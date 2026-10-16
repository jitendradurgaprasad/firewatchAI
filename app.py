"""FireWatch FIRMS observation triage dashboard with audited review notifications."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from threading import Thread
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from dotenv import load_dotenv

from firewatch.association import associate_observations
from firewatch.evaluation import run_functional_evaluation
from firewatch.exports import investigation_report_markdown, safe_csv_bytes
from firewatch.forest_context import ForestContextError, match_forest_context, parse_fsi_points
from firewatch.firms_api import FirmsAPIError, PRODUCTS, fetch_recent_area
from firewatch.firms_ingest import IngestionError, MAX_CSV_BYTES, ingest_firms_csv
from firewatch.map_view import create_observation_map, nearest_observation
from firewatch.notifications import (SimulatedNotificationProvider, build_notification_payload,
                                     SMTPTestNotificationProvider,
                                     FormSubmitNotificationProvider,
                                     dispatch_notification,
                                     notification_dedupe_key)
from firewatch.ranking import CORE_WEIGHTS, DEFAULT_WEIGHTS, score_observations
from firewatch.storage import (OUTCOMES, add_authority, add_notification, connect, database_path,
                               claim_startup_email_run, update_startup_email_run, get_startup_email_run,
                               get_setting, latest_feedback, list_authorities, list_imports,
                               load_evaluations, load_feedback, load_notifications, load_observations,
                               load_forest_context_points, list_forest_context_imports,
                               load_forest_context_rejections,
                               load_observation_sources, notification_history,
                               load_rejections, resolve_recipient,
                               save_evaluation, save_feedback, save_import,
                               save_forest_context_import, set_setting)
from firewatch.synthetic_demo import generate_synthetic_forest_demo, write_synthetic_forest_demo
from firewatch.satellite_imagery import GIBS_LAYERS, build_gibs_image_url, fetch_gibs_image


APP_ROOT = Path(__file__).resolve().parent
load_dotenv()
SAMPLE_PATH = APP_ROOT / "sample_data" / "nasa_firms_viirs_snpp_071223.csv"
SAMPLE_META_PATH = APP_ROOT / "sample_data" / "nasa_firms_viirs_snpp_071223.json"
NAV_ITEMS = ["Live Monitoring", "Overview", "Import Data", "Observation Map", "Observation Table",
             "Priority Review Queue", "Observation Details", "Feedback and Review History",
             "Spatial and Temporal Analysis", "Satellite Image Viewer", "Notification Centre",
             "Data Validation", "Evaluation", "Settings and About"]


@st.cache_resource
def _app_process_run_id() -> str:
    """One stable ID for the lifetime of this Streamlit server process."""
    return uuid.uuid4().hex


def _send_run_start_notice(product: str, feed_status: str, detection_count: int) -> dict | None:
    """Send at most one status email per app process, never a fire alarm."""
    if os.environ.get("FIREWATCH_RUN_NOTICE_ENABLED", "0").strip().lower() not in {"1", "true", "yes"}:
        return None
    # AppTest and pytest must never cause real external messages, even if a local
    # developer .env has run notices enabled.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    recipient = os.environ.get("FIREWATCH_ERS_EMAIL", "").strip()
    if not recipient:
        return {"status": "disabled", "details": "FIREWATCH_ERS_EMAIL is not configured."}
    run_id = _app_process_run_id()
    conn = connect(database_path())
    try:
        if not claim_startup_email_run(conn, run_id=run_id, recipient=recipient):
            return get_startup_email_run(conn, run_id)
        Thread(target=_deliver_run_start_notice,
               args=(run_id, recipient, product, feed_status, detection_count),
               name="firewatch-run-email", daemon=True).start()
        return get_startup_email_run(conn, run_id)
    finally:
        conn.close()


def _deliver_run_start_notice(run_id: str, recipient: str, product: str,
                              feed_status: str, detection_count: int) -> None:
    conn = connect(database_path())
    try:
        response = FormSubmitNotificationProvider().deliver_run_notice({
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
            "product": product,
            "feed_status": feed_status,
            "detection_count": detection_count,
        }, recipient)
        update_startup_email_run(conn, run_id=run_id, status=response["status"],
                                 details=response["response"])
    except Exception as exc:
        update_startup_email_run(conn, run_id=run_id, status="failed",
                                 details=f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()


@st.fragment(run_every=10)
def _live_satellite_image(layer_name: str, image_date: date) -> None:
    """Refresh only a rendered GIBS image every ten seconds; perform no analysis."""
    layer = GIBS_LAYERS[layer_name]
    try:
        image, mime = fetch_gibs_image(build_gibs_image_url(layer, image_date), timeout=15.0)
        st.image(image, caption=f"NASA GIBS rendered {layer_name} image · imagery date {image_date.isoformat()} · {mime}",
                 width="stretch")
        st.caption(f"Image request refreshed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))


def _radar_animation() -> None:
    """Draw a decorative scan sweep; it is not a detection algorithm."""
    st.html("""
    <div style="height:270px;display:flex;align-items:center;gap:26px;background:#081b20;border:1px solid #17434b;
                border-radius:16px;padding:18px;color:#c7f9e9;font-family:Segoe UI,Arial,sans-serif">
      <div style="position:relative;width:230px;height:230px;border-radius:50%;overflow:hidden;flex:none;
                  background:radial-gradient(circle,#0b3840 0 2px,transparent 3px),
                  repeating-radial-gradient(circle,#17434b 0 1px,transparent 2px 38px),
                  linear-gradient(90deg,transparent 49.7%,#22616a 50%,transparent 50.3%),
                  linear-gradient(0deg,transparent 49.7%,#22616a 50%,transparent 50.3%),#061216">
        <div style="position:absolute;inset:0;border-radius:50%;transform-origin:50% 50%;
                    background:conic-gradient(from 0deg,transparent 0 78%,rgba(44,255,183,.02) 84%,rgba(44,255,183,.5) 99%,transparent 100%);
                    animation:firewatchSweep 4s linear infinite"></div>
        <span style="position:absolute;left:50%;top:50%;width:10px;height:10px;margin:-5px;border-radius:50%;
                     background:#4effad;box-shadow:0 0 16px #4effad;animation:firewatchPulse 1.6s ease-out infinite"></span>
      </div>
      <div><div style="font-size:22px;font-weight:700;letter-spacing:1px">LIVE SATELLITE WATCH</div>
        <div style="margin-top:8px;color:#6fffc2">● FIRMS query cycle: 60 seconds</div>
        <div style="margin-top:5px;color:#86a9aa">Decorative radar sweep · detections come from FIRMS records</div>
        <div style="margin-top:5px;color:#86a9aa">Not image-pixel analysis or a confirmed-fire alarm</div></div>
    </div>
    <style>@keyframes firewatchSweep{to{transform:rotate(360deg)}}
    @keyframes firewatchPulse{0%{transform:scale(.8);opacity:1}100%{transform:scale(3.5);opacity:0}}</style>
    """)


@st.fragment(run_every=60)
def _live_hotspot_monitor(product: str) -> None:
    """Poll one bounded India FIRMS query and show its validated current result."""
    try:
        download = fetch_recent_area(area="60,5,100,38", product=product, day_range=1,
                                     timeout_seconds=10)
        result = ingest_firms_csv(download.content, source_description=download.source,
                                  product=download.product,
                                  retrieved_at=datetime.fromisoformat(download.retrieved_at),
                                  original_filename=download.filename,
                                  licence_attribution="NASA FIRMS Area API; possible satellite thermal anomalies.")
        valid = result.valid.sort_values("acq_datetime", ascending=False) if not result.valid.empty else result.valid
        run_notice = _send_run_start_notice(product, "FIRMS query succeeded", len(valid))
        cols = st.columns(3)
        cols[0].metric("FIRMS detections returned", f"{len(valid):,}")
        cols[1].metric("Rejected rows", f"{result.report['invalid_rows']:,}")
        cols[2].metric("Last FIRMS check (UTC)", download.retrieved_at[11:19])
        st.caption(f"Source: NASA FIRMS {download.product} · India region · recent {download.day_range} day · checked {download.retrieved_at}")
        if run_notice:
            st.caption(f"ERS session notice: {run_notice.get('status', 'unknown')}. Provider acceptance is not inbox-delivery confirmation.")
        if valid.empty:
            st.info("No detections were returned for this query window. This does not prove that no fire exists.")
        else:
            st.map(valid[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}),
                   color="#ff5722", size=80, zoom=4, height=380)
            display_fields = [field for field in (
                "acq_datetime", "latitude", "longitude", "satellite", "confidence", "frp", "bright_ti4")
                if field in valid.columns]
            st.dataframe(valid[display_fields].head(500), width="stretch", hide_index=True)
        st.caption("This live feed displays FIRMS' processed detection records only. FireWatch does not infer hotspots from the image shown below.")
    except (FirmsAPIError, IngestionError, ValueError) as exc:
        run_notice = _send_run_start_notice(product, f"FIRMS query unavailable: {exc}", 0)
        if run_notice:
            st.caption(f"ERS session notice: {run_notice.get('status', 'unknown')}. Any email reports feed status only.")
        st.error(f"Live FIRMS check unavailable: {exc}")


def bootstrap_sample(conn) -> None:
    """Load the versioned public NASA sample once into the local database."""
    if list_imports(conn) or os.environ.get("FIREWATCH_AUTOLOAD_SAMPLE", "1").lower() in {"0", "false", "no"}:
        return
    if not SAMPLE_PATH.is_file():
        return
    retrieved = datetime.now(timezone.utc)
    attribution = "NASA FIRMS / NASA ESDIS. NASA recommends source acknowledgement; data from NASA-led missions are generally CC0 unless specifically restricted. This app is not endorsed by NASA."
    if SAMPLE_META_PATH.is_file():
        metadata = json.loads(SAMPLE_META_PATH.read_text(encoding="utf-8"))
        retrieved = datetime.fromisoformat(metadata["retrieved_at"])
        source = metadata["source"]
        attribution = metadata.get("licence_attribution", metadata.get("license_attribution", attribution))
        digest = hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
        if (metadata.get("source_sha256") and digest.lower() != metadata["source_sha256"].lower()) or (
                metadata.get("size_bytes") and SAMPLE_PATH.stat().st_size != int(metadata["size_bytes"])):
            st.error("The bundled NASA sample failed its manifest hash/size check. It was not imported; use a verified CSV or restore the repository sample.")
            return
    else:
        source = "NASA FIRMS Academy public VIIRS SNPP sample CSV (NRT and URT; acquisition date 2023-07-12)"
    result = ingest_firms_csv(SAMPLE_PATH, source_description=source,
                              licence_attribution=attribution, retrieved_at=retrieved,
                              product="VIIRS SNPP; Academy static sample")
    digest = hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()
    save_import(conn, result, import_id=f"nasa-sample-{digest[:20]}")


@st.cache_data(show_spinner="Preparing observation table and review priorities…", max_entries=6)
def cached_scored_data(db_file: str, cache_token: str, weights_json: str,
                       reference_mode: str, half_life_hours: float,
                       repeat_radius_km: float, repeat_window_hours: float,
                       frp_cap_mw: float, now_utc: str, minimum_coverage: float,
                       medium_threshold: float, high_threshold: float) -> pd.DataFrame:
    del cache_token  # Included as a cache invalidation key.
    conn = connect(db_file)
    observations = load_observations(conn)
    if observations.empty:
        return observations
    observations = match_forest_context(observations, load_forest_context_points(conn))
    weights = json.loads(weights_json)
    reference = observations["acq_datetime"].max() if reference_mode == "Newest observation in dataset" else now_utc
    scored = score_observations(observations, weights=weights, reference_time=reference,
                                half_life_hours=half_life_hours,
                                repeat_radius_km=repeat_radius_km,
                                repeat_window_hours=repeat_window_hours,
                                frp_cap_mw=frp_cap_mw,
                                minimum_coverage=minimum_coverage,
                                medium_threshold=medium_threshold,
                                high_threshold=high_threshold)
    # Association is an explicit, bounded user action on the Spatial page.
    # Avoid blocking ordinary dashboard startup on pairwise spatial analysis.
    scored["association_id"] = None
    scored["nearby_observation_count"] = 0
    scored["association_status"] = "not_calculated"
    reviews = latest_feedback(conn)
    scored["review_status"] = scored["observation_id"].map(
        lambda value: reviews.get(str(value), {}).get("outcome", "Awaiting review"))
    scored["verified_jurisdiction"] = scored["observation_id"].map(
        lambda value: reviews.get(str(value), {}).get("verified_jurisdiction")
        if reviews.get(str(value), {}).get("jurisdiction_verified") else None)
    return scored


def _imports_token(conn) -> str:
    row = conn.execute("SELECT COALESCE(MAX(created_at),''), COUNT(*) FROM imports").fetchone()
    feedback = conn.execute("SELECT COALESCE(MAX(feedback_id),0) FROM feedback").fetchone()[0]
    forest = conn.execute("SELECT COALESCE(MAX(created_at),''), COUNT(*) FROM forest_context_imports").fetchone()
    return f"{row[0]}:{row[1]}:{forest[0]}:{forest[1]}:{feedback}"


def _selected_observation(frame: pd.DataFrame, key: str = "selected_observation_id") -> dict | None:
    if frame.empty:
        st.info("Import FIRMS observations to review them here.")
        return None
    ids = frame["observation_id"].astype(str).tolist()
    current = st.session_state.get(key) or st.session_state.get("selected_observation_id")
    if current not in ids:
        current = ids[0]
    st.session_state[key] = current
    labels = {str(r["observation_id"]):
              f"{r['observation_id']} · {r.get('acq_datetime', '')} · {r.get('priority_category', '')}"
              for r in frame.to_dict(orient="records")}
    picker_key = f"{key}_picker"
    if st.session_state.get(picker_key) not in ids or st.session_state.get(picker_key) != current:
        st.session_state[picker_key] = current
    selected = st.selectbox("Selected observation", ids,
                            format_func=lambda value: labels.get(value, value), key=picker_key)
    st.session_state[key] = selected
    return frame.loc[frame["observation_id"].astype(str) == str(selected)].iloc[0].to_dict()


def _date_filter(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty or "acq_datetime" not in frame:
        return frame
    dates = pd.to_datetime(frame["acq_datetime"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return frame.iloc[0:0]
    minimum, maximum = dates.min().date(), dates.max().date()
    bounds = st.date_input("Acquisition date range (UTC)", value=(minimum, maximum),
                           min_value=minimum, max_value=maximum, key=f"{prefix}_dates")
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        begin, end = bounds
        parsed = pd.to_datetime(frame["acq_datetime"], utc=True, errors="coerce")
        return frame.loc[(parsed.dt.date >= begin) & (parsed.dt.date <= end)].copy()
    if isinstance(bounds, date):
        parsed = pd.to_datetime(frame["acq_datetime"], utc=True, errors="coerce")
        return frame.loc[parsed.dt.date == bounds].copy()
    return frame


def _filters(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = _date_filter(frame, prefix)
    satellites = sorted(str(x) for x in result.get("satellite", pd.Series(dtype=str)).dropna().unique() if str(x))
    chosen_satellites = st.multiselect("Satellite", satellites, default=satellites, key=f"{prefix}_sat")
    if chosen_satellites and "satellite" in result:
        result = result.loc[result["satellite"].astype(str).isin(chosen_satellites)]
    confidences = sorted(str(x) for x in result.get("confidence", pd.Series(dtype=str)).dropna().unique() if str(x))
    chosen_conf = st.multiselect("Confidence", confidences, default=confidences, key=f"{prefix}_conf")
    if chosen_conf and "confidence" in result:
        result = result.loc[result["confidence"].astype(str).isin(chosen_conf)]
    categories = [x for x in ("HIGH REVIEW PRIORITY", "MEDIUM REVIEW PRIORITY", "LOW REVIEW PRIORITY", "INSUFFICIENT INFORMATION")
                  if x in set(result.get("priority_category", []))]
    chosen_priority = st.multiselect("Review priority", categories, default=categories, key=f"{prefix}_priority")
    if chosen_priority:
        result = result.loc[result["priority_category"].isin(chosen_priority)]
    statuses = sorted(str(x) for x in result.get("review_status", pd.Series(dtype=str)).dropna().unique())
    chosen_status = st.multiselect("Review status", statuses, default=statuses, key=f"{prefix}_status")
    if chosen_status:
        result = result.loc[result["review_status"].isin(chosen_status)]
    numeric_frp = pd.to_numeric(result.get("frp", pd.Series(dtype=float)), errors="coerce")
    if numeric_frp.notna().any():
        low, high = float(numeric_frp.min()), float(numeric_frp.max())
        if high > low:
            frp_range = st.slider("FRP (MW)", min_value=low, max_value=high,
                                  value=(low, high), key=f"{prefix}_frp")
            value = pd.to_numeric(result["frp"], errors="coerce")
            result = result.loc[value.isna() | value.between(*frp_range)]
    search = st.text_input("Search raw values / satellite / observation ID", key=f"{prefix}_search")
    if search.strip():
        searchable = result.drop(columns=["source_record"], errors="ignore")
        mask = searchable.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, regex=False, na=False)).any(axis=1)
        result = result.loc[mask]
    if st.checkbox("Limit to a geographic bounding box", key=f"{prefix}_bbox"):
        with st.expander("Bounding box (west, south, east, north)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            west = c1.number_input("West longitude", -180.0, 180.0, -180.0, key=f"{prefix}_west")
            south = c2.number_input("South latitude", -90.0, 90.0, -90.0, key=f"{prefix}_south")
            east = c3.number_input("East longitude", -180.0, 180.0, 180.0, key=f"{prefix}_east")
            north = c4.number_input("North latitude", -90.0, 90.0, 90.0, key=f"{prefix}_north")
        if west < east and south < north:
            result = result.loc[pd.to_numeric(result.latitude, errors="coerce").between(south, north) &
                                pd.to_numeric(result.longitude, errors="coerce").between(west, east)]
        else:
            st.warning("Bounding box must have west < east and south < north; geographic filter not applied.")
    return result


def _show_validation(result) -> None:
    report = result.report
    a, b, c, d = st.columns(4)
    a.metric("Rows", report["total_rows"])
    b.metric("Valid", report["valid_rows"])
    c.metric("Rejected", report["invalid_rows"])
    d.metric("Exact duplicates", report["duplicate_rows"])
    st.write("Missing supported-field values")
    missing = {k: v for k, v in report["missing_field_counts"].items() if v}
    st.json({"missing_columns": report["missing_columns"], "missing_field_counts": missing,
             "rejection_reasons": report["rejection_reasons"]})
    if not result.rejected.empty:
        st.dataframe(result.rejected[["source_row_number", "rejection_reasons"]].head(500),
                     width="stretch", hide_index=True)
        st.download_button("Download rejected source rows with reasons",
                           safe_csv_bytes(result.rejected),
                           file_name="firms_rejected_rows.csv", mime="text/csv")


def _review_form(conn, observation: dict, authorities: pd.DataFrame) -> None:
    jurisdictions = []
    if not authorities.empty:
        jurisdictions = sorted(authorities.loc[(authorities["active"] == 1) & (authorities["verified"] == 1),
                                               "jurisdiction"].dropna().unique().tolist())
    reviewer_default = os.environ.get("FIREWATCH_REVIEWER_ID", "")
    st.subheader("Record a human review")
    st.caption("Reviewer assessments are retained as an audit trail. They are not independent ground truth.")
    with st.form("feedback_form", clear_on_submit=True):
        outcome = st.selectbox("Review outcome", OUTCOMES)
        assessment = st.text_input("Reviewer assessment", value=outcome)
        confidence_choice = st.selectbox("Confidence in this assessment", ["Not stated", "Low", "Medium", "High"])
        notes = st.text_area("Notes (optional)")
        reviewer = st.text_input("Reviewer identifier (optional)", value=reviewer_default,
                                 disabled=bool(reviewer_default))
        jurisdiction = st.selectbox("Manually verified jurisdiction", ["Unknown / not verified", *jurisdictions])
        verified = st.checkbox("I verified this jurisdiction using an authoritative source")
        evidence_source = st.selectbox("Evidence source", ["Not recorded", "Official incident report",
            "Field inspection / fire service report", "Post-fire imagery or burned-area record", "Other"])
        evidence_reference = st.text_input("Evidence link or record reference (optional)")
        event_id = st.text_input("Fire event ID (needed for event-aware evaluation)")
        independent_evidence_reviewed = st.checkbox(
            "I checked evidence independent of the FIRMS/VIIRS hotspot")
        st.caption("FSI forest-area alert matches and the FIRMS point are correlated satellite evidence; they do not qualify as independent ground truth.")
        submitted = st.form_submit_button("Save review to audit history")
    if submitted:
        if verified and jurisdiction == "Unknown / not verified":
            st.error("Choose a configured jurisdiction before marking it verified.")
            return
        confidence = {"Not stated": None, "Low": .25, "Medium": .5, "High": .9}[confidence_choice]
        if evidence_source == "Not recorded":
            evidence_source = ""
        try:
            save_feedback(conn, observation_id=str(observation["observation_id"]), outcome=outcome,
                          assessment=assessment, confidence=confidence, notes=notes,
                          reviewer_id=reviewer,
                          verified_jurisdiction=jurisdiction if verified else None,
                          jurisdiction_verified=verified,
                          evidence_source=evidence_source,
                          evidence_reference=evidence_reference,
                          event_id=event_id,
                          independent_evidence_reviewed=independent_evidence_reviewed)
            st.success("Review saved. The original FIRMS row was preserved.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def _render_details(conn, scored: pd.DataFrame, authorities: pd.DataFrame) -> None:
    observation = _selected_observation(scored)
    if observation is None:
        return
    st.subheader("Observation details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Priority", observation.get("priority_category", "INSUFFICIENT INFORMATION"))
    c2.metric("Review score", f"{observation.get('priority_score', 0):.1f} / 100")
    c3.metric("Review status", observation.get("review_status", "Awaiting review"))
    st.write(f"**Coordinates:** {observation['latitude']:.5f}, {observation['longitude']:.5f}")
    st.write(f"**Observation time:** {observation.get('acq_datetime', 'Unknown')} (UTC)")
    st.write(f"**Satellite / instrument:** {observation.get('satellite') or 'Unknown'} / {observation.get('instrument') or 'Unknown'}")
    st.write(f"**Product:** {observation.get('product') or 'Not specified'} · **Source:** {observation.get('source_name') or 'Unknown'}")
    forest_summary = observation.get("forest_context_summary") or "No matched FSI forest-area context"
    st.write(f"**Forest context:** {forest_summary}")
    if observation.get("forest_context") is not None and not pd.isna(observation.get("forest_context")):
        st.caption("FSI context comes from forest-area-filtered satellite alerts; it is not independent confirmation of fire.")
    st.write(f"**Original file:** {observation.get('source_filename') or 'Unknown'} · **SHA-256:** {(observation.get('source_file_sha256') or 'Unavailable')[:16]}")
    available_fields = observation.get("available_fields") or []
    st.write("**Available source fields:** " + (", ".join(available_fields) if available_fields else "Not recorded"))
    st.write(f"**Confidence:** {observation.get('confidence') if observation.get('confidence') is not None else 'Missing'}")
    st.write(f"**FRP:** {observation.get('frp') if observation.get('frp') is not None else 'Missing'} MW")
    st.write(f"**Priority explanation:** {observation.get('explanation', '')}")
    st.write(f"**Missing features:** {observation.get('missing_factors') or 'None'}")
    st.write(f"**Data-quality warnings:** {observation.get('source_quality_warnings') or observation.get('data_quality_warnings') or 'None reported'}")
    st.write(f"**Spatial/time association:** {observation.get('association_id') or 'No associated observation'} · {int(observation.get('nearby_observation_count') or 0)} direct nearby detections")
    if observation.get("association_id"):
        related = scored.loc[scored["association_id"] == observation["association_id"]].copy()
        related = related.loc[related["observation_id"].astype(str) != str(observation["observation_id"])]
        if not related.empty:
            st.write("**Related observations (same connected association)**")
            st.dataframe(related.drop(columns=["source_record"], errors="ignore"), width="stretch", hide_index=True)
    jurisdiction = observation.get("verified_jurisdiction")
    if jurisdiction:
        st.info(f"Manually verified jurisdiction: {jurisdiction}")
    else:
        st.warning("Jurisdiction unknown — manual review required.")
    st.write("**Original source record**")
    st.json(json.loads(observation["source_record"]))
    source_history = load_observation_sources(conn, str(observation["observation_id"]))
    if not source_history.empty:
        st.write("**Import provenance for this observation**")
        st.dataframe(source_history, width="stretch", hide_index=True)
    st.write("**Review history**")
    history = load_feedback(conn, str(observation["observation_id"]))
    if history.empty:
        st.info("No reviewer feedback has been recorded.")
    else:
        st.dataframe(history.drop(columns=["observation_id"], errors="ignore"),
                     width="stretch", hide_index=True)
    notification_rows = load_notifications(conn)
    if not notification_rows.empty:
        notification_rows = notification_rows.loc[notification_rows["observation_id"] == str(observation["observation_id"])]
    st.write("**Notification history**")
    if notification_rows.empty:
        st.info("No notifications recorded for this observation.")
    else:
        st.dataframe(notification_rows.drop(columns=["payload_json"], errors="ignore"), width="stretch", hide_index=True)
    report_md = investigation_report_markdown(observation, feedback=history, notifications=notification_rows)
    st.download_button("Download investigation report (Markdown)", report_md.encode("utf-8"),
                       file_name=f"firewatch_{observation['observation_id']}_report.md", mime="text/markdown")
    _review_form(conn, observation, authorities)


def main() -> None:
    st.set_page_config(page_title="FireWatch AI | FIRMS Triage", page_icon="🔥",
                       layout="wide", initial_sidebar_state="collapsed")
    conn = connect()
    bootstrap_sample(conn)

    settings = get_setting(conn, "dashboard", {})
    weights = settings.get("weights", DEFAULT_WEIGHTS.copy())
    if set(weights) == set(CORE_WEIGHTS):
        weights = {**{key: float(value) * .85 for key, value in weights.items()},
                   "forest_context": .15}
    with st.sidebar:
        st.title("🔥 FireWatch AI")
        st.caption("Forest-fire hotspot triage · Human-in-the-loop")
        page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
        st.divider()
        st.metric("Stored observations", int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]))
        st.caption("ERS email: one-click manual anomaly review")

    imports = list_imports(conn)
    token = _imports_token(conn)
    pages_without_scored_data = {"Live Monitoring", "Satellite Image Viewer", "Import Data"}
    if imports and page not in pages_without_scored_data:
        weights_json = json.dumps(weights, sort_keys=True)
        reference_mode = settings.get("reference_mode", "Newest observation in dataset")
        now_utc = (datetime.now(timezone.utc).isoformat()
                   if reference_mode == "Current UTC time" else "unused-for-dataset-reference")
        scored = cached_scored_data(str(database_path()), token, weights_json, reference_mode,
                                    float(settings.get("half_life_hours", 72)),
                                    float(settings.get("repeat_radius_km", 1)),
                                    float(settings.get("repeat_window_hours", 24)),
                                    float(settings.get("frp_cap_mw", 500)),
                                    now_utc,
                                    float(settings.get("minimum_coverage", .60)),
                                    float(settings.get("medium_threshold", 45)),
                                    float(settings.get("high_threshold", 70)))
    else:
        scored = pd.DataFrame()
    saved_associations = st.session_state.get("association_result")
    if (not scored.empty and saved_associations and saved_associations.get("cache_token") == token and
            bool(settings.get("association_enabled", True)) and
            saved_associations.get("distance_km") == float(settings.get("association_distance_km", 5)) and
            saved_associations.get("time_window_hours") == float(settings.get("association_window_hours", 6))):
        assoc_rows = saved_associations["frame"]
        scored = scored.drop(columns=["association_id", "nearby_observation_count", "association_status"], errors="ignore").merge(
            assoc_rows, on="observation_id", how="left")
        scored["association_status"] = scored["association_status"].fillna("not_in_analysis_subset")
        scored["nearby_observation_count"] = scored["nearby_observation_count"].fillna(0).astype(int)
    authorities = list_authorities(conn)
    latest_import = imports[0] if imports else None

    st.title(page)
    if page != "Live Monitoring":
        st.caption("Forest-fire hotspot triage. FIRMS and FSI provide satellite thermal-anomaly cues; a match adds forest-area context but does not confirm a wildfire or measure severity.")
    if page != "Live Monitoring" and not scored.empty and scored.get("source_name", pd.Series(dtype=str)).astype(str).str.contains("SYNTHETIC", case=False).any():
        st.warning("Synthetic records are loaded. Their generator-defined labels and simulated locations are for software demonstration only; they do not represent real observations or establish real-world accuracy.")

    if page == "Live Monitoring":
        st.write("Live monitoring polls NASA FIRMS for processed satellite hotspot records. The animated radar is a visual status display; it does not scan or analyze the imagery pixels.")
        _radar_animation()
        st.caption("FIRMS hotspot products are observations derived from satellite instrument measurements. They are possible thermal anomalies, not confirmed wildfires. Image refresh (10 sec) and FIRMS query refresh (60 sec) are separate; neither changes satellite acquisition cadence.")
        product = st.selectbox("FIRMS satellite product", PRODUCTS, index=0, key="live_monitor_product")
        _live_hotspot_monitor(product)
        st.subheader("Latest satellite image")
        st.caption("Read-only VIIRS true-color image from NASA GIBS, requested every 10 seconds. No image inference or automatic alert decision is performed.")
        _live_satellite_image("Suomi-NPP VIIRS true color", datetime.now(timezone.utc).date() - timedelta(days=1))

    elif page == "Satellite Image Viewer":
        st.write("A read-only satellite imagery viewer. It displays NASA GIBS rendered VIIRS imagery and does not run FireWatch ranking, import records, or generate alerts.")
        st.warning("The FIRMS MAP_KEY is not used here: that key retrieves FIRMS fire-detection records/map services, not imagery. This tab uses NASA GIBS imagery separately and requires no FIRMS key.")
        st.caption("The image display refreshes every 10 seconds. The selected satellite image itself usually updates only when NASA publishes another processed scene; many GIBS near-real-time products appear a few hours after observation. Repeated refreshes can show the same image and do not mean a new satellite capture arrived.")
        col_layer, col_date = st.columns(2)
        today_utc = datetime.now(timezone.utc).date()
        with col_layer:
            layer_name = st.selectbox("Satellite imagery layer", list(GIBS_LAYERS))
        with col_date:
            image_date = st.date_input("Imagery date (UTC)", value=today_utc - timedelta(days=1),
                                       max_value=today_utc)
        st.caption("Regional view: India and surrounding South Asia · rendered corrected-reflectance true color. Imagery is visual context only; the viewer does not interpret it as a fire or use it for model predictions.")
        _live_satellite_image(layer_name, image_date)

    elif page == "Overview":
        if scored.empty:
            st.info("No observations are loaded. Import a CSV or enable the bundled sample.")
            return
        status_counts = scored["review_status"].value_counts()
        categories = scored["priority_category"].value_counts()
        valid_count = len(scored)
        rejected_count = sum(json.loads(item["report_json"]).get("invalid_rows", 0) for item in imports)
        feedback_count = int(conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
        notification_count = int(conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0])
        high, medium, low = (int(categories.get(name, 0)) for name in
                             ("HIGH REVIEW PRIORITY", "MEDIUM REVIEW PRIORITY", "LOW REVIEW PRIORITY"))
        cols = st.columns(9)
        for col, label, value in zip(cols,
            ["Stored unique", "Valid source rows", "Rejected source rows", "High review", "Medium review", "Low review", "Awaiting review", "Feedback records", "Notifications"],
            [valid_count, sum(json.loads(i["report_json"]).get("valid_rows", 0) for i in imports), rejected_count,
             high, medium, low, int(status_counts.get("Awaiting review", 0)), feedback_count, notification_count]):
            col.metric(label, value)
        st.caption(f"Latest retrieval timestamp: {latest_import['retrieved_at'] if latest_import else 'Unavailable'}")
        st.caption(f"Source: {latest_import['source'] if latest_import else 'Unknown'}")
        latest_meta, latest_report = json.loads(latest_import["metadata_json"]), json.loads(latest_import["report_json"])
        st.caption(f"Latest source date coverage: {latest_meta.get('date_range_start') or 'unknown'} to {latest_meta.get('date_range_end') or 'unknown'} · Latest import: {latest_report.get('total_rows', 0):,} rows")
        st.caption("Stored unique counts may be lower than accepted source-row counts when the same source rows are imported again.")
        left, right = st.columns(2)
        with left:
            counts = scored["priority_category"].value_counts().rename_axis("Review category").reset_index(name="Observations")
            st.plotly_chart(px.bar(counts, x="Review category", y="Observations", color="Review category",
                                   title="Review-priority queue"), width="stretch")
        with right:
            daily = (scored.assign(day=pd.to_datetime(scored["acq_datetime"], utc=True).dt.date)
                     .groupby("day", as_index=False).size().rename(columns={"size": "Observations"}))
            st.plotly_chart(px.line(daily, x="day", y="Observations", title="Acquisition observations by day"),
                            width="stretch")
        st.subheader("Next items for review")
        st.dataframe(scored.sort_values("priority_score", ascending=False)[
            ["observation_id", "acq_datetime", "latitude", "longitude", "satellite", "confidence", "frp",
             "forest_context_matches",
             "priority_score", "priority_category", "review_status"]].head(20),
            width="stretch", hide_index=True)
        st.warning("Priority categories are for investigation ordering only. They do not assert fire presence, severity, or emergency status.")

    elif page == "Import Data":
        st.write("Upload FIRMS observations and, optionally, an FSI Fire Point Search CSV. FSI matches add forest-area context to the review score; they are not independent confirmation because FSI uses MODIS/VIIRS detections.")
        st.caption(f"Acquisition timestamps are interpreted as UTC. Interactive CSV limit: {MAX_CSV_BYTES // (1024 * 1024)} MiB. The bundled archive sample works without credentials.")
        st.subheader("Synthetic forest-fire workflow demo")
        st.warning("Synthetic scenario data only. These generated examples are not real satellite observations, verified forest locations, or evidence of model accuracy. Do not use for emergency decisions or training a deployable model.")
        with st.form("synthetic_demo_import"):
            st.caption("Creates 800 reproducible FIRMS-shaped observations across 40 synthetic events (320 generated forest-fire, 480 generated non-fire) and 480 correlated synthetic forest-context points. The files are saved outside the repository.")
            include_demo = st.form_submit_button("Generate and load synthetic demo")
        if include_demo:
            try:
                demo = generate_synthetic_forest_demo()
                generated_files = write_synthetic_forest_demo()
                if not any(item["import_id"] == "synthetic-forest-demo-v1" for item in imports):
                    result = ingest_firms_csv(demo.observations_csv,
                        source_description="SYNTHETIC DEMO — generated scenario records, NOT REAL observations",
                        licence_attribution="Generated locally by FireWatch; no external dataset or real labels.",
                        product="Synthetic VIIRS-shaped workflow test data")
                    save_import(conn, result, import_id="synthetic-forest-demo-v1")
                context_imports_now = list_forest_context_imports(conn)
                if not any(item["import_id"] == "synthetic-forest-demo-context-v1" for item in context_imports_now):
                    context = parse_fsi_points(demo.context_csv)
                    save_forest_context_import(conn, context,
                        import_id="synthetic-forest-demo-context-v1",
                        source="SYNTHETIC DEMO — generated forest context, NOT FSI",
                        filename="synthetic_forest_context.csv",
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                        metadata={**demo.manifest, "interpretation": "Generated correlated software-demo context; not real forest boundary or independent truth."})
                st.session_state["data_version"] = str(uuid.uuid4())
                cached_scored_data.clear()
                st.success(f"Synthetic demo loaded. Files: {generated_files['output_dir']}. These labels are generator-defined and cannot establish real-world performance.")
                st.rerun()
            except (IngestionError, ForestContextError, ValueError, OSError) as exc:
                st.error(f"Synthetic demo could not be loaded: {exc}")
        with st.form("csv_import", clear_on_submit=True):
            uploaded = st.file_uploader("FIRMS CSV file", type=["csv"], accept_multiple_files=False)
            local_path = st.text_input("Or local CSV path (optional)")
            source = st.text_input("Data source", value="User-uploaded CSV; source not yet verified")
            product = st.text_input("Product (optional)", value="")
            attribution = st.text_input("Licence / attribution", value="Uploader must confirm source terms and attribution")
            imported = st.form_submit_button("Validate and import")
        if imported:
            if uploaded is not None and local_path.strip():
                st.error("Choose either an upload or a local path, not both.")
            elif uploaded is None and not local_path.strip():
                st.error("Choose a CSV file or enter a local path.")
            else:
                try:
                    selected_file = uploaded if uploaded is not None else Path(local_path.strip())
                    if uploaded is not None and uploaded.size > MAX_CSV_BYTES:
                        raise IngestionError(f"Upload exceeds {MAX_CSV_BYTES // (1024 * 1024)} MiB; use a smaller regional/time slice.")
                    result = ingest_firms_csv(selected_file, source_description=source,
                                              licence_attribution=attribution, product=product)
                    import_id = str(uuid.uuid4())
                    save_import(conn, result, import_id=import_id)
                    st.session_state["last_validation"] = {"report": result.report,
                        "metadata": result.metadata,
                        "rejected_csv": safe_csv_bytes(result.rejected)}
                    st.session_state["data_version"] = str(uuid.uuid4())
                    cached_scored_data.clear()
                    st.success(f"Import recorded: {result.report['valid_rows']} valid, {result.report['invalid_rows']} rejected of {result.report['total_rows']} rows.")
                    st.rerun()
                except (IngestionError, UnicodeDecodeError, ValueError) as exc:
                    st.error(f"CSV import failed: {exc}")
        if st.session_state.get("last_validation"):
            item = st.session_state["last_validation"]
            st.subheader("Latest validation report")
            st.json({"report": item["report"], "source_metadata": item["metadata"]})
            st.caption("Available canonical fields: " + ", ".join(item["metadata"].get("available_fields", [])))
            if item["rejected_csv"].strip():
                st.download_button("Download rejected rows", item["rejected_csv"],
                                   file_name="firms_rejected_rows.csv", mime="text/csv")
        st.subheader("Imported source files")
        if imports:
            table = []
            for item in imports:
                table.append({"source": item["source"], "filename": item["filename"],
                              **json.loads(item["metadata_json"]), **json.loads(item["report_json"])})
            st.dataframe(pd.DataFrame(table)[["source", "filename", "retrieved_at", "date_range_start",
                 "date_range_end", "total_records", "valid_records", "rejected_records"]],
                 width="stretch", hide_index=True)
        else:
            st.info("No imported sources yet.")

        st.divider()
        st.subheader("Import Forest Survey of India forest-area context")
        st.write("Download a CSV from the [FSI Fire Point Search](https://fsiforestfire.gov.in/FirePointSearch/Search) and import it here. FireWatch matches points within 1 km and 12 hours. FSI defines these as satellite hotspots filtered to forest-area boundaries, so matches raise forest-context priority but are not confirmed-fire labels.")
        with st.form("fsi_context_import", clear_on_submit=True):
            fsi_csv = st.file_uploader("FSI Fire Point Search CSV", type=["csv"], key="fsi_context_file")
            fsi_imported = st.form_submit_button("Validate and import FSI forest context")
        if fsi_imported:
            if fsi_csv is None:
                st.error("Choose an FSI Fire Point Search CSV.")
            else:
                try:
                    fsi_bytes = fsi_csv.getvalue()
                    points = parse_fsi_points(fsi_bytes)
                    report = dict(points.attrs)
                    save_forest_context_import(conn, points, import_id=str(uuid.uuid4()),
                        source="Forest Survey of India Fire Point Search",
                        filename=fsi_csv.name, retrieved_at=datetime.now(timezone.utc).isoformat(),
                        metadata={"portal": "https://fsiforestfire.gov.in/FirePointSearch/Search",
                                  "file_size_bytes": len(fsi_bytes),
                                  "file_sha256": hashlib.sha256(fsi_bytes).hexdigest(),
                                  "match_radius_km": 1.0, "match_time_window_hours": 12,
                                  "interpretation": "Satellite alerts filtered by FSI forest-area boundaries; not ground truth."})
                    st.session_state["data_version"] = str(uuid.uuid4())
                    cached_scored_data.clear()
                    st.success(f"FSI context imported: {report['valid_rows']:,} valid points; {report['rejected_rows']:,} rejected of {report['total_rows']:,} rows.")
                    st.rerun()
                except (ForestContextError, UnicodeDecodeError, ValueError) as exc:
                    st.error(f"FSI context import failed: {exc}")
        fsi_imports = list_forest_context_imports(conn)
        if fsi_imports:
            summaries = [{"source": item["source"], "filename": item["filename"],
                          **json.loads(item["report_json"]), "retrieved_at": item["retrieved_at"]}
                         for item in fsi_imports]
            st.dataframe(pd.DataFrame(summaries), width="stretch", hide_index=True)
            fsi_rejections = load_forest_context_rejections(conn)
            if not fsi_rejections.empty:
                st.subheader("Rejected FSI point rows")
                st.dataframe(fsi_rejections.drop(columns=["source_record"]).head(500),
                             width="stretch", hide_index=True)
                st.download_button("Export rejected FSI rows", safe_csv_bytes(fsi_rejections),
                    file_name="fsi_forest_context_rejections.csv", mime="text/csv")

        st.divider()
        st.subheader("Refresh from NASA FIRMS Area API")
        st.write("Optional recent data import. The official API requires a free MAP_KEY; it is read from `FIRMS_MAP_KEY`, never stored in SQLite, and is not required to use the sample or upload CSVs.")
        with st.form("firms_api_import"):
            api_product = st.selectbox("FIRMS product", PRODUCTS, index=0)
            api_area = st.text_input("Bounding box: west,south,east,north", value="60,5,100,38")
            api_days = st.slider("Recent day range", 1, 5, 1)
            api_import = st.form_submit_button("Fetch and validate recent FIRMS CSV")
        if api_import:
            try:
                download = fetch_recent_area(area=api_area, product=api_product, day_range=api_days)
                result = ingest_firms_csv(download.content, source_description=download.source,
                    product=download.product, retrieved_at=datetime.fromisoformat(download.retrieved_at),
                    original_filename=download.filename,
                    licence_attribution="NASA FIRMS Area API; source acknowledgement requested. See https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy")
                save_import(conn, result, import_id=str(uuid.uuid4()))
                cached_scored_data.clear()
                st.success(f"Retrieved at {download.retrieved_at}; {result.report['valid_rows']:,} accepted, {result.report['invalid_rows']:,} rejected of {result.report['total_rows']:,} rows. Imported as recent API data.")
                st.rerun()
            except (FirmsAPIError, IngestionError, ValueError) as exc:
                st.error(str(exc))

    elif page == "Observation Map":
        if scored.empty:
            st.info("No valid observations to map.")
            return
        st.sidebar.subheader("Map filters")
        filtered = _filters(scored, "map")
        max_markers = int(settings.get("max_map_markers", 5000))
        map_widget, shown, excluded = create_observation_map(filtered, max_markers=max_markers)
        if excluded:
            st.warning(f"{excluded} observation(s) had invalid coordinates and were excluded from the map.")
        if map_widget is None:
            st.info("No valid coordinates remain after filtering.")
        else:
            st.caption(f"Showing {shown:,} of {len(filtered):,} filtered observations. Map tiles use OpenStreetMap and require internet access.")
            response = st_folium(map_widget, height=650, use_container_width=True,
                                 key="firms_observation_map", returned_objects=["last_object_clicked"])
            clicked = response.get("last_object_clicked") if response else None
            if clicked:
                observation_id = nearest_observation(filtered, float(clicked["lat"]), float(clicked["lng"]))
                if observation_id:
                    st.session_state["selected_observation_id"] = observation_id
                    st.session_state["selected_observation_id_picker"] = observation_id
                    st.info(f"Selected {observation_id}. Open Observation Details to review it.")
        st.download_button("Export filtered map observations", safe_csv_bytes(filtered),
                           file_name="firms_map_observations.csv", mime="text/csv")

    elif page == "Observation Table":
        if scored.empty:
            st.info("No observations to display.")
            return
        st.sidebar.subheader("Table filters")
        filtered = _filters(scored, "table")
        st.write(f"Showing **{len(filtered):,}** matching observations.")
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
        page_count = max(1, (len(filtered) + page_size - 1) // page_size)
        page_number = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
        start = (int(page_number) - 1) * page_size
        selectable = filtered.iloc[start:start + page_size].drop(columns=["source_record"], errors="ignore").reset_index(drop=True)
        event = st.dataframe(selectable, width="stretch", hide_index=True,
                             on_select="rerun", selection_mode="single-row")
        if event and event.selection.rows:
            chosen = selectable.iloc[event.selection.rows[0]]["observation_id"]
            st.session_state["selected_observation_id"] = str(chosen)
            st.session_state["selected_observation_id_picker"] = str(chosen)
        st.download_button("Export filtered observations", safe_csv_bytes(filtered),
                           file_name="firms_filtered_observations.csv", mime="text/csv")

    elif page == "Priority Review Queue":
        if scored.empty:
            st.info("No observations are available.")
            return
        queue = scored.loc[scored["priority_category"].isin(
            ["HIGH REVIEW PRIORITY", "MEDIUM REVIEW PRIORITY", "LOW REVIEW PRIORITY"])]
        queue = queue.loc[queue["review_status"] == "Awaiting review"]
        st.metric("Unreviewed scored observations", len(queue))
        st.sidebar.subheader("Queue filters")
        filtered = _filters(scored, "queue")
        filtered = filtered.loc[filtered["review_status"] == "Awaiting review"].sort_values("priority_score", ascending=False)
        selectable = filtered.drop(columns=["source_record"], errors="ignore").reset_index(drop=True)
        event = st.dataframe(selectable, width="stretch", hide_index=True,
                             on_select="rerun", selection_mode="single-row")
        if event and event.selection.rows:
            chosen = selectable.iloc[event.selection.rows[0]]["observation_id"]
            st.session_state["selected_observation_id"] = str(chosen)
            st.session_state["observation_picker"] = str(chosen)
        st.download_button("Export prioritized queue", safe_csv_bytes(filtered),
                           file_name="firms_priority_queue.csv", mime="text/csv")

    elif page == "Observation Details":
        if scored.empty:
            st.info("No observations to review.")
            return
        _render_details(conn, scored, authorities)

    elif page == "Feedback and Review History":
        history = load_feedback(conn)
        if history.empty:
            st.info("No reviewer feedback has been recorded yet.")
        else:
            outcomes = sorted(history["outcome"].dropna().astype(str).unique())
            selected_outcomes = st.multiselect("Filter by review outcome", outcomes, default=outcomes)
            filtered_history = history.loc[history["outcome"].isin(selected_outcomes)]
            st.dataframe(filtered_history, width="stretch", hide_index=True)
            st.download_button("Export filtered feedback audit history", safe_csv_bytes(filtered_history),
                               file_name="firms_feedback_history.csv", mime="text/csv")
        if not scored.empty:
            st.subheader("Review an observation")
            observation = _selected_observation(scored, key="feedback_observation_id")
            if observation:
                _review_form(conn, observation, authorities)

    elif page == "Notification Centre":
        st.warning("FIRMS detections are unconfirmed thermal anomalies. Any email here requests manual review only; FireWatch does not dispatch response teams or confirm fires.")
        if not scored.empty:
            st.subheader("Queue an observation for review")
            observation = _selected_observation(scored, key="notification_observation_id")
            if observation:
                reason = st.text_input("Trigger reason", value="Human reviewer requested follow-up")
                jurisdiction = observation.get("verified_jurisdiction")
                recipient = resolve_recipient(conn, jurisdiction)
                st.write(f"**Routing:** {recipient['name']} — {recipient['routing_reason']}")
                if st.button("Create simulated notification", type="primary"):
                    payload = build_notification_payload(observation, trigger_reason=reason,
                                routing_reason=recipient["routing_reason"], recipient=recipient["name"])
                    notification_id = payload["alert_id"]
                    dedupe = notification_dedupe_key(str(observation["observation_id"]), reason)
                    created = add_notification(conn, notification_id=notification_id,
                        observation_id=str(observation["observation_id"]), recipient=recipient["name"],
                        channel="simulation", trigger_reason=reason, dedupe_key=dedupe, payload=payload)
                    if not created:
                        st.warning("A notification with this observation and trigger reason already exists. No duplicate was added.")
                    else:
                        response = dispatch_notification(conn, notification_id,
                                                         SimulatedNotificationProvider())
                        st.success(f"Notification {notification_id} recorded with status `{response['status']}`; no message was sent.")
                        st.rerun()
                st.divider()
                st.subheader("Send a FIRMS review request to ERS support")
                ers_email = os.environ.get("FIREWATCH_ERS_EMAIL", "").strip()
                source_name = str(observation.get("source_name") or "")
                acquired = pd.to_datetime(observation.get("acq_datetime"), utc=True, errors="coerce")
                age_hours = ((pd.Timestamp.now(tz="UTC") - acquired).total_seconds() / 3600
                             if pd.notna(acquired) else float("inf"))
                eligible_live_record = ("NASA FIRMS Area API" in source_name and
                    "SYNTHETIC" not in source_name.upper() and -0.25 <= age_hours <= 144)
                st.caption("Uses FormSubmit to email the configured support inbox. The recipient must complete FormSubmit's one-time email activation before submissions can be delivered. Coordinates and record metadata are sent to FormSubmit; its documentation says submissions are retained for 30 days.")
                st.write(f"**ERS support inbox:** {ers_email or 'not configured in private .env'}")
                if not eligible_live_record:
                    st.info("Email is enabled only for NASA FIRMS Area API observations acquired within the last six days. Historical sample, uploaded, and synthetic rows cannot be emailed through this route.")
                if not ers_email:
                    st.info("Set FIREWATCH_ERS_EMAIL in the private local .env file and restart FireWatch.")
                st.caption("One click sends this selected, recent FIRMS anomaly to ERS support for human review. It does not confirm a fire or dispatch responders.")
                send_ers_enabled = bool(ers_email and eligible_live_record)
                if st.button("Email this anomaly to ERS support", type="primary", key="send_ers_formsubmit",
                             disabled=not send_ers_enabled):
                    email_payload = build_notification_payload(observation,
                        trigger_reason="Manual ERS authority review request",
                        routing_reason="User-designated ERS inbox; recipient is responsible for manual verification",
                        recipient=ers_email)
                    email_id = email_payload["alert_id"]
                    email_dedupe = notification_dedupe_key(str(observation["observation_id"]),
                        "ers_formsubmit_review_request", ers_email)
                    created = add_notification(conn, notification_id=email_id,
                        observation_id=str(observation["observation_id"]), recipient=ers_email,
                        channel="formsubmit", trigger_reason="ers_formsubmit_review_request",
                        dedupe_key=email_dedupe, payload=email_payload)
                    if not created:
                        st.warning("An ERS email request for this observation and inbox was already recorded. Use the audited retry action only if the previous attempt failed.")
                    else:
                        response = dispatch_notification(conn, email_id,
                            FormSubmitNotificationProvider(), max_attempts=1)
                        if response["status"] == "provider-accepted":
                            st.success(f"FormSubmit accepted the review request ({email_id}). This does not confirm inbox delivery; the recipient may need to activate FormSubmit first.")
                        else:
                            st.error(f"Review email was not confirmed: {response['response']}")
                        st.rerun()
                st.divider()
                st.subheader("Send synthetic hackathon test email")
                test_recipients = [value.strip() for value in os.environ.get("FIREWATCH_TEST_RECIPIENTS", "").split(",") if value.strip()]
                smtp_config = {key: os.environ.get(key, "") for key in (
                    "FIREWATCH_SMTP_HOST", "FIREWATCH_SMTP_USERNAME", "FIREWATCH_SMTP_PASSWORD", "FIREWATCH_SMTP_FROM")}
                smtp_port = int(os.environ.get("FIREWATCH_SMTP_PORT", "587"))
                synthetic_only = "SYNTHETIC" in str(observation.get("source_name") or "").upper()
                st.caption("Recipients: " + (", ".join(test_recipients) if test_recipients else "not configured"))
                if not synthetic_only:
                    st.info("Select a row from the generated synthetic demo to enable email. Real observation rows are blocked from this test sender.")
                if not all(smtp_config.values()):
                    st.info("SMTP is not configured. Add FIREWATCH_SMTP_HOST, FIREWATCH_SMTP_USERNAME, FIREWATCH_SMTP_PASSWORD, FIREWATCH_SMTP_FROM and FIREWATCH_SMTP_PORT to the private local .env file to enable delivery. Simulation still works without SMTP.")
                send_enabled = synthetic_only and bool(test_recipients) and all(smtp_config.values())
                if st.button("Send synthetic test alert email", disabled=not send_enabled, type="primary", key="send_synthetic_email"):
                    provider = SMTPTestNotificationProvider(host=smtp_config["FIREWATCH_SMTP_HOST"], port=smtp_port,
                        username=smtp_config["FIREWATCH_SMTP_USERNAME"], password=smtp_config["FIREWATCH_SMTP_PASSWORD"],
                        from_address=smtp_config["FIREWATCH_SMTP_FROM"])
                    delivery_results = []
                    for email_address in test_recipients:
                        email_payload = build_notification_payload(observation,
                            trigger_reason="synthetic_hackathon_test_email", routing_reason="User-configured test recipient",
                            recipient=email_address)
                        email_payload["test_only"] = True
                        email_payload["synthetic_data"] = True
                        email_payload["synthetic_scenario"] = "generated forest-fire workflow demo"
                        email_payload["verification_notice"] = "SYNTHETIC HACKATHON TEST ONLY — not a real fire alert."
                        email_id = email_payload["alert_id"]
                        email_dedupe = notification_dedupe_key(str(observation["observation_id"]),
                            "synthetic_hackathon_test_email", email_address)
                        created = add_notification(conn, notification_id=email_id,
                            observation_id=str(observation["observation_id"]), recipient=email_address,
                            channel="smtp-test", trigger_reason="synthetic_hackathon_test_email",
                            dedupe_key=email_dedupe, payload=email_payload)
                        if created:
                            delivery_results.append((email_address, dispatch_notification(conn, email_id, provider)))
                        else:
                            delivery_results.append((email_address, {"status": "duplicate_suppressed", "response": "Already attempted for this record and recipient."}))
                    for address, delivery in delivery_results:
                        st.info(f"{address}: {delivery['status']} — {delivery['response']}")
                    st.rerun()
        notifications = load_notifications(conn)
        st.subheader("Notification audit log")
        if notifications.empty:
            st.info("No notification records.")
        else:
            counts = notifications["status"].value_counts().to_dict()
            st.caption("Simulation/status counts: " + ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
            display_notifications = notifications.drop(columns=["payload_json", "recipient"], errors="ignore")
            st.dataframe(display_notifications,
                         width="stretch", hide_index=True)
            selected_notification = st.selectbox("Notification payload", notifications["notification_id"].tolist())
            row = notifications.loc[notifications["notification_id"] == selected_notification].iloc[0]
            st.json(json.loads(row["payload_json"]))
            events = notification_history(conn, selected_notification)
            st.dataframe(events, width="stretch", hide_index=True)
            if row["status"] in {"queued", "failed"} and st.button("Retry through simulation provider", key="retry_notification"):
                response = dispatch_notification(conn, selected_notification, SimulatedNotificationProvider())
                st.info(f"Retry audit result: {response['status']} (simulation only).")
                st.rerun()
            if row["channel"] == "formsubmit" and row["status"] == "failed" and st.button(
                    "Retry failed FormSubmit request", key="retry_formsubmit_notification"):
                response = dispatch_notification(conn, selected_notification,
                    FormSubmitNotificationProvider(), max_attempts=1)
                st.info(f"FormSubmit retry status: {response['status']} — {response['response']}")
                st.rerun()
            st.download_button("Export notification audit", safe_csv_bytes(display_notifications),
                               file_name="firms_notification_audit.csv", mime="text/csv")

    elif page == "Spatial and Temporal Analysis":
        st.write("Associations connect observations when both geographic distance and acquisition-time difference are within the configured limits. Connected groups help organise review; they are not fire perimeters or proof that points share an incident.")
        if scored.empty:
            st.info("Import observations to analyse spatial and temporal associations.")
        elif not bool(settings.get("association_enabled", True)):
            st.info("Association analysis is disabled in Settings and About.")
        else:
            selected = _filters(scored, "association")
            current = st.session_state.get("association_result")
            distance = float(settings.get("association_distance_km", 5))
            time_window = float(settings.get("association_window_hours", 6))
            max_rows = int(settings.get("max_association_records", 10000))
            selection_key = hashlib.sha256("|".join(selected["observation_id"].astype(str)).encode("utf-8")).hexdigest()
            st.caption(f"Analysis limit: newest {max_rows:,} filtered observations; rule: ≤{distance:g} km and ≤{time_window:g} hours. This bound keeps analysis responsive on a laptop.")
            if st.button("Calculate spatial/time associations", type="primary"):
                working = selected.sort_values("acq_datetime", ascending=False).head(max_rows).copy()
                with st.spinner(f"Associating up to {len(working):,} observations…"):
                    associated_rows, _ = associate_observations(working, distance_km=distance,
                                                               time_window_hours=time_window)
                st.session_state["association_result"] = {"cache_token": token,
                    "distance_km": distance, "time_window_hours": time_window,
                    "selection_key": selection_key,
                    "frame": associated_rows[["observation_id", "association_id",
                        "nearby_observation_count", "association_status"]].copy(),
                    "analyzed_count": len(working), "selected_count": len(selected)}
                st.rerun()
            current = st.session_state.get("association_result")
            if (not current or current.get("cache_token") != token or current.get("distance_km") != distance or
                    current.get("time_window_hours") != time_window or current.get("selection_key") != selection_key):
                st.info("Run association analysis to calculate spatial groups. Ordinary dashboard startup does not run this heavier analysis automatically.")
                return
            if current.get("selected_count", 0) > current.get("analyzed_count", 0):
                st.warning(f"Analysis covered the newest {current['analyzed_count']:,} of {current['selected_count']:,} filtered observations. Other observations have no computed association in this run.")
            summaries = []
            analyzed_ids = set(current["frame"]["observation_id"].astype(str))
            analyzed = selected.loc[selected["observation_id"].astype(str).isin(analyzed_ids)]
            for association_id, group in analyzed.dropna(subset=["association_id"]).groupby("association_id"):
                times = pd.to_datetime(group["acq_datetime"], utc=True, errors="coerce").dropna()
                satellites = sorted(group["satellite"].dropna().astype(str).unique())
                frp = pd.to_numeric(group["frp"], errors="coerce")
                summaries.append({"association_id": association_id, "observations": len(group),
                    "time_start_utc": times.min().isoformat() if len(times) else None,
                    "time_end_utc": times.max().isoformat() if len(times) else None,
                    "latitude_min": group["latitude"].min(), "latitude_max": group["latitude"].max(),
                    "longitude_min": group["longitude"].min(), "longitude_max": group["longitude"].max(),
                    "satellites": ", ".join(satellites), "distinct_satellite_count": len(satellites),
                    "max_frp_mw": frp.max() if frp.notna().any() else None})
            summary = pd.DataFrame(summaries)
            associated = analyzed.loc[analyzed["association_id"].notna()]
            unassociated = len(analyzed) - len(associated)
            not_analyzed = len(selected) - len(analyzed)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Associated observations", len(associated))
            c2.metric("Connected groups", len(summary))
            c3.metric("Analyzed without a nearby pair", unassociated)
            c4.metric("Not analyzed due to limit", not_analyzed)
            if summary.empty:
                st.info("No pairs met the current distance and time rule.")
            else:
                st.dataframe(summary.sort_values("observations", ascending=False), width="stretch", hide_index=True)
                st.download_button("Export association summary", safe_csv_bytes(summary),
                    file_name="firms_association_summary.csv", mime="text/csv")
                st.subheader("Observations in associations")
                st.dataframe(associated.drop(columns=["source_record"], errors="ignore"), width="stretch", hide_index=True)
                st.download_button("Export associated observations", safe_csv_bytes(associated),
                    file_name="firms_associated_observations.csv", mime="text/csv")

    elif page == "Data Validation":
        st.write("Every imported row is accounted for as valid or rejected. Exact duplicate records are rejected after their first occurrence; their original values and reasons remain in the audit database.")
        if imports:
            records = []
            for item in imports:
                meta, report = json.loads(item["metadata_json"]), json.loads(item["report_json"])
                records.append({"import_id": item["import_id"], **meta, **report})
            st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
            st.download_button("Export validation report", safe_csv_bytes(pd.DataFrame(records)),
                               file_name="firms_validation_report.csv", mime="text/csv")
            rejected = load_rejections(conn)
            if not rejected.empty:
                st.subheader("Rejected records (original values preserved)")
                st.dataframe(rejected.drop(columns=["source_record"], errors="ignore").head(1000),
                             width="stretch", hide_index=True)
                st.download_button("Export all rejected records", safe_csv_bytes(rejected),
                                   file_name="firms_validation_rejections.csv", mime="text/csv")
        else:
            st.info("No imports available for validation review.")

    elif page == "Evaluation":
        st.write("This page runs synthetic, labeled-as-synthetic checks of scoring behavior. It does not compare against independent fire labels and reports no real-world accuracy metrics.")
        evidence_reviews = load_feedback(conn)
        if evidence_reviews.empty:
            eligible_reviews, evidence_events = 0, 0
        else:
            eligible = evidence_reviews.loc[evidence_reviews["independent_evidence_reviewed"].astype(bool)]
            eligible_reviews = len(eligible)
            evidence_events = eligible["event_id"].replace("", pd.NA).dropna().nunique()
        st.metric("Evidence-backed reviews ready for event-aware evaluation",
                  f"{eligible_reviews} reviews / {evidence_events} events")
        st.caption("FSI/FIRMS matches are not independent labels. Event IDs keep observations from one incident together when evaluating the priority queue.")
        if st.button("Run functional scoring checks", type="primary"):
            result = run_functional_evaluation()
            save_evaluation(conn, result)
            st.session_state["last_evaluation"] = result
        if st.session_state.get("last_evaluation"):
            result = st.session_state["last_evaluation"]
            st.metric("Rule checks", f"{result['checks_passed']} / {result['checks_total']}")
            st.json(result)
        runs = load_evaluations(conn)
        if not runs.empty:
            st.subheader("Evaluation history")
            st.dataframe(runs, width="stretch", hide_index=True)
            st.download_button("Export evaluation results", safe_csv_bytes(runs),
                               file_name="firms_functional_evaluation.csv", mime="text/csv")

    elif page == "Settings and About":
        st.subheader("Priority score configuration")
        with st.form("scoring_settings"):
            new_weights = {}
            cols = st.columns(6)
            for col, key in zip(cols, DEFAULT_WEIGHTS):
                new_weights[key] = col.number_input(f"{key.title()} weight", min_value=0.0, max_value=1.0,
                    value=float(weights.get(key, DEFAULT_WEIGHTS[key])), step=.05, key=f"weight_{key}")
            half_life = st.number_input("Recency half-life (hours)", min_value=1.0, max_value=8760.0,
                                        value=float(settings.get("half_life_hours", 72)), step=6.0)
            repeat_radius = st.number_input("Repeat radius (km)", min_value=.1, max_value=50.0,
                                            value=float(settings.get("repeat_radius_km", 1)), step=.1)
            repeat_window = st.number_input("Repeat time window (hours)", min_value=1.0, max_value=168.0,
                                            value=float(settings.get("repeat_window_hours", 24)), step=1.0)
            frp_cap = st.number_input("FRP saturation scale (MW)", min_value=1.0, max_value=100000.0,
                                      value=float(settings.get("frp_cap_mw", 500)), step=10.0)
            minimum_coverage = st.number_input("Minimum factor-weight coverage", min_value=0.0, max_value=1.0,
                                      value=float(settings.get("minimum_coverage", .60)), step=.05)
            medium_threshold = st.number_input("Medium-priority score threshold", min_value=0.0, max_value=99.0,
                                      value=float(settings.get("medium_threshold", 45)), step=1.0)
            high_threshold = st.number_input("High-priority score threshold", min_value=1.0, max_value=100.0,
                                      value=float(settings.get("high_threshold", 70)), step=1.0)
            association_enabled = st.checkbox("Enable spatial/time associations", value=bool(settings.get("association_enabled", True)))
            association_distance = st.number_input("Association distance (km)", min_value=.1, max_value=500.0,
                                      value=float(settings.get("association_distance_km", 5)), step=.5)
            association_window = st.number_input("Association time window (hours)", min_value=.1, max_value=720.0,
                                      value=float(settings.get("association_window_hours", 6)), step=.5)
            max_association_records = st.number_input("Association record limit", min_value=500, max_value=50000,
                                      value=int(settings.get("max_association_records", 10000)), step=500)
            reference_mode = st.selectbox("Recency reference", ["Newest observation in dataset", "Current UTC time"],
                index=0 if settings.get("reference_mode", "Newest observation in dataset") == "Newest observation in dataset" else 1)
            max_map = st.number_input("Maximum map markers", min_value=100, max_value=20000,
                                      value=int(settings.get("max_map_markers", 5000)), step=100)
            save = st.form_submit_button("Save configuration")
        if save:
            if sum(new_weights.values()) <= 0:
                st.error("At least one weight must be greater than zero.")
            elif medium_threshold >= high_threshold:
                st.error("High-priority threshold must be greater than medium-priority threshold.")
            else:
                new_settings = {"weights": new_weights, "half_life_hours": half_life,
                    "repeat_radius_km": repeat_radius, "repeat_window_hours": repeat_window,
                    "frp_cap_mw": frp_cap, "reference_mode": reference_mode,
                    "max_map_markers": int(max_map), "minimum_coverage": minimum_coverage,
                    "medium_threshold": medium_threshold, "high_threshold": high_threshold,
                    "association_enabled": association_enabled,
                    "association_distance_km": association_distance,
                    "association_window_hours": association_window,
                    "max_association_records": int(max_association_records)}
                set_setting(conn, "dashboard", new_settings)
                cached_scored_data.clear()
                st.success("Scoring settings saved locally.")
                st.rerun()
        st.markdown("""
**Default formula.** The prior five cues retain their relative weights and are scaled to 85% when FSI context is available: recency 0.17, confidence 0.2125, FRP 0.2125, repeat detections 0.2125, and completeness 0.0425. A match to an FSI forest-area-filtered point within 1 km / 12 h contributes 0.15. With no imported FSI points, that factor is omitted and the old five-factor score is unchanged after normalization. FSI uses MODIS/VIIRS alerts and can overlap with FIRMS, so it is forest context, not independent proof of fire. Confidence maps low/nominal/high to 0.30/0.65/0.90; numeric confidence maps 0–100 linearly. FRP uses `log1p(FRP) / log1p(500 MW)`, capped at 1. Recency uses a configurable exponential half-life; by default it is relative to the newest observation in the loaded dataset. A repeat is another observation within 1 km and 24 hours; counts saturate at five. Missing signals are omitted and remaining weights are renormalized. Less than 0.60 weight coverage is labelled **INSUFFICIENT INFORMATION**. Score thresholds are low below 45, medium 45–69.99, high 70 and above. These are configurable operational heuristics, not scientifically validated fire severity estimates.

**Spatial/time associations.** Connected components use unit-sphere KD-tree candidate search and haversine-equivalent distance, then retain edges only when acquisition timestamps are within the configured time window. The default is 5 km and 6 hours. Association IDs and nearby counts are organizational cues, not evidence of one fire. Distinct satellites are reported from observed metadata only.

**Jurisdiction.** FIRMS CSVs do not reliably provide administrative boundaries. The dashboard displays “Jurisdiction unknown — manual review required” unless a reviewer explicitly verifies a configured jurisdiction. Unknown routes to the internal simulation queue.

**Delivery.** Notifications are simulated by default. A manually requested FormSubmit review email is available only for recent NASA FIRMS Area API observations and the user-configured ERS inbox; it labels the record as an unconfirmed thermal anomaly requiring human review. FormSubmit must be activated by the recipient. A provider acceptance is not proof of inbox delivery, and FireWatch does not dispatch response teams. The separate optional SMTP path remains synthetic-demo-only.

**NASA sample attribution.** NASA FIRMS Academy VIIRS SNPP sample, 2023-07-12; see `sample_data/nasa_firms_viirs_snpp_071223.json`. NASA ESDIS generally provides NASA-sponsored Earth science data as open data/CC0 unless a specific restriction applies; acknowledge NASA and do not imply NASA endorsement. FIRMS thermal anomalies are not confirmed wildfires.
""")
        st.subheader("Authority directory")
        st.caption("Contacts are user-configured only. No government or emergency contacts are pre-filled. Active entries require user confirmation of verification date.")
        with st.form("authority_form", clear_on_submit=True):
            name = st.text_input("Authority name")
            jurisdiction_name = st.text_input("Jurisdiction label")
            channel = st.selectbox("Contact channel", ["email", "webhook", "internal review"])
            contact = st.text_input("Contact detail / endpoint")
            verified = st.checkbox("I verified these contact details")
            active = st.checkbox("Mark active", value=False)
            verified_date = st.date_input("Last contact verification date", value=date.today())
            add = st.form_submit_button("Save configured authority")
        if add:
            try:
                add_authority(conn, name=name, jurisdiction=jurisdiction_name,
                              contact_channel=channel, contact_details=contact,
                              active=active, verified=verified,
                              last_verified=verified_date.isoformat() if verified else None)
                st.success("Authority configuration saved. Notification delivery remains simulation-only.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if not authorities.empty:
            st.dataframe(authorities.drop(columns=["authority_id"], errors="ignore"),
                         width="stretch", hide_index=True)
        st.subheader("Import history")
        if imports:
            st.dataframe(pd.DataFrame([{"source": i["source"], "filename": i["filename"],
                **json.loads(i["metadata_json"]), **json.loads(i["report_json"])} for i in imports]),
                width="stretch", hide_index=True)
        st.subheader("Application information")
        st.write("This Streamlit dashboard triages NASA FIRMS thermal-anomaly records for human review. It uses no image classifier, GPU, Google Earth Engine, external reverse geocoder, or automated dispatch. FormSubmit email is a manual review request for recent FIRMS API records only.")
        st.write("Map tiles are served by OpenStreetMap and require internet access. Data persistence uses a local SQLite file. For local data storage configuration use `FIREWATCH_DATA_DIR` or `FIREWATCH_DB_PATH`.")


if __name__ == "__main__":
    main()
