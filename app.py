"""FireWatch FIRMS observation review dashboard (simulation-only notifications)."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from dotenv import load_dotenv

from firewatch.association import associate_observations
from firewatch.evaluation import run_functional_evaluation
from firewatch.exports import investigation_report_markdown, safe_csv_bytes
from firewatch.firms_api import FirmsAPIError, PRODUCTS, fetch_recent_area
from firewatch.firms_ingest import IngestionError, MAX_CSV_BYTES, ingest_firms_csv
from firewatch.map_view import create_observation_map, nearest_observation
from firewatch.notifications import (SimulatedNotificationProvider, build_notification_payload,
                                     dispatch_notification,
                                     notification_dedupe_key)
from firewatch.ranking import DEFAULT_WEIGHTS, score_observations
from firewatch.storage import (OUTCOMES, add_authority, add_notification, connect, database_path,
                               get_setting, latest_feedback, list_authorities, list_imports,
                               load_evaluations, load_feedback, load_notifications, load_observations,
                               load_observation_sources, notification_history,
                               load_rejections, resolve_recipient,
                               save_evaluation, save_feedback, save_import, set_setting)


APP_ROOT = Path(__file__).resolve().parent
load_dotenv()
SAMPLE_PATH = APP_ROOT / "sample_data" / "nasa_firms_viirs_snpp_071223.csv"
SAMPLE_META_PATH = APP_ROOT / "sample_data" / "nasa_firms_viirs_snpp_071223.json"
NAV_ITEMS = ["Overview", "Import Data", "Observation Map", "Observation Table",
             "Priority Review Queue", "Observation Details", "Feedback and Review History",
             "Spatial and Temporal Analysis", "Notification Centre", "Data Validation", "Evaluation", "Settings and About"]


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
    return f"{row[0]}:{row[1]}:{feedback}"


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
        submitted = st.form_submit_button("Save review to audit history")
    if submitted:
        if verified and jurisdiction == "Unknown / not verified":
            st.error("Choose a configured jurisdiction before marking it verified.")
            return
        confidence = {"Not stated": None, "Low": .25, "Medium": .5, "High": .9}[confidence_choice]
        try:
            save_feedback(conn, observation_id=str(observation["observation_id"]), outcome=outcome,
                          assessment=assessment, confidence=confidence, notes=notes,
                          reviewer_id=reviewer,
                          verified_jurisdiction=jurisdiction if verified else None,
                          jurisdiction_verified=verified)
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
                       layout="wide", initial_sidebar_state="expanded")
    conn = connect()
    bootstrap_sample(conn)

    settings = get_setting(conn, "dashboard", {})
    weights = settings.get("weights", DEFAULT_WEIGHTS.copy())
    with st.sidebar:
        st.title("🔥 FireWatch AI")
        st.caption("Satellite thermal-anomaly review · Human-in-the-loop")
        page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
        st.divider()
        st.metric("Stored observations", int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]))
        st.caption("Notification delivery: SIMULATION ONLY")

    imports = list_imports(conn)
    token = _imports_token(conn)
    if imports:
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
    st.caption("FIRMS reports satellite thermal anomalies. A detection is a review cue, not confirmation of a wildfire or a measure of its severity.")

    if page == "Overview":
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
             "priority_score", "priority_category", "review_status"]].head(20),
            width="stretch", hide_index=True)
        st.warning("Priority categories are for investigation ordering only. They do not assert fire presence, severity, or emergency status.")

    elif page == "Import Data":
        st.write("Upload a FIRMS-compatible CSV. Original fields and rejected source rows are retained in the local audit database.")
        st.caption(f"Acquisition timestamps are interpreted as UTC. Interactive CSV limit: {MAX_CSV_BYTES // (1024 * 1024)} MiB. The bundled archive sample works without credentials.")
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
        st.warning("Simulation only. No email, webhook, SMS, or authority contact is sent by this application.")
        if not scored.empty:
            st.subheader("Queue an observation for simulated review")
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
            cols = st.columns(5)
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
**Default formula.** Five review cues are combined with initial weights: recency 0.20, confidence 0.25, FRP 0.25, repeat detections 0.25, and completeness 0.05. Confidence maps low/nominal/high to 0.30/0.65/0.90; numeric confidence maps 0–100 linearly. FRP uses `log1p(FRP) / log1p(500 MW)`, capped at 1. Recency uses a configurable exponential half-life; by default it is relative to the newest observation in the loaded dataset. A repeat is another observation within 1 km and 24 hours; counts saturate at five. Confidence and FRP are related sensor signals and together are capped at half the default weight. Missing signals are omitted and remaining weights are renormalized. Less than 0.60 weight coverage is labelled **INSUFFICIENT INFORMATION**. Score thresholds are low below 45, medium 45–69.99, high 70 and above. These are configurable operational heuristics, not scientifically validated fire severity estimates.

**Spatial/time associations.** Connected components use unit-sphere KD-tree candidate search and haversine-equivalent distance, then retain edges only when acquisition timestamps are within the configured time window. The default is 5 km and 6 hours. Association IDs and nearby counts are organizational cues, not evidence of one fire. Distinct satellites are reported from observed metadata only.

**Jurisdiction.** FIRMS CSVs do not reliably provide administrative boundaries. The dashboard displays “Jurisdiction unknown — manual review required” unless a reviewer explicitly verifies a configured jurisdiction. Unknown routes to the internal simulation queue.

**Delivery.** Simulated notification records only. There is no real email, webhook, SMS, or emergency-authority delivery integration.

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
        st.write("This Streamlit dashboard triages NASA FIRMS thermal-anomaly records for human review. It uses no image classifier, GPU, Google Earth Engine, external reverse geocoder, or automated dispatch.")
        st.write("Map tiles are served by OpenStreetMap and require internet access. Data persistence uses a local SQLite file. For local data storage configuration use `FIREWATCH_DATA_DIR` or `FIREWATCH_DB_PATH`.")


if __name__ == "__main__":
    main()
