import json
from pathlib import Path

import pandas as pd
import pytest

from firewatch.association import associate_observations, haversine_km
from firewatch.evaluation import run_functional_evaluation
from firewatch.exports import investigation_report_markdown, safe_csv_bytes
from firewatch.firms_api import FirmsAPIError, fetch_recent_area
from firewatch.firms_ingest import IngestionError, ingest_firms_csv
from firewatch.forest_context import ForestContextError, match_forest_context, parse_fsi_points
from firewatch.map_view import create_observation_map, map_ready_observations, nearest_observation
from firewatch.notifications import (SimulatedNotificationProvider, build_notification_payload,
                                     dispatch_notification, notification_dedupe_key)
from firewatch.ranking import score_observations
from firewatch.storage import (add_authority, add_notification, connect, latest_feedback,
                               initialize, load_feedback, load_observation_sources, load_observations, resolve_recipient, save_feedback,
                               save_forest_context_import, load_forest_context_points,
                               load_forest_context_rejections, save_import)
from firewatch.synthetic_demo import generate_synthetic_forest_demo, write_synthetic_forest_demo

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "nasa_firms_viirs_snpp_071223.csv"
BASE = "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp\n"


def test_ingestion_preserves_rows_and_rejects_invalid_and_duplicates():
    text = BASE + "10,20,2024-01-01,1234,NOAA-20,VIIRS,h,3\n10,20,2024-01-01,1234,NOAA-20,VIIRS,h,3\n91,200,nope,2500,,,,bad\n"
    result = ingest_firms_csv(text.encode())
    assert (result.report["total_rows"], result.report["valid_rows"], result.report["invalid_rows"]) == (3, 1, 2)
    assert result.report["duplicate_rows"] == 1
    assert "duplicate_record" in result.rejected.iloc[0]["rejection_reasons"]
    assert result.valid.iloc[0]["source_record"]


def test_synthetic_forest_demo_is_reproducible_valid_and_explicitly_labeled(tmp_path):
    demo = generate_synthetic_forest_demo()
    same = generate_synthetic_forest_demo()
    assert demo.observations_csv == same.observations_csv
    assert demo.context_csv == same.context_csv
    assert demo.manifest["synthetic"] is True
    assert demo.manifest["real_world_observations"] is False
    assert demo.manifest["class_counts"] == {"forest_fire": 320, "non_fire": 480}
    assert demo.manifest["event_count"] == 40
    imported = ingest_firms_csv(demo.observations_csv, source_description="SYNTHETIC demo")
    context = parse_fsi_points(demo.context_csv)
    assert imported.report["valid_rows"] == 800
    assert imported.valid["source_record"].str.contains("synthetic_truth").all()
    assert context.attrs["valid_rows"] == 480
    scored = match_forest_context(imported.valid.head(5), context)
    assert scored["forest_context"].eq(1.0).all()
    assert scored["forest_context_summary"].str.contains("SYNTHETIC-VIIRS").all()
    files = write_synthetic_forest_demo(tmp_path)
    assert Path(files["observations_path"]).parent == tmp_path.resolve()
    assert Path(files["manifest_path"]).is_file()
def test_ingestion_empty_and_unsupported_headers_fail_closed():
    with pytest.raises(IngestionError):
        ingest_firms_csv(b"")
    result = ingest_firms_csv(b"other,value\nx,y\n")
    assert result.report["valid_rows"] == 0
    assert result.report["invalid_rows"] == 1
    assert "missing_required_columns" in result.rejected.iloc[0]["rejection_reasons"]
    with pytest.raises(IngestionError):
        ingest_firms_csv((BASE + '"unterminated,record').encode())
    with pytest.raises(IngestionError, match="size limit"):
        ingest_firms_csv((BASE + "1").encode(), max_bytes=4)
    with pytest.raises(IngestionError, match="rows"):
        ingest_firms_csv((BASE + "10,20,2024-01-01,1234,,,,\n" * 2).encode(), max_rows=1)


def test_official_sample_actual_counts():
    import hashlib
    result = ingest_firms_csv(SAMPLE)
    manifest = json.loads(SAMPLE.with_suffix(".json").read_text(encoding="utf-8"))
    assert SAMPLE.stat().st_size == manifest["size_bytes"]
    assert hashlib.sha256(SAMPLE.read_bytes()).hexdigest() == manifest["source_sha256"]
    assert result.report["total_rows"] == 74605
    assert result.report["valid_rows"] == 74599
    assert result.report["duplicate_rows"] == 6
    assert result.report["rejection_reasons"] == {"duplicate_record": 6}
    assert result.valid["acq_datetime"].dt.tz is not None
    assert result.valid["confidence"].value_counts().to_dict() == {"nominal":64895, "high":4912, "low":4792}
    assert result.valid["version"].value_counts().to_dict() == {"2.0NRT":69507, "2.0URT":5092}
    raw = pd.read_csv(SAMPLE)
    assert raw["confidence"].value_counts().to_dict() == {"n":64901, "h":4912, "l":4792}
    assert raw["version"].value_counts().to_dict() == {"2.0NRT":69507, "2.0URT":5098}
    assert result.valid["acq_date"].nunique() == 1


def test_scoring_is_explainable_bounded_and_abstains_on_missing():
    df = pd.DataFrame([
        {"observation_id":"a", "latitude":0, "longitude":0, "acq_datetime":"2026-01-01T00:00:00Z", "confidence":"high", "frp":100},
        {"observation_id":"b", "latitude":20, "longitude":20, "acq_datetime":"2025-01-01T00:00:00Z", "confidence":"low", "frp":.1},
        {"observation_id":"c", "latitude":40, "longitude":40, "acq_datetime":"2026-01-01T00:00:00Z", "confidence":None, "frp":None},
    ])
    out = score_observations(df, reference_time="2026-01-01T00:00:00Z")
    assert out.priority_score.between(0,100).all()
    assert out.iloc[0].priority_score > out.iloc[1].priority_score
    assert out.iloc[2].priority_category == "INSUFFICIENT INFORMATION"
    assert "Not scored because missing" in out.iloc[2].explanation
    assert score_observations(df, reference_time="2026-01-01T00:00:00Z").priority_score.tolist() == out.priority_score.tolist()


def test_scoring_rejects_bad_weights():
    frame = pd.DataFrame([{"latitude":0,"longitude":0,"acq_datetime":"2026-01-01","confidence":"high","frp":1}])
    with pytest.raises(ValueError):
        score_observations(frame, weights={"recency":-1,"confidence":0,"frp":0,"repeat":0,"completeness":0})


def test_fsi_context_csv_validates_rows_and_preserves_portal_fields():
    csv = ("Fire Date,Time,Source,Lat,Lon,State,District,Division,Range,Beat\n"
           "2026-10-15,01:09:55,VIIRS,17.464672,80.653850,TELANGANA,DISTRICT,DIVISION,RANGE,BEAT\n"
           "2026-10-15,01:09:55,VIIRS,17.464672,80.653850,TELANGANA,DISTRICT,DIVISION,RANGE,BEAT\n"
           "2026-10-15,01:09:55,MODIS,999,80,TELANGANA,DISTRICT,DIVISION,RANGE,BEAT\n")
    points = parse_fsi_points(csv.encode())
    assert points.attrs == {"filename":"fsi_forest_fire_points.csv", "total_rows":3,
                            "valid_rows":1, "rejected_rows":2}
    assert points.iloc[0].fire_datetime == "2026-10-15T01:09:55"
    assert points.iloc[0].division == "DIVISION"
    assert points.iloc[1].rejection_reason == "duplicate_point"
    assert "invalid_latitude" in points.iloc[2].rejection_reason
    with pytest.raises(ForestContextError, match="Missing required FSI columns"):
        parse_fsi_points(b"lat,lon\n10,20\n")


def test_fsi_spatiotemporal_match_is_context_only_and_changes_review_rank():
    rows = pd.DataFrame([
        {"observation_id":"match", "latitude":17.46467, "longitude":80.65385,
         "acq_datetime":"2026-10-15T01:00:00Z", "confidence":"nominal", "frp":10},
        {"observation_id":"far", "latitude":12, "longitude":75,
         "acq_datetime":"2026-10-15T01:00:00Z", "confidence":"nominal", "frp":10},
        {"observation_id":"late", "latitude":17.46467, "longitude":80.65385,
         "acq_datetime":"2026-10-16T14:00:00Z", "confidence":"nominal", "frp":10},
    ])
    points = parse_fsi_points(("Fire Date,Time,Source,Lat,Lon,State\n"
                               "2026-10-15,01:09:55,VIIRS,17.464672,80.653850,TELANGANA\n").encode())
    matched = match_forest_context(rows, points)
    assert matched.forest_context.tolist()[:2] == [1.0, None]
    assert matched.forest_context_matches.tolist() == [1, 0, 0]
    baseline = score_observations(rows, reference_time="2026-10-16T14:00:00Z")
    forest_scored = score_observations(matched, reference_time="2026-10-16T14:00:00Z")
    assert baseline.score_coverage.eq(1.0).all()
    assert forest_scored.score_coverage.eq(1.0).all()
    assert forest_scored.iloc[0].priority_score > baseline.iloc[0].priority_score
    assert forest_scored.iloc[1].priority_score == pytest.approx(baseline.iloc[1].priority_score)
    assert "not ground truth" in forest_scored.iloc[0].explanation
    # Old saved five-factor user settings reproduce the original uncontextualized score.
    legacy = score_observations(rows, weights={"recency":.20,"confidence":.25,"frp":.25,
                                "repeat":.25,"completeness":.05},
                                reference_time="2026-10-16T14:00:00Z")
    assert legacy.priority_score.tolist() == baseline.priority_score.tolist()


def test_fsi_context_persistence_and_evidence_backed_review_gate(tmp_path):
    conn = connect(tmp_path / "forest.sqlite")
    rows = ingest_firms_csv((BASE + "17.464672,80.653850,2026-10-15,0100,N20,VIIRS,n,10\n").encode())
    save_import(conn, rows, import_id="firms")
    points = parse_fsi_points(("Fire Date,Time,Source,Lat,Lon,State\n"
                               "2026-10-15,01:09:55,VIIRS,17.464672,80.653850,TELANGANA\n").encode())
    save_forest_context_import(conn, points, import_id="fsi", source="FSI", filename="fsi.csv",
                               retrieved_at="2026-10-16T00:00:00Z")
    assert len(load_forest_context_points(conn)) == 1
    assert load_forest_context_rejections(conn).empty
    observation_id = load_observations(conn).iloc[0]["observation_id"]
    with pytest.raises(ValueError, match="source, reference, and event ID"):
        save_feedback(conn, observation_id=observation_id,
                      outcome="Confirmed fire based on available evidence", assessment="confirmed",
                      confidence=.9, independent_evidence_reviewed=True)
    save_feedback(conn, observation_id=observation_id,
                  outcome="Confirmed fire based on available evidence", assessment="confirmed",
                  confidence=.9, evidence_source="Field inspection / fire service report",
                  evidence_reference="inspection-42", event_id="event-1",
                  independent_evidence_reviewed=True)
    feedback = load_feedback(conn, observation_id).iloc[0]
    assert feedback.evidence_reference == "inspection-42"
    assert feedback.event_id == "event-1" and feedback.independent_evidence_reviewed == 1
    conn.close()


def test_repeat_detection_uses_utc_elapsed_hours_not_pandas_storage_units():
    rows = pd.DataFrame([
        {"observation_id":"a", "latitude":0, "longitude":0, "acq_datetime":"2026-01-01T00:00:00Z", "confidence":"high", "frp":1},
        {"observation_id":"b", "latitude":0, "longitude":.001, "acq_datetime":"2026-01-01T01:00:00Z", "confidence":"high", "frp":1},
        {"observation_id":"c", "latitude":0, "longitude":.001, "acq_datetime":"2026-01-02T02:00:00Z", "confidence":"high", "frp":1},
    ])
    scored = score_observations(rows, reference_time="2026-01-02T02:00:00Z")
    assert scored.repeat_count.tolist() == [1, 1, 0]


def test_map_coordinate_validation_nearest_and_rendering():
    frame = pd.DataFrame([{"observation_id":"x","latitude":10,"longitude":20,"priority_category":"HIGH REVIEW PRIORITY"},
                          {"observation_id":"bad","latitude":100,"longitude":20}])
    ready, excluded = map_ready_observations(frame)
    assert len(ready) == 1 and excluded == 1
    assert nearest_observation(ready, 10, 20) == "x"
    assert nearest_observation(ready, -20, -40) is None
    fmap, shown, excluded = create_observation_map(frame)
    assert fmap is not None and shown == 1 and excluded == 1
    assert "HIGH REVIEW PRIORITY" in fmap.get_root().render()
    assert create_observation_map(frame.iloc[0:0])[0] is None


def test_feedback_audit_authority_routing_and_notification_dedupe(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    result = ingest_firms_csv((BASE + "10,20,2024-01-01,1234,NOAA-20,VIIRS,h,3\n").encode())
    save_import(conn, result, import_id="import-1")
    second = ingest_firms_csv((BASE + "10,20,2024-01-01,1234,NOAA-20,VIIRS,h,3\n").encode(),
                              source_description="Second source copy")
    save_import(conn, second, import_id="import-2")
    obs = load_observations(conn).iloc[0].to_dict()
    oid = obs["observation_id"]
    assert len(load_observation_sources(conn, oid)) == 2
    save_feedback(conn, observation_id=oid, outcome="Unable to determine", assessment="Pending", confidence=.5, reviewer_id="tester")
    save_feedback(conn, observation_id=oid, outcome="Possible false alarm", assessment="Recheck", confidence=.7, reviewer_id="tester")
    assert len(load_feedback(conn, oid)) == 2
    assert latest_feedback(conn)[oid]["outcome"] == "Possible false alarm"
    assert resolve_recipient(conn, None)["name"] == "Internal review queue"
    add_authority(conn, name="County Desk", jurisdiction="X", contact_channel="email", contact_details="test@example.invalid", active=True, verified=True, last_verified="2026-10-01")
    assert resolve_recipient(conn, "X")["name"] == "County Desk"
    obs.update(priority_score=75, priority_category="HIGH REVIEW PRIORITY", explanation="transparent factors")
    payload = build_notification_payload(obs, trigger_reason="review", routing_reason="internal", recipient="Internal")
    assert "not a confirmed wildfire" in payload["verification_notice"]
    key = notification_dedupe_key(oid, "review")
    assert add_notification(conn, notification_id=payload["alert_id"], observation_id=oid, recipient="Internal", channel="simulation", trigger_reason="review", dedupe_key=key, payload=payload)
    assert not add_notification(conn, notification_id="other", observation_id=oid, recipient="Internal", channel="simulation", trigger_reason="review", dedupe_key=key, payload=payload)
    assert conn.execute("select duplicate_attempts from notifications where notification_id=?", (payload["alert_id"],)).fetchone()[0] == 1
    assert conn.execute("select status from notification_events order by event_id desc limit 1").fetchone()[0] == "duplicate_suppressed"
    result = dispatch_notification(conn, payload["alert_id"], SimulatedNotificationProvider())
    assert result["status"] == "simulated"
    assert conn.execute("select status from notifications").fetchone()[0] == "simulated"
    conn.close()


def test_notification_retries_are_audited(tmp_path):
    conn = connect(tmp_path / "retry.sqlite")
    result = ingest_firms_csv((BASE + "10,20,2024-01-01,1234,,,,\n").encode())
    save_import(conn, result, import_id="retry-import")
    oid = load_observations(conn).iloc[0]["observation_id"]
    payload = {"recipient":"queue"}
    add_notification(conn, notification_id="n1", observation_id=oid, recipient="queue", channel="test", trigger_reason="test", dedupe_key="d1", payload=payload)
    class Provider:
        attempts = 0
        def deliver(self, payload, recipient):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary")
            return {"status":"provider-accepted", "response":"accepted by fake provider"}
    provider = Provider()
    outcome = dispatch_notification(conn, "n1", provider, max_attempts=2)
    assert outcome["status"] == "provider-accepted" and outcome["attempts"] == 2
    statuses = [r[0] for r in conn.execute("select status from notification_events order by event_id")]
    assert statuses == ["queued", "failed", "queued", "provider-accepted"]
    conn.close()


def test_csv_formula_injection_and_functional_evaluation(tmp_path):
    data = safe_csv_bytes(pd.DataFrame({"text":["=1+1", " normal"], "value":[-3, 2]})).decode("utf-8-sig")
    assert "'=1+1" in data and ", -3" not in data
    evaluation = run_functional_evaluation()
    assert evaluation["checks_passed"] == evaluation["checks_total"]
    assert "not an accuracy evaluation" in evaluation["interpretation"]
    conn = connect(tmp_path / "evaluation.sqlite")
    from firewatch.storage import load_evaluations, save_evaluation
    save_evaluation(conn, evaluation)
    assert json.loads(load_evaluations(conn).iloc[0]["result_json"])["checks_passed"] == 11
    conn.close()


def test_import_normalizes_aliases_optional_missing_and_blank_lines():
    csv = "lat,lon,date,time,platform,sensor,conf,fire_radiative_power\n10,20,2024/02/01,5,N21,VIIRS,medium,4.5\n\n"
    result = ingest_firms_csv(csv.encode())
    assert result.report["total_rows"] == 2
    assert result.report["valid_rows"] == 1
    assert result.report["invalid_rows"] == 1
    assert result.valid.iloc[0]["acq_time"] == "0005"
    assert result.valid.iloc[0]["confidence"] == "nominal"
    assert "frp" in result.metadata["available_fields"]
    assert "invalid_latitude" in result.rejected.iloc[0]["rejection_reasons"]


def test_invalid_measurements_timestamp_and_confidence_are_rejected():
    csv = BASE + "0,0,not-a-date,2561,VIIRS,VIIRS,impossible,-4\n"
    result = ingest_firms_csv(csv.encode())
    reason = result.rejected.iloc[0]["rejection_reasons"]
    assert "missing_or_invalid_acq_date" in reason
    assert "missing_or_invalid_acq_time" in reason
    assert "unexpected_confidence" in reason
    assert "invalid_frp" in reason


def test_priority_threshold_boundaries_and_factor_contributions():
    rows = pd.DataFrame([{"observation_id":str(score), "latitude":i*20, "longitude":i*20,
        "acq_datetime":"2026-01-01T00:00:00Z", "confidence":score, "frp":None}
        for i, score in enumerate((44.99, 45, 69.99, 70))])
    weights = {"recency":0, "confidence":1, "frp":0, "repeat":0, "completeness":0}
    out = score_observations(rows, weights=weights, reference_time="2026-01-01T00:00:00Z")
    assert out.priority_category.tolist() == ["LOW REVIEW PRIORITY", "MEDIUM REVIEW PRIORITY",
                                               "MEDIUM REVIEW PRIORITY", "HIGH REVIEW PRIORITY"]
    assert all(abs(sum(row.contributing_factors.values()) - row.priority_score / 100) < 1e-5
               for _, row in out.iterrows())
    assert "confidence: 45.0 points" in out.iloc[1].explanation
    with pytest.raises(ValueError):
        score_observations(rows, medium_threshold=70, high_threshold=70)


def test_haversine_and_spatiotemporal_connected_associations():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=.01)
    frame = pd.DataFrame([
        {"observation_id":"a", "latitude":0, "longitude":0, "acq_datetime":"2026-01-01T00:00:00Z", "satellite":"N20", "frp":1},
        {"observation_id":"b", "latitude":0, "longitude":.01, "acq_datetime":"2026-01-01T01:00:00Z", "satellite":"N21", "frp":2},
        {"observation_id":"c", "latitude":0, "longitude":.02, "acq_datetime":"2026-01-01T20:00:00Z", "satellite":"N21", "frp":3},
        {"observation_id":"d", "latitude":0, "longitude":0, "acq_datetime":"bad", "satellite":"N20", "frp":None},
    ])
    associated, summary = associate_observations(frame, distance_km=2, time_window_hours=6)
    assert associated.iloc[0].association_id == associated.iloc[1].association_id
    assert pd.isna(associated.iloc[2].association_id)
    assert associated.iloc[3].association_status == "missing_or_invalid_coordinate_or_time"
    assert associated.iloc[0].nearby_observation_count == 1
    assert summary.iloc[0].observation_count == 2
    assert summary.iloc[0].distinct_satellite_count == 2
    with pytest.raises(ValueError):
        haversine_km(91, 0, 0, 0)


def test_api_client_is_keyed_bounded_and_does_not_leak_credentials(monkeypatch):
    key = "a" * 32
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(FirmsAPIError, match="FIRMS_MAP_KEY"):
        fetch_recent_area(area="60,5,100,38")
    with pytest.raises(ValueError):
        fetch_recent_area(area="100,5,60,38", map_key=key)
    class Response:
        headers = {"Content-Length":"78"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, amount): return b"latitude,longitude,acq_date,acq_time\n10,20,2026-01-01,1234\n"
    import firewatch.firms_api as api
    captured = {}
    def fake_open(request, timeout):
        captured["url"] = request.full_url
        return Response()
    monkeypatch.setattr(api.urllib.request, "urlopen", fake_open)
    data = fetch_recent_area(area="60,5,100,38", product="VIIRS_NOAA20_NRT", day_range=1, map_key=key)
    assert data.content.startswith(b"latitude")
    assert ingest_firms_csv(data.content, original_filename=data.filename).metadata["filename"] == data.filename
    assert key in captured["url"]
    assert key not in data.source and key not in data.filename


def test_old_feedback_schema_migrates_without_losing_rows(tmp_path):
    import sqlite3
    db = tmp_path / "old.sqlite"
    raw = sqlite3.connect(db)
    raw.executescript("CREATE TABLE feedback(feedback_id INTEGER PRIMARY KEY, observation_id TEXT, outcome TEXT, assessment TEXT, confidence REAL, notes TEXT, reviewer_id TEXT, created_at TEXT); INSERT INTO feedback VALUES(1,'x','Awaiting review','old',NULL,'','', '2025-01-01');")
    raw.close()
    conn = connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    assert {"verified_jurisdiction", "jurisdiction_verified"}.issubset(cols)
    assert conn.execute("SELECT assessment FROM feedback WHERE feedback_id=1").fetchone()[0] == "old"
    conn.close()


def test_investigation_report_contains_disclaimer_provenance_and_history():
    obs = {"observation_id":"obs-x", "source_name":"Example", "source_attribution":"NASA attribution",
           "latitude":1, "longitude":2, "priority_score":81, "priority_category":"HIGH REVIEW PRIORITY",
           "explanation":"confidence contributes 20 points", "source_record":"{\"latitude\": \"1\"}"}
    report = investigation_report_markdown(obs)
    assert "does not confirm a wildfire" in report
    assert "NASA attribution" in report
    assert "application-generated" in report


def test_streamlit_empty_state_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWATCH_DATA_DIR", str(tmp_path / "streamlit-empty"))
    monkeypatch.setenv("FIREWATCH_AUTOLOAD_SAMPLE", "0")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("No observations are loaded" in item.value for item in at.info)
    for page in ("Import Data", "Observation Map", "Observation Table", "Priority Review Queue",
                 "Observation Details", "Feedback and Review History", "Spatial and Temporal Analysis",
                 "Notification Centre", "Data Validation", "Evaluation", "Settings and About"):
        at.radio[0].set_value(page).run()
        assert not at.exception, f"Empty-state page failed: {page}"


def test_streamlit_populated_spatial_analysis_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWATCH_DATA_DIR", str(tmp_path / "streamlit-synthetic"))
    monkeypatch.setenv("FIREWATCH_AUTOLOAD_SAMPLE", "0")
    conn = connect(tmp_path / "streamlit-synthetic" / "firewatch.sqlite3")
    csv = BASE + (
        "10,20,2026-01-01,0000,N20,VIIRS,n,2\n"
        "10,20.001,2026-01-01,0100,N21,VIIRS,n,3\n"
        "40,40,2026-01-02,1200,N20,VIIRS,h,4\n")
    result = ingest_firms_csv(csv.encode(), source_description="Synthetic UI test data")
    save_import(conn, result, import_id="synthetic-ui-import")
    conn.close()
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.radio[0].set_value("Spatial and Temporal Analysis").run()
    assert not at.exception
    next(button for button in at.button if button.label == "Calculate spatial/time associations").click().run()
    assert not at.exception
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Associated observations"] == "2"
    assert metrics["Connected groups"] == "1"
    assert metrics["Analyzed without a nearby pair"] == "1"
    assert metrics["Not analyzed due to limit"] == "0"
