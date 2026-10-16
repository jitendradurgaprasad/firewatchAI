import json
from pathlib import Path

import pandas as pd
import pytest

from firewatch.evaluation import run_functional_evaluation
from firewatch.exports import safe_csv_bytes
from firewatch.firms_ingest import IngestionError, ingest_firms_csv
from firewatch.map_view import create_observation_map, map_ready_observations, nearest_observation
from firewatch.notifications import (SimulatedNotificationProvider, build_notification_payload,
                                     dispatch_notification, notification_dedupe_key)
from firewatch.ranking import score_observations
from firewatch.storage import (add_authority, add_notification, connect, latest_feedback,
                               load_feedback, load_observations, resolve_recipient, save_feedback,
                               save_import)

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


def test_ingestion_empty_and_unsupported_headers_fail_closed():
    with pytest.raises(IngestionError):
        ingest_firms_csv(b"")
    result = ingest_firms_csv(b"other,value\nx,y\n")
    assert result.report["valid_rows"] == 0
    assert result.report["invalid_rows"] == 1
    assert "missing_required_columns" in result.rejected.iloc[0]["rejection_reasons"]


def test_official_sample_actual_counts():
    result = ingest_firms_csv(SAMPLE)
    assert result.report["total_rows"] == 74605
    assert result.report["valid_rows"] == 74599
    assert result.report["duplicate_rows"] == 6
    assert result.report["rejection_reasons"] == {"duplicate_record": 6}
    assert result.valid["acq_datetime"].dt.tz is not None


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
    obs = load_observations(conn).iloc[0].to_dict()
    oid = obs["observation_id"]
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


def test_csv_formula_injection_and_functional_evaluation():
    data = safe_csv_bytes(pd.DataFrame({"text":["=1+1", " normal"], "value":[-3, 2]})).decode("utf-8-sig")
    assert "'=1+1" in data and ", -3" not in data
    evaluation = run_functional_evaluation()
    assert evaluation["checks_passed"] == evaluation["checks_total"]
    assert "not an accuracy evaluation" in evaluation["interpretation"]
