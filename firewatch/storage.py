"""SQLite persistence for imports, append-only review feedback and notifications."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTCOMES = (
    "Confirmed fire based on available evidence",
    "Likely fire; additional verification needed",
    "Possible false alarm",
    "Non-fire thermal source",
    "Unable to determine",
    "Awaiting review",
    "Duplicate",
    "Inconclusive",
    "Unable to verify",
)


def default_data_dir() -> Path:
    configured = os.environ.get("FIREWATCH_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "FireWatchAI-dashboard"
    return Path.home() / ".local" / "share" / "firewatch-ai"


def database_path() -> Path:
    return Path(os.environ.get("FIREWATCH_DB_PATH", default_data_dir() / "firewatch.sqlite3")).expanduser().resolve()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    initialize(conn)
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS imports (
        import_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        filename TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        import_id TEXT NOT NULL REFERENCES imports(import_id),
        source_record TEXT NOT NULL,
        normalized_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS observation_sources (
        observation_id TEXT NOT NULL REFERENCES observations(observation_id),
        import_id TEXT NOT NULL REFERENCES imports(import_id),
        source_row_number INTEGER NOT NULL,
        PRIMARY KEY(observation_id, import_id, source_row_number)
    );
    CREATE INDEX IF NOT EXISTS idx_observations_datetime ON observations(json_extract(normalized_json, '$.acq_datetime'));
    CREATE TABLE IF NOT EXISTS import_rejections (
        rejection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        import_id TEXT NOT NULL REFERENCES imports(import_id),
        source_row_number INTEGER,
        reason TEXT NOT NULL,
        source_record TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS forest_context_imports (
        import_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        filename TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS forest_context_points (
        point_id TEXT PRIMARY KEY,
        import_id TEXT NOT NULL REFERENCES forest_context_imports(import_id),
        source_row_number INTEGER NOT NULL,
        point_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(import_id, source_row_number)
    );
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_id TEXT NOT NULL REFERENCES observations(observation_id),
        outcome TEXT NOT NULL,
        assessment TEXT NOT NULL,
        confidence REAL,
        notes TEXT NOT NULL DEFAULT '',
        reviewer_id TEXT NOT NULL DEFAULT '',
        verified_jurisdiction TEXT,
        jurisdiction_verified INTEGER NOT NULL DEFAULT 0,
        evidence_source TEXT NOT NULL DEFAULT '',
        evidence_reference TEXT NOT NULL DEFAULT '',
        event_id TEXT NOT NULL DEFAULT '',
        independent_evidence_reviewed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_feedback_observation ON feedback(observation_id, feedback_id DESC);
    CREATE TABLE IF NOT EXISTS authorities (
        authority_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        jurisdiction TEXT NOT NULL,
        contact_channel TEXT NOT NULL,
        contact_details TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 0,
        verified INTEGER NOT NULL DEFAULT 0,
        last_verified TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(name, jurisdiction, contact_channel, contact_details)
    );
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id TEXT PRIMARY KEY,
        observation_id TEXT NOT NULL REFERENCES observations(observation_id),
        recipient TEXT NOT NULL,
        channel TEXT NOT NULL,
        trigger_reason TEXT NOT NULL,
        dedupe_key TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        attempted_at TEXT,
        provider_response TEXT NOT NULL DEFAULT '',
        retry_count INTEGER NOT NULL DEFAULT 0,
        duplicate_attempts INTEGER NOT NULL DEFAULT 0,
        error_details TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS notification_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        notification_id TEXT NOT NULL REFERENCES notifications(notification_id),
        status TEXT NOT NULL,
        details TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)
    feedback_columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "verified_jurisdiction" not in feedback_columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN verified_jurisdiction TEXT")
    if "jurisdiction_verified" not in feedback_columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN jurisdiction_verified INTEGER NOT NULL DEFAULT 0")
    for column, definition in (
        ("evidence_source", "TEXT NOT NULL DEFAULT ''"),
        ("evidence_reference", "TEXT NOT NULL DEFAULT ''"),
        ("event_id", "TEXT NOT NULL DEFAULT ''"),
        ("independent_evidence_reviewed", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if column not in feedback_columns:
            conn.execute(f"ALTER TABLE feedback ADD COLUMN {column} {definition}")
    notification_columns = {row[1] for row in conn.execute("PRAGMA table_info(notifications)")}
    if "duplicate_attempts" not in notification_columns:
        conn.execute("ALTER TABLE notifications ADD COLUMN duplicate_attempts INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def _json_value(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def save_import(conn: sqlite3.Connection, result, *, import_id: str) -> None:
    metadata = dict(result.metadata)
    report = dict(result.report)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO imports VALUES(?, ?, ?, ?, ?, ?, ?)",
                 (import_id, metadata["source"], metadata["filename"], metadata["retrieved_at"],
                  json.dumps(metadata, ensure_ascii=False), json.dumps(report, ensure_ascii=False), now))
    for row in result.valid.to_dict(orient="records"):
        source_record = row.get("source_record", "{}")
        normalized = {k: _json_value(v) for k, v in row.items() if k != "source_record"}
        normalized.update({"source_name": metadata["source"],
                           "product": metadata.get("product", ""),
                           "source_retrieved_at": metadata["retrieved_at"],
                           "source_attribution": metadata.get("licence_attribution", ""),
                           "source_quality_warnings": metadata.get("quality_warnings", []),
                           "available_fields": metadata.get("available_fields", []),
                           "source_filename": metadata.get("filename", ""),
                           "source_file_sha256": metadata.get("file_sha256", ""),
                           "source_file_size_bytes": metadata.get("file_size_bytes"),
                           "import_id": import_id})
        conn.execute("INSERT OR IGNORE INTO observations VALUES(?, ?, ?, ?, ?)",
                     (row["observation_id"], import_id, source_record,
                      json.dumps(normalized, ensure_ascii=False, allow_nan=False), now))
        conn.execute("INSERT OR IGNORE INTO observation_sources(observation_id,import_id,source_row_number) VALUES(?,?,?)",
                     (row["observation_id"], import_id, int(row.get("source_row_number", 0))))
    for row in result.rejected.to_dict(orient="records"):
        conn.execute("INSERT INTO import_rejections(import_id, source_row_number, reason, source_record) VALUES(?, ?, ?, ?)",
                     (import_id, row.get("source_row_number"), row.get("rejection_reasons", "invalid"),
                      row.get("source_record", "{}")))
    conn.commit()


def list_imports(conn: sqlite3.Connection) -> list[dict]:
    return [dict(row) for row in conn.execute("SELECT * FROM imports ORDER BY created_at DESC")]


def save_forest_context_import(conn: sqlite3.Connection, points: pd.DataFrame, *, import_id: str,
                               source: str, filename: str, retrieved_at: str,
                               metadata: dict | None = None) -> None:
    """Persist forest-context points and their row audit separately from FIRMS observations."""
    metadata = dict(metadata or {})
    valid = points.loc[points["rejection_reason"].eq("")] if not points.empty else points
    rejected = points.loc[points["rejection_reason"].ne("")] if not points.empty else points
    now = datetime.now(timezone.utc).isoformat()
    report = {"total_rows": int(len(points)), "valid_rows": int(len(valid)),
              "rejected_rows": int(len(rejected))}
    conn.execute("INSERT INTO forest_context_imports VALUES(?,?,?,?,?,?,?)",
                 (import_id, source, filename, retrieved_at,
                  json.dumps(metadata, ensure_ascii=False), json.dumps(report), now))
    for row in points.to_dict(orient="records"):
        point_id = f"{import_id}:{row['source_row_number']}"
        row["context_source"] = source
        conn.execute("INSERT INTO forest_context_points VALUES(?,?,?,?,?)",
                     (point_id, import_id, int(row["source_row_number"]),
                      json.dumps(row, ensure_ascii=False, allow_nan=False), now))
    conn.commit()


def list_forest_context_imports(conn: sqlite3.Connection) -> list[dict]:
    return [dict(row) for row in conn.execute(
        "SELECT * FROM forest_context_imports ORDER BY created_at DESC")]


def load_forest_context_points(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT point_json FROM forest_context_points ORDER BY created_at, point_id").fetchall()
    if not rows:
        return pd.DataFrame(columns=["fire_datetime", "source_sensor", "latitude", "longitude",
                                     "rejection_reason", "state", "district", "division", "range", "beat"])
    return pd.DataFrame([json.loads(row[0]) for row in rows])


def load_forest_context_rejections(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("""SELECT i.import_id, i.filename, p.source_row_number, p.point_json
                          FROM forest_context_points p JOIN forest_context_imports i USING(import_id)
                          WHERE json_extract(p.point_json, '$.rejection_reason') != ''
                          ORDER BY p.point_id DESC""").fetchall()
    records = []
    for row in rows:
        point = json.loads(row["point_json"])
        records.append({"import_id": row["import_id"], "filename": row["filename"],
                        "source_row_number": row["source_row_number"],
                        "rejection_reason": point.get("rejection_reason", ""),
                        "source_record": json.dumps(point.get("source_record", {}), ensure_ascii=False)})
    return pd.DataFrame(records)


def load_observations(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT normalized_json FROM observations ORDER BY created_at, observation_id").fetchall()
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame([json.loads(row[0]) for row in rows])
    if "acq_datetime" in result:
        result["acq_datetime"] = pd.to_datetime(result["acq_datetime"], utc=True, errors="coerce")
    return result


def load_rejections(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("""SELECT r.import_id, r.source_row_number, r.reason, r.source_record,
                                 i.source, i.filename, i.retrieved_at
                          FROM import_rejections r JOIN imports i USING(import_id)
                          ORDER BY r.rejection_id DESC""").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def load_observation_sources(conn: sqlite3.Connection, observation_id: str) -> pd.DataFrame:
    """Return every import/file/row that supplied this deduplicated observation."""
    rows = conn.execute("""SELECT s.source_row_number, i.import_id, i.source, i.filename,
                          i.retrieved_at, i.metadata_json
                          FROM observation_sources s JOIN imports i USING(import_id)
                          WHERE s.observation_id=? ORDER BY i.created_at,s.source_row_number""",
                        (observation_id,)).fetchall()
    result = pd.DataFrame([dict(row) for row in rows])
    if not result.empty:
        result["product"] = result["metadata_json"].map(lambda raw: json.loads(raw).get("product", ""))
        result = result.drop(columns=["metadata_json"])
    return result


def save_feedback(conn: sqlite3.Connection, *, observation_id: str, outcome: str,
                  assessment: str, confidence: float | None, notes: str = "",
                  reviewer_id: str = "", verified_jurisdiction: str | None = None,
                  jurisdiction_verified: bool = False, evidence_source: str = "",
                  evidence_reference: str = "", event_id: str = "",
                  independent_evidence_reviewed: bool = False) -> int:
    if outcome not in OUTCOMES:
        raise ValueError("Unsupported review outcome.")
    if confidence is not None and not 0 <= float(confidence) <= 1:
        raise ValueError("Reviewer confidence must be from 0 to 1.")
    stamp = datetime.now(timezone.utc).isoformat()
    if jurisdiction_verified and not verified_jurisdiction:
        raise ValueError("A verified jurisdiction must be specified when marking it verified.")
    if independent_evidence_reviewed and not all((evidence_source.strip(), evidence_reference.strip(), event_id.strip())):
        raise ValueError("Independent evidence review requires its source, reference, and event ID.")
    if independent_evidence_reviewed and outcome not in {
        "Confirmed fire based on available evidence", "Non-fire thermal source"
    }:
        raise ValueError("Only a decisive fire/non-fire assessment can be marked for evaluation.")
    cur = conn.execute("""INSERT INTO feedback(observation_id,outcome,assessment,confidence,notes,reviewer_id,
                          verified_jurisdiction,jurisdiction_verified,evidence_source,evidence_reference,
                          event_id,independent_evidence_reviewed,created_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (observation_id, outcome, assessment.strip() or outcome,
                        float(confidence) if confidence is not None else None,
                        notes.strip(), reviewer_id.strip(), verified_jurisdiction,
                        int(jurisdiction_verified), evidence_source.strip(), evidence_reference.strip(),
                        event_id.strip(), int(independent_evidence_reviewed), stamp))
    conn.commit()
    return int(cur.lastrowid)


def load_feedback(conn: sqlite3.Connection, observation_id: str | None = None) -> pd.DataFrame:
    if observation_id:
        rows = conn.execute("SELECT * FROM feedback WHERE observation_id=? ORDER BY feedback_id DESC", (observation_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM feedback ORDER BY feedback_id DESC").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def latest_feedback(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute("""SELECT f.* FROM feedback f JOIN
                          (SELECT observation_id, MAX(feedback_id) AS newest FROM feedback GROUP BY observation_id) x
                          ON f.feedback_id=x.newest""").fetchall()
    return {row["observation_id"]: dict(row) for row in rows}


def add_authority(conn: sqlite3.Connection, *, name: str, jurisdiction: str,
                  contact_channel: str, contact_details: str, active: bool,
                  verified: bool, last_verified: str | None) -> None:
    if not all((name.strip(), jurisdiction.strip(), contact_channel.strip(), contact_details.strip())):
        raise ValueError("Authority name, jurisdiction, channel, and contact details are required.")
    if active and (not verified or not last_verified):
        raise ValueError("An active authority contact must be marked verified and include its last verification date.")
    conn.execute("""INSERT INTO authorities(name,jurisdiction,contact_channel,contact_details,active,verified,last_verified,created_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(name,jurisdiction,contact_channel,contact_details) DO UPDATE SET
                    active=excluded.active, verified=excluded.verified, last_verified=excluded.last_verified""",
                 (name.strip(), jurisdiction.strip(), contact_channel.strip(), contact_details.strip(),
                  int(active), int(verified), last_verified,
                  datetime.now(timezone.utc).isoformat()))
    conn.commit()


def list_authorities(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM authorities ORDER BY jurisdiction,name", conn)


def resolve_recipient(conn: sqlite3.Connection, jurisdiction: str | None) -> dict:
    if jurisdiction and jurisdiction.strip():
        row = conn.execute("""SELECT * FROM authorities WHERE lower(jurisdiction)=lower(?)
                             AND active=1 AND verified=1 ORDER BY last_verified DESC LIMIT 1""",
                           (jurisdiction.strip(),)).fetchone()
        if row:
            return {**dict(row), "routing_reason": f"Verified configured jurisdiction match: {jurisdiction.strip()}"}
    return {"authority_id": None, "name": "Internal review queue", "jurisdiction": "Unknown",
            "contact_channel": "simulation", "contact_details": "Internal simulation queue",
            "active": 1, "verified": 1,
            "routing_reason": "Jurisdiction unknown or no verified active match; routed to internal review queue."}


def add_notification(conn: sqlite3.Connection, *, notification_id: str, observation_id: str,
                     recipient: str, channel: str, trigger_reason: str,
                     dedupe_key: str, payload: dict) -> bool:
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute("""INSERT INTO notifications(notification_id,observation_id,recipient,channel,trigger_reason,
                       dedupe_key,payload_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                     (notification_id, observation_id, recipient, channel, trigger_reason,
                      dedupe_key, json.dumps(payload, ensure_ascii=False), "queued", stamp))
    except sqlite3.IntegrityError as exc:
        if "dedupe_key" in str(exc) or "UNIQUE constraint failed: notifications.dedupe_key" in str(exc):
            existing = conn.execute("SELECT notification_id FROM notifications WHERE dedupe_key=?", (dedupe_key,)).fetchone()
            if existing:
                stamp = datetime.now(timezone.utc).isoformat()
                details = f"Duplicate request suppressed for trigger purpose: {trigger_reason}"
                conn.execute("UPDATE notifications SET duplicate_attempts=duplicate_attempts+1 WHERE notification_id=?",
                             (existing["notification_id"],))
                conn.execute("INSERT INTO notification_events(notification_id,status,details,created_at) VALUES(?,?,?,?)",
                             (existing["notification_id"], "duplicate_suppressed", details, stamp))
                conn.commit()
            return False
        raise
    conn.execute("INSERT INTO notification_events(notification_id,status,details,created_at) VALUES(?,?,?,?)",
                 (notification_id, "queued", "Queued; no external delivery attempted yet.", stamp))
    conn.commit()
    return True


def mark_notification_simulated(conn: sqlite3.Connection, notification_id: str, response: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE notifications SET status='simulated',attempted_at=?,provider_response=? WHERE notification_id=? AND status='queued'",
                 (stamp, response, notification_id))
    conn.execute("INSERT INTO notification_events(notification_id,status,details,created_at) VALUES(?,?,?,?)",
                 (notification_id, "simulated", response, stamp))
    conn.commit()


def record_notification_attempt(conn: sqlite3.Connection, notification_id: str, *, status: str,
                                details: str, retry_count: int, error_details: str = "") -> None:
    allowed = {"queued", "simulated", "sent", "provider-accepted", "delivered", "failed"}
    if status not in allowed:
        raise ValueError(f"Unsupported notification status: {status}")
    stamp = datetime.now(timezone.utc).isoformat()
    conn.execute("""UPDATE notifications SET status=?, attempted_at=?, provider_response=?,
                    retry_count=?, error_details=? WHERE notification_id=?""",
                 (status, stamp, details, retry_count, error_details, notification_id))
    conn.execute("INSERT INTO notification_events(notification_id,status,details,created_at) VALUES(?,?,?,?)",
                 (notification_id, status, details, stamp))
    conn.commit()


def load_notifications(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM notifications ORDER BY created_at DESC").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def notification_history(conn: sqlite3.Connection, notification_id: str) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM notification_events WHERE notification_id=? ORDER BY event_id", (notification_id,)).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def save_evaluation(conn: sqlite3.Connection, result: dict) -> None:
    conn.execute("INSERT INTO evaluation_runs(result_json,created_at) VALUES(?,?)",
                 (json.dumps(result, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
    conn.commit()


def load_evaluations(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM evaluation_runs ORDER BY evaluation_id DESC").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_setting(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value_json FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("""INSERT INTO settings(key,value_json,updated_at) VALUES(?,?,?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
                 (key, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
    conn.commit()
