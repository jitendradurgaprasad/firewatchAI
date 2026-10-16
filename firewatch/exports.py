"""Spreadsheet-safe CSV exports for user-supplied and FIRMS values."""

from __future__ import annotations

import pandas as pd
import json


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize CSV while neutralizing formula-like text cells for spreadsheets."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        values = safe[column].astype("string")
        starts_formula = values.str.lstrip().str.startswith(FORMULA_PREFIXES, na=False)
        safe.loc[starts_formula, column] = "'" + values.loc[starts_formula].astype(str)
    return safe.to_csv(index=False).encode("utf-8-sig")


def investigation_report_markdown(observation: dict, *, feedback: pd.DataFrame | None = None,
                                  notifications: pd.DataFrame | None = None) -> str:
    """Create an individual review report without presenting detections as confirmed fires."""
    fields = [
        ("Observation ID", observation.get("observation_id")),
        ("Source", observation.get("source_name")),
        ("Product", observation.get("product")),
        ("Original filename", observation.get("source_filename")),
        ("Source file SHA-256", observation.get("source_file_sha256")),
        ("Retrieved at", observation.get("source_retrieved_at")),
        ("Acquisition time (UTC)", observation.get("acq_datetime")),
        ("Coordinates", f"{observation.get('latitude')}, {observation.get('longitude')}"),
        ("Satellite / instrument", f"{observation.get('satellite') or 'Unknown'} / {observation.get('instrument') or 'Unknown'}"),
        ("Confidence", observation.get("confidence")),
        ("FRP (MW)", observation.get("frp")),
        ("Priority score (application-generated)", observation.get("priority_score")),
        ("Priority category", observation.get("priority_category")),
        ("Review status", observation.get("review_status")),
        ("Association ID", observation.get("association_id")),
        ("Nearby observation count", observation.get("nearby_observation_count")),
        ("Explanation", observation.get("explanation")),
        ("Missing inputs", observation.get("missing_factors")),
        ("Data-quality warnings", observation.get("data_quality_warnings")),
    ]
    lines = ["# FireWatch AI — Observation investigation report", "",
             "> Unverified satellite thermal anomaly. This report does not confirm a wildfire, severity, or emergency.",
             "", "## Observation"]
    lines.extend(f"- **{label}:** {value if value not in (None, '') else 'Not available'}" for label, value in fields)
    raw = observation.get("source_record", "{}")
    try:
        raw = json.dumps(json.loads(raw) if isinstance(raw, str) else raw, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        raw = str(raw)
    lines.extend(["", "## Original source record", "", "```json", raw, "```", "", "## Reviewer history"])
    if feedback is None or feedback.empty:
        lines.append("No reviews recorded.")
    else:
        for row in feedback.to_dict(orient="records"):
            lines.append(f"- {row.get('created_at')}: {row.get('outcome')} — {row.get('assessment')} (reviewer: {row.get('reviewer_id') or 'unspecified'})")
    lines.extend(["", "## Notification history"])
    if notifications is None or notifications.empty:
        lines.append("No notifications recorded.")
    else:
        for row in notifications.to_dict(orient="records"):
            lines.append(f"- {row.get('created_at')}: {row.get('status')} — {row.get('trigger_reason')}")
    attribution = observation.get("source_attribution") or "Attribution not recorded for this import; verify source terms."
    lines.extend(["", "## Provenance and attribution", "", str(attribution), "",
                  "Priority score is an application-generated review heuristic, not a wildfire probability or fire-severity estimate.", ""])
    return "\n".join(lines)
