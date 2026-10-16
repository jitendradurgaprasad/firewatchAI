"""Import and match Forest Survey of India forest-fire point context.

FSI points are satellite thermal-anomaly alerts filtered using forest-area
boundaries. They provide useful forest context, but are not ground truth for
whether a wildfire occurred because the source sensors overlap with FIRMS.
"""

from __future__ import annotations

import csv
import io
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAX_CONTEXT_BYTES = 50 * 1024 * 1024
MAX_CONTEXT_ROWS = 100_000
EARTH_RADIUS_KM = 6371.0088


class ForestContextError(ValueError):
    """Raised when an FSI forest-fire point export is invalid or unsupported."""


def _header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


ALIASES = {
    "date": {"firedate", "date", "acqdate"},
    "time": {"firetime", "time", "acqtime"},
    "source": {"source", "sensor"},
    "latitude": {"lat", "latitude"},
    "longitude": {"lon", "long", "longitude"},
    "state": {"state", "stateut", "stateunionterritory"},
    "district": {"district"},
    "circle": {"circle"},
    "division": {"division"},
    "range": {"range"},
    "block": {"block", "sectionround"},
    "beat": {"beat"},
}


def _parse_date(value: str) -> str | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _parse_time(value: str) -> str | None:
    value = value.strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.strptime(value, fmt).strftime("%H:%M:%S")
        except ValueError:
            pass
    if re.fullmatch(r"\d{1,4}", value):
        value = value.zfill(4)
        hour, minute = int(value[:2]), int(value[2:])
        if hour < 24 and minute < 60:
            return f"{hour:02d}:{minute:02d}:00"
    return None


def parse_fsi_points(source: str | Path | bytes | bytearray) -> pd.DataFrame:
    """Parse an FSI Fire Point Search CSV; every row is accepted or rejected."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ForestContextError(f"Forest-point CSV not found: {path}")
        if path.stat().st_size > MAX_CONTEXT_BYTES:
            raise ForestContextError("Forest-point CSV exceeds the 50 MiB import limit.")
        data = path.read_bytes()
        filename = path.name
    else:
        data = bytes(source)
        filename = "fsi_forest_fire_points.csv"
        if len(data) > MAX_CONTEXT_BYTES:
            raise ForestContextError("Forest-point CSV exceeds the 50 MiB import limit.")
    if not data.strip():
        raise ForestContextError("Forest-point CSV is empty.")
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    except UnicodeDecodeError as exc:
        raise ForestContextError("Forest-point CSV must be UTF-8 or UTF-8 with BOM.") from exc
    if not reader.fieldnames:
        raise ForestContextError("Forest-point CSV has no header row.")
    normalized_headers = {_header(name): name for name in reader.fieldnames}
    selected = {canonical: next((normalized_headers[a] for a in aliases if a in normalized_headers), None)
                for canonical, aliases in ALIASES.items()}
    missing = [key for key in ("date", "time", "source", "latitude", "longitude") if not selected[key]]
    if missing:
        raise ForestContextError("Missing required FSI columns: " + ", ".join(missing))
    rows, seen = [], set()
    for row_number, raw in enumerate(reader, 1):
        if row_number > MAX_CONTEXT_ROWS:
            raise ForestContextError("Forest-point CSV exceeds the 100,000-row import limit.")
        record = {key: (raw.get(column) or "").strip() if column else ""
                  for key, column in selected.items()}
        reasons = []
        date = _parse_date(record["date"])
        time = _parse_time(record["time"])
        try:
            lat, lon = float(record["latitude"]), float(record["longitude"])
            if not (math.isfinite(lat) and -90 <= lat <= 90):
                raise ValueError
        except ValueError:
            lat = None
            reasons.append("invalid_latitude")
        try:
            lon = float(record["longitude"])
            if not (math.isfinite(lon) and -180 <= lon <= 180):
                raise ValueError
        except ValueError:
            lon = None
            reasons.append("invalid_longitude")
        if date is None:
            reasons.append("invalid_fire_date")
        if time is None:
            reasons.append("invalid_fire_time")
        if not record["source"]:
            reasons.append("missing_sensor_source")
        key = (date, time, record["source"].casefold(), lat, lon)
        if not reasons and key in seen:
            reasons.append("duplicate_point")
        if not reasons:
            seen.add(key)
            rows.append({"source_row_number": row_number, "fire_datetime": f"{date}T{time}",
                         "source_sensor": record["source"], "latitude": lat, "longitude": lon,
                         **{field: record[field] or None for field in
                            ("state", "district", "circle", "division", "range", "block", "beat")},
                         "source_record": {str(k): str(v or "") for k, v in raw.items()},
                         "rejection_reason": ""})
        else:
            rows.append({"source_row_number": row_number, "fire_datetime": None,
                         "source_sensor": record["source"] or None, "latitude": lat, "longitude": lon,
                         **{field: record[field] or None for field in
                            ("state", "district", "circle", "division", "range", "block", "beat")},
                         "source_record": {str(k): str(v or "") for k, v in raw.items()},
                         "rejection_reason": ";".join(reasons)})
    frame = pd.DataFrame(rows)
    frame.attrs.update({"filename": filename, "total_rows": len(rows),
                        "valid_rows": int((frame["rejection_reason"] == "").sum()) if not frame.empty else 0,
                        "rejected_rows": int((frame["rejection_reason"] != "").sum()) if not frame.empty else 0})
    return frame


def match_forest_context(observations: pd.DataFrame, points: pd.DataFrame, *,
                         radius_km: float = 1.0, time_window_hours: float = 12.0) -> pd.DataFrame:
    """Mark FIRMS rows with nearby FSI forest-area alerts; unmatched is unknown.

    A 12-hour window allows for differing UTC/local display times in exports.
    The FSI points and FIRMS rows may use the same satellite detection.
    """
    if radius_km <= 0 or time_window_hours <= 0:
        raise ValueError("Forest match distance and time window must be positive.")
    out = observations.copy()
    out["forest_context"] = None
    out["forest_context_matches"] = 0
    out["forest_context_summary"] = "No forest-area context imported"
    if out.empty or points.empty:
        return out
    out["forest_context_summary"] = "No matching forest-area context point in the configured window"
    valid_points = points.loc[points.get("rejection_reason", pd.Series("", index=points.index)).eq("")].copy()
    valid_points = valid_points.dropna(subset=["latitude", "longitude", "fire_datetime"])
    if valid_points.empty:
        out["forest_context_summary"] = "Forest-area context imported, but no valid point rows are available"
        return out
    valid_points = valid_points.drop_duplicates(
        subset=["fire_datetime", "source_sensor", "latitude", "longitude"])
    pcoords = np.radians(valid_points[["latitude", "longitude"]].astype(float).to_numpy())
    xyz = np.column_stack((np.cos(pcoords[:, 0]) * np.cos(pcoords[:, 1]),
                           np.cos(pcoords[:, 0]) * np.sin(pcoords[:, 1]), np.sin(pcoords[:, 0])))
    tree = cKDTree(xyz)
    radius_chord = 2 * math.sin(radius_km / (2 * EARTH_RADIUS_KM))
    point_times = (pd.to_datetime(valid_points["fire_datetime"], utc=True, errors="coerce")
                   .astype("datetime64[ns, UTC]").astype("int64").to_numpy())
    valid_indices = valid_points.index.to_numpy()
    obs_coords = np.radians(out[["latitude", "longitude"]].astype(float).to_numpy())
    obs_xyz = np.column_stack((np.cos(obs_coords[:, 0]) * np.cos(obs_coords[:, 1]),
                               np.cos(obs_coords[:, 0]) * np.sin(obs_coords[:, 1]), np.sin(obs_coords[:, 0])))
    obs_times = (pd.to_datetime(out["acq_datetime"], utc=True, errors="coerce")
                 .astype("datetime64[ns, UTC]").astype("int64").to_numpy())
    window_ns = int(time_window_hours * 3600 * 1_000_000_000)
    for pos, candidates in enumerate(tree.query_ball_point(obs_xyz, radius_chord)):
        if not candidates or obs_times[pos] == np.iinfo(np.int64).min:
            continue
        cand = np.asarray(candidates, dtype=np.int64)
        time_delta = np.abs(point_times[cand] - obs_times[pos])
        matched = cand[time_delta <= window_ns]
        if len(matched):
            out.iloc[pos, out.columns.get_loc("forest_context")] = 1.0
            out.iloc[pos, out.columns.get_loc("forest_context_matches")] = int(len(matched))
            labels = [str(valid_points.loc[valid_indices[i], "source_sensor"]) for i in matched[:3]]
            sources = sorted({str(valid_points.iloc[i].get("context_source") or
                                  valid_points.iloc[i].get("source_sensor") or "forest-area source")
                              for i in matched})
            out.iloc[pos, out.columns.get_loc("forest_context_summary")] = (
                f"Matched {len(matched)} forest-area context point(s) from {', '.join(sources)}; "
                f"sensor(s): {', '.join(labels)}")
    return out
