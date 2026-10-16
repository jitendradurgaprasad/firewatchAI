"""FIRMS CSV ingestion with explicit row-level validation and provenance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pandas as pd


MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_INTERACTIVE_ROWS = 100_000


ALIASES = {
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "long"),
    "acq_date": ("acq_date", "date", "acquisition_date", "yyyymmdd"),
    "acq_time": ("acq_time", "time", "acquisition_time", "hhmm"),
    "satellite": ("satellite", "sat", "platform"),
    "instrument": ("instrument", "sensor"),
    "confidence": ("confidence", "conf"),
    "bright_ti4": ("bright_ti4", "t_i4", "t4"),
    "bright_ti5": ("bright_ti5", "t_i5", "t5"),
    "brightness": ("brightness", "bright_t31", "brightness_temperature"),
    "frp": ("frp", "fire_radiative_power"),
    "daynight": ("daynight", "day_night", "dnflag"),
    "scan": ("scan", "scan_size"),
    "track": ("track", "track_size"),
    "version": ("version", "product_version"),
    "type": ("type", "hotspot_type"),
    "jurisdiction": ("jurisdiction", "admin_region", "state", "province"),
}
REQUIRED = ("latitude", "longitude", "acq_date", "acq_time")
NUMERIC_FIELDS = ("bright_ti4", "bright_ti5", "brightness", "frp", "scan", "track")
SUPPORTED = tuple(ALIASES)


class IngestionError(ValueError):
    """Raised when an input cannot be interpreted as a supported CSV."""


@dataclass
class ImportResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame
    report: dict
    metadata: dict


def _safe_chunk_iterator(reader):
    try:
        yield from reader
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise IngestionError(f"Malformed or unsupported CSV row: {exc}") from exc


def _read_source(source: str | Path | bytes | bytearray | BinaryIO,
                 max_bytes: int = MAX_CSV_BYTES) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise IngestionError(f"CSV file was not found: {path}")
        if path.stat().st_size > max_bytes:
            raise IngestionError(f"CSV exceeds the interactive size limit of {max_bytes // (1024 * 1024)} MiB.")
        return path.read_bytes(), path.name
    if isinstance(source, (bytes, bytearray)):
        if len(source) > max_bytes:
            raise IngestionError(f"CSV exceeds the interactive size limit of {max_bytes // (1024 * 1024)} MiB.")
        return bytes(source), "uploaded.csv"
    name = Path(getattr(source, "name", "uploaded.csv")).name
    size = getattr(source, "size", None)
    if size is not None and size > max_bytes:
        raise IngestionError(f"CSV exceeds the interactive size limit of {max_bytes // (1024 * 1024)} MiB.")
    data = source.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) > max_bytes:
        raise IngestionError(f"CSV exceeds the interactive size limit of {max_bytes // (1024 * 1024)} MiB.")
    return bytes(data), name


def _parse_date(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    formats = ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%m/%d/%Y")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_time(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    if re.fullmatch(r"\d{1,4}", value):
        value = value.zfill(4)
        hour, minute = int(value[:2]), int(value[2:])
    elif re.fullmatch(r"\d{1,2}:\d{2}", value):
        hour, minute = (int(part) for part in value.split(":"))
    else:
        return None
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}{minute:02d}"


def _parse_confidence(raw: str) -> str | float | None:
    value = raw.strip().lower()
    if not value:
        return None
    categories = {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal",
                  "medium": "nominal", "med": "nominal", "h": "high", "high": "high"}
    if value in categories:
        return categories[value]
    try:
        number = float(value)
    except ValueError:
        return "__INVALID__"
    if not math.isfinite(number) or not 0 <= number <= 100:
        return "__INVALID__"
    return number


def ingest_firms_csv(
    source: str | Path | bytes | bytearray | BinaryIO,
    *,
    source_description: str | None = None,
    licence_attribution: str | None = None,
    retrieved_at: datetime | None = None,
    chunk_size: int = 50_000,
    product: str | None = None,
    original_filename: str | None = None,
    max_bytes: int = MAX_CSV_BYTES,
    max_rows: int = MAX_INTERACTIVE_ROWS,
) -> ImportResult:
    """Read a FIRMS CSV, retaining every source row and rejecting with reasons.

    The pandas parser reads bounded chunks; valid and rejected records are
    returned separately, with exact original row values retained as JSON.
    """
    if chunk_size < 1 or max_rows < 1:
        raise IngestionError("chunk_size and max_rows must be positive integers.")
    data, detected_filename = _read_source(source, max_bytes=max_bytes)
    filename = Path(original_filename).name if original_filename else detected_filename
    if not data.strip():
        raise IngestionError("The CSV file is empty.")
    try:
        dialect = csv.Sniffer().sniff(data[:8192].decode("utf-8-sig", errors="replace"), delimiters=",;\t|")
    except csv.Error as exc:
        raise IngestionError(f"Could not identify a supported comma-separated CSV: {exc}") from exc
    if dialect.delimiter != ",":
        raise IngestionError(f"Unsupported CSV delimiter {dialect.delimiter!r}; expected comma-separated FIRMS data.")
    try:
        chunks = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, chunksize=chunk_size,
                             encoding="utf-8-sig", on_bad_lines="error", skip_blank_lines=False)
        columns = list(chunks._engine.names)
    except Exception as exc:
        raise IngestionError(f"Unsupported or malformed CSV: {exc}") from exc
    normalized_headers = {str(column).strip().lower().replace(" ", "_"): column for column in columns}
    selected = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized_headers:
                selected[canonical] = normalized_headers[alias]
                break
    missing_columns = [field for field in REQUIRED if field not in selected]
    valid_rows, rejected_rows = [], []
    missing_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    seen: set[str] = set()
    total = duplicates = row_number = 0

    for chunk in _safe_chunk_iterator(chunks):
        for raw_values in chunk.to_dict(orient="records"):
            if total >= max_rows:
                raise IngestionError(f"CSV exceeds the interactive limit of {max_rows:,} rows; query a smaller area or time window.")
            total += 1
            row_number += 1
            raw = {str(k): str(v) for k, v in raw_values.items()}
            source_record = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            reasons: list[str] = []
            normalized = {key: raw.get(column, "").strip() for key, column in selected.items()}
            for field in SUPPORTED:
                value = normalized.get(field, "")
                if not value:
                    missing_counts[field] += 1
            if missing_columns:
                reasons.append("missing_required_columns:" + ",".join(missing_columns))
            try:
                latitude = float(normalized.get("latitude", ""))
                if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                    raise ValueError
            except ValueError:
                latitude = None
                reasons.append("invalid_latitude")
            try:
                longitude = float(normalized.get("longitude", ""))
                if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                    raise ValueError
            except ValueError:
                longitude = None
                reasons.append("invalid_longitude")
            date_value = _parse_date(normalized.get("acq_date", ""))
            if date_value is None:
                reasons.append("missing_or_invalid_acq_date")
            time_value = _parse_time(normalized.get("acq_time", ""))
            if time_value is None:
                reasons.append("missing_or_invalid_acq_time")

            confidence = _parse_confidence(normalized.get("confidence", ""))
            if confidence == "__INVALID__":
                confidence = None
                reasons.append("unexpected_confidence")
            numeric = {}
            for field in NUMERIC_FIELDS:
                value = normalized.get(field, "")
                if not value:
                    numeric[field] = None
                    continue
                try:
                    parsed = float(value)
                    if not math.isfinite(parsed) or (field in {"frp", "scan", "track"} and parsed < 0):
                        raise ValueError
                    if field in {"scan", "track"} and parsed == 0:
                        raise ValueError
                    if field in {"bright_ti4", "bright_ti5", "brightness"} and parsed <= 0:
                        raise ValueError
                    numeric[field] = parsed
                except ValueError:
                    numeric[field] = None
                    reasons.append(f"invalid_{field}")
            daynight = normalized.get("daynight", "").upper()
            if daynight and daynight not in {"D", "N"}:
                reasons.append("unexpected_daynight")
            if "type" in normalized and normalized["type"]:
                try:
                    type_value = float(normalized["type"])
                    if not math.isfinite(type_value) or not type_value.is_integer() or int(type_value) not in {0, 1, 2, 3}:
                        raise ValueError
                except ValueError:
                    reasons.append("unexpected_hotspot_type")
            digest = hashlib.sha256(source_record.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                reasons.append("duplicate_record")
            else:
                seen.add(digest)
            row = {
                **raw,
                "observation_id": "obs_" + digest[:20],
                "latitude": latitude,
                "longitude": longitude,
                "acq_date": date_value or normalized.get("acq_date", ""),
                "acq_time": time_value or normalized.get("acq_time", ""),
                "acq_datetime": (f"{date_value} {time_value[:2]}:{time_value[2:]}:00+00:00"
                                 if date_value and time_value else ""),
                "satellite": normalized.get("satellite", ""),
                "instrument": normalized.get("instrument", ""),
                "confidence": confidence,
                "bright_ti4": numeric.get("bright_ti4"),
                "bright_ti5": numeric.get("bright_ti5"),
                "brightness": numeric.get("brightness"),
                "frp": numeric.get("frp"),
                "scan": numeric.get("scan"),
                "track": numeric.get("track"),
                "daynight": daynight,
                "version": normalized.get("version", ""),
                "type": normalized.get("type", ""),
                "jurisdiction": normalized.get("jurisdiction", ""),
                "source_record": source_record,
                "source_row_number": row_number,
            }
            if reasons:
                row["rejection_reasons"] = ";".join(reasons)
                rejected_rows.append(row)
                rejection_counts.update(reasons)
            else:
                valid_rows.append(row)
    valid = pd.DataFrame(valid_rows)
    rejected = pd.DataFrame(rejected_rows)
    if "acq_datetime" in valid:
        valid["acq_datetime"] = pd.to_datetime(valid["acq_datetime"], utc=True, errors="coerce")
    dates = valid["acq_datetime"].dropna() if not valid.empty else pd.Series(dtype="datetime64[ns, UTC]")
    retrieved = retrieved_at or datetime.now(timezone.utc)
    if retrieved.tzinfo is None:
        retrieved = retrieved.replace(tzinfo=timezone.utc)
    metadata = {
        "source": source_description or f"User-uploaded CSV: {filename}",
        "product": (product or "").strip(),
        "filename": filename,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "file_size_bytes": len(data),
        "retrieved_at": retrieved.astimezone(timezone.utc).isoformat(),
        "date_range_start": dates.min().isoformat() if len(dates) else None,
        "date_range_end": dates.max().isoformat() if len(dates) else None,
        "total_records": total,
        "valid_records": len(valid_rows),
        "rejected_records": len(rejected_rows),
        "licence_attribution": licence_attribution or "Attribution supplied by the uploader; verify source terms before reuse.",
        "available_fields": sorted(selected),
        "quality_warnings": ([f"Optional supported fields missing from file: {', '.join(k for k in SUPPORTED if k not in selected)}"]
                             if any(k not in selected for k in SUPPORTED if k not in {"jurisdiction", "type", "brightness"}) else []),
    }
    report = {
        "total_rows": total,
        "valid_rows": len(valid_rows),
        "invalid_rows": len(rejected_rows),
        "duplicate_rows": duplicates,
        "missing_field_counts": {field: int(missing_counts.get(field, 0)) for field in SUPPORTED},
        "missing_columns": missing_columns,
        "rejection_reasons": dict(rejection_counts),
    }
    return ImportResult(valid=valid, rejected=rejected, report=report, metadata=metadata)
