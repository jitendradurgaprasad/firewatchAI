"""Manifest validation and event-aware split helpers."""

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

LABELS = {"ACTIVE_FIRE_EVIDENCE", "NO_VISIBLE_FIRE_EVIDENCE"}
REQUIRED_COLUMNS = {"image", "label", "event_id"}


class DatasetError(ValueError):
    """Raised for invalid or unsafe training manifests."""


def validate_manifest(csv_path: str | Path, *, check_duplicates: bool = True) -> list[dict[str, str]]:
    """Read image,label,event_id CSV and reject invalid labels/files/duplicates."""
    manifest_path = Path(csv_path).resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise DatasetError(f"Manifest must contain columns: {', '.join(sorted(REQUIRED_COLUMNS))}.")
        rows = []
        seen_hashes: dict[str, str] = {}
        for line_number, raw in enumerate(reader, start=2):
            label = (raw.get("label") or "").strip()
            event_id = (raw.get("event_id") or "").strip()
            image_value = (raw.get("image") or "").strip()
            if label not in LABELS:
                raise DatasetError(f"Invalid label on line {line_number}: {label!r}.")
            if not event_id or not image_value:
                raise DatasetError(f"Missing event_id or image path on line {line_number}.")
            image_path = (manifest_path.parent / image_value).resolve()
            if not image_path.is_file():
                raise DatasetError(f"Image does not exist on line {line_number}: {image_value}.")
            if check_duplicates:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    raise DatasetError(f"Duplicate image on line {line_number}: {image_value} duplicates {seen_hashes[digest]}.")
                seen_hashes[digest] = image_value
            rows.append({"image": str(image_path), "label": label, "event_id": event_id})
    if not rows:
        raise DatasetError("Manifest contains no samples.")
    return rows


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {label: counts.get(label, 0) for label in sorted(LABELS)}


def split_by_event(rows: list[dict[str, str]], *, seed: int = 42,
                   train: float = 0.7, validation: float = 0.15) -> dict[str, list[dict[str, str]]]:
    """Deterministically split by event to prevent event/chip leakage."""
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("train and validation fractions must be positive and sum to less than 1.")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["event_id"], []).append(row)
    event_ids = list(groups)
    if len(event_ids) < 3:
        raise DatasetError("At least three distinct events are required for train/validation/test splitting.")
    random.Random(seed).shuffle(event_ids)
    train_end = max(1, round(len(event_ids) * train))
    val_end = max(train_end + 1, round(len(event_ids) * (train + validation)))
    val_end = min(val_end, len(event_ids) - 1)
    partitions = {"train": event_ids[:train_end], "validation": event_ids[train_end:val_end], "test": event_ids[val_end:]}
    return {name: [row for event in ids for row in groups[event]] for name, ids in partitions.items()}
