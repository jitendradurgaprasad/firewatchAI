"""Dataset and training readiness reports used by the desktop application."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .california_dataset import prepare_california_manifest
from .data import LABELS, split_by_event, validate_manifest


@dataclass(frozen=True)
class DatasetReadiness:
    location: str
    available: bool
    validated_pairs: int
    weak_positive_candidates: int
    reviewed_negative_examples: int
    uncertain_or_excluded: int
    skipped_nodata: int
    json_export_available: bool
    validation_status: str
    training_ready: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_california_dataset(dataset_root: str | Path) -> DatasetReadiness:
    """Run the existing pair/mask checks and report label readiness conservatively."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return DatasetReadiness(str(root), False, 0, 0, 0, 0, 0, False,
                                "unavailable", False, "Dataset folder was not found.")
    try:
        report = prepare_california_manifest(root)
    except Exception as exc:
        return DatasetReadiness(str(root), True, 0, 0, 0, 0, 0, False,
                                "failed", False, f"Dataset validation failed: {exc}")

    summary = root / "summary.csv"
    import csv
    with summary.open(newline="", encoding="utf-8-sig") as handle:
        pairs = sum(1 for _ in csv.DictReader(handle))
    json_export = any(root.rglob("*.json"))
    negatives = 0  # No importable Label Studio JSON/export is available.
    uncertain = int(report["uncertain_no_fire_mask"])
    ready = False  # Positive HOTMAP mask candidates alone cannot train a classifier.
    return DatasetReadiness(
        location=str(root), available=True, validated_pairs=pairs,
        weak_positive_candidates=int(report["weak_positive_candidates"]),
        reviewed_negative_examples=negatives, uncertain_or_excluded=uncertain,
        skipped_nodata=int(report["skipped_nodata"]), json_export_available=json_export,
        validation_status="passed", training_ready=ready,
        message=("Training blocked: mask-zero chips are not reviewed negatives, and the "
                 "required Label Studio JSON export is absent."),
    )


def inspect_training_manifest(manifest_path: str | Path) -> tuple[bool, str, dict[str, int]]:
    """Verify both classes exist in train/validation and labels have defensible sources."""
    try:
        rows = validate_manifest(manifest_path)
        parts = split_by_event(rows)
    except Exception as exc:
        return False, f"Manifest validation failed: {exc}", {}
    counts = {}
    for split in ("train", "validation", "test"):
        split_rows = parts[split]
        labels = {row["label"] for row in split_rows}
        counts[split] = len(split_rows)
        if labels != LABELS:
            return False, f"{split.title()} data must contain both validated classes for training and held-out evaluation; found {sorted(labels)}.", counts
        for row in split_rows:
            source = row.get("label_source", "").strip().lower()
            if not source:
                return False, "Training rows need a label_source documenting reviewed or provider-verified labels.", counts
            if "hotmap" in source or "mask_presence_weak" in source:
                return False, "Weak HOTMAP mask labels cannot supply reviewed negative examples.", counts
    return True, "Validated train and validation labels contain both classes with documented sources.", counts
