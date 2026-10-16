"""Prepare weak chip labels from the California Sentinel-2 mask dataset."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .data import DatasetError
from .image_io import ImageInputError, load_rgb_image

SPLIT_NAMES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
MASK_LABEL_SOURCE = "HOTMAP_SWIR_mask_presence_weak"
MASK_VALUES = {0, 1, 255}


def prepare_california_manifest(dataset_root: str | Path, output_csv: str | Path | None = None,
                                review_queue_csv: str | Path | None = None,
                                *, max_nodata_percent: float = 50.0) -> dict:
    """Validate all pairs; emit weak positives and quarantine mask-zero chips.

    An image is positive if its supplied weak mask contains value 1. A mask with
    no positive pixels is *not* turned into a negative: cloud-obscured valid
    imagery cannot be distinguished from clear no-fire imagery using this mask.
    Such chips are written to a review queue with UNCERTAIN_OR_UNSUPPORTED.
    Mask value 255 is ignored as no-data.
    """
    if not 0 <= max_nodata_percent < 100:
        raise ValueError("max_nodata_percent must be in [0, 100).")
    root = Path(dataset_root).resolve()
    summary_path, partitions_path = root / "summary.csv", root / "partitions.csv"
    if not (root / "images").is_dir() or not (root / "masks").is_dir():
        raise DatasetError("Dataset root must contain images/ and masks/ directories.")
    for path in (summary_path, partitions_path):
        if not path.is_file():
            raise DatasetError(f"Dataset file is missing: {path.name}.")

    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        summary_reader = csv.DictReader(handle)
        required = {"file_name", "fire", "fire_px", "nodata_pct"}
        if not summary_reader.fieldnames or not required.issubset(summary_reader.fieldnames):
            raise DatasetError(f"summary.csv must contain {', '.join(sorted(required))}.")
        summary = list(summary_reader)
    with partitions_path.open(newline="", encoding="utf-8-sig") as handle:
        split_reader = csv.DictReader(handle)
        required = {"fire", "split", "chips", "fire_chips", "fire_px"}
        if not split_reader.fieldnames or not required.issubset(split_reader.fieldnames):
            raise DatasetError(f"partitions.csv must contain {', '.join(sorted(required))}.")
        partition_rows = list(split_reader)

    event_to_split: dict[str, str] = {}
    for row in partition_rows:
        event, raw_split = row["fire"].strip(), row["split"].strip().lower()
        if not event or raw_split not in SPLIT_NAMES or event in event_to_split:
            raise DatasetError(f"Invalid or duplicate fire in partitions.csv: {event!r}.")
        event_to_split[event] = SPLIT_NAMES[raw_split]
    if not summary:
        raise DatasetError("summary.csv contains no image rows.")

    rows, review_rows, skipped_nodata = [], [], 0
    observed_by_fire = Counter()
    fire_chip_counts = Counter()
    fire_pixel_counts = Counter()
    for source_row in summary:
        file_name, fire = source_row["file_name"].strip(), source_row["fire"].strip()
        if not file_name or Path(file_name).name != file_name:
            raise DatasetError(f"Invalid image basename in summary.csv: {file_name!r}.")
        if fire not in event_to_split:
            raise DatasetError(f"Fire {fire!r} is absent from partitions.csv.")
        observed_by_fire[fire] += 1
        image_path, mask_path = root / "images" / file_name, root / "masks" / file_name
        try:
            loaded = load_rgb_image(image_path)
            with Image.open(mask_path) as source_mask:
                if source_mask.size != loaded.image.size:
                    raise DatasetError(f"Image/mask dimensions differ for {file_name}.")
                mask = np.asarray(source_mask.convert("L"), dtype=np.uint8)
        except ImageInputError as exc:
            raise DatasetError(f"Invalid dataset image {file_name}: {exc}") from exc
        except OSError as exc:
            raise DatasetError(f"Image or mask is missing for {file_name}.") from exc
        values = set(int(value) for value in np.unique(mask))
        if not values.issubset(MASK_VALUES):
            raise DatasetError(f"Unexpected mask values for {file_name}: {sorted(values - MASK_VALUES)}.")
        fire_pixels = int(np.count_nonzero(mask == 1))
        nodata_pixels = int(np.count_nonzero(mask == 255))
        if fire_pixels != int(source_row["fire_px"]):
            raise DatasetError(f"fire_px disagrees with mask for {file_name}.")
        fire_chip_counts[fire] += fire_pixels > 0
        fire_pixel_counts[fire] += fire_pixels
        actual_nodata_percent = nodata_pixels * 100.0 / mask.size
        if abs(actual_nodata_percent - float(source_row["nodata_pct"])) > 0.002:
            raise DatasetError(f"nodata_pct disagrees with mask for {file_name}.")
        valid_pixels = mask.size - nodata_pixels
        if valid_pixels == 0 or actual_nodata_percent > max_nodata_percent:
            skipped_nodata += 1
            continue
        common = {
            "image": str(root / "images" / file_name),
            "event_id": fire,
            "split": event_to_split[fire],
            "fire_px": str(fire_pixels),
            "nodata_pct": f"{actual_nodata_percent:.3f}",
        }
        if fire_pixels:
            rows.append({**common, "label": "ACTIVE_FIRE_EVIDENCE", "label_source": MASK_LABEL_SOURCE})
        else:
            review_rows.append({**common, "status": "UNCERTAIN_OR_UNSUPPORTED",
                                "reason": "No HOTMAP positive; cloud-obscured chips cannot be separated from clear negatives by this mask."})

    for part in partition_rows:
        fire = part["fire"].strip()
        if observed_by_fire[fire] != int(part["chips"]):
            raise DatasetError(f"Chip count for {fire!r} disagrees with partitions.csv.")
        if fire_chip_counts[fire] != int(part["fire_chips"]):
            raise DatasetError(f"Positive chip count for {fire!r} disagrees with partitions.csv.")
        if fire_pixel_counts[fire] != int(part["fire_px"]):
            raise DatasetError(f"Fire pixel count for {fire!r} disagrees with partitions.csv.")
    if not rows:
        raise DatasetError("No mask-positive chips remained after no-data filtering.")

    output_path = Path(output_csv).resolve() if output_csv else root / "firewatch_manifest.csv"
    review_path = Path(review_queue_csv).resolve() if review_queue_csv else root / "firewatch_review_queue.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "label", "event_id", "split", "fire_px", "nodata_pct", "label_source"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    review_fields = ["image", "event_id", "split", "fire_px", "nodata_pct", "status", "reason"]
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)
    counts = Counter(row["split"] for row in rows)
    review_counts = Counter(row["split"] for row in review_rows)
    return {
        "positive_candidates_manifest": str(output_path), "review_queue": str(review_path),
        "weak_positive_candidates": len(rows), "uncertain_no_fire_mask": len(review_rows),
        "skipped_nodata": skipped_nodata, "training_ready": False,
        "splits": {split: {"weak_positive_candidates": counts[split],
                           "uncertain_no_fire_mask": review_counts[split]}
                   for split in ("train", "validation", "test")},
        "events": len(event_to_split), "label_source": MASK_LABEL_SOURCE,
    }
