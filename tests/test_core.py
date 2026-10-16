import csv

import pytest
from PIL import Image

from firewatch.california_dataset import prepare_california_manifest
from firewatch.data import DatasetError, split_by_event, validate_manifest
from firewatch.image_io import ImageInputError, load_rgb_image
from firewatch.inference import predict
from firewatch.metrics import classification_metrics
from firewatch.model import PREPROCESS, build_model, load_checkpoint, save_checkpoint, train_model


def make_rgb(path, color=(20, 30, 40)):
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_load_rgb_jpeg_png(tmp_path):
    for suffix in (".jpg", ".png"):
        path = make_rgb(tmp_path / f"image{suffix}")
        loaded = load_rgb_image(path)
        assert loaded.image.mode == "RGB"
        assert loaded.width == loaded.height == 64


def test_rejects_unsupported_corrupt_and_non_rgb(tmp_path):
    with pytest.raises(ImageInputError, match="extension"):
        load_rgb_image(tmp_path / "map.tif")
    corrupt = tmp_path / "broken.png"
    corrupt.write_text("not an image")
    with pytest.raises(ImageInputError):
        load_rgb_image(corrupt)
    gray = tmp_path / "gray.png"
    Image.new("L", (64, 64)).save(gray)
    with pytest.raises(ImageInputError, match="3-channel"):
        load_rgb_image(gray)


def test_rejects_tiny_image(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8)).save(path)
    with pytest.raises(ImageInputError, match="at least"):
        load_rgb_image(path)


def test_preprocessing_preserves_classification_tensor_contract(tmp_path):
    import torch

    path = make_rgb(tmp_path / "image.png")
    tensor = PREPROCESS(load_rgb_image(path).image)
    assert tensor.shape == (3, 384, 384)
    assert tensor.dtype == torch.float32


def test_manifest_validates_label_and_duplicate(tmp_path):
    image = make_rgb(tmp_path / "a.png")
    manifest = tmp_path / "labels.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "event_id"])
        writer.writeheader()
        writer.writerow({"image": image.name, "label": "ACTIVE_FIRE_EVIDENCE", "event_id": "fire-1"})
        writer.writerow({"image": image.name, "label": "NO_VISIBLE_FIRE_EVIDENCE", "event_id": "fire-2"})
    with pytest.raises(DatasetError, match="Duplicate"):
        validate_manifest(manifest)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "event_id"])
        writer.writeheader()
        writer.writerow({"image": image.name, "label": "wildfire", "event_id": "fire-1"})
    with pytest.raises(DatasetError, match="Invalid label"):
        validate_manifest(manifest)


def test_event_split_is_deterministic_and_disjoint():
    rows = [{"image": f"{event}-{i}", "label": "ACTIVE_FIRE_EVIDENCE", "event_id": event}
            for event in "abcdef" for i in range(2)]
    result = split_by_event(rows, seed=13)
    event_sets = [{row["event_id"] for row in result[key]} for key in ("train", "validation", "test")]
    assert all(event_sets[i].isdisjoint(event_sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert result == split_by_event(rows, seed=13)


def test_missing_weights_abstain(tmp_path):
    image = make_rgb(tmp_path / "image.png")
    result = predict(image, tmp_path / "missing.pt")
    assert result.outcome == "UNCERTAIN_OR_UNSUPPORTED"
    assert result.predicted_class is None
    assert result.confidence_score is None
    assert "missing" in result.message


def test_invalid_image_abstains_and_metrics():
    result = predict("does-not-exist.jpg", "missing.pt")
    assert result.outcome == "UNCERTAIN_OR_UNSUPPORTED"
    metrics = classification_metrics(["ACTIVE_FIRE_EVIDENCE", "NO_VISIBLE_FIRE_EVIDENCE"],
                                     ["ACTIVE_FIRE_EVIDENCE", "ACTIVE_FIRE_EVIDENCE"])
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["NO_VISIBLE_FIRE_EVIDENCE"]["ACTIVE_FIRE_EVIDENCE"] == 1


def test_model_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(build_model(), checkpoint, epoch=1, seed=5)
    model, metadata = load_checkpoint(checkpoint)
    assert metadata["architecture"] == "resnet18"
    assert metadata["epoch"] == 1
    assert not model.training


def test_confidence_threshold_abstains(tmp_path, monkeypatch):
    import torch
    import firewatch.inference as inference
    from firewatch.model import PREPROCESSING

    class FixedModel:
        def __call__(self, _tensor):
            return torch.log(torch.tensor([[0.6, 0.4]]))

    image = make_rgb(tmp_path / "image.png")
    checkpoint = tmp_path / "exists.pt"
    checkpoint.touch()
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: (FixedModel(), {
        "preprocessing": PREPROCESSING
    }))
    result = predict(image, checkpoint, threshold=0.7)
    assert result.outcome == "UNCERTAIN_OR_UNSUPPORTED"
    assert result.confidence_score == pytest.approx(0.6)
    accepted = predict(image, checkpoint, threshold=0.6)
    assert accepted.outcome == "FIRE EVIDENCE DETECTED"


def test_official_event_splits_are_preserved_and_checked():
    rows = [
        {"image": f"{event}-{label}", "label": label, "event_id": event, "split": split}
        for event, split in (("fire-a", "train"), ("fire-b", "validation"), ("fire-c", "test"))
        for label in ("ACTIVE_FIRE_EVIDENCE", "NO_VISIBLE_FIRE_EVIDENCE")
    ]
    parts = split_by_event(rows, seed=999)
    assert {row["event_id"] for row in parts["test"]} == {"fire-c"}
    rows[-1] = {**rows[-1], "split": "train"}
    with pytest.raises(DatasetError, match="multiple splits"):
        split_by_event(rows)


def test_california_adapter_uses_masks_and_excludes_all_nodata(tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    fires = [("Fire A", "train"), ("Fire B", "val"), ("Fire C", "test")]
    summary_rows, partition_rows = [], []
    for fire_index, (fire, split) in enumerate(fires):
        total_fire_pixels = 0
        fire_chips = 0
        for kind_index, kind in enumerate(("positive", "negative", "nodata")):
            filename = f"fire{fire_index}_{kind}.png"
            make_rgb(root / "images" / filename, (20 + fire_index * 3 + kind_index, 30, 40))
            mask = Image.new("L", (64, 64), 255 if kind == "nodata" else 0)
            if kind == "positive":
                mask.putpixel((4, 4), 1)
                total_fire_pixels += 1
                fire_chips += 1
            mask.save(root / "masks" / filename)
            summary_rows.append({"file_name": filename, "fire": fire, "fire_px": str(int(kind == "positive")),
                                 "nodata_pct": "100.000" if kind == "nodata" else "0.000"})
        partition_rows.append({"fire": fire, "split": split, "chips": "3",
                               "fire_chips": str(fire_chips), "fire_px": str(total_fire_pixels)})
    for filename, rows in (("summary.csv", summary_rows), ("partitions.csv", partition_rows)):
        with (root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = prepare_california_manifest(root)
    assert report["weak_positive_candidates"] == 3
    assert report["uncertain_no_fire_mask"] == 3
    assert report["skipped_nodata"] == 3
    assert report["training_ready"] is False
    assert report["splits"]["validation"]["weak_positive_candidates"] == 1
    rows = validate_manifest(report["positive_candidates_manifest"])
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all(row["label"] == "ACTIVE_FIRE_EVIDENCE" for row in rows)
    parts = split_by_event(rows)
    assert {row["event_id"] for row in parts["test"]} == {"Fire C"}
    assert all(row["label_source"] == "HOTMAP_SWIR_mask_presence_weak" for row in rows)
    with open(report["review_queue"], newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))
    assert len(review_rows) == 3
    assert {row["status"] for row in review_rows} == {"UNCERTAIN_OR_UNSUPPORTED"}


def test_training_refuses_unvalidated_single_class_data(tmp_path):
    rows = [{"image": "missing.png", "label": "ACTIVE_FIRE_EVIDENCE", "event_id": "fire-a"}]
    with pytest.raises(ValueError, match="both validated classes"):
        train_model(rows, rows, tmp_path / "model.pt")
