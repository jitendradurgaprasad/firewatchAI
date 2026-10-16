import csv

import pytest
from PIL import Image

from firewatch.data import DatasetError, split_by_event, validate_manifest
from firewatch.image_io import ImageInputError, load_rgb_image
from firewatch.inference import predict
from firewatch.metrics import classification_metrics
from firewatch.model import build_model, load_checkpoint, save_checkpoint


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

    class FixedModel:
        def __call__(self, _tensor):
            return torch.log(torch.tensor([[0.6, 0.4]]))

    image = make_rgb(tmp_path / "image.png")
    checkpoint = tmp_path / "exists.pt"
    checkpoint.touch()
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args, **_kwargs: (FixedModel(), {
        "preprocessing": "RGB resize 224x224 bilinear; tensor scaled to [0,1], normalized with mean/std 0.5"
    }))
    result = predict(image, checkpoint, threshold=0.7)
    assert result.outcome == "UNCERTAIN_OR_UNSUPPORTED"
    assert result.confidence_score == pytest.approx(0.6)
    accepted = predict(image, checkpoint, threshold=0.6)
    assert accepted.outcome == "FIRE EVIDENCE DETECTED"
