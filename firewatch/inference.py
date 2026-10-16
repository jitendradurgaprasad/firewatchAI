"""Structured inference with explicit abstention and no implicit fallback model."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .image_io import ImageInputError, load_rgb_image
from .model import LABEL_ORDER, PREPROCESS, PREPROCESSING, load_checkpoint

MODEL_ID = "firewatch-resnet18"
MODEL_VERSION = "0.1.0"
LIMITATIONS = [
    "RGB input only; multispectral bands and georeferencing are not read from ordinary PNG/JPEG files.",
    "A model score is not a calibrated probability of real-world fire.",
    "Training labels, when used, are weakly derived from satellite SWIR masks and do not establish universal fire detection.",
]


@dataclass(frozen=True)
class Prediction:
    outcome: str
    predicted_class: str | None
    confidence_score: float | None
    score_kind: str
    model_id: str
    model_version: str
    input_validation: str
    inference_device: str
    message: str
    limitations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def predict(image_path: str | Path, checkpoint: str | Path, *, threshold: float = 0.7,
            device: str = "cpu") -> Prediction:
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.5 and 1.0.")
    try:
        loaded = load_rgb_image(image_path)
    except ImageInputError as exc:
        return Prediction("UNCERTAIN_OR_UNSUPPORTED", None, None, "unavailable", MODEL_ID,
                          MODEL_VERSION, "invalid", device, str(exc), LIMITATIONS)
    if not Path(checkpoint).is_file():
        return Prediction("UNCERTAIN_OR_UNSUPPORTED", None, None, "unavailable", MODEL_ID,
                          MODEL_VERSION, "valid", device, "Prediction unavailable: trained model weights are missing.", LIMITATIONS)
    try:
        model, meta = load_checkpoint(checkpoint, device=device)
        if meta.get("preprocessing") != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing metadata is unsupported.")
        tensor = PREPROCESS(loaded.image).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = torch.max(scores, dim=0)
        confidence_value = float(confidence.item())
        label = LABEL_ORDER[int(index.item())]
        if confidence_value < threshold:
            return Prediction("UNCERTAIN_OR_UNSUPPORTED", None, confidence_value, "uncalibrated_softmax_score",
                              MODEL_ID, MODEL_VERSION, "valid", device, "Model score is below the abstention threshold.", LIMITATIONS)
        outcome = "FIRE EVIDENCE DETECTED" if label == "ACTIVE_FIRE_EVIDENCE" else "NO VISIBLE FIRE EVIDENCE DETECTED"
        return Prediction(outcome, label, confidence_value, "uncalibrated_softmax_score", MODEL_ID,
                          MODEL_VERSION, "valid", device, "Classification from the loaded checkpoint.", LIMITATIONS)
    except Exception as exc:
        return Prediction("UNCERTAIN_OR_UNSUPPORTED", None, None, "unavailable", MODEL_ID,
                          MODEL_VERSION, "valid", device, f"Inference unavailable: {exc}", LIMITATIONS)
