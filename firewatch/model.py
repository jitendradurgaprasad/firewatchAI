"""Small RGB ResNet-18 classifier and reproducible training utilities."""

from pathlib import Path
from threading import Event
from typing import Callable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import LABELS

IMAGE_SIZE = 384
PREPROCESSING = "RGB resize 384x384 bilinear; tensor scaled to [0,1], normalized with mean/std 0.5"
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
LABEL_ORDER = ["ACTIVE_FIRE_EVIDENCE", "NO_VISIBLE_FIRE_EVIDENCE"]


class ImageManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            tensor = PREPROCESS(image)
        return tensor, LABEL_ORDER.index(row["label"])


def build_model(*, pretrained: bool = False) -> nn.Module:
    """Construct ResNet-18; pretrained weights are opt-in and may need network access."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    network = models.resnet18(weights=weights)
    network.fc = nn.Linear(network.fc.in_features, len(LABEL_ORDER))
    return network


def save_checkpoint(model: nn.Module, path: str | Path, *, epoch: int, seed: int,
                    metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": "resnet18",
                "labels": LABEL_ORDER, "preprocessing": PREPROCESSING,
                "epoch": epoch, "seed": seed, **(metadata or {})}, path)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("architecture") != "resnet18" or payload.get("labels") != LABEL_ORDER:
        raise ValueError("Checkpoint architecture or label order is unsupported.")
    if not payload.get("trained"):
        raise ValueError("Checkpoint is not marked as a completed training run.")
    model = build_model()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def update_checkpoint_metadata(path: str | Path, metadata: dict) -> None:
    """Add evaluation metadata to an already validated model checkpoint."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "resnet18" or payload.get("labels") != LABEL_ORDER:
        raise ValueError("Cannot update metadata on an unsupported checkpoint.")
    payload.update(metadata)
    torch.save(payload, checkpoint_path)


def train_model(train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]],
                checkpoint: str | Path, *, epochs: int = 5, batch_size: int = 4,
                learning_rate: float = 1e-3, seed: int = 42, device: str = "cuda",
                on_progress: Callable[[dict], None] | None = None,
                stop_event: Event | None = None, num_workers: int = 0) -> nn.Module:
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        labels = {row.get("label") for row in rows}
        if labels != set(LABEL_ORDER):
            raise ValueError(f"{split_name} data must contain both validated classes; found {sorted(labels)}.")
        for row in rows:
            source = row.get("label_source", "").strip().lower()
            if not source:
                raise ValueError(f"{split_name} rows must document label_source provenance.")
            if "hotmap" in source or "mask_presence_weak" in source:
                raise ValueError(f"{split_name} contains weak HOTMAP mask labels; they are not suitable class labels.")
    train_events = {row.get("event_id") for row in train_rows}
    validation_events = {row.get("event_id") for row in validation_rows}
    if None in train_events or None in validation_events or train_events & validation_events:
        raise ValueError("Training and validation must use non-empty, disjoint fire-event IDs.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if device != "cuda":
        raise ValueError("Training is restricted to the NVIDIA CUDA device; CPU fallback is disabled.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training was not started and no CPU fallback was attempted.")
    torch.manual_seed(seed)
    random_gen = torch.Generator().manual_seed(seed)
    cuda_name = torch.cuda.get_device_name(0)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(ImageManifestDataset(train_rows), batch_size=batch_size,
                              shuffle=True, generator=random_gen, num_workers=num_workers)
    validation_loader = DataLoader(ImageManifestDataset(validation_rows), batch_size=batch_size,
                                   num_workers=num_workers)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Training was stopped at a batch boundary.")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()
            if on_progress:
                on_progress({"epoch": epoch, "epochs": epochs,
                              "batch": batch_index, "batches": len(train_loader),
                              "device": cuda_name, "status": "training"})
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                total_loss += loss_fn(model(images), labels).item() * len(labels)
        val_loss = total_loss / len(validation_loader.dataset)
        if on_progress:
            on_progress({"epoch": epoch, "epochs": epochs, "batch": len(train_loader),
                         "batches": len(train_loader), "validation_loss": val_loss,
                         "device": cuda_name, "status": "validation"})
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, checkpoint, epoch=epoch, seed=seed,
                            metadata={"model_id": "firewatch-resnet18", "model_version": "0.1.0",
                                      "device": f"cuda:0 ({cuda_name})", "batch_size": batch_size,
                                      "learning_rate": learning_rate, "epochs_requested": epochs,
                                      "num_workers": num_workers, "best_validation_loss": val_loss})
    update_checkpoint_metadata(checkpoint, {"trained": True})
    model, _ = load_checkpoint(checkpoint, device=device)
    return model
