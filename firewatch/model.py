"""Small RGB ResNet-18 classifier and reproducible training utilities."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import LABELS

IMAGE_SIZE = 224
PREPROCESSING = "RGB resize 224x224 bilinear; tensor scaled to [0,1], normalized with mean/std 0.5"
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


def save_checkpoint(model: nn.Module, path: str | Path, *, epoch: int, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": "resnet18",
                "labels": LABEL_ORDER, "preprocessing": PREPROCESSING,
                "epoch": epoch, "seed": seed}, path)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("architecture") != "resnet18" or payload.get("labels") != LABEL_ORDER:
        raise ValueError("Checkpoint architecture or label order is unsupported.")
    model = build_model()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def train_model(train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]],
                checkpoint: str | Path, *, epochs: int = 5, batch_size: int = 16,
                learning_rate: float = 1e-3, seed: int = 42, device: str | None = None) -> nn.Module:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    torch.manual_seed(seed)
    random_gen = torch.Generator().manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(ImageManifestDataset(train_rows), batch_size=batch_size,
                              shuffle=True, generator=random_gen)
    validation_loader = DataLoader(ImageManifestDataset(validation_rows), batch_size=batch_size)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                total_loss += loss_fn(model(images), labels).item() * len(labels)
        val_loss = total_loss / len(validation_loader.dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, checkpoint, epoch=epoch, seed=seed)
    model, _ = load_checkpoint(checkpoint, device=device)
    return model
