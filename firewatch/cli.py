"""Local command-line interface for dataset checks, training, and inference."""

import argparse
import json
from pathlib import Path

import torch

from .data import class_counts, split_by_event, validate_manifest
from .image_io import load_rgb_image
from .inference import predict
from .metrics import classification_metrics
from .model import ImageManifestDataset, PREPROCESS, load_checkpoint, train_model


def _evaluate(model, rows, device: str) -> dict:
    actual, predicted = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            with load_rgb_image(row["image"]).image as image:
                scores = model(PREPROCESS(image).unsqueeze(0).to(device))[0]
            actual.append(row["label"])
            predicted.append(("ACTIVE_FIRE_EVIDENCE", "NO_VISIBLE_FIRE_EVIDENCE")[int(scores.argmax().item())])
    return classification_metrics(actual, predicted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="firewatch", description="Local satellite image fire-evidence classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate an image/label/event manifest")
    inspect.add_argument("manifest")
    train = commands.add_parser("train", help="train a small ResNet-18 using event-disjoint splits")
    train.add_argument("manifest")
    train.add_argument("--checkpoint", default="artifacts/firewatch-resnet18.pt")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint on a held-out event split")
    evaluate.add_argument("manifest")
    evaluate.add_argument("checkpoint")
    infer = commands.add_parser("predict", help="classify a local JPEG or PNG")
    infer.add_argument("image")
    infer.add_argument("--checkpoint", default="artifacts/firewatch-resnet18.pt")
    infer.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        rows = validate_manifest(args.manifest)
        print(json.dumps({"samples": len(rows), "class_counts": class_counts(rows),
                          "events": len({row['event_id'] for row in rows})}, indent=2))
    elif args.command == "train":
        rows = validate_manifest(args.manifest)
        partitions = split_by_event(rows, seed=args.seed)
        model = train_model(partitions["train"], partitions["validation"], args.checkpoint,
                            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        print(json.dumps({"checkpoint": str(Path(args.checkpoint)),
                          "split_sizes": {key: len(value) for key, value in partitions.items()},
                          "test_metrics": _evaluate(model, partitions["test"], "cuda" if torch.cuda.is_available() else "cpu")}, indent=2))
    elif args.command == "evaluate":
        rows = validate_manifest(args.manifest)
        partitions = split_by_event(rows)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_checkpoint(args.checkpoint, device=device)
        print(json.dumps({"test_metrics": _evaluate(model, partitions["test"], device)}, indent=2))
    else:
        print(json.dumps(predict(args.image, args.checkpoint, threshold=args.threshold).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
