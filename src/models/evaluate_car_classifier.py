from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_car_classifier import build_model, build_transforms
from src.models.train_scene_classifier import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CompCars classifier.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--report-path", type=str, default="")
    parser.add_argument("--figure-path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    _, eval_transform = build_transforms(checkpoint["image_size"])
    dataset = datasets.ImageFolder(Path(args.data_root) / args.split, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(num_classes=len(checkpoint["classes"]), backbone=checkpoint["backbone"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=dataset.classes, digits=4, zero_division=0)

    output_text = "Classification Report:\n" + report
    print(output_text)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output_text, encoding="utf-8")

    if args.figure_path:
        figure_path = Path(args.figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig_width = max(10, len(dataset.classes) * 0.7)
        fig_height = max(8, len(dataset.classes) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({args.split})")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(len(dataset.classes)))
        ax.set_yticks(range(len(dataset.classes)))
        ax.set_xticklabels(dataset.classes, rotation=45, ha="right")
        ax.set_yticklabels(dataset.classes)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
