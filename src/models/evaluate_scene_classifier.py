from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_scene_classifier import (
    ImageFolderWithObjectFeatures,
    ImageFolderWithObjectTokens,
    ImageFolderWithSegmentationFeatures,
    build_model,
    build_transforms,
    forward_model,
    get_device,
    unpack_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained VisionCraft scene classifier.")
    parser.add_argument("--data-root", type=str, required=True, help="Subset root with train/ and val/ folders.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--object-features-path", type=str, default="")
    parser.add_argument("--object-tokens-path", type=str, default="")
    parser.add_argument("--segmentation-features-path", type=str, default="")
    parser.add_argument("--scene-text-embeddings-path", type=str, default="")
    parser.add_argument("--report-path", type=str, default="", help="Optional path to save the text report.")
    parser.add_argument("--figure-path", type=str, default="", help="Optional path to save confusion matrix heatmap PNG.")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    return torch.load(checkpoint_path, map_location="cpu")


def evaluate(args):
    device = get_device()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    image_size = checkpoint["image_size"]
    classes = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "resnet18")
    fusion_mode = checkpoint.get("fusion_mode", "visual-only")
    object_feature_dim = checkpoint.get("object_feature_dim", 0)
    object_max_objects = checkpoint.get("object_max_objects", 16)
    segmentation_feature_dim = checkpoint.get("segmentation_feature_dim", 0)
    scene_text_embeddings = checkpoint.get("scene_text_embeddings", None)
    cross_attention_dropout = checkpoint.get("cross_attention_dropout", 0.1)

    _, eval_transform = build_transforms(image_size)
    dataset = datasets.ImageFolder(Path(args.data_root) / args.split, transform=eval_transform)
    if fusion_mode == "late-fusion":
        if not args.object_features_path:
            raise ValueError("late-fusion evaluation requires --object-features-path")
        dataset = ImageFolderWithObjectFeatures(
            dataset,
            Path(args.data_root),
            Path(args.object_features_path),
        )
    elif fusion_mode == "cross-attention":
        if not args.object_tokens_path:
            raise ValueError("cross-attention evaluation requires --object-tokens-path")
        dataset = ImageFolderWithObjectTokens(
            dataset,
            Path(args.data_root),
            Path(args.object_tokens_path),
        )
    elif fusion_mode == "segmentation-cross-attention":
        if not args.segmentation_features_path:
            raise ValueError("segmentation-cross-attention evaluation requires --segmentation-features-path")
        dataset = ImageFolderWithSegmentationFeatures(
            dataset,
            Path(args.data_root),
            Path(args.segmentation_features_path),
        )
    elif fusion_mode == "text-cross-attention":
        if scene_text_embeddings is None:
            if not args.scene_text_embeddings_path:
                raise ValueError("text-cross-attention evaluation requires checkpoint embeddings or --scene-text-embeddings-path")
            text_data = np.load(args.scene_text_embeddings_path, allow_pickle=True)
            scene_text_embeddings = torch.from_numpy(text_data["embeddings"].astype(np.float32))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        num_classes=len(classes),
        backbone=backbone,
        fusion_mode=fusion_mode,
        object_feature_dim=object_feature_dim,
        object_max_objects=object_max_objects,
        segmentation_feature_dim=segmentation_feature_dim,
        scene_text_embeddings=scene_text_embeddings,
        cross_attention_dropout=cross_attention_dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch_data = unpack_batch(batch, device)
            logits = forward_model(
                model,
                batch_data["images"],
                batch_data["object_features"],
                batch_data["segmentation_features"],
                batch_data["object_class_ids"],
                batch_data["object_geometry"],
                batch_data["object_valid_mask"],
            )
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(batch_data["labels"].cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=dataset.classes,
        digits=4,
        zero_division=0,
    )

    cm_lines = ["Confusion Matrix:"]
    header = "pred-> " + " ".join(f"{idx:>4}" for idx in range(len(dataset.classes)))
    cm_lines.append(header)
    for idx, row in enumerate(cm):
        cm_lines.append(f"true {idx:>2}: " + " ".join(f"{value:>4}" for value in row))

    class_lines = ["", "Class Index Mapping:"]
    for idx, class_name in enumerate(dataset.classes):
        class_lines.append(f"{idx}: {class_name}")

    output_text = "\n".join(cm_lines + class_lines + ["", "Classification Report:", report])
    print(output_text)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output_text, encoding="utf-8")
        print(f"\nSaved report to {report_path}")

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

        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                value = cm[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > cm.max() * 0.5 else "black",
                    fontsize=8,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix figure to {figure_path}")


if __name__ == "__main__":
    evaluate(parse_args())
