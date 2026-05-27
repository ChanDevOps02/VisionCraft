from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier
from src.models.train_scene_classifier import build_model, build_transforms, get_device, unpack_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze fused latent space of text cross-attention scene classifier.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tsne-path", type=str, default="logs/text_crossattn_tsne.png")
    parser.add_argument("--cosine-path", type=str, default="logs/text_crossattn_cosine.png")
    parser.add_argument("--report-path", type=str, default="logs/text_crossattn_latent_report.json")
    parser.add_argument("--max-samples", type=int, default=3000)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    return torch.load(checkpoint_path, map_location="cpu")


def main():
    args = parse_args()
    device = get_device()
    checkpoint = load_checkpoint(Path(args.checkpoint))

    fusion_mode = checkpoint.get("fusion_mode", "visual-only")
    if fusion_mode != "text-cross-attention":
        raise ValueError(f"This analysis script expects a text-cross-attention checkpoint, got {fusion_mode}")

    image_size = checkpoint["image_size"]
    classes = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "resnet18")
    cross_attention_dropout = checkpoint.get("cross_attention_dropout", 0.1)
    scene_text_embeddings = checkpoint.get("scene_text_embeddings", None)
    if scene_text_embeddings is None:
        raise ValueError("Checkpoint does not contain scene_text_embeddings.")

    _, eval_transform = build_transforms(image_size)
    dataset = datasets.ImageFolder(Path(args.data_root) / args.split, transform=eval_transform)
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
        scene_text_embeddings=scene_text_embeddings,
        cross_attention_dropout=cross_attention_dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if not isinstance(model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("Loaded model is not ResNetTextCrossAttentionSceneClassifier")

    latents: list[np.ndarray] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch_data = unpack_batch(batch, device)
            fused_latent = model.extract_fused_latent(batch_data["images"])
            latents.append(fused_latent.cpu().numpy())
            labels.extend(batch_data["labels"].cpu().numpy().tolist())
            if sum(len(chunk) for chunk in latents) >= args.max_samples:
                break

    latent_matrix = np.concatenate(latents, axis=0)[: args.max_samples]
    label_array = np.array(labels[: args.max_samples], dtype=np.int64)

    with torch.no_grad():
        projected_text_tokens = model.get_projected_text_tokens().cpu().numpy()

    latent_norm = latent_matrix / np.clip(np.linalg.norm(latent_matrix, axis=1, keepdims=True), 1e-12, None)
    text_norm = projected_text_tokens / np.clip(np.linalg.norm(projected_text_tokens, axis=1, keepdims=True), 1e-12, None)
    cosine_matrix = latent_norm @ text_norm.T
    correct_cosines = cosine_matrix[np.arange(len(label_array)), label_array]
    predicted_text_indices = cosine_matrix.argmax(axis=1)
    retrieval_accuracy = float((predicted_text_indices == label_array).mean())

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_coords = tsne.fit_transform(latent_matrix)

    tsne_path = Path(args.tsne_path)
    tsne_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=label_array,
        cmap="tab20",
        s=8,
        alpha=0.75,
    )
    handles, _ = scatter.legend_elements(num=len(classes))
    plt.legend(handles, classes, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.title(f"t-SNE of Text Cross-Attention Fused Latents ({args.split})")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=220, bbox_inches="tight")
    plt.close()

    cosine_path = Path(args.cosine_path)
    cosine_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(correct_cosines, bins=30, color="#2b6cb0", alpha=0.85, edgecolor="white")
    plt.axvline(correct_cosines.mean(), color="#c53030", linestyle="--", linewidth=2, label=f"mean={correct_cosines.mean():.4f}")
    plt.title(f"Correct-Class Cosine Similarity to Text Prototype ({args.split})")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Sample count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cosine_path, dpi=220, bbox_inches="tight")
    plt.close()

    per_class_stats = {}
    for class_index, class_name in enumerate(classes):
        mask = label_array == class_index
        if not np.any(mask):
            continue
        values = correct_cosines[mask]
        per_class_stats[class_name] = {
            "count": int(mask.sum()),
            "mean_cosine": float(values.mean()),
            "std_cosine": float(values.std()),
            "min_cosine": float(values.min()),
            "max_cosine": float(values.max()),
        }

    report = {
        "split": args.split,
        "num_samples": int(len(label_array)),
        "latent_dim": int(latent_matrix.shape[1]),
        "overall_correct_cosine_mean": float(correct_cosines.mean()),
        "overall_correct_cosine_std": float(correct_cosines.std()),
        "text_prototype_retrieval_accuracy": retrieval_accuracy,
        "per_class_stats": per_class_stats,
        "tsne_path": str(tsne_path),
        "cosine_path": str(cosine_path),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved t-SNE figure to {tsne_path}")
    print(f"Saved cosine histogram to {cosine_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
