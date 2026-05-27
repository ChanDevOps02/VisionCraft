from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models as tv_models
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import text_cross_attention as text_cross_attention_module
from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier
from src.models.train_scene_classifier import build_transforms, get_device, unpack_batch


DEFAULT_DATA_ROOT = "data/visioncraft_subset_small_v11"
DEFAULT_BASELINE_CHECKPOINT = "checkpoint/scene_classifier_resnet50_v11_visual_only_e20.pt"
DEFAULT_TEXT_CHECKPOINT = "checkpoint/scene_classifier_resnet50_v11_text_crossattn_e20.pt"
DEFAULT_PAIRS = [
    "kitchen_dining:restaurant_cafe",
    "waterfront:mountain_valley",
    "public_large_indoor:corridor_lobby",
]
ALL_STAGES = ["umap", "centroid", "similarity", "prototype", "pairwise", "attention", "report"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare visual-only and text cross-attention latent spaces with multiple visualizations."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-checkpoint", type=str, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=str, default=DEFAULT_TEXT_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/latent_comparison_v11")
    parser.add_argument("--pair", action="append", default=[], help="Class pair in the form class_a:class_b.")
    parser.add_argument("--attention-samples", type=int, default=6)
    parser.add_argument("--cache-file", type=str, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--load-cache", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES + ["all"],
        default=["all"],
        help="Which stages to run after latent extraction/loading.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu")


def build_dataset(data_root: Path, split: str, image_size: int):
    _, eval_transform = build_transforms(image_size)
    return datasets.ImageFolder(data_root / split, transform=eval_transform)


def select_balanced_indices(dataset: datasets.ImageFolder, samples_per_class: int, seed: int) -> list[int]:
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(index)

    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(indices_by_class):
        label_indices = list(indices_by_class[label])
        rng.shuffle(label_indices)
        selected.extend(label_indices[:samples_per_class])
    selected.sort()
    return selected


def build_loader(dataset, indices: list[int], batch_size: int, num_workers: int):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return subset, loader


def extract_visual_latent(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.conv1(images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def load_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device):
    fusion_mode = checkpoint.get("fusion_mode", "visual-only")
    backbone = checkpoint.get("backbone", "resnet18")

    if fusion_mode == "visual-only":
        if backbone == "resnet50":
            model = tv_models.resnet50(weights=None)
        else:
            model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    elif fusion_mode == "text-cross-attention":
        original_resnet18 = text_cross_attention_module.models.resnet18
        original_resnet50 = text_cross_attention_module.models.resnet50

        def _resnet18_no_weights(*args, **kwargs):
            kwargs["weights"] = None
            return original_resnet18(*args, **kwargs)

        def _resnet50_no_weights(*args, **kwargs):
            kwargs["weights"] = None
            return original_resnet50(*args, **kwargs)

        text_cross_attention_module.models.resnet18 = _resnet18_no_weights
        text_cross_attention_module.models.resnet50 = _resnet50_no_weights
        try:
            model = ResNetTextCrossAttentionSceneClassifier(
                backbone_name=backbone,
                num_classes=len(checkpoint["classes"]),
                scene_text_embeddings=checkpoint.get("scene_text_embeddings", None),
                dropout=checkpoint.get("cross_attention_dropout", 0.1),
            )
        finally:
            text_cross_attention_module.models.resnet18 = original_resnet18
            text_cross_attention_module.models.resnet50 = original_resnet50
    else:
        raise ValueError(f"Unsupported fusion mode for this comparison script: {fusion_mode}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def extract_model_outputs(
    model: torch.nn.Module,
    fusion_mode: str,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    latents: list[np.ndarray] = []
    labels: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch_data = unpack_batch(batch, device)
            images = batch_data["images"]
            if fusion_mode == "visual-only":
                latent = extract_visual_latent(model, images)
                logits = model(images)
            elif fusion_mode == "text-cross-attention":
                latent = model.extract_fused_latent(images)
                logits = model.classifier(latent)
            else:
                raise ValueError(f"Unsupported fusion mode for this comparison script: {fusion_mode}")

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            max_probs = probs.max(dim=1).values

            latents.append(latent.cpu().numpy())
            labels.extend(batch_data["labels"].cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            confidences.extend(max_probs.cpu().numpy().tolist())

    return {
        "latents": np.concatenate(latents, axis=0),
        "labels": np.array(labels, dtype=np.int64),
        "predictions": np.array(predictions, dtype=np.int64),
        "confidences": np.array(confidences, dtype=np.float32),
    }


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def compute_centroid_distance_matrix(normalized_latents: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []
    for class_index in range(num_classes):
        class_vectors = normalized_latents[labels == class_index]
        centroid = class_vectors.mean(axis=0)
        centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-12, None)
        centroids.append(centroid)
    centroid_matrix = np.stack(centroids, axis=0)
    cosine_similarity = centroid_matrix @ centroid_matrix.T
    return 1.0 - cosine_similarity


def sample_pairwise_cosine_similarity(
    normalized_latents: np.ndarray,
    labels: np.ndarray,
    max_same_pairs: int = 12000,
    max_diff_pairs: int = 12000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    same_similarities: list[float] = []
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        pair_count = min(max_same_pairs // max(len(label_to_indices), 1) + 1, len(indices) * 2)
        for _ in range(pair_count):
            i, j = rng.choice(indices, size=2, replace=False)
            same_similarities.append(float(normalized_latents[i] @ normalized_latents[j]))

    diff_similarities: list[float] = []
    unique_labels = sorted(label_to_indices)
    for _ in range(max_diff_pairs):
        label_a, label_b = rng.choice(unique_labels, size=2, replace=False)
        idx_a = int(rng.choice(label_to_indices[label_a]))
        idx_b = int(rng.choice(label_to_indices[label_b]))
        diff_similarities.append(float(normalized_latents[idx_a] @ normalized_latents[idx_b]))

    return {
        "same": np.array(same_similarities[:max_same_pairs], dtype=np.float32),
        "diff": np.array(diff_similarities, dtype=np.float32),
    }


def run_umap(latents: np.ndarray, seed: int) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(latents)


def plot_umap_comparison(
    baseline_coords: np.ndarray,
    text_coords: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    titles = ["Visual-Only Baseline UMAP", "Text Cross-Attention UMAP"]
    coords_list = [baseline_coords, text_coords]

    for ax, coords, title in zip(axes, coords_list, titles, strict=True):
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="tab20",
            s=10,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    handles, _ = scatter.legend_elements(num=len(class_names))
    fig.legend(handles, class_names, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_heatmaps(
    baseline_matrix: np.ndarray,
    text_matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    matrices = [baseline_matrix, text_matrix]
    titles = ["Baseline Centroid Cosine Distance", "Text Cross-Attention Centroid Cosine Distance"]

    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)

    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        im = ax.imshow(matrix, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_boxplot(
    baseline_pairwise: dict[str, np.ndarray],
    text_pairwise: dict[str, np.ndarray],
    output_path: Path,
):
    data = [
        baseline_pairwise["same"],
        baseline_pairwise["diff"],
        text_pairwise["same"],
        text_pairwise["diff"],
    ]
    labels = [
        "baseline same-class",
        "baseline different-class",
        "text same-class",
        "text different-class",
    ]

    plt.figure(figsize=(11, 7))
    box = plt.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
    colors = ["#4c78a8", "#9ecae9", "#e45756", "#f4a6a6"]
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    plt.ylabel("Cosine similarity")
    plt.title("Intra-Class vs Inter-Class Cosine Similarity")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_text_prototype_histogram(
    correct_cosines: np.ndarray,
    margins: np.ndarray,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(correct_cosines, bins=30, color="#2b6cb0", alpha=0.85, edgecolor="white")
    axes[0].axvline(correct_cosines.mean(), color="#c53030", linestyle="--", linewidth=2)
    axes[0].set_title("Correct-Class Cosine to Text Prototype")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Sample count")

    axes[1].hist(margins, bins=30, color="#2f855a", alpha=0.85, edgecolor="white")
    axes[1].axvline(margins.mean(), color="#c53030", linestyle="--", linewidth=2)
    axes[1].set_title("Correct-vs-Rival Prototype Margin")
    axes[1].set_xlabel("Cosine margin")
    axes[1].set_ylabel("Sample count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"Pair must be in class_a:class_b format, got: {raw}")
        left, right = raw.split(":", 1)
        parsed.append((left, right))
    return parsed


def plot_pairwise_umap(
    baseline_latents: np.ndarray,
    text_latents: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    pairs: list[tuple[str, str]],
    output_path: Path,
    seed: int,
):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 5 * len(pairs)))
    if len(pairs) == 1:
        axes = np.array([axes])

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    colors = ["#1f77b4", "#d62728"]

    for row, (class_a, class_b) in enumerate(pairs):
        idx_a = class_to_index[class_a]
        idx_b = class_to_index[class_b]
        mask = np.logical_or(labels == idx_a, labels == idx_b)
        pair_labels = labels[mask]

        for col, (latents, title_prefix) in enumerate(
            [
                (baseline_latents[mask], "Baseline"),
                (text_latents[mask], "Text Cross-Attention"),
            ]
        ):
            coords = run_umap(latents, seed=seed + row * 17 + col)
            ax = axes[row, col]
            for class_index, color in zip([idx_a, idx_b], colors, strict=True):
                class_mask = pair_labels == class_index
                ax.scatter(
                    coords[class_mask, 0],
                    coords[class_mask, 1],
                    s=12,
                    alpha=0.8,
                    color=color,
                    label=class_names[class_index],
                )
            ax.set_title(f"{title_prefix}: {class_a} vs {class_b}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_to_rgb(base_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    import cv2

    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / max(float(heatmap.max()), 1e-12)
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_image, 0.58, colored, 0.42, 0)


def extract_attention_map(
    model: ResNetTextCrossAttentionSceneClassifier,
    image_tensor: torch.Tensor,
    class_index: int,
) -> np.ndarray:
    with torch.no_grad():
        visual_tokens = model.encode_visual_tokens(image_tensor)
        text_tokens = model.get_text_tokens(image_tensor.size(0))
        _ = model.cross_attention(visual_tokens, text_tokens)
        attention_weights = model.cross_attention.last_attention_weights
        if attention_weights is None:
            raise RuntimeError("Attention weights were not captured.")

        attention = attention_weights[0].mean(dim=0)[:, class_index].cpu().numpy()
        side = int(round(math.sqrt(attention.shape[0])))
        if side * side != attention.shape[0]:
            raise ValueError("Visual token count is not a square number; cannot reshape to map.")
        return attention.reshape(side, side)


def choose_attention_examples(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    pairs: list[tuple[str, str]],
    attention_samples: int,
) -> list[int]:
    preferred_classes = []
    for pair in pairs:
        preferred_classes.extend(pair)
    seen = set()
    ordered_classes = []
    for class_name in preferred_classes:
        if class_name not in seen:
            seen.add(class_name)
            ordered_classes.append(class_name)

    selected: list[int] = []
    for class_name in ordered_classes:
        class_index = class_names.index(class_name)
        candidates = np.where(np.logical_and(labels == class_index, predictions == class_index))[0]
        if len(candidates) == 0:
            continue
        best_idx = int(candidates[np.argmax(confidences[candidates])])
        selected.append(best_idx)
        if len(selected) >= attention_samples:
            break

    if len(selected) < attention_samples:
        remaining = [idx for idx in np.argsort(-confidences) if idx not in selected and labels[idx] == predictions[idx]]
        selected.extend(remaining[: attention_samples - len(selected)])
    return selected[:attention_samples]


def plot_attention_examples(
    model: ResNetTextCrossAttentionSceneClassifier,
    dataset: datasets.ImageFolder,
    selected_indices: list[int],
    subset_indices: list[int],
    class_names: list[str],
    image_size: int,
    output_path: Path,
    device: torch.device,
):
    _, eval_transform = build_transforms(image_size)
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(12, 4 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = np.array([axes])

    for row, subset_idx in enumerate(selected_indices):
        dataset_idx = subset_indices[subset_idx]
        image_path, label = dataset.samples[dataset_idx]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        image_tensor = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            pred_conf = float(probs[pred_idx].item())

        true_map = extract_attention_map(model, image_tensor, class_index=label)
        pred_map = extract_attention_map(model, image_tensor, class_index=pred_idx)

        true_overlay = heatmap_to_rgb(image_np, np.array(Image.fromarray((true_map * 255).astype(np.uint8)).resize(image.size)))
        pred_overlay = heatmap_to_rgb(image_np, np.array(Image.fromarray((pred_map * 255).astype(np.uint8)).resize(image.size)))

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"Original\ntrue={class_names[label]}")
        axes[row, 1].imshow(true_overlay)
        axes[row, 1].set_title(f"True-class attention\n{class_names[label]}")
        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f"Pred attention\n{class_names[pred_idx]} ({pred_conf:.3f})")

        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_metrics_summary(
    normalized_latents: np.ndarray,
    labels: np.ndarray,
    pairwise: dict[str, np.ndarray],
) -> dict[str, float]:
    return {
        "mean_same_class_cosine": float(pairwise["same"].mean()),
        "mean_diff_class_cosine": float(pairwise["diff"].mean()),
        "same_minus_diff_margin": float(pairwise["same"].mean() - pairwise["diff"].mean()),
        "silhouette_score": float(silhouette_score(normalized_latents, labels, metric="cosine")),
    }


def save_latent_cache(
    cache_path: Path,
    baseline_latents: np.ndarray,
    text_latents: np.ndarray,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    text_predictions: np.ndarray,
    baseline_confidences: np.ndarray,
    text_confidences: np.ndarray,
    selected_indices: list[int],
):
    np.savez_compressed(
        cache_path,
        baseline_latents=baseline_latents,
        text_latents=text_latents,
        labels=labels,
        baseline_predictions=baseline_predictions,
        text_predictions=text_predictions,
        baseline_confidences=baseline_confidences,
        text_confidences=text_confidences,
        selected_indices=np.array(selected_indices, dtype=np.int64),
    )


def load_latent_cache(cache_path: Path) -> dict[str, np.ndarray]:
    cached = np.load(cache_path)
    return {key: cached[key] for key in cached.files}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_file) if args.cache_file else output_dir / "latent_cache.npz"
    stages = ALL_STAGES if "all" in args.stages else args.stages

    device = get_device()
    baseline_checkpoint = load_checkpoint(Path(args.baseline_checkpoint))
    text_checkpoint = load_checkpoint(Path(args.text_checkpoint))

    baseline_fusion = baseline_checkpoint.get("fusion_mode", "visual-only")
    text_fusion = text_checkpoint.get("fusion_mode", "visual-only")
    if baseline_fusion != "visual-only":
        raise ValueError(f"Baseline checkpoint must be visual-only, got {baseline_fusion}")
    if text_fusion != "text-cross-attention":
        raise ValueError(f"Text checkpoint must be text-cross-attention, got {text_fusion}")

    class_names = baseline_checkpoint["classes"]
    if class_names != text_checkpoint["classes"]:
        raise ValueError("Checkpoint class orders do not match.")

    baseline_dataset = build_dataset(Path(args.data_root), args.split, baseline_checkpoint["image_size"])
    baseline_model = load_model_from_checkpoint(baseline_checkpoint, device)
    text_model = load_model_from_checkpoint(text_checkpoint, device)
    if not isinstance(text_model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("Text checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")

    if args.load_cache and cache_path.exists():
        print(f"loading latent cache from {cache_path}...", flush=True)
        cached = load_latent_cache(cache_path)
        selected_indices = cached["selected_indices"].astype(np.int64).tolist()
        labels = cached["labels"].astype(np.int64)
        baseline_latents = cached["baseline_latents"]
        text_latents = cached["text_latents"]
        baseline_outputs = {
            "predictions": cached["baseline_predictions"].astype(np.int64),
            "confidences": cached["baseline_confidences"].astype(np.float32),
        }
        text_outputs = {
            "predictions": cached["text_predictions"].astype(np.int64),
            "confidences": cached["text_confidences"].astype(np.float32),
        }
    else:
        selected_indices = select_balanced_indices(baseline_dataset, args.samples_per_class, args.seed)
        _, loader = build_loader(baseline_dataset, selected_indices, args.batch_size, args.num_workers)

        print("extracting baseline latents...", flush=True)
        baseline_outputs = extract_model_outputs(baseline_model, baseline_fusion, loader, device)
        print("extracting text latents...", flush=True)
        text_outputs = extract_model_outputs(text_model, text_fusion, loader, device)
        labels = baseline_outputs["labels"]

        baseline_latents = normalize_rows(baseline_outputs["latents"])
        text_latents = normalize_rows(text_outputs["latents"])
        print(f"saving latent cache to {cache_path}...", flush=True)
        save_latent_cache(
            cache_path=cache_path,
            baseline_latents=baseline_latents,
            text_latents=text_latents,
            labels=labels,
            baseline_predictions=baseline_outputs["predictions"],
            text_predictions=text_outputs["predictions"],
            baseline_confidences=baseline_outputs["confidences"],
            text_confidences=text_outputs["confidences"],
            selected_indices=selected_indices,
        )

    if args.cache_only:
        print(f"Saved latent cache to {cache_path}", flush=True)
        return

    umap_path = output_dir / "baseline_vs_text_umap.png"
    centroid_path = output_dir / "centroid_cosine_distance_heatmaps.png"
    boxplot_path = output_dir / "intra_inter_class_cosine_boxplot.png"
    cosine_hist_path = output_dir / "text_prototype_cosine_histograms.png"
    pairs = parse_pairs(args.pair or DEFAULT_PAIRS)
    pairwise_path = output_dir / "confusion_pair_umap_comparison.png"
    attention_path = output_dir / "text_attention_examples.png"
    baseline_pairwise = None
    text_pairwise = None
    correct_cosines = None
    best_rival = None
    cosine_matrix = None

    if "umap" in stages:
        print("running UMAP comparison...", flush=True)
        baseline_umap = run_umap(baseline_latents, args.seed)
        text_umap = run_umap(text_latents, args.seed)
        plot_umap_comparison(baseline_umap, text_umap, labels, class_names, umap_path)

    if "centroid" in stages:
        print("building centroid heatmaps...", flush=True)
        baseline_centroid = compute_centroid_distance_matrix(baseline_latents, labels, len(class_names))
        text_centroid = compute_centroid_distance_matrix(text_latents, labels, len(class_names))
        plot_centroid_heatmaps(baseline_centroid, text_centroid, class_names, centroid_path)

    if "similarity" in stages or "report" in stages:
        print("sampling intra/inter cosine similarities...", flush=True)
        baseline_pairwise = sample_pairwise_cosine_similarity(
            baseline_latents,
            labels,
            seed=args.seed,
        )
        text_pairwise = sample_pairwise_cosine_similarity(
            text_latents,
            labels,
            seed=args.seed + 1,
        )
        if "similarity" in stages:
            plot_similarity_boxplot(baseline_pairwise, text_pairwise, boxplot_path)

    if "prototype" in stages or "report" in stages:
        print("computing text prototype metrics...", flush=True)
        with torch.no_grad():
            projected_text_tokens = text_model.get_projected_text_tokens().cpu().numpy()
        projected_text_tokens = normalize_rows(projected_text_tokens)
        cosine_matrix = text_latents @ projected_text_tokens.T
        correct_cosines = cosine_matrix[np.arange(len(labels)), labels]
        rival_cosines = cosine_matrix.copy()
        rival_cosines[np.arange(len(labels)), labels] = -np.inf
        best_rival = rival_cosines.max(axis=1)
        if "prototype" in stages:
            plot_text_prototype_histogram(correct_cosines, correct_cosines - best_rival, cosine_hist_path)

    if "pairwise" in stages:
        print("running confusion-pair UMAP plots...", flush=True)
        plot_pairwise_umap(baseline_latents, text_latents, labels, class_names, pairs, pairwise_path, args.seed)

    if "attention" in stages:
        print("rendering attention examples...", flush=True)
        attention_example_indices = choose_attention_examples(
            labels=labels,
            predictions=text_outputs["predictions"],
            confidences=text_outputs["confidences"],
            class_names=class_names,
            pairs=pairs,
            attention_samples=args.attention_samples,
        )
        plot_attention_examples(
            model=text_model,
            dataset=baseline_dataset,
            selected_indices=attention_example_indices,
            subset_indices=selected_indices,
            class_names=class_names,
            image_size=text_checkpoint["image_size"],
            output_path=attention_path,
            device=device,
        )

    if "report" not in stages:
        return

    print("computing summary metrics...", flush=True)
    report = {
        "data_root": args.data_root,
        "split": args.split,
        "samples_per_class": args.samples_per_class,
        "num_samples": int(len(labels)),
        "baseline_checkpoint": args.baseline_checkpoint,
        "text_checkpoint": args.text_checkpoint,
        "cache_file": str(cache_path),
        "pairs": [f"{left}:{right}" for left, right in pairs],
        "baseline_metrics": build_metrics_summary(baseline_latents, labels, baseline_pairwise),
        "text_metrics": build_metrics_summary(text_latents, labels, text_pairwise),
        "text_prototype_metrics": {
            "mean_correct_class_cosine": float(correct_cosines.mean()),
            "std_correct_class_cosine": float(correct_cosines.std()),
            "mean_correct_vs_rival_margin": float((correct_cosines - best_rival).mean()),
            "prototype_retrieval_accuracy": float((cosine_matrix.argmax(axis=1) == labels).mean()),
        },
        "outputs": {
            "umap": str(umap_path),
            "centroid_heatmaps": str(centroid_path),
            "intra_inter_boxplot": str(boxplot_path),
            "text_prototype_histograms": str(cosine_hist_path),
            "pairwise_umap": str(pairwise_path),
            "attention_examples": str(attention_path),
        },
    }

    report_path = output_dir / "latent_comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
