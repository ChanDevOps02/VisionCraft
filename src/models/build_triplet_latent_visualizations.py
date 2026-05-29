from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import (  # noqa: E402
    compute_centroid_distance_matrix,
    load_checkpoint,
    load_latent_cache,
    load_model_from_checkpoint,
    normalize_rows,
    parse_pairs,
    run_umap,
    sample_pairwise_cosine_similarity,
)
from src.models.infonce_rerun_viz_common import DEFAULT_PAIRS  # noqa: E402
from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier  # noqa: E402
from src.models.train_scene_classifier import get_device  # noqa: E402


VANILLA_DIR = PROJECT_ROOT / "logs" / "latent_comparison_v11_full180"
INFO_NCE_DIR = PROJECT_ROOT / "logs" / "latent_comparison_v11_infonce_rerun_full180"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "latent_comparison_triplet_full180"

VANILLA_CACHE = VANILLA_DIR / "latent_cache.npz"
INFO_NCE_CACHE = INFO_NCE_DIR / "latent_cache.npz"
VANILLA_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_e20.pt"
INFO_NCE_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_infonce_e20_rerun.pt"

CLASS_NAMES = [
    "bedroom",
    "corridor_lobby",
    "forest_nature",
    "industrial_area",
    "kitchen_dining",
    "mountain_valley",
    "office_study",
    "open_field_landscape",
    "public_large_indoor",
    "residential_outdoor",
    "restaurant_cafe",
    "street_downtown",
    "transportation_hub_road",
    "waterfront",
]


def require_matching_caches() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not VANILLA_CACHE.exists():
        raise FileNotFoundError(f"Vanilla cache not found: {VANILLA_CACHE}")
    if not INFO_NCE_CACHE.exists():
        raise FileNotFoundError(f"InfoNCE cache not found: {INFO_NCE_CACHE}")

    vanilla = load_latent_cache(VANILLA_CACHE)
    infonce = load_latent_cache(INFO_NCE_CACHE)

    vanilla_indices = vanilla["selected_indices"].astype(np.int64)
    infonce_indices = infonce["selected_indices"].astype(np.int64)
    vanilla_labels = vanilla["labels"].astype(np.int64)
    infonce_labels = infonce["labels"].astype(np.int64)

    if not np.array_equal(vanilla_indices, infonce_indices):
        raise ValueError("selected_indices do not match between vanilla and InfoNCE caches.")
    if not np.array_equal(vanilla_labels, infonce_labels):
        raise ValueError("labels do not match between vanilla and InfoNCE caches.")

    return vanilla, infonce


def run_tsne(latents: np.ndarray, seed: int) -> np.ndarray:
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return reducer.fit_transform(latents)


def plot_triplet_umap(
    baseline_latents: np.ndarray,
    vanilla_latents: np.ndarray,
    infonce_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    coords_list = [
        run_umap(baseline_latents, seed=42),
        run_umap(vanilla_latents, seed=42),
        run_umap(infonce_latents, seed=42),
    ]
    titles = [
        "Visual-Only Baseline UMAP",
        "Text Cross-Attention UMAP",
        "Text Cross-Attention + InfoNCE UMAP",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=False, sharey=False)
    for ax, coords, title in zip(axes, coords_list, titles, strict=True):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    handles, _ = scatter.legend_elements(num=len(CLASS_NAMES))
    fig.legend(handles, CLASS_NAMES, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_tsne(
    baseline_latents: np.ndarray,
    vanilla_latents: np.ndarray,
    infonce_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    coords_list = [
        run_tsne(baseline_latents, seed=42),
        run_tsne(vanilla_latents, seed=42),
        run_tsne(infonce_latents, seed=42),
    ]
    titles = [
        "Visual-Only Baseline t-SNE",
        "Text Cross-Attention t-SNE",
        "Text Cross-Attention + InfoNCE t-SNE",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=False, sharey=False)
    for ax, coords, title in zip(axes, coords_list, titles, strict=True):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")

    handles, _ = scatter.legend_elements(num=len(CLASS_NAMES))
    fig.legend(handles, CLASS_NAMES, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_centroid_heatmaps(
    baseline_latents: np.ndarray,
    vanilla_latents: np.ndarray,
    infonce_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    matrices = [
        compute_centroid_distance_matrix(baseline_latents, labels, len(CLASS_NAMES)),
        compute_centroid_distance_matrix(vanilla_latents, labels, len(CLASS_NAMES)),
        compute_centroid_distance_matrix(infonce_latents, labels, len(CLASS_NAMES)),
    ]
    titles = [
        "Baseline Centroid Cosine Distance",
        "Text Cross-Attention Centroid Cosine Distance",
        "Text + InfoNCE Centroid Cosine Distance",
    ]
    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        im = ax.imshow(matrix, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_similarity_boxplot(
    baseline_latents: np.ndarray,
    vanilla_latents: np.ndarray,
    infonce_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    baseline_pairwise = sample_pairwise_cosine_similarity(baseline_latents, labels, seed=42)
    vanilla_pairwise = sample_pairwise_cosine_similarity(vanilla_latents, labels, seed=43)
    infonce_pairwise = sample_pairwise_cosine_similarity(infonce_latents, labels, seed=44)

    data = [
        baseline_pairwise["same"],
        baseline_pairwise["diff"],
        vanilla_pairwise["same"],
        vanilla_pairwise["diff"],
        infonce_pairwise["same"],
        infonce_pairwise["diff"],
    ]
    tick_labels = [
        "baseline\nsame",
        "baseline\ndiff",
        "text\nsame",
        "text\ndiff",
        "text+InfoNCE\nsame",
        "text+InfoNCE\ndiff",
    ]
    colors = ["#4c78a8", "#9ecae9", "#e45756", "#f4a6a6", "#2f855a", "#9fd6b3"]

    plt.figure(figsize=(14, 7))
    box = plt.boxplot(data, patch_artist=True, tick_labels=tick_labels, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    plt.ylabel("Cosine similarity")
    plt.title("Intra-Class vs Inter-Class Cosine Similarity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _pair_umap_coords(latents: np.ndarray, labels: np.ndarray, idx_a: int, idx_b: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.logical_or(labels == idx_a, labels == idx_b)
    coords = run_umap(latents[mask], seed=seed)
    pair_labels = labels[mask]
    return coords, pair_labels


def plot_triplet_pairwise_umap(
    baseline_latents: np.ndarray,
    vanilla_latents: np.ndarray,
    infonce_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    pairs = parse_pairs([f"{left}:{right}" for left, right in DEFAULT_PAIRS])
    class_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    titles = ["Baseline", "Text Cross-Attention", "Text + InfoNCE"]
    colors = ["#1f77b4", "#d62728"]

    fig, axes = plt.subplots(len(pairs), 3, figsize=(20, 5 * len(pairs)))
    if len(pairs) == 1:
        axes = np.array([axes])

    for row, (class_a, class_b) in enumerate(pairs):
        idx_a = class_to_index[class_a]
        idx_b = class_to_index[class_b]
        triplet = [
            _pair_umap_coords(baseline_latents, labels, idx_a, idx_b, 42 + row * 17 + 0),
            _pair_umap_coords(vanilla_latents, labels, idx_a, idx_b, 42 + row * 17 + 1),
            _pair_umap_coords(infonce_latents, labels, idx_a, idx_b, 42 + row * 17 + 2),
        ]

        for col, ((coords, pair_labels), title_prefix) in enumerate(zip(triplet, titles, strict=True)):
            ax = axes[row, col]
            for class_index, color in zip([idx_a, idx_b], colors, strict=True):
                class_mask = pair_labels == class_index
                ax.scatter(coords[class_mask, 0], coords[class_mask, 1], s=12, alpha=0.8, color=color, label=CLASS_NAMES[class_index])
            ax.set_title(f"{title_prefix}: {class_a} vs {class_b}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_prototype_histogram(
    vanilla_text_latents: np.ndarray,
    infonce_text_latents: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    device = get_device()
    vanilla_model = load_model_from_checkpoint(load_checkpoint(VANILLA_CHECKPOINT), device)
    infonce_model = load_model_from_checkpoint(load_checkpoint(INFO_NCE_CHECKPOINT), device)

    if not isinstance(vanilla_model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("Vanilla text checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")
    if not isinstance(infonce_model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("InfoNCE checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")

    vanilla_tokens = normalize_rows(vanilla_model.get_projected_text_tokens().detach().cpu().numpy())
    infonce_tokens = normalize_rows(infonce_model.get_projected_text_tokens().detach().cpu().numpy())

    vanilla_cos = vanilla_text_latents @ vanilla_tokens.T
    infonce_cos = infonce_text_latents @ infonce_tokens.T

    vanilla_correct = vanilla_cos[np.arange(len(labels)), labels]
    infonce_correct = infonce_cos[np.arange(len(labels)), labels]

    vanilla_rival = vanilla_cos.copy()
    infonce_rival = infonce_cos.copy()
    vanilla_rival[np.arange(len(labels)), labels] = -np.inf
    infonce_rival[np.arange(len(labels)), labels] = -np.inf

    vanilla_margin = vanilla_correct - vanilla_rival.max(axis=1)
    infonce_margin = infonce_correct - infonce_rival.max(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].axis("off")
    axes[0].text(
        0.5,
        0.5,
        "Baseline has no text prototype\nalignment by design.",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[0].set_title("Visual-Only Baseline")

    for ax, correct, margin, title, color_a, color_b in [
        (axes[1], vanilla_correct, vanilla_margin, "Text Cross-Attention", "#2b6cb0", "#2f855a"),
        (axes[2], infonce_correct, infonce_margin, "Text Cross-Attention + InfoNCE", "#7b2cbf", "#c05621"),
    ]:
        ax.hist(correct, bins=28, alpha=0.72, color=color_a, edgecolor="white", label="correct cosine")
        ax.hist(margin, bins=28, alpha=0.55, color=color_b, edgecolor="white", label="correct-rival margin")
        ax.axvline(correct.mean(), color=color_a, linestyle="--", linewidth=2)
        ax.axvline(margin.mean(), color=color_b, linestyle="--", linewidth=2)
        ax.set_xlabel("Cosine / Margin")
        ax.set_ylabel("Sample count")
        ax.legend(fontsize=8)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vanilla, infonce = require_matching_caches()

    labels = vanilla["labels"].astype(np.int64)
    baseline_latents = normalize_rows(vanilla["baseline_latents"])
    vanilla_text_latents = normalize_rows(vanilla["text_latents"])
    infonce_text_latents = normalize_rows(infonce["text_latents"])

    plot_triplet_umap(
        baseline_latents,
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_umap.png",
    )
    plot_triplet_tsne(
        baseline_latents,
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_tsne.png",
    )
    plot_triplet_centroid_heatmaps(
        baseline_latents,
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_centroid_cosine_distance_heatmaps.png",
    )
    plot_triplet_similarity_boxplot(
        baseline_latents,
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_intra_inter_class_cosine_boxplot.png",
    )
    plot_triplet_pairwise_umap(
        baseline_latents,
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_confusion_pair_umap_comparison.png",
    )
    plot_triplet_prototype_histogram(
        vanilla_text_latents,
        infonce_text_latents,
        labels,
        OUTPUT_DIR / "triplet_prototype_alignment_overview.png",
    )
    print(f"Saved triplet visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
