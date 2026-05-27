from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import (
    build_dataset,
    build_metrics_summary,
    choose_attention_examples,
    compute_centroid_distance_matrix,
    load_checkpoint,
    load_latent_cache,
    load_model_from_checkpoint,
    normalize_rows,
    parse_pairs,
    plot_attention_examples,
    plot_centroid_heatmaps,
    plot_similarity_boxplot,
    plot_text_prototype_histogram,
    sample_pairwise_cosine_similarity,
)
from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier
from src.models.train_scene_classifier import get_device


DATA_ROOT = PROJECT_ROOT / "data" / "visioncraft_subset_small_v11"
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_visual_only_e20.pt"
INFO_NCE_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_infonce_e20.pt"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "latent_comparison_v11_infonce_full180"
CACHE_PATH = OUTPUT_DIR / "latent_cache.npz"
DEFAULT_PAIRS = [
    ("kitchen_dining", "restaurant_cafe"),
    ("waterfront", "mountain_valley"),
    ("public_large_indoor", "corridor_lobby"),
]


def ensure_cache() -> None:
    if CACHE_PATH.exists():
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "models" / "analyze_latent_comparison.py"),
        "--data-root",
        str(DATA_ROOT),
        "--baseline-checkpoint",
        str(BASELINE_CHECKPOINT),
        "--text-checkpoint",
        str(INFO_NCE_CHECKPOINT),
        "--samples-per-class",
        "180",
        "--num-workers",
        "0",
        "--output-dir",
        str(OUTPUT_DIR),
        "--cache-only",
    ]
    env = {
        **dict(__import__("os").environ),
        "MPLCONFIGDIR": "/private/tmp/mpl",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def run_pca(latents: np.ndarray, seed: int) -> np.ndarray:
    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(latents)


def plot_pca_comparison(
    baseline_coords: np.ndarray,
    text_coords: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    titles = ["Visual-Only Baseline PCA", "Text Cross-Attention + InfoNCE PCA"]
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
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")

    handles, _ = scatter.legend_elements(num=len(class_names))
    fig.legend(handles, class_names, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_pca(
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
                (text_latents[mask], "Text Cross-Attention + InfoNCE"),
            ]
        ):
            coords = run_pca(latents, seed + row * 17 + col)
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
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_cache()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = load_checkpoint(BASELINE_CHECKPOINT)
    text_checkpoint = load_checkpoint(INFO_NCE_CHECKPOINT)
    class_names = baseline_checkpoint["classes"]
    device = get_device()

    baseline_dataset = build_dataset(DATA_ROOT, "val", baseline_checkpoint["image_size"])
    text_model = load_model_from_checkpoint(text_checkpoint, device)
    if not isinstance(text_model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("InfoNCE checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")

    cached = load_latent_cache(CACHE_PATH)
    selected_indices = cached["selected_indices"].astype(np.int64).tolist()
    labels = cached["labels"].astype(np.int64)
    baseline_latents = normalize_rows(cached["baseline_latents"])
    text_latents = normalize_rows(cached["text_latents"])
    baseline_predictions = cached["baseline_predictions"].astype(np.int64)
    text_predictions = cached["text_predictions"].astype(np.int64)
    baseline_confidences = cached["baseline_confidences"].astype(np.float32)
    text_confidences = cached["text_confidences"].astype(np.float32)

    baseline_pca = run_pca(baseline_latents, seed=42)
    text_pca = run_pca(text_latents, seed=42)
    pca_path = OUTPUT_DIR / "baseline_vs_text_pca.png"
    plot_pca_comparison(baseline_pca, text_pca, labels, class_names, pca_path)

    baseline_centroid = compute_centroid_distance_matrix(baseline_latents, labels, len(class_names))
    text_centroid = compute_centroid_distance_matrix(text_latents, labels, len(class_names))
    centroid_path = OUTPUT_DIR / "centroid_cosine_distance_heatmaps.png"
    plot_centroid_heatmaps(baseline_centroid, text_centroid, class_names, centroid_path)

    baseline_pairwise = sample_pairwise_cosine_similarity(baseline_latents, labels, seed=42)
    text_pairwise = sample_pairwise_cosine_similarity(text_latents, labels, seed=43)
    boxplot_path = OUTPUT_DIR / "intra_inter_class_cosine_boxplot.png"
    plot_similarity_boxplot(baseline_pairwise, text_pairwise, boxplot_path)

    with np.errstate(invalid="ignore"):
        projected_text_tokens = text_model.get_projected_text_tokens().detach().cpu().numpy()
    projected_text_tokens = normalize_rows(projected_text_tokens)
    cosine_matrix = text_latents @ projected_text_tokens.T
    correct_cosines = cosine_matrix[np.arange(len(labels)), labels]
    rival_cosines = cosine_matrix.copy()
    rival_cosines[np.arange(len(labels)), labels] = -np.inf
    best_rival = rival_cosines.max(axis=1)
    prototype_path = OUTPUT_DIR / "text_prototype_cosine_histograms.png"
    plot_text_prototype_histogram(correct_cosines, correct_cosines - best_rival, prototype_path)

    pairwise_path = OUTPUT_DIR / "confusion_pair_pca_comparison.png"
    plot_pairwise_pca(
        baseline_latents=baseline_latents,
        text_latents=text_latents,
        labels=labels,
        class_names=class_names,
        pairs=DEFAULT_PAIRS,
        output_path=pairwise_path,
        seed=42,
    )

    attention_example_indices = choose_attention_examples(
        labels=labels,
        predictions=text_predictions,
        confidences=text_confidences,
        class_names=class_names,
        pairs=DEFAULT_PAIRS,
        attention_samples=6,
    )
    attention_path = OUTPUT_DIR / "text_attention_examples.png"
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

    report = {
        "data_root": str(DATA_ROOT),
        "split": "val",
        "samples_per_class": 180,
        "num_samples": int(len(labels)),
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "text_checkpoint": str(INFO_NCE_CHECKPOINT),
        "cache_file": str(CACHE_PATH),
        "embedding_method": "pca",
        "pairs": [f"{left}:{right}" for left, right in DEFAULT_PAIRS],
        "baseline_metrics": build_metrics_summary(baseline_latents, labels, baseline_pairwise),
        "text_metrics": build_metrics_summary(text_latents, labels, text_pairwise),
        "text_prototype_metrics": {
            "mean_correct_class_cosine": float(correct_cosines.mean()),
            "std_correct_class_cosine": float(correct_cosines.std()),
            "mean_correct_vs_rival_margin": float((correct_cosines - best_rival).mean()),
            "prototype_retrieval_accuracy": float((cosine_matrix.argmax(axis=1) == labels).mean()),
        },
        "outputs": {
            "pca": str(pca_path),
            "centroid_heatmaps": str(centroid_path),
            "intra_inter_boxplot": str(boxplot_path),
            "text_prototype_histograms": str(prototype_path),
            "pairwise_pca": str(pairwise_path),
            "attention_examples": str(attention_path),
        },
        "baseline_predictions_shape": list(baseline_predictions.shape),
        "baseline_confidences_shape": list(baseline_confidences.shape),
    }
    report_path = OUTPUT_DIR / "latent_comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved InfoNCE comparison artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
