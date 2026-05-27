from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import (
    build_dataset,
    build_loader,
    extract_model_outputs,
    load_checkpoint,
    load_latent_cache,
    load_model_from_checkpoint,
    normalize_rows,
    save_latent_cache,
    select_balanced_indices,
)
DATA_ROOT = PROJECT_ROOT / "data" / "visioncraft_subset_small_v11"
BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_visual_only_e20.pt"
TEXT_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_e20.pt"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "latent_comparison_v11_full180"
CACHE_PATH = OUTPUT_DIR / "latent_cache.npz"
OUTPUT_PATH = OUTPUT_DIR / "baseline_vs_text_tsne.png"
SAMPLES_PER_CLASS = 180
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42
SPLIT = "val"


def run_tsne(latents: np.ndarray, seed: int) -> np.ndarray:
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return reducer.fit_transform(latents)


def plot_tsne_comparison(
    baseline_coords: np.ndarray,
    text_coords: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    titles = ["Visual-Only Baseline t-SNE", "Text Cross-Attention t-SNE"]
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
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")

    handles, _ = scatter.legend_elements(num=len(class_names))
    fig.legend(handles, class_names, loc="center right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def ensure_cache() -> tuple[dict[str, np.ndarray], list[str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"using device: {device}", flush=True)

    baseline_checkpoint = load_checkpoint(BASELINE_CHECKPOINT)
    text_checkpoint = load_checkpoint(TEXT_CHECKPOINT)
    class_names = baseline_checkpoint["classes"]

    if CACHE_PATH.exists():
        return load_latent_cache(CACHE_PATH), class_names

    dataset = build_dataset(DATA_ROOT, SPLIT, baseline_checkpoint["image_size"])
    selected_indices = select_balanced_indices(dataset, SAMPLES_PER_CLASS, SEED)
    _, loader = build_loader(dataset, selected_indices, BATCH_SIZE, NUM_WORKERS)

    baseline_model = load_model_from_checkpoint(baseline_checkpoint, device)
    text_model = load_model_from_checkpoint(text_checkpoint, device)

    print("extracting baseline latents...", flush=True)
    baseline_outputs = extract_model_outputs(
        baseline_model,
        baseline_checkpoint.get("fusion_mode", "visual-only"),
        loader,
        device,
    )
    print("extracting text latents...", flush=True)
    text_outputs = extract_model_outputs(
        text_model,
        text_checkpoint.get("fusion_mode", "text-cross-attention"),
        loader,
        device,
    )

    cached = {
        "baseline_latents": normalize_rows(baseline_outputs["latents"]),
        "text_latents": normalize_rows(text_outputs["latents"]),
        "labels": baseline_outputs["labels"],
        "baseline_predictions": baseline_outputs["predictions"],
        "text_predictions": text_outputs["predictions"],
        "baseline_confidences": baseline_outputs["confidences"],
        "text_confidences": text_outputs["confidences"],
        "selected_indices": np.array(selected_indices, dtype=np.int64),
    }

    save_latent_cache(
        cache_path=CACHE_PATH,
        baseline_latents=cached["baseline_latents"],
        text_latents=cached["text_latents"],
        labels=cached["labels"],
        baseline_predictions=cached["baseline_predictions"],
        text_predictions=cached["text_predictions"],
        baseline_confidences=cached["baseline_confidences"],
        text_confidences=cached["text_confidences"],
        selected_indices=selected_indices,
    )
    return cached, class_names


def main() -> None:
    cached, class_names = ensure_cache()
    baseline_coords = run_tsne(cached["baseline_latents"], seed=SEED)
    text_coords = run_tsne(cached["text_latents"], seed=SEED)
    plot_tsne_comparison(
        baseline_coords,
        text_coords,
        cached["labels"],
        class_names,
        OUTPUT_PATH,
    )
    print(f"Saved t-SNE comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
