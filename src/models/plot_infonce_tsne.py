from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.infonce_viz_common import OUTPUT_DIR, load_metadata, require_cache


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
    output_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    titles = ["Visual-Only Baseline t-SNE", "Text Cross-Attention + InfoNCE t-SNE"]
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


def main() -> None:
    cached = require_cache()
    metadata = load_metadata()
    baseline_coords = run_tsne(cached["baseline_latents"], seed=42)
    text_coords = run_tsne(cached["text_latents"], seed=42)
    output_path = OUTPUT_DIR / "baseline_vs_text_tsne.png"
    plot_tsne_comparison(
        baseline_coords,
        text_coords,
        cached["labels"],
        metadata["class_names"],
        output_path,
    )
    print(f"Saved t-SNE comparison to {output_path}")


if __name__ == "__main__":
    main()
