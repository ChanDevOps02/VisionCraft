from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import plot_umap_comparison, run_umap
from src.models.infonce_viz_common import OUTPUT_DIR, load_metadata, require_cache


def main() -> None:
    cached = require_cache()
    metadata = load_metadata()

    baseline_coords = run_umap(cached["baseline_latents"], seed=42)
    text_coords = run_umap(cached["text_latents"], seed=42)
    output_path = OUTPUT_DIR / "baseline_vs_text_umap.png"
    plot_umap_comparison(
        baseline_coords,
        text_coords,
        cached["labels"],
        metadata["class_names"],
        output_path,
    )
    print(f"Saved UMAP comparison to {output_path}")


if __name__ == "__main__":
    main()
