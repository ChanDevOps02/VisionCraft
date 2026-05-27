from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import compute_centroid_distance_matrix, plot_centroid_heatmaps
from src.models.infonce_rerun_viz_common import OUTPUT_DIR, load_metadata, require_cache


def main() -> None:
    cached = require_cache()
    metadata = load_metadata()
    baseline_matrix = compute_centroid_distance_matrix(
        cached["baseline_latents"],
        cached["labels"],
        len(metadata["class_names"]),
    )
    text_matrix = compute_centroid_distance_matrix(
        cached["text_latents"],
        cached["labels"],
        len(metadata["class_names"]),
    )
    output_path = OUTPUT_DIR / "centroid_cosine_distance_heatmaps.png"
    plot_centroid_heatmaps(baseline_matrix, text_matrix, metadata["class_names"], output_path)
    print(f"Saved centroid heatmaps to {output_path}")


if __name__ == "__main__":
    main()
