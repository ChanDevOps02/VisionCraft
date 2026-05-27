from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import plot_similarity_boxplot, sample_pairwise_cosine_similarity
from src.models.infonce_rerun_viz_common import OUTPUT_DIR, require_cache


def main() -> None:
    cached = require_cache()
    baseline_pairwise = sample_pairwise_cosine_similarity(cached["baseline_latents"], cached["labels"], seed=42)
    text_pairwise = sample_pairwise_cosine_similarity(cached["text_latents"], cached["labels"], seed=43)
    output_path = OUTPUT_DIR / "intra_inter_class_cosine_boxplot.png"
    plot_similarity_boxplot(baseline_pairwise, text_pairwise, output_path)
    print(f"Saved similarity boxplot to {output_path}")


if __name__ == "__main__":
    main()
