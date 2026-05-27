from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import parse_pairs, plot_pairwise_umap
from src.models.infonce_viz_common import DEFAULT_PAIRS, OUTPUT_DIR, load_metadata, require_cache


def main() -> None:
    cached = require_cache()
    metadata = load_metadata()
    output_path = OUTPUT_DIR / "confusion_pair_umap_comparison.png"
    plot_pairwise_umap(
        baseline_latents=cached["baseline_latents"],
        text_latents=cached["text_latents"],
        labels=cached["labels"],
        class_names=metadata["class_names"],
        pairs=parse_pairs([f"{left}:{right}" for left, right in DEFAULT_PAIRS]),
        output_path=output_path,
        seed=42,
    )
    print(f"Saved pairwise UMAP to {output_path}")


if __name__ == "__main__":
    main()
