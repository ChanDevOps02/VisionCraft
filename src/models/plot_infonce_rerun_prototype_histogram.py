from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import normalize_rows, plot_text_prototype_histogram
from src.models.infonce_rerun_viz_common import OUTPUT_DIR, load_text_model, require_cache


def main() -> None:
    cached = require_cache()
    text_model = load_text_model()

    projected_text_tokens = text_model.get_projected_text_tokens().detach().cpu().numpy()
    projected_text_tokens = normalize_rows(projected_text_tokens)
    cosine_matrix = cached["text_latents"] @ projected_text_tokens.T
    labels = cached["labels"]
    correct_cosines = cosine_matrix[np.arange(len(labels)), labels]
    rival_cosines = cosine_matrix.copy()
    rival_cosines[np.arange(len(labels)), labels] = -np.inf
    best_rival = rival_cosines.max(axis=1)

    output_path = OUTPUT_DIR / "text_prototype_cosine_histograms.png"
    plot_text_prototype_histogram(correct_cosines, correct_cosines - best_rival, output_path)
    print(f"Saved prototype histogram to {output_path}")


if __name__ == "__main__":
    main()
