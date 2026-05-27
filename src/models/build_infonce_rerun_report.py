from __future__ import annotations

import json
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import build_metrics_summary, normalize_rows, sample_pairwise_cosine_similarity
from src.models.infonce_rerun_viz_common import (
    BASELINE_CHECKPOINT,
    CACHE_PATH,
    DEFAULT_PAIRS,
    INFO_NCE_CHECKPOINT,
    OUTPUT_DIR,
    load_text_model,
    require_cache,
)


def main() -> None:
    cached = require_cache()
    baseline_pairwise = sample_pairwise_cosine_similarity(cached["baseline_latents"], cached["labels"], seed=42)
    text_pairwise = sample_pairwise_cosine_similarity(cached["text_latents"], cached["labels"], seed=43)
    text_model = load_text_model()

    projected_text_tokens = text_model.get_projected_text_tokens().detach().cpu().numpy()
    projected_text_tokens = normalize_rows(projected_text_tokens)
    cosine_matrix = cached["text_latents"] @ projected_text_tokens.T
    labels = cached["labels"]
    correct_cosines = cosine_matrix[np.arange(len(labels)), labels]
    rival_cosines = cosine_matrix.copy()
    rival_cosines[np.arange(len(labels)), labels] = -np.inf
    best_rival = rival_cosines.max(axis=1)

    report = {
        "data_root": "data/visioncraft_subset_small_v11",
        "split": "val",
        "samples_per_class": 180,
        "num_samples": int(len(labels)),
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "text_checkpoint": str(INFO_NCE_CHECKPOINT),
        "cache_file": str(CACHE_PATH),
        "pairs": [f"{left}:{right}" for left, right in DEFAULT_PAIRS],
        "baseline_metrics": build_metrics_summary(cached["baseline_latents"], labels, baseline_pairwise),
        "text_metrics": build_metrics_summary(cached["text_latents"], labels, text_pairwise),
        "text_prototype_metrics": {
            "mean_correct_class_cosine": float(correct_cosines.mean()),
            "std_correct_class_cosine": float(correct_cosines.std()),
            "mean_correct_vs_rival_margin": float((correct_cosines - best_rival).mean()),
            "prototype_retrieval_accuracy": float((cosine_matrix.argmax(axis=1) == labels).mean()),
        },
        "outputs": {
            "umap": str(OUTPUT_DIR / "baseline_vs_text_umap.png"),
            "tsne": str(OUTPUT_DIR / "baseline_vs_text_tsne.png"),
            "centroid_heatmaps": str(OUTPUT_DIR / "centroid_cosine_distance_heatmaps.png"),
            "intra_inter_boxplot": str(OUTPUT_DIR / "intra_inter_class_cosine_boxplot.png"),
            "text_prototype_histograms": str(OUTPUT_DIR / "text_prototype_cosine_histograms.png"),
            "pairwise_umap": str(OUTPUT_DIR / "confusion_pair_umap_comparison.png"),
            "attention_examples": str(OUTPUT_DIR / "text_attention_examples.png"),
        },
    }
    report_path = OUTPUT_DIR / "latent_comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
