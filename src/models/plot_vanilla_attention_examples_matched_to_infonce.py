from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import (
    choose_attention_examples,
    load_checkpoint,
    load_latent_cache,
    load_model_from_checkpoint,
    plot_attention_examples,
    build_dataset,
)
from src.models.infonce_rerun_viz_common import (
    CACHE_PATH as INFONCE_CACHE_PATH,
    DEFAULT_PAIRS,
    load_metadata as load_infonce_metadata,
)
from src.models.text_cross_attention import ResNetTextCrossAttentionSceneClassifier
from src.models.train_scene_classifier import get_device


VANILLA_OUTPUT_DIR = PROJECT_ROOT / "logs" / "latent_comparison_v11_full180"
VANILLA_CACHE_PATH = VANILLA_OUTPUT_DIR / "latent_cache.npz"
VANILLA_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_e20.pt"
OUTPUT_PATH = VANILLA_OUTPUT_DIR / "text_attention_examples_matched_to_infonce.png"


def main() -> None:
    if not VANILLA_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Vanilla latent cache not found: {VANILLA_CACHE_PATH}\n"
            "먼저 analyze_latent_comparison.py 로 vanilla cache를 생성하세요."
        )
    if not INFONCE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"InfoNCE latent cache not found: {INFONCE_CACHE_PATH}\n"
            "먼저 build_infonce_rerun_latent_cache.py 를 실행하세요."
        )

    vanilla_cache = load_latent_cache(VANILLA_CACHE_PATH)
    infonce_cache = load_latent_cache(INFONCE_CACHE_PATH)

    vanilla_selected = vanilla_cache["selected_indices"].astype(np.int64)
    infonce_selected = infonce_cache["selected_indices"].astype(np.int64)
    vanilla_labels = vanilla_cache["labels"].astype(np.int64)
    infonce_labels = infonce_cache["labels"].astype(np.int64)

    if not np.array_equal(vanilla_selected, infonce_selected):
        raise ValueError("Vanilla cache and InfoNCE cache use different selected_indices.")
    if not np.array_equal(vanilla_labels, infonce_labels):
        raise ValueError("Vanilla cache and InfoNCE cache use different label ordering.")

    metadata = load_infonce_metadata()
    selected = choose_attention_examples(
        labels=infonce_labels,
        predictions=infonce_cache["text_predictions"].astype(np.int64),
        confidences=infonce_cache["text_confidences"].astype(np.float32),
        class_names=metadata["class_names"],
        pairs=DEFAULT_PAIRS,
        attention_samples=6,
    )

    checkpoint = load_checkpoint(VANILLA_CHECKPOINT)
    device = get_device()
    model = load_model_from_checkpoint(checkpoint, device)
    if not isinstance(model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("Vanilla checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")

    dataset = build_dataset(PROJECT_ROOT / "data" / "visioncraft_subset_small_v11", "val", checkpoint["image_size"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_attention_examples(
        model=model,
        dataset=dataset,
        selected_indices=selected,
        subset_indices=vanilla_selected.tolist(),
        class_names=metadata["class_names"],
        image_size=checkpoint["image_size"],
        output_path=OUTPUT_PATH,
        device=device,
    )
    print(f"Saved matched vanilla attention examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
