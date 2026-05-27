from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import (
    build_dataset,
    load_checkpoint,
    load_latent_cache,
    load_model_from_checkpoint,
    normalize_rows,
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


def require_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Latent cache not found: {CACHE_PATH}\n"
            "먼저 build_infonce_latent_cache.py 를 실행하세요."
        )

    cached = load_latent_cache(CACHE_PATH)
    return {
        "baseline_latents": normalize_rows(cached["baseline_latents"]),
        "text_latents": normalize_rows(cached["text_latents"]),
        "labels": cached["labels"].astype(np.int64),
        "baseline_predictions": cached["baseline_predictions"].astype(np.int64),
        "text_predictions": cached["text_predictions"].astype(np.int64),
        "baseline_confidences": cached["baseline_confidences"].astype(np.float32),
        "text_confidences": cached["text_confidences"].astype(np.float32),
        "selected_indices": cached["selected_indices"].astype(np.int64).tolist(),
    }


def load_metadata() -> dict[str, Any]:
    baseline_checkpoint = load_checkpoint(BASELINE_CHECKPOINT)
    text_checkpoint = load_checkpoint(INFO_NCE_CHECKPOINT)
    class_names = baseline_checkpoint["classes"]
    return {
        "baseline_checkpoint": baseline_checkpoint,
        "text_checkpoint": text_checkpoint,
        "class_names": class_names,
    }


def load_text_model() -> ResNetTextCrossAttentionSceneClassifier:
    device = get_device()
    checkpoint = load_checkpoint(INFO_NCE_CHECKPOINT)
    model = load_model_from_checkpoint(checkpoint, device)
    if not isinstance(model, ResNetTextCrossAttentionSceneClassifier):
        raise TypeError("InfoNCE checkpoint did not load as ResNetTextCrossAttentionSceneClassifier.")
    return model


def load_eval_dataset(image_size: int):
    return build_dataset(DATA_ROOT, "val", image_size)
