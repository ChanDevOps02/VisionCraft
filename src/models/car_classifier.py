from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.train_car_classifier import build_model, build_transforms
from src.models.train_scene_classifier import get_device


DEFAULT_CAR_MAKE_CHECKPOINT = Path("checkpoint/car_make_classifier_resnet50.pt")
DEFAULT_CAR_MODEL_CHECKPOINT_DIR = Path("checkpoint/car_model_classifiers")

_MAKE_MODEL = None
_MAKE_METADATA: dict[str, Any] | None = None
_MAKE_DEVICE = None
_MODEL_CACHE: dict[str, tuple[torch.nn.Module, dict[str, Any], torch.device]] = {}


def _load_checkpoint_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(num_classes=len(checkpoint["classes"]), backbone=checkpoint["backbone"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model.to(device)
    model.eval()
    return model, checkpoint, device


def _load_make_model():
    global _MAKE_MODEL, _MAKE_METADATA, _MAKE_DEVICE
    if _MAKE_MODEL is not None:
        return _MAKE_MODEL, _MAKE_METADATA, _MAKE_DEVICE
    if not DEFAULT_CAR_MAKE_CHECKPOINT.exists():
        return None, None, None
    _MAKE_MODEL, _MAKE_METADATA, _MAKE_DEVICE = _load_checkpoint_model(DEFAULT_CAR_MAKE_CHECKPOINT)
    return _MAKE_MODEL, _MAKE_METADATA, _MAKE_DEVICE


def _load_brand_model(make_name: str):
    normalized = make_name.replace(" ", "_").replace("-", "_").lower()
    if normalized in _MODEL_CACHE:
        return _MODEL_CACHE[normalized]

    checkpoint_path = DEFAULT_CAR_MODEL_CHECKPOINT_DIR / f"{normalized}.pt"
    if not checkpoint_path.exists():
        return None, None, None

    model_bundle = _load_checkpoint_model(checkpoint_path)
    _MODEL_CACHE[normalized] = model_bundle
    return model_bundle


def _predict_from_model(image: np.ndarray, model, metadata: dict[str, Any], device: torch.device):
    _, eval_transform = build_transforms(metadata["image_size"])
    pil_image = Image.fromarray(image.astype(np.uint8))
    image_tensor = eval_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probabilities))
    return {
        "label": metadata["classes"][pred_idx],
        "confidence": float(probabilities[pred_idx]),
        "top3": [
            {
                "label": metadata["classes"][idx],
                "confidence": float(probabilities[idx]),
            }
            for idx in np.argsort(probabilities)[::-1][:3]
        ],
    }


def classify_car_make(image: np.ndarray) -> dict[str, Any]:
    model, metadata, device = _load_make_model()
    if model is None or metadata is None or device is None:
        return {
            "status": "unavailable",
            "label": "unknown",
            "confidence": 0.0,
            "top3": [],
            "reason": "Car make checkpoint가 없어 차량 브랜드 분류를 수행하지 않았습니다.",
        }

    prediction = _predict_from_model(image, model, metadata, device)
    prediction["status"] = "ok"
    prediction["reason"] = "학습된 CompCars make classifier 예측 결과입니다."
    return prediction


def classify_car_model(image: np.ndarray, make_name: str) -> dict[str, Any]:
    model, metadata, device = _load_brand_model(make_name)
    if model is None or metadata is None or device is None:
        return {
            "status": "unavailable",
            "label": "unknown",
            "confidence": 0.0,
            "top3": [],
            "reason": f"{make_name} 전용 model classifier checkpoint가 없어 모델 분류를 수행하지 않았습니다.",
        }

    prediction = _predict_from_model(image, model, metadata, device)
    prediction["status"] = "ok"
    prediction["reason"] = f"학습된 {make_name} 전용 CompCars model classifier 예측 결과입니다."
    return prediction
