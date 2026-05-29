from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.object_features import build_object_feature_vector
from src.models.train_scene_classifier import build_model, build_transforms, get_device


DEFAULT_SCENE_CHECKPOINT = Path("checkpoint/scene_classifier_resnet50_v11_text_crossattn_infonce_A.pt")
BASELINE_SCENE_CHECKPOINT = Path("checkpoint/scene_classifier_resnet50_v11_visual_only_e20.pt")
VANILLA_TEXT_SCENE_CHECKPOINT = Path("checkpoint/scene_classifier_resnet50_v11_text_crossattn_e20.pt")
INFONCE_SCENE_CHECKPOINT = Path("checkpoint/scene_classifier_resnet50_v11_text_crossattn_infonce_A.pt")

_SCENE_MODELS: dict[str, Any] = {}
_SCENE_METADATA_BY_PATH: dict[str, dict[str, Any]] = {}
_SCENE_DEVICES: dict[str, torch.device] = {}


def _heuristic_scene(image: np.ndarray) -> dict[str, str]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mean_brightness = float(np.mean(gray))
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    blue_ratio = float(np.mean(image[:, :, 2] > image[:, :, 0]))
    green_ratio = float(np.mean(image[:, :, 1] > image[:, :, 0]))

    if blue_ratio > 0.55 and mean_brightness > 120:
        return {
            "label": "outdoor sky-dominant",
            "reason": "학습 모델을 불러오지 못해 휴리스틱으로 추정했습니다. 밝은 톤과 높은 청색 비중으로 실외 하늘 장면일 가능성이 높습니다.",
        }

    if green_ratio > 0.52 and mean_saturation > 80:
        return {
            "label": "nature",
            "reason": "학습 모델을 불러오지 못해 휴리스틱으로 추정했습니다. 녹색 계열 비중과 채도가 높아 자연 장면으로 추정됩니다.",
        }

    if mean_brightness < 110:
        return {
            "label": "indoor",
            "reason": "학습 모델을 불러오지 못해 휴리스틱으로 추정했습니다. 전반적인 밝기가 낮고 색 분포가 실내 장면 특성에 가깝습니다.",
        }

    return {
        "label": "urban/outdoor",
        "reason": "학습 모델을 불러오지 못해 휴리스틱으로 추정했습니다. 중간 이상의 밝기와 혼합 색 분포로 일반 실외 또는 도심 장면으로 보입니다.",
    }


def _load_scene_model(checkpoint_path: Path = DEFAULT_SCENE_CHECKPOINT):
    cache_key = str(checkpoint_path)
    if cache_key in _SCENE_MODELS:
        return _SCENE_MODELS[cache_key], _SCENE_METADATA_BY_PATH[cache_key], _SCENE_DEVICES[cache_key]

    if not checkpoint_path.exists():
        return None, None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "resnet18")
    fusion_mode = checkpoint.get("fusion_mode", "visual-only")
    object_feature_dim = checkpoint.get("object_feature_dim", 0)
    object_max_objects = checkpoint.get("object_max_objects", 16)
    segmentation_feature_dim = checkpoint.get("segmentation_feature_dim", 0)
    scene_text_embeddings = checkpoint.get("scene_text_embeddings", None)
    cross_attention_dropout = checkpoint.get("cross_attention_dropout", 0.1)

    model = build_model(
        num_classes=len(checkpoint["classes"]),
        backbone=backbone,
        fusion_mode=fusion_mode,
        object_feature_dim=object_feature_dim,
        object_max_objects=object_max_objects,
        segmentation_feature_dim=segmentation_feature_dim,
        scene_text_embeddings=scene_text_embeddings,
        cross_attention_dropout=cross_attention_dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model.to(device)
    model.eval()

    _SCENE_MODELS[cache_key] = model
    _SCENE_METADATA_BY_PATH[cache_key] = checkpoint
    _SCENE_DEVICES[cache_key] = device
    return _SCENE_MODELS[cache_key], _SCENE_METADATA_BY_PATH[cache_key], _SCENE_DEVICES[cache_key]


def _predict_scene_with_loaded_model(
    model,
    metadata: dict[str, Any],
    device: torch.device,
    image: np.ndarray,
    detection_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_size = metadata["image_size"]
    classes = metadata["classes"]
    fusion_mode = metadata.get("fusion_mode", "visual-only")
    _, eval_transform = build_transforms(image_size)

    pil_image = Image.fromarray(image.astype(np.uint8))
    image_tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    object_features_tensor = None
    if fusion_mode == "late-fusion":
        detections = detection_result["detections"] if detection_result is not None else []
        object_feature_vector = build_object_feature_vector(detections)
        object_features_tensor = torch.from_numpy(object_feature_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        if object_features_tensor is not None:
            logits = model(image_tensor, object_features_tensor)
        else:
            logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probabilities))
    pred_label = classes[pred_idx]
    pred_confidence = float(probabilities[pred_idx])
    topk_indices = np.argsort(probabilities)[::-1][:3]
    topk_parts = [f"{classes[idx]}={probabilities[idx]:.3f}" for idx in topk_indices]
    reason = (
        f"학습된 scene classifier({metadata.get('backbone', 'resnet18')}, "
        f"{metadata.get('fusion_mode', 'visual-only')}) 기준 추정 결과입니다. "
        f"예측 신뢰도는 {pred_confidence:.3f}이며, 상위 후보는 {', '.join(topk_parts)} 입니다."
    )

    return {
        "label": pred_label,
        "reason": reason,
        "confidence": round(pred_confidence, 4),
        "top3": [
            {"label": classes[idx], "confidence": round(float(probabilities[idx]), 4)}
            for idx in topk_indices
        ],
        "source": "learned-checkpoint",
        "checkpoint": str(metadata.get("checkpoint_path", "")),
        "backbone": metadata.get("backbone", "resnet18"),
        "fusion_mode": metadata.get("fusion_mode", "visual-only"),
    }


def classify_scene(
    image: np.ndarray,
    detection_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model, metadata, device = _load_scene_model()
    if model is None or metadata is None or device is None:
        return _heuristic_scene(image)
    return _predict_scene_with_loaded_model(model, metadata, device, image, detection_result=detection_result)


def compare_scene_classifiers(
    image: np.ndarray,
    detection_result: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    configs = [
        ("visual_only_baseline", BASELINE_SCENE_CHECKPOINT),
        ("text_cross_attention", VANILLA_TEXT_SCENE_CHECKPOINT),
        ("text_cross_attention_infonce", INFONCE_SCENE_CHECKPOINT),
    ]
    results: dict[str, dict[str, Any]] = {}
    for key, checkpoint_path in configs:
        model, metadata, device = _load_scene_model(checkpoint_path)
        if model is None or metadata is None or device is None:
            fallback = _heuristic_scene(image)
            fallback.update(
                {
                    "confidence": None,
                    "top3": [],
                    "source": "heuristic-fallback",
                    "backbone": None,
                    "fusion_mode": None,
                    "checkpoint": str(checkpoint_path),
                }
            )
            results[key] = fallback
            continue

        metadata = dict(metadata)
        metadata["checkpoint_path"] = str(checkpoint_path)
        results[key] = _predict_scene_with_loaded_model(
            model,
            metadata,
            device,
            image,
            detection_result=detection_result,
        )
    return results
