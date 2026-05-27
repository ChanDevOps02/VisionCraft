from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

ADE20K_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

_SEGMENTATION_MODEL = None
_SEGMENTATION_PROCESSOR = None
_SEGMENTATION_DEVICE = None
_SEGMENTATION_ERROR: str | None = None
_ID2LABEL: dict[int, str] = {}

_PALETTE = [
    (80, 170, 255),
    (255, 120, 120),
    (120, 220, 120),
    (255, 200, 80),
    (200, 120, 255),
    (120, 255, 220),
]


def _get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned > 0


def _normalize_label(label: str) -> str:
    return label.replace("-", " ").replace("_", " ").strip().lower()


def _load_segmentation_model():
    global _SEGMENTATION_MODEL, _SEGMENTATION_PROCESSOR, _SEGMENTATION_DEVICE, _SEGMENTATION_ERROR, _ID2LABEL

    if _SEGMENTATION_MODEL is not None or _SEGMENTATION_ERROR is not None:
        return _SEGMENTATION_MODEL, _SEGMENTATION_PROCESSOR, _SEGMENTATION_DEVICE, _SEGMENTATION_ERROR, _ID2LABEL

    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        processor = AutoImageProcessor.from_pretrained(ADE20K_MODEL_NAME)
        model = AutoModelForSemanticSegmentation.from_pretrained(ADE20K_MODEL_NAME)
        device = _get_device()
        model.to(device)
        model.eval()

        _SEGMENTATION_MODEL = model
        _SEGMENTATION_PROCESSOR = processor
        _SEGMENTATION_DEVICE = device
        _ID2LABEL = {int(key): value for key, value in model.config.id2label.items()}
    except Exception as exc:
        _SEGMENTATION_ERROR = str(exc)

    return _SEGMENTATION_MODEL, _SEGMENTATION_PROCESSOR, _SEGMENTATION_DEVICE, _SEGMENTATION_ERROR, _ID2LABEL


def _person_mask_from_yolo(detection_result: dict[str, Any] | None, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)
    if detection_result is None:
        return mask

    for detection in detection_result.get("detections", []):
        if detection.get("label") != "person":
            continue
        x1, y1, x2, y2 = detection["bbox"]
        mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True

    return _clean_mask(mask, kernel_size=7) if np.any(mask) else mask


def _extract_semantic_map(image: np.ndarray) -> tuple[np.ndarray | None, dict[int, str], str | None]:
    model, processor, device, error, id2label = _load_segmentation_model()
    if model is None or processor is None or device is None:
        return None, id2label, error

    try:
        inputs = processor(images=Image.fromarray(image.astype(np.uint8)), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        semantic_map = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.shape[:2]],
        )[0].cpu().numpy()
        return semantic_map.astype(np.int32), id2label, None
    except Exception as exc:
        return None, id2label, str(exc)


def _collect_class_stats(semantic_map: np.ndarray, id2label: dict[int, str]) -> list[dict[str, Any]]:
    total_pixels = float(semantic_map.size)
    stats: list[dict[str, Any]] = []
    for class_id in np.unique(semantic_map):
        class_id_int = int(class_id)
        mask = semantic_map == class_id_int
        ratio = float(mask.mean())
        label = _normalize_label(id2label.get(class_id_int, f"class_{class_id_int}"))
        stats.append(
            {
                "id": class_id_int,
                "label": label,
                "mask": mask,
                "pixel_count": int(mask.sum()),
                "ratio": ratio,
                "percent": round(ratio * 100.0, 1),
            }
        )
    stats.sort(key=lambda item: item["pixel_count"], reverse=True)
    return stats


def _match_labels(stats: list[dict[str, Any]], candidates: tuple[str, ...]) -> np.ndarray | None:
    masks = [item["mask"] for item in stats if item["label"] in candidates]
    if not masks:
        return None
    merged = np.any(np.stack(masks, axis=0), axis=0)
    return _clean_mask(merged, kernel_size=5) if np.any(merged) else merged


def _build_segmentation_overlay(
    image: np.ndarray,
    class_stats: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    top_classes = [item for item in class_stats if item["label"] != "background"][:6]

    for index, item in enumerate(reversed(top_classes)):
        color = np.array(_PALETTE[index % len(_PALETTE)], dtype=np.float32)
        overlay[item["mask"]] = overlay[item["mask"]] * 0.5 + color * 0.5

    result = np.clip(overlay, 0, 255).astype(np.uint8)
    y = 18
    for index, item in enumerate(top_classes):
        color = _PALETTE[index % len(_PALETTE)]
        cv2.rectangle(result, (10, y - 12), (28, y + 4), color, -1)
        cv2.putText(
            result,
            f"{item['label']} {item['percent']:.1f}%",
            (36, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return result


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(
        labeled,
        text,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _masked_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(image)
    masked[mask] = image[mask]
    return masked


def _build_component_visualization(image: np.ndarray, class_stats: list[dict[str, Any]]) -> np.ndarray:
    top_classes = [item for item in class_stats if item["label"] != "background"][:3]
    if not top_classes:
        return _label_panel(image.copy(), "no semantic components")

    panels: list[np.ndarray] = []
    for item in top_classes:
        panels.append(_label_panel(_masked_region(image, item["mask"]), f"{item['label']} only"))
    return cv2.hconcat(panels)


def segment_regions(image: np.ndarray, detection_result: dict[str, Any] | None = None) -> dict[str, Any]:
    height, width = image.shape[:2]
    semantic_map, id2label, error = _extract_semantic_map(image)

    source = "segformer-ade20k"
    status = "ok" if semantic_map is not None else "fallback"

    if semantic_map is not None:
        class_stats = _collect_class_stats(semantic_map, id2label)
        person_mask = _match_labels(class_stats, ("person",))
        sky_mask = _match_labels(class_stats, ("sky",))
    else:
        class_stats = []
        person_mask = None
        sky_mask = None

    if person_mask is None or not np.any(person_mask):
        person_mask = _person_mask_from_yolo(detection_result, (height, width))
        if np.any(person_mask) and source == "segformer-ade20k":
            source = "segformer-ade20k + yolo-person-fallback"

    if sky_mask is None:
        sky_mask = np.zeros((height, width), dtype=bool)

    background_mask = ~(person_mask | sky_mask)
    background_mask = background_mask.astype(bool)

    if class_stats:
        summary_parts = [f"{item['label']} {item['percent']:.1f}%" for item in class_stats[:6]]
        ratios = {item["label"]: round(item["ratio"], 4) for item in class_stats[:10]}
    else:
        person_ratio = float(person_mask.mean())
        sky_ratio = float(sky_mask.mean())
        background_ratio = float(background_mask.mean())
        summary_parts = []
        if person_ratio > 0:
            summary_parts.append(f"person {person_ratio * 100:.1f}%")
        if sky_ratio > 0:
            summary_parts.append(f"sky {sky_ratio * 100:.1f}%")
        summary_parts.append(f"background {background_ratio * 100:.1f}%")
        ratios = {
            "person": round(person_ratio, 4),
            "sky": round(sky_ratio, 4),
            "background": round(background_ratio, 4),
        }

    if error:
        summary_parts.append(f"scene parsing fallback reason: {error}")

    if class_stats:
        overlay_image = _build_segmentation_overlay(image, class_stats)
        component_visualization = _build_component_visualization(image, class_stats)
    else:
        fallback_stats = [
            {"label": "person", "mask": person_mask, "percent": round(float(person_mask.mean()) * 100.0, 1)},
            {"label": "sky", "mask": sky_mask, "percent": round(float(sky_mask.mean()) * 100.0, 1)},
            {"label": "background", "mask": background_mask, "percent": round(float(background_mask.mean()) * 100.0, 1)},
        ]
        overlay_image = _build_segmentation_overlay(image, fallback_stats)
        component_visualization = _build_component_visualization(image, fallback_stats)

    return {
        "status": status,
        "source": source,
        "person_mask": person_mask,
        "sky_mask": sky_mask,
        "background_mask": background_mask,
        "semantic_map": semantic_map,
        "class_stats": class_stats,
        "overlay_image": overlay_image,
        "component_visualization": component_visualization,
        "summary": ", ".join(summary_parts),
        "ratios": ratios,
    }
