from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _clamp_crop(origin: int, crop_size: int, image_size: int) -> int:
    return max(0, min(origin, image_size - crop_size))


def _snap_crop_to_include_bbox(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox

    if x1 < crop_x:
        crop_x = x1
    if x2 > crop_x + crop_w:
        crop_x = x2 - crop_w
    if y1 < crop_y:
        crop_y = y1
    if y2 > crop_y + crop_h:
        crop_y = y2 - crop_h

    crop_x = _clamp_crop(crop_x, crop_w, width)
    crop_y = _clamp_crop(crop_y, crop_h, height)
    return crop_x, crop_y


def _movement_description(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    image_shape: tuple[int, int],
) -> str:
    height, width = image_shape
    dx = (crop_x + crop_w / 2.0) - (width / 2.0)
    dy = (crop_y + crop_h / 2.0) - (height / 2.0)

    horizontal = ""
    vertical = ""

    if abs(dx) > width * 0.04:
        horizontal = "왼쪽" if dx < 0 else "오른쪽"
    if abs(dy) > height * 0.04:
        vertical = "위쪽" if dy < 0 else "아래쪽"

    if horizontal and vertical:
        return f"{vertical} {horizontal} 방향으로"
    if horizontal:
        return f"{horizontal} 방향으로"
    if vertical:
        return f"{vertical} 방향으로"
    return "중앙 기준으로"


def _build_crop_result(
    image: np.ndarray,
    crop_box: tuple[int, int, int, int],
    summary: str,
    main_label: str,
) -> dict[str, Any]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    annotated = image.copy()
    cv2.rectangle(annotated, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 180, 40), 3)
    cv2.putText(
        annotated,
        "recommended crop",
        (crop_x1, max(22, crop_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 220, 120),
        2,
        cv2.LINE_AA,
    )

    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return {
        "status": "ok",
        "summary": summary,
        "annotated_image": annotated,
        "cropped_image": cropped_image,
        "crop_box": crop_box,
        "main_label": main_label,
    }


def _suggest_crop_from_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> dict[str, Any]:
    height, width = image.shape[:2]
    main_object = max(detections, key=lambda item: item["area_ratio"])
    x1, y1, x2, y2 = main_object["bbox"]
    object_cx = (x1 + x2) / 2.0
    object_cy = (y1 + y2) / 2.0
    object_w = max(1, x2 - x1)
    object_h = max(1, y2 - y1)

    thirds_points = [
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ]

    best_ratio_x, best_ratio_y = min(
        thirds_points,
        key=lambda point: np.hypot(object_cx - point[0] * width, object_cy - point[1] * height),
    )

    crop_scale = 0.88 if main_object["thirds_distance"] > 0.18 else 0.94
    min_crop_w = int(object_w * 1.9)
    min_crop_h = int(object_h * 1.9)

    crop_w = max(min_crop_w, int(width * crop_scale))
    crop_h = max(min_crop_h, int(height * crop_scale))

    crop_w = min(width, crop_w)
    crop_h = min(height, crop_h)

    crop_x = int(round(object_cx - best_ratio_x * crop_w))
    crop_y = int(round(object_cy - best_ratio_y * crop_h))
    crop_x = _clamp_crop(crop_x, crop_w, width)
    crop_y = _clamp_crop(crop_y, crop_h, height)
    crop_x, crop_y = _snap_crop_to_include_bbox(crop_x, crop_y, crop_w, crop_h, main_object["bbox"], (height, width))

    crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
    crop_ratio = (crop_w * crop_h) / float(width * height)

    if crop_ratio >= 0.92:
        crop_strength = "가벼운"
    elif crop_ratio >= 0.80:
        crop_strength = "중간 정도의"
    else:
        crop_strength = "비교적 강한"

    movement = _movement_description(crop_x, crop_y, crop_w, crop_h, (height, width))
    summary = (
        f"주 피사체 `{main_object['label']}`를 rule-of-thirds 기준으로 더 안정적으로 보이게 하기 위해 "
        f"{movement} {crop_strength} 크롭을 추천했습니다 "
        f"({crop_w}x{crop_h})."
    )

    return _build_crop_result(image, crop_box, summary, main_object["label"])


def _suggest_crop_from_segmentation(
    image: np.ndarray,
    segmentation_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not segmentation_result:
        return None

    class_stats = segmentation_result.get("class_stats") or []
    if not class_stats:
        return None

    preferred_labels = {
        "sky",
        "sea",
        "water",
        "river",
        "mountain",
        "hill",
        "tree",
        "plant",
        "earth",
        "road",
        "building",
        "rock",
        "sand",
    }
    selected = [item for item in class_stats if item["label"] in preferred_labels][:4]
    if not selected:
        selected = class_stats[:3]

    if not selected:
        return None

    merged_mask = np.any(np.stack([item["mask"] for item in selected], axis=0), axis=0)
    ys, xs = np.where(merged_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    height, width = image.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    content_w = max(1, x2 - x1)
    content_h = max(1, y2 - y1)
    crop_w = min(width, max(int(width * 0.88), int(content_w * 1.12)))
    crop_h = min(height, max(int(height * 0.88), int(content_h * 1.12)))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x = _clamp_crop(int(round(cx - crop_w / 2.0)), crop_w, width)
    crop_y = _clamp_crop(int(round(cy - crop_h / 2.0)), crop_h, height)
    crop_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)

    labels_preview = ", ".join(item["label"] for item in selected[:3])
    summary = (
        f"객체 검출이 비어 있어 scene parsing 결과(`{labels_preview}`)를 중심으로 "
        f"가벼운 장면 크롭을 추천했습니다 ({crop_w}x{crop_h})."
    )

    return _build_crop_result(image, crop_box, summary, selected[0]["label"])


def suggest_crop(
    image: np.ndarray,
    detection_result: dict[str, Any],
    segmentation_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detections = detection_result.get("detections", [])
    if detections:
        return _suggest_crop_from_detections(image, detections)

    segmentation_based = _suggest_crop_from_segmentation(image, segmentation_result)
    if segmentation_based is not None:
        return segmentation_based

    return {
        "status": "unavailable",
        "summary": "객체와 장면 구성 단서를 충분히 찾지 못해 추천 크롭을 계산하지 않았습니다.",
        "annotated_image": image.copy(),
        "cropped_image": image.copy(),
        "crop_box": None,
    }
