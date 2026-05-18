from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_ratio: float
    thirds_distance: float


def _try_load_yolo():
    try:
        from ultralytics import YOLO

        return YOLO("yolov8n.pt")
    except Exception:
        return None


_YOLO_MODEL = None


def _get_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = _try_load_yolo()
    return _YOLO_MODEL


def _thirds_distance(center_x: float, center_y: float, width: int, height: int) -> float:
    thirds_points = [
        (width / 3.0, height / 3.0),
        (2.0 * width / 3.0, height / 3.0),
        (width / 3.0, 2.0 * height / 3.0),
        (2.0 * width / 3.0, 2.0 * height / 3.0),
    ]
    distances = [
        np.hypot(center_x - point_x, center_y - point_y) / max(width, height)
        for point_x, point_y in thirds_points
    ]
    return float(min(distances))


def _annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 120), 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def detect_objects(image: np.ndarray) -> dict[str, Any]:
    model = _get_model()
    if model is None:
        return {
            "detections": [],
            "annotated_image": image.copy(),
            "status": "unavailable",
            "summary": "Ultralytics YOLO가 설치되지 않았거나 가중치를 불러오지 못했습니다.",
            "composition_basis": "edge structure fallback",
        }

    try:
        results = model.predict(image, verbose=False, conf=0.35)
    except Exception as exc:
        return {
            "detections": [],
            "annotated_image": image.copy(),
            "status": "failed",
            "summary": f"YOLO 추론 중 오류가 발생했습니다: {exc}",
            "composition_basis": "edge structure fallback",
        }

    height, width = image.shape[:2]
    detections: list[Detection] = []

    if results:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            names = result.names
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label_index = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
                detections.append(
                    Detection(
                        label=names[label_index],
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        area_ratio=float(area_ratio),
                        thirds_distance=_thirds_distance(center_x, center_y, width, height),
                    )
                )

    detections = sorted(detections, key=lambda item: item.confidence, reverse=True)
    annotated_image = _annotate_image(image, detections)

    return {
        "detections": [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "bbox": detection.bbox,
                "area_ratio": round(detection.area_ratio, 4),
                "thirds_distance": round(detection.thirds_distance, 4),
            }
            for detection in detections
        ],
        "annotated_image": annotated_image,
        "status": "ok",
        "summary": f"{len(detections)} objects detected by YOLO." if detections else "YOLO ran successfully but found no objects.",
        "composition_basis": "object-centered thirds analysis" if detections else "edge structure fallback",
    }
