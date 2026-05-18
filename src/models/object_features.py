from __future__ import annotations

from typing import Any

import numpy as np

from src.models.object_detector import detect_objects


# COCO 80-class order used by YOLOv8 default models.
COCO80_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO80_INDEX = {label: idx for idx, label in enumerate(COCO80_CLASSES)}


def build_object_feature_vector(
    detections: list[dict[str, Any]],
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Encode YOLO detections into a fixed-size vector.

    The feature layout is:
    - object counts per class (80)
    - max confidence per class (80)
    - summed area ratio per class (80)
    """

    if class_names is None:
        class_names = COCO80_CLASSES

    feature_dim = len(class_names)
    counts = np.zeros(feature_dim, dtype=np.float32)
    max_conf = np.zeros(feature_dim, dtype=np.float32)
    total_area = np.zeros(feature_dim, dtype=np.float32)

    for detection in detections:
        label = detection["label"]
        if label not in COCO80_INDEX:
            continue
        idx = COCO80_INDEX[label]
        counts[idx] += 1.0
        max_conf[idx] = max(max_conf[idx], float(detection.get("confidence", 0.0)))
        total_area[idx] += float(detection.get("area_ratio", 0.0))

    return np.concatenate([counts, max_conf, total_area], axis=0)


def extract_object_feature_vector(image: np.ndarray) -> dict[str, Any]:
    detection_result = detect_objects(image)
    detections = detection_result["detections"]
    feature_vector = build_object_feature_vector(detections)

    return {
        "feature_vector": feature_vector,
        "detections": detections,
        "status": detection_result["status"],
        "summary": detection_result["summary"],
        "feature_dim": int(feature_vector.shape[0]),
    }
