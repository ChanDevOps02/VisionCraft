from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.object_features import COCO80_CLASSES, COCO80_INDEX
from src.models.object_detector import detect_objects


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute YOLO object tokens for cross-attention-based VisionCraft experiments."
    )
    parser.add_argument("--data-root", type=str, required=True, help="ImageFolder-style dataset root.")
    parser.add_argument("--output-path", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for quick experiments. 0 means no cap.")
    parser.add_argument("--max-objects", type=int, default=16, help="Maximum number of object tokens to keep per image.")
    return parser.parse_args()


def list_image_paths(data_root: Path, split: str) -> list[Path]:
    splits = ["train", "val"] if split == "all" else [split]
    image_paths: list[Path] = []
    for split_name in splits:
        split_root = data_root / split_name
        if not split_root.exists():
            continue
        for path in split_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                image_paths.append(path)
    return sorted(image_paths)


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detections_to_padded_arrays(
    detections: list[dict],
    image_width: int,
    image_height: int,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert YOLO detections to padded token arrays.

    geometry layout:
    - confidence
    - center_x / width
    - center_y / height
    - bbox_width / width
    - bbox_height / height
    - area_ratio
    - thirds_distance
    """

    class_ids = np.zeros((max_objects,), dtype=np.int64)
    geometry = np.zeros((max_objects, 7), dtype=np.float32)
    valid_mask = np.zeros((max_objects,), dtype=bool)

    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)

    for idx, detection in enumerate(detections[:max_objects]):
        label = detection.get("label", "")
        if label not in COCO80_INDEX:
            continue

        x1, y1, x2, y2 = detection.get("bbox", (0, 0, 0, 0))
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        bbox_width = max((x2 - x1) / width, 0.0)
        bbox_height = max((y2 - y1) / height, 0.0)

        class_ids[idx] = COCO80_INDEX[label]
        geometry[idx] = np.array(
            [
                float(detection.get("confidence", 0.0)),
                float(center_x),
                float(center_y),
                float(bbox_width),
                float(bbox_height),
                float(detection.get("area_ratio", 0.0)),
                float(detection.get("thirds_distance", 0.0)),
            ],
            dtype=np.float32,
        )
        valid_mask[idx] = True

    return class_ids, geometry, valid_mask


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(data_root, args.split)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    relative_paths: list[str] = []
    statuses: list[str] = []
    class_id_rows: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    valid_mask_rows: list[np.ndarray] = []
    detection_counts: list[int] = []

    print(f"Precomputing YOLO object tokens from: {data_root}")
    print(f"Split: {args.split}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Max objects per image: {args.max_objects}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = load_rgb_image(image_path)
        height, width = image.shape[:2]
        detection_result = detect_objects(image)
        detections = detection_result["detections"]

        class_ids, geometry, valid_mask = detections_to_padded_arrays(
            detections,
            image_width=width,
            image_height=height,
            max_objects=args.max_objects,
        )

        relative_paths.append(str(image_path.relative_to(data_root)))
        statuses.append(detection_result["status"])
        class_id_rows.append(class_ids)
        geometry_rows.append(geometry)
        valid_mask_rows.append(valid_mask)
        detection_counts.append(len(detections))

        if idx % 200 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed")

    class_ids_array = (
        np.stack(class_id_rows, axis=0) if class_id_rows else np.zeros((0, args.max_objects), dtype=np.int64)
    )
    geometry_array = (
        np.stack(geometry_rows, axis=0)
        if geometry_rows
        else np.zeros((0, args.max_objects, 7), dtype=np.float32)
    )
    valid_mask_array = (
        np.stack(valid_mask_rows, axis=0) if valid_mask_rows else np.zeros((0, args.max_objects), dtype=bool)
    )

    np.savez_compressed(
        output_path,
        paths=np.array(relative_paths, dtype=object),
        statuses=np.array(statuses, dtype=object),
        detection_counts=np.array(detection_counts, dtype=np.int32),
        class_ids=class_ids_array,
        geometry=geometry_array,
        valid_mask=valid_mask_array,
        class_names=np.array(COCO80_CLASSES, dtype=object),
    )

    metadata = {
        "data_root": str(data_root),
        "split": args.split,
        "num_images": len(relative_paths),
        "max_objects": args.max_objects,
        "geometry_layout": [
            "confidence",
            "center_x_norm",
            "center_y_norm",
            "bbox_width_norm",
            "bbox_height_norm",
            "area_ratio",
            "thirds_distance",
        ],
        "num_object_classes": len(COCO80_CLASSES),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved object tokens to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
