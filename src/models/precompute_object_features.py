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

from src.models.object_features import COCO80_CLASSES, extract_object_feature_vector


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute YOLO object feature vectors for a VisionCraft subset.")
    parser.add_argument("--data-root", type=str, required=True, help="ImageFolder-style dataset root.")
    parser.add_argument("--output-path", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for quick experiments. 0 means no cap.")
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
    features: list[np.ndarray] = []
    detection_counts: list[int] = []

    print(f"Precomputing YOLO object features from: {data_root}")
    print(f"Split: {args.split}")
    print(f"Images to process: {len(image_paths)}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = load_rgb_image(image_path)
        result = extract_object_feature_vector(image)

        relative_paths.append(str(image_path.relative_to(data_root)))
        statuses.append(result["status"])
        features.append(result["feature_vector"])
        detection_counts.append(len(result["detections"]))

        if idx % 200 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed")

    feature_matrix = np.stack(features, axis=0) if features else np.zeros((0, len(COCO80_CLASSES) * 3), dtype=np.float32)

    np.savez_compressed(
        output_path,
        paths=np.array(relative_paths, dtype=object),
        statuses=np.array(statuses, dtype=object),
        detection_counts=np.array(detection_counts, dtype=np.int32),
        features=feature_matrix,
        class_names=np.array(COCO80_CLASSES, dtype=object),
    )

    metadata = {
        "data_root": str(data_root),
        "split": args.split,
        "num_images": len(relative_paths),
        "feature_dim": int(feature_matrix.shape[1]) if feature_matrix.size > 0 else len(COCO80_CLASSES) * 3,
        "layout": {
            "counts": len(COCO80_CLASSES),
            "max_confidence": len(COCO80_CLASSES),
            "total_area_ratio": len(COCO80_CLASSES),
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved feature matrix to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
