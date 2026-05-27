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

from src.models.segmenter import ADE20K_MODEL_NAME, _load_segmentation_model, segment_regions


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute segmentation composition vectors for VisionCraft.")
    parser.add_argument("--data-root", type=str, required=True, help="ImageFolder-style dataset root.")
    parser.add_argument("--output-path", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for quick experiments.")
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

    _, _, _, error, id2label = _load_segmentation_model()
    if error:
        raise RuntimeError(f"Segmentation model is unavailable: {error}")

    num_classes = len(id2label)
    image_paths = list_image_paths(data_root, args.split)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    relative_paths: list[str] = []
    statuses: list[str] = []
    features: list[np.ndarray] = []
    summaries: list[str] = []

    print(f"Precomputing segmentation composition features from: {data_root}")
    print(f"Split: {args.split}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Segmentation model: {ADE20K_MODEL_NAME}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = load_rgb_image(image_path)
        result = segment_regions(image)

        vector = np.zeros((num_classes,), dtype=np.float32)
        for item in result.get("class_stats", []):
            class_id = int(item["id"])
            if 0 <= class_id < num_classes:
                vector[class_id] = float(item["ratio"])

        relative_paths.append(str(image_path.relative_to(data_root)))
        statuses.append(result.get("status", "unknown"))
        features.append(vector)
        summaries.append(result.get("summary", ""))

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed")

    feature_matrix = np.stack(features, axis=0) if features else np.zeros((0, num_classes), dtype=np.float32)

    np.savez_compressed(
        output_path,
        paths=np.array(relative_paths, dtype=object),
        statuses=np.array(statuses, dtype=object),
        summaries=np.array(summaries, dtype=object),
        features=feature_matrix,
        class_names=np.array([id2label[idx] for idx in range(num_classes)], dtype=object),
        model_name=np.array([ADE20K_MODEL_NAME], dtype=object),
    )

    metadata = {
        "data_root": str(data_root),
        "split": args.split,
        "num_images": len(relative_paths),
        "feature_dim": int(feature_matrix.shape[1]) if feature_matrix.size > 0 else num_classes,
        "model_name": ADE20K_MODEL_NAME,
        "num_classes": num_classes,
        "class_names": [id2label[idx] for idx in range(num_classes)],
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved segmentation feature matrix to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
