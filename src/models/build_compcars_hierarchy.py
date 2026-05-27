from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.compcars_config import (
    CompCarsHierarchyConfig,
    NON_CHINA_BRAND_ALLOWLIST,
    normalize_brand_name,
)


@dataclass(frozen=True)
class CarRecord:
    image_path: Path
    make: str
    model: str
    year: str
    split: str
    relative_key: str


def parse_args():
    parser = argparse.ArgumentParser(description="Build a non-China CompCars hierarchy subset.")
    parser.add_argument("--compcars-root", type=str, required=True, help="Root directory of downloaded CompCars.")
    parser.add_argument("--output-root", type=str, required=True, help="Output directory for train/val subsets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-make-samples", type=int, default=200)
    parser.add_argument("--min-model-samples", type=int, default=80)
    parser.add_argument("--min-models-per-make", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--task",
        choices=["make", "model", "hierarchy"],
        default="hierarchy",
        help="make: build make-only subset, model: build model-only subset, hierarchy: build both.",
    )
    parser.add_argument(
        "--use-official-split",
        action="store_true",
        help="Use CompCars official classification train/test split files.",
    )
    parser.add_argument(
        "--crop-with-bbox",
        action="store_true",
        help="Crop each training image with the official CompCars bounding box from label/*.txt.",
    )
    return parser.parse_args()


def _load_name_list(array) -> dict[int, str]:
    def _unwrap(value) -> str:
        if isinstance(value, str):
            return value.strip()
        if hasattr(value, "dtype") and getattr(value, "size", 0) == 1:
            try:
                return _unwrap(value.item())
            except Exception:
                pass
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list) and len(converted) == 1:
                return _unwrap(converted[0])
            if isinstance(converted, list):
                flat_parts: list[str] = []
                for item in converted:
                    text = _unwrap(item)
                    if text:
                        flat_parts.append(text)
                return " ".join(part for part in flat_parts if part).strip()
        return str(value).strip()

    mapping: dict[int, str] = {}
    for index, item in enumerate(array, start=1):
        value = item[0]
        mapping[index] = _unwrap(value)
    return mapping


def load_make_model_mappings(compcars_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    mat_path = compcars_root / "misc" / "make_model_name.mat"
    metadata = loadmat(mat_path)
    make_names = _load_name_list(metadata["make_names"])
    model_names = _load_name_list(metadata["model_names"])
    return make_names, model_names


def load_official_split(compcars_root: Path, split_name: str) -> list[str]:
    split_file = compcars_root / "train_test_split" / "classification" / f"{split_name}.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_record_from_relative_key(
    relative_key: str,
    image_root: Path,
    make_map: dict[int, str],
    model_map: dict[int, str],
    split_name: str,
) -> CarRecord | None:
    relative_parts = Path(relative_key).parts
    if len(relative_parts) != 4:
        return None

    make_id = int(relative_parts[0])
    model_id = int(relative_parts[1])
    year = relative_parts[2].strip()
    image_name = relative_parts[3]

    make = normalize_brand_name(make_map.get(make_id, f"make_{make_id}"))
    if make not in NON_CHINA_BRAND_ALLOWLIST:
        return None

    model = model_map.get(model_id, f"model_{model_id}").strip()
    image_path = image_root / relative_key
    if not image_path.exists():
        return None

    return CarRecord(
        image_path=image_path,
        make=make,
        model=model,
        year=year,
        split=split_name,
        relative_key=relative_key,
    )


def collect_records(compcars_root: Path, config: CompCarsHierarchyConfig, use_official_split: bool) -> list[CarRecord]:
    make_map, model_map = load_make_model_mappings(compcars_root)
    image_root = compcars_root / "image"
    records: list[CarRecord] = []
    if use_official_split:
        for split_name in ("train", "test"):
            for relative_key in load_official_split(compcars_root, split_name):
                record = _parse_record_from_relative_key(relative_key, image_root, make_map, model_map, split_name)
                if record is not None:
                    records.append(record)
    else:
        for image_path in image_root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in config.image_extensions:
                continue
            relative_key = str(image_path.relative_to(image_root))
            record = _parse_record_from_relative_key(relative_key, image_root, make_map, model_map, "train")
            if record is not None:
                records.append(record)
    return records


def filter_records(records: list[CarRecord], config: CompCarsHierarchyConfig) -> list[CarRecord]:
    make_counts = Counter(record.make for record in records)
    model_counts = Counter((record.make, record.model) for record in records)
    models_per_make = defaultdict(set)
    for record in records:
        models_per_make[record.make].add(record.model)

    filtered: list[CarRecord] = []
    for record in records:
        if make_counts[record.make] < config.min_make_samples:
            continue
        if model_counts[(record.make, record.model)] < config.min_model_samples:
            continue
        if len(models_per_make[record.make]) < config.min_models_per_make:
            continue
        filtered.append(record)
    return filtered


def build_split(records: list[CarRecord], use_official_split: bool, train_ratio: float) -> dict[str, list[CarRecord]]:
    train_records: list[CarRecord] = []
    val_records: list[CarRecord] = []
    if use_official_split:
        for record in records:
            if record.split == "train":
                train_records.append(record)
            else:
                val_records.append(record)
    else:
        grouped = defaultdict(list)
        for record in records:
            grouped[(record.make, record.model)].append(record)
        for _, group in grouped.items():
            split_idx = max(1, min(len(group) - 1, int(round(len(group) * train_ratio))))
            train_records.extend(group[:split_idx])
            val_records.extend(group[split_idx:])

    return {"train": train_records, "val": val_records}


def _safe_label(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        dst.write_bytes(src.read_bytes())


def _read_bbox(label_root: Path, relative_key: str) -> tuple[int, int, int, int] | None:
    label_path = label_root / Path(relative_key).with_suffix(".txt")
    if not label_path.exists():
        return None
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    parts = lines[2].split()
    if len(parts) != 4:
        return None
    x1, y1, x2, y2 = map(int, parts)
    return x1, y1, x2, y2


def _prepare_image(record: CarRecord, compcars_root: Path, crop_with_bbox: bool) -> Path:
    if not crop_with_bbox:
        return record.image_path

    bbox = _read_bbox(compcars_root / "label", record.relative_key)
    if bbox is None:
        return record.image_path

    image = cv2.imread(str(record.image_path))
    if image is None:
        return record.image_path

    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return record.image_path

    cropped = image[y1:y2, x1:x2]
    cropped_root = compcars_root / "_cropped_cache"
    cropped_path = cropped_root / record.relative_key
    cropped_path.parent.mkdir(parents=True, exist_ok=True)
    if not cropped_path.exists():
        cv2.imwrite(str(cropped_path), cropped)
    return cropped_path


def write_make_subset(output_root: Path, split_map: dict[str, list[CarRecord]], compcars_root: Path, crop_with_bbox: bool):
    for split_name, records in split_map.items():
        for index, record in enumerate(records):
            label_dir = output_root / "make" / split_name / _safe_label(record.make)
            source_image = _prepare_image(record, compcars_root, crop_with_bbox)
            dst = label_dir / f"{index:06d}{source_image.suffix.lower()}"
            _link_or_copy(source_image, dst)


def write_model_subset(output_root: Path, split_map: dict[str, list[CarRecord]], compcars_root: Path, crop_with_bbox: bool):
    for split_name, records in split_map.items():
        for index, record in enumerate(records):
            label = f"{record.make}__{record.model}"
            label_dir = output_root / "model" / split_name / _safe_label(label)
            source_image = _prepare_image(record, compcars_root, crop_with_bbox)
            dst = label_dir / f"{index:06d}{source_image.suffix.lower()}"
            _link_or_copy(source_image, dst)


def build_metadata(records: list[CarRecord], filtered_records: list[CarRecord], split_map: dict[str, list[CarRecord]]):
    make_counts = Counter(record.make for record in filtered_records)
    model_counts = Counter(f"{record.make}__{record.model}" for record in filtered_records)
    make_to_models = defaultdict(set)
    for record in filtered_records:
        make_to_models[record.make].add(record.model)

    return {
        "raw_record_count": len(records),
        "filtered_record_count": len(filtered_records),
        "num_makes": len(make_counts),
        "num_models": len(model_counts),
        "make_counts": dict(sorted(make_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "make_to_models": {make: sorted(models) for make, models in sorted(make_to_models.items())},
        "train_count": len(split_map["train"]),
        "val_count": len(split_map["val"]),
    }


def main():
    args = parse_args()
    config = CompCarsHierarchyConfig(
        min_make_samples=args.min_make_samples,
        min_model_samples=args.min_model_samples,
        min_models_per_make=args.min_models_per_make,
        train_ratio=args.train_ratio,
    )

    compcars_root = Path(args.compcars_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning CompCars root: {compcars_root}")
    records = collect_records(compcars_root, config, args.use_official_split)
    print(f"Collected {len(records)} non-China candidate records before filtering.")

    filtered_records = filter_records(records, config)
    print(f"Kept {len(filtered_records)} records after sample-count filtering.")

    split_map = build_split(filtered_records, args.use_official_split, config.train_ratio)
    print(f"Train records: {len(split_map['train'])}, Val records: {len(split_map['val'])}")

    if args.task in {"make", "hierarchy"}:
        write_make_subset(output_root, split_map, compcars_root, args.crop_with_bbox)
        print(f"Wrote make subset under {output_root / 'make'}")

    if args.task in {"model", "hierarchy"}:
        write_model_subset(output_root, split_map, compcars_root, args.crop_with_bbox)
        print(f"Wrote model subset under {output_root / 'model'}")

    metadata = build_metadata(records, filtered_records, split_map)
    metadata["config"] = {
        "min_make_samples": config.min_make_samples,
        "min_model_samples": config.min_model_samples,
        "min_models_per_make": config.min_models_per_make,
        "train_ratio": config.train_ratio,
        "seed": args.seed,
        "task": args.task,
        "use_official_split": args.use_official_split,
        "crop_with_bbox": args.crop_with_bbox,
    }
    metadata_path = output_root / "compcars_hierarchy_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
