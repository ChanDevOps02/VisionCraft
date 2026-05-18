from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from torchvision.datasets import Places365

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.visioncraft_scene_mapping import get_mapping, build_reverse_mapping

#1. Places365 train/val을 읽음
#2. 각 이미지의 원본 클래스 이름을 확인
#3. visioncraft_scene_mapping.py를 참고해서 상위 클래스 결정
#4. output_root/train/<class_name>/...과 output_root/val/<class_name>/...과 같은 구조로 이미지 연결
#5. 최종적으로 train_scene_classifier.py가 바로 읽을 수 있는 데이터셋 생성

def parse_args():
    parser = argparse.ArgumentParser(description="Build a VisionCraft subset from Places365.")
    parser.add_argument("--places-root", type=str, default="data/places365", help="Root directory of downloaded Places365.") #원본 Places365가 있는 위치
    parser.add_argument("--output-root", type=str, default="data/visioncraft_subset", help="Output directory for ImageFolder-style subset.") #새로 만들 VisionCraft subset위치
    parser.add_argument("--small", action="store_true", help="Use the Places365 small layout.") #Places365 small버전을 쓸지 여부
    parser.add_argument(    #원본 이미지를 실제 복사할지, 아니면 symlink(이미지를 복사하지 않고 원본을 가리키는 링크만 만드는 방식)만 만들지
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize subset samples. symlink saves disk space.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap per target class per split. 0 means no cap.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove an existing output directory before rebuilding.",
    )
    parser.add_argument(
        "--mapping-version",
        choices=["14", "14-refined", "10", "16"],
        default="14",
        help="Which VisionCraft taxonomy preset to use.",
    )
    return parser.parse_args()


def ensure_mapping_is_unique(mapping: dict[str, list[str]], mapping_version: str) -> None:
    reverse_mapping = build_reverse_mapping(mapping_version)
    total_sources = sum(len(source_classes) for source_classes in mapping.values())
    if len(reverse_mapping) != total_sources:
        raise ValueError("Duplicate Places365 source class found in VisionCraft mapping.")


def make_target_path(output_root: Path, split_name: str, target_class: str, source_path: Path) -> Path:
    source_stem = source_path.stem
    suffix = source_path.suffix
    parent_hint = source_path.parent.name.replace("/", "_")
    filename = f"{parent_hint}__{source_stem}{suffix}"
    return output_root / split_name / target_class / filename


def materialize_file(source_path: Path, target_path: Path, link_mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        return

    if link_mode == "symlink":
        target_path.symlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, target_path)


def build_split(
    dataset: Places365,
    split_name: str,
    output_root: Path,
    mapping: dict[str, list[str]],
    reverse_mapping: dict[str, str],
    link_mode: str,
    max_per_class: int,
) -> Counter:
    counts: Counter[str] = Counter()
    skipped: defaultdict[str, int] = defaultdict(int)

    for source_path_str, target_idx in dataset.imgs:
        source_class = dataset.classes[target_idx]
        target_class = reverse_mapping.get(source_class)
        if target_class is None:
            continue

        if max_per_class > 0 and counts[target_class] >= max_per_class:
            skipped[target_class] += 1
            continue

        source_path = Path(source_path_str)
        target_path = make_target_path(output_root, split_name, target_class, source_path)
        materialize_file(source_path, target_path, link_mode)
        counts[target_class] += 1

    print(f"\n[{split_name}] subset counts")
    for class_name in sorted(mapping):
        print(f"- {class_name}: {counts[class_name]}")

    if max_per_class > 0:
        print(f"\n[{split_name}] skipped due to cap")
        for class_name in sorted(skipped):
            print(f"- {class_name}: {skipped[class_name]}")

    return counts


def main():
    args = parse_args()
    mapping = get_mapping(args.mapping_version)
    ensure_mapping_is_unique(mapping, args.mapping_version)

    places_root = Path(args.places_root)
    output_root = Path(args.output_root)
    reverse_mapping = build_reverse_mapping(args.mapping_version)

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    train_dataset = Places365(root=places_root, split="train-standard", small=args.small, download=False)
    val_dataset = Places365(root=places_root, split="val", small=args.small, download=False)

    print(f"Using Places365 root: {places_root}")
    print(f"Writing subset to: {output_root}")
    print(f"Mapping preset: {args.mapping_version}-class")
    print(f"Materialization mode: {args.link_mode}")
    if args.max_per_class > 0:
        print(f"Max per class per split: {args.max_per_class}")

    build_split(
        dataset=train_dataset,
        split_name="train",
        output_root=output_root,
        mapping=mapping,
        reverse_mapping=reverse_mapping,
        link_mode=args.link_mode,
        max_per_class=args.max_per_class,
    )
    build_split(
        dataset=val_dataset,
        split_name="val",
        output_root=output_root,
        mapping=mapping,
        reverse_mapping=reverse_mapping,
        link_mode=args.link_mode,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
