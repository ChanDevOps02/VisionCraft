from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import Places365


def ensure_split(root: Path, split: str, small: bool) -> None:
    print(f"Preparing split={split}, small={small}, root={root}")
    dataset = Places365(root=root, split=split, small=small, download=True)
    print(f"Completed split={split}: {len(dataset)} samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Download Places365 for VisionCraft.")
    parser.add_argument("--root", type=str, default="data/places365", help="Dataset root directory.")
    parser.add_argument("--small", action="store_true", help="Download the 256x256 small version.")
    parser.add_argument("--include-val", action="store_true", help="Also download the validation split.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    ensure_split(root=root, split="train-standard", small=args.small)

    if args.include_val:
        ensure_split(root=root, split="val", small=args.small)


if __name__ == "__main__":
    main()
