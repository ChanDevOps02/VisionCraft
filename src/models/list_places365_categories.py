from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import Places365


def parse_args():
    parser = argparse.ArgumentParser(description="List or filter Places365 categories.")
    parser.add_argument("--root", type=str, default="data/places365", help="Dataset root directory.")
    parser.add_argument("--small", action="store_true", help="Use the small metadata layout.")
    parser.add_argument(
        "--keyword",
        nargs="*",
        default=[],
        help="Optional keywords used to filter category names.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = Places365(root=Path(args.root), split="train-standard", small=args.small, download=False)
    categories = dataset.classes

    if args.keyword:
        lowered_keywords = [keyword.lower() for keyword in args.keyword]
        categories = [
            category
            for category in categories
            if any(keyword in category.lower() for keyword in lowered_keywords)
        ]

    for category in categories:
        print(category)

    print(f"\nTotal categories shown: {len(categories)}")


if __name__ == "__main__":
    main()
