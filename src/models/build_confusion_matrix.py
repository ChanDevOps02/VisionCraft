from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate_scene_classifier import evaluate


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "visioncraft_subset_small_v11"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_infonce_e20_rerun.pt"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "logs" / "eval_resnet50_v11_text_crossattn_infonce_e20_rerun_report.txt"
)
DEFAULT_FIGURE_PATH = (
    PROJECT_ROOT / "logs" / "eval_resnet50_v11_text_crossattn_infonce_e20_rerun_confusion.png"
)
DEFAULT_TEXT_EMBEDDINGS = PROJECT_ROOT / "data" / "scene_text_embeddings_clip_sentence_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix PNG and text report for a VisionCraft scene-classifier checkpoint."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--object-features-path", type=Path, default=Path(""))
    parser.add_argument("--object-tokens-path", type=Path, default=Path(""))
    parser.add_argument("--segmentation-features-path", type=Path, default=Path(""))
    parser.add_argument("--scene-text-embeddings-path", type=Path, default=DEFAULT_TEXT_EMBEDDINGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating confusion matrix with settings:")
    print(f"- data_root: {args.data_root}")
    print(f"- checkpoint: {args.checkpoint}")
    print(f"- split: {args.split}")
    print(f"- report_path: {args.report_path}")
    print(f"- figure_path: {args.figure_path}")

    evaluate(args)


if __name__ == "__main__":
    main()
