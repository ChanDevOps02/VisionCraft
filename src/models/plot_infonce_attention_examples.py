from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_latent_comparison import choose_attention_examples, plot_attention_examples
from src.models.infonce_viz_common import DEFAULT_PAIRS, OUTPUT_DIR, load_eval_dataset, load_metadata, load_text_model, require_cache
from src.models.train_scene_classifier import get_device


def main() -> None:
    cached = require_cache()
    metadata = load_metadata()
    text_model = load_text_model()
    device = get_device()
    dataset = load_eval_dataset(metadata["text_checkpoint"]["image_size"])

    selected = choose_attention_examples(
        labels=cached["labels"],
        predictions=cached["text_predictions"],
        confidences=cached["text_confidences"],
        class_names=metadata["class_names"],
        pairs=DEFAULT_PAIRS,
        attention_samples=6,
    )
    output_path = OUTPUT_DIR / "text_attention_examples.png"
    plot_attention_examples(
        model=text_model,
        dataset=dataset,
        selected_indices=selected,
        subset_indices=cached["selected_indices"],
        class_names=metadata["class_names"],
        image_size=metadata["text_checkpoint"]["image_size"],
        output_path=output_path,
        device=device,
    )
    print(f"Saved attention examples to {output_path}")


if __name__ == "__main__":
    main()
