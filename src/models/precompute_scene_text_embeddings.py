from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import CLIPModel, CLIPTokenizer

from src.models.scene_text_prompts import SCENE_CLASS_NAMES, get_scene_text_prompts
from src.models.train_scene_classifier import get_device


DEFAULT_CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"


def extract_text_embedding_tensor(model_output) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "text_embeds") and model_output.text_embeds is not None:
        return model_output.text_embeds
    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output
    if hasattr(model_output, "last_hidden_state") and model_output.last_hidden_state is not None:
        return model_output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported CLIP text output type: {type(model_output)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute scene-class text embeddings for VisionCraft.")
    parser.add_argument("--output-path", type=str, required=True, help="Output .npz path.")
    parser.add_argument(
        "--prompt-style",
        choices=["sentence_v1", "keyword_v1"],
        default="sentence_v1",
        help="Prompt template style for scene text.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=DEFAULT_CLIP_TEXT_MODEL,
        help="CLIP model name used for text encoding.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model only from local cache or a local model directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = get_scene_text_prompts(args.prompt_style)
    ordered_class_names = list(SCENE_CLASS_NAMES)
    ordered_prompts = [prompts[class_name] for class_name in ordered_class_names]

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading CLIP text encoder: {args.clip_model}")

    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.clip_model,
            local_files_only=args.local_files_only,
        )
        model = CLIPModel.from_pretrained(
            args.clip_model,
            local_files_only=args.local_files_only,
        ).to(device)
    except OSError as error:
        raise SystemExit(
            "Failed to load the CLIP text model. "
            "If you are offline, either pass a local model directory via --clip-model "
            "or rerun after the model has been downloaded once.\n"
            f"Original error: {error}"
        ) from error
    model.eval()

    tokenized = tokenizer(
        ordered_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.no_grad():
        text_output = model.get_text_features(**tokenized)
        text_embeddings = extract_text_embedding_tensor(text_output)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    embedding_matrix = text_embeddings.cpu().numpy().astype(np.float32)

    np.savez_compressed(
        output_path,
        class_names=np.array(ordered_class_names, dtype=object),
        prompts=np.array(ordered_prompts, dtype=object),
        embeddings=embedding_matrix,
        prompt_style=np.array([args.prompt_style], dtype=object),
        clip_model=np.array([args.clip_model], dtype=object),
    )

    metadata = {
        "num_classes": len(ordered_class_names),
        "embedding_dim": int(embedding_matrix.shape[1]) if embedding_matrix.ndim == 2 else 0,
        "class_names": ordered_class_names,
        "prompts": ordered_prompts,
        "prompt_style": args.prompt_style,
        "clip_model": args.clip_model,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved scene text embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
