from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "visioncraft_subset_small_v11" / "val"
VANILLA_CACHE = PROJECT_ROOT / "logs" / "latent_comparison_v11_full180" / "latent_cache.npz"
INFONCE_CACHE = PROJECT_ROOT / "logs" / "latent_comparison_v11_infonce_rerun_full180" / "latent_cache.npz"
CHECKPOINT = PROJECT_ROOT / "checkpoint" / "scene_classifier_resnet50_v11_text_crossattn_infonce_A.pt"
OUTPUT_PATH = PROJECT_ROOT / "logs" / "scene_confusion_case_gallery.png"

SEMANTIC_NEIGHBOR_PAIRS = {
    (4, 10),  # kitchen_dining, restaurant_cafe
    (10, 4),
    (13, 5),  # waterfront, mountain_valley
    (5, 13),
    (8, 1),   # public_large_indoor, corridor_lobby
    (1, 8),
}


def _load_classes() -> list[str]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    return checkpoint["classes"]


def _load_caches():
    vanilla = np.load(VANILLA_CACHE, allow_pickle=True)
    infonce = np.load(INFONCE_CACHE, allow_pickle=True)
    if not np.array_equal(vanilla["labels"], infonce["labels"]):
        raise ValueError("Vanilla and InfoNCE caches use different labels.")
    if not np.array_equal(vanilla["selected_indices"], infonce["selected_indices"]):
        raise ValueError("Vanilla and InfoNCE caches use different selected_indices.")
    return vanilla, infonce


def _select_cases(vanilla, infonce) -> list[tuple[str, int]]:
    labels = vanilla["labels"].astype(np.int64)
    baseline_preds = vanilla["baseline_predictions"].astype(np.int64)
    vanilla_preds = vanilla["text_predictions"].astype(np.int64)
    infonce_preds = infonce["text_predictions"].astype(np.int64)
    infonce_conf = infonce["text_confidences"].astype(np.float32)

    corrected = np.where((baseline_preds != labels) & (infonce_preds == labels))[0]
    corrected = corrected[np.argsort(infonce_conf[corrected])[::-1]]

    all_wrong = np.where((baseline_preds != labels) & (vanilla_preds != labels) & (infonce_preds != labels))[0]
    all_wrong = all_wrong[np.argsort(infonce_conf[all_wrong])[::-1]]

    semantic_neighbor = np.array(
        [
            idx
            for idx, (label, bp, vp, ip) in enumerate(zip(labels, baseline_preds, vanilla_preds, infonce_preds, strict=True))
            if (label, bp) in SEMANTIC_NEIGHBOR_PAIRS or (label, vp) in SEMANTIC_NEIGHBOR_PAIRS or (label, ip) in SEMANTIC_NEIGHBOR_PAIRS
        ],
        dtype=np.int64,
    )
    semantic_neighbor = semantic_neighbor[np.argsort(infonce_conf[semantic_neighbor])[::-1]]

    picks: list[tuple[str, int]] = []
    if len(corrected):
        picks.append(("Baseline wrong -> InfoNCE correct", int(corrected[0])))
    if len(all_wrong):
        picks.append(("All three still confused", int(all_wrong[0])))
    if len(semantic_neighbor):
        semantic_idx = next((int(idx) for idx in semantic_neighbor if int(idx) not in {case_idx for _, case_idx in picks}), int(semantic_neighbor[0]))
        picks.append(("Representative semantic-neighbor case", semantic_idx))
    return picks


def _title(model_name: str, pred_idx: int, label_idx: int, conf: float, classes: list[str]) -> str:
    mark = "CORRECT" if pred_idx == label_idx else "WRONG"
    return f"{model_name}\n{classes[pred_idx]} ({conf:.3f})\n{mark}"


def main() -> None:
    classes = _load_classes()
    vanilla, infonce = _load_caches()
    cases = _select_cases(vanilla, infonce)
    dataset = datasets.ImageFolder(DATA_ROOT)
    selected_indices = vanilla["selected_indices"].astype(np.int64)
    labels = vanilla["labels"].astype(np.int64)
    baseline_preds = vanilla["baseline_predictions"].astype(np.int64)
    baseline_conf = vanilla["baseline_confidences"].astype(np.float32)
    vanilla_preds = vanilla["text_predictions"].astype(np.int64)
    vanilla_conf = vanilla["text_confidences"].astype(np.float32)
    infonce_preds = infonce["text_predictions"].astype(np.int64)
    infonce_conf = infonce["text_confidences"].astype(np.float32)

    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 5 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for row, (case_title, subset_idx) in enumerate(cases):
        dataset_idx = int(selected_indices[subset_idx])
        image_path, label = dataset.samples[dataset_idx]
        image = np.array(Image.open(image_path).convert("RGB"))

        row_axes = axes[row]
        row_axes[0].imshow(image)
        row_axes[0].set_title(f"{case_title}\nOriginal / true={classes[label]}", fontsize=12)
        row_axes[1].imshow(image)
        row_axes[1].set_title(_title("Baseline", int(baseline_preds[subset_idx]), int(label), float(baseline_conf[subset_idx]), classes), fontsize=11)
        row_axes[2].imshow(image)
        row_axes[2].set_title(_title("Vanilla Text", int(vanilla_preds[subset_idx]), int(label), float(vanilla_conf[subset_idx]), classes), fontsize=11)
        row_axes[3].imshow(image)
        row_axes[3].set_title(_title("Text + InfoNCE", int(infonce_preds[subset_idx]), int(label), float(infonce_conf[subset_idx]), classes), fontsize=11)

        for ax in row_axes:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scene confusion case gallery to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
