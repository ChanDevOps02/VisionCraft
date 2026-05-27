from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(
        labeled,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def build_difference_heatmap(source_image: np.ndarray, enhanced_image: np.ndarray) -> dict[str, Any]:
    abs_diff = cv2.absdiff(source_image, enhanced_image)
    diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(source_image, 0.55, heatmap_rgb, 0.45, 0)
    mean_change = float(diff_gray.mean())
    max_change = int(diff_gray.max())

    if mean_change < 6:
        intensity = "subtle"
    elif mean_change < 14:
        intensity = "moderate"
    else:
        intensity = "strong"

    visualization = cv2.hconcat(
        [
            _label_panel(source_image, "original"),
            _label_panel(overlay, "difference heatmap"),
        ]
    )

    summary = (
        f"원본 대비 보정 변화 강도는 {intensity} 수준이며 "
        f"(mean change: {mean_change:.1f}, max change: {max_change})."
    )

    return {
        "status": "ok",
        "summary": summary,
        "mean_change": round(mean_change, 2),
        "max_change": max_change,
        "change_intensity": intensity,
        "visualization_image": visualization,
    }


def build_segmentation_guided_enhancement_map(
    image: np.ndarray,
    segmentation_result: dict[str, Any] | None,
    enhancement_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not segmentation_result:
        return {
            "status": "unavailable",
            "summary": "세그멘테이션 결과가 없어 영역별 보정 맵을 생성하지 않았습니다.",
            "visualization_image": _label_panel(image.copy(), "enhancement map unavailable"),
        }

    person_mask = segmentation_result.get("person_mask")
    sky_mask = segmentation_result.get("sky_mask")
    background_mask = segmentation_result.get("background_mask")
    if person_mask is None or sky_mask is None or background_mask is None:
        return {
            "status": "unavailable",
            "summary": "영역 마스크가 충분하지 않아 보정 맵을 생성하지 않았습니다.",
            "visualization_image": _label_panel(image.copy(), "enhancement map unavailable"),
        }

    applied = set((enhancement_report or {}).get("applied_steps", []))
    overlay = image.copy().astype(np.float32)
    legend: list[tuple[str, tuple[int, int, int]]] = []

    if "background-aware denoise" in applied and np.any(background_mask):
        color = np.array([80, 220, 120], dtype=np.float32)
        overlay[background_mask] = overlay[background_mask] * 0.55 + color * 0.45
        legend.append(("background denoise", (80, 220, 120)))

    if "sky-aware color boost" in applied and np.any(sky_mask):
        color = np.array([90, 170, 255], dtype=np.float32)
        overlay[sky_mask] = overlay[sky_mask] * 0.45 + color * 0.55
        legend.append(("sky boost", (90, 170, 255)))

    if "person-preserving blend" in applied and np.any(person_mask):
        color = np.array([255, 110, 110], dtype=np.float32)
        overlay[person_mask] = overlay[person_mask] * 0.4 + color * 0.6
        legend.append(("person preserve", (255, 110, 110)))

    visualization = np.clip(overlay, 0, 255).astype(np.uint8)
    visualization = _label_panel(visualization, "segmentation-guided enhancement map")

    y = 54
    if not legend:
        cv2.putText(
            visualization,
            "no region-specific enhancement applied",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        summary = "영역별 보정이 적용되지 않아 enhancement map 변화가 제한적입니다."
    else:
        for label, color in legend:
            cv2.rectangle(visualization, (12, y - 15), (36, y + 4), color, -1)
            cv2.putText(
                visualization,
                label,
                (46, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
        summary = "영역별 보정 맵에서 sky / person / background에 적용된 후처리를 시각화했습니다."

    return {
        "status": "ok",
        "summary": summary,
        "visualization_image": visualization,
    }


def build_crop_preview(image: np.ndarray, crop_result: dict[str, Any]) -> dict[str, Any]:
    annotated = crop_result.get("annotated_image", image.copy())
    cropped = crop_result.get("cropped_image", image.copy())

    preview_height = image.shape[0]
    preview_width = image.shape[1]
    resized_crop = cv2.resize(cropped, (preview_width, preview_height), interpolation=cv2.INTER_LINEAR)

    visualization = cv2.hconcat(
        [
            _label_panel(annotated, "recommended crop area"),
            _label_panel(resized_crop, "cropped preview"),
        ]
    )

    return {
        "status": crop_result.get("status", "unavailable"),
        "summary": crop_result.get("summary", ""),
        "visualization_image": visualization,
    }
