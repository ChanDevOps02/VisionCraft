from __future__ import annotations

import html

import gradio as gr
import numpy as np

from src.analyzer.blur import calculate_blur_score
from src.analyzer.brightness import calculate_brightness_score
from src.analyzer.color_balance import analyze_color_balance
from src.analyzer.contrast import calculate_contrast_score
from src.analyzer.crop_suggestion import suggest_crop
from src.analyzer.document_text import (
    build_manual_point_overlay,
    extract_text_information,
    rectify_text_document_from_points,
)
from src.analyzer.edge_density import calculate_edge_density
from src.analyzer.enhancement_visuals import build_crop_preview, build_difference_heatmap
from src.analyzer.exposure import analyze_exposure
from src.analyzer.histogram import build_histogram_figure
from src.analyzer.orb_matching import analyze_orb_feature_matching
from src.analyzer.quality import generate_feedback, summarize_scores
from src.analyzer.tilt_correction import analyze_tilt_and_horizon
from src.enhancer.traditional_enhance import enhance_image
from src.models.object_detector import detect_objects
from src.models.scene_classifier import classify_scene, compare_scene_classifiers
from src.models.segmenter import segment_regions
from src.utils.visualization import build_metrics_markdown


def _apply_recommended_crop_to_image(image: np.ndarray, crop_result: dict) -> np.ndarray:
    crop_box = crop_result.get("crop_box")
    if not crop_box:
        return image

    x1, y1, x2, y2 = crop_box
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2].copy()


def _reset_manual_points(image: np.ndarray):
    if image is None:
        return [], None, None, "이미지를 먼저 업로드해주세요."
    return [], image, image, "좌상단부터 시계 방향으로 4개 점을 클릭하세요."


def _add_manual_point(image: np.ndarray, points: list, evt: gr.SelectData):
    if image is None:
        return points, None, None, "이미지를 먼저 업로드해주세요."

    if evt is None or evt.index is None:
        overlay = build_manual_point_overlay(image, points)
        return points, overlay, image, "점 선택 이벤트를 읽지 못했습니다."

    index = evt.index
    if isinstance(index, tuple) and len(index) >= 2:
        x, y = int(index[0]), int(index[1])
    elif isinstance(index, list) and len(index) >= 2:
        x, y = int(index[0]), int(index[1])
    else:
        overlay = build_manual_point_overlay(image, points)
        return points, overlay, image, "클릭 좌표를 읽지 못했습니다."

    updated = list(points)
    if len(updated) >= 4:
        updated = []
    updated.append([x, y])
    overlay = build_manual_point_overlay(image, updated)
    if len(updated) < 4:
        message = f"{len(updated)}/4 points selected. 좌상단부터 시계 방향으로 계속 클릭하세요."
        rectified = image
    else:
        rectified_result = rectify_text_document_from_points(image, updated)
        message = rectified_result["summary"]
        rectified = rectified_result["rectified_image"]
    return updated, overlay, rectified, message


def _clear_manual_points(image: np.ndarray):
    if image is None:
        return [], None, None, "이미지를 먼저 업로드해주세요."
    return [], image, image, "수동 점 선택을 초기화했습니다."


def _apply_auto_straighten(image: np.ndarray) -> tuple[np.ndarray, dict]:
    tilt_result = analyze_tilt_and_horizon(image)
    angle = float(tilt_result.get("tilt_angle_deg", 0.0))
    state = tilt_result.get("tilt_state", "unknown")
    if tilt_result.get("status") == "ok" and angle >= 0.8 and state in {"slight", "noticeable"}:
        return tilt_result.get("corrected_preview", image), tilt_result
    return image, tilt_result


def _build_scene_comparison_markdown(results: dict[str, dict]) -> str:
    model_titles = {
        "visual_only_baseline": "Visual-Only Baseline",
        "text_cross_attention": "Text Cross-Attention",
        "text_cross_attention_infonce": "Text Cross-Attention + InfoNCE",
    }
    rows = [
        "| Model | Predicted Scene | Confidence | Top-3 | Source |",
        "|---|---|---:|---|---|",
    ]
    for key in ["visual_only_baseline", "text_cross_attention", "text_cross_attention_infonce"]:
        result = results.get(key, {})
        label = result.get("label", "unknown")
        confidence = result.get("confidence")
        confidence_text = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "-"
        top3 = result.get("top3", []) or []
        top3_text = ", ".join(f"{item['label']}={item['confidence']:.3f}" for item in top3) if top3 else "-"
        source = result.get("source", "unknown")
        rows.append(f"| {model_titles[key]} | `{label}` | {confidence_text} | {top3_text} | `{source}` |")

    summary_lines = [
        "## Scene Classifier Comparison",
        "",
        "동일한 입력 이미지에 대해 `baseline`, `vanilla text cross-attention`, `InfoNCE` 세 모델의 scene prediction을 나란히 비교한 결과이다.",
        "",
        *rows,
    ]
    return "\n".join(summary_lines)


def _format_status_badge(status: str) -> str:
    normalized = (status or "unknown").lower()
    color = "#388e3c"
    if normalized in {"fallback", "heuristic-fallback", "manual_required", "disabled", "unavailable"}:
        color = "#f57c00"
    elif normalized in {"failed", "error"}:
        color = "#d32f2f"
    return f"<code style='color:{color};font-weight:700;'>{html.escape(str(status))}</code>"


def _build_model_status_markdown(
    detection_result: dict,
    segmentation_result: dict,
    scene_result: dict,
    scene_comparison_result: dict[str, dict],
    ocr_result: dict,
) -> str:
    comparison_sources = [
        result.get("source", "unknown")
        for result in scene_comparison_result.values()
    ]
    learned_count = sum(source == "learned-checkpoint" for source in comparison_sources)
    comparison_status = f"{learned_count}/{len(comparison_sources)} checkpoints loaded"

    rows = [
        ("YOLO Object Detection", detection_result.get("status", "unknown"), detection_result.get("summary", "")),
        (
            "Scene Classifier",
            scene_result.get("source", "heuristic-fallback"),
            scene_result.get("reason", ""),
        ),
        ("Scene Model Comparison", comparison_status, ", ".join(comparison_sources) or "unknown"),
        (
            "SegFormer Segmentation",
            segmentation_result.get("status", "unknown"),
            f"{segmentation_result.get('source', 'unknown')} | {segmentation_result.get('summary', '')}",
        ),
        (
            "OCR Backend",
            ocr_result.get("status", "unknown"),
            f"{ocr_result.get('engine', 'unknown')} | {ocr_result.get('summary', '')}",
        ),
    ]

    body = "".join(
        "<tr>"
        f"<th style='text-align:left;padding:6px 10px;border-bottom:1px solid #333;width:26%;'>{html.escape(name)}</th>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #333;width:20%;'>{_format_status_badge(status)}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #333;'>{html.escape(str(detail or '-'))}</td>"
        "</tr>"
        for name, status, detail in rows
    )
    return (
        "<h2 style='margin:0 0 8px 0;'>Model Status</h2>"
        "<table style='width:100%;border-collapse:collapse;margin:6px 0 14px 0;font-size:14px;'>"
        f"{body}</table>"
    )


def process_image(
    image: np.ndarray,
    enable_text_processing: bool,
    manual_points: list[list[int]] | None,
    manual_rectified_for_ocr: np.ndarray | None,
):
    if image is None:
        raise gr.Error("이미지를 업로드해주세요.")

    analysis_image, tilt_result = _apply_auto_straighten(image)

    brightness = calculate_brightness_score(analysis_image)
    contrast = calculate_contrast_score(analysis_image)
    color_balance = analyze_color_balance(analysis_image)
    exposure = analyze_exposure(analysis_image)
    blur = calculate_blur_score(analysis_image)
    edge_density = calculate_edge_density(analysis_image)
    detection_result = detect_objects(analysis_image)
    scene_result = classify_scene(analysis_image, detection_result=detection_result)
    scene_comparison_result = compare_scene_classifiers(analysis_image, detection_result=detection_result)
    segmentation_result = segment_regions(analysis_image, detection_result=detection_result)
    crop_result = suggest_crop(analysis_image, detection_result, segmentation_result=segmentation_result)
    manual_points = manual_points or []
    if enable_text_processing and len(manual_points) == 4 and manual_rectified_for_ocr is not None:
        ocr_result = extract_text_information(manual_rectified_for_ocr)
    elif enable_text_processing:
        ocr_result = {
            "status": "manual_required",
            "raw_text": "OCR를 사용하려면 Manual 4-Point Rectification 탭에서 4개 꼭짓점을 먼저 지정해주세요.",
            "korean_interpretation": "",
            "summary": "",
            "engine": "manual_rectification_required",
        }
    else:
        ocr_result = {
            "status": "disabled",
            "raw_text": "",
            "korean_interpretation": "",
            "summary": "",
            "engine": "disabled",
        }

    metrics = summarize_scores(
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        edge_density=edge_density,
        color_balance=color_balance,
        exposure=exposure,
        scene_result=scene_result,
        detection_result=detection_result,
        segmentation_result=segmentation_result,
        crop_result=crop_result,
        ocr_result=ocr_result,
        tilt_result=tilt_result,
    )

    enhanced, enhancement_report = enhance_image(analysis_image, metrics, region_result=segmentation_result)
    final_output = _apply_recommended_crop_to_image(enhanced, crop_result)
    orb_result = analyze_orb_feature_matching(analysis_image, enhanced)
    crop_preview_result = build_crop_preview(analysis_image, crop_result)
    difference_result = build_difference_heatmap(analysis_image, enhanced)
    metrics["applied_enhancements"] = enhancement_report["applied_steps"]
    metrics["enhancement_reasons"] = enhancement_report["reason_steps"]
    metrics["enhancement_summary"] = enhancement_report["summary"]
    metrics["orb_status"] = orb_result["status"]
    metrics["orb_summary"] = orb_result["summary"]
    metrics["orb_match_count"] = orb_result["match_count"]
    metrics["orb_matching_quality"] = orb_result.get("matching_quality", "unknown")
    metrics["difference_status"] = difference_result["status"]
    metrics["difference_summary"] = difference_result["summary"]
    feedback = generate_feedback(metrics)
    histogram_fig = build_histogram_figure(analysis_image, final_output)
    metrics_md = build_metrics_markdown(metrics, feedback)
    model_status_md = _build_model_status_markdown(
        detection_result,
        segmentation_result,
        scene_result,
        scene_comparison_result,
        ocr_result,
    )
    scene_comparison_md = _build_scene_comparison_markdown(scene_comparison_result)

    return (
        model_status_md,
        metrics_md,
        scene_comparison_md,
        histogram_fig,
        tilt_result["visualization_image"],
        detection_result["annotated_image"],
        segmentation_result["overlay_image"],
        segmentation_result["component_visualization"],
        crop_preview_result["visualization_image"],
        difference_result["visualization_image"],
        orb_result["visualization_image"],
        ocr_result["raw_text"],
        ocr_result["korean_interpretation"],
        final_output,
    )


with gr.Blocks(title="VisionCraft") as demo:
    gr.Markdown(
        """
        # VisionCraft
        Scene Understanding 기반 이미지 개선·생성 시스템

        이미지를 업로드하면 장면 유형, 객체, 기본 품질 지표를 분석하고
        OpenCV 기반 자동 보정 결과를 함께 비교할 수 있습니다.
        """
    )

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="numpy", scale=1)
        output_image = gr.Image(label="Enhanced Image", type="numpy", scale=1)

    with gr.Tabs():
        with gr.Tab("Auto Straighten"):
            tilt_visualization_image = gr.Image(label="Tilt / Horizon Correction", type="numpy")
        with gr.Tab("Detection"):
            detection_image = gr.Image(label="Detection View", type="numpy")
        with gr.Tab("Segmentation Overlay"):
            segmentation_image = gr.Image(label="Segmentation View", type="numpy")
        with gr.Tab("Segmentation Components"):
            segmentation_components_image = gr.Image(label="Segmentation Components View", type="numpy")
        with gr.Tab("Auto Crop Preview"):
            crop_preview_image = gr.Image(label="Auto Crop Preview", type="numpy")
        with gr.Tab("Manual 4-Point Rectification"):
            gr.Markdown("문서/간판의 네 꼭짓점을 `좌상단 -> 우상단 -> 우하단 -> 좌하단` 순서로 클릭하세요.")
            manual_points_state = gr.State([])
            manual_status = gr.Markdown("이미지를 먼저 업로드해주세요.")
            with gr.Row():
                manual_rectification_input = gr.Image(label="Click 4 Points", type="numpy")
                manual_rectified_image = gr.Image(label="Manual Homography Result", type="numpy")
            with gr.Row():
                clear_manual_points_btn = gr.Button("Clear Manual Points")
        with gr.Tab("Difference Heatmap"):
            difference_heatmap_image = gr.Image(label="Before/After Difference Heatmap", type="numpy")
        with gr.Tab("ORB Matching"):
            orb_matching_image = gr.Image(label="ORB Matching View", type="numpy")

    analyze_btn = gr.Button("Analyze and Enhance", variant="primary")
    enable_text_processing = gr.Checkbox(
        label="Enable Text Processing (OCR)",
        value=False,
        info="체크한 경우에만 OCR과 한국어 해석을 수행합니다.",
    )
    model_status_output = gr.Markdown(
        "## Model Status\n분석을 실행하면 YOLO, Scene Classifier, SegFormer, OCR backend 상태가 여기에 표시됩니다.",
        label="Model Status",
    )

    with gr.Row():
        metrics_output = gr.Markdown(label="Analysis Report")
        scene_comparison_output = gr.Markdown(label="Scene Model Comparison")
    with gr.Row():
        histogram_output = gr.Plot(label="Original vs Enhanced RGB Histogram")
    with gr.Row():
        ocr_text_output = gr.Textbox(label="Detected Text", lines=8)
        ocr_interpretation_output = gr.Textbox(label="Korean Interpretation", lines=8)

    analyze_btn.click(
        fn=process_image,
        inputs=[input_image, enable_text_processing, manual_points_state, manual_rectified_image],
        outputs=[
            model_status_output,
            metrics_output,
            scene_comparison_output,
            histogram_output,
            tilt_visualization_image,
            detection_image,
            segmentation_image,
            segmentation_components_image,
            crop_preview_image,
            difference_heatmap_image,
            orb_matching_image,
            ocr_text_output,
            ocr_interpretation_output,
            output_image,
        ],
    )

    input_image.change(
        fn=_reset_manual_points,
        inputs=[input_image],
        outputs=[manual_points_state, manual_rectification_input, manual_rectified_image, manual_status],
    )

    manual_rectification_input.select(
        fn=_add_manual_point,
        inputs=[manual_rectification_input, manual_points_state],
        outputs=[manual_points_state, manual_rectification_input, manual_rectified_image, manual_status],
    )

    clear_manual_points_btn.click(
        fn=_clear_manual_points,
        inputs=[input_image],
        outputs=[manual_points_state, manual_rectification_input, manual_rectified_image, manual_status],
    )


if __name__ == "__main__":
    demo.launch()
