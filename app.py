from __future__ import annotations

import gradio as gr
import numpy as np

from src.analyzer.blur import calculate_blur_score
from src.analyzer.brightness import calculate_brightness_score
from src.analyzer.contrast import calculate_contrast_score
from src.analyzer.edge_density import calculate_edge_density
from src.analyzer.histogram import build_histogram_figure
from src.analyzer.quality import generate_feedback, summarize_scores
from src.enhancer.traditional_enhance import enhance_image
from src.models.object_detector import detect_objects
from src.models.scene_classifier import classify_scene
from src.utils.visualization import build_metrics_markdown


def process_image(image: np.ndarray):
    if image is None:
        raise gr.Error("이미지를 업로드해주세요.")

    brightness = calculate_brightness_score(image)
    contrast = calculate_contrast_score(image)
    blur = calculate_blur_score(image)
    edge_density = calculate_edge_density(image)
    detection_result = detect_objects(image)
    scene_result = classify_scene(image, detection_result=detection_result)

    metrics = summarize_scores(
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        edge_density=edge_density,
        scene_result=scene_result,
        detection_result=detection_result,
    )

    enhanced = enhance_image(image, metrics)
    feedback = generate_feedback(metrics)
    histogram_fig = build_histogram_figure(image)
    metrics_md = build_metrics_markdown(metrics, feedback)

    return (
        metrics_md,
        histogram_fig,
        detection_result["annotated_image"],
        enhanced,
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
        input_image = gr.Image(label="Input Image", type="numpy")
        detection_image = gr.Image(label="Detection View", type="numpy")
        output_image = gr.Image(label="Enhanced Image", type="numpy")

    analyze_btn = gr.Button("Analyze and Enhance", variant="primary")

    with gr.Row():
        metrics_output = gr.Markdown(label="Analysis Report")
        histogram_output = gr.Plot(label="Color Histogram")

    analyze_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[metrics_output, histogram_output, detection_image, output_image],
    )


if __name__ == "__main__":
    demo.launch()
