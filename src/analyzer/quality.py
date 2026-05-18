from __future__ import annotations

from typing import Any


def _calculate_composition_score(detection_result: dict[str, Any], edge_density: float) -> float:
    detections = detection_result.get("detections", [])
    if not detections:
        return round(min(100.0, max(0.0, 40.0 + edge_density * 2.0)), 2)

    main_object = max(detections, key=lambda item: item["area_ratio"])
    third_distance = main_object["thirds_distance"]

    thirds_score = max(0.0, 100.0 - third_distance * 180.0)
    size_bonus = min(20.0, main_object["area_ratio"] * 120.0)
    return round(min(100.0, thirds_score + size_bonus), 2)


def summarize_scores(
    brightness: float,
    contrast: float,
    blur: float,
    edge_density: float,
    scene_result: dict[str, Any],
    detection_result: dict[str, Any],
) -> dict[str, Any]:
    composition = _calculate_composition_score(detection_result, edge_density)

    overall = round(
        min(
            100.0,
            max(
                0.0,
                0.24 * brightness
                + 0.22 * contrast
                + 0.20 * blur
                + 0.14 * edge_density
                + 0.20 * composition,
            ),
        ),
        2,
    )

    detections = detection_result.get("detections", [])
    object_names = [item["label"] for item in detections]
    main_subject = max(detections, key=lambda item: item["area_ratio"])["label"] if detections else "not detected"

    return {
        "scene": scene_result["label"],
        "scene_reason": scene_result["reason"],
        "brightness": brightness,
        "contrast": contrast,
        "blur": blur,
        "edge_density": edge_density,
        "composition": composition,
        "overall": overall,
        "detected_objects": object_names,
        "detection_count": len(detections),
        "main_subject": main_subject,
        "detection_status": detection_result.get("status", "unknown"),
        "detection_summary": detection_result.get("summary", ""),
        "composition_basis": detection_result.get("composition_basis", "edge structure"),
    }


def generate_feedback(metrics: dict[str, Any]) -> str:
    comments: list[str] = []

    if metrics["brightness"] < 45:
        comments.append("이미지가 전체적으로 어두워 밝기 보정이 필요합니다.")
    elif metrics["brightness"] > 75:
        comments.append("이미지가 다소 밝아 하이라이트가 날아갈 수 있습니다.")

    if metrics["contrast"] < 45:
        comments.append("대비가 낮아 피사체 구분이 약하므로 contrast enhancement가 유효합니다.")

    if metrics["blur"] < 40:
        comments.append("선명도가 낮아 sharpening 적용이 도움이 됩니다.")

    if metrics["edge_density"] < 8:
        comments.append("에지 밀도가 낮아 장면 구조가 다소 평면적으로 보일 수 있습니다.")

    if metrics["composition"] < 50:
        comments.append("구도 점수가 낮아 피사체 배치와 시선 유도가 약할 수 있습니다.")
    elif metrics["composition"] > 75:
        comments.append("주요 피사체 배치가 비교적 안정적입니다.")

    if metrics["detection_count"] > 0:
        object_preview = ", ".join(metrics["detected_objects"][:4])
        comments.append(
            f"객체 검출 기준 주요 피사체는 {metrics['main_subject']}이며, 감지된 객체는 {object_preview}입니다."
        )
    else:
        comments.append(
            "현재 환경에서 YOLO 객체가 감지되지 않았습니다. 기본 품질 지표 중심으로 분석을 진행했습니다."
        )

    comments.append(f"추정 장면 유형은 {metrics['scene']}입니다.")
    return " ".join(comments)
