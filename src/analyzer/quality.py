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
    color_balance: dict[str, Any],
    exposure: dict[str, Any],
    scene_result: dict[str, Any],
    detection_result: dict[str, Any],
    segmentation_result: dict[str, Any] | None = None,
    crop_result: dict[str, Any] | None = None,
    orb_result: dict[str, Any] | None = None,
    ocr_result: dict[str, Any] | None = None,
    tilt_result: dict[str, Any] | None = None,
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
        "color_cast_score": color_balance["color_cast_score"],
        "white_balance_shift": color_balance["white_balance_shift"],
        "channel_means": color_balance["channel_means"],
        "channel_scale": color_balance["channel_scale"],
        "dominant_cast": color_balance["dominant_cast"],
        "exposure_state": exposure["exposure_state"],
        "shadow_ratio": exposure["shadow_ratio"],
        "highlight_ratio": exposure["highlight_ratio"],
        "dynamic_range_score": exposure["dynamic_range_score"],
        "p5_intensity": exposure["p5_intensity"],
        "p95_intensity": exposure["p95_intensity"],
        "composition": composition,
        "overall": overall,
        "detected_objects": object_names,
        "detection_count": len(detections),
        "main_subject": main_subject,
        "detection_status": detection_result.get("status", "unknown"),
        "detection_summary": detection_result.get("summary", ""),
        "composition_basis": detection_result.get("composition_basis", "edge structure"),
        "segmentation_status": segmentation_result.get("status", "unavailable") if segmentation_result else "unavailable",
        "segmentation_summary": segmentation_result.get("summary", "") if segmentation_result else "",
        "segmentation_source": segmentation_result.get("source", "none") if segmentation_result else "none",
        "tilt_status": tilt_result.get("status", "unavailable") if tilt_result else "unavailable",
        "tilt_summary": tilt_result.get("summary", "") if tilt_result else "",
        "tilt_angle_deg": tilt_result.get("tilt_angle_deg", 0.0) if tilt_result else 0.0,
        "tilt_state": tilt_result.get("tilt_state", "unknown") if tilt_result else "unknown",
        "tilt_correction_direction": tilt_result.get("correction_direction", "none") if tilt_result else "none",
        "crop_status": crop_result.get("status", "unavailable") if crop_result else "unavailable",
        "crop_summary": crop_result.get("summary", "") if crop_result else "",
        "orb_status": orb_result.get("status", "unavailable") if orb_result else "unavailable",
        "orb_summary": orb_result.get("summary", "") if orb_result else "",
        "orb_match_count": orb_result.get("match_count", 0) if orb_result else 0,
        "orb_matching_quality": orb_result.get("matching_quality", "unknown") if orb_result else "unknown",
        "ocr_status": ocr_result.get("status", "unavailable") if ocr_result else "unavailable",
        "ocr_summary": ocr_result.get("summary", "") if ocr_result else "",
        "ocr_text": ocr_result.get("raw_text", "") if ocr_result else "",
        "ocr_interpretation": ocr_result.get("korean_interpretation", "") if ocr_result else "",
        "ocr_engine": ocr_result.get("engine", "none") if ocr_result else "none",
    }


def generate_feedback(metrics: dict[str, Any]) -> str:
    comments: list[str] = []

    if metrics["brightness"] < 45:
        comments.append("이미지가 전체적으로 어두워 밝기 보정이 필요합니다.")
    elif metrics["brightness"] > 75:
        comments.append("이미지가 다소 밝아 하이라이트가 날아갈 수 있습니다.")

    if metrics["contrast"] < 45:
        comments.append("대비가 낮아 피사체 구분이 약하므로 contrast enhancement가 유효합니다.")

    if metrics["exposure_state"] == "underexposed":
        comments.append(
            f"노출 상태가 underexposed로 판단되며 어두운 영역 비율이 {metrics['shadow_ratio']}%로 높습니다."
        )
    elif metrics["exposure_state"] == "overexposed":
        comments.append(
            f"노출 상태가 overexposed로 판단되며 밝은 하이라이트 비율이 {metrics['highlight_ratio']}%입니다."
        )
    elif metrics["exposure_state"] == "low_dynamic_range":
        comments.append(
            f"명암 분포가 좁아 low dynamic range 상태로 보이며 dynamic range score는 {metrics['dynamic_range_score']}입니다."
        )

    if metrics["color_cast_score"] >= 22:
        comments.append(
            f"색 편향이 비교적 강해 white balance 보정의 영향이 크게 나타날 수 있습니다 ({metrics['dominant_cast']})."
        )
    elif metrics["color_cast_score"] >= 10:
        comments.append(
            f"약한 색 편향이 있어 white balance 보정이 유효합니다 ({metrics['dominant_cast']})."
        )

    if metrics["blur"] < 40:
        comments.append("선명도가 낮아 sharpening 적용이 도움이 됩니다.")

    if metrics["edge_density"] < 8:
        comments.append("에지 밀도가 낮아 장면 구조가 다소 평면적으로 보일 수 있습니다.")

    if metrics.get("tilt_summary"):
        comments.append(f"수평 보정 분석 결과는 {metrics['tilt_summary']}")

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

    if metrics.get("segmentation_summary"):
        comments.append(f"영역 분할 기준 요약은 {metrics['segmentation_summary']} 입니다.")

    if metrics.get("crop_summary"):
        comments.append(f"추천 크롭 가이드는 {metrics['crop_summary']}")

    if metrics.get("orb_summary"):
        comments.append(f"특징점 대응 분석은 {metrics['orb_summary']}")

    if metrics.get("difference_summary"):
        comments.append(f"보정 변화 요약은 {metrics['difference_summary']}")

    comments.append(f"추정 장면 유형은 {metrics['scene']}입니다.")
    return " ".join(comments)
