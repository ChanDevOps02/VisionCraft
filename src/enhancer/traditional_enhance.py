#분석 점수에 따라 어떤 보정을 얼마나 할지 결정함.
from __future__ import annotations

import cv2
import numpy as np


def _blend_images(base: np.ndarray, adjusted: np.ndarray, adjusted_weight: float) -> np.ndarray:
    adjusted_weight = float(np.clip(adjusted_weight, 0.0, 1.0))
    if adjusted_weight >= 1.0:
        return adjusted
    if adjusted_weight <= 0.0:
        return base
    return cv2.addWeighted(adjusted, adjusted_weight, base, 1.0 - adjusted_weight, 0)


def _composite_mask(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return base
    result = base.copy()
    result[mask] = overlay[mask]
    return result


def _get_scene_enhancement_policy(metrics: dict) -> dict[str, float | str]:
    scene_name = metrics.get("scene", "")
    main_subject = metrics.get("main_subject", "")
    ocr_status = metrics.get("ocr_status", "disabled")

    policy: dict[str, float | str] = {
        "name": "balanced",
        "white_balance_blend": 1.0,
        "brightness_strength": 1.0,
        "contrast_strength": 1.0,
        "gamma_strength": 1.0,
        "clahe_clip": 1.8,
        "clahe_blend": 0.72,
        "sharpen_strength": 1.0,
        "sky_saturation_boost": 1.06,
        "sky_value_boost": 1.025,
        "sky_original_blend": 0.08,
        "green_original_blend": 0.0,
        "green_saturation_floor": 0.0,
        "person_original_blend": 0.36,
        "background_denoise_blend": 0.85,
        "final_original_blend": 0.0,
    }

    if metrics.get("color_cast_score", 0.0) < 8:
        policy["white_balance_blend"] = 0.35

    if scene_name in {"bedroom", "restaurant_cafe", "kitchen_dining", "office_study"}:
        policy.update(
            {
                "name": "indoor-natural",
                "white_balance_blend": min(float(policy["white_balance_blend"]), 0.55),
                "brightness_strength": 0.82,
                "contrast_strength": 0.72,
                "gamma_strength": 0.84,
                "clahe_clip": 1.35,
                "clahe_blend": 0.42,
                "sharpen_strength": 0.62,
                "person_original_blend": 0.44,
                "background_denoise_blend": 0.72,
                "final_original_blend": 0.10,
            }
        )
    elif scene_name in {"waterfront", "mountain_valley", "forest_nature", "open_field_landscape"}:
        policy.update(
            {
                "name": "landscape-natural",
                "white_balance_blend": min(float(policy["white_balance_blend"]), 0.45),
                "brightness_strength": 0.90,
                "contrast_strength": 0.82,
                "gamma_strength": 0.86,
                "clahe_clip": 1.45,
                "clahe_blend": 0.40,
                "sharpen_strength": 0.78,
                "sky_saturation_boost": 1.0,
                "sky_value_boost": 1.0,
                "sky_original_blend": 0.62,
                "green_original_blend": 0.38,
                "green_saturation_floor": 0.94,
                "final_original_blend": 0.20,
            }
        )
    elif scene_name in {"street_downtown", "transportation_hub_road", "residential_outdoor"}:
        policy.update(
            {
                "name": "urban-balanced",
                "white_balance_blend": min(float(policy["white_balance_blend"]), 0.52),
                "brightness_strength": 0.70,
                "contrast_strength": 0.72,
                "gamma_strength": 0.72,
                "clahe_clip": 1.30,
                "clahe_blend": 0.34,
                "sharpen_strength": 0.68,
                "sky_saturation_boost": 1.0,
                "sky_value_boost": 1.0,
                "sky_original_blend": 0.58,
                "background_denoise_blend": 0.62,
                "final_original_blend": 0.24,
            }
        )
    elif scene_name in {"corridor_lobby", "public_large_indoor", "industrial_area"}:
        policy.update(
            {
                "name": "structure-preserving",
                "contrast_strength": 0.86,
                "gamma_strength": 0.92,
                "clahe_clip": 1.55,
                "clahe_blend": 0.56,
                "sharpen_strength": 0.76,
                "final_original_blend": 0.06,
            }
        )

    if main_subject == "person":
        policy["name"] = f"{policy['name']} + person-safe"
        policy["sharpen_strength"] = min(float(policy["sharpen_strength"]), 0.58)
        policy["clahe_blend"] = min(float(policy["clahe_blend"]), 0.50)
        policy["person_original_blend"] = max(float(policy["person_original_blend"]), 0.46)
        policy["final_original_blend"] = max(float(policy["final_original_blend"]), 0.08)

    if ocr_status in {"ok", "low_confidence"}:
        policy.update(
            {
                "name": "document-readable",
                "white_balance_blend": 0.75,
                "brightness_strength": 0.95,
                "contrast_strength": 0.95,
                "gamma_strength": 0.95,
                "clahe_clip": 1.7,
                "clahe_blend": 0.70,
                "sharpen_strength": 0.72,
                "final_original_blend": 0.03,
            }
        )

    return policy


# Gray-world white balance to reduce global color cast.
def _apply_white_balance(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    channel_means = image_f.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())
    scale = overall_mean / np.clip(channel_means, 1e-6, None)
    balanced = image_f * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


#이 함수는 이미지와 분석 결과 metrics를 받아서 밝기와 대비를 먼저 조정함.
def _apply_brightness_contrast(image: np.ndarray, metrics: dict, policy: dict) -> np.ndarray:
    #new_image = alpha * image + beta
    alpha = 1.0
    beta = 0

    #대비 점수가 낮을 때만 대비를 올리는 부분 (대비가 부족할 수록 alpha를 키워줌)
    if metrics["contrast"] < 50:
        alpha += ((50 - metrics["contrast"]) / 120.0) * float(policy["contrast_strength"])
    #밝기 점수에 따라 보정 (밝기가 낮으면 beta키워줌)
    if metrics["brightness"] < 50:
        beta += int((50 - metrics["brightness"]) * 1.5 * float(policy["brightness_strength"]))
    elif metrics["brightness"] > 75:
        beta -= int((metrics["brightness"] - 75) * 1.2 * float(policy["brightness_strength"]))

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _apply_gamma_correction(image: np.ndarray, metrics: dict, policy: dict) -> np.ndarray:
    brightness = metrics["brightness"]
    if brightness < 40:
        gamma = 0.78
    elif brightness < 50:
        gamma = 0.88
    elif brightness > 82:
        gamma = 1.12
    else:
        gamma = 1.0

    gamma = 1.0 + (gamma - 1.0) * float(policy["gamma_strength"])
    if gamma == 1.0:
        return image

    lookup = np.array(
        [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lookup)


#Contrast Limited Adaptive Histogram Equalization
#히스토그램 평활화를 너무 과하게 하지 않으면서, 지역적으로 대비를 개선
def _apply_clahe(image: np.ndarray, policy: dict) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) #LAB로 바꾸는 이유 : 밝기 정보(L)와 생상 정보 (A, B)가 구분되어 있어서 밝기만 따로 개선하기 좋음
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(policy["clahe_clip"]), tileGridSize=(8, 8)) #clipLimit: 장면별로 과보정을 막기 위해 조절
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel)) #밝기 채널만 보정한 후 색상 채널과 다시 합치기
    clahe_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return _blend_images(image, clahe_image, float(policy["clahe_blend"]))


def _apply_adaptive_sharpen(image: np.ndarray, metrics: dict, policy: dict) -> np.ndarray:
    blur_score = metrics["blur"]
    scene_name = metrics.get("scene", "")
    main_subject = metrics.get("main_subject", "")

    if blur_score >= 70:
        return image

    strength = 0.0
    if blur_score < 30:
        strength = 1.2
    elif blur_score < 45:
        strength = 0.8
    elif blur_score < 60:
        strength = 0.45

    if main_subject == "person" or scene_name in {"bedroom", "office_study"}:
        strength *= 0.75
    strength *= float(policy["sharpen_strength"])

    if strength <= 0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _apply_adaptive_denoise(image: np.ndarray, metrics: dict) -> np.ndarray:
    edge_density = metrics["edge_density"]
    blur_score = metrics["blur"]
    brightness = metrics["brightness"]

    if edge_density < 4:
        return cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)

    if blur_score < 35 and brightness < 55:
        return cv2.medianBlur(image, 3)

    return cv2.fastNlMeansDenoisingColored(image, None, 4, 4, 7, 21)


def _apply_region_aware_adjustments(
    image: np.ndarray,
    original_image: np.ndarray,
    region_result: dict | None,
    policy: dict,
) -> tuple[np.ndarray, list[str], list[str]]:
    if not region_result:
        return image, [], []

    person_mask = region_result.get("person_mask")
    sky_mask = region_result.get("sky_mask")
    background_mask = region_result.get("background_mask")

    if person_mask is None or sky_mask is None or background_mask is None:
        return image, [], []

    result = image.copy()
    applied_steps: list[str] = []
    reason_steps: list[str] = []

    if np.any(background_mask):
        background_variant = cv2.bilateralFilter(result, d=5, sigmaColor=25, sigmaSpace=25)
        background_variant = _blend_images(result, background_variant, float(policy["background_denoise_blend"]))
        result = _composite_mask(result, background_variant, background_mask)
        applied_steps.append("background-aware denoise")
        reason_steps.append("배경 영역은 디테일 손실이 적도록 추가 노이즈 완화를 적용했습니다.")

    if np.any(sky_mask):
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(policy["sky_saturation_boost"]), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(policy["sky_value_boost"]), 0, 255)
        sky_variant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        sky_original_blend = float(policy["sky_original_blend"])
        sky_variant = cv2.addWeighted(sky_variant, 1.0 - sky_original_blend, original_image, sky_original_blend, 0)
        result = _composite_mask(result, sky_variant, sky_mask)
        applied_steps.append("sky-preserving color blend")
        reason_steps.append("하늘 영역은 원본 색을 더 많이 보존해 보라색/분홍색 색 밀림을 줄였습니다.")

    if np.any(person_mask):
        original_blend = float(policy["person_original_blend"])
        person_variant = cv2.addWeighted(result, 1.0 - original_blend, original_image, original_blend, 0)
        result = _composite_mask(result, person_variant, person_mask)
        applied_steps.append("person-preserving blend")
        reason_steps.append("인물 영역은 과도한 샤프닝을 줄이기 위해 원본과 부드럽게 혼합했습니다.")

    return result, applied_steps, reason_steps


def _apply_greenery_color_preservation(
    image: np.ndarray,
    original_image: np.ndarray,
    policy: dict,
) -> tuple[np.ndarray, list[str], list[str]]:
    green_original_blend = float(policy.get("green_original_blend", 0.0))
    if green_original_blend <= 0:
        return image, [], []

    original_hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    result_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    hue = original_hsv[:, :, 0]
    saturation = original_hsv[:, :, 1]
    value = original_hsv[:, :, 2]
    greenery_mask = (
        (hue >= 28)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 25)
    )

    if not np.any(greenery_mask):
        return image, [], []

    saturation_floor = float(policy.get("green_saturation_floor", 0.0))
    if saturation_floor > 0:
        original_saturation = original_hsv[:, :, 1].astype(np.float32)
        result_hsv[:, :, 1] = np.where(
            greenery_mask,
            np.maximum(result_hsv[:, :, 1], original_saturation * saturation_floor),
            result_hsv[:, :, 1],
        )

    saturation_preserved = cv2.cvtColor(np.clip(result_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    original_blended = cv2.addWeighted(
        saturation_preserved,
        1.0 - green_original_blend,
        original_image,
        green_original_blend,
        0,
    )
    result = _composite_mask(saturation_preserved, original_blended, greenery_mask)

    return (
        result,
        ["greenery color preservation"],
        ["식생 영역은 원본 초록 채도와 색감을 일부 되살려 보정 후 색이 죽는 현상을 줄였습니다."],
    )


def enhance_image(
    image: np.ndarray,
    metrics: dict,
    region_result: dict | None = None,
) -> tuple[np.ndarray, dict[str, list[str] | str]]:
    enhanced = image.copy() #원본 복사본 생성
    applied_steps: list[str] = []
    reason_steps: list[str] = []
    policy = _get_scene_enhancement_policy(metrics)

    white_balanced = _apply_white_balance(enhanced) #채널 평균을 맞춰 전반적인 색 편향 완화
    enhanced = _blend_images(enhanced, white_balanced, float(policy["white_balance_blend"]))
    applied_steps.append("white balance")
    reason_steps.append("전반적인 색 편향을 줄이되 장면의 원래 색감을 보존하도록 white balance 강도를 조절했습니다.")

    if metrics["contrast"] < 50 or metrics["brightness"] < 50 or metrics["brightness"] > 75:
        applied_steps.append("brightness / contrast scaling")
        if metrics["brightness"] < 50:
            reason_steps.append("밝기 점수가 낮아 전역 밝기 보정을 적용했습니다.")
        elif metrics["brightness"] > 75:
            reason_steps.append("밝기 점수가 높아 하이라이트 억제를 위해 밝기를 낮췄습니다.")
        if metrics["contrast"] < 50:
            reason_steps.append("대비 점수가 낮아 전역 contrast scaling을 적용했습니다.")
    enhanced = _apply_brightness_contrast(enhanced, metrics, policy) #분석 점수를 보고 밝기와 대비를 먼저 큰 틀에서 잡기

    gamma_applied = metrics["brightness"] < 50 or metrics["brightness"] > 82
    enhanced = _apply_gamma_correction(enhanced, metrics, policy) #저조도/과노출 구간은 감마로 자연스럽게 보정
    if gamma_applied:
        applied_steps.append("gamma correction")
        if metrics["brightness"] < 50:
            reason_steps.append("저조도 구간을 더 자연스럽게 밝히기 위해 gamma correction을 적용했습니다.")
        else:
            reason_steps.append("과도한 밝기 구간을 완화하기 위해 gamma correction을 적용했습니다.")

    enhanced = _apply_clahe(enhanced, policy) #지역 대비 개선 (어두운 부분과 밝은 부분의 디테일 살려주기)
    applied_steps.append("CLAHE")
    reason_steps.append("지역 대비와 디테일 복원을 위해 CLAHE를 적용하되 장면별 blend 강도로 과보정을 제한했습니다.")

    sharpen_applied = metrics["blur"] < 60
    enhanced = _apply_adaptive_sharpen(enhanced, metrics, policy) #scene/blur 기반 샤프닝 강도 조절
    if sharpen_applied:
        applied_steps.append("adaptive sharpening")
        reason_steps.append("blur 점수가 낮아 에지와 디테일을 살리기 위해 adaptive sharpening을 적용했습니다.")

    if metrics["edge_density"] < 4:
        denoise_mode = "bilateral filtering"
        denoise_reason = "에지 밀도가 낮아 구조 보존형 bilateral filtering을 적용했습니다."
    elif metrics["blur"] < 35 and metrics["brightness"] < 55:
        denoise_mode = "median filtering"
        denoise_reason = "저조도와 낮은 선명도를 함께 보여 median filtering으로 노이즈를 정리했습니다."
    else:
        denoise_mode = "fast non-local means denoising"
        denoise_reason = "기본 컬러 노이즈 제거를 위해 fast non-local means denoising을 적용했습니다."

    enhanced = _apply_adaptive_denoise(enhanced, metrics) #장면 구조에 맞는 필터 선택
    applied_steps.append(denoise_mode)
    reason_steps.append(denoise_reason)

    enhanced, region_steps, region_reasons = _apply_region_aware_adjustments(
        enhanced,
        image,
        region_result,
        policy,
    )
    applied_steps.extend(region_steps)
    reason_steps.extend(region_reasons)

    enhanced, greenery_steps, greenery_reasons = _apply_greenery_color_preservation(
        enhanced,
        image,
        policy,
    )
    applied_steps.extend(greenery_steps)
    reason_steps.extend(greenery_reasons)

    final_original_blend = float(policy["final_original_blend"])
    if final_original_blend > 0:
        enhanced = cv2.addWeighted(enhanced, 1.0 - final_original_blend, image, final_original_blend, 0)
        applied_steps.append("scene-aware preservation blend")
        reason_steps.append("장면별 색감과 질감이 과하게 변하지 않도록 원본을 소량 다시 혼합했습니다.")

    enhancement_report = {
        "applied_steps": applied_steps,
        "reason_steps": reason_steps,
        "summary": f"{policy['name']}: " + ", ".join(applied_steps),
    }
    return enhanced, enhancement_report
