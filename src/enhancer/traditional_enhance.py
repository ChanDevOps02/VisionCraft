#분석 점수에 따라 어떤 보정을 얼마나 할지 결정함.
from __future__ import annotations

import cv2
import numpy as np


def _composite_mask(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return base
    result = base.copy()
    result[mask] = overlay[mask]
    return result


# Gray-world white balance to reduce global color cast.
def _apply_white_balance(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    channel_means = image_f.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())
    scale = overall_mean / np.clip(channel_means, 1e-6, None)
    balanced = image_f * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


#이 함수는 이미지와 분석 결과 metrics를 받아서 밝기와 대비를 먼저 조정함.
def _apply_brightness_contrast(image: np.ndarray, metrics: dict) -> np.ndarray:
    #new_image = alpha * image + beta
    alpha = 1.0
    beta = 0

    #대비 점수가 낮을 때만 대비를 올리는 부분 (대비가 부족할 수록 alpha를 키워줌)
    if metrics["contrast"] < 50:
        alpha += (50 - metrics["contrast"]) / 120.0
    #밝기 점수에 따라 보정 (밝기가 낮으면 beta키워줌)
    if metrics["brightness"] < 50:
        beta += int((50 - metrics["brightness"]) * 1.5)
    elif metrics["brightness"] > 75:
        beta -= int((metrics["brightness"] - 75) * 1.2)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _apply_gamma_correction(image: np.ndarray, metrics: dict) -> np.ndarray:
    brightness = metrics["brightness"]
    if brightness < 40:
        gamma = 0.78
    elif brightness < 50:
        gamma = 0.88
    elif brightness > 82:
        gamma = 1.12
    else:
        gamma = 1.0

    if gamma == 1.0:
        return image

    lookup = np.array(
        [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lookup)


#Contrast Limited Adaptive Histogram Equalization
#히스토그램 평활화를 너무 과하게 하지 않으면서, 지역적으로 대비를 개선
def _apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) #LAB로 바꾸는 이유 : 밝기 정보(L)와 생상 정보 (A, B)가 구분되어 있어서 밝기만 따로 개선하기 좋음
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #clipLimit = 2.0 : 대비가 너무 과하게 늘어나지 않도록 함 | tileGridSize = (8, 8)이미지를 더 작은 영역들로 나눠서 각 영역별 적응형 보정
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel)) #밝기 채널만 보정한 후 색상 채널과 다시 합치기
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _apply_adaptive_sharpen(image: np.ndarray, metrics: dict) -> np.ndarray:
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
        result = _composite_mask(result, background_variant, background_mask)
        applied_steps.append("background-aware denoise")
        reason_steps.append("배경 영역은 디테일 손실이 적도록 추가 노이즈 완화를 적용했습니다.")

    if np.any(sky_mask):
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.04, 0, 255)
        sky_variant = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = _composite_mask(result, sky_variant, sky_mask)
        applied_steps.append("sky-aware color boost")
        reason_steps.append("하늘 영역은 채도와 밝기를 소폭 올려 색감을 더 자연스럽게 복원했습니다.")

    if np.any(person_mask):
        person_variant = cv2.addWeighted(result, 0.68, original_image, 0.32, 0)
        result = _composite_mask(result, person_variant, person_mask)
        applied_steps.append("person-preserving blend")
        reason_steps.append("인물 영역은 과도한 샤프닝을 줄이기 위해 원본과 부드럽게 혼합했습니다.")

    return result, applied_steps, reason_steps


def enhance_image(
    image: np.ndarray,
    metrics: dict,
    region_result: dict | None = None,
) -> tuple[np.ndarray, dict[str, list[str] | str]]:
    enhanced = image.copy() #원본 복사본 생성
    applied_steps: list[str] = []
    reason_steps: list[str] = []

    enhanced = _apply_white_balance(enhanced) #채널 평균을 맞춰 전반적인 색 편향 완화
    applied_steps.append("white balance")
    reason_steps.append("전반적인 색 편향을 줄이기 위해 채널 평균을 보정했습니다.")

    if metrics["contrast"] < 50 or metrics["brightness"] < 50 or metrics["brightness"] > 75:
        applied_steps.append("brightness / contrast scaling")
        if metrics["brightness"] < 50:
            reason_steps.append("밝기 점수가 낮아 전역 밝기 보정을 적용했습니다.")
        elif metrics["brightness"] > 75:
            reason_steps.append("밝기 점수가 높아 하이라이트 억제를 위해 밝기를 낮췄습니다.")
        if metrics["contrast"] < 50:
            reason_steps.append("대비 점수가 낮아 전역 contrast scaling을 적용했습니다.")
    enhanced = _apply_brightness_contrast(enhanced, metrics) #분석 점수를 보고 밝기와 대비를 먼저 큰 틀에서 잡기

    gamma_applied = metrics["brightness"] < 50 or metrics["brightness"] > 82
    enhanced = _apply_gamma_correction(enhanced, metrics) #저조도/과노출 구간은 감마로 자연스럽게 보정
    if gamma_applied:
        applied_steps.append("gamma correction")
        if metrics["brightness"] < 50:
            reason_steps.append("저조도 구간을 더 자연스럽게 밝히기 위해 gamma correction을 적용했습니다.")
        else:
            reason_steps.append("과도한 밝기 구간을 완화하기 위해 gamma correction을 적용했습니다.")

    enhanced = _apply_clahe(enhanced) #지역 대비 개선 (어두운 부분과 밝은 부분의 디테일 살려주기)
    applied_steps.append("CLAHE")
    reason_steps.append("지역 대비와 디테일 복원을 위해 CLAHE를 적용했습니다.")

    sharpen_applied = metrics["blur"] < 60
    enhanced = _apply_adaptive_sharpen(enhanced, metrics) #scene/blur 기반 샤프닝 강도 조절
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
    )
    applied_steps.extend(region_steps)
    reason_steps.extend(region_reasons)

    enhancement_report = {
        "applied_steps": applied_steps,
        "reason_steps": reason_steps,
        "summary": ", ".join(applied_steps),
    }
    return enhanced, enhancement_report
