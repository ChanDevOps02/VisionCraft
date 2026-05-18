#분석 점수에 따라 어떤 보정을 얼마나 할지 결정함.
from __future__ import annotations

import cv2
import numpy as np

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

#Contrast Limited Adaptive Histogram Equalization
#히스토그램 평활화를 너무 과하게 하지 않으면서, 지역적으로 대비를 개선
def _apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) #LAB로 바꾸는 이유 : 밝기 정보(L)와 생상 정보 (A, B)가 구분되어 있어서 밝기만 따로 개선하기 좋음
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #clipLimit = 2.0 : 대비가 너무 과하게 늘어나지 않도록 함 | tileGridSize = (8, 8)이미지를 더 작은 영역들로 나눠서 각 영역별 적응형 보정
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel)) #밝기 채널만 보정한 후 색상 채널과 다시 합치기
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

#흐릿한 이미지를 더 선명하게 보이도록 해줌 (edge, detail강조됨)
def _apply_sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) #전형적인 sharpening kernel사용
    return cv2.filter2D(image, -1, kernel)


def enhance_image(image: np.ndarray, metrics: dict) -> np.ndarray:
    enhanced = image.copy() #원본 복사본 생성
    enhanced = _apply_brightness_contrast(enhanced, metrics) #분석 점수를 보고 밝기와 대비를 먼저 큰 틀에서 잡기
    enhanced = _apply_clahe(enhanced) #지역 대비 개선 (어두운 부분과 밝은 부분의 디테일 살려주기)

    if metrics["blur"] < 60:    #blur점수가 낮다는 건 이미지가 상대적으로 흐리다는 뜻이므로 그 경우에만 sharpening적용해주기
        enhanced = _apply_sharpen(enhanced)

    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 4, 4, 7, 21) #컬러 이미지용 노이즈 제거
    return enhanced
