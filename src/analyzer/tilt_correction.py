from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _normalize_angle(angle_deg: float) -> float:
    while angle_deg <= -90.0:
        angle_deg += 180.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    return angle_deg


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    valid_mask = cv2.warpAffine(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    crop_box = _largest_valid_rectangle(valid_mask > 0)
    if crop_box is None:
        return cv2.resize(rotated, (width, height), interpolation=cv2.INTER_LINEAR)

    x1, y1, x2, y2 = crop_box
    cropped = rotated[y1:y2, x1:x2]

    if cropped.size == 0:
        return cv2.resize(rotated, (width, height), interpolation=cv2.INTER_LINEAR)

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2 or mask.size == 0:
        return None

    height, width = mask.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_box: tuple[int, int, int, int] | None = None

    for row in range(height):
        histogram = np.where(mask[row], histogram + 1, 0)
        stack: list[int] = []

        for col in range(width + 1):
            current_height = int(histogram[col]) if col < width else 0
            while stack and current_height < int(histogram[stack[-1]]):
                top = stack.pop()
                rect_height = int(histogram[top])
                if rect_height <= 0:
                    continue
                left = stack[-1] + 1 if stack else 0
                rect_width = col - left
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    x1 = left
                    x2 = col
                    y2 = row + 1
                    y1 = y2 - rect_height
                    best_box = (x1, y1, x2, y2)
            stack.append(col)

    return best_box


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(values)
    sorted_values = np.array(values, dtype=np.float32)[order]
    sorted_weights = np.array(weights, dtype=np.float32)[order]
    cumulative = np.cumsum(sorted_weights)
    midpoint = float(sorted_weights.sum()) / 2.0
    index = int(np.searchsorted(cumulative, midpoint, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _weighted_median_deviation(values: list[float], weights: list[float], center: float) -> float:
    if not values:
        return 0.0
    deviations = [abs(value - center) for value in values]
    return _weighted_median(deviations, weights)


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(
        labeled,
        text,
        (12, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _build_tilt_visualization(
    line_overlay: np.ndarray,
    corrected_preview: np.ndarray,
    angle_deg: float,
    correction_direction: str,
) -> np.ndarray:
    direction_text = {
        "clockwise": "rotate CW",
        "counterclockwise": "rotate CCW",
        "none": "no rotation",
    }.get(correction_direction, "unknown")

    left = _label_panel(line_overlay, "detected guide lines")
    right = _label_panel(corrected_preview, f"{direction_text} {angle_deg:.1f} deg")
    return cv2.hconcat([left, right])


def analyze_tilt_and_horizon(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=55,
        minLineLength=max(30, int(min(image.shape[:2]) * 0.09)),
        maxLineGap=24,
    )

    overlay = image.copy()
    if lines is None:
        visualization = _build_tilt_visualization(overlay, image.copy(), 0.0, "none")
        return {
            "status": "unavailable",
            "summary": "지배적인 직선을 찾지 못해 기울기 보정 제안을 생성하지 않았습니다.",
            "tilt_angle_deg": 0.0,
            "tilt_state": "unknown",
            "correction_direction": "none",
            "line_overlay": overlay,
            "corrected_preview": image.copy(),
            "visualization_image": visualization,
        }

    candidate_angles: list[float] = []
    candidate_weights: list[float] = []
    horizontal_angles: list[float] = []
    horizontal_weights: list[float] = []
    vertical_angles: list[float] = []
    vertical_weights: list[float] = []
    selected_lines: list[tuple[int, int, int, int]] = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line.tolist())
        length = float(np.hypot(x2 - x1, y2 - y1))

        if length < 1.0:
            continue

        hx1, hy1, hx2, hy2 = x1, y1, x2, y2
        if hx2 < hx1:
            hx1, hy1, hx2, hy2 = hx2, hy2, hx1, hy1
        horizontal_angle = float(np.degrees(np.arctan2(hy2 - hy1, hx2 - hx1)))
        horizontal_angle = _normalize_angle(horizontal_angle)

        # Near-horizontal lines directly contribute their angle.
        if abs(horizontal_angle) <= 20.0:
            candidate_angles.append(horizontal_angle)
            candidate_weights.append(length)
            horizontal_angles.append(horizontal_angle)
            horizontal_weights.append(length)
            selected_lines.append((hx1, hy1, hx2, hy2))
            continue

        vx1, vy1, vx2, vy2 = x1, y1, x2, y2
        if vy2 < vy1:
            vx1, vy1, vx2, vy2 = vx2, vy2, vx1, vy1
        vertical_angle = float(np.degrees(np.arctan2(vy2 - vy1, vx2 - vx1)))
        vertical_angle = _normalize_angle(vertical_angle)

        # Near-vertical lines are converted into equivalent tilt around the vertical axis.
        vertical_offset = min(abs(vertical_angle - 90.0), abs(vertical_angle + 90.0))
        if vertical_offset <= 12.0:
            equivalent_tilt = float(np.degrees(np.arctan2(vx2 - vx1, vy2 - vy1)))
            candidate_angles.append(equivalent_tilt)
            candidate_weights.append(length * 0.85)
            vertical_angles.append(equivalent_tilt)
            vertical_weights.append(length * 0.85)
            selected_lines.append((vx1, vy1, vx2, vy2))

    if not candidate_angles:
        visualization = _build_tilt_visualization(overlay, image.copy(), 0.0, "none")
        return {
            "status": "unavailable",
            "summary": "수평 또는 수직 기준이 될 직선이 부족해 기울기 보정 제안을 생성하지 않았습니다.",
            "tilt_angle_deg": 0.0,
            "tilt_state": "unknown",
            "correction_direction": "none",
            "line_overlay": overlay,
            "corrected_preview": image.copy(),
            "visualization_image": visualization,
        }

    horizontal_weight_total = float(sum(horizontal_weights))
    vertical_weight_total = float(sum(vertical_weights))
    horizontal_dominant = _weighted_median(horizontal_angles, horizontal_weights) if horizontal_angles else 0.0
    vertical_dominant = _weighted_median(vertical_angles, vertical_weights) if vertical_angles else 0.0

    if horizontal_angles and horizontal_weight_total >= max(120.0, vertical_weight_total * 0.45):
        dominant_angle = horizontal_dominant
    elif vertical_angles:
        dominant_angle = vertical_dominant
    else:
        dominant_angle = _weighted_median(candidate_angles, candidate_weights)

    if horizontal_angles and vertical_angles:
        disagreement = abs(horizontal_dominant - vertical_dominant)
        if disagreement > 4.0 and horizontal_weight_total >= 120.0:
            dominant_angle = horizontal_dominant

    consistency = _weighted_median_deviation(candidate_angles, candidate_weights, dominant_angle)
    if consistency > 3.5 or abs(dominant_angle) > 8.0:
        visualization = _build_tilt_visualization(overlay, image.copy(), 0.0, "none")
        return {
            "status": "unavailable",
            "summary": "직선 방향의 합의가 약해 자동 수평 보정을 적용하지 않았습니다.",
            "tilt_angle_deg": 0.0,
            "tilt_state": "unknown",
            "correction_direction": "none",
            "line_overlay": overlay,
            "corrected_preview": image.copy(),
            "visualization_image": visualization,
        }

    # OpenCV uses positive angles for counterclockwise rotation.
    # The detected line angle itself is the correction angle needed to re-level
    # the image in image-coordinate space.
    corrected_preview = _rotate_image(image, dominant_angle)

    for x1, y1, x2, y2 in selected_lines[:20]:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 220, 80), 2, cv2.LINE_AA)

    height, width = image.shape[:2]
    cv2.line(overlay, (0, height // 2), (width, height // 2), (80, 255, 255), 1, cv2.LINE_AA)

    abs_angle = abs(dominant_angle)
    if abs_angle < 1.0:
        tilt_state = "stable"
    elif abs_angle < 2.5:
        tilt_state = "slight"
    else:
        tilt_state = "noticeable"

    if dominant_angle > 0.8:
        correction_direction = "counterclockwise"
        summary = f"사진이 약 {abs_angle:.1f}도 시계 방향으로 기울어 보여 반시계 방향 보정을 적용했습니다."
    elif dominant_angle < -0.8:
        correction_direction = "clockwise"
        summary = f"사진이 약 {abs_angle:.1f}도 반시계 방향으로 기울어 보여 시계 방향 보정을 적용했습니다."
    else:
        correction_direction = "none"
        summary = "수평선 기준으로 기울기가 크지 않아 별도의 straighten 보정이 크게 필요하지 않습니다."

    visualization = _build_tilt_visualization(
        line_overlay=overlay,
        corrected_preview=corrected_preview,
        angle_deg=abs_angle,
        correction_direction=correction_direction,
    )

    return {
        "status": "ok",
        "summary": summary,
        "tilt_angle_deg": round(abs_angle, 2),
        "tilt_state": tilt_state,
        "correction_direction": correction_direction,
        "line_overlay": overlay,
        "corrected_preview": corrected_preview,
        "visualization_image": visualization,
    }
