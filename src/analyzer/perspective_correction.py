from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def _warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = _order_points(quad)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(1, int(round(max(width_a, width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


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


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * (height / image.shape[0]))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _build_visualization(overlay: np.ndarray, rectified: np.ndarray) -> np.ndarray:
    target_height = overlay.shape[0]
    rectified_resized = _resize_to_height(rectified, target_height)
    left = _label_panel(overlay, "detected planar quad")
    right = _label_panel(rectified_resized, "perspective preview")
    return cv2.hconcat([left, right])


def analyze_perspective_correction(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    image_area = float(height * width)

    best_quad = None
    best_score = -1.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        area = cv2.contourArea(quad)
        if area < image_area * 0.06:
            continue
        if not cv2.isContourConvex(quad.astype(np.int32)):
            continue

        rect = _order_points(quad)
        top_width = np.linalg.norm(rect[1] - rect[0])
        bottom_width = np.linalg.norm(rect[2] - rect[3])
        left_height = np.linalg.norm(rect[3] - rect[0])
        right_height = np.linalg.norm(rect[2] - rect[1])

        horizontal_skew = abs(top_width - bottom_width) / max(max(top_width, bottom_width), 1.0)
        vertical_skew = abs(left_height - right_height) / max(max(left_height, right_height), 1.0)
        skew_score = horizontal_skew + vertical_skew
        score = (area / image_area) + skew_score * 0.6

        if score > best_score:
            best_score = score
            best_quad = rect

    overlay = image.copy()
    if best_quad is None:
        return {
            "status": "unavailable",
            "summary": "원근 보정에 적합한 사각형 평면 후보를 찾지 못했습니다.",
            "annotated_image": overlay,
            "rectified_preview": image.copy(),
            "visualization_image": _build_visualization(overlay, image.copy()),
        }

    cv2.polylines(overlay, [best_quad.astype(np.int32)], isClosed=True, color=(80, 220, 255), thickness=3)
    rectified = _warp_quad(image, best_quad)

    top_width = np.linalg.norm(best_quad[1] - best_quad[0])
    bottom_width = np.linalg.norm(best_quad[2] - best_quad[3])
    left_height = np.linalg.norm(best_quad[3] - best_quad[0])
    right_height = np.linalg.norm(best_quad[2] - best_quad[1])
    skew_amount = max(
        abs(top_width - bottom_width) / max(max(top_width, bottom_width), 1.0),
        abs(left_height - right_height) / max(max(left_height, right_height), 1.0),
    )

    if skew_amount < 0.08:
        strength = "가벼운"
    elif skew_amount < 0.18:
        strength = "중간 정도의"
    else:
        strength = "비교적 강한"

    summary = f"사각형 평면 후보가 감지되어 {strength} perspective correction을 추천합니다."

    return {
        "status": "ok",
        "summary": summary,
        "annotated_image": overlay,
        "rectified_preview": rectified,
        "visualization_image": _build_visualization(overlay, rectified),
        "skew_amount": round(float(skew_amount), 3),
    }
