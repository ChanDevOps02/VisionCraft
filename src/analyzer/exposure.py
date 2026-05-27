from __future__ import annotations

import cv2
import numpy as np


def analyze_exposure(image: np.ndarray) -> dict[str, float | str]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)

    shadow_ratio = float(np.mean(gray <= 45))
    highlight_ratio = float(np.mean(gray >= 225))
    p5 = float(np.percentile(gray_f, 5))
    p95 = float(np.percentile(gray_f, 95))
    dynamic_range = p95 - p5

    if shadow_ratio > 0.32 and p95 < 190:
        exposure_state = "underexposed"
    elif highlight_ratio > 0.18 and p5 > 40:
        exposure_state = "overexposed"
    elif dynamic_range < 85:
        exposure_state = "low_dynamic_range"
    else:
        exposure_state = "balanced"

    return {
        "exposure_state": exposure_state,
        "shadow_ratio": round(shadow_ratio * 100.0, 2),
        "highlight_ratio": round(highlight_ratio * 100.0, 2),
        "dynamic_range_score": round(min((dynamic_range / 180.0) * 100.0, 100.0), 2),
        "p5_intensity": round(p5, 2),
        "p95_intensity": round(p95, 2),
    }
