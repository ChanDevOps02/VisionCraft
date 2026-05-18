from __future__ import annotations

import cv2
import numpy as np


def calculate_brightness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_intensity = float(np.mean(gray))
    return round((mean_intensity / 255.0) * 100.0, 2)
