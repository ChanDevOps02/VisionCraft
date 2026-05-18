from __future__ import annotations

import cv2
import numpy as np


def calculate_contrast_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    std_value = float(np.std(gray))
    return round(min((std_value / 64.0) * 100.0, 100.0), 2)
