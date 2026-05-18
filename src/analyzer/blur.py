from __future__ import annotations

import cv2
import numpy as np


def calculate_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(min((laplacian_var / 500.0) * 100.0, 100.0), 2)
