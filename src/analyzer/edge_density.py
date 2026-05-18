from __future__ import annotations

import cv2
import numpy as np


def calculate_edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = float(np.count_nonzero(edges) / edges.size)
    return round(density * 100.0, 2)
