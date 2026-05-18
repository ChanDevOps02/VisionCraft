from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def build_histogram_figure(image: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ("red", "green", "blue")

    for idx, color in enumerate(colors):
        hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1.5)

    ax.set_title("RGB Histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 256])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
