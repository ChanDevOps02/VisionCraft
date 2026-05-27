from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _plot_rgb_histogram(ax, image: np.ndarray, title: str):
    colors = ("red", "green", "blue")

    for idx, color in enumerate(colors):
        hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel Count")
    ax.set_xlim([0, 256])
    ax.grid(alpha=0.2)


def build_histogram_figure(image: np.ndarray, enhanced_image: np.ndarray | None = None):
    if enhanced_image is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_rgb_histogram(ax, image, "Original RGB Histogram")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    _plot_rgb_histogram(axes[0], image, "Original RGB Histogram")
    _plot_rgb_histogram(axes[1], enhanced_image, "Enhanced RGB Histogram")
    fig.tight_layout()
    return fig
