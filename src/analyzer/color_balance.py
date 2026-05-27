from __future__ import annotations

import numpy as np


def analyze_color_balance(image: np.ndarray) -> dict[str, object]:
    image_f = image.astype(np.float32)
    channel_means = image_f.reshape(-1, 3).mean(axis=0)
    overall_mean = float(channel_means.mean())

    if overall_mean <= 1e-6:
        channel_scale = np.ones(3, dtype=np.float32)
    else:
        channel_scale = overall_mean / np.clip(channel_means, 1e-6, None)

    imbalance = float(np.max(np.abs(channel_means - overall_mean)) / max(overall_mean, 1e-6))
    color_cast_score = round(min(100.0, imbalance * 220.0), 2)

    if color_cast_score < 10:
        shift = "mild"
    elif color_cast_score < 22:
        shift = "moderate"
    else:
        shift = "strong"

    dominant_index = int(np.argmax(np.abs(channel_means - overall_mean)))
    channel_names = ["R", "G", "B"]
    dominant_channel = channel_names[dominant_index]
    dominant_direction = "high" if channel_means[dominant_index] > overall_mean else "low"

    return {
        "color_cast_score": color_cast_score,
        "white_balance_shift": shift,
        "channel_means": {
            "R": round(float(channel_means[0]), 2),
            "G": round(float(channel_means[1]), 2),
            "B": round(float(channel_means[2]), 2),
        },
        "channel_scale": {
            "R": round(float(channel_scale[0]), 3),
            "G": round(float(channel_scale[1]), 3),
            "B": round(float(channel_scale[2]), 3),
        },
        "dominant_cast": f"{dominant_channel} {dominant_direction}",
    }
