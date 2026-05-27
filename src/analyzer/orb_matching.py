from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def analyze_orb_feature_matching(source_image: np.ndarray, target_image: np.ndarray) -> dict[str, Any]:
    gray_src = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
    gray_tgt = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=600)
    keypoints_src, desc_src = orb.detectAndCompute(gray_src, None)
    keypoints_tgt, desc_tgt = orb.detectAndCompute(gray_tgt, None)

    if desc_src is None or desc_tgt is None or not keypoints_src or not keypoints_tgt:
        fallback = cv2.hconcat([source_image, target_image])
        return {
            "status": "unavailable",
            "summary": "유효한 ORB 특징점을 충분히 찾지 못해 대응점 시각화를 생성하지 않았습니다.",
            "match_count": 0,
            "visualization_image": fallback,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_src, desc_tgt)
    matches = sorted(matches, key=lambda item: item.distance)
    good_matches = matches[: min(40, len(matches))]

    visualization = cv2.drawMatches(
        source_image,
        keypoints_src,
        target_image,
        keypoints_tgt,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if not good_matches:
        return {
            "status": "unavailable",
            "summary": "ORB 특징점은 찾았지만 신뢰할 만한 매칭을 충분히 만들지 못했습니다.",
            "match_count": 0,
            "visualization_image": visualization,
        }

    mean_distance = float(np.mean([match.distance for match in good_matches]))
    if mean_distance < 24:
        quality = "strong"
    elif mean_distance < 40:
        quality = "moderate"
    else:
        quality = "weak"

    summary = (
        f"원본과 보정 결과 사이에서 ORB 대응점 {len(good_matches)}개를 시각화했습니다 "
        f"(matching quality: {quality}, mean distance: {mean_distance:.1f})."
    )

    return {
        "status": "ok",
        "summary": summary,
        "match_count": len(good_matches),
        "matching_quality": quality,
        "mean_distance": round(mean_distance, 2),
        "visualization_image": visualization,
    }
