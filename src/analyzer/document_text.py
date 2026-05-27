from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np


def _touches_border(points: np.ndarray, width: int, height: int, margin: int = 18) -> bool:
    xs = points[:, 0]
    ys = points[:, 1]
    return (
        xs.min() <= margin
        or ys.min() <= margin
        or xs.max() >= width - 1 - margin
        or ys.max() >= height - 1 - margin
    )


def _quad_score(points: np.ndarray, area: float, width: int, height: int) -> float:
    rect = _order_points(points.astype(np.float32))
    tl, tr, br, bl = rect
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    if min(top, bottom, left, right) < 8:
        return -1.0

    horizontal_ratio = min(top, bottom) / max(top, bottom)
    vertical_ratio = min(left, right) / max(left, right)
    balance = 0.5 * (horizontal_ratio + vertical_ratio)
    border_penalty = 0.45 if _touches_border(points, width, height) else 0.0
    center = points.mean(axis=0)
    center_distance = np.linalg.norm(center - np.array([width / 2.0, height / 2.0], dtype=np.float32))
    center_penalty = min(0.18, center_distance / max(width, height) * 0.18)
    return area * (0.55 + 0.45 * balance) * (1.0 - border_penalty - center_penalty)


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_points(points.astype(np.float32))
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    if max_width < 10 or max_height < 10:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _largest_document_quad(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        7,
    )
    merged = cv2.bitwise_or(edged, thresh)
    contours, _ = cv2.findContours(merged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = height * width
    best_quad = None
    best_score = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < image_area * 0.08:
            continue
        points = approx.reshape(4, 2).astype(np.float32)
        score = _quad_score(points, area, width, height)
        if score > best_score:
            best_score = score
            best_quad = points

    if best_quad is not None:
        return best_quad.astype(np.float32)

    if not contours:
        return None

    inner_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * 0.08
    ]
    if not inner_contours:
        return None

    inner_contours.sort(
        key=lambda contour: cv2.contourArea(contour)
        * (0.55 if _touches_border(cv2.boxPoints(cv2.minAreaRect(contour)), width, height) else 1.0),
        reverse=True,
    )
    largest = inner_contours[0]
    if cv2.contourArea(largest) < image_area * 0.12:
        return None
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def rectify_text_document(image: np.ndarray) -> dict[str, object]:
    quad = _largest_document_quad(image)
    overlay = image.copy()
    if quad is None:
        return {
            "status": "unavailable",
            "summary": "정면 보정용 사각형 텍스트/문서 영역을 찾지 못했습니다.",
            "overlay_image": overlay,
            "rectified_image": image,
            "quad": None,
        }

    quad_i = quad.astype(np.int32)
    cv2.polylines(overlay, [quad_i], True, (255, 200, 0), 4)
    for point in quad_i:
        cv2.circle(overlay, tuple(point), 7, (255, 80, 80), -1)
    rectified = _four_point_transform(image, quad)

    return {
        "status": "ok",
        "summary": "기울어진 문서/텍스트 평면을 검출해 정면 보정 미리보기를 생성했습니다.",
        "overlay_image": overlay,
        "rectified_image": rectified,
        "quad": quad.tolist(),
    }


def build_manual_point_overlay(image: np.ndarray, points: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    overlay = image.copy()
    if not points:
        return overlay

    for index, point in enumerate(points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(overlay, (x, y), 10, (255, 80, 80), -1)
        cv2.putText(
            overlay,
            str(index),
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        poly = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [poly], False, (255, 220, 0), 3)
    if len(points) == 4:
        poly = np.array(points, dtype=np.int32)
        cv2.polylines(overlay, [poly], True, (255, 220, 0), 4)
    return overlay


def rectify_text_document_from_points(
    image: np.ndarray,
    points: list[list[int]] | list[tuple[int, int]],
) -> dict[str, object]:
    overlay = build_manual_point_overlay(image, points)
    if len(points) != 4:
        return {
            "status": "unavailable",
            "summary": "수동 보정을 위해 4개의 꼭짓점을 찍어주세요.",
            "overlay_image": overlay,
            "rectified_image": image,
            "quad": None,
        }

    quad = np.array(points, dtype=np.float32)
    rectified = _four_point_transform(image, quad)
    return {
        "status": "ok",
        "summary": "수동 4점 선택으로 homography 기반 정면 보정을 적용했습니다.",
        "overlay_image": overlay,
        "rectified_image": rectified,
        "quad": quad.tolist(),
    }


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 35, 35)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def _generate_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    processed = _preprocess_for_ocr(image)
    upscaled_rgb = cv2.resize(image, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    upscaled_processed = cv2.resize(processed, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    adaptive = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    otsu = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return [
        image,
        gray,
        processed,
        adaptive,
        otsu,
        upscaled_rgb,
        upscaled_processed,
    ]


def _normalize_text_key(text: str) -> str:
    return " ".join(text.lower().split())


def _clean_ocr_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _extract_paddle_lines(result: object) -> list[tuple[str, float]]:
    lines: list[tuple[str, float]] = []
    if result is None:
        return lines

    if isinstance(result, list):
        for item in result:
            if isinstance(item, list):
                if len(item) == 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                    text = _clean_ocr_text(str(item[1][0]))
                    try:
                        confidence = float(item[1][1])
                    except Exception:
                        confidence = 0.0
                    if text:
                        lines.append((text, confidence))
                else:
                    lines.extend(_extract_paddle_lines(item))
    return lines


@lru_cache(maxsize=1)
def _get_paddleocr_reader():
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None

    try:
        return PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    except Exception:
        return None


def _run_paddleocr(image: np.ndarray) -> tuple[str, str]:
    reader = _get_paddleocr_reader()
    if reader is None:
        return "unavailable", "PaddleOCR를 사용할 수 없습니다."

    variants = _generate_ocr_variants(image)
    paragraph_candidates: list[tuple[str, float, int]] = []
    line_candidates: dict[str, tuple[str, float]] = {}

    for variant in variants:
        try:
            result = reader.ocr(variant, cls=True)
        except Exception:
            continue

        extracted = _extract_paddle_lines(result)
        if not extracted:
            continue

        texts = [text for text, _ in extracted if len(text) >= 2]
        confidences = [conf for text, conf in extracted if len(text) >= 2]
        if texts:
            merged_text = _clean_ocr_text("\n".join(texts))
            char_count = len(merged_text.replace("\n", " ").strip())
            if char_count >= 16:
                mean_conf = float(np.mean(confidences)) if confidences else 0.0
                paragraph_candidates.append((merged_text, mean_conf, char_count))

        for text, confidence in extracted:
            if len(text) < 2:
                continue
            key = _normalize_text_key(text)
            previous = line_candidates.get(key)
            if previous is None or confidence > previous[1]:
                line_candidates[key] = (text, confidence)

    if paragraph_candidates:
        paragraph_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
        best_text, best_conf, _ = paragraph_candidates[0]
        if best_conf >= 0.35:
            return "ok", best_text
        if best_conf >= 0.18:
            return "low_confidence", best_text

    if line_candidates:
        ranked = sorted(line_candidates.values(), key=lambda item: item[1], reverse=True)
        high_conf_lines = [text for text, conf in ranked if conf >= 0.45]
        low_conf_lines = [text for text, conf in ranked if 0.2 <= conf < 0.45]
        if high_conf_lines:
            return "ok", "\n".join(high_conf_lines[:10])
        if low_conf_lines:
            return "low_confidence", "\n".join(low_conf_lines[:10])

    return "unavailable", "PaddleOCR가 현재 이미지에서 읽을 수 있는 텍스트를 찾지 못했습니다."


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    try:
        import easyocr  # type: ignore
    except Exception:
        return None

    try:
        project_root = Path(__file__).resolve().parents[2]
        model_dir = project_root / ".cache" / "easyocr"
        model_dir.mkdir(parents=True, exist_ok=True)
        return easyocr.Reader(
            ["en", "ko"],
            gpu=False,
            verbose=False,
            model_storage_directory=str(model_dir),
            user_network_directory=str(model_dir),
        )
    except Exception:
        return None


def _run_easyocr(image: np.ndarray) -> tuple[str, str]:
    reader = _get_easyocr_reader()
    if reader is None:
        return "unavailable", "EasyOCR를 사용할 수 없습니다."

    variants = _generate_ocr_variants(image)
    paragraph_candidates: list[tuple[str, float, int]] = []

    for variant in variants:
        try:
            paragraph_results = reader.readtext(
                variant,
                detail=1,
                paragraph=True,
                decoder="greedy",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                width_ths=0.7,
                y_ths=0.35,
                mag_ratio=1.8,
            )
        except Exception:
            continue

        if not paragraph_results:
            continue

        texts: list[str] = []
        confidences: list[float] = []
        for item in paragraph_results:
            if len(item) < 3:
                continue
            text = _clean_ocr_text(str(item[1]))
            confidence = float(item[2])
            if len(text) < 8:
                continue
            texts.append(text)
            confidences.append(confidence)

        if not texts:
            continue

        merged_text = _clean_ocr_text("\n".join(texts))
        char_count = len(merged_text.replace("\n", " ").strip())
        if char_count < 16:
            continue
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        paragraph_candidates.append((merged_text, mean_conf, char_count))

    if paragraph_candidates:
        paragraph_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
        best_text, best_conf, _ = paragraph_candidates[0]
        if best_conf >= 0.08:
            return "ok", best_text
        return "low_confidence", best_text

    collected: dict[str, tuple[str, float]] = {}

    for variant in variants:
        try:
            results = reader.readtext(variant, detail=1, paragraph=False)
        except Exception:
            continue

        for item in results:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            confidence = float(item[2])
            if len(text) < 2:
                continue
            key = _normalize_text_key(text)
            previous = collected.get(key)
            if previous is None or confidence > previous[1]:
                collected[key] = (text, confidence)

    if not collected:
        return "unavailable", "EasyOCR가 현재 이미지에서 읽을 수 있는 텍스트를 찾지 못했습니다."

    ranked = sorted(collected.values(), key=lambda item: item[1], reverse=True)
    high_conf_lines = [text for text, conf in ranked if conf >= 0.18]
    low_conf_lines = [text for text, conf in ranked if 0.04 <= conf < 0.18]

    if high_conf_lines:
        return "ok", "\n".join(high_conf_lines[:8])
    if low_conf_lines:
        return "low_confidence", "\n".join(low_conf_lines[:8])
    return "unavailable", "EasyOCR 결과가 있었지만 신뢰도 기준을 통과한 텍스트가 없습니다."


def _run_tesseract(image: np.ndarray) -> tuple[str, str]:
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return "unavailable", "Tesseract OCR가 설치되어 있지 않아 텍스트 추출을 건너뛰었습니다."

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "ocr_input.png"
        processed = _preprocess_for_ocr(image)
        cv2.imwrite(str(image_path), processed)

        commands = [
            [tesseract_path, str(image_path), "stdout", "-l", "eng+kor"],
            [tesseract_path, str(image_path), "stdout", "-l", "eng"],
        ]
        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            except Exception:
                continue
            text = result.stdout.strip()
            if result.returncode == 0 and text:
                return "ok", text

    return "unavailable", "OCR 엔진은 감지됐지만 현재 이미지에서 읽을 수 있는 텍스트를 찾지 못했습니다."


def _translate_keywords_to_korean(raw_text: str) -> str:
    replacements = {
        "building": "빌딩",
        "university": "대학교",
        "law": "법학",
        "street": "거리",
        "st": "번지",
        "auditorium": "강당",
        "museum": "박물관",
        "library": "도서관",
        "office": "사무실",
        "floor": "층",
        "room": "호실",
        "entrance": "입구",
        "exit": "출구",
        "freedom trail": "프리덤 트레일",
        "revolution": "혁명",
    }
    text = raw_text
    for source, target in replacements.items():
        text = text.replace(source.title(), target)
        text = text.replace(source.upper(), target)
        text = text.replace(source, target)
    return text


def _build_korean_interpretation(text_status: str, raw_text: str) -> str:
    if text_status not in {"ok", "low_confidence"}:
        return raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "텍스트는 감지됐지만 정리할 내용이 부족합니다."

    translated_preview = " / ".join(_translate_keywords_to_korean(line) for line in lines[:3])
    if len(lines) == 1 and len(lines[0].split()) <= 8:
        return f"감지된 텍스트의 한국어 해석: {translated_preview}"
    if len(lines) <= 4:
        prefix = "주요 문구를 한국어 기준으로 정리하면"
        if text_status == "low_confidence":
            prefix = "낮은 신뢰도로 읽힌 주요 문구를 한국어 기준으로 정리하면"
        return f"{prefix}: {translated_preview}"
    return f"영문 안내문/설명문이 감지되었으며 핵심 문구는 다음과 같습니다: {translated_preview}"


def _build_ocr_summary(text_status: str, raw_text: str) -> str:
    if text_status not in {"ok", "low_confidence"}:
        return raw_text

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    preview = " / ".join(lines[:2])
    if text_status == "low_confidence":
        return f"OCR가 낮은 신뢰도로 {len(lines)}개 줄의 텍스트 후보를 감지했습니다. 주요 문구: {preview}"
    return f"OCR로 {len(lines)}개 줄의 텍스트를 감지했습니다. 주요 문구: {preview}"


def _encode_image_to_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("이미지를 PNG로 인코딩하지 못했습니다.")
    payload = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def _run_openai_vision_translation(image: np.ndarray) -> tuple[str, str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "unavailable", "", "OPENAI_API_KEY가 없어 멀티모달 번역을 건너뜁니다."

    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
    prompt = (
        "You are an OCR and translation assistant. Read the English text visible in this image as accurately as possible. "
        "Return strict JSON with keys: raw_text, korean_translation, summary. "
        "raw_text should preserve line breaks naturally. "
        "korean_translation should be a natural Korean translation of the extracted text. "
        "summary should be one short Korean sentence summarizing what kind of text this is. "
        "If the text is partial or uncertain, still return the best reconstruction you can."
    )

    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_image_to_data_url(image)},
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }

    req = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        return "unavailable", "", f"멀티모달 번역 호출에 실패했습니다: {detail}"
    except Exception as exc:
        return "unavailable", "", f"멀티모달 번역 호출에 실패했습니다: {exc}"

    output_text = ""
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                output_text += content.get("text", "")

    if not output_text.strip():
        return "unavailable", "", "멀티모달 번역 결과가 비어 있습니다."

    try:
        parsed = json.loads(output_text)
    except Exception:
        return "unavailable", "", "멀티모달 번역 결과를 JSON으로 해석하지 못했습니다."

    raw_text = _clean_ocr_text(str(parsed.get("raw_text", "")))
    korean_translation = _clean_ocr_text(str(parsed.get("korean_translation", "")))
    summary = _clean_ocr_text(str(parsed.get("summary", "")))

    if not raw_text:
        return "unavailable", "", "멀티모달 번역이 읽을 수 있는 텍스트를 찾지 못했습니다."
    if not korean_translation:
        korean_translation = "한국어 번역을 생성하지 못했습니다."
    if not summary:
        summary = "멀티모달 모델이 이미지 속 텍스트를 읽고 번역했습니다."

    return "ok", raw_text, json.dumps(
        {
            "korean_translation": korean_translation,
            "summary": summary,
        },
        ensure_ascii=False,
    )


def extract_text_information(image: np.ndarray) -> dict[str, str]:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_status, openai_raw_text, openai_payload = _run_openai_vision_translation(image)
    if openai_status == "ok":
        parsed = json.loads(openai_payload)
        korean_translation = parsed.get("korean_translation", "")
        summary = parsed.get("summary", "")
        return {
            "status": "ok",
            "raw_text": openai_raw_text,
            "korean_interpretation": korean_translation,
            "summary": summary,
            "engine": "openai_vision",
        }

    # If the user configured an OpenAI key, prefer surfacing the actual
    # multimodal failure instead of silently degrading to weak local OCR.
    if openai_api_key:
        return {
            "status": "unavailable",
            "raw_text": "",
            "korean_interpretation": "",
            "summary": openai_payload or "OpenAI 멀티모달 텍스트 번역 경로를 사용하지 못했습니다.",
            "engine": "openai_vision",
        }

    status, raw_text = _run_paddleocr(image)
    engine = "paddleocr"
    if status not in {"ok", "low_confidence"}:
        easy_status, easy_text = _run_easyocr(image)
        if easy_status in {"ok", "low_confidence"}:
            status, raw_text = easy_status, easy_text
            engine = "easyocr"

    if status not in {"ok", "low_confidence"}:
        tesseract_status, tesseract_text = _run_tesseract(image)
        if tesseract_status == "ok":
            status, raw_text = tesseract_status, tesseract_text
            engine = "tesseract"

    interpretation = _build_korean_interpretation(status, raw_text)
    summary = _build_ocr_summary(status, raw_text)

    return {
        "status": status,
        "raw_text": raw_text,
        "korean_interpretation": interpretation,
        "summary": summary if status in {"ok", "low_confidence"} else openai_payload or summary,
        "engine": engine,
    }
