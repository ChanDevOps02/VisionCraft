from __future__ import annotations


def _html_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th style='text-align:left;padding:6px 10px;border-bottom:1px solid #333;width:34%;'>{label}</th>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #333;'>{value}</td></tr>"
        for label, value in rows
    )
    return (
        "<table style='width:100%;border-collapse:collapse;margin:6px 0 14px 0;font-size:14px;'>"
        f"{body}</table>"
    )


def _multiline_html_block(title: str, lines: list[str]) -> str:
    items = "".join(f"<li style='margin:4px 0;'>{line}</li>" for line in lines)
    return (
        f"<h3 style='margin:14px 0 8px 0;'>{title}</h3>"
        f"<ul style='margin:0 0 14px 18px;padding:0;'>{items}</ul>"
    )


def build_metrics_markdown(metrics: dict, feedback: str) -> str:
    object_list = ", ".join(metrics["detected_objects"][:6]) if metrics["detected_objects"] else "None"
    applied = ", ".join(metrics.get("applied_enhancements", [])) or "None"
    reasons = metrics.get("enhancement_reasons", []) or ["자동 보정 설명이 없습니다."]
    channel_scale = metrics.get("channel_scale", {})
    channel_scale_text = (
        f"R {channel_scale.get('R', 1.0)}, G {channel_scale.get('G', 1.0)}, B {channel_scale.get('B', 1.0)}"
    )

    scene_object_rows = [
        ("Scene", f"<code>{metrics['scene']}</code>"),
        ("Scene Reason", metrics["scene_reason"]),
        ("Main Subject", f"<code>{metrics['main_subject']}</code>"),
        ("Detected Objects", f"<code>{object_list}</code>"),
        ("Detection Status", f"<code>{metrics['detection_status']}</code>"),
        ("Detection Summary", metrics["detection_summary"]),
        ("Composition Basis", f"<code>{metrics['composition_basis']}</code>"),
    ]

    quality_rows = [
        ("Brightness Score", f"<code>{metrics['brightness']} / 100</code>"),
        ("Contrast Score", f"<code>{metrics['contrast']} / 100</code>"),
        ("Blur Score", f"<code>{metrics['blur']} / 100</code>"),
        ("Edge Density Score", f"<code>{metrics['edge_density']} / 100</code>"),
        ("Composition Score", f"<code>{metrics['composition']} / 100</code>"),
        ("Overall Quality Score", f"<code>{metrics['overall']} / 100</code>"),
    ]

    color_exposure_rows = [
        ("Color Cast Score", f"<code>{metrics.get('color_cast_score', 0.0)} / 100</code>"),
        ("White Balance Shift", f"<code>{metrics.get('white_balance_shift', 'unknown')}</code>"),
        ("White Balance Scale", f"<code>{channel_scale_text}</code>"),
        ("Exposure State", f"<code>{metrics.get('exposure_state', 'unknown')}</code>"),
        ("Shadow Ratio", f"<code>{metrics.get('shadow_ratio', 0.0)}%</code>"),
        ("Highlight Ratio", f"<code>{metrics.get('highlight_ratio', 0.0)}%</code>"),
        ("Dynamic Range Score", f"<code>{metrics.get('dynamic_range_score', 0.0)} / 100</code>"),
    ]

    analysis_rows = [
        ("Tilt Status", f"<code>{metrics.get('tilt_status', 'unavailable')}</code>"),
        ("Tilt Angle", f"<code>{metrics.get('tilt_angle_deg', 0.0)} deg</code>"),
        ("Tilt State", f"<code>{metrics.get('tilt_state', 'unknown')}</code>"),
        ("Tilt Summary", metrics.get("tilt_summary", "") or "-"),
        ("Segmentation Status", f"<code>{metrics.get('segmentation_status', 'unavailable')}</code>"),
        ("Segmentation Source", f"<code>{metrics.get('segmentation_source', 'none')}</code>"),
        ("Segmentation Summary", metrics.get("segmentation_summary", "")),
        ("Crop Suggestion Status", f"<code>{metrics.get('crop_status', 'unavailable')}</code>"),
        ("Crop Suggestion Summary", metrics.get("crop_summary", "")),
        ("Difference Heatmap Status", f"<code>{metrics.get('difference_status', 'unavailable')}</code>"),
        ("Difference Heatmap Summary", metrics.get("difference_summary", "")),
        ("ORB Matching Status", f"<code>{metrics.get('orb_status', 'unavailable')}</code>"),
        ("ORB Match Count", f"<code>{metrics.get('orb_match_count', 0)}</code>"),
        ("ORB Matching Quality", f"<code>{metrics.get('orb_matching_quality', 'unknown')}</code>"),
        ("ORB Matching Summary", metrics.get("orb_summary", "")),
        ("OCR Status", f"<code>{metrics.get('ocr_status', 'unavailable')}</code>"),
        ("OCR Engine", f"<code>{metrics.get('ocr_engine', 'none')}</code>"),
        ("OCR Summary", metrics.get("ocr_summary", "") or "-"),
        ("Korean Interpretation", metrics.get("ocr_interpretation", "") or "-"),
    ]

    html = [
        "<h2 style='margin:0 0 10px 0;'>Analysis Result</h2>",
        "<h3 style='margin:14px 0 8px 0;'>Scene & Objects</h3>",
        _html_table(scene_object_rows),
        "<h3 style='margin:14px 0 8px 0;'>Quality Scores</h3>",
        _html_table(quality_rows),
        "<h3 style='margin:14px 0 8px 0;'>Color & Exposure</h3>",
        _html_table(color_exposure_rows),
        "<h3 style='margin:14px 0 8px 0;'>Additional Analysis</h3>",
        _html_table(analysis_rows),
        "<h3 style='margin:14px 0 8px 0;'>Applied Enhancements</h3>",
        f"<p style='margin:0 0 12px 0;'><code>{applied}</code></p>",
        _multiline_html_block("Enhancement Rationale", reasons),
        "<h2 style='margin:18px 0 10px 0;'>AI Feedback</h2>",
        f"<p style='margin:0;line-height:1.8;'>{feedback}</p>",
    ]
    return "".join(html)
