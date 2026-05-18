from __future__ import annotations


def build_metrics_markdown(metrics: dict, feedback: str) -> str:
    object_list = ", ".join(metrics["detected_objects"][:6]) if metrics["detected_objects"] else "None"

    return f"""
## Analysis Result

- Scene: `{metrics["scene"]}`
- Scene Reason: {metrics["scene_reason"]}
- Brightness Score: `{metrics["brightness"]} / 100`
- Contrast Score: `{metrics["contrast"]} / 100`
- Blur Score: `{metrics["blur"]} / 100`
- Edge Density Score: `{metrics["edge_density"]} / 100`
- Composition Score: `{metrics["composition"]} / 100`
- Overall Quality Score: `{metrics["overall"]} / 100`
- Main Subject: `{metrics["main_subject"]}`
- Detected Objects: `{object_list}`
- Detection Status: `{metrics["detection_status"]}`
- Detection Summary: {metrics["detection_summary"]}
- Composition Basis: `{metrics["composition_basis"]}`

## AI Feedback

{feedback}
"""
