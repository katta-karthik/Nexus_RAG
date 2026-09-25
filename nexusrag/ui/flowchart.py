from typing import List, Dict, Any


def render_flowchart_html(stages: List[Dict[str, Any]], current_stage_idx: int = -1) -> str:
    """
    Renders an executive, modern visual RAG lifecycle flowchart with stage nodes,
    status badges, live metrics, and directional connectors.
    Zero-whitespace formatting prevents Markdown from rendering HTML as a code block.
    """
    parts = ['<div class="rag-flowchart">']

    for i, s in enumerate(stages):
        status = s.get("status", "pending")
        icon = s.get("icon", "•")
        title = s.get("title", f"Step {i+1}")
        metric = s.get("metric", "")
        status_text = "READY" if status == "completed" else ("RUNNING" if status == "running" else "WAITING")
        pill_class = f"pill-{status}"

        parts.append(
            f'<div class="flow-node {status}">'
            f'<div class="flow-step-num">STAGE 0{i+1}</div>'
            f'<div class="flow-icon">{icon}</div>'
            f'<div class="flow-title">{title}</div>'
            f'<div class="flow-metric">{metric}</div>'
            f'<div class="flow-status-pill {pill_class}">{status_text}</div>'
            f'</div>'
        )

        if i < len(stages) - 1:
            arrow_color = "#10b981" if status == "completed" and stages[i+1].get("status") in ["running", "completed"] else "#475569"
            parts.append(f'<div class="flow-arrow" style="color: {arrow_color};">➔</div>')

    parts.append('</div>')
    return "".join(parts)
