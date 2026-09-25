from typing import List, Dict, Any


def render_flowchart_html(stages: List[Dict[str, Any]], current_stage_idx: int = -1) -> str:
    """
    Renders an executive, modern visual RAG lifecycle flowchart with stage nodes,
    status badges, live metrics, and directional connectors.
    """
    html_parts = [
        """
        <style>
        .rag-flowchart {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: stretch;
            justify-content: space-between;
            margin: 20px 0 24px 0;
            padding: 16px;
            background: linear-gradient(145deg, #0b1120 0%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-radius: 14px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .flow-node {
            flex: 1 1 125px;
            min-width: 120px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
        }
        .flow-node.pending {
            opacity: 0.45;
            border-color: #334155;
            background: #0f172a;
        }
        .flow-node.running {
            opacity: 1;
            border-color: #38bdf8;
            background: linear-gradient(180deg, rgba(56, 189, 248, 0.12) 0%, #1e293b 100%);
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
            animation: pulse-glow 1.5s infinite;
        }
        .flow-node.completed {
            opacity: 1;
            border-color: #10b981;
            background: linear-gradient(180deg, rgba(16, 185, 129, 0.1) 0%, #1e293b 100%);
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        }
        .flow-step-num {
            font-size: 0.7rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .flow-icon {
            font-size: 1.5rem;
            margin-bottom: 6px;
        }
        .flow-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 4px;
            line-height: 1.2;
        }
        .flow-metric {
            font-size: 0.72rem;
            color: #38bdf8;
            font-weight: 600;
            margin-top: auto;
            padding-top: 4px;
        }
        .flow-status-pill {
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 999px;
            margin-top: 6px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .pill-pending {
            background: rgba(148, 163, 184, 0.15);
            color: #94a3b8;
        }
        .pill-running {
            background: rgba(56, 189, 248, 0.2);
            color: #38bdf8;
        }
        .pill-completed {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }
        .flow-arrow {
            display: flex;
            align-items: center;
            color: #475569;
            font-weight: bold;
            font-size: 1.1rem;
        }
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 4px rgba(56, 189, 248, 0.2); }
            50% { box-shadow: 0 0 16px rgba(56, 189, 248, 0.6); }
            100% { box-shadow: 0 0 4px rgba(56, 189, 248, 0.2); }
        }
        </style>
        <div class="rag-flowchart">
        """
    ]

    for i, s in enumerate(stages):
        status = s.get("status", "pending")
        icon = s.get("icon", "•")
        title = s.get("title", f"Step {i+1}")
        metric = s.get("metric", "")
        status_text = "READY" if status == "completed" else ("RUNNING" if status == "running" else "WAITING")
        pill_class = f"pill-{status}"

        node_html = f"""
        <div class="flow-node {status}">
            <div class="flow-step-num">STAGE 0{i+1}</div>
            <div class="flow-icon">{icon}</div>
            <div class="flow-title">{title}</div>
            <div class="flow-metric">{metric}</div>
            <div class="flow-status-pill {pill_class}">{status_text}</div>
        </div>
        """
        html_parts.append(node_html)

        if i < len(stages) - 1:
            arrow_color = "#10b981" if status == "completed" and stages[i+1].get("status") in ["running", "completed"] else "#475569"
            html_parts.append(f'<div class="flow-arrow" style="color: {arrow_color};">➔</div>')

    html_parts.append("</div>")
    return "\n".join(html_parts)
