"""
UI module for NexusRAG.
Provides reusable Streamlit components, custom CSS, and visual RAG lifecycle widgets.
"""

from .components import (
    render_header,
    render_active_doc_bar,
    render_chunk_card,
    render_citations,
)
from .flowchart import render_flowchart_html
from .styles import get_custom_css

__all__ = [
    "render_header",
    "render_active_doc_bar",
    "render_chunk_card",
    "render_citations",
    "render_flowchart_html",
    "get_custom_css",
]
