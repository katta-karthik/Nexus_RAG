"""
UI module for NexusRAG.
Provides reusable Streamlit components, custom CSS, and visual RAG lifecycle widgets.
"""

from .components import (
    render_lifecycle_pipeline,
    render_chunk_card,
    render_citations,
    render_evaluation_chart,
)
from .styles import get_custom_css

__all__ = [
    "render_lifecycle_pipeline",
    "render_chunk_card",
    "render_citations",
    "render_evaluation_chart",
    "get_custom_css",
]
