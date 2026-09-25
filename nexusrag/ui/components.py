import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Any, Optional
from nexusrag.generation.answer import Citation


def render_header():
    """Renders the NexusRAG application header banner."""
    st.markdown(
        """
        <div class="nexus-header">
            <div class="nexus-title">⚡ NexusRAG — RAG Lifecycle Platform</div>
            <div class="nexus-subtitle">
                An interactive RAG engineering laboratory demonstrating the full lifecycle: 
                Ingestion ➔ Cleaning ➔ Chunking ➔ Embeddings ➔ Hybrid Retrieval ➔ Reranking ➔ Grounded Generation ➔ Evaluation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lifecycle_pipeline(stages: List[Dict[str, Any]]):
    """
    Renders live RAG ingestion lifecycle stage cards.
    Each item in stages is {'name', 'icon', 'status', 'detail'}
    status: 'pending', 'running', 'complete', 'error'
    """
    for stage in stages:
        status = stage.get("status", "pending")
        name = stage.get("name", "")
        detail = stage.get("detail", "")
        icon = stage.get("icon", "•")

        if status == "complete":
            badge = "✅"
            color_border = "#10b981"
        elif status == "running":
            badge = "🔄"
            color_border = "#3b82f6"
        elif status == "error":
            badge = "❌"
            color_border = "#ef4444"
        else:
            badge = "⏳"
            color_border = "#94a3b8"

        st.markdown(
            f"""
            <div style="border-left: 4px solid {color_border}; background: rgba(30, 41, 59, 0.05); padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 0.95rem;">{badge} {icon} {name}</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chunk_card(chunk: Dict[str, Any], show_metadata: bool = True):
    """
    Renders a formatted chunk inspection card with provenance details.
    """
    cid = chunk.get("chunk_id", "N/A")
    meta = chunk.get("metadata", {})
    page = meta.get("page", 1)
    doc_name = meta.get("filename", meta.get("source", "Document"))
    strategy = meta.get("strategy", "recursive")
    tokens = meta.get("token_estimate", len(chunk.get("text", "")) // 4)
    char_count = meta.get("char_count", len(chunk.get("text", "")))

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.03); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="margin-bottom: 8px;">
                <span style="background: #e0f2fe; color: #0369a1; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">{doc_name}</span>
                <span style="background: #ede9fe; color: #6d28d9; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">Page {page}</span>
                <span style="background: #f1f5f9; color: #475569; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">ID: {cid}</span>
                <span style="float: right; font-size: 0.75rem; color: #64748b;">{char_count} chars (~{tokens} tokens)</span>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: #334155; font-family: monospace; white-space: pre-wrap; background: rgba(0,0,0,0.02); padding: 8px; border-radius: 4px;">{chunk.get("text", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_metadata and meta:
        with st.expander(f"Inspect Metadata: {cid}", expanded=False):
            st.json(meta)


def render_citations(citations: List[Citation]):
    """
    Renders structured citation sources with document provenance.
    """
    if not citations:
        return

    st.markdown("#### 📚 Verified Source Citations")
    for i, cit in enumerate(citations, start=1):
        with st.expander(
            f"Source #{i}: {cit.source_document} — Page {cit.page_number} (Relevance: {cit.relevance_score:.2f})",
            expanded=False,
        ):
            st.markdown(f"**Chunk Identifier:** `{cit.chunk_id}`")
            st.markdown(f"**Relevance Confidence:** `{cit.relevance_score:.4f}`")
            st.markdown("**Verbatim Passage:**")
            st.info(cit.excerpt)


def render_evaluation_chart(df: pd.DataFrame):
    """
    Renders an interactive Altair grouped bar chart comparing RAG strategies.
    """
    melted = df.melt(
        id_vars=["Strategy"],
        value_vars=["Context Precision", "Context Recall", "Faithfulness", "Answer Relevance"],
        var_name="Metric",
        value_name="Score",
    )

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", axis=alt.Axis(title=None, labelAngle=-20)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1.0]), axis=alt.Axis(title="Benchmark Score")),
            color=alt.Color("Strategy:N", scale=alt.Scale(scheme="category10")),
            xOffset="Strategy:N",
            tooltip=["Strategy", "Metric", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=340)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def render_pipeline_diagram():
    """
    Renders an ASCII / styled diagram of the complete RAG lifecycle flow.
    """
    st.markdown(
        """
        ```text
           [ User Query ]
                 │
                 ▼
        ┌──────────────────┐
        │  Query Rewriting │ (Optional: Expands synonyms & entities)
        └────────┬─────────┘
                 │
        ┌────────┴─────────────────────────┐
        ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│  Semantic Search │              │   Keyword Search │
│ (Dense Embeddings│              │   (Sparse BM25)  │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
         └───────────────┬─────────────────┘
                         ▼
             ┌─────────────────────────┐
             │ Reciprocal Rank Fusion  │ (Merges & deduplicates)
             └───────────┬─────────────┘
                         ▼
             ┌─────────────────────────┐
             │  Cross-Encoder Rerank   │ (Optional: FlashRank score)
             └───────────┬─────────────┘
                         ▼
             ┌─────────────────────────┐
             │    Context Selection    │ (Top K chunks)
             └───────────┬─────────────┘
                         ▼
             ┌─────────────────────────┐
             │ Grounded LLM Generation │ (Grounded prompt + streaming)
             └───────────┬─────────────┘
                         ▼
             [ Verified Answer & Citations ]
        ```
        """
    )
