import streamlit as st
from typing import List, Dict, Any, Optional
from nexusrag.generation.answer import Citation
from .flowchart import render_flowchart_html


def render_header():
    """Renders the executive NexusRAG application banner."""
    st.markdown(
        """
        <div class="nexus-hero-banner">
            <div>
                <div class="nexus-hero-title">⚡ NexusRAG Platform</div>
                <div class="nexus-hero-subtitle">
                    Full RAG Lifecycle: Ingestion ➔ Parsing ➔ Cleaning ➔ Recursive Chunking ➔ Embeddings ➔ Hybrid Search ➔ Reranking ➔ Grounded Chat.
                </div>
            </div>
            <div class="system-status-badge">
                <span style="font-size: 0.9rem;">●</span> Engine: Hybrid + Groq LLM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_active_doc_bar(doc_name: str, pages: int, chunks: int):
    """Renders a sleek top bar showing the currently loaded document."""
    st.markdown(
        f"""
        <div class="active-doc-bar">
            <div>
                <div class="active-doc-title">
                    <span>📄</span> {doc_name}
                </div>
                <div class="active-doc-meta" style="margin-top: 4px;">
                    <span class="meta-tag">📑 {pages} Page(s)</span>
                    <span class="meta-tag">🧩 {chunks} Chunks</span>
                    <span class="meta-tag">🧬 384-d ChromaDB</span>
                    <span class="meta-tag">⚡ BM25 Indexed</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; font-weight: 700; color: #10b981; display: flex; align-items: center; gap: 6px;">
                <span>●</span> READY FOR CHAT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chunk_card(chunk: Dict[str, Any], show_metadata: bool = True):
    """Renders a formatted chunk inspection card with provenance details."""
    cid = chunk.get("chunk_id", "N/A")
    meta = chunk.get("metadata", {})
    page = meta.get("page", 1)
    doc_name = meta.get("filename", meta.get("source", "Document"))
    tokens = meta.get("token_estimate", len(chunk.get("text", "")) // 4)
    char_count = meta.get("char_count", len(chunk.get("text", "")))

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
            <div style="margin-bottom: 8px;">
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">{doc_name}</span>
                <span style="background: rgba(129, 140, 248, 0.15); color: #a5b4fc; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">Page {page}</span>
                <span style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; font-weight: 600; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">ID: {cid}</span>
                <span style="float: right; font-size: 0.75rem; color: #94a3b8;">{char_count} chars (~{tokens} tokens)</span>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.5; color: #e2e8f0; font-family: monospace; white-space: pre-wrap; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">{chunk.get("text", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_metadata and meta:
        with st.expander(f"Metadata Details: {cid}", expanded=False):
            st.json(meta)


def render_citations(citations: List[Citation]):
    """Renders structured citation sources with document provenance."""
    if not citations:
        return

    st.markdown("<div style='margin-top: 10px; font-weight: 600; font-size: 0.85rem; color: #94a3b8;'>📚 Verified Document Sources:</div>", unsafe_allow_html=True)
    c_cols = st.columns(min(len(citations), 3))
    for i, cit in enumerate(citations[:3]):
        with c_cols[i]:
            with st.expander(f"Source #{i+1}: Page {cit.page_number} ({int(cit.relevance_score * 100)}% Match)", expanded=False):
                st.caption(f"**Document:** `{cit.source_document}`")
                st.caption(f"**Chunk ID:** `{cit.chunk_id}`")
                st.markdown(f"> *\"{cit.excerpt}\"*")
