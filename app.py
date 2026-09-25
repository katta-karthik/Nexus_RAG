import os
import sys
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure local packages are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nexusrag.ingestion.loaders import DocumentLoader, LoadedDocument
from nexusrag.ingestion.cleaner import TextCleaner
from nexusrag.ingestion.chunker import Chunker, ChunkingStrategy, Chunk
from nexusrag.ingestion.metadata import enrich_chunks
from nexusrag.indexing.embeddings import EmbeddingManager, EmbeddingProvider
from nexusrag.indexing.vectorstore import VectorStoreManager
from nexusrag.retrieval.semantic import SemanticRetriever
from nexusrag.retrieval.keyword import BM25KeywordRetriever
from nexusrag.retrieval.hybrid import HybridRetriever
from nexusrag.retrieval.query_rewrite import QueryRewriter
from nexusrag.retrieval.reranker import Reranker
from nexusrag.generation.answer import AnswerGenerator
from nexusrag.ui.styles import get_custom_css
from nexusrag.ui.components import render_chunk_card, render_citations

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="NexusRAG — Chat With Your Documents",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==========================================
# Session State Initialization
# ==========================================
def init_session_state():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = VectorStoreManager(persist_directory="./chroma_db")
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
    if "keyword_retriever" not in st.session_state:
        st.session_state.keyword_retriever = BM25KeywordRetriever()
        existing = st.session_state.vectorstore.get_chunks_for_document(limit=1000)
        if existing:
            st.session_state.keyword_retriever.index_chunks(existing)
    if "reranker" not in st.session_state:
        st.session_state.reranker = Reranker()
    if "query_rewriter" not in st.session_state:
        st.session_state.query_rewriter = QueryRewriter()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_registry" not in st.session_state:
        st.session_state.documents_registry = {}


init_session_state()

# Sync document registry with vector store if empty
vs_stats = st.session_state.vectorstore.get_stats()
if vs_stats["total_vectors"] > 0 and not st.session_state.documents_registry:
    for doc_name in vs_stats["document_names"]:
        st.session_state.documents_registry[doc_name] = {
            "filename": doc_name,
            "status": "Ready",
            "chunks_count": "In Store",
        }


# ==========================================
# Document CRUD Operations
# ==========================================
def add_document(source, filename: str):
    """
    CRUD: Create / Add Document
    Executes the complete RAG ingestion lifecycle:
    Upload -> Parse -> Clean -> Chunk -> Metadata -> Embed -> Index.
    """
    status_box = st.status(f"🚀 Ingesting `{filename}` through RAG lifecycle...", expanded=True)

    with status_box:
        # 1. Parsing
        st.write("📄 **Parsing document pages...**")
        loaded_doc = DocumentLoader.load(source, filename)
        st.write(f"✓ Extracted {loaded_doc.total_pages} page(s) ({loaded_doc.total_characters:,} characters).")
        time.sleep(0.1)

        # 2. Text Cleaning
        st.write("🧹 **Normalizing text & removing artifacts...**")
        cleaned_doc, clean_stats = TextCleaner.clean_document(loaded_doc)
        st.write(f"✓ Cleaned whitespace and line breaks ({clean_stats.chars_removed:,} characters normalized).")
        time.sleep(0.1)

        # 3. Best Chunking (Recursive 800 / 120)
        st.write("✂️ **Splitting into semantic chunks...**")
        chunks = Chunker.chunk_document(
            cleaned_doc,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=800,
            chunk_overlap=120,
        )
        st.write(f"✓ Created {len(chunks)} contextual chunks with boundary preservation.")
        time.sleep(0.1)

        # 4. Metadata Enrichment
        st.write("🏷️ **Attaching provenance metadata & hashes...**")
        enriched = enrich_chunks(chunks, doc_type=cleaned_doc.file_type)
        st.write(f"✓ Attached page numbers, IDs, and token estimates to all {len(enriched)} chunks.")
        time.sleep(0.1)

        # 5. Embeddings & Vector Indexing
        st.write("🧬 **Generating embeddings & updating vector index...**")
        prog = st.progress(0, text="Generating embeddings...")

        def on_progress(completed, total, pct):
            prog.progress(int(pct), text=f"Embedded {completed}/{total} chunks ({int(pct)}%)")

        texts = [c.text for c in enriched]
        embeddings = st.session_state.embedding_manager.embed_documents_with_progress(
            texts, batch_size=16, on_progress=on_progress
        )
        prog.empty()

        # Update Chroma and BM25 index
        st.session_state.vectorstore.add_chunks(enriched, embeddings)
        all_chunks = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
        st.session_state.keyword_retriever.index_chunks(all_chunks)

        status_box.update(
            label=f"✅ '{filename}' is ready! You can now ask questions below.",
            state="complete",
            expanded=False,
        )

    # Register in session state
    st.session_state.documents_registry[filename] = {
        "filename": filename,
        "pages": loaded_doc.total_pages,
        "characters": cleaned_doc.total_characters,
        "chunks_count": len(chunks),
        "status": "Ready",
    }


def remove_document(filename: str):
    """
    CRUD: Delete / Remove Document
    Removes a document from Chroma vector store, updates the BM25 index, and clears registry.
    """
    st.session_state.vectorstore.delete_document(filename)
    if filename in st.session_state.documents_registry:
        del st.session_state.documents_registry[filename]

    # Refresh BM25 with remaining chunks
    remaining = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
    st.session_state.keyword_retriever.index_chunks(remaining)
    st.success(f"Removed '{filename}' from knowledge base.")
    st.rerun()


def clear_all_documents():
    """
    CRUD: Delete All Documents
    Clears vector database, BM25 index, and conversation history.
    """
    st.session_state.vectorstore.clear()
    st.session_state.keyword_retriever.index_chunks([])
    st.session_state.documents_registry.clear()
    st.session_state.chat_history.clear()
    st.success("All documents and memory cleared!")
    st.rerun()


# ==========================================
# Sidebar: Document Management (CRUD)
# ==========================================
with st.sidebar:
    st.markdown("## 📂 Document Manager")

    # 1. ADD / UPLOAD (Create)
    st.markdown("### 📤 Add Document")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD:",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload resumes, reports, papers, or documentation.",
    )
    if uploaded_files:
        for ufile in uploaded_files:
            if ufile.name not in st.session_state.documents_registry:
                if st.button(f"⚡ Ingest '{ufile.name}'", key=f"ingest_{ufile.name}", type="primary", use_container_width=True):
                    add_document(ufile.getvalue(), ufile.name)
                    st.rerun()

    # 1-Click Samples
    with st.expander("✨ Or Try Sample Documents", expanded=False):
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 Paper", use_container_width=True):
                p = os.path.join(sample_dir, "attention_is_all_you_need_summary.txt")
                if os.path.exists(p):
                    add_document(p, "attention_is_all_you_need_summary.txt")
                    st.rerun()
        with c2:
            if st.button("📈 Report", use_container_width=True):
                p = os.path.join(sample_dir, "techcorp_annual_report_2025.txt")
                if os.path.exists(p):
                    add_document(p, "techcorp_annual_report_2025.txt")
                    st.rerun()

    st.markdown("---")

    # 2. READ & DELETE (List active documents with Remove button)
    st.markdown("### 📚 Active Documents")
    active_docs = list(st.session_state.documents_registry.keys())

    if not active_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc_name in active_docs:
            d_info = st.session_state.documents_registry[doc_name]
            pages_str = f"{d_info.get('pages', 1)} pg"
            chunks_str = f"{d_info.get('chunks_count', 'N/A')} chunks"

            col_name, col_del = st.columns([3, 1])
            with col_name:
                st.markdown(f"**📄 {doc_name}**")
                st.caption(f"{pages_str} • {chunks_str}")
            with col_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Remove {doc_name}"):
                    remove_document(doc_name)

        st.markdown("---")
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            clear_all_documents()

    # 3. Groq API Key
    st.markdown("---")
    with st.expander("🔑 LLM Settings", expanded=False):
        groq_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="High-speed natural language reasoning",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key


# ==========================================
# Main Screen: Natural Language Interaction
# ==========================================
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; color: #f8fafc;">
        <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ⚡ NexusRAG — Document Intelligence
        </div>
        <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 4px;">
            Upload your documents, ask anything in plain natural language, and get direct, grounded answers backed by full RAG lifecycle retrieval.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_chunks = st.tabs(["💬 Chat with Documents", "🔎 Inspect Document Chunks"])


# ==========================================
# TAB 1: NATURAL LANGUAGE CHAT
# ==========================================
with tab_chat:
    active_docs = list(st.session_state.documents_registry.keys())

    if not active_docs:
        st.info("👈 **Get started:** Upload a document using the left sidebar (or click a sample document) to begin asking questions.")
    else:
        st.caption(f"🟢 **Ready to chat** with {len(active_docs)} document(s): {', '.join([f'`{d}`' for d in active_docs])}")

        # Quick Suggested Prompts
        st.markdown("💡 **Suggestions:**")
        s_cols = st.columns(3)
        sample_prompts = [
            "What are the main findings or summary?",
            "What is his education, CGPA, and background?",
            "What projects or experience are mentioned?",
        ]
        chosen_prompt = None
        for i, p_text in enumerate(sample_prompts):
            with s_cols[i]:
                if st.button(p_text, key=f"quick_{i}", use_container_width=True):
                    chosen_prompt = p_text

        # Render Conversation
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    render_citations(msg["citations"])
                if msg.get("trace"):
                    with st.expander("🔬 How RAG Found This Answer (Lifecycle Trace)", expanded=False):
                        tr = msg["trace"]
                        st.markdown(f"**Query Rewritten for Search:** `{tr.get('rewritten_query')}`")
                        st.markdown(f"**Hybrid Search:** Found `{tr.get('candidate_count')}` candidates (Dense Vectors + BM25 Lexical)")
                        st.markdown(f"**Cross-Encoder Rerank:** Filtered down to top `{tr.get('top_count')}` chunks for answer generation")

        # Chat Input
        user_input = st.chat_input("Ask any question about your document in natural language...")
        query_to_run = user_input or chosen_prompt

        if query_to_run:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            # Assistant response
            with st.chat_message("assistant"):
                status_box = st.status("🔍 Searching document evidence...", expanded=True)

                with status_box:
                    # Retrievers
                    semantic_retriever = SemanticRetriever(
                        st.session_state.vectorstore, st.session_state.embedding_manager
                    )
                    keyword_retriever = st.session_state.keyword_retriever
                    hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
                    query_rewriter = st.session_state.query_rewriter
                    reranker = st.session_state.reranker

                    # 1. Query Rewrite
                    rewritten = query_rewriter.rewrite(query_to_run)
                    st.write(f"✓ **Analyzed Query:** `{rewritten}`")

                    # 2. Hybrid Retrieval (Semantic + BM25)
                    hybrid_res = hybrid_retriever.retrieve(rewritten, top_k=8, candidate_pool_size=12)
                    st.write(
                        f"✓ **Hybrid Search:** Retrieved `{hybrid_res.merged_count}` candidates "
                        f"({hybrid_res.semantic_count} dense + {hybrid_res.keyword_count} sparse)."
                    )

                    # 3. Cross-Encoder Reranking
                    rerank_res = reranker.rerank(rewritten, hybrid_res.merged_candidates, top_n=4)
                    selected_chunks = rerank_res.after_reranking
                    st.write(f"✓ **Cross-Encoder Rerank:** Verified top `{len(selected_chunks)}` most relevant chunks.")

                    status_box.update(label="✅ Evidence verified. Generating natural language answer...", state="complete")

                # 4. Stream Natural Language Answer
                generator = AnswerGenerator(provider="groq", temperature=0.2)
                stream_gen = generator.stream_answer(
                    query_to_run, selected_chunks, st.session_state.chat_history[:-1]
                )

                full_answer = st.write_stream(stream_gen)

                # 5. Verified Citations
                citations = generator.extract_citations(selected_chunks)
                render_citations(citations)

                trace_data = {
                    "rewritten_query": rewritten,
                    "candidate_count": hybrid_res.merged_count,
                    "top_count": len(selected_chunks),
                }

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations,
                        "trace": trace_data,
                    }
                )


# ==========================================
# TAB 2: INSPECT DOCUMENT CHUNKS (READ CRUD)
# ==========================================
with tab_chunks:
    st.markdown("### 🔎 Inspect Ingested Chunks")
    st.caption("Browse how your documents were parsed and split into chunks with page numbers and metadata.")

    doc_list = list(st.session_state.documents_registry.keys())
    if doc_list:
        chosen_doc = st.selectbox("Select Document to Inspect:", options=doc_list)
        if chosen_doc:
            chunks_to_view = st.session_state.vectorstore.get_chunks_for_document(
                filename=chosen_doc, limit=30
            )
            st.caption(f"Showing chunks for `{chosen_doc}` (Total indexed: {len(chunks_to_view)}):")
            for ch in chunks_to_view:
                render_chunk_card(ch)
    else:
        st.info("No documents uploaded yet.")
