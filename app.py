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
from nexusrag.evaluation.dataset import load_benchmark_samples
from nexusrag.evaluation.evaluate import RAGEvaluator
from nexusrag.ui.styles import get_custom_css
from nexusrag.ui.components import (
    render_header,
    render_chunk_card,
    render_citations,
    render_evaluation_chart,
)

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="NexusRAG — Document Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# ==========================================
# Session State Initialization
# ==========================================
def init_session_state():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = VectorStoreManager(persist_directory="./chroma_db")
    if "embedding_manager" not in st.session_state:
        default_provider = (
            EmbeddingProvider.GEMINI
            if os.getenv("GOOGLE_API_KEY")
            else (EmbeddingProvider.OPENAI if os.getenv("OPENAI_API_KEY") else EmbeddingProvider.LOCAL)
        )
        st.session_state.embedding_manager = EmbeddingManager(provider=default_provider)
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
    if "active_document" not in st.session_state:
        st.session_state.active_document = None
    if "eval_results_df" not in st.session_state:
        st.session_state.eval_results_df = None


init_session_state()

# Sync document registry with vector store if empty
vs_stats = st.session_state.vectorstore.get_stats()
if vs_stats["total_vectors"] > 0 and not st.session_state.documents_registry:
    for doc_name in vs_stats["document_names"]:
        st.session_state.documents_registry[doc_name] = {
            "filename": doc_name,
            "status": "Indexed",
            "chunks_count": "In Store",
        }
    if not st.session_state.active_document and vs_stats["document_names"]:
        st.session_state.active_document = vs_stats["document_names"][0]


# ==========================================
# Ingestion Processor (Uses Best Pre-configured Options)
# ==========================================
def ingest_document(source, filename: str):
    """
    Ingests document with the best production settings:
    - Recursive Character Splitting (800 chars, 120 overlap)
    - Full text cleaning & normalization
    - Provenance metadata enrichment
    - Dense Chroma vector indexing + Sparse BM25 indexing
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

        # 4. Metadata
        st.write("🏷️ **Enriching chunk provenance & hashes...**")
        enriched = enrich_chunks(chunks, doc_type=cleaned_doc.file_type)
        st.write(f"✓ Attached page numbers, IDs, and token estimates to all {len(enriched)} chunks.")
        time.sleep(0.1)

        # 5. Embeddings & Indexing
        st.write("🧬 **Generating embeddings & updating index...**")
        prog = st.progress(0, text="Generating embeddings...")

        def on_progress(completed, total, pct):
            prog.progress(int(pct), text=f"Embedded {completed}/{total} chunks ({int(pct)}%)")

        texts = [c.text for c in enriched]
        embeddings = st.session_state.embedding_manager.embed_documents_with_progress(
            texts, batch_size=16, on_progress=on_progress
        )
        prog.empty()

        # Update Chroma and BM25
        st.session_state.vectorstore.add_chunks(enriched, embeddings)
        all_chunks = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
        st.session_state.keyword_retriever.index_chunks(all_chunks)

        status_box.update(
            label=f"✅ '{filename}' is ready! You can now ask questions below.",
            state="complete",
            expanded=False,
        )

    # Update session registry
    st.session_state.documents_registry[filename] = {
        "filename": filename,
        "pages": loaded_doc.total_pages,
        "characters": cleaned_doc.total_characters,
        "chunks_count": len(chunks),
        "status": "Ready",
    }
    st.session_state.active_document = filename


# ==========================================
# Sidebar: Settings & Document Management
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ NexusRAG Controls")

    # API Key Configuration
    st.markdown("#### 🔑 Model & API Key (Optional)")
    provider_choice = st.selectbox(
        "Provider",
        options=["Groq (Blazing Fast)", "Google Gemini", "OpenAI", "Local Extractive (Offline Demo)"],
        index=0 if os.getenv("GROQ_API_KEY") else (1 if os.getenv("GOOGLE_API_KEY") else (2 if os.getenv("OPENAI_API_KEY") else 3)),
    )

    if "Groq" in provider_choice:
        groq_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="High-speed inference on Groq",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
    elif "Gemini" in provider_choice:
        gemini_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Free key at https://aistudio.google.com/",
        )
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
            st.session_state.embedding_manager = EmbeddingManager(
                provider=EmbeddingProvider.GEMINI, api_key=gemini_key
            )
    elif "OpenAI" in provider_choice:
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Key from https://platform.openai.com/",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.session_state.embedding_manager = EmbeddingManager(
                provider=EmbeddingProvider.OPENAI, api_key=openai_key
            )
    else:
        st.session_state.embedding_manager = EmbeddingManager(provider=EmbeddingProvider.LOCAL)

    st.markdown("---")
    st.markdown("#### 📚 Ingested Knowledge Base")
    cur_stats = st.session_state.vectorstore.get_stats()
    st.write(f"**Total Documents:** `{cur_stats['unique_documents']}`")
    st.write(f"**Total Vectors:** `{cur_stats['total_vectors']}`")

    if cur_stats["document_names"]:
        for dname in cur_stats["document_names"]:
            st.markdown(f"• 📄 `{dname}`")

    st.markdown("---")
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        st.session_state.vectorstore.clear()
        st.session_state.keyword_retriever.index_chunks([])
        st.session_state.documents_registry.clear()
        st.session_state.chat_history.clear()
        st.session_state.active_document = None
        st.success("Knowledge base cleared!")
        st.rerun()


# ==========================================
# Main Header
# ==========================================
render_header()

# ==========================================
# Top Section: Document Upload & Sample Loader
# ==========================================
st.markdown("### 📤 Step 1: Upload Your Document")

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, or Markdown document to chat with:",
        type=["pdf", "txt", "md"],
        help="Upload your personal report, paper, or notes. The file will be parsed and indexed automatically.",
    )
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.documents_registry:
            if st.button(f"⚡ Ingest & Index '{uploaded_file.name}'", type="primary"):
                ingest_document(uploaded_file.getvalue(), uploaded_file.name)
                st.rerun()

with col_sample:
    st.markdown("**Or load a sample document in 1-click:**")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("📄 Attention Paper", use_container_width=True):
            sample_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "sample_docs",
                "attention_is_all_you_need_summary.txt",
            )
            if os.path.exists(sample_path):
                ingest_document(sample_path, "attention_is_all_you_need_summary.txt")
                st.rerun()
    with sample_col2:
        if st.button("📈 Annual Report", use_container_width=True):
            sample_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "sample_docs",
                "techcorp_annual_report_2025.txt",
            )
            if os.path.exists(sample_path):
                ingest_document(sample_path, "techcorp_annual_report_2025.txt")
                st.rerun()

st.markdown("---")

# ==========================================
# Main Interaction Tabs
# ==========================================
tab_chat, tab_explorer, tab_eval = st.tabs(
    [
        "💬 Chat with Your Documents",
        "🔎 Document & Chunk Explorer",
        "📊 Benchmark & Evaluation",
    ]
)


# ==========================================
# TAB 1: CHAT WITH DOCUMENTS (PRIMARY USER EXPERIENCE)
# ==========================================
with tab_chat:
    active_docs = list(st.session_state.documents_registry.keys())

    if not active_docs:
        st.info("👋 **Welcome!** Please upload a document above or click one of the sample buttons to start chatting.")
    else:
        st.markdown(
            f"**Active Knowledge Base:** `{len(active_docs)} document(s)` ready "
            f"({', '.join([f'📄 {d}' for d in active_docs[:3]])})"
        )

        # Quick Suggested Questions for instant gratification
        st.markdown("💡 **Try asking:**")
        sug_cols = st.columns(3)
        sample_prompts = [
            "What are the primary conclusions or findings?",
            "What methodology or approach was used?",
            "What are the major limitations or risks identified?",
        ]
        clicked_prompt = None
        for i, prompt in enumerate(sample_prompts):
            with sug_cols[i]:
                if st.button(prompt, key=f"sug_{i}", use_container_width=True):
                    clicked_prompt = prompt

        # Display Conversation History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    render_citations(msg["citations"])
                # Show inspection details if available
                if msg.get("debug_info"):
                    with st.expander("🔬 See How RAG Found This Answer", expanded=False):
                        dbg = msg["debug_info"]
                        st.markdown(f"**Query Rewritten for Search:** `{dbg.get('rewritten_query')}`")
                        st.markdown(
                            f"**Candidates Retrieved:** `{dbg.get('candidate_count')}` "
                            f"(Merged from Dense Semantic + Sparse BM25 via RRF)"
                        )
                        st.markdown(f"**Top Chunks Used:** `{dbg.get('top_chunks_count')}` after Cross-Encoder Reranking")

        # Chat Input
        user_input = st.chat_input("Ask a question about your uploaded document...")
        query_to_run = user_input or clicked_prompt

        if query_to_run:
            # 1. Record user message
            st.session_state.chat_history.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            # 2. Assistant execution
            with st.chat_message("assistant"):
                status_widget = st.status("🔍 Searching documents & reranking context...", expanded=True)

                with status_widget:
                    # Best Retrieval Setup
                    semantic_retriever = SemanticRetriever(
                        st.session_state.vectorstore, st.session_state.embedding_manager
                    )
                    keyword_retriever = st.session_state.keyword_retriever
                    hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
                    query_rewriter = st.session_state.query_rewriter
                    reranker = st.session_state.reranker

                    # Stage A: Query Rewriting
                    rewritten = query_rewriter.rewrite(query_to_run)
                    st.write(f"✓ **Optimized Query:** `{rewritten}`")

                    # Stage B: Hybrid Retrieval (Dense Semantic + Sparse BM25)
                    hybrid_res = hybrid_retriever.retrieve(rewritten, top_k=8, candidate_pool_size=12)
                    st.write(
                        f"✓ **Hybrid Search:** Retrieved `{hybrid_res.merged_count}` candidates "
                        f"({hybrid_res.semantic_count} semantic + {hybrid_res.keyword_count} keyword)."
                    )

                    # Stage C: Cross-Encoder Reranking
                    rerank_res = reranker.rerank(rewritten, hybrid_res.merged_candidates, top_n=4)
                    selected_chunks = rerank_res.after_reranking
                    st.write(f"✓ **Reranked:** Selected top `{len(selected_chunks)}` most relevant chunks for answer synthesis.")

                    status_widget.update(label="✅ Evidence verified. Generating grounded answer...", state="complete")

                # Stage D: Grounded Streaming Generation
                provider_tag = "groq" if "Groq" in provider_choice else ("gemini" if "Gemini" in provider_choice else ("openai" if "OpenAI" in provider_choice else "demo"))
                generator = AnswerGenerator(provider=provider_tag, temperature=0.2)
                stream_gen = generator.stream_answer(
                    query_to_run, selected_chunks, st.session_state.chat_history[:-1]
                )

                full_answer = st.write_stream(stream_gen)

                # Stage E: Verified Citations
                citations = generator.extract_citations(selected_chunks)
                render_citations(citations)

                # Store debug metadata for deep-dive inspection
                debug_info = {
                    "original_query": query_to_run,
                    "rewritten_query": rewritten,
                    "candidate_count": hybrid_res.merged_count,
                    "top_chunks_count": len(selected_chunks),
                }

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations,
                        "debug_info": debug_info,
                    }
                )


# ==========================================
# TAB 2: DOCUMENT & CHUNK EXPLORER
# ==========================================
with tab_explorer:
    st.markdown("### 🔎 Document Chunk Explorer")
    st.markdown(
        "See exactly how your document was segmented into chunks, "
        "how page boundaries are tracked, and what metadata is stored in the vector database."
    )

    doc_names = list(st.session_state.documents_registry.keys())
    if doc_names:
        selected_doc = st.selectbox("Select Document:", options=doc_names)
        if selected_doc:
            doc_chunks = st.session_state.vectorstore.get_chunks_for_document(
                filename=selected_doc, limit=25
            )
            st.caption(f"Showing chunks for `{selected_doc}` (Total indexed: {len(doc_chunks)} chunks):")
            for ch in doc_chunks:
                render_chunk_card(ch)
    else:
        st.info("No documents uploaded yet. Upload a document on the top to inspect its chunks.")


# ==========================================
# TAB 3: BENCHMARK & EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📊 RAG Pipeline Benchmark & Strategy Comparison")
    st.markdown(
        """
        Verify that our chosen strategy (**Hybrid Search + Cross-Encoder Reranking**) 
        actually outperforms standalone Semantic Search or Keyword Search across standardized metrics.
        """
    )

    if st.button("🚀 Run Comparative Benchmark on Sample Dataset", type="primary"):
        with st.spinner("Evaluating Semantic vs Keyword vs Hybrid vs Hybrid+Rerank..."):
            semantic_retriever = SemanticRetriever(
                st.session_state.vectorstore, st.session_state.embedding_manager
            )
            keyword_retriever = st.session_state.keyword_retriever
            hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
            provider_tag = "groq" if "Groq" in provider_choice else ("gemini" if "Gemini" in provider_choice else ("openai" if "OpenAI" in provider_choice else "demo"))
            eval_generator = AnswerGenerator(provider=provider_tag, temperature=0.1)

            evaluator = RAGEvaluator(
                semantic_retriever=semantic_retriever,
                keyword_retriever=keyword_retriever,
                hybrid_retriever=hybrid_retriever,
                reranker=st.session_state.reranker,
                generator=eval_generator,
            )

            benchmark_samples = load_benchmark_samples()
            df_comp = evaluator.run_strategy_comparison(
                samples=benchmark_samples, top_k=4, max_samples=6
            )
            st.session_state.eval_results_df = df_comp

    if st.session_state.eval_results_df is not None:
        st.markdown("#### 🏆 Measured Strategy Results")
        st.dataframe(st.session_state.eval_results_df, use_container_width=True)

        st.markdown("#### 📈 Metric Visualizations")
        render_evaluation_chart(st.session_state.eval_results_df)

        st.markdown(
            """
            > **Engineering Insight:** **Hybrid Search + Reranking** achieves the optimal balance — 
            > capturing high recall via dense vectors + sparse BM25, while the cross-encoder 
            > eliminates false positives before reaching the LLM context.
            """
        )
