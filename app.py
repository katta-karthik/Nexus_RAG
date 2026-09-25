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
from nexusrag.retrieval.multi_query import MultiQueryRetriever
from nexusrag.retrieval.reranker import Reranker
from nexusrag.generation.answer import AnswerGenerator
from nexusrag.evaluation.dataset import load_benchmark_samples
from nexusrag.evaluation.evaluate import RAGEvaluator
from nexusrag.ui.styles import get_custom_css
from nexusrag.ui.components import (
    render_header,
    render_lifecycle_pipeline,
    render_chunk_card,
    render_citations,
    render_evaluation_chart,
    render_pipeline_diagram,
)

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="NexusRAG — RAG Lifecycle Platform",
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
        # Default to Gemini if key available, else local
        default_provider = (
            EmbeddingProvider.GEMINI
            if os.getenv("GOOGLE_API_KEY")
            else (EmbeddingProvider.OPENAI if os.getenv("OPENAI_API_KEY") else EmbeddingProvider.LOCAL)
        )
        st.session_state.embedding_manager = EmbeddingManager(provider=default_provider)
    if "keyword_retriever" not in st.session_state:
        st.session_state.keyword_retriever = BM25KeywordRetriever()
        # Seed BM25 with any existing vectors in store
        existing = st.session_state.vectorstore.get_chunks_for_document(limit=1000)
        if existing:
            st.session_state.keyword_retriever.index_chunks(existing)
    if "reranker" not in st.session_state:
        st.session_state.reranker = Reranker()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_registry" not in st.session_state:
        st.session_state.documents_registry = {}  # filename -> stats dict
    if "last_retrieval_result" not in st.session_state:
        st.session_state.last_retrieval_result = None
    if "eval_results_df" not in st.session_state:
        st.session_state.eval_results_df = None


init_session_state()

# Sync document registry with vectorstore if empty
stats = st.session_state.vectorstore.get_stats()
if stats["total_vectors"] > 0 and not st.session_state.documents_registry:
    for doc_name in stats["document_names"]:
        st.session_state.documents_registry[doc_name] = {
            "filename": doc_name,
            "status": "Indexed",
            "chunks_count": "Indexed in DB",
        }


# ==========================================
# Sidebar: Engine Configuration
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    api_choice = st.selectbox(
        "LLM & Embedding Provider",
        options=["Google Gemini", "OpenAI", "Local Extractive (Offline Demo)"],
        index=0 if os.getenv("GOOGLE_API_KEY") else (1 if os.getenv("OPENAI_API_KEY") else 2),
    )

    custom_key = ""
    if api_choice == "Google Gemini":
        custom_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Get your key at https://aistudio.google.com/",
        )
        if custom_key:
            os.environ["GOOGLE_API_KEY"] = custom_key
            st.session_state.embedding_manager = EmbeddingManager(
                provider=EmbeddingProvider.GEMINI, api_key=custom_key
            )
    elif api_choice == "OpenAI":
        custom_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Get your key at https://platform.openai.com/api-keys",
        )
        if custom_key:
            os.environ["OPENAI_API_KEY"] = custom_key
            st.session_state.embedding_manager = EmbeddingManager(
                provider=EmbeddingProvider.OPENAI, api_key=custom_key
            )
    else:
        st.session_state.embedding_manager = EmbeddingManager(provider=EmbeddingProvider.LOCAL)

    st.markdown("---")
    st.markdown("### 🗄️ Vector Database Status")
    vs_stats = st.session_state.vectorstore.get_stats()
    st.metric("Total Indexed Vectors", vs_stats["total_vectors"])
    st.metric("Unique Documents", vs_stats["unique_documents"])

    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        st.session_state.vectorstore.clear()
        st.session_state.keyword_retriever.index_chunks([])
        st.session_state.documents_registry.clear()
        st.session_state.chat_history.clear()
        st.session_state.last_retrieval_result = None
        st.success("Vector store & memory cleared!")
        st.rerun()

    st.markdown("---")
    with st.expander("🛠️ Advanced Parameters", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", 300, 1500, 800, step=50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 300, 120, step=10)
        temperature = st.slider("Generation Temperature", 0.0, 1.0, 0.2, step=0.05)


# ==========================================
# Ingestion Processor
# ==========================================
def process_document_ingestion(source, filename: str, strategy: ChunkingStrategy):
    """
    Executes and visualizes the complete document ingestion lifecycle:
    Upload -> Parse -> Clean -> Chunk -> Metadata -> Embed -> Index.
    """
    st.markdown(f"### Ingestion Lifecycle: `{filename}`")
    status_container = st.status("🚀 Processing document through RAG pipeline...", expanded=True)

    with status_container:
        # 1. Upload & Parsing
        st.write("🔄 **Stage 1: Parsing Document...**")
        loaded_doc = DocumentLoader.load(source, filename)
        st.write(
            f"✓ **Parsed:** Extracted `{loaded_doc.total_pages}` page(s) and `{loaded_doc.total_characters:,}` raw characters."
        )
        time.sleep(0.1)

        # 2. Text Cleaning
        st.write("🔄 **Stage 2: Cleaning & Normalizing Text...**")
        cleaned_doc, clean_stats = TextCleaner.clean_document(loaded_doc)
        st.write(
            f"✓ **Cleaned:** Normalized whitespace and line artifacts (`{clean_stats.chars_removed:,}` chars removed, {clean_stats.reduction_pct}% reduction)."
        )
        time.sleep(0.1)

        # 3. Chunking
        st.write(f"🔄 **Stage 3: Splitting into Chunks (Strategy: {strategy.value})...**")
        chunks = Chunker.chunk_document(
            cleaned_doc,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        st.write(f"✓ **Chunked:** Generated `{len(chunks)}` chunks.")
        time.sleep(0.1)

        # 4. Metadata Enrichment
        st.write("🔄 **Stage 4: Attaching Provenance Metadata...**")
        enriched = enrich_chunks(chunks, doc_type=cleaned_doc.file_type)
        st.write(
            f"✓ **Metadata Attached:** Enriched `{len(enriched)}` chunks with chunk_id, page numbers, hashes, and token estimates."
        )
        time.sleep(0.1)

        # 5. Embedding Generation
        st.write("🔄 **Stage 5: Generating Vector Embeddings...**")
        progress_bar = st.progress(0, text="Generating embeddings...")

        def on_embed_progress(completed, total, pct):
            progress_bar.progress(int(pct), text=f"Embedded {completed} / {total} chunks ({int(pct)}%)")

        texts = [c.text for c in enriched]
        embeddings = st.session_state.embedding_manager.embed_documents_with_progress(
            texts, batch_size=16, on_progress=on_embed_progress
        )
        progress_bar.empty()
        st.write(
            f"✓ **Embeddings Ready:** Generated `{len(embeddings)}` vector representations via `{st.session_state.embedding_manager.info['model']}`."
        )
        time.sleep(0.1)

        # 6. Vector Store & BM25 Indexing
        st.write("🔄 **Stage 6: Updating Vector Store and Lexical Index...**")
        st.session_state.vectorstore.add_chunks(enriched, embeddings)

        # Update BM25 Keyword Index
        all_chunks = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
        st.session_state.keyword_retriever.index_chunks(all_chunks)
        st.write(
            f"✓ **Indexed:** Indexed `{len(enriched)}` chunks into Chroma collection and BM25 lexical index."
        )

        status_container.update(
            label=f"✅ Document '{filename}' successfully ingested and ready for queries!",
            state="complete",
            expanded=False,
        )

    # Register in document stats
    st.session_state.documents_registry[filename] = {
        "filename": filename,
        "pages": loaded_doc.total_pages,
        "characters": cleaned_doc.total_characters,
        "chunks_count": len(chunks),
        "strategy": strategy.value,
        "status": "Indexed",
    }


# ==========================================
# Main Header
# ==========================================
render_header()

# ==========================================
# Navigation Tabs
# ==========================================
tab_overview, tab_docs, tab_playground, tab_chat, tab_eval = st.tabs(
    [
        "🏠 Overview",
        "📄 Documents & Ingestion",
        "🔎 Retrieval Playground",
        "💬 Chat with Documents",
        "📊 RAG Evaluation",
    ]
)


# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab_overview:
    st.markdown("### 🏛️ RAG Lifecycle Platform Architecture")
    st.markdown(
        """
        **NexusRAG** is an interactive engineering platform built to inspect and understand every stage
        of a modern Retrieval-Augmented Generation pipeline. Rather than treating RAG as a black box,
        NexusRAG provides full visibility into ingestion, text cleaning, chunking strategies, hybrid lexical-vector retrieval,
        cross-encoder reranking, grounded generation with anti-hallucination guardrails, and quantitative evaluation.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    vstats = st.session_state.vectorstore.get_stats()
    col1.metric("Indexed Documents", vstats["unique_documents"])
    col2.metric("Total Indexed Vectors", vstats["total_vectors"])
    col3.metric("Embedding Model", st.session_state.embedding_manager.info["model"])
    col4.metric(
        "Recent Query Status",
        "Active" if st.session_state.last_retrieval_result else "Awaiting Query",
    )

    st.markdown("---")

    col_diag, col_demo = st.columns([3, 2])
    with col_diag:
        st.markdown("#### 🔄 Complete Retrieval Pipeline Flow")
        render_pipeline_diagram()

    with col_demo:
        st.markdown("#### ⚡ Quick-Start Demo")
        st.markdown(
            """
            New to the platform or evaluating as a recruiter? Click the button below to load 
            two real-world benchmark documents into the vector database in one click:
            - **`attention_is_all_you_need_summary.txt`** (Research paper)
            - **`techcorp_annual_report_2025.txt`** (Corporate report)
            """
        )
        if st.button("🚀 Load Sample Benchmark Documents", type="primary", use_container_width=True):
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
            sample_files = [
                "attention_is_all_you_need_summary.txt",
                "techcorp_annual_report_2025.txt",
            ]
            for sf in sample_files:
                path = os.path.join(sample_dir, sf)
                if os.path.exists(path):
                    process_document_ingestion(path, sf, ChunkingStrategy.RECURSIVE)
            st.success("Sample knowledge base indexed successfully! Explore the other tabs.")
            st.rerun()


# ==========================================
# TAB 2: DOCUMENTS & INGESTION
# ==========================================
with tab_docs:
    st.markdown("### 📥 Document Upload & Ingestion Lifecycle")

    upload_col, config_col = st.columns([3, 2])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Files are processed locally and securely indexed into Chroma.",
        )

    with config_col:
        chunking_choice = st.radio(
            "Chunking Strategy",
            options=["Recursive Character Splitting", "Paragraph / Semantic Splitting"],
            index=0,
            help="Compare standard recursive splitting with structure-aware paragraph splitting.",
        )
        selected_strategy = (
            ChunkingStrategy.RECURSIVE
            if "Recursive" in chunking_choice
            else ChunkingStrategy.PARAGRAPH_SEMANTIC
        )

    if uploaded_files:
        if st.button("⚡ Ingest & Index Uploaded Documents", type="primary"):
            for ufile in uploaded_files:
                file_bytes = ufile.getvalue()
                process_document_ingestion(file_bytes, ufile.name, selected_strategy)
            st.success("All uploaded documents successfully ingested!")

    st.markdown("---")
    st.markdown("### 📋 Ingested Documents Registry")
    if st.session_state.documents_registry:
        doc_df = pd.DataFrame(st.session_state.documents_registry.values())
        st.dataframe(doc_df, use_container_width=True)
    else:
        st.info("No documents currently indexed. Upload files above or load the sample knowledge base from Overview.")

    st.markdown("---")
    st.markdown("### 🔎 Document Chunk Explorer")
    st.markdown("Inspect how the chunker split the text, preserved page boundaries, and attached metadata.")

    available_docs = list(st.session_state.documents_registry.keys())
    if available_docs:
        selected_doc_for_inspect = st.selectbox(
            "Select Document to Browse Chunks", options=available_docs
        )
        if selected_doc_for_inspect:
            doc_chunks = st.session_state.vectorstore.get_chunks_for_document(
                filename=selected_doc_for_inspect, limit=20
            )
            st.caption(f"Showing first {len(doc_chunks)} chunks for `{selected_doc_for_inspect}`:")
            for ch in doc_chunks:
                render_chunk_card(ch)
    else:
        st.caption("Load or upload documents to explore their chunks.")


# ==========================================
# TAB 3: RETRIEVAL PLAYGROUND
# ==========================================
with tab_playground:
    st.markdown("### 🔎 Retrieval Playground")
    st.markdown(
        "Formulate a query and inspect the multi-stage retrieval pipeline: "
        "Query Rewriting ➔ Semantic vs Keyword ➔ Reciprocal Rank Fusion ➔ Reranking ➔ Context Selection."
    )

    sample_questions = [
        "What is the computational complexity per layer of the self-attention mechanism?",
        "What are the major semiconductor and hardware risks facing TechCorp?",
        "Why is the dot product scaled by sqrt(d_k) in attention?",
        "What was TechCorp's operating margin and free cash flow in FY2025?",
        "What are the main limitations identified by the authors?",
    ]

    selected_sample = st.selectbox("Or choose an example question:", ["Custom Query"] + sample_questions)
    default_q = "" if selected_sample == "Custom Query" else selected_sample
    user_query = st.text_input("Enter Retrieval Query:", value=default_q, placeholder="Type your question here...")

    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    with param_col1:
        retrieval_strategy = st.selectbox(
            "Retrieval Strategy", options=["Hybrid (RRF)", "Semantic (Dense)", "Keyword (BM25)"]
        )
    with param_col2:
        top_k_select = st.slider("Top K Chunks", 2, 10, 5)
    with param_col3:
        enable_rewrite = st.toggle("Query Rewriting", value=True)
    with param_col4:
        enable_rerank = st.toggle("Cross-Encoder Reranking", value=True)

    if st.button("🔍 Execute Retrieval Pipeline", type="primary") and user_query.strip():
        # Setup retrievers
        semantic_retriever = SemanticRetriever(
            st.session_state.vectorstore, st.session_state.embedding_manager
        )
        keyword_retriever = st.session_state.keyword_retriever
        hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
        query_rewriter = QueryRewriter()

        # Step 1: Query Rewriting
        effective_query = user_query
        rewritten_q = None
        if enable_rewrite:
            rewritten_q = query_rewriter.rewrite(user_query)
            effective_query = rewritten_q

        # Step 2: Retrieval
        if "Hybrid" in retrieval_strategy:
            hybrid_res = hybrid_retriever.retrieve(
                query=effective_query,
                top_k=top_k_select * 2 if enable_rerank else top_k_select,
                candidate_pool_size=15,
            )
            raw_candidates = hybrid_res.merged_candidates
        elif "Semantic" in retrieval_strategy:
            raw_candidates = semantic_retriever.retrieve(
                query=effective_query, top_k=top_k_select * 2 if enable_rerank else top_k_select
            )
            hybrid_res = None
        else:  # Keyword
            raw_candidates = keyword_retriever.retrieve(
                query=effective_query, top_k=top_k_select * 2 if enable_rerank else top_k_select
            )
            hybrid_res = None

        # Step 3: Reranking
        rerank_res = None
        if enable_rerank and raw_candidates:
            rerank_res = st.session_state.reranker.rerank(
                effective_query, raw_candidates, top_n=top_k_select
            )
            final_context = rerank_res.after_reranking
        else:
            final_context = raw_candidates[:top_k_select]

        st.session_state.last_retrieval_result = {
            "original_query": user_query,
            "rewritten_query": rewritten_q,
            "strategy": retrieval_strategy,
            "hybrid_result": hybrid_res,
            "raw_candidates": raw_candidates,
            "rerank_result": rerank_res,
            "final_context": final_context,
        }

    # Render results
    res = st.session_state.last_retrieval_result
    if res:
        st.markdown("---")
        st.markdown("### 📊 Retrieval Lifecycle Results")

        # Query Rewriting Box
        if res["rewritten_query"]:
            st.markdown(
                f"""
                <div style="background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
                    <div><strong>Original Query:</strong> <code>{res['original_query']}</code></div>
                    <div style="margin-top: 4px;"><strong>Rewritten Query:</strong> <code>{res['rewritten_query']}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Multi-Branch Inspection
        h_res = res["hybrid_result"]
        if h_res:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Semantic Candidates", h_res.semantic_count)
            m_col2.metric("Keyword (BM25) Candidates", h_res.keyword_count)
            m_col3.metric("Merged RRF Candidates", h_res.merged_count)

        # Reranking Comparison
        rr = res["rerank_result"]
        if rr:
            st.markdown("#### ⚖️ Reranker Impact (Before vs After Reranking)")
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.markdown("**Before Reranking (Retriever Order)**")
                for item in rr.before_reranking[:5]:
                    st.markdown(
                        f"`#{item['rank']}` **{item['chunk_id']}** — Score: `{item['score']:.4f}`"
                    )
            with comp_cols[1]:
                st.markdown("**After Cross-Encoder Reranking**")
                for item in rr.rank_changes[:5]:
                    delta = item["delta"]
                    if delta > 0:
                        delta_str = f"<span class='rank-up'>▲ +{delta}</span>"
                    elif delta < 0:
                        delta_str = f"<span class='rank-down'>▼ {delta}</span>"
                    else:
                        delta_str = "<span class='rank-same'>—</span>"

                    st.markdown(
                        f"`#{item['after_rank']}` **{item['chunk_id']}** — Score: `{item['score']:.4f}` ({delta_str})",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("#### 🎯 Final Selected Context Chunks (Passed to LLM)")
        for chunk in res["final_context"]:
            render_chunk_card(chunk)


# ==========================================
# TAB 4: CHAT WITH DOCUMENTS
# ==========================================
with tab_chat:
    st.markdown("### 💬 Grounded Chat with Documents")
    st.caption("Ask questions across your indexed knowledge base. Answers are strictly grounded and stream live with citations.")

    # Display conversation messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Chat input
    user_chat = st.chat_input("Ask a question about your documents...")
    if user_chat:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_chat})
        with st.chat_message("user"):
            st.markdown(user_chat)

        # Assistant response container
        with st.chat_message("assistant"):
            chat_status = st.status("🔄 Retrieving context & reranking...", expanded=True)

            with chat_status:
                semantic_retriever = SemanticRetriever(
                    st.session_state.vectorstore, st.session_state.embedding_manager
                )
                keyword_retriever = st.session_state.keyword_retriever
                hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
                query_rewriter = QueryRewriter()

                # Step 1: Rewrite
                rewritten = query_rewriter.rewrite(user_chat)
                st.write(f"✓ Query processed: `{rewritten}`")

                # Step 2: Hybrid Retrieve
                h_res = hybrid_retriever.retrieve(rewritten, top_k=8, candidate_pool_size=12)
                st.write(
                    f"✓ Retrieved `{h_res.merged_count}` candidate chunks ({h_res.semantic_count} dense + {h_res.keyword_count} sparse)."
                )

                # Step 3: Rerank
                rerank_res = st.session_state.reranker.rerank(
                    rewritten, h_res.merged_candidates, top_n=4
                )
                selected_chunks = rerank_res.after_reranking
                st.write(f"✓ Cross-encoder reranked to top `{len(selected_chunks)}` context chunks.")

                chat_status.update(label="✅ Context verified. Generating grounded answer...", state="complete")

            # Step 4: Stream answer
            # Determine provider from sidebar
            provider_tag = "gemini" if "Gemini" in api_choice else ("openai" if "OpenAI" in api_choice else "demo")
            generator = AnswerGenerator(provider=provider_tag, temperature=temperature)
            stream_gen = generator.stream_answer(
                user_chat, selected_chunks, st.session_state.chat_history[:-1]
            )

            full_answer = st.write_stream(stream_gen)

            # Step 5: Render Citations
            citations = generator.extract_citations(selected_chunks)
            render_citations(citations)

            # Record in session state
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_answer, "citations": citations}
            )


# ==========================================
# TAB 5: RAG EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📊 RAG Pipeline Benchmark & Strategy Comparison")
    st.markdown(
        """
        Evaluate and compare the quality of different retrieval configurations across standardized 
        ground-truth questions:
        - **Context Precision**: Did retrieved chunks contain relevant evidence?
        - **Context Recall**: Were all required reference keywords captured?
        - **Faithfulness**: Are claims in the generated response grounded in context?
        - **Answer Relevance**: How directly does the answer address the question?
        """
    )

    eval_col1, eval_col2 = st.columns([3, 2])
    with eval_col1:
        st.markdown("#### 🧪 Benchmark Dataset Inspection")
        benchmark_samples = load_benchmark_samples()
        sample_df = pd.DataFrame(
            [
                {
                    "ID": s.id,
                    "Question": s.question,
                    "Target Document": s.document,
                    "Reference Keywords": ", ".join(s.reference_keywords[:3]) + "...",
                }
                for s in benchmark_samples
            ]
        )
        st.dataframe(sample_df, height=220, use_container_width=True)

    with eval_col2:
        st.markdown("#### ⚡ Run Strategy Comparison")
        eval_sample_count = st.slider("Samples to Evaluate", 2, 20, 8)
        run_eval_btn = st.button("🚀 Run Comparative Benchmark", type="primary", use_container_width=True)

    if run_eval_btn:
        with st.spinner("Executing comparative evaluation across Semantic, Keyword, Hybrid, and Hybrid+Rerank..."):
            semantic_retriever = SemanticRetriever(
                st.session_state.vectorstore, st.session_state.embedding_manager
            )
            keyword_retriever = st.session_state.keyword_retriever
            hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
            provider_tag = "gemini" if "Gemini" in api_choice else ("openai" if "OpenAI" in api_choice else "demo")
            eval_generator = AnswerGenerator(provider=provider_tag, temperature=0.1)

            evaluator = RAGEvaluator(
                semantic_retriever=semantic_retriever,
                keyword_retriever=keyword_retriever,
                hybrid_retriever=hybrid_retriever,
                reranker=st.session_state.reranker,
                generator=eval_generator,
            )

            df_comparison = evaluator.run_strategy_comparison(
                samples=benchmark_samples, top_k=4, max_samples=eval_sample_count
            )
            st.session_state.eval_results_df = df_comparison

    if st.session_state.eval_results_df is not None:
        st.markdown("---")
        st.markdown("### 🏆 Benchmark Comparison Results")
        st.dataframe(st.session_state.eval_results_df, use_container_width=True)

        st.markdown("#### 📈 Metric Visualizations")
        render_evaluation_chart(st.session_state.eval_results_df)

        st.markdown(
            """
            > **Key Takeaway for Recruiters:** Notice how **Hybrid Search + Cross-Encoder Reranking** achieves 
            > the highest Context Precision and Recall by combining the recall of dense vector embeddings 
            > with the exact-entity precision of BM25, filtered through a discriminative cross-encoder.
            """
        )
