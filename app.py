import os
import sys
import time
from typing import List, Dict, Any, Optional
import streamlit as st
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
from nexusrag.generation.suggestions import DocumentQuestionSuggester
from nexusrag.ui.styles import get_custom_css
from nexusrag.ui.flowchart import render_flowchart_html
from nexusrag.ui.components import render_header, render_active_doc_bar, render_chunk_card, render_citations

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="NexusRAG — Document Intelligence",
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
    if "active_document" not in st.session_state:
        st.session_state.active_document = None
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "last_ingestion_stages" not in st.session_state:
        st.session_state.last_ingestion_stages = None
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False


init_session_state()

# Sync document registry with existing ChromaDB contents on reload
vs_stats = st.session_state.vectorstore.get_stats()
if vs_stats["total_vectors"] > 0 and not st.session_state.documents_registry:
    for doc_name in vs_stats["document_names"]:
        doc_chunks = st.session_state.vectorstore.get_chunks_for_document(doc_name, limit=100)
        full_text = " ".join([c.get("text", "") for c in doc_chunks[:5]])
        sugg = DocumentQuestionSuggester.generate_suggestions(full_text, doc_name)
        st.session_state.documents_registry[doc_name] = {
            "filename": doc_name,
            "pages": 1,
            "chunks_count": len(doc_chunks),
            "suggested_questions": sugg,
        }
    if not st.session_state.active_document and vs_stats["document_names"]:
        st.session_state.active_document = vs_stats["document_names"][0]
        st.session_state.suggested_questions = st.session_state.documents_registry[st.session_state.active_document]["suggested_questions"]


# ==========================================
# RAG Ingestion Lifecycle Runner (With Live Flowchart)
# ==========================================
def run_rag_lifecycle(source, filename: str):
    """
    Executes the full RAG lifecycle with a live interactive flowchart in the UI:
    1. Document Ingestion
    2. Document Parsing
    3. Noise Cleaning
    4. Semantic Chunking
    5. Dense Vector Embeddings
    6. ChromaDB + BM25 Indexing
    7. AI Question Synthesis
    """
    stages = [
        {"icon": "📥", "title": "Ingestion", "status": "pending", "metric": "Reading file..."},
        {"icon": "📄", "title": "Parsing", "status": "pending", "metric": "Extracting text..."},
        {"icon": "🧹", "title": "Cleaning", "status": "pending", "metric": "Normalizing..."},
        {"icon": "✂️", "title": "Chunking", "status": "pending", "metric": "Recursive split..."},
        {"icon": "🧬", "title": "Embeddings", "status": "pending", "metric": "HuggingFace MiniLM..."},
        {"icon": "🗄️", "title": "Hybrid Index", "status": "pending", "metric": "Chroma + BM25..."},
        {"icon": "💡", "title": "AI Questions", "status": "pending", "metric": "Synthesizing..."},
    ]

    flowchart_slot = st.empty()

    def update_flow():
        flowchart_slot.markdown(render_flowchart_html(stages), unsafe_allow_html=True)

    # 1. Ingestion
    stages[0]["status"] = "running"
    stages[0]["metric"] = f"Validating {filename}"
    update_flow()
    time.sleep(0.15)
    stages[0]["status"] = "completed"
    stages[0]["metric"] = "File validated ✓"
    update_flow()

    # 2. Parsing
    stages[1]["status"] = "running"
    stages[1]["metric"] = "Extracting text..."
    update_flow()
    loaded_doc = DocumentLoader.load(source, filename)
    time.sleep(0.15)
    stages[1]["status"] = "completed"
    stages[1]["metric"] = f"{loaded_doc.total_pages} pg ({loaded_doc.total_characters:,} chars)"
    update_flow()

    # 3. Text Cleaning
    stages[2]["status"] = "running"
    stages[2]["metric"] = "Normalizing whitespace..."
    update_flow()
    cleaned_doc, clean_stats = TextCleaner.clean_document(loaded_doc)
    time.sleep(0.15)
    stages[2]["status"] = "completed"
    stages[2]["metric"] = f"{clean_stats.chars_removed:,} chars filtered"
    update_flow()

    # 4. Semantic Chunking
    stages[3]["status"] = "running"
    stages[3]["metric"] = "Recursive 800/120..."
    update_flow()
    chunks = Chunker.chunk_document(
        cleaned_doc,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=800,
        chunk_overlap=120,
    )
    enriched = enrich_chunks(chunks, doc_type=cleaned_doc.file_type)
    time.sleep(0.15)
    stages[3]["status"] = "completed"
    stages[3]["metric"] = f"{len(enriched)} semantic chunks"
    update_flow()

    # 5. Embeddings
    stages[4]["status"] = "running"
    stages[4]["metric"] = f"Embedding {len(enriched)} chunks..."
    update_flow()
    texts = [c.text for c in enriched]
    embeddings = st.session_state.embedding_manager.embed_documents(texts, batch_size=16)
    time.sleep(0.15)
    stages[4]["status"] = "completed"
    stages[4]["metric"] = "384-d vectors ✓"
    update_flow()

    # 6. Hybrid Indexing
    stages[5]["status"] = "running"
    stages[5]["metric"] = "Writing Chroma + BM25..."
    update_flow()
    st.session_state.vectorstore.add_chunks(enriched, embeddings)
    all_chunks = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
    st.session_state.keyword_retriever.index_chunks(all_chunks)
    time.sleep(0.15)
    stages[5]["status"] = "completed"
    stages[5]["metric"] = "Indexed & synced ✓"
    update_flow()

    # 7. AI Question Synthesis (Document-specific suggested questions)
    stages[6]["status"] = "running"
    stages[6]["metric"] = "Analyzing document..."
    update_flow()
    full_sample_text = cleaned_doc.full_text[:3500]
    questions = DocumentQuestionSuggester.generate_suggestions(full_sample_text, filename)
    time.sleep(0.15)
    stages[6]["status"] = "completed"
    stages[6]["metric"] = f"{len(questions)} questions ready ✓"
    update_flow()

    # Save to session
    st.session_state.documents_registry[filename] = {
        "filename": filename,
        "pages": loaded_doc.total_pages,
        "characters": cleaned_doc.total_characters,
        "chunks_count": len(enriched),
        "suggested_questions": questions,
    }
    st.session_state.active_document = filename
    st.session_state.suggested_questions = questions
    st.session_state.last_ingestion_stages = stages
    st.session_state.show_uploader = False


def remove_document(filename: str):
    """CRUD: Remove a single document and refresh index."""
    st.session_state.vectorstore.delete_document(filename)
    if filename in st.session_state.documents_registry:
        del st.session_state.documents_registry[filename]

    remaining = st.session_state.vectorstore.get_chunks_for_document(limit=2000)
    st.session_state.keyword_retriever.index_chunks(remaining)

    active_list = list(st.session_state.documents_registry.keys())
    if active_list:
        st.session_state.active_document = active_list[0]
        st.session_state.suggested_questions = st.session_state.documents_registry[active_list[0]]["suggested_questions"]
    else:
        st.session_state.active_document = None
        st.session_state.suggested_questions = []
        st.session_state.last_ingestion_stages = None
    st.rerun()


def clear_all():
    """CRUD: Purge all documents and reset memory."""
    st.session_state.vectorstore.clear()
    st.session_state.keyword_retriever.index_chunks([])
    st.session_state.documents_registry.clear()
    st.session_state.chat_history.clear()
    st.session_state.active_document = None
    st.session_state.suggested_questions = []
    st.session_state.last_ingestion_stages = None
    st.rerun()


# ==========================================
# Sidebar: Document Management (CRUD)
# ==========================================
with st.sidebar:
    st.markdown("### 📂 Document Knowledge Base")

    active_docs = list(st.session_state.documents_registry.keys())

    if active_docs:
        st.markdown(f"**Indexed Documents ({len(active_docs)}):**")
        for doc_name in active_docs:
            d_info = st.session_state.documents_registry[doc_name]
            is_active = doc_name == st.session_state.active_document

            c_doc, c_del = st.columns([3, 1])
            with c_doc:
                label = f"**{'🟢 ' if is_active else '📄 '}{doc_name}**"
                if st.button(label, key=f"select_{doc_name}", help="Switch active document"):
                    st.session_state.active_document = doc_name
                    st.session_state.suggested_questions = d_info.get("suggested_questions", [])
                    st.rerun()
                st.caption(f"{d_info.get('pages', 1)} pg • {d_info.get('chunks_count', 'N/A')} chunks")
            with c_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                    remove_document(doc_name)

        st.markdown("---")
        if st.button("➕ Upload Another Document", use_container_width=True, type="secondary"):
            st.session_state.show_uploader = True
            st.rerun()

        if st.button("🗑️ Clear All Knowledge", use_container_width=True):
            clear_all()
    else:
        st.info("No documents indexed yet. Upload a document to start.")

    st.markdown("---")
    with st.expander("🔎 Inspect Chunks (Under the Hood)", expanded=False):
        if active_docs:
            inspect_doc = st.selectbox("Select document:", active_docs, key="inspect_select")
            if inspect_doc:
                chunks = st.session_state.vectorstore.get_chunks_for_document(inspect_doc, limit=10)
                st.caption(f"Showing first {len(chunks)} chunks:")
                for c in chunks:
                    render_chunk_card(c, show_metadata=False)
        else:
            st.caption("Upload a document first to inspect chunks.")

    with st.expander("⚙️ Inference Engine", expanded=False):
        st.caption("Engine: Groq High-Speed LPU")
        gkey = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        if gkey:
            os.environ["GROQ_API_KEY"] = gkey
        st.caption("Model: `qwen/qwen3.8-27b` (Ultra-low latency)")


# ==========================================
# Main App Header
# ==========================================
render_header()

# ==========================================
# Primary Screen View
# Case A: No Document Loaded OR User clicked "Upload Another"
# ==========================================
has_active_docs = len(st.session_state.documents_registry) > 0

if not has_active_docs or st.session_state.show_uploader:
    st.markdown(
        """
        <div class="hero-upload-card">
            <div class="upload-icon">📄</div>
            <h2 style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px;">
                Upload Your Document to Run the RAG Lifecycle
            </h2>
            <p style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; margin: 0 auto 16px auto;">
                NexusRAG will parse, clean, chunk, embed, and index your document live through the full RAG lifecycle, then generate instant questions tailored specifically to your content.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_samples = st.columns([1.1, 0.9], gap="large")

    with col_upload:
        st.markdown("#### 📤 Upload Any File")
        uploaded_file = st.file_uploader(
            "Drop your PDF, TXT, or Markdown document:",
            type=["pdf", "txt", "md"],
            help="Resumes, research papers, legal agreements, manuals, financial reports.",
        )
        if uploaded_file is not None:
            if st.button(f"⚡ Run RAG Lifecycle on '{uploaded_file.name}'", type="primary", use_container_width=True):
                run_rag_lifecycle(uploaded_file.getvalue(), uploaded_file.name)
                st.rerun()

    with col_samples:
        st.markdown("#### ✨ Or Try With a Ready Sample Document")
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")

        # Sample 1: Resume
        c1_path = os.path.join(sample_dir, "sample_resume_karthik.txt")
        if os.path.exists(c1_path):
            if st.button("👤 **Software Engineer Resume** — Karthik Katta (B.Tech, 8.41 CGPA)", use_container_width=True):
                run_rag_lifecycle(c1_path, "sample_resume_karthik.txt")
                st.rerun()

        # Sample 2: Research Paper
        c2_path = os.path.join(sample_dir, "attention_is_all_you_need_summary.txt")
        if os.path.exists(c2_path):
            if st.button("🔬 **AI Research Paper** — 'Attention Is All You Need'", use_container_width=True):
                run_rag_lifecycle(c2_path, "attention_is_all_you_need_summary.txt")
                st.rerun()

        # Sample 3: TechCorp Report
        c3_path = os.path.join(sample_dir, "techcorp_annual_report_2025.txt")
        if os.path.exists(c3_path):
            if st.button("📊 **Annual Financial Report** — TechCorp ($4.2B Revenue)", use_container_width=True):
                run_rag_lifecycle(c3_path, "techcorp_annual_report_2025.txt")
                st.rerun()

    if has_active_docs and st.session_state.show_uploader:
        if st.button("⬅️ Return to Chat", type="secondary"):
            st.session_state.show_uploader = False
            st.rerun()

    st.stop()


# ==========================================
# Primary Screen View
# Case B: Document Ingested -> Clean Chat Screen
# ==========================================
reg_docs = list(st.session_state.documents_registry.keys())
active_doc_name = st.session_state.active_document or (reg_docs[0] if reg_docs else "Document")
doc_info = st.session_state.documents_registry.get(active_doc_name, {})
pages_cnt = doc_info.get("pages", 1)
chunks_cnt = doc_info.get("chunks_count", 0)

# Top Bar with Document Status & Actions
render_active_doc_bar(active_doc_name, pages_cnt, chunks_cnt)

# Optional Lifecycle Flowchart Expander
with st.expander("⚡ RAG Lifecycle Architecture & Ingestion Flowchart", expanded=(st.session_state.last_ingestion_stages is not None and len(st.session_state.chat_history) == 0)):
    if st.session_state.last_ingestion_stages:
        st.markdown(render_flowchart_html(st.session_state.last_ingestion_stages), unsafe_allow_html=True)
    else:
        completed_stages = [
            {"icon": "📥", "title": "Ingestion", "status": "completed", "metric": "Validated ✓"},
            {"icon": "📄", "title": "Parsing", "status": "completed", "metric": f"{pages_cnt} page(s) ✓"},
            {"icon": "🧹", "title": "Cleaning", "status": "completed", "metric": "Normalized ✓"},
            {"icon": "✂️", "title": "Chunking", "status": "completed", "metric": f"{chunks_cnt} chunks ✓"},
            {"icon": "🧬", "title": "Embeddings", "status": "completed", "metric": "MiniLM 384-d ✓"},
            {"icon": "🗄️", "title": "Hybrid Index", "status": "completed", "metric": "Chroma + BM25 ✓"},
            {"icon": "💡", "title": "AI Questions", "status": "completed", "metric": "Synthesized ✓"},
        ]
        st.markdown(render_flowchart_html(completed_stages), unsafe_allow_html=True)

# ==========================================
# Dynamic Document-Tailored Suggested Questions
# ==========================================
suggested_questions = st.session_state.suggested_questions or doc_info.get("suggested_questions", [])
selected_suggestion = None

if suggested_questions:
    st.markdown(
        """
        <div style="font-size: 0.84rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 12px 0 8px 0; display: flex; align-items: center; gap: 6px;">
            <span>💡</span> Suggested Questions (Derived Specifically From This Document):
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(suggested_questions))
    for i, sq in enumerate(suggested_questions):
        with cols[i]:
            if st.button(f"👉 {sq}", key=f"sugg_btn_{i}", use_container_width=True):
                selected_suggestion = sq

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==========================================
# Conversation Thread
# ==========================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])
        if msg.get("trace"):
            tr = msg["trace"]
            st.markdown(
                f"""
                <div class="lifecycle-trace-box">
                    <strong>🔍 RAG Retrieval Trace:</strong> Rewritten: <code>{tr.get('rewritten_query')}</code> &nbsp;•&nbsp; 
                    Hybrid Pool: <code>{tr.get('candidate_count')} chunks (Dense + BM25)</code> &nbsp;•&nbsp; 
                    Reranked: <code>Top {tr.get('top_count')} chunks verified</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================
# Chat Input & RAG Execution
# ==========================================
user_query = st.chat_input("Ask any question about your document in natural language...")
query_to_run = selected_suggestion or user_query

if query_to_run:
    # 1. Add user message
    st.session_state.chat_history.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # 2. Assistant RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Executing RAG retrieval: Query Rewrite ➔ Hybrid Search ➔ Cross-Encoder Rerank..."):
            semantic_retriever = SemanticRetriever(
                st.session_state.vectorstore, st.session_state.embedding_manager
            )
            keyword_retriever = st.session_state.keyword_retriever
            hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
            query_rewriter = st.session_state.query_rewriter
            reranker = st.session_state.reranker

            # Step 1: Query Rewrite
            rewritten = query_rewriter.rewrite(query_to_run)

            # Step 2: Hybrid Retrieval (Dense Vector + BM25 Lexical with RRF)
            hybrid_res = hybrid_retriever.retrieve(rewritten, top_k=8, candidate_pool_size=12)

            # Step 3: Cross-Encoder Reranking
            rerank_res = reranker.rerank(rewritten, hybrid_res.merged_candidates, top_n=4)
            selected_chunks = rerank_res.after_reranking

        # Step 4: Stream Natural Language Grounded Answer
        generator = AnswerGenerator(provider="groq", temperature=0.2)
        stream_gen = generator.stream_answer(
            query_to_run, selected_chunks, st.session_state.chat_history[:-1]
        )

        full_answer = st.write_stream(stream_gen)

        # Step 5: Verified Citations
        citations = generator.extract_citations(selected_chunks)
        render_citations(citations)

        trace_data = {
            "rewritten_query": rewritten,
            "candidate_count": hybrid_res.merged_count,
            "top_count": len(selected_chunks),
        }

        st.markdown(
            f"""
            <div class="lifecycle-trace-box">
                <strong>🔍 RAG Retrieval Trace:</strong> Rewritten: <code>{rewritten}</code> &nbsp;•&nbsp; 
                Hybrid Pool: <code>{hybrid_res.merged_count} candidates (Dense + BM25)</code> &nbsp;•&nbsp; 
                Reranked: <code>Top {len(selected_chunks)} chunks verified</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": full_answer,
                "citations": citations,
                "trace": trace_data,
            }
        )
