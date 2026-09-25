# NexusRAG — Complete RAG Lifecycle Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-00A67E.svg)](https://python.langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-F55036.svg)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com)
[![FlashRank](https://img.shields.io/badge/FlashRank-Cross_Encoder-blueviolet.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **An enterprise-grade, end-to-end RAG Lifecycle Platform demonstrating the complete retrieval-augmented generation lifecycle — from ingestion, text cleaning, and recursive chunking to hybrid dense-sparse retrieval (BM25 + ChromaDB with Reciprocal Rank Fusion), cross-encoder reranking, live lifecycle flowcharts, dynamic document-specific question synthesis, and grounded natural language chat with verifiable citations.**

---

## 🌟 Overview: Beyond "Chat with PDF" Black Boxes

Most RAG tutorials and portfolio repositories treat retrieval as a black box: a file goes in, an LLM answers. 

**NexusRAG** makes the entire RAG lifecycle **visible, inspectable, and production-engineered**. Every stage of the pipeline exposes real-time telemetry, before-and-after states, intermediate retrieval candidate pools, and verifiable provenance:

```text
Document Upload (PDF / TXT / MD)
      ↓
Document Parsing & Text Extraction
      ↓
Text Cleaning & Normalization (Regex, Artifacts, Spacing)
      ↓
Semantic Chunking (Recursive Boundary Splitting + Overlap)
      ↓
Metadata Provenance (Doc ID, Page, Chunk ID, Token Estimate)
      ↓
Dense Vector Embeddings (384-d HuggingFace / MiniLM)
      ↓
Hybrid Vector & Lexical Indexing (ChromaDB + BM25 Okapi)
      ↓
AI Document Analysis & Targeted Question Synthesis
      ↓
Query Optimization & Reformulation
      ↓
Hybrid Retrieval (Dense Semantic + Sparse Lexical via RRF)
      ↓
Cross-Encoder Reranking (FlashRank CPU Scoring)
      ↓
Context Assembly & Deduplication
      ↓
Grounded LLM Generation (Groq LPU Ultra-Fast Streaming)
      ↓
Direct Natural Language Answer & Verified Citations
```

---

## 🚀 Key Features

### 1. 📥 Single Unified Upload Experience
- **Frictionless Ingestion:** Clean, single drag-and-drop zone supporting `.pdf`, `.txt`, and `.md` documents.
- **1-Click Ready Samples:** Instant test documents built right into the home screen:
  - 👤 **Software Engineer Resume** — *Karthik Katta (B.Tech CSE • 8.41 CGPA • AI Projects)*
  - 🔬 **AI Research Paper** — *'Attention Is All You Need' (Transformer Architecture)*
  - 📊 **Annual Financial Report** — *TechCorp ($4.2B Revenue • Segment Margins)*

### 2. ⚡ Live Interactive Lifecycle Flowchart
Rather than generic loading spinners, NexusRAG renders a visual, interactive 7-stage pipeline flowchart directly in the UI as your document is processed:

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ STAGE 01        │     │ STAGE 02        │     │ STAGE 03        │     │ STAGE 04        │
│ 📥 Ingestion    │ ──➔ │ 📄 Parsing      │ ──➔ │ 🧹 Cleaning     │ ──➔ │ ✂️ Chunking     │
│ Validated ✓     │     │ 2 Pages ✓       │     │ 1,450 Chars ✓   │     │ 8 Chunks ✓      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                 │
                                                                                 ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ STAGE 07        │     │ STAGE 06        │     │ STAGE 05        │
│ 💡 AI Questions │ ◄── │ 🗄️ Hybrid Index │ ◄── │ 🧬 Embeddings   │
│ 4 Ready ✓       │     │ Chroma + BM25 ✓ │     │ MiniLM 384-d ✓  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```
- Each stage pulses with a glowing blue indicator while running, and transitions to an emerald green badge `✓` displaying live execution metrics upon completion.
- You can collapse or expand this flowchart anytime at the top of the chat screen.

### 3. 💡 Dynamic, Content-Tailored Suggested Questions
No more generic questions like *"What is the company's financial growth?"* when you upload a resume!
- Once a document is ingested, our `DocumentQuestionSuggester` analyzes the content using Groq (`qwen/qwen3.8-27b`).
- **If you upload a resume**, it automatically generates tailored question pills:
  - `👉 What is Karthik's CGPA and which university did he graduate from?`
  - `👉 Which AI and GenAI frameworks are listed in his technical skills?`
  - `👉 What improvements did he achieve during his internship at NexaTech Solutions?`
  - `👉 What are the key features of the NexusRAG project?`
- **If you upload a research paper**, it asks about attention formulas, architecture differences, and experimental BLEU benchmarks.
- Clicking any suggested question immediately populates the chat and runs the RAG pipeline!

### 4. 🔍 Complete Hybrid Retrieval & Reranking
- **Query Optimization:** Translates conversational user phrasing into a high-density keyword query.
- **Dense Vector Search:** ChromaDB cosine-space search capturing deep semantic concepts and synonyms.
- **Sparse Lexical Search:** BM25 Okapi inverted indexing matching exact entities, numbers, acronyms, and formulas.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse candidates with a reciprocal rank score:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + rank_m(d)}$$
- **FlashRank Cross-Encoder Reranker:** Evaluates query-document pairs simultaneously using a lightweight CPU cross-encoder model to prune noisy candidates down to the top-4 most relevant chunks.

### 5. 💬 Grounded Natural Language Chat
- **Direct & Conversational:** Powered by Groq's high-speed inference engine (`qwen/qwen3.8-27b`), producing clean, natural language answers without raw chunk dumps.
  - *Example Query:* "what is his cgpa"
  - *Direct Answer:* **"Karthik Katta graduated from VNR Vignana Jyothi Institute of Engineering and Technology with a CGPA of 8.41."**
- **Verified Citations:** Every response includes collapsible source references with document name, page numbers, match relevance scores, and verbatim excerpts.
- **Transparent RAG Lifecycle Trace:** Expandable trace revealing rewritten queries, candidate pool sizes, and reranked selections.

### 6. 📂 Full Document CRUD (Create, Read, Update, Delete)
- **Create / Add:** Drag and drop new documents or load sample benchmarks anytime.
- **Read / List:** The sidebar actively displays all indexed documents with page and chunk counts.
- **Delete / Remove:** Individual `🗑️` delete buttons remove a document's vectors from ChromaDB and dynamically rebuild the BM25 index. Includes a **Clear All Knowledge** button.
- **Inspect Chunks:** An expandable chunk inspector lets you examine chunk text, token counts, and metadata payloads under the hood.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Executive UI                          │
│   [Hero Upload / Samples] ──➔ [RAG Flowchart] ──➔ [Grounded Chat]     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌────────────────────────┐    ┌────────────────────────┐
       │   Ingestion Pipeline   │    │  Retrieval & Synthesis │
       │  • DocumentLoader      │    │  • QueryRewriter       │
       │  • TextCleaner         │    │  • SemanticRetriever   │
       │  • Chunker (Recursive) │    │  • BM25Retriever       │
       │  • MetadataEnricher    │    │  • Hybrid (RRF Merger) │
       │  • QuestionSuggester   │    │  • FlashRank Reranker  │
       └───────────┬────────────┘    │  • AnswerGenerator     │
                   │                 └────────────┬───────────┘
                   ▼                              │
       ┌────────────────────────┐                 │
       │     Indexing Layer     │                 │
       │  • EmbeddingManager    │◄────────────────┘
       │  • Chroma VectorStore  │
       │  • BM25 Inverted Index │
       └────────────────────────┘
```

---

## 📁 Repository Structure

```text
Nexus_RAG/
├── nexusrag/
│   ├── ingestion/
│   │   ├── loaders.py          # PDF (PyPDF), TXT, MD multi-format parsing
│   │   ├── cleaner.py          # Whitespace & Unicode artifact normalization
│   │   ├── chunker.py          # Recursive boundary splitting with overlap
│   │   └── metadata.py         # Provenance enrichment & token hashing
│   ├── indexing/
│   │   ├── embeddings.py       # 384-d MiniLM, Gemini, OpenAI & Local Embeddings
│   │   └── vectorstore.py      # Chroma collection manager with cosine space
│   ├── retrieval/
│   │   ├── semantic.py         # Dense vector similarity retriever
│   │   ├── keyword.py          # BM25 Okapi lexical retriever
│   │   ├── hybrid.py           # Reciprocal Rank Fusion (RRF) candidate merger
│   │   ├── query_rewrite.py    # Groq & rule-based query reformulator
│   │   ├── multi_query.py      # Multi-perspective sub-query generator
│   │   └── reranker.py         # FlashRank CPU cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py          # Anti-hallucination grounded prompts
│   │   ├── answer.py           # Groq streaming generator with citations
│   │   └── suggestions.py      # Document-specific suggested questions engine
│   ├── graph/
│   │   └── rag_graph.py        # LangGraph StateGraph orchestration pipeline
│   ├── ui/
│   │   ├── components.py       # Active doc banner, chunk cards, citations
│   │   ├── flowchart.py        # 7-stage interactive visual flowchart renderer
│   │   └── styles.py           # Modern executive CSS styling
│   └── mcp/
│       └── server.py           # Fast JSON-RPC MCP document operations server
├── sample_docs/
│   ├── sample_resume_karthik.txt             # Realistic software engineer resume
│   ├── attention_is_all_you_need_summary.txt # AI research paper benchmark
│   └── techcorp_annual_report_2025.txt       # Enterprise financial report benchmark
├── tests/
│   └── test_rag_lifecycle.py   # Complete end-to-end unit & lifecycle tests
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Production dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # VCS ignore rules
└── README.md                   # Platform documentation
```

---

## ⚡ Quick-Start (Run Locally)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/katta-karthik/Nexus_RAG.git
cd Nexus_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your **Groq API Key** (for ultra-fast natural language answers and question synthesis):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Optional: `GOOGLE_API_KEY` or `OPENAI_API_KEY` can also be used)*

> **Offline Demo Mode:** NexusRAG includes a built-in **Local Deterministic Vectorizer & Extractive Synthesizer**, allowing complete demonstration even without any API keys!

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

### 4. Run Unit & Lifecycle Tests
```bash
python -m unittest tests/test_rag_lifecycle.py
```

---

## ⏱️ 2-Minute Walkthrough (For Recruiters & Evaluators)

1. **Step 1 — Ingest a Document:** On the home screen, click **`👤 Resume — Karthik Katta`** (or drag and drop your own PDF resume).
2. **Step 2 — Watch the RAG Lifecycle Flowchart:** Watch the 7 visual stages light up sequentially with real metrics:
   - *Validating ➔ Extracting Pages ➔ Filtering Noise ➔ Creating Chunks ➔ Embedding 384-d Vectors ➔ Indexing Chroma + BM25 ➔ Synthesizing Questions.*
3. **Step 3 — Inspect Dynamic Suggested Questions:** On the chat screen, notice the 4 questions tailored specifically to the resume (e.g. CGPA, technical skills, internships).
4. **Step 4 — Ask in Natural Language:** Click **`👉 What is Karthik's CGPA and which university did he graduate from?`** (or type your own question).
5. **Step 5 — Verify Grounding & Citations:** Observe the instant, direct answer:
   > *"Karthik Katta graduated from VNR Vignana Jyothi Institute of Engineering and Technology with a CGPA of 8.41."*
   Open the citation cards to inspect the verbatim source excerpts and view the RAG retrieval trace.
6. **Step 6 — Test Document CRUD:** In the sidebar, click the `🗑️` button to delete the document or click **`➕ Upload Another Document`** to index a second document.

---

## 🔬 Retrieval Strategy Comparison

| Retrieval Strategy | Exact Entity Matching | Conceptual Understanding | Noisy Chunk Filtering | Speed (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Lexical** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ~5ms |
| **Dense Semantic Vector** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~25ms |
| **Hybrid Search (RRF)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~30ms |
| **Hybrid + Cross-Encoder Rerank** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **~45ms** |

---

## 📄 License
This project is licensed under the Apache 2.0 License.
