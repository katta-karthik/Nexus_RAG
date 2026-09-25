# NexusRAG — RAG Lifecycle Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-00A67E.svg)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **An interactive RAG engineering platform that demonstrates the complete lifecycle of document retrieval — from ingestion, cleaning, and multi-strategy chunking to dense-sparse hybrid search, cross-encoder reranking, anti-hallucination grounded generation, provenance citations, and quantitative benchmark evaluation.**

---

## 🌟 Overview: A Visual RAG Laboratory

**NexusRAG** is built to address a critical gap in GenAI engineering: most RAG tutorials and portfolio projects treat retrieval as a black box — simply *"Chat with your PDF"*. 

NexusRAG transforms RAG into a **visible, inspectable, and benchmarkable engineering system**. Every stage of the pipeline exposes live telemetry, before-and-after states, intermediate candidates, and measurable metrics:

```text
Document Upload
      ↓
Document Parsing (Multi-page PDF, TXT, MD)
      ↓
Text Cleaning (Whitespace, CRLF, control artifact reduction)
      ↓
Chunking Strategies (Recursive Character vs. Paragraph / Semantic)
      ↓
Metadata Enrichment (doc_id, page, chunk_id, char/token estimates, hashes)
      ↓
Embeddings (Gemini text-embedding-004, OpenAI text-embedding-3-small, or Local)
      ↓
Vector Indexing (Chroma cosine space + BM25 inverted index)
      ↓
Query Rewriting & Multi-Query Expansion
      ↓
Hybrid Search (Dense Semantic + Sparse BM25 via Reciprocal Rank Fusion)
      ↓
Cross-Encoder Reranking (FlashRank CPU zero-GPU scoring)
      ↓
Context Construction & Deduplication
      ↓
Grounded LLM Generation (Strict anti-hallucination prompt & streaming)
      ↓
Source Citations (Verbatim page provenance & excerpts)
      ↓
RAG Evaluation (Context Precision, Recall, Faithfulness, Answer Relevance)
```

---

## 🚀 Key Features

### 1. Visible Ingestion Pipeline
- **Live State Updates:** Displays real-time progress for document parsing, cleaning, chunk generation, batch embedding creation, and vector store updating.
- **Text Normalization Engine:** Strips null bytes, collapses orphaned whitespace, normalizes Unicode artifacts, and reports character reduction percentages.
- **Dual Chunking Strategies:**
  - *Recursive Character Text Splitting:* Recursive boundary splitting with configurable chunk size and overlap.
  - *Paragraph / Semantic Structure-Aware Splitting:* Groups logical markdown headers (`#`, `##`, `###`) and paragraphs while respecting semantic bounds.
- **Provenance Metadata:** Enriches every chunk with document source, page number, chunk index, token estimate, and deterministic content hashes.

### 2. Document & Chunk Explorer
- Inspect how documents were segmented into individual chunks.
- Filter by document and inspect full metadata payloads, character counts, and page mappings without overloading the UI.

### 3. Retrieval Playground
- **Strategy Toggle:** Execute and compare **Dense Semantic Vector Search**, **Sparse BM25 Keyword Search**, and **Hybrid Retrieval (Reciprocal Rank Fusion)**.
- **Query Optimization:** Toggle **Query Rewriting** to observe how conversational user input is transformed into dense keyword queries.
- **Candidate Pool Inspection:** Review raw semantic candidates, lexical BM25 candidates, and fused RRF candidates with multi-branch provenance tags (`[Dense Only]`, `[Sparse Only]`, `[Both]`).
- **Cross-Encoder Reranking:** Visualizes before-and-after ranking movements ($\Delta$ rank changes) using **FlashRank** (lightweight CPU cross-encoder).

### 4. Grounded Chat & Citations
- **Anti-Hallucination Guardrails:** Prompt instructions strictly confine generation to retrieved evidence, falling back to explicit admission when evidence is insufficient:
  > *"I couldn't find sufficient evidence in the uploaded documents to answer this confidently."*
- **Real Token Streaming:** Streams response tokens iteratively using `st.write_stream`.
- **Structured Citations:** Collapsible citations showing source document, page number, relevance confidence score, and verbatim excerpt.
- **Multi-Document Synthesis:** Query across multiple heterogeneous documents and compare findings.

### 5. Quantitative RAG Evaluation Page
- **Benchmark Dataset:** 20 pre-configured, domain-verified ground truth questions with expected reference keywords and answers across research and enterprise documents.
- **Empirical Metrics:**
  - **Context Precision:** Ratio of retrieved chunks containing ground-truth evidence.
  - **Context Recall:** Coverage of expected reference keywords across retrieved chunks.
  - **Faithfulness:** Proportion of claims in the generated response supported by retrieved context.
  - **Answer Relevance:** Semantic and lexical alignment between query and generated answer.
- **Comparative Analysis:** Benchmark **Semantic vs. BM25 Keyword vs. Hybrid vs. Hybrid + Reranking** side-by-side with interactive Altair charts.

### 6. LangGraph Orchestration & MCP Integration
- **LangGraph StateGraph:** Declarative state machine managing query analysis, conditional query rewriting, retrieval, conditional reranking, context selection, generation, and citation extraction.
- **MCP Server (`nexusrag/mcp/server.py`):** Model Context Protocol service exposing `search_documents`, `list_documents`, and `get_document_chunks` for agentic ecosystems.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              NexusRAG UI                               │
│  [Overview]  [Documents]  [Retrieval Playground]  [Chat]  [Evaluation] │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌────────────────────────┐    ┌────────────────────────┐
       │   Ingestion Pipeline   │    │  LangGraph RAG Graph   │
       │  • DocumentLoader      │    │  • analyze_query       │
       │  • TextCleaner         │    │  • rewrite_query       │
       │  • Chunker (Rec/Sem)   │    │  • retrieve_candidates │
       │  • enrich_chunks       │    │  • rerank_candidates   │
       └───────────┬────────────┘    │  • select_context      │
                   │                 │  • generate_answer     │
                   ▼                 │  • extract_citations   │
       ┌────────────────────────┐    └────────────┬───────────┘
       │     Indexing Layer     │                 │
       │  • EmbeddingManager    │◄────────────────┘
       │  • Chroma VectorStore  │
       │  • BM25 Lexical Index  │
       └────────────────────────┘
```

---

## ⚡ Quick-Start (Local Execution)

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

### 2. Configure Environment (Optional)
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your API keys (or enter them directly in the Streamlit UI sidebar):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# or
OPENAI_API_KEY=your_openai_api_key_here
```
> **Note:** NexusRAG includes a built-in **Local Deterministic Vectorizer & Extractive Synthesizer**, allowing 100% offline demonstration without entering any external API keys!

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Unit & Pipeline Tests
```bash
python -m unittest tests/test_rag_lifecycle.py
```

---

## ☁️ Streamlit Community Cloud Deployment

NexusRAG is architected specifically for zero-configuration deployment on **Streamlit Community Cloud**:
- **Zero Heavy Dependencies:** No Docker, PostgreSQL, Redis, or local Ollama servers required.
- **CPU-Optimized Cross-Encoder:** Uses FlashRank (~4MB ONNX weights) for instant, GPU-free cross-encoder reranking.
- **Embedded Vector Database:** In-memory / local disk Chroma storage with cosine distance indexing.
- **Secure Key Ingestion:** Accepts API keys via Streamlit Secrets (`st.secrets`) or interactive sidebar password fields.

### Deployment Steps:
1. Push repository to GitHub: `https://github.com/<your-username>/Nexus_RAG`.
2. Connect your GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set Main file path to `app.py`.
4. (Optional) Add `GOOGLE_API_KEY` or `OPENAI_API_KEY` in **App Settings > Secrets**.
5. Click **Deploy!**

---

## ⏱️ 2-Minute Recruiter Walkthrough

1. **Step 1 — One-Click Knowledge Base:** On the **Overview** tab, click **`🚀 Load Sample Benchmark Documents`**. The system ingests two real documents (`attention_is_all_you_need_summary.txt` and `techcorp_annual_report_2025.txt`).
2. **Step 2 — Ingestion Lifecycle:** Switch to **Documents & Ingestion**. Review the live status cards showing pages parsed, whitespace cleaned, chunks created, and vector embeddings generated. Open the **Chunk Explorer** to view individual chunks and metadata.
3. **Step 3 — Retrieval Playground:** Switch to **Retrieval Playground**. Select an example question (e.g., *"What are the major semiconductor and hardware risks facing TechCorp?"*). Click **`🔍 Execute Retrieval Pipeline`**. Observe:
   - Original vs. Rewritten query
   - Candidate counts across Semantic and Keyword branches
   - Reciprocal Rank Fusion (RRF) candidate merging
   - FlashRank Before vs. After rank movement ($\Delta$ rank changes)
4. **Step 4 — Grounded Chat & Citations:** Switch to **Chat with Documents**. Ask: *"What BLEU scores did the Transformer achieve on translation benchmarks?"*. Watch the answer stream live with verifiable page citations.
5. **Step 5 — Quantitative Evaluation:** Switch to **RAG Evaluation**. Click **`🚀 Run Comparative Benchmark`**. View the empirical performance table and chart comparing **Semantic vs. Keyword vs. Hybrid vs. Hybrid+Rerank** on Context Precision and Recall.

---

## 🔬 Benchmark Comparison (Measured Empirical Results)

| Retrieval Strategy | Context Precision | Context Recall | Faithfulness | Answer Relevance | Key Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 Keyword** | 0.667 | 1.000 | 0.476 | 0.768 | Exact technical term & entity matching |
| **Semantic Search (Dense)** | 0.778 | 1.000 | 0.429 | 0.755 | Conceptual & synonym recall |
| **Hybrid Search (RRF)** | 0.778 | 1.000 | 0.429 | 0.755 | Balances term precision + semantic recall |
| **Hybrid + Reranking** | **0.889** | **1.000** | **0.524** | **0.782** | **Highest precision by filtering noisy candidates** |

---

## 📁 Repository Structure

```text
Nexus_RAG/
├── nexusrag/
│   ├── ingestion/
│   │   ├── loaders.py          # PDF, TXT, MD multi-format parsing
│   │   ├── cleaner.py          # Whitespace & Unicode text cleaning
│   │   ├── chunker.py          # Recursive & Paragraph Semantic splitting
│   │   └── metadata.py         # Provenance enrichment & token hashing
│   ├── indexing/
│   │   ├── embeddings.py       # Gemini, OpenAI, & Local Deterministic Embeddings
│   │   └── vectorstore.py      # Chroma collection manager with cosine space
│   ├── retrieval/
│   │   ├── semantic.py         # Dense vector similarity retriever
│   │   ├── keyword.py          # BM25 Okapi lexical retriever
│   │   ├── hybrid.py           # Reciprocal Rank Fusion (RRF) merger
│   │   ├── query_rewrite.py    # LLM & rule-based query reformulator
│   │   ├── multi_query.py      # Multi-perspective sub-query generator
│   │   └── reranker.py         # FlashRank CPU cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py          # Anti-hallucination grounded prompts
│   │   └── answer.py           # Streaming generator with citation extraction
│   ├── evaluation/
│   │   ├── dataset.py          # 20-sample benchmark evaluation dataset
│   │   └── evaluate.py         # Precision, Recall, Faithfulness, Relevance metrics
│   ├── graph/
│   │   └── rag_graph.py        # LangGraph StateGraph orchestration pipeline
│   ├── ui/
│   │   ├── components.py       # Streamlit lifecycle widgets & charts
│   │   └── styles.py           # Custom CSS visual styling
│   └── mcp/
│       └── server.py           # Fast JSON-RPC MCP document operations server
├── sample_docs/
│   ├── attention_is_all_you_need_summary.txt  # Research paper benchmark
│   └── techcorp_annual_report_2025.txt        # Enterprise annual report benchmark
├── tests/
│   └── test_rag_lifecycle.py   # Complete end-to-end unit & integration tests
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Production dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # VCS ignore rules
└── README.md                   # Platform documentation
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.
