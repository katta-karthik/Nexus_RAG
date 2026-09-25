import os
import sys
import unittest

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexusrag.ingestion.loaders import DocumentLoader
from nexusrag.ingestion.cleaner import TextCleaner
from nexusrag.ingestion.chunker import Chunker, ChunkingStrategy
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
from nexusrag.graph.rag_graph import create_rag_graph


class TestRAGLifecycle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "sample_docs",
                "attention_is_all_you_need_summary.txt",
            )
        )
        cls.test_db_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "test_chroma_db")
        )

    def test_01_document_loading_and_cleaning(self):
        loaded = DocumentLoader.load(self.sample_path, "attention_summary.txt")
        self.assertGreater(loaded.total_characters, 500)
        self.assertGreater(loaded.total_pages, 0)

        cleaned, stats = TextCleaner.clean_document(loaded)
        self.assertEqual(cleaned.total_pages, loaded.total_pages)
        self.assertGreaterEqual(stats.chars_before, stats.chars_after)
        print(f"\n[Test 1] Loaded {loaded.total_characters} chars -> Cleaned {cleaned.total_characters} chars.")

    def test_02_chunking_strategies(self):
        loaded = DocumentLoader.load(self.sample_path, "attention_summary.txt")
        cleaned, _ = TextCleaner.clean_document(loaded)

        # Strategy 1: Recursive
        chunks_rec = Chunker.chunk_document(cleaned, ChunkingStrategy.RECURSIVE, chunk_size=600, chunk_overlap=80)
        self.assertGreater(len(chunks_rec), 0)

        # Strategy 2: Paragraph Semantic
        chunks_sem = Chunker.chunk_document(cleaned, ChunkingStrategy.PARAGRAPH_SEMANTIC, chunk_size=600, chunk_overlap=80)
        self.assertGreater(len(chunks_sem), 0)

        enriched = enrich_chunks(chunks_rec)
        self.assertIn("citation_label", enriched[0].metadata)
        print(f"\n[Test 2] Recursive: {len(chunks_rec)} chunks, Paragraph Semantic: {len(chunks_sem)} chunks.")

    def test_03_embeddings_and_vectorstore(self):
        loaded = DocumentLoader.load(self.sample_path, "attention_summary.txt")
        cleaned, _ = TextCleaner.clean_document(loaded)
        chunks = enrich_chunks(Chunker.chunk_document(cleaned, ChunkingStrategy.RECURSIVE, chunk_size=600, chunk_overlap=80))

        embedder = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
        vectors = embedder.embed_documents_with_progress([c.text for c in chunks])
        self.assertEqual(len(vectors), len(chunks))

        vstore = VectorStoreManager(persist_directory=self.test_db_dir, collection_name="test_collection")
        vstore.clear()
        added = vstore.add_chunks(chunks, vectors)
        self.assertEqual(added, len(chunks))

        stats = vstore.get_stats()
        self.assertEqual(stats["total_vectors"], len(chunks))
        print(f"\n[Test 3] Indexed {added} vectors into test collection.")

    def test_04_retrieval_and_reranking(self):
        embedder = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
        vstore = VectorStoreManager(persist_directory=self.test_db_dir, collection_name="test_collection")
        semantic_retriever = SemanticRetriever(vstore, embedder)

        all_chunks = vstore.get_chunks_for_document(limit=100)
        keyword_retriever = BM25KeywordRetriever(all_chunks)
        hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
        reranker = Reranker()

        query = "What is scaled dot product attention and sqrt d_k?"
        hybrid_res = hybrid_retriever.retrieve(query, top_k=5)
        self.assertGreater(len(hybrid_res.merged_candidates), 0)

        rerank_res = reranker.rerank(query, hybrid_res.merged_candidates, top_n=3)
        self.assertEqual(len(rerank_res.after_reranking), min(3, len(hybrid_res.merged_candidates)))
        print(f"\n[Test 4] Hybrid retrieved {hybrid_res.merged_count} candidates -> Reranked to top {len(rerank_res.after_reranking)}.")

    def test_05_grounded_generation_and_citations(self):
        vstore = VectorStoreManager(persist_directory=self.test_db_dir, collection_name="test_collection")
        chunks = vstore.get_chunks_for_document(limit=3)

        generator = AnswerGenerator(provider="demo")
        query = "Explain the scaling factor in dot product attention"
        tokens = list(generator.stream_answer(query, chunks))
        full_answer = "".join(tokens)
        self.assertGreater(len(full_answer), 20)

        citations = generator.extract_citations(chunks)
        self.assertEqual(len(citations), len(chunks))
        print(f"\n[Test 5] Generated answer ({len(full_answer)} chars) with {len(citations)} citations.")

    def test_06_langgraph_pipeline(self):
        embedder = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
        vstore = VectorStoreManager(persist_directory=self.test_db_dir, collection_name="test_collection")
        semantic_retriever = SemanticRetriever(vstore, embedder)
        all_chunks = vstore.get_chunks_for_document(limit=100)
        keyword_retriever = BM25KeywordRetriever(all_chunks)
        hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
        query_rewriter = QueryRewriter()
        reranker = Reranker()
        generator = AnswerGenerator(provider="demo")

        rag_graph = create_rag_graph(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            hybrid_retriever=hybrid_retriever,
            query_rewriter=query_rewriter,
            reranker=reranker,
            generator=generator,
        )

        initial_state = {
            "original_query": "What is the Transformer architecture encoder stack?",
            "current_query": "",
            "enable_rewrite": True,
            "enable_rerank": True,
            "strategy": "hybrid",
            "top_k": 3,
            "filter_dict": None,
            "rewritten_query": None,
            "retrieved_candidates": [],
            "semantic_count": 0,
            "keyword_count": 0,
            "rerank_before": [],
            "rerank_after": [],
            "selected_context": [],
            "answer": "",
            "citations": [],
            "trace": [],
        }

        final_state = rag_graph.invoke(initial_state)
        self.assertGreater(len(final_state["answer"]), 0)
        self.assertGreater(len(final_state["citations"]), 0)
        self.assertGreater(len(final_state["trace"]), 3)
        safe_trace = [t.encode('ascii', errors='replace').decode('ascii') for t in final_state["trace"]]
        print(f"\n[Test 6] LangGraph completed execution trace:\n" + "\n".join(f"  {t}" for t in safe_trace))

    def test_07_rag_evaluation(self):
        embedder = EmbeddingManager(provider=EmbeddingProvider.LOCAL)
        vstore = VectorStoreManager(persist_directory=self.test_db_dir, collection_name="test_collection")
        semantic_retriever = SemanticRetriever(vstore, embedder)
        all_chunks = vstore.get_chunks_for_document(limit=100)
        keyword_retriever = BM25KeywordRetriever(all_chunks)
        hybrid_retriever = HybridRetriever(semantic_retriever, keyword_retriever)
        reranker = Reranker()
        generator = AnswerGenerator(provider="demo")

        evaluator = RAGEvaluator(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
        )

        samples = load_benchmark_samples()
        df_comp = evaluator.run_strategy_comparison(samples=samples, top_k=3, max_samples=3)
        self.assertEqual(len(df_comp), 4)  # 4 strategies
        print(f"\n[Test 7] RAG Strategy Benchmark Comparison:\n{df_comp.to_string(index=False)}")


if __name__ == "__main__":
    unittest.main()
