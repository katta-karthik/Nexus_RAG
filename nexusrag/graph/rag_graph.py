from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from nexusrag.retrieval.semantic import SemanticRetriever
from nexusrag.retrieval.keyword import BM25KeywordRetriever
from nexusrag.retrieval.hybrid import HybridRetriever
from nexusrag.retrieval.query_rewrite import QueryRewriter
from nexusrag.retrieval.reranker import Reranker
from nexusrag.generation.answer import AnswerGenerator


class RAGState(TypedDict):
    original_query: str
    current_query: str
    enable_rewrite: bool
    enable_rerank: bool
    strategy: str  # 'semantic', 'keyword', 'hybrid'
    top_k: int
    filter_dict: Optional[Dict[str, Any]]

    # Pipeline artifacts
    rewritten_query: Optional[str]
    retrieved_candidates: List[Dict[str, Any]]
    semantic_count: int
    keyword_count: int
    rerank_before: List[Dict[str, Any]]
    rerank_after: List[Dict[str, Any]]
    selected_context: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    trace: List[str]


def create_rag_graph(
    semantic_retriever: SemanticRetriever,
    keyword_retriever: BM25KeywordRetriever,
    hybrid_retriever: HybridRetriever,
    query_rewriter: QueryRewriter,
    reranker: Reranker,
    generator: AnswerGenerator,
):
    """
    Constructs a LangGraph StateGraph modeling the complete RAG query lifecycle:
    Query -> Analyze -> Rewrite? -> Retrieve -> Rerank? -> Select Context -> Generate -> Citations
    """

    def analyze_node(state: RAGState) -> Dict[str, Any]:
        q = state["original_query"]
        trace = list(state.get("trace", []))
        trace.append(f"✓ Analyzed query: '{q[:50]}...'")
        return {
            "current_query": q,
            "trace": trace,
        }

    def rewrite_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        rewritten = query_rewriter.rewrite(state["original_query"])
        trace.append(f"✓ Query Rewritten: '{rewritten[:50]}...'")
        return {
            "current_query": rewritten,
            "rewritten_query": rewritten,
            "trace": trace,
        }

    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        q = state["current_query"]
        strat = state.get("strategy", "hybrid")
        k = state.get("top_k", 5)
        flt = state.get("filter_dict")

        candidates = []
        sem_count = 0
        kw_count = 0

        if strat == "semantic":
            candidates = semantic_retriever.retrieve(q, top_k=k, filter_dict=flt)
            sem_count = len(candidates)
            trace.append(f"✓ Semantic Search retrieved {len(candidates)} chunks")

        elif strat == "keyword":
            candidates = keyword_retriever.retrieve(q, top_k=k, filter_dict=flt)
            kw_count = len(candidates)
            trace.append(f"✓ Keyword BM25 retrieved {len(candidates)} chunks")

        else:  # hybrid
            # Retrieve a slightly larger pool for reranking or final selection
            pool_size = max(10, k * 2)
            hybrid_res = hybrid_retriever.retrieve(
                q, top_k=pool_size, candidate_pool_size=pool_size, filter_dict=flt
            )
            candidates = hybrid_res.merged_candidates
            sem_count = hybrid_res.semantic_count
            kw_count = hybrid_res.keyword_count
            trace.append(
                f"✓ Hybrid Search fused {sem_count} semantic + {kw_count} keyword candidates ({len(candidates)} merged)"
            )

        return {
            "retrieved_candidates": candidates,
            "semantic_count": sem_count,
            "keyword_count": kw_count,
            "trace": trace,
        }

    def rerank_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        candidates = state.get("retrieved_candidates", [])
        q = state["current_query"]
        k = state.get("top_k", 5)

        rerank_res = reranker.rerank(q, candidates, top_n=k)
        trace.append(f"✓ Reranked {len(candidates)} candidates down to top {len(rerank_res.after_reranking)}")

        return {
            "rerank_before": rerank_res.before_reranking,
            "rerank_after": rerank_res.after_reranking,
            "retrieved_candidates": rerank_res.after_reranking,
            "trace": trace,
        }

    def select_context_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        candidates = state.get("retrieved_candidates", [])
        k = state.get("top_k", 5)
        selected = candidates[:k]
        trace.append(f"✓ Selected {len(selected)} context chunks for prompt synthesis")
        return {
            "selected_context": selected,
            "trace": trace,
        }

    def generate_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        q = state["original_query"]
        selected = state.get("selected_context", [])

        tokens = list(generator.stream_answer(q, selected))
        answer = "".join(tokens)
        trace.append("✓ Generated grounded response")

        return {
            "answer": answer,
            "trace": trace,
        }

    def citations_node(state: RAGState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        selected = state.get("selected_context", [])
        citations = [c.to_dict() for c in generator.extract_citations(selected)]
        trace.append(f"✓ Extracted {len(citations)} citations")
        return {
            "citations": citations,
            "trace": trace,
        }

    # Conditional branching logic
    def should_rewrite(state: RAGState) -> str:
        return "rewrite" if state.get("enable_rewrite", False) else "retrieve"

    def should_rerank(state: RAGState) -> str:
        return "rerank" if state.get("enable_rerank", False) else "select_context"

    # Build StateGraph
    builder = StateGraph(RAGState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("select_context", select_context_node)
    builder.add_node("generate", generate_node)
    builder.add_node("citations", citations_node)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", should_rewrite, {"rewrite": "rewrite", "retrieve": "retrieve"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges("retrieve", should_rerank, {"rerank": "rerank", "select_context": "select_context"})
    builder.add_edge("rerank", "select_context")
    builder.add_edge("select_context", "generate")
    builder.add_edge("generate", "citations")
    builder.add_edge("citations", END)

    return builder.compile()
