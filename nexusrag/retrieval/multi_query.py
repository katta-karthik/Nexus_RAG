from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage


class MultiQueryRetriever:
    """
    Generates 2-3 alternative perspectives or sub-queries for a user prompt,
    executes retrieval across all variations, and produces a deduplicated,
    fused candidate pool.
    """

    SYSTEM_PROMPT = (
        "You are an AI language model assistant for a search retrieval system.\n"
        "Your task is to generate 2 to 3 distinct perspectives or alternative queries "
        "for the user's question to retrieve relevant documents from a vector database.\n"
        "Output each query on a new line. Do not number them. Do not add intro or outro text."
    )

    def __init__(self, base_retriever: Any, llm: Optional[Any] = None):
        self.base_retriever = base_retriever
        self.llm = llm

    def generate_queries(self, original_query: str) -> List[str]:
        """
        Generates 2-3 query variations.
        """
        queries = [original_query]

        if self.llm is not None:
            try:
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=original_query),
                ]
                resp = self.llm.invoke(messages)
                content = resp.content if hasattr(resp, "content") else str(resp)
                lines = [line.strip().lstrip("123456789.-* ") for line in content.split("\n") if line.strip()]
                for line in lines[:3]:
                    if line and line.lower() != original_query.lower() and line not in queries:
                        queries.append(line)
            except Exception:
                pass

        # If LLM unavailable or failed, create synthetic perspective variations
        if len(queries) == 1:
            queries.append(f"{original_query} methodology and technical architecture")
            queries.append(f"{original_query} key metrics and experimental results")

        return queries[:3]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates multi-queries, executes retrieval across all, and fuses candidates.
        """
        sub_queries = self.generate_queries(query)
        all_candidates: Dict[str, Dict[str, Any]] = {}

        for sub_q in sub_queries:
            results = self.base_retriever.retrieve(sub_q, top_k=top_k, filter_dict=filter_dict)
            for r in results:
                cid = r["chunk_id"]
                if cid not in all_candidates:
                    all_candidates[cid] = dict(r)
                else:
                    # Accumulate score
                    all_candidates[cid]["score"] = max(all_candidates[cid]["score"], r["score"])

        sorted_candidates = sorted(
            all_candidates.values(), key=lambda x: x["score"], reverse=True
        )

        for rank, c in enumerate(sorted_candidates, start=1):
            c["rank"] = rank

        return {
            "original_query": query,
            "generated_queries": sub_queries,
            "candidates": sorted_candidates[:top_k],
        }
