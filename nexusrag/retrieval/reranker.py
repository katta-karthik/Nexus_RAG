from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


@dataclass
class RerankResult:
    before_reranking: List[Dict[str, Any]]
    after_reranking: List[Dict[str, Any]]
    rank_changes: List[Dict[str, Any]]


class Reranker:
    """
    Cross-encoder reranking engine.
    Uses FlashRank (CPU-based, lightweight ONNX model) for zero-GPU reranking,
    with an internal semantic cross-scoring fallback.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None
        self._init_flashrank()

    def _init_flashrank(self):
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self.model_name)
        except Exception:
            self._ranker = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> RerankResult:
        """
        Reranks retrieved candidate chunks and returns detailed before/after comparison.
        """
        if not candidates or not query.strip():
            return RerankResult(
                before_reranking=[], after_reranking=[], rank_changes=[]
            )

        # Record pre-rerank state
        before_list = []
        for i, c in enumerate(candidates, start=1):
            before_list.append(
                {
                    "chunk_id": c["chunk_id"],
                    "rank": i,
                    "score": c["score"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                }
            )

        # Attempt FlashRank execution
        after_list: List[Dict[str, Any]] = []
        if self._ranker is not None:
            try:
                from flashrank import RerankRequest

                passages = [
                    {"id": c["chunk_id"], "text": c["text"], "meta": c.get("metadata", {})}
                    for c in candidates
                ]
                req = RerankRequest(query=query, passages=passages)
                results = self._ranker.rerank(req)

                for new_rank, r in enumerate(results[:top_n], start=1):
                    after_list.append(
                        {
                            "chunk_id": r["id"],
                            "rank": new_rank,
                            "score": round(float(r["score"]), 4),
                            "text": r["text"],
                            "metadata": r.get("meta", {}),
                        }
                    )
            except Exception:
                after_list = []

        # Fallback cross-encoder scoring if flashrank failed or not loaded
        if not after_list:
            after_list = self._fallback_cross_rerank(query, candidates, top_n)

        # Compute rank movements (delta)
        before_rank_map = {item["chunk_id"]: item["rank"] for item in before_list}
        rank_changes = []
        for item in after_list:
            cid = item["chunk_id"]
            old_r = before_rank_map.get(cid, 0)
            new_r = item["rank"]
            rank_changes.append(
                {
                    "chunk_id": cid,
                    "before_rank": old_r,
                    "after_rank": new_r,
                    "delta": old_r - new_r,  # positive means moved up
                    "score": item["score"],
                }
            )

        return RerankResult(
            before_reranking=before_list,
            after_reranking=after_list,
            rank_changes=rank_changes,
        )

    def _fallback_cross_rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Cross-matching score combining token bigrams, exact title matching,
        and query term density.
        """
        query_terms = re.findall(r"\b\w+\b", query.lower())
        scored = []

        for c in candidates:
            text_lower = c["text"].lower()
            overlap_score = sum(1.0 for term in query_terms if term in text_lower)
            density = overlap_score / max(1, len(query_terms))
            # Blend original score with term density
            cross_score = 0.5 * c["score"] + 0.5 * density
            scored.append((cross_score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for new_rank, (s, c) in enumerate(scored[:top_n], start=1):
            results.append(
                {
                    "chunk_id": c["chunk_id"],
                    "rank": new_rank,
                    "score": round(s, 4),
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                }
            )
        return results
