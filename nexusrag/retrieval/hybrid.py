from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .semantic import SemanticRetriever
from .keyword import BM25KeywordRetriever


@dataclass
class HybridResult:
    query: str
    semantic_count: int
    keyword_count: int
    merged_count: int
    semantic_candidates: List[Dict[str, Any]]
    keyword_candidates: List[Dict[str, Any]]
    merged_candidates: List[Dict[str, Any]]


class HybridRetriever:
    """
    Executes hybrid retrieval combining dense semantic similarity and sparse BM25 scoring.
    Fuses rankings via Reciprocal Rank Fusion (RRF) and preserves multi-branch provenance.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever: BM25KeywordRetriever,
        rrf_k: int = 60,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int = 15,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> HybridResult:
        """
        Retrieves candidates from both semantic and keyword search, merges them via RRF,
        and returns the top_k fused candidates alongside candidate metrics.
        """
        # Fetch candidate pools from each retriever
        semantic_candidates = self.semantic_retriever.retrieve(
            query=query, top_k=candidate_pool_size, filter_dict=filter_dict
        )
        keyword_candidates = self.keyword_retriever.retrieve(
            query=query, top_k=candidate_pool_size, filter_dict=filter_dict
        )

        # Reciprocal Rank Fusion (RRF)
        # score(d) = sum(1 / (k + rank))
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        provenance: Dict[str, Dict[str, Any]] = {}

        # 1. Process Semantic candidates
        for rank, c in enumerate(semantic_candidates, start=1):
            cid = c["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_score
            chunk_lookup[cid] = c
            provenance[cid] = {
                "in_semantic": True,
                "in_keyword": False,
                "semantic_rank": rank,
                "semantic_score": c["score"],
                "keyword_rank": None,
                "keyword_score": None,
            }

        # 2. Process Keyword candidates
        for rank, c in enumerate(keyword_candidates, start=1):
            cid = c["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_score
            if cid not in chunk_lookup:
                chunk_lookup[cid] = c
                provenance[cid] = {
                    "in_semantic": False,
                    "in_keyword": True,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "keyword_rank": rank,
                    "keyword_score": c["score"],
                }
            else:
                provenance[cid]["in_keyword"] = True
                provenance[cid]["keyword_rank"] = rank
                provenance[cid]["keyword_score"] = c["score"]

        # Sort by fused RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        max_rrf = fused_scores[sorted_cids[0]] if sorted_cids else 1.0

        merged_candidates: List[Dict[str, Any]] = []
        for new_rank, cid in enumerate(sorted_cids, start=1):
            base_chunk = chunk_lookup[cid]
            prov = provenance[cid]
            # Normalize RRF score to [0, 1] relative to top candidate
            norm_score = round(fused_scores[cid] / max_rrf, 4) if max_rrf > 0 else 1.0

            merged_candidates.append(
                {
                    "chunk_id": cid,
                    "text": base_chunk["text"],
                    "metadata": base_chunk["metadata"],
                    "score": norm_score,
                    "raw_rrf_score": round(fused_scores[cid], 5),
                    "rank": new_rank,
                    "method": "hybrid_rrf",
                    "provenance": prov,
                }
            )

        final_candidates = merged_candidates[:top_k]

        return HybridResult(
            query=query,
            semantic_count=len(semantic_candidates),
            keyword_count=len(keyword_candidates),
            merged_count=len(merged_candidates),
            semantic_candidates=semantic_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=final_candidates,
        )
