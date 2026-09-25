import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class BM25KeywordRetriever:
    """
    Lexical BM25 retriever for exact keyword and term frequency matching.
    Builds an in-memory inverted index over the current document chunks.
    """

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of chunks ({'chunk_id', 'text', 'metadata'}).
        """
        self.chunks = list(chunks)
        self.corpus_tokens = [self._tokenize(c["text"]) for c in self.chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks based on BM25 scores.
        """
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_pairs = []

        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]

            # Apply metadata filters if specified
            if filter_dict:
                meta = chunk.get("metadata", {})
                match = all(meta.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue

            if score > 0.0:
                scored_pairs.append((idx, float(score)))

        # Sort by BM25 score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        if not top_pairs:
            return []

        max_score = top_pairs[0][1] if top_pairs[0][1] > 0 else 1.0

        results = []
        for rank, (idx, raw_score) in enumerate(top_pairs, start=1):
            chunk = self.chunks[idx]
            norm_score = round(raw_score / max_score, 4) if max_score > 0 else 0.0
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": norm_score,
                    "raw_score": round(raw_score, 4),
                    "rank": rank,
                    "method": "keyword",
                }
            )

        return results
