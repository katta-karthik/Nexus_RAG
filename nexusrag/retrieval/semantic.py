from typing import List, Dict, Any, Optional
from nexusrag.indexing.embeddings import EmbeddingManager
from nexusrag.indexing.vectorstore import VectorStoreManager


class SemanticRetriever:
    """
    Executes dense vector similarity search across indexed document chunks.
    """

    def __init__(self, vectorstore: VectorStoreManager, embedder: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embeds the query and queries Chroma for the nearest top_k neighbors.
        Returns a list of standardized candidate dicts.
        """
        if not query.strip():
            return []

        query_vec = self.embedder.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(
            query_vector=query_vec,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        candidates = []
        for rank, r in enumerate(results, start=1):
            candidates.append(
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": r["score"],
                    "rank": rank,
                    "method": "semantic",
                }
            )

        return candidates
