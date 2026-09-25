import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from nexusrag.ingestion.chunker import Chunk


class VectorStoreManager:
    """
    Manages the Chroma vector database collection for NexusRAG.
    Supports persistent and in-memory indexing, similarity search with cosine distance,
    chunk inspection for the UI explorer, and live metrics.
    """

    DEFAULT_COLLECTION = "nexusrag_documents"

    def __init__(
        self,
        persist_directory: Optional[str] = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = self._init_client()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_client(self) -> chromadb.ClientAPI:
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            return chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

    def add_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> int:
        """
        Adds chunks and precomputed embeddings to the Chroma collection.
        Returns the number of chunks added.
        """
        if not chunks or not embeddings:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        # Chroma metadata values must be primitive types (str, int, float, bool)
        metadatas = []
        for c in chunks:
            clean_meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search. Returns list of result dicts:
        [{'chunk_id', 'text', 'metadata', 'score', 'distance'}]
        Note: With cosine space in Chroma, distance is 1 - cosine_similarity.
        Score is converted to similarity: 1.0 - distance.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=actual_k,
            where=filter_dict if filter_dict else None,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance is in [0, 2]; similarity = max(0.0, 1.0 - dist)
            score = max(0.0, 1.0 - dist)
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                    "score": round(float(score), 4),
                    "distance": round(float(dist), 4),
                }
            )

        return output

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns stats about the current vector collection.
        """
        count = self.collection.count()
        docs = set()
        if count > 0:
            # Sample up to 1000 items to find unique document sources
            sample = self.collection.get(limit=min(1000, count), include=["metadatas"])
            if sample["metadatas"]:
                for m in sample["metadatas"]:
                    if m and "filename" in m:
                        docs.add(m["filename"])

        return {
            "collection_name": self.collection_name,
            "total_vectors": count,
            "unique_documents": len(docs),
            "document_names": sorted(list(docs)),
            "status": "Ready" if count > 0 else "Empty",
        }

    def get_chunks_for_document(
        self, filename: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetches a slice of chunks for UI inspection in the Chunk Explorer.
        """
        count = self.collection.count()
        if count == 0:
            return []

        where_clause = {"filename": filename} if filename else None
        res = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas"],
        )

        chunks_out = []
        if res["ids"]:
            for chunk_id, doc_text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                chunks_out.append(
                    {
                        "chunk_id": chunk_id,
                        "text": doc_text,
                        "metadata": meta or {},
                    }
                )

        # Sort by chunk_index if present
        chunks_out.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks_out

    def delete_document(self, filename: str) -> None:
        """Deletes all chunks associated with a given filename."""
        self.collection.delete(where={"filename": filename})

    def clear(self) -> None:
        """Empties the current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
