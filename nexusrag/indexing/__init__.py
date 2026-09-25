"""
Indexing module for NexusRAG.
Handles multi-provider embedding generation and Chroma vector store management.
"""

from .embeddings import EmbeddingManager, EmbeddingProvider
from .vectorstore import VectorStoreManager

__all__ = ["EmbeddingManager", "EmbeddingProvider", "VectorStoreManager"]
