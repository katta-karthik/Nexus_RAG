"""
Ingestion module for NexusRAG.
Handles document loading, text cleaning, chunking strategies, and metadata enrichment.
"""

from .loaders import DocumentLoader, LoadedDocument
from .cleaner import TextCleaner, CleaningStats
from .chunker import ChunkingStrategy, Chunker, Chunk
from .metadata import enrich_chunks

__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "TextCleaner",
    "CleaningStats",
    "ChunkingStrategy",
    "Chunker",
    "Chunk",
    "enrich_chunks",
]
