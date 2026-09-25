import hashlib
import time
from typing import List
from .chunker import Chunk


def enrich_chunks(chunks: List[Chunk], doc_type: str = "txt") -> List[Chunk]:
    """
    Enriches each Chunk with detailed provenance metadata, token estimates,
    and formatted display labels for citations.
    """
    timestamp = int(time.time())

    for chunk in chunks:
        # Generate stable chunk hash
        text_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:10]
        token_estimate = max(1, len(chunk.text) // 4)
        citation_label = f"{chunk.filename} — Page {chunk.page_number}"

        chunk.metadata.update(
            {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "strategy": chunk.strategy,
                "char_count": chunk.char_count,
                "word_count": chunk.word_count,
                "token_estimate": token_estimate,
                "content_hash": text_hash,
                "doc_type": doc_type,
                "indexed_at": timestamp,
                "citation_label": citation_label,
            }
        )

    return chunks
