from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loaders import LoadedDocument


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    PARAGRAPH_SEMANTIC = "paragraph_semantic"


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    text: str
    doc_id: str
    filename: str
    page_number: int
    char_count: int
    word_count: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


class Chunker:
    """
    Implements multiple chunking strategies:
    1. Recursive Character Text Splitting (LangChain)
    2. Paragraph / Structure-Aware Semantic Splitting
    """

    @classmethod
    def chunk_document(
        cls,
        doc: LoadedDocument,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> List[Chunk]:
        """
        Chunks a document across its pages, tagging each chunk with page information
        and document metadata.
        """
        if strategy == ChunkingStrategy.RECURSIVE:
            return cls._chunk_recursive(doc, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH_SEMANTIC:
            return cls._chunk_paragraph_semantic(doc, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    @classmethod
    def _chunk_recursive(
        cls, doc: LoadedDocument, chunk_size: int, chunk_overlap: int
    ) -> List[Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
        )

        all_chunks: List[Chunk] = []
        global_idx = 0

        for page in doc.pages:
            if not page.text.strip():
                continue

            raw_splits = splitter.split_text(page.text)
            for split_text in raw_splits:
                cleaned_split = split_text.strip()
                if not cleaned_split:
                    continue

                chunk_id = f"{doc.doc_id}_c{global_idx + 1}"
                chunk = Chunk(
                    chunk_id=chunk_id,
                    chunk_index=global_idx + 1,
                    text=cleaned_split,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page_number=page.page_number,
                    char_count=len(cleaned_split),
                    word_count=len(cleaned_split.split()),
                    strategy=ChunkingStrategy.RECURSIVE.value,
                    metadata={
                        "source": doc.filename,
                        "doc_id": doc.doc_id,
                        "page": page.page_number,
                        "chunk_id": chunk_id,
                        "chunk_index": global_idx + 1,
                        "strategy": ChunkingStrategy.RECURSIVE.value,
                    },
                )
                all_chunks.append(chunk)
                global_idx += 1

        return all_chunks

    @classmethod
    def _chunk_paragraph_semantic(
        cls, doc: LoadedDocument, chunk_size: int, chunk_overlap: int
    ) -> List[Chunk]:
        """
        Structure-aware paragraph splitter that respects heading boundaries (#, ##, ###)
        and paragraph clusters before falling back to character bounds.
        """
        all_chunks: List[Chunk] = []
        global_idx = 0

        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ". ", " ", ""],
        )

        for page in doc.pages:
            if not page.text.strip():
                continue

            # Split on double newline or markdown headers
            raw_paragraphs = re.split(r"(?:\n\s*\n)|(?=^#{1,4}\s+)", page.text, flags=re.MULTILINE)
            paragraphs = [p.strip() for p in raw_paragraphs if p and p.strip()]

            current_group: List[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)

                # If this individual paragraph is larger than chunk_size, split it using fallback
                if para_len > chunk_size:
                    # Flush current group first
                    if current_group:
                        merged = "\n\n".join(current_group)
                        chunk_id = f"{doc.doc_id}_c{global_idx + 1}"
                        all_chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                chunk_index=global_idx + 1,
                                text=merged,
                                doc_id=doc.doc_id,
                                filename=doc.filename,
                                page_number=page.page_number,
                                char_count=len(merged),
                                word_count=len(merged.split()),
                                strategy=ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                                metadata={
                                    "source": doc.filename,
                                    "doc_id": doc.doc_id,
                                    "page": page.page_number,
                                    "chunk_id": chunk_id,
                                    "chunk_index": global_idx + 1,
                                    "strategy": ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                                },
                            )
                        )
                        global_idx += 1
                        current_group = []
                        current_len = 0

                    sub_splits = fallback_splitter.split_text(para)
                    for sub in sub_splits:
                        sub = sub.strip()
                        if not sub:
                            continue
                        chunk_id = f"{doc.doc_id}_c{global_idx + 1}"
                        all_chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                chunk_index=global_idx + 1,
                                text=sub,
                                doc_id=doc.doc_id,
                                filename=doc.filename,
                                page_number=page.page_number,
                                char_count=len(sub),
                                word_count=len(sub.split()),
                                strategy=ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                                metadata={
                                    "source": doc.filename,
                                    "doc_id": doc.doc_id,
                                    "page": page.page_number,
                                    "chunk_id": chunk_id,
                                    "chunk_index": global_idx + 1,
                                    "strategy": ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                                },
                            )
                        )
                        global_idx += 1

                # If adding para exceeds chunk_size, flush group and start new group with overlap
                elif current_len + para_len > chunk_size and current_group:
                    merged = "\n\n".join(current_group)
                    chunk_id = f"{doc.doc_id}_c{global_idx + 1}"
                    all_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            chunk_index=global_idx + 1,
                            text=merged,
                            doc_id=doc.doc_id,
                            filename=doc.filename,
                            page_number=page.page_number,
                            char_count=len(merged),
                            word_count=len(merged.split()),
                            strategy=ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                            metadata={
                                "source": doc.filename,
                                "doc_id": doc.doc_id,
                                "page": page.page_number,
                                "chunk_id": chunk_id,
                                "chunk_index": global_idx + 1,
                                "strategy": ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                            },
                        )
                    )
                    global_idx += 1

                    # Keep last paragraph for overlap if within chunk_overlap limit
                    if current_group and len(current_group[-1]) <= chunk_overlap:
                        current_group = [current_group[-1], para]
                        current_len = len(current_group[0]) + para_len + 2
                    else:
                        current_group = [para]
                        current_len = para_len
                else:
                    current_group.append(para)
                    current_len += para_len + 2

            # Flush remaining group for the page
            if current_group:
                merged = "\n\n".join(current_group)
                chunk_id = f"{doc.doc_id}_c{global_idx + 1}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        chunk_index=global_idx + 1,
                        text=merged,
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        page_number=page.page_number,
                        char_count=len(merged),
                        word_count=len(merged.split()),
                        strategy=ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                        metadata={
                            "source": doc.filename,
                            "doc_id": doc.doc_id,
                            "page": page.page_number,
                            "chunk_id": chunk_id,
                            "chunk_index": global_idx + 1,
                            "strategy": ChunkingStrategy.PARAGRAPH_SEMANTIC.value,
                        },
                    )
                )
                global_idx += 1

        return all_chunks
