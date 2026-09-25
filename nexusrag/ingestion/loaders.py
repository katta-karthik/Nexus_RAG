import os
import io
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import pypdf


@dataclass
class DocumentPage:
    page_number: int
    text: str
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    total_characters: int
    pages: List[DocumentPage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class DocumentLoader:
    """
    Robust multi-format document loader supporting PDF, TXT, and Markdown.
    Accepts both disk file paths and in-memory byte streams.
    """

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @classmethod
    def load(cls, source: Union[str, bytes, io.BytesIO], filename: str) -> LoadedDocument:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )

        # Generate a deterministic doc_id from filename and content hash
        if isinstance(source, str):
            with open(source, "rb") as f:
                content_bytes = f.read()
        elif isinstance(source, io.BytesIO):
            content_bytes = source.getvalue()
        else:
            content_bytes = source

        content_hash = hashlib.sha256(content_bytes[:4096]).hexdigest()[:12]
        doc_id = f"doc_{content_hash}"

        if ext == ".pdf":
            return cls._load_pdf(content_bytes, filename, doc_id)
        elif ext in {".txt", ".md"}:
            return cls._load_text(content_bytes, filename, doc_id, ext)
        else:
            raise ValueError(f"Unhandled extension: {ext}")

    @classmethod
    def _load_pdf(cls, content_bytes: bytes, filename: str, doc_id: str) -> LoadedDocument:
        stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(stream)
        pages: List[DocumentPage] = []
        total_chars = 0

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            char_count = len(page_text)
            total_chars += char_count
            pages.append(
                DocumentPage(
                    page_number=i + 1,
                    text=page_text,
                    char_count=char_count,
                    metadata={"source": filename, "page": i + 1},
                )
            )

        # In case reader finds 0 pages or empty PDF
        if not pages:
            pages.append(
                DocumentPage(
                    page_number=1,
                    text="",
                    char_count=0,
                    metadata={"source": filename, "page": 1},
                )
            )

        return LoadedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            total_pages=len(pages),
            total_characters=total_chars,
            pages=pages,
            metadata={"source": filename, "type": "pdf", "size_bytes": len(content_bytes)},
        )

    @classmethod
    def _load_text(
        cls, content_bytes: bytes, filename: str, doc_id: str, ext: str
    ) -> LoadedDocument:
        # Try UTF-8 first, fallback to latin-1
        try:
            raw_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content_bytes.decode("latin-1", errors="replace")

        # For text files, if long, we can break into logical pages of ~3,000 characters
        # or keep as 1 page if under threshold
        page_size_chars = 3000
        if len(raw_text) <= page_size_chars:
            pages = [
                DocumentPage(
                    page_number=1,
                    text=raw_text,
                    char_count=len(raw_text),
                    metadata={"source": filename, "page": 1},
                )
            ]
        else:
            # Segment along paragraph breaks where possible
            paragraphs = raw_text.split("\n\n")
            pages = []
            curr_text = []
            curr_len = 0
            page_num = 1

            for p in paragraphs:
                p_len = len(p)
                if curr_len + p_len > page_size_chars and curr_text:
                    p_content = "\n\n".join(curr_text)
                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            text=p_content,
                            char_count=len(p_content),
                            metadata={"source": filename, "page": page_num},
                        )
                    )
                    page_num += 1
                    curr_text = [p]
                    curr_len = p_len
                else:
                    curr_text.append(p)
                    curr_len += p_len + 2

            if curr_text:
                p_content = "\n\n".join(curr_text)
                pages.append(
                    DocumentPage(
                        page_number=page_num,
                        text=p_content,
                        char_count=len(p_content),
                        metadata={"source": filename, "page": page_num},
                    )
                )

        total_chars = sum(p.char_count for p in pages)
        return LoadedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=ext.replace(".", ""),
            total_pages=len(pages),
            total_characters=total_chars,
            pages=pages,
            metadata={"source": filename, "type": ext.replace(".", ""), "size_bytes": len(content_bytes)},
        )
