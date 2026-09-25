import re
from dataclasses import dataclass
from typing import Tuple, List
from .loaders import LoadedDocument, DocumentPage


@dataclass
class CleaningStats:
    chars_before: int
    chars_after: int
    chars_removed: int
    reduction_pct: float
    pages_processed: int


class TextCleaner:
    """
    Cleans and normalizes extracted text across pages, removing control characters,
    excessive whitespace, and line-break artifacts while preserving semantic layout.
    """

    @classmethod
    def clean_text(cls, text: str) -> Tuple[str, int]:
        """
        Cleans a text string and returns (cleaned_text, characters_removed).
        """
        initial_len = len(text)

        # 1. Normalize line breaks: replace CRLF or CR with LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip non-printable and null control characters (keep \n and \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 3. Replace non-breaking spaces and tabs with standard spaces
        text = text.replace("\xa0", " ").replace("\t", "    ")

        # 4. Collapse lines with only whitespace into empty lines
        text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)

        # 5. Condense 3+ consecutive newlines to 2 newlines (preserve paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 6. Collapse multiple horizontal spaces to single space
        text = re.sub(r"[ ]{2,}", " ", text)

        cleaned = text.strip()
        chars_removed = max(0, initial_len - len(cleaned))
        return cleaned, chars_removed

    @classmethod
    def clean_document(cls, doc: LoadedDocument) -> Tuple[LoadedDocument, CleaningStats]:
        """
        Cleans all pages in a LoadedDocument and returns a new LoadedDocument along with CleaningStats.
        """
        cleaned_pages: List[DocumentPage] = []
        chars_before = 0
        chars_after = 0

        for page in doc.pages:
            chars_before += page.char_count
            cleaned_text, _ = cls.clean_text(page.text)
            cleaned_len = len(cleaned_text)
            chars_after += cleaned_len

            cleaned_pages.append(
                DocumentPage(
                    page_number=page.page_number,
                    text=cleaned_text,
                    char_count=cleaned_len,
                    metadata=dict(page.metadata),
                )
            )

        chars_removed = chars_before - chars_after
        reduction_pct = (chars_removed / chars_before * 100) if chars_before > 0 else 0.0

        stats = CleaningStats(
            chars_before=chars_before,
            chars_after=chars_after,
            chars_removed=chars_removed,
            reduction_pct=round(reduction_pct, 2),
            pages_processed=len(cleaned_pages),
        )

        cleaned_doc = LoadedDocument(
            doc_id=doc.doc_id,
            filename=doc.filename,
            file_type=doc.file_type,
            total_pages=len(cleaned_pages),
            total_characters=chars_after,
            pages=cleaned_pages,
            metadata={**doc.metadata, "cleaned": True, "reduction_pct": stats.reduction_pct},
        )

        return cleaned_doc, stats
