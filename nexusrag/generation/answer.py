import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Generator
from .prompts import GroundedPromptBuilder


@dataclass
class Citation:
    source_document: str
    page_number: int
    chunk_id: str
    relevance_score: float
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_document": self.source_document,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "relevance_score": self.relevance_score,
            "excerpt": self.excerpt,
        }


@dataclass
class GenerationResult:
    answer: str
    citations: List[Citation]
    model_name: str
    generation_time_sec: float
    context_chunks_count: int


class AnswerGenerator:
    """
    Orchestrates grounded answer generation with live streaming and structured citations.
    Supports Google Gemini (via langchain-google-genai), OpenAI (via langchain-openai),
    and a local extractive synthesizer for offline demo execution.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.llm = self._init_llm()

    def _init_llm(self) -> Any:
        if self.provider == "gemini":
            key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                self.provider = "demo"
                self.model_name = "nexusrag-extractive-demo"
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                model = self.model_name or "gemini-2.5-flash"
                self.model_name = model
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=self.temperature,
                    google_api_key=key,
                    streaming=True,
                )
            except Exception:
                self.provider = "demo"
                self.model_name = "nexusrag-extractive-demo"
                return None

        elif self.provider == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                self.provider = "demo"
                self.model_name = "nexusrag-extractive-demo"
                return None
            try:
                from langchain_openai import ChatOpenAI
                model = self.model_name or "gpt-4o-mini"
                self.model_name = model
                return ChatOpenAI(
                    model=model,
                    temperature=self.temperature,
                    api_key=key,
                    streaming=True,
                )
            except Exception:
                self.provider = "demo"
                self.model_name = "nexusrag-extractive-demo"
                return None

        else:
            self.provider = "demo"
            self.model_name = "nexusrag-extractive-demo"
            return None

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """
        Creates structured citations from the selected context chunks.
        """
        citations = []
        for c in chunks:
            meta = c.get("metadata", {})
            citations.append(
                Citation(
                    source_document=meta.get("filename", "Unknown Document"),
                    page_number=int(meta.get("page", 1)),
                    chunk_id=c.get("chunk_id", ""),
                    relevance_score=float(c.get("score", 0.0)),
                    excerpt=c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
                )
            )
        return citations

    def stream_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """
        Streams response tokens iteratively for Streamlit st.write_stream.
        """
        if not chunks:
            yield "I couldn't find sufficient evidence in the uploaded documents to answer this confidently."
            return

        messages = GroundedPromptBuilder.build_messages(query, chunks, chat_history)

        if self.llm is not None:
            try:
                for chunk in self.llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        yield content
                return
            except Exception:
                # If streaming API error occurs, fallback to demo synthesizer
                pass

        # Local Extractive Demo Synthesizer
        # Provides an authentic grounded answer without API keys
        yield from self._stream_demo_answer(query, chunks)

    def _stream_demo_answer(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Iterator[str]:
        top_chunk = chunks[0]
        meta = top_chunk.get("metadata", {})
        source = meta.get("filename", "document")
        page = meta.get("page", 1)

        intro = (
            f"Based on **{source}** (Page {page}), here is the relevant factual finding:\n\n"
        )
        for word in intro.split(" "):
            yield word + " "
            time.sleep(0.01)

        # Highlight key sentences from top chunks
        summary_sentences = []
        for c in chunks[:3]:
            text = c.get("text", "")
            # Pick first 2 informative sentences
            sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 20]
            if sentences:
                c_meta = c.get("metadata", {})
                c_source = c_meta.get("filename", source)
                c_page = c_meta.get("page", page)
                summary_sentences.append(
                    f"• {sentences[0]}. [{c_source} — Page {c_page}]"
                )

        full_body = "\n\n".join(summary_sentences)
        for word in full_body.split(" "):
            yield word + " "
            time.sleep(0.015)
