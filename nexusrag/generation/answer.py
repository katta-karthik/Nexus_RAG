import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator
from dotenv import load_dotenv
from .prompts import GroundedPromptBuilder

load_dotenv()


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
    Generates concise, natural language answers from retrieved context with live streaming.
    Supports Groq (LLaMA/Qwen/GPT-OSS), Google Gemini, OpenAI, and extractive fallback.
    """

    def __init__(
        self,
        provider: str = "groq",
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.temperature = temperature
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.model_name = model_name
        self.groq_client = None
        self._init_client()

    def _init_client(self):
        groq_key = os.getenv("GROQ_API_KEY") or (self.api_key if self.provider == "groq" else None)
        if (self.provider == "groq" or groq_key) and groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                self.provider = "groq"
                self.model_name = self.model_name or "qwen/qwen3.8-27b"
            except Exception:
                self.groq_client = None

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        citations = []
        for c in chunks:
            meta = c.get("metadata", {})
            citations.append(
                Citation(
                    source_document=meta.get("filename", "Uploaded Document"),
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
        if not chunks:
            yield "I couldn't find sufficient evidence in the uploaded documents to answer this confidently."
            return

        context_str = GroundedPromptBuilder.format_context_block(chunks)
        system_content = GroundedPromptBuilder.SYSTEM_PROMPT

        # 1. Groq ultra-fast streaming
        if self.groq_client is not None:
            try:
                messages = [{"role": "system", "content": system_content}]
                if chat_history:
                    for turn in chat_history[-4:]:
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        messages.append({"role": role, "content": content})

                user_content = (
                    f"<document_context>\n{context_str}\n</document_context>\n\n"
                    f"User Question: {query}\n\n"
                    "Answer the question directly and concisely in natural language:"
                )
                messages.append({"role": "user", "content": user_content})

                # Prioritize qwen3.8-27b for fast, direct, concise natural responses
                models_to_try = [self.model_name or "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
                for model in models_to_try:
                    try:
                        stream = self.groq_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=self.temperature,
                            stream=True,
                        )
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content
                        return
                    except Exception:
                        continue
            except Exception:
                pass

        # 2. Gemini fallback if key available
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=self.temperature,
                    google_api_key=gemini_key,
                    streaming=True,
                )
                messages = GroundedPromptBuilder.build_messages(query, chunks, chat_history)
                for chunk in llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        yield content
                return
            except Exception:
                pass

        # 3. Intelligent extractive fallback (concise extraction, not raw dump)
        yield from self._stream_concise_fallback(query, chunks)

    def _stream_concise_fallback(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Iterator[str]:
        top_chunk = chunks[0]
        text = top_chunk.get("text", "")
        q_tokens = [w for w in query.lower().split() if len(w) > 2]

        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        matched_sentences = []
        for s in sentences:
            if any(q in s.lower() for q in q_tokens):
                matched_sentences.append(s)

        if matched_sentences:
            ans = matched_sentences[0] + "."
        elif sentences:
            ans = sentences[0] + "."
        else:
            ans = "Information found in document."

        for word in ans.split(" "):
            yield word + " "
            time.sleep(0.02)
