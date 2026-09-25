import os
import re
import hashlib
from enum import Enum
from typing import List, Optional, Callable, Any
import numpy as np


class EmbeddingProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"


class LocalDeterministicEmbeddings:
    """
    Fast, deterministic dense vectorizer (384-dimensions) based on subword hashing
    and character n-grams with L2 normalization.
    Requires NO external API keys or heavy GPU downloads, making NexusRAG 100% functional
    even in offline or unauthenticated demo sessions.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_token(self, token: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        # Use multiple hash seeds for dense projection
        for seed in range(3):
            h = int(hashlib.md5(f"{seed}:{token}".encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vec[idx] += sign
        return vec

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def _embed_single(self, text: str) -> List[float]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
            return vec.tolist()

        vec = np.zeros(self.dim, dtype=np.float32)
        # Add word-level and 3-gram subwords
        for token in tokens:
            vec += self._hash_token(token)
            if len(token) >= 4:
                for i in range(len(token) - 2):
                    sub = token[i : i + 3]
                    vec += 0.5 * self._hash_token(sub)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
        return vec.tolist()


class EmbeddingManager:
    """
    Manages embedding model initialization, provider selection,
    and batched embedding execution with live progress reporting.
    """

    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.GEMINI,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self._embedder = self._initialize_embedder()

    def _initialize_embedder(self) -> Any:
        if self.provider == EmbeddingProvider.GEMINI:
            key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                # Fallback to local deterministic if key missing
                self.provider = EmbeddingProvider.LOCAL
                self.model_name = "local-deterministic-384"
                return LocalDeterministicEmbeddings(dim=384)
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                model = self.model_name or "models/text-embedding-004"
                self.model_name = model
                return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)
            except Exception:
                self.provider = EmbeddingProvider.LOCAL
                self.model_name = "local-deterministic-384"
                return LocalDeterministicEmbeddings(dim=384)

        elif self.provider == EmbeddingProvider.OPENAI:
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                self.provider = EmbeddingProvider.LOCAL
                self.model_name = "local-deterministic-384"
                return LocalDeterministicEmbeddings(dim=384)
            try:
                from langchain_openai import OpenAIEmbeddings

                model = self.model_name or "text-embedding-3-small"
                self.model_name = model
                return OpenAIEmbeddings(model=model, api_key=key)
            except Exception:
                self.provider = EmbeddingProvider.LOCAL
                self.model_name = "local-deterministic-384"
                return LocalDeterministicEmbeddings(dim=384)

        else:
            self.provider = EmbeddingProvider.LOCAL
            self.model_name = "local-deterministic-384"
            return LocalDeterministicEmbeddings(dim=384)

    @property
    def info(self) -> dict:
        return {
            "provider": self.provider.value if isinstance(self.provider, EmbeddingProvider) else str(self.provider),
            "model": self.model_name,
        }

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text)

    def embed_documents_with_progress(
        self,
        texts: List[str],
        batch_size: int = 16,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> List[List[float]]:
        """
        Embeds a list of texts in batches, invoking on_progress(completed_count, total_count, percentage)
        after each batch to drive visible UI progress.
        """
        total = len(texts)
        if total == 0:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_vectors = self._embedder.embed_documents(batch)
            all_embeddings.extend(batch_vectors)

            completed = min(i + len(batch), total)
            pct = (completed / total) * 100.0
            if on_progress:
                on_progress(completed, total, pct)

        return all_embeddings
