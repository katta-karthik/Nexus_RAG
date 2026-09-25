import os
import re
from typing import Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage


class QueryRewriter:
    """
    Reformulates and expands user queries into keyword-dense, disambiguated search queries.
    Supports LLM-powered rewriting (Gemini/OpenAI) with an automated algorithmic fallback
    for zero-API-key environments.
    """

    SYSTEM_PROMPT = (
        "You are an expert search query optimization engine for an enterprise RAG system.\n"
        "Your goal is to rewrite the user's conversational question into a concise, keyword-dense "
        "search query that maximizes vector and lexical recall.\n"
        "Rules:\n"
        "- Strip conversational filler ('tell me about', 'can you explain', 'what did they say about')\n"
        "- Expand core technical terms and acronyms\n"
        "- Include key nouns, entities, metrics, and relationships\n"
        "- Return ONLY the rewritten query on a single line with no markdown or explanation."
    )

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """
        Rewrites a query into a dense retrieval query.
        """
        query = query.strip()
        if not query:
            return ""

        if self.llm is not None:
            try:
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=f"Original Query: {query}\nRewritten Query:"),
                ]
                response = self.llm.invoke(messages)
                rewritten = response.content if hasattr(response, "content") else str(response)
                rewritten = rewritten.strip().replace('"', '').replace('\n', ' ')
                if rewritten:
                    return rewritten
            except Exception:
                pass

        # Algorithmic fallback: clean conversational stop words and focus on core semantic entities
        return self._rule_based_rewrite(query)

    def _rule_based_rewrite(self, query: str) -> str:
        fillers = [
            r"^what\s+(is|are|was|were|did)\s+(the\s+|they\s+)?",
            r"^can\s+you\s+(explain|tell\s+me\s+about)\s+",
            r"^tell\s+me\s+about\s+",
            r"^how\s+does\s+",
            r"^show\s+me\s+",
            r"\bplease\b",
            r"\babout\b",
            r"\bdetails\b",
        ]
        text = query.lower()
        for f in fillers:
            text = re.sub(f, "", text).strip()

        # Remove trailing question marks and punctuation
        text = re.sub(r"[?!.,;]+$", "", text).strip()
        tokens = [w for w in text.split() if len(w) > 2]

        if not tokens:
            return query

        return " ".join(tokens)
