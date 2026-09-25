import os
import re
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()


class QueryRewriter:
    """
    Reformulates and expands user queries into keyword-dense, disambiguated search queries.
    Uses Groq (Qwen/LLaMA), Gemini/OpenAI, or an algorithmic keyword fallback.
    """

    SYSTEM_PROMPT = (
        "You are an expert search query optimization engine for an enterprise RAG system.\n"
        "Rewrite the user's conversational question into a concise, keyword-dense search query "
        "that maximizes retrieval recall.\n"
        "Rules:\n"
        "- Strip conversational filler\n"
        "- Include key nouns, entities, metrics, and relationships\n"
        "- Return ONLY the rewritten query on a single line."
    )

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.groq_client = None
        self._init_groq()

    def _init_groq(self):
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
            except Exception:
                self.groq_client = None

    def rewrite(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""

        # 1. Use Groq if available
        if self.groq_client is not None:
            try:
                resp = self.groq_client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                rewritten = resp.choices[0].message.content.strip().replace('"', '').replace('\n', ' ')
                if rewritten and len(rewritten) < 150:
                    return rewritten
            except Exception:
                pass

        # 2. Use passed LLM if provided
        if self.llm is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
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

        text = re.sub(r"[?!.,;]+$", "", text).strip()
        tokens = [w for w in text.split() if len(w) > 2]

        if not tokens:
            return query

        return " ".join(tokens)
