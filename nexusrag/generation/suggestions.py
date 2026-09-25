import os
import re
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class DocumentQuestionSuggester:
    """
    Analyzes an ingested document and dynamically generates 3-4 concise,
    hyper-relevant questions tailored specifically to that document's content.
    """

    SYSTEM_PROMPT = (
        "You are an expert AI document analyst.\n"
        "Your task: Read the provided document excerpt and generate 3 to 4 natural, concise questions "
        "that a user would realistically ask about THIS specific document.\n"
        "Rules:\n"
        "- Tailor questions directly to the document domain (e.g., if a resume: ask about CGPA/education, skills, experience, projects; "
        "if a research paper: ask about methodology, key findings; if a financial report: ask about revenue, margins, guidance).\n"
        "- Make each question clear, natural, and under 15 words.\n"
        "- Return ONLY the questions, exactly one per line.\n"
        "- Do NOT include numbers, bullet symbols, quotation marks, or introductory words."
    )

    @classmethod
    def generate_suggestions(
        cls,
        document_text: str,
        filename: str = "",
        api_key: Optional[str] = None,
    ) -> List[str]:
        if not document_text.strip():
            return cls._fallback_questions(filename, "")

        sample_text = document_text[:3000].strip()
        groq_key = api_key or os.getenv("GROQ_API_KEY")

        # 1. Try Groq with Qwen
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": cls.SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Document: {filename}\n\nContent Excerpt:\n{sample_text}\n\nGenerate 3-4 specific questions:",
                        },
                    ],
                    temperature=0.3,
                    max_tokens=180,
                )
                raw_output = response.choices[0].message.content.strip()
                lines = [
                    re.sub(r"^[0-9]+[\.\)\-]\s*", "", line).strip(' "•-*')
                    for line in raw_output.split("\n")
                    if line.strip()
                ]
                valid_questions = [q for q in lines if len(q) > 8 and "?" in q][:4]
                if valid_questions:
                    return valid_questions
            except Exception:
                pass

        # 2. Intelligent Heuristic Fallback
        return cls._fallback_questions(filename, sample_text)

    @classmethod
    def _fallback_questions(cls, filename: str, text: str) -> List[str]:
        lower = (filename + " " + text).lower()

        # Resume / CV detection
        if any(w in lower for w in ["resume", "curriculum vitae", "education", "cgpa", "gpa", "b.tech", "skills", "experience"]):
            return [
                "What is his education background and CGPA?",
                "What technical skills and frameworks does he specialize in?",
                "Can you summarize his professional experience and internships?",
                "What major projects has he built?",
            ]

        # Financial / Business report detection
        if any(w in lower for w in ["annual report", "financial", "revenue", "ebitda", "fiscal", "q1", "q2", "q3", "q4"]):
            return [
                "What was the total revenue and annual growth rate?",
                "What are the primary operational risks mentioned?",
                "What is the company's future outlook and strategic priorities?",
                "What were the segment margins and earnings?",
            ]

        # Research Paper / Technical Paper detection
        if any(w in lower for w in ["abstract", "attention", "transformer", "neural", "dataset", "benchmark", "accuracy"]):
            return [
                "What is the primary contribution or problem solved?",
                "What architecture or methodology does the paper propose?",
                "What were the key experimental results and benchmarks?",
                "How does this approach compare to prior methods?",
            ]

        # General Document fallback
        return [
            "Can you provide a comprehensive summary of this document?",
            "What are the main key points and takeaways?",
            "Who are the key entities or topics discussed?",
        ]
