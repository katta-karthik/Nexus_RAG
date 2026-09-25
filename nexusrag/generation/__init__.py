"""
Generation module for NexusRAG.
Handles grounded prompt templates, streaming LLM answer generation,
and citation parsing.
"""

from .prompts import GroundedPromptBuilder
from .answer import AnswerGenerator, GenerationResult, Citation

__all__ = [
    "GroundedPromptBuilder",
    "AnswerGenerator",
    "GenerationResult",
    "Citation",
]
