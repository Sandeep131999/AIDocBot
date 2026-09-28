"""
src/prompts.py — Enterprise Prompt Management
───────────────────────────────────────────────
Centralized prompt access layer with versioning support.

Design decisions:
  • All prompts live in .env (Config) for runtime override without redeploy
  • PromptLoader provides typed static methods for each prompt
  • GradeDocuments Pydantic model used for structured grading output
  • Prompt versioning: add PROMPT_VERSION env key for A/B testing
  • Prompts are lazy-loaded (no import-time cost)
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schemas (used with .with_structured_output())
# ─────────────────────────────────────────────────────────────────────────────

class GradeDocuments(BaseModel):
    """Relevance grade for a retrieved document."""
    relevance: Literal["relevant", "not_relevant"] = Field(
        description="'relevant' if the document helps answer the question, 'not_relevant' otherwise."
    )
    confidence: float = Field(
        default=0.8,
        description="Confidence in the grade, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the relevance decision.",
    )


class RewriteResult(BaseModel):
    """Output of the query rewriting node."""
    rewritten_question: str = Field(description="Improved version of the original question")
    reasoning: str = Field(default="", description="Why this rewrite is better")


class HyDEDocument(BaseModel):
    """Output of HyDE hypothetical document generation."""
    hypothetical_document: str = Field(
        description="A hypothetical document that would perfectly answer the question"
    )


class ReflectionResult(BaseModel):
    """Output of the reflection/self-correction node."""
    needs_correction: bool = Field(description="Does the answer need improvement?")
    issues: list[str] = Field(default_factory=list, description="Identified problems with the answer")
    improved_answer: str = Field(default="", description="Corrected answer, empty if no correction needed")
    confidence: float = Field(default=1.0, description="Confidence in the final answer", ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loader
# ─────────────────────────────────────────────────────────────────────────────

class PromptLoader:
    """
    Static factory methods for every prompt in the system.
    All prompts sourced from Config (which reads from .env).
    Supports runtime override without code changes.
    """

    @staticmethod
    def system() -> str:
        """Main system prompt for generate_query_or_respond node."""
        from src.config import Config
        return Config.get_prompt("GENERATE_QUERY_SYSTEM_PROMPT")

    @staticmethod
    def grading(question: str, context: str) -> str:
        """Relevance grading prompt."""
        from src.config import Config
        return Config.get_prompt("GRADING_PROMPT").format(
            question=question, context=context
        )

    @staticmethod
    def rewrite(question: str) -> str:
        """Query rewriting prompt."""
        from src.config import Config
        return Config.get_prompt("REWRITE_PROMPT").format(question=question)

    @staticmethod
    def generation(question: str, context: str) -> str:
        """Final answer generation prompt."""
        from src.config import Config
        return Config.get_prompt("GENERATION_PROMPT").format(
            question=question, context=context
        )

    @staticmethod
    def reflection(question: str, context: str, answer: str) -> str:
        """Self-correction reflection prompt."""
        from src.config import Config
        return Config.get_prompt("REFLECTION_PROMPT").format(
            question=question, context=context, answer=answer
        )

    @staticmethod
    def hyde(question: str) -> str:
        """Hypothetical Document Embeddings generation prompt."""
        from src.config import Config
        return Config.get_prompt("HYDE_PROMPT").format(question=question)

    @staticmethod
    def summarize_history(history: str) -> str:
        """Conversation history summarization prompt."""
        from src.config import Config
        return Config.get_prompt("SUMMARIZE_HISTORY_PROMPT").format(history=history)

    @staticmethod
    def guardrail_input(query: str) -> str:
        """Input safety check prompt."""
        from src.config import Config
        return Config.get_prompt("GUARDRAIL_PROMPT").format(query=query)

    @classmethod
    def all_prompts(cls) -> dict[str, str]:
        """Return all prompts as a dict (for prompt management UI / LangSmith)."""
        return {
            "system":            cls.system(),
            "grading":           cls.grading("{question}", "{context}"),
            "rewrite":           cls.rewrite("{question}"),
            "generation":        cls.generation("{question}", "{context}"),
            "reflection":        cls.reflection("{question}", "{context}", "{answer}"),
            "hyde":              cls.hyde("{question}"),
            "summarize_history": cls.summarize_history("{history}"),
            "guardrail_input":   cls.guardrail_input("{query}"),
        }
