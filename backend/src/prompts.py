from src.config import Config
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    relevance: Literal["relevant", "not_relevant"] = Field(
        description="'relevant' if document helps answer question, 'not_relevant' otherwise."
    )
    reasoning: str = Field(description="Brief explanation.")

class PromptLoader:
    @staticmethod
    def system() -> str:
        return Config.get_prompt("GENERATE_QUERY_SYSTEM_PROMPT")

    @staticmethod
    def grading(question: str, context: str) -> str:
        return Config.get_prompt("GRADING_PROMPT").format(question=question, context=context)

    @staticmethod
    def rewrite(question: str) -> str:
        return Config.get_prompt("REWRITE_PROMPT").format(question=question)

    @staticmethod
    def generation(question: str, context: str) -> str:
        return Config.get_prompt("GENERATION_PROMPT").format(question=question, context=context)