"""
src/state.py — Enterprise AgentState
──────────────────────────────────────
Extends LangGraph MessagesState with all fields needed for:
  • Multi-turn conversation (session_id, thread_id)
  • Agentic RAG (question, context, retrieved_docs, rewritten_question)
  • Guardrails (guard_blocked, guard_reason, pii_detected)
  • Evaluation (retrieval_scores)
  • Observability (trace_id, latency_ms, token_usage)
  • Human-in-the-loop (requires_approval, approval_granted)
  • Streaming (streaming_tokens)
  • Reflection / self-correction (reflection_count, correction_notes)
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
import operator


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic sub-models (used inside the state for structured data)
# ─────────────────────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RetrievalScores(BaseModel):
    hit_at_k: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    f1_score_at_k: float = 0.0


class GuardResult(BaseModel):
    safe: bool = True
    reason: str = ""
    violation: str = "none"          # none | prompt_injection | pii | toxic | too_long
    hallucination_score: float = 0.0
    pii_entities: List[Dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main AgentState
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(MessagesState, total=False):
    """
    Enterprise-grade LangGraph state.

    Convention for list fields that nodes append to:
        Annotated[List[X], operator.add]
    This lets multiple nodes safely extend lists without overwriting them.
    """

    # ── Identity / session ───────────────────────────────────────────────────
    session_id: str                # user/browser session
    thread_id: str                 # LangGraph checkpointer thread
    user_id: str                   # authenticated user (RBAC)
    trace_id: str                  # LangSmith / OTEL trace

    # ── Core RAG fields ──────────────────────────────────────────────────────
    question: str
    rewritten_question: str
    context: str                   # final assembled context string for generation
    retrieved_docs: Annotated[List[Document], operator.add]  # all raw docs
    relevant_docs: Annotated[List[Document], operator.add]   # graded-relevant docs
    answer: str

    # ── Retrieval metadata ───────────────────────────────────────────────────
    retrieval_scores: Optional[RetrievalScores]
    retrieval_strategy: str       # dense | bm25 | hybrid | hyde

    # ── Reflection / self-correction ─────────────────────────────────────────
    reflection_count: int
    max_reflections: int
    correction_notes: Annotated[List[str], operator.add]

    # ── Guardrails ───────────────────────────────────────────────────────────
    input_guard: Optional[GuardResult]
    output_guard: Optional[GuardResult]
    guard_blocked: bool
    guard_reason: str
    pii_detected: bool
    pii_entities: Annotated[List[Dict[str, Any]], operator.add]

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    requires_approval: bool
    approval_granted: Optional[bool]
    approval_reason: str

    # ── Streaming ────────────────────────────────────────────────────────────
    streaming_tokens: Annotated[List[str], operator.add]
    is_streaming: bool

    # ── Observability ────────────────────────────────────────────────────────
    token_usage: Optional[TokenUsage]
    latency_ms: float
    llm_provider_used: str
    node_trace: Annotated[List[str], operator.add]  # execution path log

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_context: str             # injected from long-term memory store
    conversation_summary: str       # summarized history for long threads

    # ── MCP / Tool calls ─────────────────────────────────────────────────────
    tool_calls_made: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
