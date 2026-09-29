"""
Enterprise Agentic RAG System
─────────────────────────────
LangGraph + FastAPI + Hybrid Search + Guardrails + MCP + Observability

All imports here are lazy to avoid loading heavy optional deps
(torch, structlog, prometheus, opentelemetry) at package import time.
Call the getter functions to pull in specific modules on demand.
"""

# Only truly lightweight, always-available imports at package level
from src.config import Config, get_settings
from src.agent.state import AgentState, GuardResult, TokenUsage, RetrievalScores

# ── Lazy getters — call these when you need the actual functions ─────────────

def get_document_loader():
    from src.retrieval.document_loader import load_and_split, load_and_split_async, get_supported_extensions
    return load_and_split, load_and_split_async, get_supported_extensions


def get_vector_store_module():
    from src.retrieval.vector_store import get_vector_store, get_retriever, retrieve, retrieve_async
    return get_vector_store, get_retriever, retrieve, retrieve_async


def get_tools_module():
    from src.routing.tools import get_retriever_tool, get_all_tools
    return get_retriever_tool, get_all_tools


def get_graph_module():
    from src.agent.graph import build_agentic_rag_graph, build_supervisor_graph
    return build_agentic_rag_graph, build_supervisor_graph


def get_guardrails_module():
    from src.safety.guardrails import check_input_guard, check_output_guard, redact_pii
    return check_input_guard, check_output_guard, redact_pii


def get_llm_module():
    from src.routing.multi_llm import (
        build_llm, get_llm, get_fast_llm, get_quality_llm,
        get_grader_llm, get_guardrail_llm, estimate_cost,
    )
    return build_llm, get_llm, get_fast_llm, get_quality_llm, get_grader_llm, get_guardrail_llm, estimate_cost


def get_memory_module():
    from src.observability.memory import (
        trim_messages, summarize_conversation,
        save_user_memory, get_user_memory, build_memory_context,
    )
    return trim_messages, summarize_conversation, save_user_memory, get_user_memory, build_memory_context


def get_cache_module():
    from src.retrieval.cache import get_cached_response, cache_response, get_cache_stats, invalidate_cache
    return get_cached_response, cache_response, get_cache_stats, invalidate_cache


def get_evaluator_module():
    from src.observability.evaluator import Evaluator, run_ragas_evaluation, golden_dataset_eval
    return Evaluator, run_ragas_evaluation, golden_dataset_eval


def get_observability_module():
    from src.observability.observability import get_logger, record_chat_metrics, observe
    return get_logger, record_chat_metrics, observe


__all__ = [
    "Config", "get_settings",
    "AgentState", "GuardResult", "TokenUsage", "RetrievalScores",
    "get_document_loader", "get_vector_store_module", "get_tools_module",
    "get_graph_module", "get_guardrails_module", "get_llm_module",
    "get_memory_module", "get_cache_module", "get_evaluator_module",
    "get_observability_module",
]
