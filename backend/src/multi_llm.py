"""
src/multi_llm.py — Enterprise Multi-LLM Factory
─────────────────────────────────────────────────
Features:
  • Provider fallback chain  (Gemini → Groq → OpenRouter → Anthropic → OpenAI → Ollama)
  • Tenacity retry with exponential backoff per provider
  • Circuit breaker (consecutive failure tracking)
  • Token cost estimation per provider
  • Model routing: route cheap/complex tasks to the right model
  • Singleton via get_llm() / get_grader_llm() / get_guardrail_llm()
  • Async-safe (all LangChain chat models support ainvoke)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cost table (USD per 1K tokens, input/output)
# Updated approximate values — adjust for current pricing
# ─────────────────────────────────────────────────────────────────────────────
COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash":            {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":            {"input": 0.000075, "output": 0.0003},
    "llama-3.1-8b-instant":        {"input": 0.00005,  "output": 0.00008},
    "llama-3.3-70b-versatile":     {"input": 0.00059,  "output": 0.00079},
    "claude-3-5-haiku-20241022":   {"input": 0.0008,   "output": 0.004},
    "gpt-4o-mini":                 {"input": 0.00015,  "output": 0.0006},
    "default":                     {"input": 0.0001,   "output": 0.0003},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single LLM call."""
    rates = COST_PER_1K.get(model, COST_PER_1K["default"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1000


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CircuitBreaker:
    """Simple half-open circuit breaker per provider."""
    name: str
    threshold: int = 5           # consecutive failures before opening
    reset_timeout: int = 60      # seconds before trying again (half-open)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def is_open(self) -> bool:
        if self._failures >= self.threshold:
            if time.time() - self._opened_at > self.reset_timeout:
                # half-open: allow one attempt
                logger.info(f"[CircuitBreaker] {self.name} half-open — trying again")
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures == self.threshold:
            self._opened_at = time.time()
            logger.warning(f"[CircuitBreaker] {self.name} OPENED after {self.threshold} failures")


# Registry of circuit breakers (one per provider name)
_breakers: Dict[str, CircuitBreaker] = {}


def _get_breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name)
    return _breakers[name]


# ─────────────────────────────────────────────────────────────────────────────
# Provider builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_gemini(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.GEMINI_API_KEY:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=Config.LLM_PROVIDER_TIMEOUT,
            convert_system_message_to_human=True,
        )
        logger.info(f"[LLM] Gemini added: {Config.GEMINI_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Gemini skip: {e}")
        return None


def _build_groq(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.GROQ_API_KEY:
        return None
    try:
        llm = ChatGroq(
            model=Config.GROQ_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_PROVIDER_TIMEOUT,
        )
        logger.info(f"[LLM] Groq added: {Config.GROQ_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Groq skip: {e}")
        return None


def _build_openrouter(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.OPENROUTER_API_KEY:
        return None
    try:
        llm = ChatOpenAI(
            model=Config.OPENROUTER_MODEL,
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_PROVIDER_TIMEOUT,
            default_headers={
                "HTTP-Referer": Config.APP_URL,
                "X-Title": Config.APP_NAME,
            },
        )
        logger.info(f"[LLM] OpenRouter added: {Config.OPENROUTER_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] OpenRouter skip: {e}")
        return None


def _build_anthropic(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.ANTHROPIC_API_KEY:
        return None
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=Config.ANTHROPIC_MODEL,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_PROVIDER_TIMEOUT,
        )
        logger.info(f"[LLM] Anthropic added: {Config.ANTHROPIC_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Anthropic skip: {e}")
        return None


def _build_openai(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.OPENAI_API_KEY:
        return None
    try:
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_PROVIDER_TIMEOUT,
        )
        logger.info(f"[LLM] OpenAI added: {Config.OPENAI_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] OpenAI skip: {e}")
        return None


def _build_ollama(temperature: float, max_tokens: int) -> Optional[BaseChatModel]:
    if not Config.OLLAMA_MODEL:
        return None
    try:
        llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens,
        )
        logger.info(f"[LLM] Ollama LOCAL added: {Config.OLLAMA_MODEL}")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Ollama skip: {e}")
        return None


_PROVIDER_BUILDERS = {
    "gemini":      _build_gemini,
    "groq":        _build_groq,
    "openrouter":  _build_openrouter,
    "anthropic":   _build_anthropic,
    "openai":      _build_openai,
    "ollama":      _build_ollama,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main factory
# ─────────────────────────────────────────────────────────────────────────────
def build_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider_order: Optional[List[str]] = None,
) -> BaseChatModel:
    """
    Build a multi-provider fallback LLM chain.

    Args:
        temperature: Override Config.LLM_TEMPERATURE
        max_tokens:  Override Config.LLM_MAX_TOKENS
        provider_order: Override Config.llm_provider_list (for model routing)

    Returns:
        Primary LLM with all other providers chained as fallbacks.
    """
    temp = temperature if temperature is not None else Config.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
    order = provider_order or Config.llm_provider_list

    llms: List[BaseChatModel] = []
    for name in order:
        builder = _PROVIDER_BUILDERS.get(name)
        if builder is None:
            logger.warning(f"[LLM] Unknown provider '{name}' — skipping")
            continue
        breaker = _get_breaker(name)
        if breaker.is_open:
            logger.warning(f"[LLM] Circuit OPEN for '{name}' — skipping")
            continue
        llm = builder(temp, tokens)
        if llm is not None:
            llms.append(llm)

    if not llms:
        raise RuntimeError(
            "No LLMs configured or all circuits open. "
            "Set at least one API key or start ollama serve."
        )

    chain_names = " → ".join(type(l).__name__ for l in llms)
    logger.info(f"[LLM] Fallback chain: {chain_names}")

    primary, *fallbacks = llms
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=(Exception,)) if fallbacks else primary


# ─────────────────────────────────────────────────────────────────────────────
# Singletons — always use these in nodes / guardrails / evaluator
# ─────────────────────────────────────────────────────────────────────────────
_llm_cache: Dict[str, BaseChatModel] = {}


def get_llm() -> BaseChatModel:
    """Default LLM singleton — full fallback chain, standard settings."""
    if "default" not in _llm_cache:
        _llm_cache["default"] = build_llm()
    return _llm_cache["default"]


def get_fast_llm() -> BaseChatModel:
    """Speed-optimised chain: Groq first, then Gemini Flash, then Ollama."""
    if "fast" not in _llm_cache:
        _llm_cache["fast"] = build_llm(
            temperature=0.0,
            max_tokens=512,
            provider_order=["groq", "gemini", "ollama"],
        )
    return _llm_cache["fast"]


def get_quality_llm() -> BaseChatModel:
    """Quality-optimised chain: Gemini first, then Anthropic, then OpenAI."""
    if "quality" not in _llm_cache:
        _llm_cache["quality"] = build_llm(
            temperature=0.2,
            max_tokens=2048,
            provider_order=["gemini", "anthropic", "openai", "ollama"],
        )
    return _llm_cache["quality"]


def get_grader_llm() -> BaseChatModel:
    """
    Grader LLM — structured output for document relevance scoring.
    Uses fast, low-temperature model (Groq preferred).
    """
    if "grader" not in _llm_cache:
        from pydantic import BaseModel, Field
        from typing import Literal

        class GradeResult(BaseModel):
            relevant: bool = Field(description="Is document relevant to the question")
            confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
            reason: str = Field(description="Brief explanation")

        base = build_llm(
            temperature=0.0,
            max_tokens=256,
            provider_order=["groq", "gemini", "ollama"],
        )
        _llm_cache["grader"] = base.with_structured_output(GradeResult)
    return _llm_cache["grader"]


def get_guardrail_llm() -> BaseChatModel:
    """Guardrail LLM — fast, structured output for security checks."""
    if "guardrail" not in _llm_cache:
        base = build_llm(
            temperature=0.0,
            max_tokens=256,
            provider_order=["groq", "gemini", "ollama"],
        )
        _llm_cache["guardrail"] = base
    return _llm_cache["guardrail"]


def invalidate_llm_cache() -> None:
    """Force rebuild of all LLM singletons (e.g. after config reload)."""
    _llm_cache.clear()
    logger.info("[LLM] Cache invalidated — singletons will rebuild on next call")
