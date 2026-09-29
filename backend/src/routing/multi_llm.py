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

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    Retrying,
    stop_after_attempt,
    wait_exponential,
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
# Retry + circuit-breaker wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_with_resilience(llm: Runnable, name: str) -> Runnable:
    """
    Wrap a single provider so that:
      • transient failures are retried in-process with tenacity
        (exponential backoff, Config.LLM_RETRY_*) before giving up on
        this provider, and
      • every real success/failure is reported to that provider's
        CircuitBreaker, so it actually opens after repeated failures
        instead of being tried forever.

    Previously neither of these were wired to anything — the CircuitBreaker
    methods and the tenacity import existed but were never called.
    """
    breaker = _get_breaker(name)
    def _invoke(input_, config=None, **kwargs):
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(Config.LLM_RETRY_ATTEMPTS),
                wait=wait_exponential(
                    min=Config.LLM_RETRY_MIN_WAIT,
                    max=Config.LLM_RETRY_MAX_WAIT,
                ),
                reraise=True,
            ):
                with attempt:
                    result = llm.invoke(input_, config=config, **kwargs)
        except Exception:
            breaker.record_failure()
            raise
        else:
            breaker.record_success()
            return result

    async def _ainvoke(input_, config=None, **kwargs):
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(Config.LLM_RETRY_ATTEMPTS),
                wait=wait_exponential(
                    min=Config.LLM_RETRY_MIN_WAIT,
                    max=Config.LLM_RETRY_MAX_WAIT,
                ),
                reraise=True,
            ):
                with attempt:
                    result = await llm.ainvoke(input_, config=config, **kwargs)
        except Exception:
            breaker.record_failure()
            raise
        else:
            breaker.record_success()
            return result

    return RunnableLambda(_invoke, afunc=_ainvoke, name=f"{name}_resilient")


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
            api_key=Config.GROQ_API_KEY,
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
            model_name=Config.ANTHROPIC_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
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
            max_completion_tokens=max_tokens,
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
    structured_output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[list] = None,
) -> Runnable:
    """
    Build a multi-provider fallback LLM chain.

    Args:
        temperature: Override Config.LLM_TEMPERATURE
        max_tokens:  Override Config.LLM_MAX_TOKENS
        provider_order: Override Config.llm_provider_list (for model routing)
        structured_output_schema: If given, apply `.with_structured_output()`
            to EACH provider individually before chaining fallbacks. Doing
            this after `.with_fallbacks()` (the old behaviour) doesn't work —
            `with_fallbacks()` returns a RunnableWithFallbacks, which has no
            `.with_structured_output()` method, so it raised AttributeError
            the first time a grader/guardrail LLM was actually invoked.
        tools: If given, apply `.bind_tools()` to EACH provider individually
            before chaining fallbacks. This must be done on base LLM classes
            (ChatOpenAI, ChatGoogleGenerativeAI, etc.) before they are wrapped
            with resilience/fallbacks, as RunnableWithFallbacks and RunnableLambda
            don't have a `bind_tools` method.

    Returns:
        Primary LLM (retry- and circuit-breaker-wrapped) with all other
        providers chained as fallbacks.
    """
    temp = temperature if temperature is not None else Config.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
    order = provider_order or Config.llm_provider_list

    llms: List[Runnable] = []
    names: List[str] = []
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
        if llm is None:
            continue
        if structured_output_schema is not None:
            llm = llm.with_structured_output(structured_output_schema)
        if tools is not None:
            llm = llm.bind_tools(tools)
        llms.append(_wrap_with_resilience(llm, name))
        names.append(name)

    if not llms:
        raise RuntimeError(
            "No LLMs configured or all circuits open. "
            "Set at least one API key or start ollama serve."
        )

    logger.info(f"[LLM] Fallback chain: {' → '.join(names)}")

    primary, *fallbacks = llms
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


# ─────────────────────────────────────────────────────────────────────────────
# Singletons — always use these in nodes / guardrails / evaluator
# ─────────────────────────────────────────────────────────────────────────────
_llm_cache: Dict[str, Runnable] = {}


def get_llm() -> Runnable:
    """Default LLM singleton — full fallback chain, standard settings."""
    if "default" not in _llm_cache:
        _llm_cache["default"] = build_llm()
    return _llm_cache["default"]


def get_ragas_llm() -> BaseChatModel:
    """Return a native chat model for RAGAS, which requires BaseLanguageModel."""
    for name in Config.llm_provider_list:
        builder = _PROVIDER_BUILDERS.get(name)
        if builder is None:
            continue
        llm = builder(0.0, Config.LLM_MAX_TOKENS)
        if llm is not None:
            return llm
    raise RuntimeError("No configured chat model is available for RAGAS evaluation")


def get_fast_llm(
    structured_output_schema: Optional[Type[BaseModel]] = None,
) -> Runnable:
    """Speed-optimised chain: Groq first, then Gemini Flash, then Ollama."""
    key = "fast" if structured_output_schema is None else (
        f"fast:{structured_output_schema.__module__}.{structured_output_schema.__qualname__}"
    )
    if key not in _llm_cache:
        _llm_cache[key] = build_llm(
            temperature=0.0,
            max_tokens=512,
            provider_order=["groq", "gemini", "ollama"],
            structured_output_schema=structured_output_schema,
        )
    return _llm_cache[key]


def get_quality_llm() -> Runnable:
    """Quality-optimised chain: Gemini first, then Anthropic, then OpenAI."""
    if "quality" not in _llm_cache:
        _llm_cache["quality"] = build_llm(
            temperature=0.2,
            max_tokens=2048,
            provider_order=["gemini", "anthropic", "openai", "ollama"],
        )
    return _llm_cache["quality"]


def get_grader_llm() -> Runnable:
    """
    Grader LLM — structured output for document relevance scoring.
    Uses fast, low-temperature model (Groq preferred).
    """
    if "grader" not in _llm_cache:
        from pydantic import Field

        class GradeResult(BaseModel):
            relevant: bool = Field(description="Is document relevant to the question")
            confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
            reason: str = Field(description="Brief explanation")

        # structured_output_schema is applied per-provider, before the
        # fallback chain is built — see build_llm() docstring for why
        # applying it afterward (the old code) crashed on first use.
        _llm_cache["grader"] = build_llm(
            temperature=0.0,
            max_tokens=256,
            provider_order=["groq", "gemini", "ollama"],
            structured_output_schema=GradeResult,
        )
    return _llm_cache["grader"]


def get_guardrail_llm(
    structured_output_schema: Optional[Type[BaseModel]] = None,
) -> Runnable:
    """Guardrail LLM — fast, structured output for security checks."""
    key = "guardrail" if structured_output_schema is None else (
        f"guardrail:{structured_output_schema.__module__}.{structured_output_schema.__qualname__}"
    )
    if key not in _llm_cache:
        _llm_cache[key] = build_llm(
            temperature=0.0,
            max_tokens=256,
            provider_order=["groq", "gemini", "ollama"],
            structured_output_schema=structured_output_schema,
        )
    return _llm_cache[key]


def get_llm_with_tools(tools: list) -> Runnable:
    """
    Get an LLM with tools bound — builds a fresh fallback chain with tools
    bound to each base provider before resilience wrapping.
    
    This is needed because bind_tools() only works on base LLM classes
    (ChatOpenAI, ChatGoogleGenerativeAI, etc.), not on RunnableWithFallbacks
    or RunnableLambda wrappers.
    """
    cache_key = f"with_tools_{id(tuple(tools))}"
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = build_llm(tools=tools)
    return _llm_cache[cache_key]


def invalidate_llm_cache() -> None:
    """Force rebuild of all LLM singletons (e.g. after config reload)."""
    _llm_cache.clear()
    logger.info("[LLM] Cache invalidated — singletons will rebuild on next call")