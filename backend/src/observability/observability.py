
"""
src/observability.py — Enterprise Observability
─────────────────────────────────────────────────
Integrations:
  • LangSmith: run tracing, datasets, evals, prompt management
  • OpenTelemetry: distributed tracing (OTLP exporter)
  • Prometheus: token cost, latency, error rates, cache hits
  • Structured logging (structlog → JSON in production)

Usage:
    from src.observability import tracer, record_chat_metrics, get_logger

    logger = get_logger(__name__)
    with tracer.start_as_current_span("my-span") as span:
        span.set_attribute("query", query)
        ...
    record_chat_metrics(latency_ms=120, tokens=500, provider="groq", success=True)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import structlog

from src.config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Structured logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _configure_structlog() -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if Config.APP_ENV == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_structlog_configured = False


def get_logger(name: str) -> structlog.BoundLogger:
    global _structlog_configured
    if not _structlog_configured:
        _configure_structlog()
        _structlog_configured = True
    return structlog.get_logger(name)


# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry tracer
# ─────────────────────────────────────────────────────────────────────────────

def _build_otel_tracer():
    if not Config.OTEL_ENABLED:
        # Return a no-op tracer
        from opentelemetry import trace
        return trace.get_tracer("noop")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": Config.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=Config.OTEL_EXPORTER_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(Config.OTEL_SERVICE_NAME)
        logging.info(f"[Observability] OTEL tracer ready → {Config.OTEL_EXPORTER_ENDPOINT}")
        return tracer
    except Exception as e:
        logging.warning(f"[Observability] OTEL init failed: {e} — using no-op tracer")
        from opentelemetry import trace
        return trace.get_tracer("noop")


tracer = _build_otel_tracer()


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

_prom_metrics: Optional[Dict[str, Any]] = None


def _build_prometheus():
    if not Config.PROMETHEUS_ENABLED:
        return None
    try:
        from prometheus_client import Counter, Histogram, Gauge, Summary
        metrics = {
            "chat_requests_total": Counter(
                "rag_chat_requests_total",
                "Total chat requests",
                ["provider", "status"],
            ),
            "chat_latency_ms": Histogram(
                "rag_chat_latency_ms",
                "Chat request latency in milliseconds",
                ["provider"],
                buckets=[100, 250, 500, 1000, 2000, 5000, 10000],
            ),
            "tokens_used_total": Counter(
                "rag_tokens_used_total",
                "Total tokens consumed",
                ["provider", "type"],   # type: prompt | completion
            ),
            "cost_usd_total": Counter(
                "rag_cost_usd_total",
                "Estimated total cost in USD",
                ["provider"],
            ),
            "cache_hits_total": Counter(
                "rag_cache_hits_total",
                "Cache hits",
                ["cache_type"],   # exact | semantic
            ),
            "retrieval_docs_returned": Histogram(
                "rag_retrieval_docs_returned",
                "Number of documents returned by retrieval",
                buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
            ),
            "guard_blocks_total": Counter(
                "rag_guard_blocks_total",
                "Total requests blocked by guardrails",
                ["violation_type"],
            ),
            "upload_files_total": Counter(
                "rag_upload_files_total",
                "Total files uploaded",
                ["format", "status"],
            ),
            "active_sessions": Gauge(
                "rag_active_sessions",
                "Currently active chat sessions",
            ),
            "llm_errors_total": Counter(
                "rag_llm_errors_total",
                "LLM provider errors",
                ["provider"],
            ),
        }
        logging.info("[Observability] Prometheus metrics registered")
        return metrics
    except ImportError:
        logging.warning("[Observability] prometheus_client not available")
        return None


_prom_metrics = _build_prometheus()


def _prom(metric_name: str):
    """Safe Prometheus metric accessor — returns None if disabled."""
    if _prom_metrics is None:
        return None
    return _prom_metrics.get(metric_name)


# ─────────────────────────────────────────────────────────────────────────────
# Metric recording helpers (called from nodes / API)
# ─────────────────────────────────────────────────────────────────────────────

def record_chat_metrics(
    *,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    provider: str = "unknown",
    success: bool = True,
    cache_hit: bool = False,
    cache_type: str = "none",
    docs_returned: int = 0,
) -> None:
    """Record all metrics for a single chat request."""
    status = "success" if success else "error"

    m = _prom("chat_requests_total")
    if m:
        m.labels(provider=provider, status=status).inc()

    m = _prom("chat_latency_ms")
    if m:
        m.labels(provider=provider).observe(latency_ms)

    m = _prom("tokens_used_total")
    if m:
        if prompt_tokens:
            m.labels(provider=provider, type="prompt").inc(prompt_tokens)
        if completion_tokens:
            m.labels(provider=provider, type="completion").inc(completion_tokens)

    m = _prom("cost_usd_total")
    if m and estimated_cost_usd:
        m.labels(provider=provider).inc(estimated_cost_usd)

    if cache_hit:
        m = _prom("cache_hits_total")
        if m:
            m.labels(cache_type=cache_type).inc()

    m = _prom("retrieval_docs_returned")
    if m:
        m.observe(docs_returned)


def record_guard_block(violation_type: str) -> None:
    m = _prom("guard_blocks_total")
    if m:
        m.labels(violation_type=violation_type).inc()


def record_upload(file_format: str, success: bool) -> None:
    m = _prom("upload_files_total")
    if m:
        m.labels(format=file_format, status="success" if success else "error").inc()


def record_llm_error(provider: str) -> None:
    m = _prom("llm_errors_total")
    if m:
        m.labels(provider=provider).inc()


def inc_active_sessions(delta: int = 1) -> None:
    m = _prom("active_sessions")
    if m:
        m.inc(delta) if delta > 0 else m.dec(abs(delta))


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_langsmith_client():
    """Return a LangSmith client if configured, else None."""
    if not Config.LANGSMITH_ENABLED or not Config.LANGSMITH_API_KEY:
        return None
    try:
        from langsmith import Client
        return Client(
            api_url=Config.LANGSMITH_ENDPOINT,
            api_key=Config.LANGSMITH_API_KEY,
        )
    except ImportError:
        logging.warning("[Observability] langsmith not installed")
        return None
    except Exception as e:
        logging.warning(f"[Observability] LangSmith client error: {e}")
        return None


async def log_to_langsmith(
    run_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    error: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log a custom run to LangSmith.
    Returns run_id on success, None on failure.
    LangChain/LangGraph automatically traces when LANGCHAIN_TRACING_V2=true —
    this is for custom non-LangChain code paths.
    """
    client = get_langsmith_client()
    if client is None:
        return None
    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _create_run():
            run = client.create_run(
                name=run_name,
                run_type="chain",
                inputs=inputs,
                project_name=Config.LANGSMITH_PROJECT,
                tags=tags or [],
                extra={"metadata": metadata or {}},
            )
            run.end(outputs=outputs, error=error)
            client.update_run(run.id, outputs=outputs, error=error)
            return str(run.id)

        return await loop.run_in_executor(None, _create_run)
    except Exception as e:
        logging.warning(f"[Observability] LangSmith log error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Context manager for timing + tracing a block
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def observe(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Async context manager that:
    - Creates an OTEL span
    - Times the block
    - Logs structured entry/exit

    Usage:
        async with observe("retrieval", {"query": query}) as obs:
            results = await retrieve(query)
            obs["docs_count"] = len(results)
    """
    log = get_logger("observe")
    context: Dict[str, Any] = {}
    start = time.perf_counter()

    span_attrs = attributes or {}
    with tracer.start_as_current_span(span_name) as span:
        for k, v in span_attrs.items():
            span.set_attribute(k, str(v))
        log.debug("span_start", span=span_name, **span_attrs)
        try:
            yield context
            elapsed = (time.perf_counter() - start) * 1000
            context["latency_ms"] = elapsed
            span.set_attribute("latency_ms", elapsed)
            log.debug("span_end", span=span_name, latency_ms=f"{elapsed:.1f}", **context)
        except Exception as e:
            span.record_exception(e)
            span.set_status(
                __import__("opentelemetry.trace", fromlist=["StatusCode"]).StatusCode.ERROR,
                str(e),
            )
            log.error("span_error", span=span_name, error=str(e))
            raise
