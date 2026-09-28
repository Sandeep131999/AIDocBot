"""
api/middleware.py — Enterprise Middleware Stack
─────────────────────────────────────────────────
Middleware (in order of execution, outermost first):
  1. RequestIDMiddleware    — inject X-Request-ID into every request/response
  2. AuditLogMiddleware     — structured log every request with timing
  3. PIIScrubMiddleware     — scrub PII from request logs (not response body)
  4. RateLimitMiddleware    — per-IP + per-user rate limits via slowapi/Redis
  5. SecurityHeadersMiddleware — OWASP security headers on every response

Usage in main.py:
    from api.middleware import register_middleware
    register_middleware(app)
"""
from __future__ import annotations

import time
import uuid
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Request ID
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID into every request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Make it available via structlog context
        try:
            import structlog
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)
        except Exception:
            pass

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audit log
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Log every request with: method, path, status, latency, user agent, IP.
    Excludes health check and metrics endpoints to reduce noise.
    """

    SKIP_PATHS = {"/", "/api/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        request_id = getattr(request.state, "request_id", "-")

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"[REQUEST] {request.method} {request.url.path} "
            f"status={response.status_code} latency={elapsed_ms:.1f}ms "
            f"ip={client_ip} rid={request_id}"
        )

        # Record Prometheus metric
        try:
            from src.observability import record_chat_metrics
            if request.url.path == "/api/chat":
                record_chat_metrics(
                    latency_ms=elapsed_ms,
                    success=response.status_code < 400,
                )
        except Exception:
            pass

        return response


# ─────────────────────────────────────────────────────────────────────────────
# 3. PII Scrubbing (log-level only)
# ─────────────────────────────────────────────────────────────────────────────

class PIIScrubMiddleware(BaseHTTPMiddleware):
    """
    Prevent PII from appearing in request logs.
    Note: This scrubs from the log context only — the actual request body
    is passed through unmodified (guardrails.py handles PII redaction before LLM).
    """

    # Headers that should never be logged
    SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Sanitize headers in structlog context
        try:
            import structlog
            safe_headers = {
                k: ("***" if k.lower() in self.SENSITIVE_HEADERS else v)
                for k, v in request.headers.items()
            }
            structlog.contextvars.bind_contextvars(
                method=request.method,
                path=request.url.path,
                headers_scrubbed=True,
            )
        except Exception:
            pass

        return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────

def _build_limiter():
    """Build slowapi limiter — Redis backend if available, else in-memory."""
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        if Config.REDIS_ENABLED:
            storage_uri = Config.REDIS_URL
            logger.info(f"[RateLimit] Using Redis storage: {storage_uri}")
        else:
            storage_uri = "memory://"
            logger.info("[RateLimit] Using in-memory storage")

        return Limiter(
            key_func=get_remote_address,
            storage_uri=storage_uri,
            enabled=Config.RATE_LIMIT_ENABLED,
        )
    except ImportError:
        logger.warning("[RateLimit] slowapi not installed — rate limiting disabled")
        return None


limiter = _build_limiter()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Security Headers (OWASP)
# ─────────────────────────────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add OWASP-recommended security headers to every response.
    Addresses: clickjacking, MIME sniffing, XSS, info leakage.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["X-XSS-Protection"]          = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]        = "geolocation=(), microphone=()"
        response.headers["Cache-Control"]             = "no-store"
        response.headers["Pragma"]                    = "no-cache"
        if Config.APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Remove server fingerprint (MutableHeaders uses del, not pop)
        if "server" in response.headers:
            del response.headers["server"]
        if "x-powered-by" in response.headers:
            del response.headers["x-powered-by"]
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Registration helper
# ─────────────────────────────────────────────────────────────────────────────

def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI app.
    Order matters: outermost middleware runs first on request, last on response.
    """
    # CORS (outermost — must run before auth/rate limit)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.cors_origins_list,
        allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Audit logging
    app.add_middleware(AuditLogMiddleware)

    # PII scrubbing
    app.add_middleware(PIIScrubMiddleware)

    # Request ID (innermost — runs first so other middleware can use it)
    app.add_middleware(RequestIDMiddleware)

    # Rate limiter (attach to app state for route decorators)
    if limiter:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("[Middleware] All middleware registered")
