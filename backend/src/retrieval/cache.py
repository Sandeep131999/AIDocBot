"""
src/cache.py — Enterprise Semantic & Prompt Cache
───────────────────────────────────────────────────
Two cache layers:
  1. Exact prompt cache   — hash(query) → cached answer (Redis, fast)
  2. Semantic cache       — embedding similarity → nearest cached answer
     (ChromaDB collection "cache", cosine threshold)

Features:
  • Async-first (Redis async client)
  • TTL-aware (configurable per cache type)
  • Cache invalidation by key pattern
  • Hit/miss metrics for observability
  • Graceful degradation when Redis is down or embeddings fail
"""
import hashlib
import json
import logging
import time
from typing import Any

import redis.asyncio as redis

from src.config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        self.redis_client: redis.Redis | None = None
        self.hits = 0
        self.misses = 0
        self._redis_retry_after = 0.0
        self._redis_available = False
        if Config.REDIS_ENABLED:
            try:
                self.redis_client = redis.from_url(
                    Config.REDIS_URL,
                    password=Config.REDIS_PASSWORD or None,
                    decode_responses=True,
                    socket_connect_timeout=1,
                )
            except Exception as e:
                self._mark_redis_failure(e)

        # Lazy initialization of embedding model - only load when needed
        # This avoids DLL load issues at import time (Windows App Control policies)
        self._embed_model = None
        self._embed_model_failed = False

    def _can_use_redis(self) -> bool:
        return (
            Config.REDIS_ENABLED
            and self.redis_client is not None
            and time.monotonic() >= self._redis_retry_after
        )

    def _mark_redis_failure(self, error: Exception) -> None:
        self._redis_available = False
        self._redis_retry_after = time.monotonic() + 30
        logger.warning("[Cache] Redis unavailable; retrying in 30 seconds: %s", error)

    def _mark_redis_success(self) -> None:
        self._redis_available = True
        self._redis_retry_after = 0.0

    def _get_embed_model(self):
        """Lazily initialize the embedding model with error handling."""
        if self._embed_model is not None:
            return self._embed_model
        if self._embed_model_failed:
            return None

        try:
            # Import here to avoid loading at module import time
            from sentence_transformers import SentenceTransformer
            logger.info("[Embeddings] Loading HF model: BAAI/bge-base-en-v1.5")
            self._embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
            logger.info("[Embeddings] Model loaded successfully")
        except Exception as e:
            # Handle Windows App Control policy blocking DLLs
            error_msg = str(e)
            if "DLL load failed" in error_msg or "Application Control policy" in error_msg:
                logger.warning(
                    "[Cache] Embedding model failed to load due to Windows App Control policy "
                    "blocking native DLLs (likely onnxruntime). Semantic cache disabled. "
                    "Exact cache (Redis) will still work. Error: %s", error_msg
                )
            else:
                logger.warning(f"[Cache] Embed model failed to load: {e}")
            self._embed_model_failed = True
            self._embed_model = None
        return self._embed_model

    async def get_embedding(self, text: str):
        model = self._get_embed_model()
        if not model:
            return None
        try:
            return model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.warning(f"[Cache] Embedding generation failed: {e}")
            return None

    async def ping(self):
        if not self._can_use_redis():
            return False
        try:
            await self.redis_client.ping()
            self._mark_redis_success()
            return True
        except redis.RedisError as exc:
            self._mark_redis_failure(exc)
            return False


cache = CacheManager()


def _cache_key(query: str) -> str:
    digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
    return f"cache:exact:{digest}"


async def get_cached_response(query: str) -> dict[str, Any] | None:
    """Return an exact-query cache entry, or None when it is unavailable."""
    if not cache._can_use_redis():
        cache.misses += 1
        return None

    try:
        value = await cache.redis_client.get(_cache_key(query))
        cache._mark_redis_success()
        if value is None:
            cache.misses += 1
            return None
        response = json.loads(value)
        if not isinstance(response, dict):
            cache.misses += 1
            logger.warning("[Cache] Ignoring cached response with invalid shape")
            return None
        cache.hits += 1
        response.setdefault("cache_type", "exact")
        return response
    except redis.RedisError as exc:
        cache.misses += 1
        cache._mark_redis_failure(exc)
        return None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        cache.misses += 1
        logger.warning("[Cache] Invalid cached response: %s", exc)
        return None


async def cache_response(query: str, response: dict[str, Any]) -> None:
    """Store a response under its normalized query hash with the configured TTL."""
    if not cache._can_use_redis():
        return

    try:
        await cache.redis_client.set(
            _cache_key(query),
            json.dumps(response, ensure_ascii=False),
            ex=Config.REDIS_CACHE_TTL,
        )
        cache._mark_redis_success()
    except redis.RedisError as exc:
        cache._mark_redis_failure(exc)
    except (TypeError, ValueError) as exc:
        logger.warning("[Cache] Response could not be serialized: %s", exc)


def get_cache_stats() -> dict[str, int | float | bool]:
    """Return process-local cache counters and Redis availability."""
    total = cache.hits + cache.misses
    return {
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": cache.hits / total if total else 0.0,
        "redis_available": cache._redis_available,
    }


async def invalidate_cache(pattern: str = "cache:*") -> int:
    """Delete cache keys matching a Redis glob pattern."""
    if not cache._can_use_redis():
        return 0

    deleted = 0
    try:
        async for key in cache.redis_client.scan_iter(match=pattern):
            deleted += await cache.redis_client.delete(key)
    except redis.RedisError as exc:
        cache._mark_redis_failure(exc)
    else:
        cache._mark_redis_success()
    return deleted