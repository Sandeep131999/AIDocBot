"""
src/cache.py — Enterprise Semantic & Prompt Cache
───────────────────────────────────────────────────
Two cache layers:
    Exact prompt cache — hash(query) → cached answer in a bounded in-process TTL cache

Features:
    • Async-compatible API, with process-local storage
  • TTL-aware (configurable per cache type)
  • Cache invalidation by key pattern
  • Hit/miss metrics for observability
"""
import fnmatch
import hashlib
import json
import logging
import threading
from typing import Any

from cachetools import TTLCache

from src.config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    """Small process-local exact-query response cache; shared by no external service."""

    def __init__(self):
        self.responses: TTLCache[str, str] = TTLCache(maxsize=500, ttl=Config.CACHE_TTL)
        self.hits = 0
        self.misses = 0
        self.project_hits: dict[str, int] = {}
        self.project_misses: dict[str, int] = {}
        self._lock = threading.RLock()


cache = CacheManager()


def _cache_key(query: str, project_id: str = "default") -> str:
    project_digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
    digest = hashlib.sha256(f"{project_id}\0{query.strip()}".encode("utf-8")).hexdigest()
    return f"cache:exact:{project_digest}:{digest}"


async def get_cached_response(
    query: str, project_id: str = "default"
) -> dict[str, Any] | None:
    """Return an exact-query cache entry, or None when it is unavailable."""
    with cache._lock:
        value = cache.responses.get(_cache_key(query, project_id))
        if value is None:
            cache.misses += 1
            cache.project_misses[project_id] = cache.project_misses.get(project_id, 0) + 1
            return None
        try:
            response = json.loads(value)
        except json.JSONDecodeError:
            cache.misses += 1
            cache.project_misses[project_id] = cache.project_misses.get(project_id, 0) + 1
            logger.warning("[Cache] Ignoring invalid cached response")
            return None
        if not isinstance(response, dict):
            cache.misses += 1
            cache.project_misses[project_id] = cache.project_misses.get(project_id, 0) + 1
            return None
        cache.hits += 1
        cache.project_hits[project_id] = cache.project_hits.get(project_id, 0) + 1
        response.setdefault("cache_type", "exact")
        return response


async def cache_response(
    query: str, response: dict[str, Any], project_id: str = "default"
) -> None:
    """Store an exact-query response in the bounded process-local cache."""
    try:
        serialized = json.dumps(response, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning("[Cache] Response could not be serialized: %s", exc)
        return
    with cache._lock:
        cache.responses[_cache_key(query, project_id)] = serialized


def get_cache_stats(project_id: str | None = None) -> dict[str, int | float | bool]:
    if project_id is None:
        hits, misses = cache.hits, cache.misses
    else:
        hits = cache.project_hits.get(project_id, 0)
        misses = cache.project_misses.get(project_id, 0)
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "cache_enabled": True,
    }


async def invalidate_cache(
    pattern: str = "cache:*", project_id: str | None = None
) -> int:
    """Delete process-local cache entries matching a glob pattern."""
    if project_id is not None:
        project_digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
        pattern = f"cache:exact:{project_digest}:*"
    with cache._lock:
        matching = [key for key in cache.responses if fnmatch.fnmatch(key, pattern)]
        for key in matching:
            del cache.responses[key]
    return len(matching)