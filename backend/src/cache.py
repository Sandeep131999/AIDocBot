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
  • Graceful degradation when Redis is down
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics counters (picked up by observability.py Prometheus exporter)
# ─────────────────────────────────────────────────────────────────────────────
_stats = {
    "exact_hits": 0,
    "exact_misses": 0,
    "semantic_hits": 0,
    "semantic_misses": 0,
}


def get_cache_stats() -> Dict[str, int]:
    return dict(_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Redis client
# ─────────────────────────────────────────────────────────────────────────────

_redis = None


async def _get_redis():
    global _redis
    if _redis is None and Config.REDIS_ENABLED:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(
                Config.REDIS_URL,
                password=Config.REDIS_PASSWORD or None,
                encoding="utf-8",
                decode_responses=True,
                max_connections=Config.REDIS_POOL_SIZE,
            )
            await _redis.ping()
            logger.info("[Cache] Redis connected")
        except Exception as e:
            logger.warning(f"[Cache] Redis unavailable: {e}")
            _redis = None
    return _redis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _query_hash(query: str, namespace: str = "chat") -> str:
    """Stable SHA-256 cache key for an exact query."""
    normalized = query.strip().lower()
    raw = f"{namespace}:{normalized}"
    return "cache:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _serialize(data: Dict[str, Any]) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


def _deserialize(raw: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Exact prompt cache
# ─────────────────────────────────────────────────────────────────────────────

async def get_exact_cache(query: str, namespace: str = "chat") -> Optional[Dict[str, Any]]:
    """
    Check Redis for an exact-match cached response.
    Returns the cached payload dict or None on miss.
    """
    redis = await _get_redis()
    if redis is None:
        _stats["exact_misses"] += 1
        return None
    try:
        key = _query_hash(query, namespace)
        raw = await redis.get(key)
        if raw:
            data = _deserialize(raw)
            if data:
                _stats["exact_hits"] += 1
                logger.debug(f"[Cache] Exact HIT: {key[:20]}...")
                data["cache_hit"] = True
                data["cache_type"] = "exact"
                return data
    except Exception as e:
        logger.warning(f"[Cache] get_exact_cache error: {e}")
    _stats["exact_misses"] += 1
    return None


async def set_exact_cache(
    query: str,
    payload: Dict[str, Any],
    ttl: Optional[int] = None,
    namespace: str = "chat",
) -> bool:
    """Store a response in the exact cache."""
    redis = await _get_redis()
    if redis is None:
        return False
    try:
        key = _query_hash(query, namespace)
        ttl = ttl or Config.REDIS_CACHE_TTL
        payload_clean = {k: v for k, v in payload.items() if k not in ("cache_hit", "cache_type")}
        payload_clean["cached_at"] = time.time()
        await redis.setex(key, ttl, _serialize(payload_clean))
        logger.debug(f"[Cache] Exact SET: {key[:20]}... (TTL={ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"[Cache] set_exact_cache error: {e}")
        return False


async def invalidate_cache(pattern: str = "cache:*", namespace: str = "") -> int:
    """Delete all keys matching pattern. Returns count deleted."""
    redis = await _get_redis()
    if redis is None:
        return 0
    try:
        if namespace:
            full_pattern = f"cache:{namespace}:*"
        else:
            full_pattern = pattern
        keys = await redis.keys(full_pattern)
        if keys:
            deleted = await redis.delete(*keys)
            logger.info(f"[Cache] Invalidated {deleted} keys matching '{full_pattern}'")
            return deleted
        return 0
    except Exception as e:
        logger.warning(f"[Cache] invalidate_cache error: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Semantic cache (ChromaDB collection)
# ─────────────────────────────────────────────────────────────────────────────

_semantic_collection = None


def _get_semantic_collection():
    """Lazy-initialize the Chroma semantic cache collection."""
    global _semantic_collection
    if _semantic_collection is None:
        try:
            import chromadb
            from src.embeddings import get_embeddings
            client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
            embedding_fn = get_embeddings()

            # Wrap LangChain embeddings for Chroma's EmbeddingFunction interface
            class _LCEmbeddingFn:
                name = "lc-embedding"   # required by chromadb >= 0.5

                def __call__(self, input: List[str]) -> List[List[float]]:
                    return embedding_fn.embed_documents(input)

            _semantic_collection = client.get_or_create_collection(
                name="semantic_cache",
                metadata={"hnsw:space": "cosine"},
                embedding_function=_LCEmbeddingFn(),
            )
            logger.info("[Cache] Semantic cache collection ready")
        except Exception as e:
            logger.warning(f"[Cache] Semantic collection init failed: {e}")
    return _semantic_collection


SEMANTIC_SIMILARITY_THRESHOLD = 0.92   # cosine similarity ≥ this → cache hit


async def get_semantic_cache(query: str) -> Optional[Dict[str, Any]]:
    """
    Search the semantic cache for a similar past query.
    Returns cached payload if cosine similarity ≥ SEMANTIC_SIMILARITY_THRESHOLD.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    def _search():
        col = _get_semantic_collection()
        if col is None or col.count() == 0:
            return None
        try:
            results = col.query(
                query_texts=[query],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            distances = results.get("distances", [[]])[0]
            if not distances:
                return None
            # Chroma cosine returns distance (0=identical, 2=opposite)
            # Convert: similarity = 1 - distance/2
            similarity = 1 - (distances[0] / 2)
            if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                meta = results.get("metadatas", [[{}]])[0][0]
                payload_raw = meta.get("payload", "{}")
                data = _deserialize(payload_raw)
                if data:
                    data["cache_hit"] = True
                    data["cache_type"] = "semantic"
                    data["similarity"] = round(similarity, 4)
                    return data
            return None
        except Exception as e:
            logger.warning(f"[Cache] Semantic query error: {e}")
            return None

    result = await loop.run_in_executor(None, _search)
    if result:
        _stats["semantic_hits"] += 1
        logger.debug(f"[Cache] Semantic HIT (sim={result.get('similarity', '?')})")
    else:
        _stats["semantic_misses"] += 1
    return result


async def set_semantic_cache(query: str, payload: Dict[str, Any]) -> bool:
    """Store a query+response pair in the semantic cache."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _store():
        col = _get_semantic_collection()
        if col is None:
            return False
        try:
            doc_id = _query_hash(query, "semantic")
            payload_clean = {k: v for k, v in payload.items() if k not in ("cache_hit", "cache_type", "similarity")}
            col.upsert(
                ids=[doc_id],
                documents=[query],
                metadatas=[{"payload": _serialize(payload_clean), "cached_at": str(time.time())}],
            )
            logger.debug(f"[Cache] Semantic SET: '{query[:50]}'")
            return True
        except Exception as e:
            logger.warning(f"[Cache] set_semantic_cache error: {e}")
            return False

    return await loop.run_in_executor(None, _store)


# ─────────────────────────────────────────────────────────────────────────────
# Unified cache lookup (exact first, then semantic)
# ─────────────────────────────────────────────────────────────────────────────

async def get_cached_response(query: str) -> Optional[Dict[str, Any]]:
    """
    Try exact cache first, then semantic cache.
    Returns the cached response dict or None.
    """
    # Exact match (fastest)
    exact = await get_exact_cache(query)
    if exact:
        return exact

    # Semantic match
    semantic = await get_semantic_cache(query)
    return semantic


async def cache_response(query: str, payload: Dict[str, Any]) -> None:
    """
    Write to both exact and semantic caches concurrently.
    Fire-and-forget — does not block the response.
    """
    import asyncio
    await asyncio.gather(
        set_exact_cache(query, payload, ttl=Config.REDIS_CACHE_TTL),
        set_semantic_cache(query, payload),
        return_exceptions=True,   # never raise — cache failures are non-fatal
    )
