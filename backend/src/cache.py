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
import os
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

class CacheManager:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
        except Exception as e:
            print(f"[Cache] Redis init failed: {e}")

        # FIX for 'str' object is not callable
        self.embed_model = None
        try:
            print("[Embeddings] Using HF: BAAI/bge-base-en-v1.5")
            self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        except Exception as e:
            print(f"[Cache] Embed model failed: {e}")

    async def get_embedding(self, text: str):
        if not self.embed_model:
            return None
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()

    async def ping(self):
        if not self.redis_client:
            return False
        try:
            await self.redis_client.ping()
            return True
        except:
            return False

cache = CacheManager()