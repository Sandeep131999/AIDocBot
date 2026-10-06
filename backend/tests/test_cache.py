import asyncio

from src.retrieval import cache as cache_module


def test_cache_read_write_stats_and_invalidation(monkeypatch):
    cache_module.cache.responses.clear()
    monkeypatch.setattr(cache_module.cache, "hits", 0)
    monkeypatch.setattr(cache_module.cache, "misses", 0)

    async def exercise_cache():
        assert await cache_module.get_cached_response("query") is None
        await cache_module.cache_response("query", {"answer": "ok"})

        assert await cache_module.get_cached_response(" query ") == {
            "answer": "ok",
            "cache_type": "exact",
        }
        assert cache_module.get_cache_stats() == {
            "hits": 1,
            "misses": 1,
            "hit_rate": 0.5,
            "cache_enabled": True,
        }
        assert await cache_module.invalidate_cache("cache:*") == 1
        assert await cache_module.get_cached_response("query") is None

    asyncio.run(exercise_cache())
