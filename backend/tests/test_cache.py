import asyncio

from src.retrieval import cache as cache_module


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def scan_iter(self, match):
        prefix = match.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key


class FailingRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.get_calls = 0

    async def get(self, key):
        self.get_calls += 1
        raise cache_module.redis.RedisError("connection timed out")


def test_cache_read_write_stats_and_invalidation(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(cache_module.cache, "redis_client", client)
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
            "redis_available": True,
        }
        assert await cache_module.invalidate_cache("cache:*") == 1
        assert await cache_module.get_cached_response("query") is None

    asyncio.run(exercise_cache())


def test_redis_failure_enters_retry_cooldown(monkeypatch, caplog):
    client = FailingRedis()
    monkeypatch.setattr(cache_module.cache, "redis_client", client)
    monkeypatch.setattr(cache_module.cache, "_redis_retry_after", 0.0)
    monkeypatch.setattr(cache_module.cache, "_redis_available", True)

    async def exercise_failure():
        assert await cache_module.get_cached_response("query") is None
        assert await cache_module.get_cached_response("another query") is None

    asyncio.run(exercise_failure())

    assert client.get_calls == 1
    assert cache_module.get_cache_stats()["redis_available"] is False
    assert caplog.text.count("Redis unavailable") == 1