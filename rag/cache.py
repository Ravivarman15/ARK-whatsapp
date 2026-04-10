"""
rag/cache.py
────────────
Question-answer cache for instant responses to frequent queries.

Supports two backends:
  1. In-memory dict with TTL (always available, zero dependencies)
  2. Redis (auto-detected when REDIS_URL is set)

Normalises question keys (lowercase, stripped of punctuation/whitespace)
so slight variations of the same question ("NEET fees?" vs "neet fees")
hit the same cache entry.

Target: < 1 second response for cached answers.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Optional

logger = logging.getLogger("ark.cache")


# ─── Key Normalisation ───────────────────────────────────────────

def _normalise_key(question: str) -> str:
    """
    Create a deterministic cache key from a question.

    Steps:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove all non-alphanumeric characters (except spaces)
      4. Collapse multiple spaces into one
      5. SHA-256 hash for fixed-length key
    """
    q = question.lower().strip()
    q = re.sub(r"[^a-z0-9\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return hashlib.sha256(q.encode()).hexdigest()


# ─── In-Memory Cache ─────────────────────────────────────────────

class InMemoryCache:
    """
    Simple TTL-based in-memory cache using a Python dict.
    Thread-safe enough for async FastAPI (GIL protects dict ops).
    """

    def __init__(self, ttl: int = 3600):
        self._store: dict[str, tuple[str, float]] = {}  # key → (answer, expiry)
        self._ttl = ttl

    def get(self, question: str) -> Optional[str]:
        key = _normalise_key(question)
        entry = self._store.get(key)
        if entry is None:
            return None
        answer, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        logger.info("Cache HIT for key %s", key[:12])
        return answer

    def set(self, question: str, answer: str) -> None:
        key = _normalise_key(question)
        self._store[key] = (answer, time.time() + self._ttl)
        logger.debug("Cache SET for key %s", key[:12])

    def clear(self) -> None:
        self._store.clear()


# ─── Redis Cache ──────────────────────────────────────────────────

class RedisCache:
    """
    Redis-backed cache. Preferred for production (persistent across
    restarts, shared across workers).
    """

    def __init__(self, redis_url: str, ttl: int = 3600):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._prefix = "ark:cache:"
        logger.info("Redis cache connected: %s", redis_url)

    def get(self, question: str) -> Optional[str]:
        key = self._prefix + _normalise_key(question)
        answer = self._client.get(key)
        if answer:
            logger.info("Redis cache HIT for key %s", key[-12:])
        return answer

    def set(self, question: str, answer: str) -> None:
        key = self._prefix + _normalise_key(question)
        self._client.setex(key, self._ttl, answer)
        logger.debug("Redis cache SET for key %s", key[-12:])

    def clear(self) -> None:
        keys = self._client.keys(self._prefix + "*")
        if keys:
            self._client.delete(*keys)


# ─── Factory ──────────────────────────────────────────────────────

_cache_instance = None


def get_cache(redis_url: str = "", ttl: int = 3600):
    """
    Return a singleton cache instance.

    If redis_url is provided and non-empty, uses Redis.
    Otherwise falls back to in-memory cache.
    """
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    if redis_url:
        try:
            _cache_instance = RedisCache(redis_url, ttl=ttl)
        except Exception as e:
            logger.warning("Redis unavailable (%s), falling back to in-memory cache.", e)
            _cache_instance = InMemoryCache(ttl=ttl)
    else:
        logger.info("Using in-memory cache (set REDIS_URL for Redis).")
        _cache_instance = InMemoryCache(ttl=ttl)

    return _cache_instance
