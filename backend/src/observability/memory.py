"""
src/memory.py — Enterprise Memory Manager
──────────────────────────────────────────
Short-term memory: LangGraph thread state (managed by checkpointer)
Long-term memory: Redis (fast K/V) + Postgres (persistent, searchable)

Patterns:
  • Thread-scoped context window (last N messages)
  • Conversation summarization (compress long threads)
  • User-level persistent memory (cross-session facts)
  • Async-first API
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.config import Config

logger = logging.getLogger(__name__)

# Maximum messages before triggering summarization
CONTEXT_WINDOW_MESSAGES = 20
SUMMARY_TRIGGER = 15


# ─────────────────────────────────────────────────────────────────────────────
# Redis memory backend
# ─────────────────────────────────────────────────────────────────────────────

_redis_client = None


async def _get_redis():
    global _redis_client
    if _redis_client is None:
        if not Config.REDIS_ENABLED:
            return None
        try:
            import redis.asyncio as aioredis
            _redis_client = aioredis.from_url(
                Config.REDIS_URL,
                password=Config.REDIS_PASSWORD or None,
                encoding="utf-8",
                decode_responses=True,
                max_connections=Config.REDIS_POOL_SIZE,
            )
            await _redis_client.ping()
            logger.info("[Memory] Redis connected")
        except Exception as e:
            logger.warning(f"[Memory] Redis unavailable: {e}")
            _redis_client = None
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# Short-term: thread-scoped window
# ─────────────────────────────────────────────────────────────────────────────

def trim_messages(
    messages: List[BaseMessage],
    max_messages: int = CONTEXT_WINDOW_MESSAGES,
) -> List[BaseMessage]:
    """
    Keep the last max_messages, always preserving the first system message
    if present. Used to prevent context window overflow.
    """
    if len(messages) <= max_messages:
        return messages

    # Keep system message + last (max_messages - 1) messages
    system_msgs = [m for m in messages if getattr(m, "type", "") == "system"]
    non_system = [m for m in messages if getattr(m, "type", "") != "system"]
    trimmed = non_system[-(max_messages - len(system_msgs)):]
    return system_msgs + trimmed


# ─────────────────────────────────────────────────────────────────────────────
# Conversation summarizer
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_conversation(messages: List[BaseMessage]) -> str:
    """
    Compress conversation history into a summary string.
    Used when thread exceeds SUMMARY_TRIGGER messages.
    """
    if len(messages) < SUMMARY_TRIGGER:
        return ""

    from src.routing.multi_llm import get_fast_llm
    history_text = "\n".join(
        f"{getattr(m, 'type', 'unknown').upper()}: {m.content}"
        for m in messages[-SUMMARY_TRIGGER:]
        if hasattr(m, "content")
    )
    prompt = Config.get_prompt("SUMMARIZE_HISTORY_PROMPT").format(history=history_text)
    try:
        resp = await get_fast_llm().ainvoke([{"role": "user", "content": prompt}])
        summary = resp.content if isinstance(resp.content, str) else str(resp.content)
        logger.debug(f"[Memory] Summarized {len(messages)} messages → {len(summary)} chars")
        return summary
    except Exception as e:
        logger.warning(f"[Memory] Summarization failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Long-term: Redis K/V memory store
# ─────────────────────────────────────────────────────────────────────────────

async def save_user_memory(user_id: str, key: str, value: Any, ttl: int = 0) -> bool:
    """Persist a fact about a user to Redis long-term memory."""
    redis = await _get_redis()
    if redis is None:
        return False
    try:
        redis_key = f"memory:{user_id}:{key}"
        payload = json.dumps({"value": value, "updated_at": datetime.now(timezone.utc).isoformat()})
        if ttl > 0:
            await redis.setex(redis_key, ttl, payload)
        else:
            await redis.set(redis_key, payload)
        return True
    except Exception as e:
        logger.warning(f"[Memory] save_user_memory error: {e}")
        return False


async def get_user_memory(user_id: str, key: str) -> Optional[Any]:
    """Retrieve a user memory fact from Redis."""
    redis = await _get_redis()
    if redis is None:
        return None
    try:
        redis_key = f"memory:{user_id}:{key}"
        raw = await redis.get(redis_key)
        if raw:
            return json.loads(raw).get("value")
        return None
    except Exception as e:
        logger.warning(f"[Memory] get_user_memory error: {e}")
        return None


async def get_all_user_memories(user_id: str) -> Dict[str, Any]:
    """Retrieve all memory entries for a user."""
    redis = await _get_redis()
    if redis is None:
        return {}
    try:
        pattern = f"memory:{user_id}:*"
        keys = await redis.keys(pattern)
        result = {}
        for k in keys:
            raw = await redis.get(k)
            if raw:
                fact_key = k.split(":", 2)[-1]
                result[fact_key] = json.loads(raw).get("value")
        return result
    except Exception as e:
        logger.warning(f"[Memory] get_all_user_memories error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Session cache: store/retrieve full conversation state
# ─────────────────────────────────────────────────────────────────────────────

async def save_session_context(session_id: str, context: Dict[str, Any]) -> bool:
    """Save session-level context (e.g. user preferences, last topic)."""
    redis = await _get_redis()
    if redis is None:
        return False
    try:
        key = f"session:{session_id}"
        payload = json.dumps(context, default=str)
        await redis.setex(key, Config.REDIS_CACHE_TTL, payload)
        return True
    except Exception as e:
        logger.warning(f"[Memory] save_session_context error: {e}")
        return False


async def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session context."""
    redis = await _get_redis()
    if redis is None:
        return None
    try:
        key = f"session:{session_id}"
        raw = await redis.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"[Memory] get_session_context error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Memory context builder (injected into AgentState.memory_context)
# ─────────────────────────────────────────────────────────────────────────────

async def build_memory_context(
    user_id: str,
    session_id: str,
    messages: List[BaseMessage],
) -> str:
    """
    Build a memory context string to inject into the system prompt.
    Combines: conversation summary + user long-term facts.
    """
    parts: List[str] = []

    # Conversation summary (if thread is long)
    if len(messages) >= SUMMARY_TRIGGER:
        summary = await summarize_conversation(messages)
        if summary:
            parts.append(f"Previous conversation summary:\n{summary}")

    # Long-term user facts
    if user_id:
        facts = await get_all_user_memories(user_id)
        if facts:
            fact_lines = [f"  • {k}: {v}" for k, v in list(facts.items())[:10]]
            parts.append("Known user preferences:\n" + "\n".join(fact_lines))

    return "\n\n".join(parts)
