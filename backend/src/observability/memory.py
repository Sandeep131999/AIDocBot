"""
src/memory.py — Enterprise Memory Manager
──────────────────────────────────────────
Short-term memory: LangGraph thread state (managed by checkpointer)
Long-term and session memory: PostgreSQL JSONB

Patterns:
  • Thread-scoped context window (last N messages)
  • Conversation summarization (compress long threads)
  • User-level persistent memory (cross-session facts)
  • Async-first API
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from src.config import Config

logger = logging.getLogger(__name__)

# Maximum messages before triggering summarization
CONTEXT_WINDOW_MESSAGES = 20
SUMMARY_TRIGGER = 15


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL memory backend
# ─────────────────────────────────────────────────────────────────────────────

_postgres_pool = None


async def _get_pool():
    global _postgres_pool
    if _postgres_pool is None:
        if not Config.POSTGRES_ENABLED:
            return None
        try:
            import asyncpg

            dsn = Config.POSTGRES_URL.replace("+asyncpg", "").replace("+psycopg", "")
            _postgres_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            async with _postgres_pool.acquire() as connection:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_memory (
                        kind TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        entry_key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        expires_at TIMESTAMPTZ,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (kind, owner_id, entry_key)
                    )
                    """
                )
            logger.info("[Memory] PostgreSQL memory store connected")
        except Exception as e:
            logger.warning("[Memory] PostgreSQL memory store unavailable: %s", e)
            _postgres_pool = None
    return _postgres_pool


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
# Long-term: PostgreSQL JSONB memory store
# ─────────────────────────────────────────────────────────────────────────────

async def save_user_memory(user_id: str, key: str, value: Any, ttl: int = 0) -> bool:
    """Persist a fact about a user in PostgreSQL."""
    pool = await _get_pool()
    if pool is None:
        return False
    try:
        payload = json.dumps(value, default=str)
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO rag_memory (kind, owner_id, entry_key, value, expires_at, updated_at)
                VALUES ('user', $1, $2, $3::jsonb,
                        CASE WHEN $4 > 0 THEN now() + $4 * interval '1 second' END, now())
                ON CONFLICT (kind, owner_id, entry_key) DO UPDATE SET
                    value = EXCLUDED.value, expires_at = EXCLUDED.expires_at, updated_at = now()
                """,
                user_id,
                key,
                payload,
                ttl,
            )
        return True
    except Exception as e:
        logger.warning(f"[Memory] save_user_memory error: {e}")
        return False


async def get_user_memory(user_id: str, key: str) -> Optional[Any]:
    """Retrieve a user memory fact from PostgreSQL."""
    pool = await _get_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT value FROM rag_memory WHERE kind = 'user' AND owner_id = $1
                   AND entry_key = $2 AND (expires_at IS NULL OR expires_at > now())""",
                user_id,
                key,
            )
        return json.loads(row["value"]) if row else None
    except Exception as e:
        logger.warning(f"[Memory] get_user_memory error: {e}")
        return None


async def get_all_user_memories(user_id: str) -> Dict[str, Any]:
    """Retrieve all memory entries for a user."""
    pool = await _get_pool()
    if pool is None:
        return {}
    try:
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT entry_key, value FROM rag_memory WHERE kind = 'user' AND owner_id = $1
                   AND (expires_at IS NULL OR expires_at > now())""",
                user_id,
            )
        return {row["entry_key"]: json.loads(row["value"]) for row in rows}
    except Exception as e:
        logger.warning(f"[Memory] get_all_user_memories error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Session cache: store/retrieve full conversation state
# ─────────────────────────────────────────────────────────────────────────────

async def save_session_context(session_id: str, context: Dict[str, Any]) -> bool:
    """Save session-level context (e.g. user preferences, last topic)."""
    pool = await _get_pool()
    if pool is None:
        return False
    try:
        payload = json.dumps(context, default=str)
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO rag_memory (kind, owner_id, entry_key, value, expires_at, updated_at)
                VALUES ('session', $1, 'context', $2::jsonb,
                        now() + $3 * interval '1 second', now())
                ON CONFLICT (kind, owner_id, entry_key) DO UPDATE SET
                    value = EXCLUDED.value, expires_at = EXCLUDED.expires_at, updated_at = now()
                """,
                session_id,
                payload,
                Config.CACHE_TTL,
            )
        return True
    except Exception as e:
        logger.warning(f"[Memory] save_session_context error: {e}")
        return False


async def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session context."""
    pool = await _get_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT value FROM rag_memory WHERE kind = 'session' AND owner_id = $1
                   AND entry_key = 'context' AND (expires_at IS NULL OR expires_at > now())""",
                session_id,
            )
        return json.loads(row["value"]) if row else None
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


async def close_memory_pool() -> None:
    """Close the optional PostgreSQL memory pool on application shutdown."""
    global _postgres_pool
    if _postgres_pool is not None:
        await _postgres_pool.close()
        _postgres_pool = None
