"""
tests/conftest.py — Shared pytest fixtures
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment before any imports
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("APP_NAME", "test-rag")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("GUARDRAIL_ENABLED", "true")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("POSTGRES_ENABLED", "false")
os.environ.setdefault("LANGSMITH_ENABLED", "false")
os.environ.setdefault("PROMETHEUS_ENABLED", "false")
os.environ.setdefault("CHUNK_SIZE", "800")
os.environ.setdefault("CHUNK_OVERLAP", "150")
os.environ.setdefault("CHUNK_STRATEGY", "recursive")
os.environ.setdefault("TOP_K", "5")
os.environ.setdefault("MIN_RELEVANCE_SCORE", "0.35")
os.environ.setdefault("LLM_TEMPERATURE", "0.1")
os.environ.setdefault("LLM_MAX_TOKENS", "512")
os.environ.setdefault("LLM_PROVIDER_ORDER", "ollama")
os.environ.setdefault("VECTOR_DB_PATH", "./test_chroma_db")
os.environ.setdefault("VECTOR_COLLECTION", "test_docs")
os.environ.setdefault("GUARDRAIL_MAX_INPUT_CHARS", "5000")
os.environ.setdefault("MAX_FILE_SIZE_MB", "10")
os.environ.setdefault("ALLOWED_EXTENSIONS", ".pdf,.txt,.md,.json")
os.environ.setdefault("UPLOAD_DIR", "./test_uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_documents():
    """Minimal Document objects for testing."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content="The reset password process requires you to click Forgot Password on the login page.",
            metadata={"source": "faq.pdf", "filename": "faq.pdf", "format": "pdf", "file_hash": "abc123", "chunk_index": 0},
        ),
        Document(
            page_content="To contact support, email support@example.com or call 1-800-SUPPORT.",
            metadata={"source": "contact.md", "filename": "contact.md", "format": "md", "file_hash": "def456", "chunk_index": 0},
        ),
        Document(
            page_content="The enterprise plan includes unlimited users, SSO, and a dedicated account manager.",
            metadata={"source": "pricing.pdf", "filename": "pricing.pdf", "format": "pdf", "file_hash": "ghi789", "chunk_index": 0},
        ),
    ]


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a predictable response."""
    from langchain_core.messages import AIMessage
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test answer from mock LLM."))
    llm.invoke = MagicMock(return_value=AIMessage(content="Test answer from mock LLM."))
    llm.bind_tools = MagicMock(return_value=llm)
    llm.with_fallbacks = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_vector_store(sample_documents):
    """A mock ChromaDB vector store."""
    vs = MagicMock()
    vs._collection.count.return_value = len(sample_documents)
    vs._collection.get.return_value = {"metadatas": [d.metadata for d in sample_documents], "ids": ["1", "2", "3"]}
    vs.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.92) for doc in sample_documents
    ]
    vs.add_documents = MagicMock(return_value=["1", "2", "3"])
    return vs


@pytest.fixture
def fast_api_client():
    """Async HTTP client for the FastAPI app."""
    from fastapi.testclient import TestClient
    # Patch heavy components so tests don't need full env
    with patch("src.graph.build_agentic_rag_graph") as mock_graph:
        mock_graph.return_value = AsyncMock()
        mock_graph.return_value.ainvoke = AsyncMock(return_value={
            "answer": "Test answer",
            "messages": [],
            "guard_blocked": False,
            "guard_reason": "",
            "retrieved_docs": [],
            "relevant_docs": [],
            "token_usage": None,
            "llm_provider_used": "mock",
            "node_trace": ["test"],
        })
        from api.main import app
        yield TestClient(app)
