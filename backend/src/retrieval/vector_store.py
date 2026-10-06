"""
src/vector_store.py — Enterprise Hybrid Search + Reranking
────────────────────────────────────────────────────────────
Features:
    • Hybrid search: PostgreSQL full-text + pgvector dense, RRF fusion
  • Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  • HyDE: Hypothetical Document Embeddings for better query coverage
  • Relevance score filtering (MIN_RELEVANCE_SCORE)
  • Query rewriting before retrieval
  • Thread-safe singleton vector store
  • Async retrieval wrapper
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from contextvars import ContextVar
from functools import lru_cache, partial
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.config import Config
from src.retrieval.embeddings import get_embeddings
from src.retrieval.postgres_store import PostgresVectorStore

logger = logging.getLogger(__name__)
_project_id: ContextVar[str] = ContextVar("rag_project_id", default="default")


def set_project_scope(project_id: str):
    """Set the project scope for retrieval performed during this request."""
    return _project_id.set(project_id)


def reset_project_scope(token) -> None:
    _project_id.reset(token)


def current_project_scope() -> str:
    return _project_id.get()


def _document_identity(doc: Document) -> str:
    metadata = doc.metadata
    vector_id = metadata.get("_vector_id")
    if vector_id:
        return str(vector_id)
    document_id = metadata.get("document_id")
    if document_id:
        chunk_index = metadata.get("chunk_index")
        return f"{document_id}#chunk-{chunk_index}" if chunk_index is not None else str(document_id)
    source = metadata.get("file_hash") or metadata.get("source")
    if source:
        return f"{source}#chunk-{metadata.get('chunk_index', 0)}"
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Vector store singleton
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> PostgresVectorStore:
    """Return the process-wide PostgreSQL/pgvector store."""
    connection_string = Config.POSTGRES_URL.replace("+asyncpg", "+psycopg")
    logger.info("[VectorStore] Initializing PostgreSQL pgvector store")
    return PostgresVectorStore(connection_string, get_embeddings())


def close_vector_store() -> None:
    """Close the database pool if the vector store was initialized."""
    if get_vector_store.cache_info().currsize:
        vector_store = get_vector_store()
        get_vector_store.cache_clear()
        vector_store.close()


def index_documents(documents: List[Document]) -> List[str]:
    """Index one version of a source file without destroying its history."""
    if not documents:
        return []

    filenames = {str(doc.metadata.get("filename", "")) for doc in documents}
    if len(filenames) != 1 or not next(iter(filenames)):
        raise ValueError("All indexed chunks must belong to one named source file")
    filename = next(iter(filenames))
    project_ids = {str(doc.metadata.get("project_id", "default")) for doc in documents}
    if len(project_ids) != 1:
        raise ValueError("All indexed chunks must belong to exactly one project")
    project_id = next(iter(project_ids))

    ids = []
    for position, doc in enumerate(documents):
        source = str(doc.metadata.get("source", filename))
        document_id = str(
            doc.metadata.get("document_id")
            or doc.metadata.get("file_hash")
            or filename
        )
        file_hash = str(doc.metadata.get("file_hash", document_id))
        chunk_index = str(doc.metadata.get("chunk_index", position))
        doc.metadata.setdefault("active_version", True)
        identity = f"{project_id}\0{source}\0{file_hash}\0{document_id}\0{chunk_index}"
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())

    if len(ids) != len(set(ids)):
        raise ValueError("Indexed chunks produced duplicate stable IDs")

    vector_store = get_vector_store()
    vector_store.add_documents(documents, ids=ids)

    return ids


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL native full-text search
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_fusion(
    dense_results: List[Document],
    sparse_results: List[Tuple[Document, float]],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    k: int = 60,
) -> List[Document]:
    """
    Combine dense + sparse rankings using Reciprocal Rank Fusion.
    k=60 is the standard RRF constant from the original paper.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Dense scores
    for rank, doc in enumerate(dense_results):
        doc_id = _document_identity(doc)
        scores[doc_id] = scores.get(doc_id, 0) + dense_weight * (1.0 / (k + rank + 1))
        doc_map[doc_id] = doc

    # Sparse scores
    for rank, (doc, _score) in enumerate(sparse_results):
        doc_id = _document_identity(doc)
        scores[doc_id] = scores.get(doc_id, 0) + sparse_weight * (1.0 / (k + rank + 1))
        doc_map[doc_id] = doc

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in ranked_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker
# ─────────────────────────────────────────────────────────────────────────────

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(Config.RERANKER_MODEL)
            logger.info(f"[Reranker] Loaded: {Config.RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load: {e}. Reranking disabled.")
            _reranker = False
    return _reranker if _reranker is not False else None


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int,
) -> List[Document]:
    """Rerank docs with cross-encoder, return top_k."""
    if not docs:
        return docs
    reranker = _get_reranker()
    if reranker is None:
        return docs[:top_k]
    try:
        pairs = [(query, doc.page_content[:512]) for doc in docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked = [d for d, _ in scored[:top_k]]
        logger.debug(f"[Reranker] {len(docs)} → {len(reranked)} docs")
        return reranked
    except Exception as e:
        logger.warning(f"[Reranker] Error: {e} — falling back to original order")
        return docs[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# HyDE — Hypothetical Document Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def generate_hypothetical_document(query: str) -> str:
    """
    Generate a hypothetical document that would answer the query.
    Used to create a better embedding vector than the raw question.
    """
    from src.routing.multi_llm import get_fast_llm
    prompt = Config.get_prompt("HYDE_PROMPT").format(question=query)
    try:
        resp = get_fast_llm().invoke([{"role": "user", "content": prompt}])
        content = resp.content
        if isinstance(content, list):
            content = " ".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in content
            )
        return str(content)
    except Exception as e:
        logger.warning(f"[HyDE] Generation failed: {e} — using raw query")
        return query


# ─────────────────────────────────────────────────────────────────────────────
# Main retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    k: Optional[int] = None,
    strategy: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """
    Enterprise retrieval with hybrid search, HyDE, and reranking.

    Args:
        query:           User query or rewritten query
        k:               Number of results (default: Config.TOP_K)
        strategy:        Override retrieval strategy
        score_threshold: Min relevance score (default: Config.MIN_RELEVANCE_SCORE)

    Returns:
        List of Document objects, ordered by relevance.
    """
    top_k = k or Config.TOP_K
    strat = strategy or Config.RETRIEVAL_STRATEGY
    threshold = score_threshold if score_threshold is not None else Config.MIN_RELEVANCE_SCORE
    project_id = _project_id.get()

    vs = get_vector_store()
    fetch_k = top_k * 4   # fetch more candidates for fusion/reranking

    # ── HyDE: embed a hypothetical document instead of raw query ──────────
    embed_query = query
    if Config.HYDE_ENABLED and strat in ("hyde", "hybrid"):
        hypo_doc = generate_hypothetical_document(query)
        logger.debug(f"[HyDE] Generated hypothetical doc ({len(hypo_doc)} chars)")
        embed_query = hypo_doc

    # ── Dense retrieval ────────────────────────────────────────────────────
    dense_results: List[Document] = []
    if strat in ("dense", "hybrid", "hyde"):
        try:
            results_with_scores = vs.similarity_search_with_relevance_scores(
                embed_query, k=fetch_k, project_id=project_id
            )
            # Filter by score threshold
            dense_results = [
                doc for doc, score in results_with_scores
                if score >= threshold
            ]
            logger.debug(f"[Dense] {len(results_with_scores)} retrieved → {len(dense_results)} above threshold")
        except Exception as e:
            logger.warning(f"[Dense] Retrieval error: {e}")

    # ── PostgreSQL full-text sparse retrieval ───────────────────────────────
    sparse_results: List[Tuple[Document, float]] = []
    if strat in ("bm25", "hybrid"):
        try:
            sparse_results = vs.keyword_search(query, k=fetch_k, project_id=project_id)
            logger.debug(f"[BM25] {len(sparse_results)} results")
        except Exception as e:
            logger.warning(f"[BM25] Error: {e}")

    # ── Fusion ─────────────────────────────────────────────────────────────
    if strat == "hybrid" and dense_results and sparse_results:
        candidates = _rrf_fusion(
            dense_results, sparse_results,
            dense_weight=Config.DENSE_WEIGHT,
            sparse_weight=Config.BM25_WEIGHT,
        )
    elif dense_results:
        candidates = dense_results
    elif sparse_results:
        candidates = [doc for doc, _ in sparse_results]
    else:
        logger.warning("[Retrieval] No results from any strategy")
        return []

    # ── Reranking ──────────────────────────────────────────────────────────
    if Config.RERANKER_ENABLED and len(candidates) > top_k:
        final = rerank_documents(query, candidates, top_k=top_k)
    else:
        final = candidates[:top_k]

    logger.info(f"[Retrieval] Strategy={strat} | Query='{query[:60]}...' | Results={len(final)}")
    return final


async def retrieve_async(
    query: str,
    k: Optional[int] = None,
    strategy: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """Async wrapper for retrieve() — safe for FastAPI endpoints."""
    loop = asyncio.get_event_loop()
    fn = partial(retrieve, query, k, strategy, score_threshold)
    return await loop.run_in_executor(None, fn)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible retriever (for ToolNode / bind_tools)
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(k: Optional[int] = None):
    """
    Returns a LangChain BaseRetriever wrapping our enterprise retrieve().
    Used by get_retriever_tool() and the evaluator.
    """
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    top_k = k or Config.TOP_K

    class EnterpriseRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return retrieve(query, k=top_k)

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return await retrieve_async(query, k=top_k)

    return EnterpriseRetriever()
