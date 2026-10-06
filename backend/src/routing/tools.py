"""
src/tools.py — Enterprise LangChain Tools
──────────────────────────────────────────
Features:
  • retrieve_documents — hybrid search tool with idempotency key
  • search_with_filter — metadata-filtered retrieval
  • summarize_document — summarize a specific source
  • All tools have clear JSON schemas, timeouts, and error handling
  • Tools update AgentState.retrieved_docs and tool_calls_made
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool, create_retriever_tool
from pydantic import BaseModel, Field

from src.config import Config
from src.retrieval.vector_store import (
    current_project_scope,
    get_retriever,
    retrieve,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (Pydantic v2 — for structured tool calling)
# ─────────────────────────────────────────────────────────────────────────────

class RetrieveInput(BaseModel):
    query: str = Field(description="The search query to find relevant documents in the knowledge base")
    k: int = Field(default=5, description="Number of documents to retrieve", ge=1, le=20)
    strategy: Optional[str] = Field(
        default=None,
        description="Retrieval strategy: 'hybrid' (default), 'dense', 'bm25', 'hyde'"
    )


class FilteredRetrieveInput(BaseModel):
    query: str = Field(description="The search query")
    filename: Optional[str] = Field(default=None, description="Filter by specific filename")
    format: Optional[str] = Field(default=None, description="Filter by file format (pdf, docx, etc.)")
    k: int = Field(default=5, description="Number of documents", ge=1, le=20)


class WebSearchInput(BaseModel):
    query: str = Field(description="A concise web search query")
    max_results: int = Field(default=5, ge=1, le=10)


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieveInput)
def retrieve_documents(query: str, k: int = 5, strategy: Optional[str] = None) -> str:
    """
    Search the knowledge base for documents relevant to the query.
    Uses hybrid BM25 + dense vector search with cross-encoder reranking.
    Returns formatted document excerpts with source citations.

    Use this tool whenever the user asks a question that requires
    information from uploaded documents.
    """
    try:
        docs = retrieve(query, k=k, strategy=strategy)
        if not docs:
            return "No relevant documents found in the knowledge base for this query."

        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            excerpt = doc.page_content[:800].strip()
            formatted_parts.append(
                f"[Document {i}] Source: {source}{page_info}\n{excerpt}"
            )

        result = "\n\n---\n\n".join(formatted_parts)
        logger.info(f"[Tool:retrieve] Query='{query[:50]}' → {len(docs)} docs")
        return result

    except Exception as e:
        logger.error(f"[Tool:retrieve] Error: {e}")
        return f"Retrieval error: {str(e)}"


@tool(args_schema=FilteredRetrieveInput)
def retrieve_with_filter(
    query: str,
    filename: Optional[str] = None,
    format: Optional[str] = None,
    k: int = 5,
) -> str:
    """
    Search the knowledge base with metadata filters.
    Use this when the user specifies a particular document or file type.
    """
    try:
        from src.retrieval.vector_store import get_vector_store
        vs = get_vector_store()

        where_filter = {}
        if filename:
            where_filter["filename"] = {"$eq": filename}
        if format:
            where_filter["format"] = {"$eq": format.lstrip(".")}

        filters = {}
        if filename:
            filters["filename"] = filename
        if format:
            filters["format"] = format.lstrip(".")
        docs = [
            document
            for document, _score in vs.similarity_search_with_relevance_scores(
                query,
                k=k,
                metadata_filter=filters,
                project_id=current_project_scope(),
            )
        ]

        if not docs:
            filter_desc = f" (filename={filename}, format={format})" if where_filter else ""
            return f"No documents found matching the filter{filter_desc}."

        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("filename", "Unknown")
            parts.append(f"[Document {i}] Source: {src}\n{doc.page_content[:800]}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"[Tool:retrieve_filtered] Error: {e}")
        return f"Filtered retrieval error: {str(e)}"


@tool
def list_indexed_documents() -> str:
    """
    List all documents currently indexed in the knowledge base.
    Returns filenames, formats, and chunk counts.
    Use this when the user asks what documents are available.
    """
    try:
        from src.retrieval.vector_store import get_vector_store
        vs = get_vector_store()
        metadatas = [
            document.metadata
            for document in vs.list_documents(
                project_id=current_project_scope(),
                active_only=True,
            )
        ]

        if not metadatas:
            return "No documents are currently indexed in the knowledge base."

        # Aggregate by filename
        files: dict[str, dict] = {}
        for meta in metadatas:
            fname = meta.get("filename", "Unknown")
            if fname not in files:
                files[fname] = {
                    "format": meta.get("format", "?"),
                    "chunks": 0,
                    "indexed_at": meta.get("indexed_at", "?"),
                }
            files[fname]["chunks"] += 1

        lines = [f"📚 Indexed Documents ({len(files)} files):"]
        for fname, info in sorted(files.items()):
            lines.append(f"  • {fname} ({info['format']}) — {info['chunks']} chunks, indexed {info['indexed_at'][:10]}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[Tool:list_docs] Error: {e}")
        return f"Error listing documents: {str(e)}"


@tool(args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """Search current public web information through DuckDuckGo; no API key is required."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
        formatted = [
            f"[{index}] {item.get('title', 'Untitled')}\n"
            f"{item.get('body', '')}\nSource: {item.get('href', '')}"
            for index, item in enumerate(results, 1)
        ]
        return "\n\n".join(formatted) if formatted else "No web results found."
    except Exception as exc:
        logger.warning("[Tool:web_search] DuckDuckGo search failed: %s", exc)
        return "Web search is temporarily unavailable."


# ─────────────────────────────────────────────────────────────────────────────
# Standard LangChain retriever tool (for ToolNode compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever_tool():
    """
    Returns a LangChain create_retriever_tool wrapping EnterpriseRetriever.
    Used by nodes.py for LLM tool binding and ToolNode.
    """
    retriever = get_retriever(k=Config.TOP_K)
    return create_retriever_tool(
        retriever,
        name="retrieve_documents",
        description=(
            "Search and retrieve relevant documents from the enterprise knowledge base. "
            "Use this tool to find information before answering questions about uploaded documents. "
            "Input should be a clear, specific search query."
        ),
        response_format="content_and_artifact",
    )


def get_all_tools():
    """Return all available tools for agent binding."""
    return [
        get_retriever_tool(),
        retrieve_with_filter,
        list_indexed_documents,
        web_search,
    ]
