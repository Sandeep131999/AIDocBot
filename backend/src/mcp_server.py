"""
src/mcp_server.py — Enterprise MCP Server (FastMCP)
─────────────────────────────────────────────────────
Model Context Protocol server exposing the RAG system as MCP tools/resources.
Enables A2A (agent-to-agent) and IDE integration (Cursor, VS Code, Claude Desktop).

Transport: Streamable HTTP (default) or stdio
Auth: Bearer token (MCP_AUTH_TOKEN in .env)

Tools exposed:
  • search_knowledge_base  — hybrid RAG search
  • upload_document        — add a document to the vector store
  • list_documents         — list all indexed documents
  • get_document_chunks    — retrieve chunks from a specific file

Resources exposed:
  • rag://status           — system health and index stats
  • rag://config           — current retrieval config (non-sensitive)

Prompts exposed:
  • rag_system_prompt      — the system prompt used by the RAG agent
  • document_qa_prompt     — template for Q&A over a specific document
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FastMCP app
# ─────────────────────────────────────────────────────────────────────────────

def create_mcp_server():
    """Build and return the FastMCP server instance."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        logger.error("[MCP] fastmcp not installed. Run: pip install fastmcp")
        raise

    from src.config import Config

    mcp = FastMCP(
        name=f"{Config.APP_NAME} MCP",
        version=Config.APP_VERSION,
        description="Enterprise Agentic RAG — search, index, and query your knowledge base via MCP.",
    )

    # ── Tools ───────────────────────────────────────────────────────────────

    @mcp.tool(
        description=(
            "Search the enterprise knowledge base for documents relevant to a query. "
            "Returns formatted document excerpts with source citations. "
            "Use this to answer questions about indexed documents."
        )
    )
    async def search_knowledge_base(
        query: str,
        k: int = 5,
        strategy: str = "hybrid",
    ) -> str:
        """
        Hybrid RAG search (BM25 + dense + reranking).
        Args:
            query: Natural language search query
            k: Number of results to return (1-20)
            strategy: 'hybrid' | 'dense' | 'bm25' | 'hyde'
        """
        if not query.strip():
            return "Error: query cannot be empty"
        if k < 1 or k > 20:
            return "Error: k must be between 1 and 20"

        from src.vector_store import retrieve
        try:
            docs = retrieve(query, k=k, strategy=strategy)
            if not docs:
                return "No relevant documents found for this query."

            parts = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page", "")
                page_str = f", p.{page}" if page else ""
                excerpt = doc.page_content[:600].strip()
                parts.append(f"[{i}] {src}{page_str}\n{excerpt}")

            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"[MCP:search] Error: {e}")
            return f"Search error: {str(e)}"

    @mcp.tool(
        description=(
            "Add a document to the knowledge base by providing its file path. "
            "Supports PDF, DOCX, TXT, MD, CSV, JSON, HTML, PPTX, XLSX. "
            "The document will be chunked, embedded, and indexed for retrieval."
        )
    )
    async def index_document(file_path: str, strategy: str = "recursive") -> str:
        """
        Index a document from a file path.
        Args:
            file_path: Absolute path to the document file
            strategy: Chunking strategy: 'recursive' | 'markdown' | 'sentence'
        """
        from pathlib import Path
        from src.document_loader import load_and_split_async
        from src.vector_store import get_vector_store

        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        try:
            chunks = await load_and_split_async(str(path), strategy=strategy)
            vs = get_vector_store()
            vs.add_documents(chunks)
            return f"Successfully indexed {len(chunks)} chunks from '{path.name}'"
        except Exception as e:
            logger.error(f"[MCP:index] Error: {e}")
            return f"Indexing error: {str(e)}"

    @mcp.tool(
        description="List all documents currently indexed in the knowledge base."
    )
    async def list_indexed_documents() -> str:
        """Returns a formatted list of all indexed files with chunk counts."""
        from src.vector_store import get_vector_store
        try:
            vs = get_vector_store()
            result = vs._collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])

            if not metadatas:
                return "No documents indexed yet."

            files: dict[str, dict] = {}
            for meta in metadatas:
                fname = meta.get("filename", "Unknown")
                if fname not in files:
                    files[fname] = {"format": meta.get("format", "?"), "chunks": 0}
                files[fname]["chunks"] += 1

            lines = [f"📚 {len(files)} documents indexed:"]
            for fname, info in sorted(files.items()):
                lines.append(f"  • {fname} ({info['format']}) — {info['chunks']} chunks")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool(
        description="Run a full agentic RAG query through the LangGraph pipeline and return the answer."
    )
    async def ask_agent(
        question: str,
        session_id: str = "mcp-default",
    ) -> str:
        """
        Full agentic RAG: input guard → retrieve → grade → generate → output guard.
        Args:
            question: The question to ask the RAG agent
            session_id: Session identifier for conversation continuity
        """
        from src.graph import build_agentic_rag_graph
        try:
            graph = build_agentic_rag_graph()
            config = {"configurable": {"thread_id": session_id}}
            result = await graph.ainvoke(
                {
                    "messages": [{"role": "user", "content": question}],
                    "question": question,
                    "session_id": session_id,
                },
                config=config,
            )
            answer = result.get("answer", "")
            if not answer:
                msgs = result.get("messages", [])
                if msgs:
                    answer = getattr(msgs[-1], "content", "")
            if isinstance(answer, list):
                answer = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in answer
                )
            return str(answer) if answer else "No answer generated."
        except Exception as e:
            logger.error(f"[MCP:ask_agent] Error: {e}")
            return f"Agent error: {str(e)}"

    # ── Resources ───────────────────────────────────────────────────────────

    @mcp.resource("rag://status")
    async def get_status() -> str:
        """System health and index statistics."""
        from src.vector_store import get_vector_store
        from src.config import Config
        try:
            vs = get_vector_store()
            doc_count = vs._collection.count()
        except Exception:
            doc_count = -1

        return (
            f"# RAG System Status\n\n"
            f"- App: {Config.APP_NAME} v{Config.APP_VERSION}\n"
            f"- Environment: {Config.APP_ENV}\n"
            f"- Indexed documents: {doc_count} chunks\n"
            f"- Retrieval strategy: {Config.RETRIEVAL_STRATEGY}\n"
            f"- Reranker: {'enabled' if Config.RERANKER_ENABLED else 'disabled'}\n"
            f"- HyDE: {'enabled' if Config.HYDE_ENABLED else 'disabled'}\n"
            f"- Guardrails: {'enabled' if Config.GUARDRAIL_ENABLED else 'disabled'}\n"
            f"- Redis cache: {'enabled' if Config.REDIS_ENABLED else 'disabled'}\n"
        )

    @mcp.resource("rag://config")
    async def get_config() -> str:
        """Current retrieval configuration (non-sensitive values only)."""
        from src.config import Config
        return (
            f"# RAG Configuration\n\n"
            f"## Retrieval\n"
            f"- Strategy: {Config.RETRIEVAL_STRATEGY}\n"
            f"- Top-K: {Config.TOP_K}\n"
            f"- Min relevance score: {Config.MIN_RELEVANCE_SCORE}\n"
            f"- BM25 weight: {Config.BM25_WEIGHT}\n"
            f"- Dense weight: {Config.DENSE_WEIGHT}\n\n"
            f"## Chunking\n"
            f"- Strategy: {Config.CHUNK_STRATEGY}\n"
            f"- Chunk size: {Config.CHUNK_SIZE} tokens\n"
            f"- Overlap: {Config.CHUNK_OVERLAP} tokens\n\n"
            f"## LLM\n"
            f"- Provider order: {Config.LLM_PROVIDER_ORDER}\n"
            f"- Temperature: {Config.LLM_TEMPERATURE}\n"
            f"- Max tokens: {Config.LLM_MAX_TOKENS}\n"
        )

    # ── Prompts ─────────────────────────────────────────────────────────────

    @mcp.prompt(description="The system prompt used by the RAG agent")
    def rag_system_prompt() -> str:
        from src.config import Config
        return Config.get_prompt("GENERATE_QUERY_SYSTEM_PROMPT")

    @mcp.prompt(description="Template for Q&A over a specific document")
    def document_qa_prompt(document_name: str, question: str) -> str:
        return (
            f"You are answering questions about the document: {document_name}\n\n"
            f"Use the search_knowledge_base tool to find relevant sections, "
            f"then answer this question:\n\n{question}"
        )

    return mcp


# ─────────────────────────────────────────────────────────────────────────────
# Standalone server entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_mcp_server() -> None:
    """Run the MCP server as a standalone process."""
    from src.config import Config

    if not Config.MCP_ENABLED:
        logger.warning("[MCP] MCP_ENABLED=false — server not starting")
        return

    mcp = create_mcp_server()
    transport = Config.MCP_TRANSPORT

    logger.info(f"[MCP] Starting server on {Config.MCP_SERVER_HOST}:{Config.MCP_SERVER_PORT} ({transport})")

    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=Config.MCP_SERVER_HOST,
            port=Config.MCP_SERVER_PORT,
        )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
