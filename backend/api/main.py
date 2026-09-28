"""
api/main.py — Enterprise FastAPI Application
──────────────────────────────────────────────
Endpoints:
  GET  /                           — root health ping
  GET  /api/health                 — detailed health (vector DB, Redis, LLM)
  GET  /metrics                    — Prometheus metrics (if enabled)

  POST /api/chat                   — standard JSON chat
  POST /api/chat/stream            — SSE streaming chat
  WS   /api/chat/ws/{session_id}   — WebSocket real-time chat

  POST /api/documents/upload       — upload + index a document (background task)
  GET  /api/documents              — list all indexed documents
  DELETE /api/documents/{filename} — remove a document from the index

  POST /api/evaluate               — RAGAS + retrieval evaluation
  GET  /api/cache/stats            — cache hit/miss statistics
  POST /api/cache/invalidate       — flush cache (admin only)
  GET  /api/config                 — current config dump (admin only)

  POST /auth/token                 — issue JWT (dev helper, disabled in prod)

Features:
  • SSE streaming via sse-starlette
  • WebSocket with session continuity
  • Background tasks for indexing (non-blocking upload)
  • Dependency injection (get_graph, get_cache)
  • JWT + API Key auth with RBAC (bypass when AUTH_ENABLED=false)
  • Rate limiting via slowapi
  • Prometheus metrics exposure
  • LangSmith trace_id in response headers
  • Semantic + exact cache layer
  • Async document loading (no blocking)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import Config
from src.observability import (
    get_logger,
    record_chat_metrics,
    record_guard_block,
    record_upload,
    inc_active_sessions,
)

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Application state (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    graph = None
    supervisor_graph = None


_app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("startup", app=Config.APP_NAME, version=Config.APP_VERSION, env=Config.APP_ENV)
    try:
        # Build RAG graph
        from src.graph import build_agentic_rag_graph
        _app_state.graph = build_agentic_rag_graph()
        logger.info("graph_ready", type="agentic_rag")

        # Ensure directories exist
        Path(Config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

        # Log config (masks secrets)
        if Config.DEBUG:
            Config.print_config()

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("shutdown", app=Config.APP_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=Config.APP_NAME,
        version=Config.APP_VERSION,
        description=(
            "Enterprise Agentic RAG — LangGraph + FastAPI\n\n"
            "Features: Hybrid Search, Reranking, HyDE, CRAG, Reflection, "
            "Guardrails, PII Detection, Semantic Cache, MCP, RAGAS Eval"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Register all middleware
    from api.middleware import register_middleware
    register_middleware(app)

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# Dependency injectors
# ─────────────────────────────────────────────────────────────────────────────

def get_graph():
    if _app_state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized — startup failed")
    return _app_state.graph


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.GUARDRAIL_MAX_INPUT_CHARS)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    top_k: int = Field(default=Config.TOP_K, ge=1, le=20)
    retrieval_strategy: Optional[str] = Field(default=None)
    stream: bool = False
    use_cache: bool = True


class SourceDoc(BaseModel):
    filename: str
    page: Optional[int] = None
    excerpt: str
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc] = []
    session_id: str = ""
    trace_id: str = ""
    guard_blocked: bool = False
    guard_reason: str = ""
    cache_hit: bool = False
    cache_type: str = "none"
    latency_ms: float = 0.0
    token_usage: Optional[Dict[str, Any]] = None
    provider_used: str = ""
    node_trace: List[str] = []


class UploadResponse(BaseModel):
    filename: str
    indexed_chunks: int
    message: str
    file_hash: str = ""
    job_id: str = ""


class EvalRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None
    relevant_doc_ids: Optional[List[List[str]]] = None


class TokenRequest(BaseModel):
    user_id: str = "dev-user"
    role: str = "admin"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    """Handle multi-modal Gemini 2.5 content (list of dicts) or plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", str(p)))
            else:
                parts.append(getattr(p, "text", str(p)))
        return "\n".join(parts)
    return str(content)


def _build_sources(result: Dict[str, Any]) -> List[SourceDoc]:
    """Extract structured source citations from graph result."""
    sources = []
    docs = result.get("relevant_docs") or result.get("retrieved_docs", [])
    for doc in docs[:5]:   # cap at 5 sources
        meta = getattr(doc, "metadata", {}) or {}
        sources.append(SourceDoc(
            filename=meta.get("filename", meta.get("source", "Unknown")),
            page=meta.get("page"),
            excerpt=getattr(doc, "page_content", "")[:300],
            chunk_index=meta.get("chunk_index"),
        ))
    return sources


async def _run_graph(
    query: str,
    session_id: str,
    graph,
) -> Dict[str, Any]:
    """Invoke the LangGraph graph with proper config and return the state dict."""
    config = {"configurable": {"thread_id": session_id}}
    return await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": query}],
            "question": query,
            "session_id": session_id,
        },
        config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": Config.APP_NAME,
        "version": Config.APP_VERSION,
        "environment": Config.APP_ENV,
        "graph_ready": _app_state.graph is not None,
    }


@app.get("/api/health", tags=["Health"])
async def health():
    checks: Dict[str, Any] = {"status": "healthy"}

    # Vector DB
    try:
        from src.vector_store import get_vector_store
        vs = get_vector_store()
        checks["vector_db"] = {"status": "ok", "chunks": vs._collection.count()}
    except Exception as e:
        checks["vector_db"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    # Redis
    if Config.REDIS_ENABLED:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(Config.REDIS_URL)
            await r.ping()
            await r.aclose()
            checks["redis"] = {"status": "ok"}
        except Exception as e:
            checks["redis"] = {"status": "error", "detail": str(e)}
            checks["status"] = "degraded"
    else:
        checks["redis"] = {"status": "disabled"}

    # Features
    checks["features"] = {
        "guardrails":           Config.GUARDRAIL_ENABLED,
        "pii_detection":        Config.PII_DETECTION_ENABLED,
        "reranker":             Config.RERANKER_ENABLED,
        "hyde":                 Config.HYDE_ENABLED,
        "hybrid_search":        Config.RETRIEVAL_STRATEGY == "hybrid",
        "semantic_cache":       Config.REDIS_ENABLED,
        "langsmith_tracing":    Config.LANGSMITH_ENABLED,
        "mcp_server":           Config.MCP_ENABLED,
        "auth":                 Config.AUTH_ENABLED,
        "rate_limiting":        Config.RATE_LIMIT_ENABLED,
    }

    return checks


# ── Chat: standard JSON ──────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    graph=Depends(get_graph),
):
    """
    Standard synchronous chat endpoint.
    Includes semantic cache, guardrails, hybrid retrieval, and full observability.
    """
    start = time.perf_counter()
    query = request.query.strip()
    session_id = request.session_id

    # ── Cache check ───────────────────────────────────────────────────────────
    if request.use_cache:
        try:
            from src.cache import get_cached_response
            cached = await get_cached_response(query)
            if cached:
                logger.info("cache_hit", query=query[:60], cache_type=cached.get("cache_type"))
                return ChatResponse(
                    answer=cached.get("answer", ""),
                    sources=[SourceDoc(**s) for s in cached.get("sources", [])],
                    session_id=session_id,
                    cache_hit=True,
                    cache_type=cached.get("cache_type", "exact"),
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                )
        except Exception as e:
            logger.warning("cache_check_error", error=str(e))

    # ── Graph invocation ──────────────────────────────────────────────────────
    try:
        result = await _run_graph(query, session_id, graph)
    except Exception as e:
        logger.error("graph_invoke_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # ── Build response ────────────────────────────────────────────────────────
    answer = result.get("answer", "")
    if not answer:
        msgs = result.get("messages", [])
        if msgs:
            answer = _extract_text(getattr(msgs[-1], "content", ""))
    if not answer:
        answer = "No answer generated."

    sources = _build_sources(result)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    token_usage = None
    if result.get("token_usage"):
        tu = result["token_usage"]
        token_usage = {
            "prompt_tokens": getattr(tu, "prompt_tokens", 0),
            "completion_tokens": getattr(tu, "completion_tokens", 0),
            "total_tokens": getattr(tu, "total_tokens", 0),
            "estimated_cost_usd": getattr(tu, "estimated_cost_usd", 0.0),
        }

    response = ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
        guard_blocked=result.get("guard_blocked", False),
        guard_reason=result.get("guard_reason", ""),
        latency_ms=latency_ms,
        token_usage=token_usage,
        provider_used=result.get("llm_provider_used", ""),
        node_trace=result.get("node_trace", []),
    )

    # ── Observability ─────────────────────────────────────────────────────────
    record_chat_metrics(
        latency_ms=latency_ms,
        prompt_tokens=token_usage.get("prompt_tokens", 0) if token_usage else 0,
        completion_tokens=token_usage.get("completion_tokens", 0) if token_usage else 0,
        estimated_cost_usd=token_usage.get("estimated_cost_usd", 0.0) if token_usage else 0.0,
        provider=result.get("llm_provider_used", "unknown"),
        success=not result.get("guard_blocked", False),
        docs_returned=len(sources),
    )

    if result.get("guard_blocked"):
        record_guard_block(result.get("guard_reason", "unknown"))

    # ── Write-through cache ───────────────────────────────────────────────────
    if request.use_cache and not result.get("guard_blocked") and answer:
        try:
            from src.cache import cache_response
            asyncio.create_task(cache_response(
                query,
                {
                    "answer": answer,
                    "sources": [s.model_dump() for s in sources],
                },
            ))
        except Exception:
            pass

    return response


# ── Chat: SSE Streaming ──────────────────────────────────────────────────────

@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    graph=Depends(get_graph),
):
    """
    SSE streaming chat. Streams token-by-token via LangGraph astream_events.
    Client reads text/event-stream.
    """
    query = request.query.strip()
    session_id = request.session_id

    async def _event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        inc_active_sessions(1)
        config = {"configurable": {"thread_id": session_id}}
        accumulated_answer = ""
        try:
            async for event in graph.astream_events(
                {
                    "messages": [{"role": "user", "content": query}],
                    "question": query,
                    "session_id": session_id,
                },
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")

                # Stream LLM tokens from generate_answer node
                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node in ("generate_answer", "generate_query_or_respond"):
                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            token = _extract_text(getattr(chunk, "content", ""))
                            if token:
                                accumulated_answer += token
                                yield {
                                    "event": "token",
                                    "data": json.dumps({"token": token, "session_id": session_id}),
                                }

                # Emit node progress events
                elif kind == "on_chain_start":
                    node = event.get("name", "")
                    if node not in ("LangGraph",):
                        yield {
                            "event": "node_start",
                            "data": json.dumps({"node": node}),
                        }

                elif kind == "on_chain_end":
                    node = event.get("name", "")
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("guard_blocked"):
                        yield {
                            "event": "blocked",
                            "data": json.dumps({
                                "reason": output.get("guard_reason", "Blocked by guardrails"),
                            }),
                        }

            yield {
                "event": "done",
                "data": json.dumps({"answer": accumulated_answer, "session_id": session_id}),
            }

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
        finally:
            inc_active_sessions(-1)

    return EventSourceResponse(_event_generator())


# ── Chat: WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/api/chat/ws/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str,
    graph=Depends(get_graph),
):
    """
    WebSocket endpoint for real-time multi-turn chat.
    Client sends: {"query": "..."}
    Server streams: {"event": "token", "data": "..."}
                    {"event": "done", "answer": "...", "sources": [...]}
                    {"event": "error", "detail": "..."}
    """
    await websocket.accept()
    inc_active_sessions(1)
    logger.info("ws_connected", session_id=session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                query = data.get("query", "").strip()
            except json.JSONDecodeError:
                query = raw.strip()

            if not query:
                await websocket.send_json({"event": "error", "detail": "Empty query"})
                continue

            config = {"configurable": {"thread_id": session_id}}
            accumulated = ""

            try:
                async for event in graph.astream_events(
                    {
                        "messages": [{"role": "user", "content": query}],
                        "question": query,
                        "session_id": session_id,
                    },
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event", "")
                    if kind == "on_chat_model_stream":
                        node = event.get("metadata", {}).get("langgraph_node", "")
                        if node in ("generate_answer", "generate_query_or_respond"):
                            chunk = event.get("data", {}).get("chunk")
                            if chunk:
                                token = _extract_text(getattr(chunk, "content", ""))
                                if token:
                                    accumulated += token
                                    await websocket.send_json({"event": "token", "data": token})

                await websocket.send_json({
                    "event": "done",
                    "answer": accumulated,
                    "session_id": session_id,
                })

            except Exception as e:
                logger.error("ws_graph_error", error=str(e))
                await websocket.send_json({"event": "error", "detail": str(e)})

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    finally:
        inc_active_sessions(-1)


# ── Documents ────────────────────────────────────────────────────────────────

async def _index_document_background(file_path: str, job_id: str) -> None:
    """Background task: load, chunk, embed and index a document."""
    try:
        from src.document_loader import load_and_split_async
        from src.vector_store import get_vector_store
        docs = await load_and_split_async(file_path)
        vs = get_vector_store()
        vs.add_documents(docs)
        logger.info("index_complete", job_id=job_id, chunks=len(docs), file=Path(file_path).name)
    except Exception as e:
        logger.error("index_failed", job_id=job_id, error=str(e))


@app.post("/api/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and index a document asynchronously.
    The file is saved immediately; indexing runs as a background task.
    Returns job_id for tracking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in Config.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Extension '{ext}' not allowed. Supported: {Config.allowed_extensions_list}",
        )

    # Size check
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File {size_mb:.1f}MB exceeds {Config.MAX_FILE_SIZE_MB}MB limit",
        )

    # Save file — read async, write via executor (no aiofiles dependency)
    save_path = Path(Config.UPLOAD_DIR) / file.filename
    job_id = str(uuid.uuid4())[:8]

    try:
        content = await file.read()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_path.write_bytes, content)
        logger.info("file_saved", filename=file.filename, size_mb=f"{size_mb:.2f}", job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Kick off background indexing
    background_tasks.add_task(_index_document_background, str(save_path), job_id)
    record_upload(ext.lstrip("."), success=True)

    # Quick synchronous chunk count estimate
    estimated_chunks = max(1, int((size_mb * 1024) / Config.CHUNK_SIZE))

    return UploadResponse(
        filename=file.filename,
        indexed_chunks=estimated_chunks,
        message=f"File accepted. Indexing in background (job_id={job_id}). Actual chunk count logged on completion.",
        job_id=job_id,
    )


@app.get("/api/documents", tags=["Documents"])
async def list_documents():
    """List all documents currently indexed in the knowledge base."""
    try:
        from src.vector_store import get_vector_store
        vs = get_vector_store()
        result = vs._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        files: Dict[str, Dict] = {}
        for meta in metadatas:
            fname = meta.get("filename", "Unknown")
            if fname not in files:
                files[fname] = {
                    "filename":   fname,
                    "format":     meta.get("format", "?"),
                    "chunks":     0,
                    "indexed_at": meta.get("indexed_at", ""),
                    "file_hash":  meta.get("file_hash", ""),
                }
            files[fname]["chunks"] += 1

        return {"documents": list(files.values()), "total_chunks": len(metadatas)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Remove all chunks for a document from the vector store."""
    try:
        from src.vector_store import get_vector_store
        vs = get_vector_store()
        ids_to_delete = vs._collection.get(
            where={"filename": {"$eq": filename}},
        ).get("ids", [])

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

        vs._collection.delete(ids=ids_to_delete)
        logger.info("document_deleted", filename=filename, chunks=len(ids_to_delete))

        # Remove upload file if it exists
        upload_path = Path(Config.UPLOAD_DIR) / filename
        if upload_path.exists():
            upload_path.unlink()

        return {"message": f"Deleted {len(ids_to_delete)} chunks for '{filename}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation ───────────────────────────────────────────────────────────────

@app.post("/api/evaluate", tags=["Evaluation"])
async def evaluate(
    request: EvalRequest,
    graph=Depends(get_graph),
):
    """
    Run RAGAS + retrieval evaluation against a question set.
    Optionally provide ground_truths for context recall.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question required")
    if len(request.questions) > 50:
        raise HTTPException(status_code=400, detail="Max 50 questions per evaluation run")

    try:
        from src.evaluator import golden_dataset_eval

        dataset = [
            {
                "question": q,
                "ground_truth": (request.ground_truths or [""])[i] if request.ground_truths else "",
                "relevant_doc_ids": (request.relevant_doc_ids or [[]])[i] if request.relevant_doc_ids else [],
            }
            for i, q in enumerate(request.questions)
        ]

        results = await golden_dataset_eval(dataset=dataset, graph=graph)
        return {"status": "ok", "results": results}

    except Exception as e:
        logger.error("eval_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


# ── Cache management ─────────────────────────────────────────────────────────

@app.get("/api/cache/stats", tags=["Cache"])
async def cache_stats():
    """Return cache hit/miss statistics."""
    from src.cache import get_cache_stats
    return get_cache_stats()


@app.post("/api/cache/invalidate", tags=["Cache"])
async def invalidate_cache(pattern: str = "cache:*"):
    """Flush cache entries matching pattern (admin only)."""
    from src.cache import invalidate_cache as _invalidate
    deleted = await _invalidate(pattern)
    return {"deleted": deleted, "pattern": pattern}


# ── Config / Admin ───────────────────────────────────────────────────────────

@app.get("/api/config", tags=["Admin"])
async def get_config():
    """Return current configuration (non-sensitive). Admin only."""
    return {
        "app_name":            Config.APP_NAME,
        "version":             Config.APP_VERSION,
        "env":                 Config.APP_ENV,
        "retrieval_strategy":  Config.RETRIEVAL_STRATEGY,
        "top_k":               Config.TOP_K,
        "chunk_strategy":      Config.CHUNK_STRATEGY,
        "chunk_size":          Config.CHUNK_SIZE,
        "reranker_enabled":    Config.RERANKER_ENABLED,
        "hyde_enabled":        Config.HYDE_ENABLED,
        "guardrails_enabled":  Config.GUARDRAIL_ENABLED,
        "pii_detection":       Config.PII_DETECTION_ENABLED,
        "redis_enabled":       Config.REDIS_ENABLED,
        "postgres_enabled":    Config.POSTGRES_ENABLED,
        "langsmith_enabled":   Config.LANGSMITH_ENABLED,
        "mcp_enabled":         Config.MCP_ENABLED,
        "auth_enabled":        Config.AUTH_ENABLED,
        "llm_provider_order":  Config.LLM_PROVIDER_ORDER,
    }


# ── Prometheus metrics ───────────────────────────────────────────────────────

if Config.PROMETHEUS_ENABLED:
    @app.get(Config.PROMETHEUS_METRICS_PATH, tags=["Observability"], include_in_schema=False)
    async def prometheus_metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return StreamingResponse(
            iter([generate_latest()]),
            media_type=CONTENT_TYPE_LATEST,
        )


# ── Auth helpers (dev / staging) ─────────────────────────────────────────────

@app.post("/auth/token", tags=["Auth"])
async def issue_token(request: TokenRequest):
    """
    Issue a JWT token. For development and testing only.
    In production, integrate with your identity provider.
    """
    from api.auth import Role, create_token_pair
    try:
        role = Role(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}. Valid: admin, user, readonly")

    tokens = create_token_pair(request.user_id, role)
    return tokens
