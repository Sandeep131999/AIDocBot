"""
api/main.py — Enterprise FastAPI Application
──────────────────────────────────────────────
Endpoints:
  GET  /                           — root health ping
    GET  /api/health                 — detailed health (PostgreSQL/pgvector, LLM)
  GET  /metrics                    — Prometheus metrics (if enabled)

  POST /api/chat                   — standard JSON chat
  POST /api/chat/stream            — SSE streaming chat

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
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import Config
from api.auth import Role
from api.projects import (
    ProjectContext,
    get_project_context,
    require_project_role,
)
from src.observability.observability import (
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
    graph: Any = None
    supervisor_graph: Any = None
    checkpointer_cleanup: Any = None


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
        from src.agent.graph import build_agentic_rag_graph, create_checkpointer
        checkpointer, _app_state.checkpointer_cleanup = await create_checkpointer()
        _app_state.graph = build_agentic_rag_graph(checkpointer=checkpointer)
        logger.info("graph_ready", type="agentic_rag")

        # Ensure directories exist
        Path(Config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        # Log config (masks secrets)
        if Config.DEBUG:
            Config.print_config()

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    if _app_state.checkpointer_cleanup:
        await _app_state.checkpointer_cleanup()
        _app_state.checkpointer_cleanup = None
    from src.retrieval.vector_store import close_vector_store
    close_vector_store()
    from src.observability.memory import close_memory_pool
    await close_memory_pool()
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
    from api.projects import router as projects_router
    app.include_router(projects_router)
    if Config.OTEL_ENABLED:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

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
    document_id: Optional[str] = None
    label: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


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
    version: int = 1


class EvalRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None
    relevant_doc_ids: Optional[List[List[str]]] = None
    k: int = Field(default=5, ge=1, le=50)
    evaluation_framework: Literal["ragas", "deepeval", "both"] = "both"


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
            document_id=meta.get("document_id"),
            label=meta.get("label"),
            title=meta.get("title"),
            url=meta.get("url"),
        ))
    return sources


async def _run_graph(
    query: str,
    session_id: str,
    graph,
    context: ProjectContext | None = None,
) -> Dict[str, Any]:
    """Invoke the LangGraph graph with proper config and return the state dict."""
    thread_id = session_id
    state: Dict[str, Any] = {
        "messages": [{"role": "user", "content": query}],
        "question": query,
        "session_id": session_id,
    }
    if context:
        thread_id = f"{context.project_id}:{context.user.user_id}:{session_id}"
        state["user_id"] = context.user.user_id
    config = {
        "configurable": {
            "thread_id": thread_id,
            **({"project_id": context.project_id} if context else {}),
        }
    }
    return await graph.ainvoke(
        state,
        config=config,
    )


async def _log_question(
    *,
    context: ProjectContext,
    request: Request,
    question: str,
    answer: str = "",
    status: str,
    error: str = "",
    latency_ms: float = 0,
    provider: str = "",
    cost_usd: float = 0,
) -> None:
    from src.retrieval.vector_store import get_vector_store

    entry = {
        "id": str(uuid.uuid4()),
        "project_id": context.project_id,
        "user_id": context.user.user_id,
        "question": question,
        "answer": answer,
        "status": status,
        "error": error,
        "latency_ms": latency_ms,
        "provider": provider,
        "estimated_cost_usd": cost_usd,
        "request_id": getattr(request.state, "request_id", ""),
    }
    try:
        await asyncio.to_thread(get_vector_store().log_question, entry)
    except Exception as exc:
        logger.error("question_log_write_failed", project_id=context.project_id, error=str(exc))


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
        from src.retrieval.vector_store import get_vector_store
        vs = get_vector_store()
        checks["vector_db"] = {"status": "ok", "chunks": vs.count(), "backend": "postgresql_pgvector"}
    except Exception as e:
        checks["vector_db"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    # Features
    checks["features"] = {
        "guardrails":           Config.GUARDRAIL_ENABLED,
        "pii_detection":        Config.PII_DETECTION_ENABLED,
        "reranker":             Config.RERANKER_ENABLED,
        "hyde":                 Config.HYDE_ENABLED,
        "hybrid_search":        Config.RETRIEVAL_STRATEGY == "hybrid",
        "response_cache":       True,
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
    http_request: Request,
    graph=Depends(get_graph),
    context: ProjectContext = Depends(get_project_context),
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
            from src.retrieval.cache import get_cached_response
            cached = await get_cached_response(query, project_id=context.project_id)
            if cached:
                logger.info("cache_hit", query=query[:60], cache_type=cached.get("cache_type"))
                latency_ms = round((time.perf_counter() - start) * 1000, 1)
                await _log_question(
                    context=context,
                    request=http_request,
                    question=query,
                    answer=cached.get("answer", ""),
                    status="success",
                    latency_ms=latency_ms,
                )
                return ChatResponse(
                    answer=cached.get("answer", ""),
                    sources=[SourceDoc(**s) for s in cached.get("sources", [])],
                    session_id=session_id,
                    cache_hit=True,
                    cache_type=cached.get("cache_type", "exact"),
                    latency_ms=latency_ms,
                )
        except Exception as e:
            logger.warning("cache_check_error", error=str(e))

    # ── Graph invocation ──────────────────────────────────────────────────────
    try:
        from src.observability.observability import observe
        async with observe(
            "rag.chat",
            {
                "project_id": context.project_id,
                "user_id": context.user.user_id,
                "request_id": getattr(http_request.state, "request_id", ""),
            },
        ):
            result = await _run_graph(query, session_id, graph, context)
    except Exception as e:
        logger.error("graph_invoke_error", error=str(e))
        await _log_question(
            context=context,
            request=http_request,
            question=query,
            status="error",
            error=str(e),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
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

    await _log_question(
        context=context,
        request=http_request,
        question=query,
        answer=answer,
        status="blocked" if result.get("guard_blocked") else "success",
        latency_ms=latency_ms,
        provider=result.get("llm_provider_used", ""),
        cost_usd=token_usage.get("estimated_cost_usd", 0.0) if token_usage else 0.0,
    )

    # ── Write-through cache ───────────────────────────────────────────────────
    if request.use_cache and not result.get("guard_blocked") and answer:
        try:
            from src.retrieval.cache import cache_response
            asyncio.create_task(cache_response(
                query,
                {
                    "answer": answer,
                    "sources": [s.model_dump() for s in sources],
                },
                project_id=context.project_id,
            ))
        except Exception:
            pass

    return response


# ── Chat: SSE Streaming ──────────────────────────────────────────────────────

@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    graph=Depends(get_graph),
    context: ProjectContext = Depends(get_project_context),
):
    """
    SSE streaming chat. Streams token-by-token via LangGraph astream_events.
    Client reads text/event-stream.
    """
    query = request.query.strip()
    session_id = request.session_id

    async def _event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        inc_active_sessions(1)
        thread_id = f"{context.project_id}:{context.user.user_id}:{session_id}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "project_id": context.project_id,
            }
        }
        accumulated_answer = ""
        try:
            async for event in graph.astream_events(
                {
                    "messages": [{"role": "user", "content": query}],
                    "question": query,
                    "session_id": session_id,
                    "user_id": context.user.user_id,
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
            await _log_question(
                context=context,
                request=http_request,
                question=query,
                answer=accumulated_answer,
                status="success",
            )

        except Exception as e:
            logger.error("stream_error", error=str(e))
            await _log_question(
                context=context,
                request=http_request,
                question=query,
                status="error",
                error=str(e),
            )
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
        finally:
            inc_active_sessions(-1)

    return EventSourceResponse(_event_generator())


# ── Documents ────────────────────────────────────────────────────────────────

async def _index_document_background(
    file_path: str,
    job_id: str,
    project_id: str,
    user_id: str,
    filename: str,
    file_hash: str,
    version: int,
    storage_key: str,
) -> None:
    """Background task: load, chunk, embed and index a document."""
    try:
        from src.retrieval.document_loader import load_and_split_async
        from src.retrieval.vector_store import index_documents
        docs = await load_and_split_async(
            file_path,
            metadata_overrides={
                "filename": filename,
                "project_id": project_id,
                "uploaded_by": user_id,
                "file_hash": file_hash,
                "document_id": file_hash,
                "version": version,
                "active_version": False,
                "storage_key": storage_key,
            },
        )
        indexed_ids = index_documents(docs)
        from src.retrieval.vector_store import get_vector_store
        get_vector_store().activate_document_version(project_id, filename, file_hash)
        logger.info(
            "index_complete",
            job_id=job_id,
            chunks=len(indexed_ids),
            file=filename,
            project_id=project_id,
            version=version,
        )
    except Exception as e:
        logger.error("index_failed", job_id=job_id, project_id=project_id, error=str(e))
        try:
            from src.retrieval.vector_store import get_vector_store
            get_vector_store().fail_document_version(project_id, filename, file_hash)
        except Exception as storage_error:
            logger.error("document_version_status_failed", job_id=job_id, error=str(storage_error))


@app.post("/api/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    context: ProjectContext = Depends(
        require_project_role(Role.ADMIN, Role.USER)
    ),
):
    """
    Upload and index a document asynchronously.
    The file is saved immediately; indexing runs as a background task.
    Returns job_id for tracking.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = Path(file.filename).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = Path(filename).suffix.lower()
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
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        file_hash = hashlib.sha256(content).hexdigest()
        from src.retrieval.vector_store import get_vector_store

        job_id = str(uuid.uuid4())
        storage_name = f"{uuid.uuid4()}{ext}"
        project_dir = Path(Config.UPLOAD_DIR) / context.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        save_path = project_dir / storage_name
        storage_key = f"{context.project_id}/{storage_name}"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_path.write_bytes, content)
        version = get_vector_store().create_document_version(
            context.project_id, filename, file_hash, context.user.user_id
        )
        logger.info(
            "file_saved",
            filename=filename,
            project_id=context.project_id,
            size_mb=f"{size_mb:.2f}",
            job_id=job_id,
        )
    except HTTPException:
        if "save_path" in locals() and save_path.exists():
            save_path.unlink()
        raise
    except Exception as e:
        if "save_path" in locals() and save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save or register file: {e}")

    # Kick off background indexing
    background_tasks.add_task(
        _index_document_background,
        str(save_path),
        job_id,
        context.project_id,
        context.user.user_id,
        filename,
        file_hash,
        version,
        storage_key,
    )
    record_upload(ext.lstrip("."), success=True)

    # Quick synchronous chunk count estimate
    estimated_chunks = max(1, int((size_mb * 1024) / Config.CHUNK_SIZE))

    return UploadResponse(
        filename=filename,
        indexed_chunks=estimated_chunks,
        message=f"File accepted. Indexing in background (job_id={job_id}). Actual chunk count logged on completion.",
        job_id=job_id,
        file_hash=file_hash,
        version=version,
    )


@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    context: ProjectContext = Depends(get_project_context),
):
    """List all documents currently indexed in the knowledge base."""
    try:
        from src.retrieval.vector_store import get_vector_store
        vs = get_vector_store()
        indexed_documents = vs.list_documents(project_id=context.project_id, active_only=True)
        metadatas = [doc.metadata for doc in indexed_documents]

        files: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            fname = str(meta.get("filename") or "Unknown")
            if fname not in files:
                files[fname] = {
                    "filename":   fname,
                    "format":     meta.get("format", "?"),
                    "chunks":     0,
                    "indexed_at": meta.get("indexed_at", ""),
                    "file_hash":  meta.get("file_hash", ""),
                    "version":    meta.get("version", 1),
                    "project_id": context.project_id,
                    "status":     "indexed",
                }
            files[fname]["chunks"] += 1

        for version in vs.list_project_document_versions(context.project_id):
            filename = version["filename"]
            entry = files.setdefault(filename, {
                "filename": filename,
                "format": Path(filename).suffix.lower().lstrip(".") or "?",
                "chunks": 0,
                "indexed_at": version["created_at"],
                "file_hash": version["file_hash"],
                "project_id": context.project_id,
            })
            entry.update({
                "version": version["version"],
                "status": version["status"],
                "file_hash": version["file_hash"],
            })

        return {"documents": list(files.values()), "total_chunks": len(metadatas)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}", tags=["Documents"])
async def delete_document(
    filename: str,
    context: ProjectContext = Depends(require_project_role(Role.ADMIN)),
):
    """Remove all chunks for a document from the vector store."""
    try:
        from src.retrieval.vector_store import get_vector_store
        vs = get_vector_store()
        version_documents = vs.list_documents(filename=filename, project_id=context.project_id)
        storage_keys = {
            doc.metadata.get("storage_key")
            for doc in version_documents
            if doc.metadata.get("storage_key")
        }
        deleted_count = vs.delete(filename=filename, project_id=context.project_id)

        if not deleted_count:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        vs.delete_document_versions(context.project_id, filename)

        logger.info("document_deleted", filename=filename, chunks=deleted_count)

        # Remove upload file if it exists
        for storage_key in storage_keys:
            upload_path = Path(Config.UPLOAD_DIR) / storage_key
            if upload_path.is_file():
                upload_path.unlink()

        return {"message": f"Deleted {deleted_count} chunks for '{filename}'"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{filename}/versions", tags=["Documents"])
async def document_versions(
    filename: str,
    context: ProjectContext = Depends(get_project_context),
):
    return {
        "filename": filename,
        "versions": get_vector_store().document_version_history(context.project_id, filename),
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

@app.post("/api/evaluate", tags=["Evaluation"])
async def evaluate(
    request: EvalRequest,
    graph=Depends(get_graph),
    context: ProjectContext = Depends(require_project_role(Role.ADMIN)),
):
    """
    Run RAGAS + retrieval evaluation against a question set.
    Optionally provide ground_truths for context recall.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question required")
    if len(request.questions) > 50:
        raise HTTPException(status_code=400, detail="Max 50 questions per evaluation run")
    if request.ground_truths is not None and len(request.ground_truths) != len(request.questions):
        raise HTTPException(status_code=400, detail="ground_truths must match the number of questions")
    if request.relevant_doc_ids is not None and len(request.relevant_doc_ids) != len(request.questions):
        raise HTTPException(status_code=400, detail="relevant_doc_ids must match the number of questions")

    try:
        from src.observability.evaluator import golden_dataset_eval

        dataset = []
        for index, question in enumerate(request.questions):
            dataset.append({
                "question": question,
                "ground_truth": request.ground_truths[index] if request.ground_truths else "",
                "relevant_doc_ids": request.relevant_doc_ids[index] if request.relevant_doc_ids else [],
            })

        thread_prefix = f"eval-{context.project_id}-{context.user.user_id}"
        results = await golden_dataset_eval(
            dataset=dataset,
            graph=graph,
            k=request.k,
            evaluation_framework=request.evaluation_framework,
            thread_prefix=thread_prefix,
        )
        hit_at_k = results.get("retrieval", {}).get("hit_at_k", {}).get("mean")
        faithfulness = results.get("ragas", {}).get("faithfulness")
        if faithfulness is None:
            faithfulness = results.get("deepeval", {}).get("FaithfulnessMetric")
        checks = {
            "hit_at_k": {
                "value": hit_at_k,
                "minimum": Config.RELEASE_MIN_HIT_AT_K,
                "passed": hit_at_k is not None and hit_at_k >= Config.RELEASE_MIN_HIT_AT_K,
            },
            "faithfulness": {
                "value": faithfulness,
                "minimum": Config.RELEASE_MIN_FAITHFULNESS,
                "passed": (
                    faithfulness is not None
                    and faithfulness >= Config.RELEASE_MIN_FAITHFULNESS
                ),
            },
        }
        release_passed = all(check["passed"] for check in checks.values())
        run_id = str(uuid.uuid4())
        results["release_gate"] = {"passed": release_passed, "checks": checks}
        from src.retrieval.vector_store import get_vector_store
        await asyncio.to_thread(
            get_vector_store().save_evaluation,
            run_id,
            context.project_id,
            context.user.user_id,
            results,
            release_passed,
        )
        return {
            "status": "ok",
            "run_id": run_id,
            "release_gate": results["release_gate"],
            "results": results,
        }

    except Exception as e:
        logger.error("eval_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@app.get("/api/evaluations", tags=["Evaluation"])
async def list_evaluations(
    limit: int = 50,
    context: ProjectContext = Depends(require_project_role(Role.ADMIN)),
):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    from src.retrieval.vector_store import get_vector_store
    return {
        "evaluations": get_vector_store().list_evaluations(context.project_id, limit),
        "project_id": context.project_id,
    }


# ── Cache management ─────────────────────────────────────────────────────────

@app.get("/api/cache/stats", tags=["Cache"])
async def cache_stats(
    context: ProjectContext = Depends(get_project_context),
):
    """Return cache hit/miss statistics."""
    from src.retrieval.cache import get_cache_stats
    return get_cache_stats(project_id=context.project_id)


@app.post("/api/cache/invalidate", tags=["Cache"])
async def invalidate_cache(
    pattern: str = "cache:*",
    context: ProjectContext = Depends(require_project_role(Role.ADMIN)),
):
    """Flush cache entries matching pattern (admin only)."""
    from src.retrieval.cache import invalidate_cache as _invalidate
    deleted = await _invalidate(pattern, project_id=context.project_id)
    return {"deleted": deleted, "pattern": pattern}


# ── Config / Admin ───────────────────────────────────────────────────────────

@app.get("/api/config", tags=["Admin"])
async def get_config(
    context: ProjectContext = Depends(require_project_role(Role.ADMIN)),
):
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
    if Config.AUTH_ENABLED or Config.APP_ENV == "production":
        raise HTTPException(status_code=404, detail="Not found")
    from api.auth import Role, create_token_pair
    try:
        role = Role(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}. Valid: admin, user, readonly")

    tokens = create_token_pair(request.user_id, role)
    return tokens
