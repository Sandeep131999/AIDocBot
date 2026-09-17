from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from src.document_loader import load_and_split
from src.vector_store import get_vector_store
from src.graph import build_agentic_rag_graph
from src.config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Lifespan instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app.state.graph = build_agentic_rag_graph()
        Path(Config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        logger.info("Agentic RAG graph compiled")
    except Exception as e:
        logger.error(f"Failed to init graph: {e}", exc_info=True)
        raise
    yield
    # Shutdown
    logger.info("Shutting down")

app = FastAPI(
    title=Config.APP_NAME,
    version="1.0.0",
    description="LangChain + LangGraph Agentic RAG with Guardrails + Ollama Local",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    top_k: int = Config.TOP_K

class ChatResponse(BaseModel):
    answer: str
    sources: list
    guard_blocked: bool = False
    guard_reason: str = ""

class UploadResponse(BaseModel):
    filename: str
    indexed_chunks: int
    message: str

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": Config.APP_NAME,
        "graph_ready": hasattr(app.state, "graph")
    }

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension {ext} not allowed. Allowed: {Config.ALLOWED_EXTENSIONS}"
        )

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024*1024)
    file.file.seek(0)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File {size_mb:.1f}MB exceeds {Config.MAX_FILE_SIZE_MB}MB limit"
        )

    path = Path(Config.UPLOAD_DIR) / file.filename
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved upload: {path}")

        docs = load_and_split(str(path))
        if not docs:
            raise HTTPException(status_code=422, detail="No content extracted from file")

        vector_store = get_vector_store()
        vector_store.add_documents(docs)
        logger.info(f"Indexed {len(docs)} chunks from {file.filename}")

        return UploadResponse(
            filename=file.filename,
            indexed_chunks=len(docs),
            message=f"Successfully indexed {len(docs)} chunks"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        if path.exists():
            path.unlink()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


def _extract_text_from_content(content) -> str:
    """Gemini 2.5 returns [{'type':'text','text':...}] - handle both str and list"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                # {'type': 'text', 'text': '...'}
                if "text" in part:
                    texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
            else:
                # LangChain content block object
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)
                else:
                    texts.append(str(part))
        return "\n".join(texts)
    return str(content)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not hasattr(app.state, "graph"):
        raise HTTPException(status_code=503, detail="Graph not initialized")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) > Config.GUARDRAIL_MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail="Query too long")

    try:
        result = await app.state.graph.ainvoke({
            "messages": [{"role": "user", "content": query}],
            "question": query,
            "guard_blocked": False
        })

        last_msg = result.get("messages", [])[-1] if result.get("messages") else None
        raw_content = getattr(last_msg, "content", "") if last_msg else ""
        
        # FIX HERE - handle list content from Gemini 2.5
        answer = _extract_text_from_content(raw_content)
        if not answer:
            answer = "No answer generated"

        # Safe context extraction
        context_val = result.get("context", "")
        sources = []
        if context_val:
            txt = _extract_text_from_content(context_val)
            sources = [txt[:500]] if txt else []

        return ChatResponse(
            answer=answer,  # Now always string
            sources=sources,
            guard_blocked=result.get("guard_blocked", False),
            guard_reason=result.get("guard_reason", "")
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    if not hasattr(app.state, "graph"):
        raise HTTPException(status_code=503, detail="Graph not initialized")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(query) > Config.GUARDRAIL_MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail="Query too long")

    try:
        result = await app.state.graph.ainvoke({
            "messages": [{"role": "user", "content": query}],
            "question": query,
            "guard_blocked": False
        })

        last_msg = result.get("messages", [])[-1] if result.get("messages") else None
        answer = getattr(last_msg, "content", "") if last_msg else "No answer generated"

        return ChatResponse(
            answer=answer,
            sources=[result.get("context","")[:500]] if result.get("context") else [],
            guard_blocked=result.get("guard_blocked", False),
            guard_reason=result.get("guard_reason", "")
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/api/health")
async def health():
    vector_store = get_vector_store()
    count = 0
    try:
        count = vector_store._collection.count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "vector_docs": count,
        "ollama_enabled": "ollama" in Config.LLM_PROVIDER_ORDER,
        "guardrails": Config.GUARDRAIL_ENABLED
    }