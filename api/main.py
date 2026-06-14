"""
MAIN.PY - RAG CHATBOT v3.4 (NO AUTHENTICATION, NO CHAT HISTORY)
===============================================================

Features:
- Document upload, list, and delete
- Hybrid search (vector + BM25)
- LLM integration with SMART FALLBACK (Gemini -> Groq -> OpenRouter)
- Automatic fallback on ALL provider errors WITH TIMEOUT
- PERMANENT provider disabling on API key leaks (403)
- LangChain for LLM interactions
- Rotating file logs with 15-day retention
- Separate error.log with full traces
- NO chat history / NO database / NO sessions
"""

# ============================================================================
# MONKEY-PATCH: Fix httpx proxies conflict with older LangChain
# ============================================================================
import httpx
_original_client_init = httpx.Client.__init__
def _patched_client_init(self, *args, **kwargs):
    kwargs.pop('proxies', None)
    return _original_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client_init

# ============================================================================
# IMPORTS
# ============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
import logging
import traceback
from logging.handlers import RotatingFileHandler
import glob
import shutil
import signal
from contextlib import contextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever
from src.evaluator import Evaluator

# LangChain imports - UPDATED to modern packages
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Environment
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# WINDOWS CONSOLE ENCODING FIX
# ============================================================================
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# TIMEOUT UTILS: Force fail-fast on Gemini 429 hangs
# ============================================================================

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutException after N seconds."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")

    # Windows doesn't support signal.SIGALRM, so we use a different approach
    if sys.platform == "win32":
        import threading
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutException(f"Timed out after {seconds} seconds")))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


def invoke_with_timeout(llm, messages, timeout_seconds=10):
    """Invoke LLM with a hard timeout. Returns response or raises TimeoutException."""
    import threading
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, messages)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutException(f"LLM call timed out after {timeout_seconds}s")


# ============================================================================
# LOG MANAGER: Rotating files + 15-day retention + separate error log
# ============================================================================

class LogManager:
    """Manages rotating log files with 15-day retention."""

    def __init__(self, log_dir="logs", retention_days=15):
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.log_dir.mkdir(exist_ok=True)
        self._cleanup_old_logs()

        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

        general_log_path = self.log_dir / "rag_chatbot.log"
        general_handler = RotatingFileHandler(
            general_log_path, maxBytes=10*1024*1024, backupCount=10, encoding='utf-8'
        )
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(general_handler)

        error_log_path = self.log_dir / "rag_chatbot_errors.log"
        error_handler = RotatingFileHandler(
            error_log_path, maxBytes=10*1024*1024, backupCount=10, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LogManager initialized. Logs: {self.log_dir}")
        self.logger.info(f"Retention: {retention_days} days")

    def _cleanup_old_logs(self):
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        log_files = glob.glob(str(self.log_dir / "*.log*"))
        deleted = 0
        for f in log_files:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(f))
                if mtime < cutoff:
                    os.remove(f)
                    deleted += 1
            except Exception:
                pass
        if deleted:
            print(f"Cleaned up {deleted} old log files (>{self.retention_days} days)")


# Initialize logging FIRST
log_manager = LogManager(log_dir="logs", retention_days=15)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".txt", ".csv", ".xls", ".xlsx", ".json", ".pdf", ".docx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory document tracking (no database)
_uploaded_documents = []
_document_counter = 0


def _get_next_doc_id():
    global _document_counter
    _document_counter += 1
    return _document_counter


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request — just query and top_k."""
    query: str
    top_k: int = int(os.getenv("TOP_K", "5"))


class ChatResponse(BaseModel):
    """Chat response — no session, no history."""
    assistant_message: str
    sources: list
    used_model: str


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    id: int
    filename: str
    size: int
    indexed: bool


class DocumentListResponse(BaseModel):
    """Document list item."""
    id: int
    filename: str
    size: int
    indexed: bool
    uploaded_at: str


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RAG Chatbot",
    description="RAG system with document management, hybrid search, and smart LLM fallback. No chat history.",
    version="3.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
retriever = None
llm_handler = None
evaluator = Evaluator()


# ============================================================================
# LLM WITH SMART FALLBACK (v3.4) — WITH TIMEOUT
# ============================================================================

class LLMWithSmartFallback:
    """
    LLM with SMART fallback + TIMEOUT:
    Fallback chain: Google Gemini -> Groq -> OpenRouter
    Each provider gets 10 seconds max before fallback kicks in.
    """

    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
    DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
    DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"

    CONTEXT_LIMITS = {
        "gemini-2.0-flash": 1_000_000,
        "gemini-2.5-flash": 1_000_000,
        "llama-3.1-70b-versatile": 8192,
        "meta-llama/llama-3.1-8b-instruct": 8192,
    }

    # Timeout per provider attempt (seconds)
    PROVIDER_TIMEOUT = 10

    def __init__(self):
        self.available_llms = []
        self.failed_providers = {}
        self.current_index = 0
        self.used_model = "unknown"
        self._initialize_all_llms()

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int, reserved: int = 1500) -> str:
        max_context_tokens = max_tokens - reserved
        max_context_chars = max_context_tokens * 4
        if len(context) > max_context_chars:
            truncated = context[:max_context_chars]
            last_period = truncated.rfind(". ")
            if last_period > max_context_chars * 0.8:
                truncated = truncated[:last_period + 1]
            return truncated + "\n\n[Content truncated due to length...]"
        return context

    def _classify_error(self, error: Exception) -> tuple:
        error_str = str(error).lower()

        key_leak_patterns = [
            "api_key_leaked", "leaked", "invalid api key", "authentication failed",
            "auth failed", "unauthorized", "401", "403",
        ]
        if any(p in error_str for p in key_leak_patterns):
            return ("api_key_leaked", "fallback", True)

        rate_limit_patterns = [
            "rate limit", "too many requests", "429", "quota exceeded", "quota", "limit exceeded",
        ]
        if any(p in error_str for p in rate_limit_patterns):
            return ("rate_limit", "fallback", False)

        token_error_patterns = [
            "too many tokens", "context length", "context window", "request payload size",
            "request too large", "max tokens", "max_output_tokens", "exceeds the limit", "token limit", "413",
        ]
        if any(p in error_str for p in token_error_patterns):
            return ("token_limit", "fallback", False)

        network_patterns = [
            "connection error", "timeout", "502", "503", "504", "bad gateway",
            "service unavailable", "gateway timeout", "network", "unable to connect", "connection refused",
        ]
        if any(p in error_str for p in network_patterns):
            return ("network_error", "fallback", False)

        # Treat timeout as network error (temporary)
        if "timed out" in error_str or "timeout" in error_str:
            return ("timeout", "fallback", False)

        return ("unknown_error", "fallback", False)

    def _initialize_all_llms(self):
        logger.info("=" * 70)
        logger.info("INITIALIZING ALL LLM PROVIDERS")
        logger.info("=" * 70)

        # [1] Google Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                logger.info("[1] Initializing Google Gemini...")
                model_name = os.getenv("GEMINI_MODEL", self.DEFAULT_GEMINI_MODEL)
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=gemini_key,
                    temperature=0.7,
                    max_output_tokens=2048,
                    timeout=5,  # Request-level timeout (may not work perfectly)
                    max_retries=0,  # Disable internal retries - we handle fallback ourselves
                )
                self.available_llms.append({
                    "name": model_name, "llm": llm, "type": "gemini", "display_name": "Google Gemini"
                })
                logger.info(f"[OK] Google Gemini ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] Google Gemini init failed: {e}")
                self.failed_providers["Google Gemini"] = f"Init failed: {str(e)}"

        # [2] Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                logger.info("[2] Initializing Groq...")
                model_name = os.getenv("GROQ_MODEL", self.DEFAULT_GROQ_MODEL)
                llm = ChatOpenAI(
                    model=model_name, api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.7, max_tokens=1024,
                    timeout=10, max_retries=1
                )
                self.available_llms.append({
                    "name": model_name, "llm": llm, "type": "groq", "display_name": "Groq"
                })
                logger.info(f"[OK] Groq ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] Groq init failed: {e}")
                self.failed_providers["Groq"] = f"Init failed: {str(e)}"

        # [3] OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                logger.info("[3] Initializing OpenRouter...")
                model_name = os.getenv("OPENROUTER_MODEL", self.DEFAULT_OPENROUTER_MODEL)
                llm = ChatOpenAI(
                    model=model_name, api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.7, max_tokens=1024,
                    timeout=10, max_retries=1
                )
                self.available_llms.append({
                    "name": model_name, "llm": llm, "type": "openrouter", "display_name": "OpenRouter"
                })
                logger.info(f"[OK] OpenRouter ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] OpenRouter init failed: {e}")
                self.failed_providers["OpenRouter"] = f"Init failed: {str(e)}"

        if not self.available_llms:
            logger.warning("[WARN] No LLM providers available at startup.")

        if self.available_llms:
            self.current_index = 0
            self.used_model = self.available_llms[0]["name"]
            logger.info(f"[PRIMARY] Primary LLM: {self.used_model}")
            logger.info(f"[FALLBACK] Fallback chain: {' -> '.join([llm['name'] for llm in self.available_llms])}")
        else:
            self.used_model = "none"
            logger.warning("[WARN] No LLM providers configured")
        logger.info("=" * 70)

    def _build_messages(self, context: str, question: str) -> list:
        """Build messages WITHOUT chat history."""
        messages = []

        raw_prompt = os.getenv("SYSTEM_PROMPT", """You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

Answer the user's question based on the context above.""")

        system_prompt = raw_prompt.format(context=context)
        messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=question))
        return messages

    def generate_response(self, context: str, question: str,
                         attempt: int = 0, max_attempts: int = 3) -> tuple:
        """Generate response with SMART automatic fallback + TIMEOUT."""

        if not self.available_llms:
            logger.warning("[WARN] No LLM providers available.")
            return "I apologize, but no AI providers are currently configured. Please check your API keys.", "none"

        if attempt >= len(self.available_llms) or attempt >= max_attempts:
            logger.error("All LLM providers exhausted")
            raise HTTPException(
                status_code=503,
                detail="All LLM providers failed. Service temporarily unavailable."
            )

        current_llm_config = self.available_llms[attempt]
        current_llm = current_llm_config["llm"]
        provider_name = current_llm_config["display_name"]
        self.used_model = current_llm_config["name"]

        logger.info(f"[LLM ATTEMPT {attempt + 1}/{len(self.available_llms)}] Provider: {provider_name}")

        model_limit = self.CONTEXT_LIMITS.get(self.used_model, 8192)
        safe_context = self._truncate_context(context, model_limit, reserved=1500)
        messages = self._build_messages(safe_context, question)

        total_text = " ".join([m.content for m in messages])
        estimated_tokens = self._estimate_tokens(total_text)
        logger.info(f"[TOK] Estimated tokens: {estimated_tokens} (limit: {model_limit})")

        try:
            # CRITICAL FIX: Use invoke_with_timeout to prevent hanging on 429
            logger.info(f"[CALL] Invoking {provider_name} with {self.PROVIDER_TIMEOUT}s timeout...")
            response = invoke_with_timeout(current_llm, messages, timeout_seconds=self.PROVIDER_TIMEOUT)
            logger.info(f"[OK] Success with {provider_name} ({self.used_model})")
            return response.content, self.used_model

        except TimeoutException as e:
            error_msg = str(e)
            logger.error(f"[ERR] {provider_name} TIMED OUT: {error_msg}")
            # Treat timeout as temporary failure -> fallback immediately
            logger.warning(f"[FALLBACK] {provider_name}: timeout - trying next provider")
            return self.generate_response(context=context, question=question, attempt=attempt + 1, max_attempts=max_attempts)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERR] {provider_name} failed: {error_msg}")
            logger.error(traceback.format_exc())

            error_type, action, permanently_disable = self._classify_error(e)
            logger.warning(f"[CHECK] Error classified as: {error_type} (action={action}, permanent_disable={permanently_disable})")

            if permanently_disable:
                self.failed_providers[provider_name] = f"{error_type}: {error_msg}"
                self.available_llms.pop(attempt)
                logger.warning(f"[WARN] {provider_name}: {error_type} - PERMANENTLY DISABLED")

                if self.available_llms:
                    return self.generate_response(context=context, question=question, attempt=attempt, max_attempts=max_attempts)
                else:
                    logger.error("All LLM providers permanently disabled")
                    return f"All AI providers are currently unavailable. Last error: {error_msg}", "none"

            elif action == "fallback":
                logger.warning(f"[FALLBACK] {provider_name}: {error_type} - trying next provider")
                return self.generate_response(context=context, question=question, attempt=attempt + 1, max_attempts=max_attempts)

            else:
                raise HTTPException(status_code=502, detail=f"LLM provider error ({provider_name}): {error_msg}")


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global retriever, llm_handler

    logger.info("=" * 70)
    logger.info("STARTING RAG CHATBOT v3.4 (NO HISTORY + TIMEOUT)")
    logger.info("=" * 70)

    try:
        retriever = Retriever(
            vector_weight=float(os.getenv("RETRIEVER_VECTOR_WEIGHT", "0.7")),
            keyword_weight=float(os.getenv("RETRIEVER_KEYWORD_WEIGHT", "0.3")),
            min_relevance_score=float(os.getenv("RETRIEVER_MIN_SCORE", "0.35"))
        )
        logger.info("[OK] Retriever initialized")

        # Try to load existing documents
        try:
            base_dir = Path(__file__).parent.parent
            default_doc_path = base_dir / "data" / "password_guide.txt"
            if default_doc_path.exists():
                num_chunks = retriever.load_and_index_documents(str(default_doc_path))
                logger.info(f"[OK] Loaded {num_chunks} chunks from default documents")
            else:
                logger.warning(f"[WARN] Default document not found at {default_doc_path}. Users can upload documents.")
        except Exception as e:
            logger.error(f"[ERR] Error loading default documents: {e}")
            logger.error(traceback.format_exc())

        llm_handler = LLMWithSmartFallback()

    except Exception as e:
        logger.error(f"[ERR] Error during startup: {e}")
        logger.error(traceback.format_exc())
        logger.warning("[WARN] Startup completed with errors. Some features may be unavailable.")


# ============================================================================
# DOCUMENT ENDPOINTS
# ============================================================================

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    logger.info(f"[UPLOAD] File: {file.filename}")
    logger.info("-" * 70)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)
        logger.info(f"[OK] File saved: {file_path} ({file_size} bytes)")

        doc_id = _get_next_doc_id()
        doc_record = {
            "id": doc_id,
            "filename": file.filename,
            "file_path": file_path,
            "size": file_size,
            "indexed": False,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        _uploaded_documents.append(doc_record)

        if retriever:
            try:
                num_chunks = retriever.load_and_index_documents(file_path)
                doc_record["indexed"] = True
                logger.info(f"[OK] Indexed {num_chunks} chunks")
                return DocumentUploadResponse(id=doc_id, filename=file.filename, size=file_size, indexed=True)
            except Exception as e:
                logger.error(f"[ERR] Indexing error: {e}")
                logger.error(traceback.format_exc())
                return DocumentUploadResponse(id=doc_id, filename=file.filename, size=file_size, indexed=False)
        else:
            raise HTTPException(status_code=500, detail="Retriever not initialized")

    except Exception as e:
        logger.error(f"[ERR] Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/api/documents", tags=["Documents"])
async def list_documents():
    """Get all uploaded documents."""
    return {
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "size": doc["size"],
                "indexed": doc["indexed"],
                "uploaded_at": doc["uploaded_at"]
            }
            for doc in _uploaded_documents
        ]
    }


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: int):
    """Delete a document by ID (file + tracking)."""
    global _uploaded_documents

    doc = next((d for d in _uploaded_documents if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with id {doc_id} not found")

    file_path = doc["file_path"]

    # Delete file from disk
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[OK] Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"[ERR] Failed to delete file {file_path}: {e}")

    # Remove from tracking list
    _uploaded_documents = [d for d in _uploaded_documents if d["id"] != doc_id]

    logger.info(f"[OK] Deleted document id={doc_id} from tracking")
    return {"success": True, "message": f"Document {doc_id} deleted", "deleted_file": file_path}


# ============================================================================
# CHAT ENDPOINT (NO HISTORY) — WITH QUERY REWRITE + RE-RANK FALLBACK
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with RAG system — NO history, NO sessions, NO database.
    Query rewriting and re-ranking use multi-LLM fallback (Gemini -> Groq -> OpenRouter).

    Request: { "query": "your question", "top_k": 5 }
    """
    logger.info(f"[CHAT] Query: {request.query}")
    logger.info("-" * 70)

    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    # FIX: Check if any documents are indexed before retrieving
    if not retriever.indexed:
        logger.warning("[WARN] No documents indexed. Cannot retrieve.")
        return ChatResponse(
            assistant_message="No documents have been uploaded yet. Please upload a document first using POST /api/documents/upload",
            sources=[],
            used_model="none"
        )

    try:
        # Step 1: Retrieve with query rewrite + re-ranking
        logger.info("[STEP 1] Retrieve documents")
        results = retriever.retrieve(
            request.query,
            top_k=request.top_k,
            use_query_rewrite=True,    # ENABLED: Uses Multi-LLM fallback internally
            use_rerank=True,           # ENABLED: Uses Multi-LLM fallback internally
            use_hyde=False
        )

        # Fallback: expand search if no results
        if not results:
            logger.warning("[WARN] No relevant documents found above threshold (0.35)")
            logger.info("[FALLBACK] Trying with expanded top_k...")
            try:
                results = retriever.retrieve(
                    request.query,
                    top_k=request.top_k * 2,
                    use_query_rewrite=True,
                    use_rerank=True
                )
            except Exception as e:
                logger.error(f"[ERR] Expanded retrieve failed: {e}")
                results = []

            if not results:
                logger.warning("[WARN] Still no results after expanding search")
                return ChatResponse(
                    assistant_message="I don't have information about that in my knowledge base.",
                    sources=[],
                    used_model="none"
                )

        # Build context
        context = "\n\n---\n\n".join([r['document'] for r in results])

        sources = []
        for i, r in enumerate(results):
            source = {
                "document": r['document'][:100] + "...",
                "combined_score": r.get('combined_score', 0),
                "vector_score": r.get('vector_score', 0),
                "keyword_score": r.get('keyword_score', 0),
                "final_score": r.get('final_score', r.get('combined_score', 0)),
                "llm_score": r.get('llm_score'),
                "explanation": r.get('explanation'),
                "llm_provider": r.get('llm_provider'),
                "employee_name": r['metadata'].get('employee_name', 'N/A')
            }
            if i == 0 and 'query_rewrite' in r:
                source["query_rewrite"] = r['query_rewrite']
            sources.append(source)

        logger.info(f"[OK] Retrieved {len(results)} documents")

        # Step 2: Generate LLM response (NO chat history) with timeout fallback
        logger.info("[STEP 2] Generate LLM response")

        if llm_handler is None:
            logger.error("[ERR] LLM handler not initialized")
            response_text = "System error: AI handler not initialized."
            used_model = "none"
        else:
            response_text, used_model = llm_handler.generate_response(
                context=context,
                question=request.query
                # NO chat_history parameter — stateless
            )

        logger.info(f"[OK] Generated response using {used_model}")

        return ChatResponse(
            assistant_message=response_text,
            sources=sources,
            used_model=used_model
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERR] Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/health", tags=["System"])
async def health():
    """Health check with provider status."""
    return {
        "status": "healthy",
        "version": "3.4.0",
        "retriever_ready": retriever is not None,
        "documents_indexed": retriever.indexed if retriever else False,
        "llm_ready": llm_handler is not None,
        "llm": {
            "providers_available": len(llm_handler.available_llms) if llm_handler else 0,
            "providers_total": (len(llm_handler.available_llms) + len(llm_handler.failed_providers)) if llm_handler else 0,
            "primary_llm": llm_handler.used_model if llm_handler else "unknown",
            "fallback_chain": [llm["name"] for llm in llm_handler.available_llms] if llm_handler else [],
            "failed_providers": llm_handler.failed_providers if llm_handler else {},
        },
        "documents": {
            "uploaded_count": len(_uploaded_documents),
            "indexed_count": sum(1 for d in _uploaded_documents if d["indexed"])
        },
        "logs": {
            "retention_days": log_manager.retention_days,
            "log_dir": str(log_manager.log_dir)
        }
    }


@app.get("/api/stats", tags=["System"])
async def stats():
    """Get system statistics — NO database."""
    return {
        "documents": {
            "total": len(_uploaded_documents),
            "indexed": sum(1 for d in _uploaded_documents if d["indexed"]),
            "not_indexed": sum(1 for d in _uploaded_documents if not d["indexed"])
        },
        "llm_providers_available": len(llm_handler.available_llms) if llm_handler else 0,
        "llm_providers_failed": len(llm_handler.failed_providers) if llm_handler else 0,
        "llm_failed_details": llm_handler.failed_providers if llm_handler else {}
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot",
        "version": "3.4.0",
        "features": [
            "Document upload (PDF, TXT, DOCX, JSON)",
            "Document list and delete",
            "Hybrid search (vector + BM25)",
            "Query rewriting + Re-ranking with Multi-LLM fallback",
            "Smart LLM fallback on ALL error types WITH 10s TIMEOUT",
            "Permanent provider disabling on API key leaks",
            "Rotating file logs with 15-day retention",
            "NO chat history / NO database / NO sessions",
            "Fallback chain: Gemini -> Groq -> OpenRouter"
        ],
        "docs": "http://localhost:8000/docs"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "status_code": exc.status_code, "detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"status": "error", "status_code": 500, "detail": f"Internal server error: {str(exc)}"}
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("RAG CHATBOT v3.4")
    logger.info("=" * 70)
    logger.info("Server will start at: http://localhost:8000")
    logger.info("API docs at: http://localhost:8000/docs")
    logger.info("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")