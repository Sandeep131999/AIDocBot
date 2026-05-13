"""
MAIN.PY - RAG CHATBOT v3.1 (NO AUTHENTICATION)
===============================================

Features:
- Document upload and management
- Hybrid search (vector + BM25)
- LLM integration with SMART FALLBACK (Gemini -> Groq -> OpenRouter)
- Automatic fallback on ALL provider errors (token, rate limit, network, key leak)
- PERMANENT provider disabling on API key leaks (403)
- SQLAlchemy database for chat history
- LangChain for LLM interactions
- Rotating file logs with 15-day retention
- Separate error.log with full traces
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

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
import secrets
import logging
import traceback
from logging.handlers import RotatingFileHandler
import glob
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever
from src.evaluator import Evaluator

# LangChain imports - UPDATED to modern packages
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

# File handling
import shutil
import PyPDF2

# Environment
from dotenv import load_dotenv

load_dotenv()
# ============================================================================
# WINDOWS CONSOLE ENCODING FIX
# ============================================================================
import sys
if sys.platform == "win32":
    import io
    # Force UTF-8 for stdout/stderr to prevent UnicodeEncodeError on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# LOG MANAGER: Rotating files + 15-day retention + separate error log
# ============================================================================

class LogManager:
    """
    Manages rotating log files with:
    - Size-based rotation (10MB per file, 10 backups)
    - 15-day retention (auto-deletes old logs on startup)
    - Separate general.log and errors.log
    - Full stack traces in error log
    """

    def __init__(self, log_dir="logs", retention_days=15):
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.log_dir.mkdir(exist_ok=True)

        # Clean up old logs first
        self._cleanup_old_logs()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler (INFO+)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

        # General log file (INFO+, rotating 10MB)
        general_log_path = self.log_dir / "rag_chatbot.log"
        general_handler = RotatingFileHandler(
            general_log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(general_handler)

        # Error log file (ERROR+, rotating 10MB) with full traces
        error_log_path = self.log_dir / "rag_chatbot_errors.log"
        error_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LogManager initialized. Logs: {self.log_dir}")
        self.logger.info(f"Retention: {retention_days} days")

    def _cleanup_old_logs(self):
        """Delete log files older than retention_days."""
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


# Initialize logging FIRST (before anything else)
log_manager = LogManager(log_dir="logs", retention_days=15)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot.db")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".txt", ".csv", ".xls", ".xlsx", ".json", ".pdf", ".docx"}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================================================
# DATABASE MODELS
# ============================================================================

class Document(Base):
    """Document model for tracking uploaded files."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    indexed = Column(Integer, default=0)  # Boolean as integer


class ChatMessage(Base):
    """Chat message model for maintaining history."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """Chat session model."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request. You only need to send these two fields:"""
    query: str          # Your question
    session_id: str     # Unique session ID (e.g., uuid or any string)
    top_k: int = os.getenv("TOP_K")    # Number of documents to retrieve


class ChatResponse(BaseModel):
    """Chat response."""
    session_id: str
    user_message: str
    assistant_message: str
    sources: list
    created_at: str
    used_model: str


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    filename: str
    size: int
    indexed: bool


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    messages: list
    session_title: str


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RAG Chatbot",
    description="Advanced RAG system with document management, chat history, and smart LLM fallback on all error types",
    version="3.1.0"
)

# Add CORS
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
# DEPENDENCY: GET DATABASE SESSION
# ============================================================================

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# LLM WITH SMART FALLBACK (v3.1)
# ============================================================================

class LLMWithSmartFallback:
    """
    LLM with SMART fallback:
    1. STARTUP: Initialize all available providers
    2. RUNTIME: Classify errors and decide action:
       - API Key Leaked (403) -> PERMANENTLY disable provider
       - Rate Limit (429) -> Try next provider
       - Token/Context Limit -> Try next provider
       - Network Error (502-504) -> Try next provider
       - Other -> Try next provider

    Fallback chain: Google Gemini -> Groq -> OpenRouter
    """

    # FIXED: Updated to current working Gemini models as of May 2026
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
    DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
    DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

    # Context window limits for each provider (conservative estimates)
    CONTEXT_LIMITS = {
        "gemini-2.0-flash": 1_000_000,
        "gemini-2.5-flash": 1_000_000,
        "llama-3.1-70b-versatile": 8192,
        "meta-llama/llama-3.1-8b-instruct:free": 8192,
    }

    def __init__(self):
        """Initialize ALL available LLM providers for runtime fallback."""
        self.available_llms = []      # Store ALL working LLMs
        self.failed_providers = {}    # Track permanently failed providers: {name: reason}
        self.current_index = 0        # Which LLM is currently active
        self.used_model = "unknown"

        self._initialize_all_llms()

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ~ 4 chars for English text)."""
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int, reserved: int = 1500) -> str:
        """
        Truncate context to fit within token limit.
        Reserves space for system prompt + chat history + user question.
        """
        max_context_tokens = max_tokens - reserved
        max_context_chars = max_context_tokens * 4  # 1 token ~ 4 chars

        if len(context) > max_context_chars:
            truncated = context[:max_context_chars]
            # Try to cut at a sentence boundary
            last_period = truncated.rfind(". ")
            if last_period > max_context_chars * 0.8:
                truncated = truncated[:last_period + 1]
            return truncated + "\n\n[Content truncated due to length...]"
        return context

    def _classify_error(self, error: Exception) -> tuple:
        """
        Classify LLM error into category and action.

        Returns:
            (error_type, action, should_permanently_disable)
            error_type: str describing the error
            action: "fallback" | "raise"
            should_permanently_disable: bool
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # [1] API Key Leaked / Invalid (403) -> PERMANENTLY DISABLE
        key_leak_patterns = [
            "api_key_leaked",
            "leaked",
            "invalid api key",
            "authentication failed",
            "auth failed",
            "unauthorized",
            "401",
            "403",
        ]
        if any(p in error_str for p in key_leak_patterns):
            return ("api_key_leaked", "fallback", True)

        # [2] Rate Limit (429) -> Fallback (temporary)
        rate_limit_patterns = [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "quota",
            "limit exceeded",
        ]
        if any(p in error_str for p in rate_limit_patterns):
            return ("rate_limit", "fallback", False)

        # [3] Token / Context Limit -> Fallback (temporary)
        token_error_patterns = [
            "too many tokens",
            "context length",
            "context window",
            "request payload size",
            "request too large",
            "max tokens",
            "max_output_tokens",
            "exceeds the limit",
            "token limit",
            "413",  # Payload Too Large
        ]
        if any(p in error_str for p in token_error_patterns):
            return ("token_limit", "fallback", False)

        # [4] Network / Server Error (502, 503, 504) -> Fallback (temporary)
        network_patterns = [
            "connection error",
            "timeout",
            "502",
            "503",
            "504",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "network",
            "unable to connect",
            "connection refused",
        ]
        if any(p in error_str for p in network_patterns):
            return ("network_error", "fallback", False)

        # [5] Unknown / Other -> Fallback (temporary) as last resort
        return ("unknown_error", "fallback", False)

    def _initialize_all_llms(self):
        """Initialize ALL providers that have API keys (not just first one)."""

        logger.info("=" * 70)
        logger.info("INITIALIZING ALL LLM PROVIDERS")
        logger.info("=" * 70)

        # [1] Google Gemini (Free Tier: 60 requests/min)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                logger.info("[1] Initializing Google Gemini...")
                model_name = os.getenv("GEMINI_MODEL", self.DEFAULT_GEMINI_MODEL)
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=gemini_key,
                    temperature=0.7,
                    max_output_tokens=2048  # Prevent runaway generation
                )
                self.available_llms.append({
                    "name": model_name,
                    "llm": llm,
                    "type": "gemini",
                    "display_name": "Google Gemini"
                })
                logger.info(f"[OK] Google Gemini ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] Google Gemini init failed: {e}")
                self.failed_providers["Google Gemini"] = f"Init failed: {str(e)}"

        # [2] Groq (Free Tier: 6000 TPM limit)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                logger.info("[2] Initializing Groq...")
                model_name = os.getenv("GROQ_MODEL", self.DEFAULT_GROQ_MODEL)
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.7,
                    max_tokens=1024  # Stay under Groq TPM limits
                )
                self.available_llms.append({
                    "name": model_name,
                    "llm": llm,
                    "type": "groq",
                    "display_name": "Groq"
                })
                logger.info(f"[OK] Groq ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] Groq init failed: {e}")
                self.failed_providers["Groq"] = f"Init failed: {str(e)}"

        # [3] OpenRouter (Free models available)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                logger.info("[3] Initializing OpenRouter...")
                model_name = os.getenv("OPENROUTER_MODEL", self.DEFAULT_OPENROUTER_MODEL)
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.7,
                    max_tokens=1024
                )
                self.available_llms.append({
                    "name": model_name,
                    "llm": llm,
                    "type": "openrouter",
                    "display_name": "OpenRouter"
                })
                logger.info(f"[OK] OpenRouter ready: {model_name}")
            except Exception as e:
                logger.error(f"[ERR] OpenRouter init failed: {e}")
                self.failed_providers["OpenRouter"] = f"Init failed: {str(e)}"

        if not self.available_llms:
            raise HTTPException(
                status_code=500,
                detail="No LLM API keys found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY"
            )

        # Set primary LLM (first available)
        self.current_index = 0
        self.used_model = self.available_llms[0]["name"]
        logger.info(f"[PRIMARY] Primary LLM: {self.used_model}")
        logger.info(f"[FALLBACK] Fallback chain: {' -> '.join([llm['name'] for llm in self.available_llms])}")
        logger.info("=" * 70)

    def _build_messages(self, context: str, question: str, chat_history: list = None) -> list:
        """Build LangChain messages with system prompt + history + question."""
        messages = []

        raw_prompt = os.getenv("SYSTEM_PROMPT")

        system_prompt = raw_prompt.format(context=context)

        # DEBUG
        logger.debug(f"Prompt first 300 chars: {system_prompt[:300]}")

        messages.append(SystemMessage(content=system_prompt))

        # Add chat history (last 5 messages for context)
        if chat_history:
            for msg in chat_history[-5:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                else:
                    messages.append(AIMessage(content=msg['content']))

        # Add current question
        messages.append(HumanMessage(content=question))

        return messages

    def generate_response(self, context: str, question: str, chat_history: list = None,
                         attempt: int = 0, max_attempts: int = 3) -> tuple:
        """
        Generate response with SMART automatic fallback on ALL errors.

        Args:
            context: Retrieved documents context
            question: User question
            chat_history: Previous chat messages from DB
            attempt: Current retry attempt (0 = primary LLM)
            max_attempts: Maximum number of LLMs to try

        Returns:
            Tuple of (response_text, used_model)

        Raises:
            HTTPException: 503 if all LLMs fail
            HTTPException: 502 for non-recoverable errors
        """

        # Safety check: Don't exceed available LLMs or max attempts
        if attempt >= len(self.available_llms) or attempt >= max_attempts:
            logger.error("All LLM providers exhausted")
            raise HTTPException(
                status_code=503,
                detail="All LLM providers failed. Service temporarily unavailable. Try again later or check API keys."
            )

        # Select current LLM based on attempt number
        current_llm_config = self.available_llms[attempt]
        current_llm = current_llm_config["llm"]
        provider_name = current_llm_config["display_name"]
        self.used_model = current_llm_config["name"]

        logger.info(f"[LLM ATTEMPT {attempt + 1}/{len(self.available_llms)}] Provider: {provider_name}")

        # Get context limit for current model and truncate if needed
        model_limit = self.CONTEXT_LIMITS.get(self.used_model, 8192)
        safe_context = self._truncate_context(context, model_limit, reserved=1500)

        # Build messages
        messages = self._build_messages(safe_context, question, chat_history)

        # Estimate total tokens for logging
        total_text = " ".join([m.content for m in messages])
        estimated_tokens = self._estimate_tokens(total_text)
        logger.info(f"[TOK] Estimated tokens: {estimated_tokens} (limit: {model_limit})")

        try:
            # Generate response
            response = current_llm.invoke(messages)
            logger.info(f"[OK] Success with {provider_name} ({self.used_model})")
            return response.content, self.used_model

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERR] {provider_name} failed: {error_msg}")
            logger.error(traceback.format_exc())  # Full trace to error log

            # SMART ERROR CLASSIFICATION
            error_type, action, permanently_disable = self._classify_error(e)
            logger.warning(f"[CHECK] Error classified as: {error_type} (action={action}, permanent_disable={permanently_disable})")

            if permanently_disable:
                # PERMANENTLY disable this provider
                self.failed_providers[provider_name] = f"{error_type}: {error_msg}"
                # Remove from available list
                self.available_llms.pop(attempt)
                logger.warning(f"[WARN]  {provider_name}: {error_type} - PERMANENTLY DISABLED")

                # Recursive fallback (same attempt index since we removed the failed one)
                if self.available_llms:
                    return self.generate_response(
                        context=context,
                        question=question,
                        chat_history=chat_history,
                        attempt=attempt,  # Same index, next provider shifted into this slot
                        max_attempts=max_attempts
                    )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"All LLM providers permanently disabled. Last error: {error_msg}"
                    )

            elif action == "fallback":
                # Temporary failure -> try next provider
                logger.warning(f"[FALLBACK] {provider_name}: {error_type} - trying next provider")

                # RECURSIVE FALLBACK: Try next available LLM
                return self.generate_response(
                    context=context,
                    question=question,
                    chat_history=chat_history,
                    attempt=attempt + 1,
                    max_attempts=max_attempts
                )

            else:
                # Non-recoverable error
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM provider error ({provider_name}): {error_msg}"
                )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""

    global retriever, llm_handler

    logger.info("=" * 70)
    logger.info("STARTING RAG CHATBOT v3.1")
    logger.info("=" * 70)

    try:
        # Initialize retriever
        # Note: If documents exist but return "not found", check src/retriever.py threshold (default 0.35)
        retriever = Retriever(vector_weight=0.7, keyword_weight=0.3,)
        retriever = Retriever(
            vector_weight=float(os.getenv("RETRIEVER_VECTOR_WEIGHT")),
            keyword_weight=float(os.getenv("RETRIEVER_KEYWORD_WEIGHT")),
            min_relevance_score=float(os.getenv("RETRIEVER_MIN_SCORE"))
        )
        logger.info("[OK] Retriever initialized")

        # Try to load existing documents
        try:
            num_chunks = retriever.load_and_index_documents("data/password_guide.txt")
            logger.info(f"[OK] Loaded {num_chunks} chunks from default documents")
        except FileNotFoundError:
            logger.warning("[WARN]  No default documents found. Users can upload documents.")

        # Initialize LLM with smart fallback
        llm_handler = LLMWithSmartFallback()

    except Exception as e:
        logger.error(f"[ERR] Error during startup: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# DOCUMENT UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and index a document.

    Supported formats: PDF, TXT, DOCX
    """

    logger.info(f"[UPLOAD] File: {file.filename}")
    logger.info("-" * 70)

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        logger.info(f"[OK] File saved: {file_path} ({file_size} bytes)")

        # Store in database
        doc = Document(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            indexed=0
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Index document
        if retriever:
            try:
                num_chunks = retriever.load_and_index_documents(file_path)

                # Update indexed status
                doc.indexed = 1
                db.commit()

                logger.info(f"[OK] Indexed {num_chunks} chunks")

                return DocumentUploadResponse(
                    filename=file.filename,
                    size=file_size,
                    indexed=True
                )

            except Exception as e:
                logger.error(f"[ERR] Indexing error: {e}")
                logger.error(traceback.format_exc())
                return DocumentUploadResponse(
                    filename=file.filename,
                    size=file_size,
                    indexed=False
                )
        else:
            raise HTTPException(status_code=500, detail="Retriever not initialized")

    except Exception as e:
        logger.error(f"[ERR] Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    db: Session = Depends(get_db)
):
    """Get all documents."""

    documents = db.query(Document).all()

    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "size": doc.file_size,
                "indexed": bool(doc.indexed),
                "uploaded_at": doc.uploaded_at.isoformat()
            }
            for doc in documents
        ]
    }


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Chat with RAG system.

    HOW TO USE:
    -----------
    You ONLY need to send:
      - query:      Your question text
      - session_id: Any unique string to identify the conversation

    The server AUTOMATICALLY:
      1. Fetches previous messages from DB using session_id
      2. Retrieves relevant documents
      3. Tries primary LLM (Gemini)
      4. SMART FALLBACK to Groq -> OpenRouter on ANY error (token, rate limit, network, key leak)
    """

    logger.info(f"[CHAT] Query: {request.query}")
    logger.info("-" * 70)

    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    try:
        # Step 1: Retrieve documents
        logger.info("[STEP 1] Retrieve documents")
        results = retriever.retrieve(request.query, top_k=request.top_k)

        # If no results pass relevance threshold, try expanding search
        if not results:
            logger.warning("[WARN] No relevant documents found above threshold (0.35)")
            logger.info("[FALLBACK] Trying with expanded top_k...")

            # Try with more documents - maybe some will pass threshold
            try:
                results = retriever.retrieve(request.query, top_k=request.top_k * 2)
            except Exception as e:
                logger.error(f"[ERR] Expanded retrieve failed: {e}")
                results = []

            if not results:
                logger.warning("[WARN] Still no results after expanding search")

            # Still save user message to history
            user_msg = ChatMessage(
                session_id=request.session_id,
                role="user",
                content=request.query
            )
            db.add(user_msg)

            # Create/ensure session exists
            session = db.query(ChatSession).filter(
                ChatSession.session_id == request.session_id
            ).first()
            if not session:
                session = ChatSession(
                    session_id=request.session_id,
                    title=request.query[:50] if request.query else "New Chat"
                )
                db.add(session)

            db.commit()

            return ChatResponse(
                session_id=request.session_id,
                user_message=request.query,
                assistant_message="I don't have information about that in my knowledge base.",
                sources=[],
                created_at=datetime.utcnow().isoformat(),
                used_model="none"
            )

        context = "\n\n---\n\n".join([r['document'] for r in results])
        sources = [
            {
                "document": r['document'][:100] + "...",
                "combined_score": r['combined_score'],
                "vector_score": r['vector_score'],
                "keyword_score": r['keyword_score'],
                "employee_name": r['metadata'].get('employee_name', 'N/A')
            }
            for r in results
        ]

        logger.info(f"[OK] Retrieved {len(results)} documents")

        # Step 2: Get chat history (AUTO-FETCHED from DB)
        logger.info("[STEP 2] Get chat history")
        chat_messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == request.session_id
        ).order_by(ChatMessage.created_at).all()

        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_messages
        ]

        logger.info(f"[OK] Retrieved {len(chat_history)} history messages")

        # Step 3: Generate LLM response (WITH SMART AUTOMATIC FALLBACK)
        logger.info("[STEP 3] Generate LLM response")
        response_text, used_model = llm_handler.generate_response(
            context=context,
            question=request.query,
            chat_history=chat_history
            # attempt=0 is default -- starts with primary LLM
        )

        logger.info(f"[OK] Generated response using {used_model}")

        # Step 4: Save to database
        logger.info("[STEP 4] Save to database")

        # Ensure session exists
        session = db.query(ChatSession).filter(
            ChatSession.session_id == request.session_id
        ).first()

        if not session:
            session = ChatSession(
                session_id=request.session_id,
                title=request.query[:50] if request.query else "New Chat"
            )
            db.add(session)
            db.commit()

        # Save user message
        user_msg = ChatMessage(
            session_id=request.session_id,
            role="user",
            content=request.query
        )
        db.add(user_msg)

        # Save assistant response
        assistant_msg = ChatMessage(
            session_id=request.session_id,
            role="assistant",
            content=response_text
        )
        db.add(assistant_msg)

        # Update session
        session.updated_at = datetime.utcnow()

        db.commit()

        logger.info("[OK] Saved to database")

        return ChatResponse(
            session_id=request.session_id,
            user_message=request.query,
            assistant_message=response_text,
            sources=sources,
            created_at=datetime.utcnow().isoformat(),
            used_model=used_model  # Shows which LLM actually answered (including fallback)
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including our 503/502 errors)
        raise
    except Exception as e:
        logger.error(f"[ERR] Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ============================================================================
# CHAT HISTORY ENDPOINTS
# ============================================================================

@app.get("/api/chat/sessions", tags=["Chat"])
async def get_sessions(
    db: Session = Depends(get_db)
):
    """Get all chat sessions."""

    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat()
            }
            for s in sessions
        ]
    }


@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse, tags=["Chat"])
async def get_chat_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get chat history for a session."""

    session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()

    return ChatHistoryResponse(
        messages=[
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ],
        session_title=session.title
    )


@app.delete("/api/chat/sessions/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a chat session and its messages."""

    # Verify session exists
    session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete messages
    db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).delete()

    # Delete session
    db.delete(session)
    db.commit()

    return {"success": True, "message": "Session deleted"}


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/health", tags=["System"])
async def health():
    """Health check with provider status."""

    return {
        "status": "healthy",
        "version": "3.1.0",
        "retriever_ready": retriever is not None,
        "llm_ready": llm_handler is not None,
        "llm": {
            "providers_available": len(llm_handler.available_llms) if llm_handler else 0,
            "providers_total": (len(llm_handler.available_llms) + len(llm_handler.failed_providers)) if llm_handler else 0,
            "primary_llm": llm_handler.used_model if llm_handler else "unknown",
            "fallback_chain": [llm["name"] for llm in llm_handler.available_llms] if llm_handler else [],
            "failed_providers": llm_handler.failed_providers if llm_handler else {},
        },
        "logs": {
            "retention_days": log_manager.retention_days,
            "log_dir": str(log_manager.log_dir)
        }
    }


@app.get("/api/stats", tags=["System"])
async def stats(db: Session = Depends(get_db)):
    """Get system statistics."""

    num_docs = db.query(Document).count()
    num_messages = db.query(ChatMessage).count()
    num_sessions = db.query(ChatSession).count()

    return {
        "documents": num_docs,
        "chat_messages": num_messages,
        "chat_sessions": num_sessions,
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
        "version": "3.1.0",
        "features": [
            "Document upload (PDF, TXT, DOCX, JSON)",
            "Hybrid search (vector + BM25)",
            "Smart LLM fallback on ALL error types",
            "Permanent provider disabling on API key leaks",
            "Chat history with sessions",
            "Rotating file logs with 15-day retention",
            "Fallback chain: Gemini -> Groq -> OpenRouter"
        ],
        "docs": "http://localhost:8000/docs"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""

    logger.error(f"HTTP {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""

    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "status_code": 500,
            "detail": f"Internal server error: {str(exc)}"
        }
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("RAG CHATBOT v3.1")
    logger.info("=" * 70)
    logger.info("Server will start at: http://localhost:8000")
    logger.info("API docs at: http://localhost:8000/docs")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )