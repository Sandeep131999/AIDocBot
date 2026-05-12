"""
MAIN.PY - RAG CHATBOT (NO AUTHENTICATION)
=========================================

Features:
- Document upload and management
- Hybrid search (vector + BM25)
- LLM integration with FALLBACK (Gemini → Groq → OpenRouter)
- Automatic fallback on token/context limit errors
- SQLAlchemy database for chat history
- LangChain for LLM interactions
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
    top_k: int = 5      # Number of documents to retrieve


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
    description="Advanced RAG system with document management, chat history, and automatic LLM fallback on token limits",
    version="3.0.0"
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
# LLM WITH FALLBACK (STARTUP + RUNTIME)
# ============================================================================

class LLMWithFallback:
    """
    LLM with TWO levels of fallback:
    1. STARTUP: Initialize all available providers (not just first working one)
    2. RUNTIME: If current provider fails on token/context limit, auto-switch to next

    Fallback chain: Google Gemini → Groq → OpenRouter
    """

    # FIXED: Updated to current working Gemini models as of May 2026
    DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
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
        self.available_llms = []  # Store ALL working LLMs
        self.current_index = 0    # Which LLM is currently active
        self.used_model = "unknown"

        self._initialize_all_llms()

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars for English text)."""
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int, reserved: int = 1000) -> str:
        """
        Truncate context to fit within token limit.
        Reserves space for system prompt + chat history + user question.
        """
        max_context_tokens = max_tokens - reserved
        max_context_chars = max_context_tokens * 4  # 1 token ≈ 4 chars

        if len(context) > max_context_chars:
            truncated = context[:max_context_chars]
            # Try to cut at a sentence boundary
            last_period = truncated.rfind(". ")
            if last_period > max_context_chars * 0.8:
                truncated = truncated[:last_period + 1]
            return truncated + "\n\n[Content truncated due to length...]"
        return context

    def _initialize_all_llms(self):
        """Initialize ALL providers that have API keys (not just first one)."""

        print("\n" + "="*70)
        print("INITIALIZING ALL LLM PROVIDERS")
        print("="*70)

        # [1] Google Gemini (Free Tier: 60 requests/min)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                print("\n[1] Initializing Google Gemini...")
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
                    "type": "gemini"
                })
                print(f"✅ Google Gemini ready: {model_name}")
            except Exception as e:
                print(f"❌ Google Gemini init failed: {e}")

        # [2] Groq (Free Tier: 6000 TPM limit)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                print("\n[2] Initializing Groq...")
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
                    "type": "groq"
                })
                print(f"✅ Groq ready: {model_name}")
            except Exception as e:
                print(f"❌ Groq init failed: {e}")

        # [3] OpenRouter (Free models available)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                print("\n[3] Initializing OpenRouter...")
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
                    "type": "openrouter"
                })
                print(f"✅ OpenRouter ready: {model_name}")
            except Exception as e:
                print(f"❌ OpenRouter init failed: {e}")

        if not self.available_llms:
            raise HTTPException(
                status_code=500,
                detail="No LLM API keys found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY"
            )

        # Set primary LLM (first available)
        self.current_index = 0
        self.used_model = self.available_llms[0]["name"]
        print(f"\n🎯 Primary LLM: {self.used_model}")
        print(f"🔄 Fallback chain: {' → '.join([llm['name'] for llm in self.available_llms])}")
        print("="*70)

    def _is_token_error(self, error: Exception) -> bool:
        """
        Detect if error is token/context limit related.
        Covers Gemini, Groq, and OpenRouter error patterns.
        """
        error_str = str(error).lower()
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
            "quota exceeded",  # Gemini rate limit
            "rate limit",  # Generic rate limiting
        ]
        return any(pattern in error_str for pattern in token_error_patterns)

    def _build_messages(self, context: str, question: str, chat_history: list = None) -> list:
        """Build LangChain messages with system prompt + history + question."""
        messages = []

        raw_prompt = os.getenv("SYSTEM_PROMPT")

        system_prompt = raw_prompt.format(context=context)
        
        print(system_prompt)
        # DEBUG
        print(f"\n[DEBUG] Prompt first 300 chars: {system_prompt[:300]}")

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
        Generate response with AUTOMATIC fallback on token/context errors.

        Args:
            context: Retrieved documents context
            question: User question
            chat_history: Previous chat messages from DB
            attempt: Current retry attempt (0 = primary LLM)
            max_attempts: Maximum number of LLMs to try

        Returns:
            Tuple of (response_text, used_model)

        Raises:
            HTTPException: 413 if all LLMs fail due to token limits
            HTTPException: 502 for non-token provider errors
        """

        # Safety check: Don't exceed available LLMs or max attempts
        if attempt >= len(self.available_llms) or attempt >= max_attempts:
            raise HTTPException(
                status_code=413,
                detail=f"All LLM providers failed. Request too large or service unavailable. Try reducing document size or question length."
            )

        # Select current LLM based on attempt number
        current_llm_config = self.available_llms[attempt]
        current_llm = current_llm_config["llm"]
        self.used_model = current_llm_config["name"]

        print(f"\n[LLM ATTEMPT {attempt + 1}/{len(self.available_llms)}] Using: {self.used_model}")

        # Get context limit for current model and truncate if needed
        model_limit = self.CONTEXT_LIMITS.get(self.used_model, 8192)
        safe_context = self._truncate_context(context, model_limit, reserved=1500)

        # Build messages
        messages = self._build_messages(safe_context, question, chat_history)

        # Estimate total tokens for logging
        total_text = " ".join([m.content for m in messages])
        estimated_tokens = self._estimate_tokens(total_text)
        print(f"📊 Estimated tokens: {estimated_tokens} (limit: {model_limit})")

        try:
            # Generate response
            response = current_llm.invoke(messages)
            print(f"✅ Success with {self.used_model}")
            return response.content, self.used_model

        except Exception as e:
            error_msg = str(e)
            print(f"❌ {self.used_model} failed: {error_msg}")

            # Check if this is a token/context limit error
            if self._is_token_error(e):
                print(f"🔄 Token/context limit error detected! Trying next LLM...")

                # RECURSIVE FALLBACK: Try next available LLM
                return self.generate_response(
                    context=context,
                    question=question,
                    chat_history=chat_history,
                    attempt=attempt + 1,
                    max_attempts=max_attempts
                )
            else:
                # Non-token error (network, auth, server error) — don't fallback, report it
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM provider error ({self.used_model}): {error_msg}"
                )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""

    global retriever, llm_handler

    print("\n" + "="*70)
    print("STARTING RAG CHATBOT v3.0")
    print("="*70)

    try:
        # Initialize retriever
        retriever = Retriever(vector_weight=0.7, keyword_weight=0.3)
        print(f"\n✅ Retriever initialized")

        # Try to load existing documents
        try:
            num_chunks = retriever.load_and_index_documents("data/password_guide.txt")
            print(f"✅ Loaded {num_chunks} chunks from default documents")
        except FileNotFoundError:
            print("⚠️  No default documents found. Users can upload documents.")

        # Initialize LLM with full fallback chain
        llm_handler = LLMWithFallback()

    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
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

    print(f"\n[UPLOAD] File: {file.filename}")
    print("-"*70)

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

        print(f"✅ File saved: {file_path} ({file_size} bytes)")

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

                print(f"✅ Indexed {num_chunks} chunks")

                return DocumentUploadResponse(
                    filename=file.filename,
                    size=file_size,
                    indexed=True
                )

            except Exception as e:
                import traceback
                print(f"❌ Indexing error: {e}")
                print(f"❌ Full traceback:\n{traceback.format_exc()}")
                return DocumentUploadResponse(
                    filename=file.filename,
                    size=file_size,
                    indexed=False
                )
        else:
            raise HTTPException(status_code=500, detail="Retriever not initialized")

    except Exception as e:
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
      4. FALLS BACK to Groq → OpenRouter if token limits hit
    """

    print(f"\n[CHAT] Query: {request.query}")
    print("-"*70)

    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    try:
        # Step 1: Retrieve documents
        print("\n[STEP 1] Retrieve documents")
        results = retriever.retrieve(request.query, top_k=request.top_k)

        # If no results pass relevance threshold, return early
        if not results:
            print("⚠️ No relevant documents found above threshold")
            
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


        print(f"✅ Retrieved {len(results)} documents")

        # Step 2: Get chat history (AUTO-FETCHED from DB)
        print("\n[STEP 2] Get chat history")
        chat_messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == request.session_id
        ).order_by(ChatMessage.created_at).all()

        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_messages
        ]

        print(f"✅ Retrieved {len(chat_history)} history messages")

        # Step 3: Generate LLM response (WITH AUTOMATIC FALLBACK)
        print("\n[STEP 3] Generate LLM response")
        response_text, used_model = llm_handler.generate_response(
            context=context,
            question=request.query,
            chat_history=chat_history
            # attempt=0 is default — starts with primary LLM
        )

        print(f"✅ Generated response using {used_model}")

        # Step 4: Save to database
        print("\n[STEP 4] Save to database")

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

        print("✅ Saved to database")

        return ChatResponse(
            session_id=request.session_id,
            user_message=request.query,
            assistant_message=response_text,
            sources=sources,
            created_at=datetime.utcnow().isoformat(),
            used_model=used_model  # Shows which LLM actually answered (including fallback)
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including our 413/502 errors)
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
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
    """Health check."""

    return {
        "status": "healthy",
        "retriever_ready": retriever is not None,
        "llm_ready": hasattr(sys.modules[__name__], 'llm_handler'),
        "llm_providers": len(llm_handler.available_llms) if hasattr(sys.modules[__name__], 'llm_handler') else 0,
        "primary_llm": llm_handler.used_model if hasattr(sys.modules[__name__], 'llm_handler') else "unknown"
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
        "llm_providers_available": len(llm_handler.available_llms) if hasattr(sys.modules[__name__], 'llm_handler') else 0
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""

    return {
        "message": "RAG Chatbot",
        "version": "3.0.0",
        "features": [
            "Document upload (PDF, TXT, DOCX,JSON)",
            "Hybrid search (vector + BM25)",
            "Automatic LLM fallback on token limits",
            "Chat history with sessions",
            "Fallback chain: Gemini → Groq → OpenRouter"
        ],
        "docs": "http://localhost:8000/docs"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""

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

    print("\n" + "="*70)
    print("RAG CHATBOT v3.0")
    print("="*70)
    print("\nServer will start at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("="*70 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )