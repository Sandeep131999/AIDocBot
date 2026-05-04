"""
API/MAIN.PY
===========

FastAPI REST server for RAG system.
Located in api/ folder as per project structure.

Endpoints:
- POST /upload    - Upload and index new documents  ← NEW!
- POST /search    - Search documents
- POST /chat      - Chat with context
- GET /health     - Health check
- GET /stats      - System statistics
"""

# ============================================================================
# IMPORTS
# ============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
from pathlib import Path
import shutil
import uuid
import os

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever
from src.evaluator import Evaluator


# ============================================================================
# CONFIGURATION
# ============================================================================

# Where uploaded files will be stored
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions (must match DocumentLoader support)
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.csv'}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for /search endpoint."""
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Single search result."""
    document: str
    combined_score: float
    vector_score: float
    keyword_score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Response model for /search endpoint."""
    query: str
    results: List[SearchResult]
    num_results: int


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    query: str
    context: List[str]
    num_context: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    indexed: bool


class UploadResponse(BaseModel):
    """Response model for /upload endpoint."""
    filename: str
    original_name: str
    file_size: int
    chunks_indexed: int
    status: str


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever and evaluator
retriever = None
evaluator = Evaluator()


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup."""
    
    global retriever
    
    print("\n" + "="*70)
    print("STARTING RAG API SERVER")
    print("="*70)
    
    try:
        retriever = Retriever(vector_weight=0.7, keyword_weight=0.3)
        
        # Load documents (optional - remove if you only want upload-based indexing)
        print(f"\nLoading default documents...")
        try:
            num_chunks = retriever.load_and_index_documents("data/password_guide.txt")
            print(f"✅ Loaded {num_chunks} chunks from default source")
        except FileNotFoundError:
            print("⚠️  No default documents found. Use /upload to add documents.")
        
    except Exception as e:
        print(f"\n❌ Error during startup: {e}")
        raise


# ============================================================================
# NEW: UPLOAD ENDPOINT
# ============================================================================

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a new document.
    
    Accepts PDF, TXT, or CSV files. The file is saved, processed into chunks,
    and indexed for both vector and keyword search.
    
    Args:
        file: The document file to upload
    
    Returns:
        Upload metadata including number of chunks indexed
    
    EXAMPLE (curl):
        curl -X POST "http://localhost:8000/upload" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@my_document.pdf"
    
    EXAMPLE (Python requests):
        >>> import requests
        >>> with open("doc.pdf", "rb") as f:
        ...     response = requests.post(
        ...         "http://localhost:8000/upload",
        ...         files={"file": ("doc.pdf", f, "application/pdf")}
        ...     )
        >>> print(response.json())
    """
    
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate safe filename: uuid + original extension
    safe_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = UPLOAD_DIR / safe_filename
    
    print(f"\n📤 UPLOAD RECEIVED")
    print(f"   Original name: {file.filename}")
    print(f"   Saved as: {safe_filename}")
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        print(f"   File size: {file_size:,} bytes")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    finally:
        file.file.close()
    
    # Process and index the document
    try:
        print(f"\n📊 Processing document...")
        
        # Use the retriever's document loader and indexing pipeline
        docs = retriever.document_loader.load_from_file(str(file_path))
        
        # Generate embeddings
        chunk_texts = [doc.page_content for doc in docs]
        embeddings = retriever.embeddings_generator.embed_texts(chunk_texts)
        
        # Index in vector store
        retriever.vector_store.add_documents(docs, embeddings)
        
        # Index for keyword search
        retriever.keyword_searcher.index_documents(chunk_texts)
        
        # Mark as indexed
        retriever.indexed = True
        
        print(f"✅ Upload complete: {len(docs)} chunks indexed")
        
        return UploadResponse(
            filename=safe_filename,
            original_name=file.filename,
            file_size=file_size,
            chunks_indexed=len(docs),
            status="success"
        )
        
    except Exception as e:
        # Clean up saved file if indexing fails
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document: {str(e)}"
        )


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of uploaded files with metadata
    """
    
    if not UPLOAD_DIR.exists():
        return {"documents": []}
    
    documents = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            stat = file_path.stat()
            documents.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "uploaded": stat.st_ctime
            })
    
    return {"documents": documents, "count": len(documents)}


# ============================================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "upload": "POST /upload",
            "search": "POST /search",
            "chat": "POST /chat",
            "health": "GET /health",
            "stats": "GET /stats",
            "documents": "GET /documents"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    return HealthResponse(
        status="healthy",
        indexed=retriever.indexed
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search for relevant documents.
    
    Args:
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of relevant documents ranked by relevance
    
    EXAMPLE:
        {
            "query": "How to reset password?",
            "top_k": 5
        }
    """
    
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    if not retriever.indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a document first using POST /upload"
        )
    
    try:
        # Retrieve documents
        results = retriever.retrieve(request.query, top_k=request.top_k)
        
        # Format results
        formatted_results = [
            SearchResult(
                document=r['document'],
                combined_score=r['combined_score'],
                vector_score=r['vector_score'],
                keyword_score=r['keyword_score'],
                metadata=r['metadata']
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            num_results=len(formatted_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint with context augmentation.
    
    Retrieves relevant context for a question.
    
    EXAMPLE:
        {
            "query": "Who is sandeep?",
            "top_k": 3
        }
    """
    
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    if not retriever.indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a document first using POST /upload"
        )
    
    try:
        # Retrieve context
        results = retriever.retrieve(request.query, top_k=request.top_k)
        
        # Extract documents as context
        context = [r['document'] for r in results]
        
        return ChatResponse(
            query=request.query,
            context=context,
            num_context=len(context)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


@app.get("/stats", tags=["System"])
async def stats():
    """Get system statistics."""
    
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    # Count uploaded documents
    doc_count = 0
    if UPLOAD_DIR.exists():
        doc_count = len([
            f for f in UPLOAD_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        ])
    
    return {
        "status": "healthy" if retriever.indexed else "not_indexed",
        "indexed": retriever.indexed,
        "vector_weight": retriever.vector_weight,
        "keyword_weight": retriever.keyword_weight,
        "embeddings_model": "all-MiniLM-L6-v2",
        "embeddings_dimension": 384,
        "uploaded_documents": doc_count,
        "upload_directory": str(UPLOAD_DIR)
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    
    return {
        "status": "error",
        "status_code": exc.status_code,
        "detail": exc.detail
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("STARTING RAG API SERVER")
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