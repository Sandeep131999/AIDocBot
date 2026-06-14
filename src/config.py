"""
CONFIG LOADER
=============
Loads all settings from .env file with UTF-8 encoding.
Centralized configuration for the entire RAG system.

FIXES:
- Updated OPENROUTER_MODEL default to working free model
"""

import os
from dotenv import load_dotenv
from typing import Optional, List

# Load .env file with UTF-8 encoding
load_dotenv(encoding="utf-8")


class Config:
    """Centralized configuration manager. All settings read from .env"""

    # --- Chunking Settings ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "recursive")

    # --- Embedding Settings ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_NORMALIZE: bool = os.getenv("EMBEDDING_NORMALIZE", "True").lower() == "true"

    # --- Vector Store Settings ---
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "documents")
    VECTOR_SIMILARITY_METRIC: str = os.getenv("VECTOR_SIMILARITY_METRIC", "cosine")

    # --- Search & Retrieval Settings ---
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    KEYWORD_WEIGHT: float = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # --- Multi-LLM Provider Settings ---
    LLM_PROVIDER_ORDER: List[str] = os.getenv("LLM_PROVIDER_ORDER", "gemini,groq,openrouter").split(",")

    # Gemini
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # OpenRouter — FIXED: Updated to working free model
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    # NOTE: If left empty, multi_llm.py will auto-discover a working free model
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "")

    # Legacy OpenAI (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")

    # --- LLM Settings ---
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # --- Re-ranking Settings ---
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "True").lower() == "true"
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "3"))
    RERANKER_PROMPT_TEMPLATE: str = os.getenv("RERANKER_PROMPT_TEMPLATE", "default")

    # --- Query Rewriting Settings ---
    QUERY_REWRITING_ENABLED: bool = os.getenv("QUERY_REWRITING_ENABLED", "True").lower() == "true"
    QUERY_REWRITING_PROMPT_TEMPLATE: str = os.getenv("QUERY_REWRITING_PROMPT_TEMPLATE", "default")

    # --- File Processing ---
    SUPPORTED_EXTENSIONS: list = os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.txt,.csv,.xlsx,.xls,.docx,.doc,.json,.md").split(",")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- App Info ---
    APP_URL: str = os.getenv("APP_URL", "https://localhost")
    APP_NAME: str = os.getenv("APP_NAME", "RAG-System")

    @classmethod
    def print_config(cls):
        """Print all loaded configuration settings."""
        print("\n" + "="*70)
        print("CONFIGURATION SETTINGS (from .env)")
        print("="*70)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and key != "print_config":
                # Mask sensitive values
                if "API_KEY" in key or "SECRET" in key or "PASSWORD" in key:
                    value = "***MASKED***" if value else "NOT SET"
                print(f"   {key}: {value}")
        print("="*70 + "\n")


if __name__ == "__main__":
    Config.print_config()