"""
CONFIG LOADER
=============
Loads all settings from .env file with UTF-8 encoding.
Centralized configuration for the entire RAG system.

RULE: NO HARDCODED DEFAULTS. Every value MUST come from .env.
If a required .env key is missing, the system raises an error.
"""

import os
from dotenv import load_dotenv
from typing import Optional, List

# Load .env file with UTF-8 encoding
load_dotenv(encoding="utf-8")


def _require_env(key: str) -> str:
    """Get a required env var. Raises if missing."""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise ValueError(f"Required .env key '{key}' is missing or empty.")
    return value


def _get_env(key: str, required: bool = True) -> str:
    """Get env var. If required=True, raises on missing."""
    value = os.getenv(key)
    if required and (value is None or value.strip() == ""):
        raise ValueError(f"Required .env key '{key}' is missing or empty.")
    return value if value is not None else ""


def _parse_list(key: str, required: bool = True) -> List[str]:
    """Parse comma-separated env value into list."""
    raw = _get_env(key, required=required)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_bool(key: str, required: bool = True) -> bool:
    """Parse env value as boolean."""
    raw = _get_env(key, required=required).lower().strip()
    return raw in ("true", "1", "yes", "on")


def _parse_int(key: str, required: bool = True) -> int:
    """Parse env value as int."""
    raw = _get_env(key, required=required)
    return int(raw)


def _parse_float(key: str, required: bool = True) -> float:
    """Parse env value as float."""
    raw = _get_env(key, required=required)
    return float(raw)


class Config:
    """Centralized configuration manager. All settings read from .env"""

    # --- Chunking Settings ---
    CHUNK_SIZE: int = _parse_int("CHUNK_SIZE")
    CHUNK_OVERLAP: int = _parse_int("CHUNK_OVERLAP")
    CHUNK_STRATEGY: str = _get_env("CHUNK_STRATEGY")

    # --- Embedding Settings ---
    EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL")
    EMBEDDING_DEVICE: str = _get_env("EMBEDDING_DEVICE")
    EMBEDDING_NORMALIZE: bool = _parse_bool("EMBEDDING_NORMALIZE")

    # --- Vector Store Settings ---
    VECTOR_DB_PATH: str = _get_env("VECTOR_DB_PATH")
    VECTOR_COLLECTION: str = _get_env("VECTOR_COLLECTION")
    VECTOR_SIMILARITY_METRIC: str = _get_env("VECTOR_SIMILARITY_METRIC")

    # --- Search & Retrieval Settings ---
    VECTOR_WEIGHT: float = _parse_float("VECTOR_WEIGHT")
    KEYWORD_WEIGHT: float = _parse_float("KEYWORD_WEIGHT")
    MIN_RELEVANCE_SCORE: float = _parse_float("MIN_RELEVANCE_SCORE")
    TOP_K: int = _parse_int("TOP_K")

    # --- Multi-LLM Provider Settings ---
    LLM_PROVIDER_ORDER: List[str] = _parse_list("LLM_PROVIDER_ORDER")

    # Gemini
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = _get_env("GEMINI_MODEL")

    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = _get_env("GROQ_MODEL")

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = _get_env("OPENROUTER_MODEL")

    # OpenRouter Free Models (comma-separated list for auto-rotation)
    OPENROUTER_FREE_MODELS: List[str] = _parse_list("OPENROUTER_FREE_MODELS")

    # Fallback error keywords (comma-separated)
    FALLBACK_ERRORS: List[str] = _parse_list("FALLBACK_ERRORS")

    # Legacy OpenAI (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = _get_env("LLM_MODEL", required=False) or ""

    # --- LLM Settings ---
    LLM_TEMPERATURE: float = _parse_float("LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = _parse_int("LLM_MAX_TOKENS")
    LLM_PROVIDER_TIMEOUT: int = _parse_int("LLM_PROVIDER_TIMEOUT")

    # --- Re-ranking Settings ---
    RERANKER_ENABLED: bool = _parse_bool("RERANKER_ENABLED")
    RERANKER_TOP_N: int = _parse_int("RERANKER_TOP_N")
    RERANKER_PROMPT_TEMPLATE: str = _get_env("RERANKER_PROMPT_TEMPLATE")

    # --- Query Rewriting Settings ---
    QUERY_REWRITING_ENABLED: bool = _parse_bool("QUERY_REWRITING_ENABLED")
    QUERY_REWRITING_PROMPT_TEMPLATE: str = _get_env("QUERY_REWRITING_PROMPT_TEMPLATE")

    # --- File Processing ---
    SUPPORTED_EXTENSIONS: List[str] = _parse_list("SUPPORTED_EXTENSIONS")
    MAX_FILE_SIZE_MB: int = _parse_int("MAX_FILE_SIZE_MB")

    # --- Logging ---
    LOG_LEVEL: str = _get_env("LOG_LEVEL")

    # --- App Info ---
    APP_URL: str = _get_env("APP_URL")
    APP_NAME: str = _get_env("APP_NAME")

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