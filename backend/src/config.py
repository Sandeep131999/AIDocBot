"""
CONFIG LOADER - ENTERPRISE
All values from.env, UTF-8, no hardcoded defaults.
Fixed _parse_int crash on empty values.
"""

import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv(encoding="utf-8")

def _get_raw(key: str) -> Optional[str]:
    """Returns None if missing OR empty string after strip"""
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    return val if val!= "" else None

def _require_env(key: str) -> str:
    val = _get_raw(key)
    if val is None:
        raise ValueError(f"Required.env key '{key}' is missing or empty")
    return val

def _get_env(key: str, required: bool = True, default: str = "") -> str:
    val = _get_raw(key)
    if val is None:
        if required:
            raise ValueError(f"Required.env key '{key}' is missing")
        return default
    return val

def _parse_list(key: str, required: bool = True) -> List[str]:
    raw = _get_raw(key)
    if raw is None:
        if required:
            raise ValueError(f"Required.env key '{key}' is missing")
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]

def _parse_int(key: str, required: bool = True, default: Optional[int] = None) -> int:
    raw = _get_raw(key)
    if raw is None:
        if default is not None:
            return default
        if not required:
            return 0
        raise ValueError(f"Required.env key '{key}' is missing")
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f".env key '{key}' must be int, got '{raw}'")

def _parse_float(key: str, required: bool = True, default: Optional[float] = None) -> float:
    raw = _get_raw(key)
    if raw is None:
        if default is not None:
            return default
        if not required:
            return 0.0
        raise ValueError(f"Required.env key '{key}' is missing")
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f".env key '{key}' must be float, got '{raw}'")

def _parse_bool(key: str, required: bool = False, default: bool = False) -> bool:
    raw = _get_raw(key)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes", "on")

class Config:
    # Chunking
    CHUNK_SIZE: int = _parse_int("CHUNK_SIZE")
    CHUNK_OVERLAP: int = _parse_int("CHUNK_OVERLAP")
    CHUNK_STRATEGY: str = _require_env("CHUNK_STRATEGY")

    # Embedding
    HUGGINGFACEHUB_API_TOKEN: str = _get_env("HUGGINGFACEHUB_API_TOKEN", required=False, default="")
    USE_OLLAMA_EMBEDDINGS: bool = _parse_bool("USE_OLLAMA_EMBEDDINGS", required=False)
    OLLAMA_EMBEDDING_MODEL: str = _get_env("OLLAMA_EMBEDDING_MODEL", required=False)
    HUGGINGFACE_EMBEDDING_MODEL: str = _get_env("HUGGINGFACE_EMBEDDING_MODEL", required=False)
    # Vector Store
    VECTOR_DB_PATH: str = _require_env("VECTOR_DB_PATH")
    VECTOR_COLLECTION: str = _require_env("VECTOR_COLLECTION")
    VECTOR_SIMILARITY_METRIC: str = _get_env("VECTOR_SIMILARITY_METRIC", required=False, default="cosine")

    # Retrieval
    TOP_K: int = _parse_int("TOP_K")
    MIN_RELEVANCE_SCORE: float = _parse_float("MIN_RELEVANCE_SCORE")

    # LLM
    LLM_PROVIDER_ORDER: List[str] = _parse_list("LLM_PROVIDER_ORDER")
    LLM_TEMPERATURE: float = _parse_float("LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = _parse_int("LLM_MAX_TOKENS")
    LLM_PROVIDER_TIMEOUT: int = _parse_int("LLM_PROVIDER_TIMEOUT", required=False, default=20)

    GEMINI_API_KEY: str = _get_env("GEMINI_API_KEY", required=False, default="")
    GEMINI_MODEL: str = _get_env("GEMINI_MODEL", required=False, default="gemini-2.0-flash")

    GROQ_API_KEY: str = _get_env("GROQ_API_KEY", required=False, default="")
    GROQ_MODEL: str = _get_env("GROQ_MODEL", required=False, default="llama-3.1-8b-instant")

    OPENROUTER_API_KEY: str = _get_env("OPENROUTER_API_KEY", required=False, default="")
    OPENROUTER_MODEL: str = _get_env("OPENROUTER_MODEL", required=False, default="meta-llama/llama-3.1-8b-instruct:free")

    OLLAMA_MODEL: str = _get_env("OLLAMA_MODEL", required=False, default="llama3.1:8b")
    OLLAMA_BASE_URL: str = _get_env("OLLAMA_BASE_URL", required=False, default="http://localhost:11434")

    # App
    APP_URL: str = _get_env("APP_URL", required=False, default="http://localhost:8000")
    APP_NAME: str = _get_env("APP_NAME", required=False, default="Enterprise-RAG")
    LOG_LEVEL: str = _get_env("LOG_LEVEL", required=False, default="INFO")
    UPLOAD_DIR: str = _get_env("UPLOAD_DIR", required=False, default="./uploads")

    # Guardrails
    GUARDRAIL_ENABLED: bool = _parse_bool("GUARDRAIL_ENABLED", required=False, default=True)
    GUARDRAIL_BLOCK_PII: bool = _parse_bool("GUARDRAIL_BLOCK_PII", required=False, default=True)
    GUARDRAIL_MAX_INPUT_CHARS: int = _parse_int("GUARDRAIL_MAX_INPUT_CHARS", required=False, default=5000)
    MAX_FILE_SIZE_MB: int = _parse_int("MAX_FILE_SIZE_MB", required=False, default=50)

    ALLOWED_EXTENSIONS: List[str] = _parse_list("ALLOWED_EXTENSIONS", required=False)
    SUPPORTED_EXTENSIONS: List[str] = _parse_list("SUPPORTED_EXTENSIONS", required=False)

    @classmethod
    def print_config(cls):
        print("\n" + "="*70)
        print("CONFIG (from.env)")
        print("="*70)
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k!= "print_config":
                if "API_KEY" in k or "TOKEN" in k:
                    v = "***MASKED***" if v else "NOT SET"
                print(f"{k}: {v}")
        print("="*70 + "\n")


    GENERATE_QUERY_SYSTEM_PROMPT: str = _get_env("GENERATE_QUERY_SYSTEM_PROMPT", required=False, default="You are helpful assistant with access to knowledge base.")
    GRADING_PROMPT: str = _get_env("GRADING_PROMPT", required=False, default="Question: {question}\nContext: {context}\nAre these relevant?")
    REWRITE_PROMPT: str = _get_env("REWRITE_PROMPT", required=False, default="Rewrite question: {question}")
    GENERATION_PROMPT: str = _get_env("GENERATION_PROMPT", required=False, default="Context: {context}\nQuestion: {question}")
    GUARDRAIL_PROMPT: str = _get_env("GUARDRAIL_PROMPT", required=False, default="Check: {query}")

    @classmethod
    def get_prompt(cls, name: str) -> str:
        raw = getattr(cls, name, "")
        return raw.replace("\\n", "\n")