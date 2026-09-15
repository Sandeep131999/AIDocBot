"""
embeddings.py
-------------
HuggingFace Inference API embeddings + Chroma vector store setup for
AIDocBot, following the same lru_cache-retriever pattern as the
reference RAG graph.

Model: BAAI/bge-base-en-v1.5, called via the HF Inference API (no local
download, so process startup is fast) — needs HUGGINGFACEHUB_API_TOKEN.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_API_TOKEN:
    raise EnvironmentError(
        "HUGGINGFACEHUB_API_TOKEN is not set. Add it to your .env "
        "(get a token at https://huggingface.co/settings/tokens)."
    )


@lru_cache
def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """Cached client for the HF Inference API — no model weights downloaded locally."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_API_TOKEN,
    )