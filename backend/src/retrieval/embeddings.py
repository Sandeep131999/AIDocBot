from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config


@lru_cache
def get_embeddings():
    """Load the compact, free sentence-transformer model once per process."""
    return HuggingFaceEmbeddings(
        model_name=Config.HUGGINGFACE_EMBEDDING_MODEL,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": Config.EMBEDDING_BATCH_SIZE},
    )