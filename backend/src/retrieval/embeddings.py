import os
from functools import lru_cache
from src.config import Config
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

@lru_cache
def get_embeddings():
    use_ollama = os.getenv("USE_OLLAMA_EMBEDDINGS","false").lower() == "true"

    if use_ollama:
        # 100% Local, no HF token needed
       
        print("[Embeddings] Using Ollama local: nomic-embed-text")
        return OllamaEmbeddings(
            model=Config.OLLAMA_EMBEDDING_MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434")
        )
    else:
        # Cloud HF Inference API
        print(f"[Embeddings] Using HF: {Config.HUGGINGFACE_EMBEDDING_MODEL}")
        return HuggingFaceEndpointEmbeddings(
            model=Config.HUGGINGFACE_EMBEDDING_MODEL,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )