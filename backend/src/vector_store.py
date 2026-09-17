from langchain_chroma import Chroma
from src.embeddings import get_embeddings
from src.config import Config

def get_vector_store():
    return Chroma(
        persist_directory=Config.VECTOR_DB_PATH,
        collection_name=Config.VECTOR_COLLECTION,
        embedding_function=get_embeddings()
    )

def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": Config.TOP_K * 3})