from langchain_core.tools import create_retriever_tool
from src.vector_store import get_retriever

def get_retriever_tool():
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and return relevant documents from the knowledge base."
    )