from pathlib import Path
import json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader
)
from src.config import Config
import os

def _load_json(path: str) -> List[Document]:
    """Enterprise JSON loader - flattens any structure."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle list of objects or single object
    if isinstance(data, list):
        docs = []
        for i, item in enumerate(data):
            text = json.dumps(item, indent=2, ensure_ascii=False) if isinstance(item, dict) else str(item)
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "format": "json", "chunk_index": i, "filename": Path(path).name}
            ))
        return docs
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return [Document(page_content=text, metadata={"source": path, "format": "json", "filename": Path(path).name})]

LOADER_REGISTRY = {
    ".pdf": lambda p: PyPDFLoader(p).load(),
    ".docx": lambda p: Docx2txtLoader(p).load(),
    ".txt": lambda p: TextLoader(p, encoding="utf-8").load(),
    ".md": lambda p: TextLoader(p, encoding="utf-8").load(),
    ".csv": lambda p: CSVLoader(p).load(),
    ".html": lambda p: UnstructuredHTMLLoader(p).load(),
    ".htm": lambda p: UnstructuredHTMLLoader(p).load(),
    ".json": _load_json,
}

def load_and_split(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(file_path)

    # Enterprise Guard: file size
    size_mb = path.stat().st_size / (1024*1024)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        raise ValueError(f"File {size_mb:.2f}MB exceeds limit {Config.MAX_FILE_SIZE_MB}MB")

    ext = path.suffix.lower()
    if ext not in LOADER_REGISTRY:
        raise ValueError(f"Unsupported {ext}. Allowed: {list(LOADER_REGISTRY.keys())}")

    docs = LOADER_REGISTRY[ext](str(path))

    # Tag metadata enterprise standard
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["filename"] = path.name
        doc.metadata["format"] = ext.lstrip(".")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)