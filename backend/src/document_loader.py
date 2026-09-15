"""
document_loader.py
-------------------
LangChain-based document loading for AIDocBot.
Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)


def _load_json(path: str) -> list[Document]:
    """JSON has no single 'right' loader shape, so flatten it ourselves."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return [Document(page_content=text, metadata={"source": path})]


# Extension -> callable that returns list[Document].
# Swap/extend this to match whatever your ContentBlock pipeline covers.
LOADER_REGISTRY: dict[str, Callable[[str], list[Document]]] = {
    ".pdf": lambda p: PyPDFLoader(p).load(),
    ".docx": lambda p: Docx2txtLoader(p).load(),
    ".pptx": lambda p: UnstructuredPowerPointLoader(p).load(),
    ".xlsx": lambda p: UnstructuredExcelLoader(p, mode="elements").load(),
    ".csv": lambda p: CSVLoader(p).load(),
    ".txt": lambda p: TextLoader(p, encoding="utf-8").load(),
    ".md": lambda p: UnstructuredMarkdownLoader(p).load(),
    ".html": lambda p: UnstructuredHTMLLoader(p).load(),
    ".htm": lambda p: UnstructuredHTMLLoader(p).load(),
    ".json": _load_json,
}


def load_document(file_path: str) -> list[Document]:
    """Load a single file into LangChain Documents, tagged with format + source."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    ext = path.suffix.lower()
    loader_fn = LOADER_REGISTRY.get(ext)
    if loader_fn is None:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {sorted(LOADER_REGISTRY)}"
        )

    docs = loader_fn(str(path))
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["format"] = ext.lstrip(".")
        doc.metadata["filename"] = path.name
    return docs


def load_documents(directory: str, recursive: bool = True) -> list[Document]:
    """Load every supported file under `directory`."""
    directory_path = Path(directory)
    pattern = "**/*" if recursive else "*"
    all_docs: list[Document] = []

    for file_path in directory_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in LOADER_REGISTRY:
            try:
                all_docs.extend(load_document(str(file_path)))
            except Exception as exc:
                # Don't let one bad file kill the whole ingestion run.
                print(f"[document_loader] skipped {file_path.name}: {exc}")

    return all_docs


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap
) -> list[Document]:
    """Token-aware chunking via LangChain's recursive splitter."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
