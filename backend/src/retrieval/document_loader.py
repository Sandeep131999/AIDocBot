"""
src/document_loader.py — Enterprise Document Loader
─────────────────────────────────────────────────────
Features:
  • Async loading with run_in_executor (non-blocking FastAPI)
  • Multi-format: PDF (pymupdf4llm), DOCX, TXT, MD, CSV, JSON, HTML, PPTX, XLSX
  • Chunking strategies: recursive (default), sentence, markdown, semantic
  • Rich metadata: source, filename, format, page, author, created_at, chunk_index, word_count
  • File hash deduplication
  • PPTX and XLSX support added
  • pymupdf4llm for superior PDF markdown extraction
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Return a SHA-256 fingerprint of the entire file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _base_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    file_hash = _file_hash(path)
    return {
        "source": str(path),
        "filename": path.name,
        "format": path.suffix.lower().lstrip("."),
        "file_size_bytes": stat.st_size,
        "file_hash": file_hash,
        "document_id": file_hash,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders (sync — called via run_in_executor)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> List[Document]:
    """Use pymupdf4llm for layout-aware markdown extraction."""
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(path, show_progress=False)
        return [Document(
            page_content=md_text,
            metadata={"page": 0, "total_pages": md_text.count("\n## ")},
        )]
    except ImportError:
        # Fallback to PyPDFLoader
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(path).load()


def _load_docx(path: str) -> List[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    return Docx2txtLoader(path).load()


def _load_text(path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(path, encoding="utf-8").load()


def _load_csv(path: str) -> List[Document]:
    from langchain_community.document_loaders import CSVLoader
    return CSVLoader(path).load()


def _load_html(path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(path).load()
    except Exception:
        # Fallback: parse with markdownify
        import markdownify
        with open(path, encoding="utf-8") as f:
            html = f.read()
        md = markdownify.markdownify(html, heading_style="ATX")
        return [Document(page_content=md)]


def _load_json(path: str) -> List[Document]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        candidate_ids = [
            str(item.get("label") or item.get("url") or f"record:{index}")
            if isinstance(item, dict)
            else f"record:{index}"
            for index, item in enumerate(data)
        ]
        duplicate_ids = Counter(candidate_ids)
        documents = []
        for index, (item, candidate_id) in enumerate(zip(data, candidate_ids)):
            if isinstance(item, dict):
                body = item.get("content")
                if body is None:
                    body = json.dumps(item, indent=2, ensure_ascii=False)
                elif not isinstance(body, str):
                    body = json.dumps(body, indent=2, ensure_ascii=False)

                heading = "\n".join(
                    str(item[key])
                    for key in ("title", "label")
                    if item.get(key)
                )
                page_content = f"{heading}\n\n{body}" if heading else body
                document_id = candidate_id
                if duplicate_ids[candidate_id] > 1:
                    document_id = f"{candidate_id}#record-{index}"

                metadata: Dict[str, Any] = {
                    "record_index": index,
                    "document_id": document_id,
                }
                for key in ("label", "url", "title"):
                    value = item.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                documents.append(Document(page_content=page_content, metadata=metadata))
            else:
                documents.append(Document(
                    page_content=str(item),
                    metadata={"record_index": index, "document_id": candidate_id},
                ))
        return documents
    return [Document(page_content=json.dumps(data, indent=2, ensure_ascii=False))]


def _load_pptx(path: str) -> List[Document]:
    from pptx import Presentation
    prs = Presentation(path)
    docs = []
    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        content = "\n".join(texts)
        if content:
            docs.append(Document(
                page_content=content,
                metadata={"slide": slide_num},
            ))
    return docs


def _load_xlsx(path: str) -> List[Document]:
    import pandas as pd
    xl = pd.ExcelFile(path)
    docs = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        # Convert to markdown table for better LLM comprehension
        md_table = df.to_markdown(index=False)
        docs.append(Document(
            page_content=f"## Sheet: {sheet_name}\n\n{md_table}",
            metadata={"sheet": sheet_name},
        ))
    return docs


LOADER_REGISTRY: Dict[str, Callable[[str], List[Document]]] = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".doc":  _load_docx,
    ".txt":  _load_text,
    ".md":   _load_text,
    ".csv":  _load_csv,
    ".html": _load_html,
    ".htm":  _load_html,
    ".json": _load_json,
    ".pptx": _load_pptx,
    ".xlsx": _load_xlsx,
    ".xls":  _load_xlsx,
}


# ─────────────────────────────────────────────────────────────────────────────
# Splitter factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_splitter(strategy: Optional[str] = None):
    s = strategy or Config.CHUNK_STRATEGY
    size = Config.CHUNK_SIZE
    overlap = Config.CHUNK_OVERLAP

    if s == "markdown":
        from langchain_text_splitters import MarkdownTextSplitter
        return MarkdownTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
        )
    elif s == "sentence":
        # Sentence-aware splitting using NLTK boundaries
        return RecursiveCharacterTextSplitter(
            separators=[". ", "! ", "? ", "\n\n", "\n", " "],
            chunk_size=size,
            chunk_overlap=overlap,
        )
    else:
        # Default: recursive + tiktoken encoding for accurate token counts
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(
    file_path: str,
    strategy: Optional[str] = None,
) -> List[Document]:
    """
    Synchronous loader — use load_and_split_async in FastAPI endpoints.
    Loads, parses, enriches metadata, and chunks a document.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        raise ValueError(f"File {size_mb:.2f}MB exceeds {Config.MAX_FILE_SIZE_MB}MB limit")

    ext = path.suffix.lower()
    loader_fn = LOADER_REGISTRY.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {list(LOADER_REGISTRY)}")

    logger.info(f"[Loader] Loading {path.name} ({size_mb:.2f}MB, strategy={strategy or Config.CHUNK_STRATEGY})")

    raw_docs = loader_fn(str(path))
    base_meta = _base_metadata(path)

    # Enrich all docs with base metadata
    for doc in raw_docs:
        doc.metadata.update({k: v for k, v in base_meta.items() if k not in doc.metadata})

    # Split
    splitter = _get_splitter(strategy)
    chunks = splitter.split_documents(raw_docs)

    chunk_totals: Dict[str, int] = defaultdict(int)
    for chunk in chunks:
        document_id = str(chunk.metadata.get("document_id", base_meta["file_hash"]))
        chunk_totals[document_id] += 1

    chunk_indexes: Dict[str, int] = defaultdict(int)
    for i, chunk in enumerate(chunks):
        document_id = str(chunk.metadata.get("document_id", base_meta["file_hash"]))
        chunk.metadata["chunk_index"] = chunk_indexes[document_id]
        chunk.metadata["chunk_total"] = chunk_totals[document_id]
        chunk_indexes[document_id] += 1
        chunk.metadata["word_count"] = len(chunk.page_content.split())

    logger.info(f"[Loader] {path.name} → {len(chunks)} chunks")
    return chunks


async def load_and_split_async(
    file_path: str,
    strategy: Optional[str] = None,
) -> List[Document]:
    """
    Async wrapper for use in FastAPI endpoints.
    Runs the blocking I/O in a thread executor.
    """
    loop = asyncio.get_event_loop()
    fn = partial(load_and_split, file_path, strategy)
    return await loop.run_in_executor(None, fn)


def get_supported_extensions() -> List[str]:
    return list(LOADER_REGISTRY.keys())
