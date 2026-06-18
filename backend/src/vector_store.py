"""
VECTOR STORE
============
ChromaDB-based vector store with incremental uploads.
All settings read from .env file (UTF-8 encoding).

FIX: Unique IDs prevent DuplicateIDError on re-indexing or duplicate content.
"""

import chromadb
from langchain_core.documents import Document
from typing import List, Dict, Optional
import uuid
import os
import hashlib
import json
import numpy as np

from src.config import Config


class VectorStore:
    """
    Manages storing and retrieving embeddings using ChromaDB.

    KEY FIXES:
    - Uses UNIQUE IDs (content hash + index + random suffix) — prevents duplicates
    - Never clears data unless explicitly requested
    - Reuses existing collection instead of recreating
    """

    def __init__(self, db_path: str = None, collection_name: str = None):
        self.db_path = db_path or Config.VECTOR_DB_PATH
        self.collection_name = collection_name or Config.VECTOR_COLLECTION

        print(f"\n📁 Initializing ChromaDB")
        print(f"   Storage path: {os.path.abspath(self.db_path)}")

        os.makedirs(self.db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_path)
        print(f"   ✅ Persistent client created")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": Config.VECTOR_SIMILARITY_METRIC}
        )

        existing_count = self.collection.count()
        print(f"   ✅ Collection '{self.collection_name}' ready")
        print(f"   Similarity metric: {Config.VECTOR_SIMILARITY_METRIC}")
        print(f"   Existing documents: {existing_count}")

    def _generate_id(self, text: str, metadata: Dict, index: int = 0) -> str:
        """
        Generate a UNIQUE ID for each chunk.

        FIX: Includes chunk index + random UUID suffix to guarantee uniqueness,
        even when identical content appears in multiple chunks or files.
        """
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 0)
        chunk_idx = metadata.get('chunk_index', index)

        # Deterministic base
        base = f"{source}:{page}:{chunk_idx}:{text[:50]}"
        content_hash = hashlib.md5(base.encode('utf-8')).hexdigest()

        # Random suffix guarantees uniqueness even on re-upload
        random_suffix = uuid.uuid4().hex[:8]

        return f"{content_hash}_{random_suffix}"

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """ChromaDB only accepts str, int, float, bool as metadata values."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def add_documents(self, documents: List[Document], embeddings_list: List, 
                      source_file: Optional[str] = None) -> Dict:
        """
        Add documents and embeddings to vector store (incremental).

        FIX: Generates truly unique IDs so identical content across chunks
        or re-uploads never causes DuplicateIDError.
        """

        print(f"\n[ADD DOCUMENTS] Adding {len(documents)} chunks...")

        if len(documents) == 0:
            return {'added': 0, 'updated': 0, 'total': self.collection.count()}

        ids = []
        metadatas = []
        documents_text = []
        embeddings = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
            # FIX: Pass index to ensure unique ID even for duplicate content
            doc_id = self._generate_id(doc.page_content, doc.metadata, index=i)
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            if source_file:
                metadata['source_file'] = os.path.basename(source_file)
                metadata['upload_batch'] = str(uuid.uuid4())[:8]

            metadata = self._sanitize_metadata(metadata)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)

            if hasattr(emb, 'tolist'):
                embeddings.append(emb.tolist())
            else:
                embeddings.append(emb)

        try:
            # Upsert — ChromaDB handles both insert and update
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )

            total = self.collection.count()
            print(f"   ✅ Added/Updated: {len(ids)} chunks | Total in DB: {total}")

            return {'added': len(ids), 'updated': 0, 'total': total}

        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise

    def search(self, query_embedding, top_k: int = None, 
               filter_source: Optional[str] = None) -> List[Dict]:
        """Search for similar documents."""

        top_k = top_k or Config.TOP_K

        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        try:
            where_filter = None
            if filter_source:
                where_filter = {"source_file": {"$eq": os.path.basename(filter_source)}}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )

            formatted_results = []

            if not results['documents'][0]:
                return []

            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance

                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity': similarity,
                    'id': results['ids'][0][i]
                })

            return formatted_results

        except Exception as e:
            print(f"❌ Error searching: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        count = self.collection.count()

        sources = set()
        if count > 0:
            try:
                all_meta = self.collection.get(include=['metadatas'])
                for meta in all_meta['metadatas']:
                    if meta and 'source' in meta:
                        sources.add(meta['source'])
                    if meta and 'source_file' in meta:
                        sources.add(meta['source_file'])
            except Exception:
                pass

        return {
            'total_chunks': count,
            'unique_sources': len(sources),
            'sources': list(sources),
            'db_path': self.db_path,
            'collection_name': self.collection_name
        }

    def list_sources(self) -> List[str]:
        """List all unique source files."""
        try:
            all_data = self.collection.get(include=['metadatas'])
            sources = set()
            for meta in all_data['metadatas']:
                if meta:
                    if 'source' in meta:
                        sources.add(meta['source'])
                    if 'source_file' in meta:
                        sources.add(meta['source_file'])
            return sorted(list(sources))
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

    def delete_by_source(self, source_file: str) -> int:
        """Delete all documents from a specific source."""
        try:
            results = self.collection.get(
                where={"source_file": {"$eq": os.path.basename(source_file)}},
                include=[]
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"🗑️  Deleted {len(results['ids'])} chunks from '{source_file}'")
                return len(results['ids'])
            else:
                print(f"ℹ️  No documents from '{source_file}'")
                return 0

        except Exception as e:
            print(f"❌ Error: {e}")
            return 0

    def clear(self, confirm: bool = False) -> None:
        """Delete ALL documents (use with caution)."""
        if not confirm:
            print("⚠️  Pass confirm=True to clear")
            return

        count_before = self.collection.count()
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": Config.VECTOR_SIMILARITY_METRIC}
        )
        print(f"✅ Cleared! Removed {count_before} documents.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VECTOR STORE TEST")
    print("="*70)

    store = VectorStore("./test_chroma_db")
    print(f"\nStats: {store.get_stats()}")