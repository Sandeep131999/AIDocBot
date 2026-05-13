"""
DAY 4: VECTOR STORE SYSTEM (FIXED FOR INCREMENTAL UPLOADS)
==========================

Store embeddings in ChromaDB vector database for fast retrieval.
FIXED: Now preserves existing data when adding new documents.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import chromadb
from langchain.schema import Document
from typing import List, Dict, Optional
import uuid
import os
import hashlib
import json


# ============================================================================
# VECTOR STORE CLASS
# ============================================================================

class VectorStore:
    """
    Manages storing and retrieving embeddings using ChromaDB.
    
    PIPELINE:
    Documents + Embeddings (from Day 2 & 3)
                ↓
            VectorStore
                ↓
        Stored in ChromaDB (PERSISTENT - survives restarts)
                ↓
        Fast similarity search
    
    KEY FIXES:
    - Uses deterministic IDs based on content hash (prevents duplicates)
    - Never clears data unless explicitly requested
    - Reuses existing collection instead of recreating
    - Checks for existing documents before adding
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            db_path: Where to save database on disk. 
                    Defaults to './chroma_db' if not provided.
        
        WHAT HAPPENS:
        1. Creates/connects to ChromaDB at db_path
        2. Uses PersistentClient (data survives program restarts)
        3. Gets existing collection or creates new one
        4. Configures cosine similarity metric
        """
        
        # Use default path if none provided
        if db_path is None:
            db_path = os.getenv("Vector_DB_Path", "./chroma_db")
        
        self.db_path = db_path
        
        print("\n" + "="*70)
        print("VECTOR STORE - INITIALIZATION")
        print("="*70)
        print(f"\n📁 Initializing ChromaDB")
        print(f"   Storage path: {os.path.abspath(db_path)}")
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        try:
            # PersistentClient = saves data to disk automatically
            self.client = chromadb.PersistentClient(path=db_path)
            print(f"✅ ChromaDB persistent client created")
            
            # Get or create collection (NEVER recreate existing one)
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            existing_count = self.collection.count()
            print(f"✅ Collection 'documents' ready")
            print(f"   Similarity metric: Cosine")
            print(f"   Existing documents: {existing_count}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
    
    # =====================================================================
    # GENERATE DETERMINISTIC ID (Prevents Duplicates)
    # =====================================================================
    
    def _generate_id(self, text: str, metadata: Dict) -> str:
        """
        Generate a deterministic unique ID based on content + source.
        
        This prevents the same document from being stored twice.
        If you upload the same file again, it will overwrite instead of duplicate.
        
        Args:
            text: The document content
            metadata: Document metadata (source, page, etc.)
        
        Returns:
            str: Deterministic hash ID
        """
        # Create a string combining content and source
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 0)
        content_hash = hashlib.md5(f"{source}:{page}:{text[:100]}".encode()).hexdigest()
        return content_hash
    
    # =====================================================================
    # SANITIZE METADATA FOR CHROMADB COMPATIBILITY
    # =====================================================================
    
    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        ChromaDB only accepts str, int, float, bool as metadata values.
        Convert lists/tuples to comma-separated strings, dicts to JSON strings.
        """
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
    
    # =====================================================================
    # ADD DOCUMENTS TO VECTOR STORE (INCREMENTAL - NO DATA LOSS)
    # =====================================================================
    
    def add_documents(self, documents: List[Document], embeddings_list: List, 
                      source_file: Optional[str] = None) -> Dict:
        """
        Add documents and their embeddings to the vector store.
        
        FIXED BEHAVIOR:
        - Appends to existing data (never deletes previous uploads)
        - Uses content-based IDs to prevent duplicates
        - If same document is added twice, it updates instead of duplicating
        
        Args:
            documents: List of Document objects from DocumentLoader
            embeddings_list: List of embedding vectors from EmbeddingGenerator
            source_file: Optional filename to tag all documents with source
        
        Returns:
            Dict with stats: {'added': int, 'updated': int, 'total': int}
        """
        
        print(f"\n[ADD DOCUMENTS TO VECTOR STORE]")
        print(f"Adding {len(documents)} chunks...")
        
        if len(documents) == 0:
            print("⚠️  No documents to add!")
            return {'added': 0, 'updated': 0, 'total': self.collection.count()}
        
        # Prepare lists for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings = []
        
        # Track what we're doing
        added_count = 0
        updated_count = 0
        
        # Process each document-embedding pair
        for doc, emb in zip(documents, embeddings_list):
            
            # Generate deterministic ID based on content
            doc_id = self._generate_id(doc.page_content, doc.metadata)
            ids.append(doc_id)
            
            # Merge source_file into metadata if provided
            metadata = dict(doc.metadata)  # Copy to avoid modifying original
            if source_file:
                metadata['source_file'] = os.path.basename(source_file)
                metadata['upload_batch'] = str(uuid.uuid4())[:8]  # Track batch
            
            # FIX: Sanitize metadata for ChromaDB compatibility
            metadata = self._sanitize_metadata(metadata)
            
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            
            # Convert embedding to list
            if hasattr(emb, 'tolist'):
                embeddings.append(emb.tolist())
            else:
                embeddings.append(emb)
        
        try:
            # Check which IDs already exist (to report correctly)
            existing_ids = set()
            try:
                existing_check = self.collection.get(ids=ids, include=[])
                existing_ids = set(existing_check['ids'])
            except Exception:
                pass  # If get fails, assume none exist
            
            # Add all to ChromaDB at once
            # ChromaDB handles duplicates by UPSERTING (updating existing)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            # Calculate stats
            for doc_id in ids:
                if doc_id in existing_ids:
                    updated_count += 1
                else:
                    added_count += 1
            
            total = self.collection.count()
            
            print(f"✅ Success!")
            print(f"   New chunks added: {added_count}")
            print(f"   Existing updated: {updated_count}")
            print(f"   Total in database: {total}")
            
            return {
                'added': added_count,
                'updated': updated_count,
                'total': total
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    # =====================================================================
    # SEARCH VECTOR STORE
    # =====================================================================
    
    def search(self, query_embedding, top_k: int = 5, 
               filter_source: Optional[str] = None) -> List[Dict]:
        """
        Search for documents similar to query embedding.
        
        Args:
            query_embedding: Query vector from EmbeddingGenerator
            top_k: How many results to return (default: 5)
            filter_source: Optional filter by source filename
        
        Returns:
            List of result dictionaries with similarity scores
        """
        
        # Convert numpy to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        try:
            # Build where filter if source specified
            where_filter = None
            if filter_source:
                where_filter = {"source_file": {"$eq": os.path.basename(filter_source)}}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
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
    
    # =====================================================================
    # GET STATISTICS
    # =====================================================================
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        count = self.collection.count()
        
        # Try to get unique sources
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
            'collection_name': 'documents'
        }
    
    def print_stats(self) -> None:
        """Print formatted statistics."""
        
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("VECTOR STORE STATISTICS")
        print("="*70)
        print(f"\nTotal document chunks: {stats['total_chunks']}")
        print(f"Unique sources: {stats['unique_sources']}")
        if stats['sources']:
            print(f"Sources: {', '.join(stats['sources'][:5])}")
            if len(stats['sources']) > 5:
                print(f"         ... and {len(stats['sources']) - 5} more")
        print(f"Database path: {stats['db_path']}")
        print(f"Collection: {stats['collection_name']}")
        
        if stats['total_chunks'] > 0:
            print(f"Status: ✅ Ready for search\n")
        else:
            print(f"Status: ⚠️  No documents yet\n")
    
    # =====================================================================
    # LIST ALL SOURCES
    # =====================================================================
    
    def list_sources(self) -> List[str]:
        """List all unique source files in the database."""
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
            print(f"❌ Error listing sources: {e}")
            return []
    
    # =====================================================================
    # DELETE DOCUMENTS BY SOURCE (Useful for re-uploading a file)
    # =====================================================================
    
    def delete_by_source(self, source_file: str) -> int:
        """
        Delete all documents from a specific source file.
        
        Use this when you want to re-upload a file (replace old version).
        
        Args:
            source_file: Filename to delete (e.g., 'document.pdf')
        
        Returns:
            int: Number of documents deleted
        """
        try:
            # Find all documents with this source
            results = self.collection.get(
                where={"source_file": {"$eq": os.path.basename(source_file)}},
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"🗑️  Deleted {len(results['ids'])} chunks from '{source_file}'")
                return len(results['ids'])
            else:
                print(f"ℹ️  No documents found from '{source_file}'")
                return 0
                
        except Exception as e:
            print(f"❌ Error deleting: {e}")
            return 0
    
    # =====================================================================
    # CLEAR ALL (USE WITH CAUTION)
    # =====================================================================
    
    def clear(self, confirm: bool = False) -> None:
        """
        Delete ALL documents from vector store.
        
        ⚠️  WARNING: This deletes everything permanently!
        
        Args:
            confirm: Must pass True to actually clear (safety measure)
        """
        if not confirm:
            print("\n⚠️  SAFETY BLOCKED: Pass confirm=True to clear all data")
            print("   store.clear(confirm=True)")
            return
        
        print("\n🚨 CLEARING ALL VECTOR STORE DATA...")
        
        # Get count before deletion
        count_before = self.collection.count()
        
        # Delete and recreate collection
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Cleared! Removed {count_before} documents.")


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("VECTOR STORE - INCREMENTAL UPLOAD TESTS")
    print("="*70)
    
    # Test 1: Initialize and check persistence
    print("\n[TEST 1] Initialize vector store")
    print("-"*70)
    
    try:
        store = VectorStore("./test_chroma_db")
        store.print_stats()
        
        print("✅ Vector store initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Test 2: Simulate multiple uploads
    print("\n[TEST 2] Simulating incremental uploads")
    print("-"*70)
    
    from  langchain.core.document import Document
    import numpy as np
    
    # Simulate JSON file 1 upload
    docs_json1 = [
        Document(page_content="User data from file 1", metadata={"source": "users.json", "type": "json"}),
        Document(page_content="More user data", metadata={"source": "users.json", "type": "json"})
    ]
    emb_json1 = [np.random.rand(384) for _ in docs_json1]
    
    result1 = store.add_documents(docs_json1, emb_json1, source_file="users.json")
    print(f"After JSON 1: {store.get_stats()['total_chunks']} total chunks")
    
    # Simulate JSON file 2 upload
    docs_json2 = [
        Document(page_content="Product data from file 2", metadata={"source": "products.json", "type": "json"}),
        Document(page_content="More products", metadata={"source": "products.json", "type": "json"})
    ]
    emb_json2 = [np.random.rand(384) for _ in docs_json2]
    
    result2 = store.add_documents(docs_json2, emb_json2, source_file="products.json")
    print(f"After JSON 2: {store.get_stats()['total_chunks']} total chunks")
    
    # Simulate PDF upload (THIS IS WHERE THE BUG WAS)
    docs_pdf = [
        Document(page_content="PDF page 1 content", metadata={"source": "report.pdf", "page": 1}),
        Document(page_content="PDF page 2 content", metadata={"source": "report.pdf", "page": 2}),
        Document(page_content="PDF page 3 content", metadata={"source": "report.pdf", "page": 3})
    ]
    emb_pdf = [np.random.rand(384) for _ in docs_pdf]
    
    result3 = store.add_documents(docs_pdf, emb_pdf, source_file="report.pdf")
    print(f"After PDF: {store.get_stats()['total_chunks']} total chunks")
    
    # Verify all data is present
    print("\n[VERIFICATION]")
    print("-"*70)
    store.print_stats()
    
    sources = store.list_sources()
    print(f"\nAll sources in DB: {sources}")
    
    # Test 3: Re-upload same file (should update, not duplicate)
    print("\n[TEST 3] Re-uploading same PDF (should update, not duplicate)")
    print("-"*70)
    
    result4 = store.add_documents(docs_pdf, emb_pdf, source_file="report.pdf")
    print(f"After re-upload: {store.get_stats()['total_chunks']} total chunks")
    print(f"Expected: Same count (updated, not duplicated)")
    
    # Cleanup test DB
    print("\n[Cleanup]")
    store.clear(confirm=True)