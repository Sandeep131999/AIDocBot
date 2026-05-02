"""
DAY 4: VECTOR STORE SYSTEM
==========================

Store embeddings in ChromaDB vector database for fast retrieval.

This is the STORAGE LAYER of the RAG system.

WHAT THIS DOES:
1. Creates ChromaDB vector database
2. Stores document embeddings with metadata
3. Searches for similar documents instantly
4. Returns results ranked by similarity

BEFORE & AFTER:
- Without vector DB: 10,000 docs = 30+ seconds per search ❌
- With vector DB: 10,000 docs = 50ms per search ✅
"""

# ============================================================================
# IMPORTS
# ============================================================================

import chromadb
from langchain.schema import Document
from typing import List, Dict
import uuid
import os


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
        Stored in ChromaDB
                ↓
        Fast similarity search
    """
    
    def __init__(self, db_path: str = "data/chroma_db"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            db_path: Where to save database on disk
        
        WHAT HAPPENS:
        1. Creates/connects to ChromaDB at db_path
        2. Sets up persistent storage (data saved to disk)
        3. Creates collection for documents
        4. Configures cosine similarity metric
        
        EXAMPLE:
            >>> store = VectorStore()
            >>> store.print_stats()
        """
        
        self.db_path = db_path
        
        print("\n" + "="*70)
        print("VECTOR STORE - INITIALIZATION")
        print("="*70)
        print(f"\n📁 Initializing ChromaDB")
        print(f"   Storage path: {db_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        try:
            # PersistentClient = saves data to disk
            self.client = chromadb.PersistentClient(path=db_path)
            print(f"✅ ChromaDB persistent client created")
            
            # Create collection (like a table in SQL)
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            print(f"✅ Collection 'documents' ready")
            print(f"   Similarity metric: Cosine")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
    
    # ========================================================================
    # ADD DOCUMENTS TO VECTOR STORE
    # ========================================================================
    
    def add_documents(self, documents: List[Document], embeddings_list: List) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        PROCESS:
        For each (document, embedding) pair:
        1. Generate unique ID
        2. Extract text content
        3. Extract metadata
        4. Convert embedding to list
        5. Store in ChromaDB
        
        Args:
            documents: List of Document objects from DocumentLoader
                      - page_content: the text
                      - metadata: dict with source, page, etc.
            
            embeddings_list: List of embedding vectors from EmbeddingGenerator
                            Each shape (384,)
        
        EXAMPLE:
            >>> from document_loader import DocumentLoader
            >>> from embeddings import EmbeddingGenerator
            >>> from vector_store import VectorStore
            >>> 
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_text("Your text...")
            >>> 
            >>> embedder = EmbeddingGenerator()
            >>> embeddings = embedder.embed_texts([d.page_content for d in docs])
            >>> 
            >>> store = VectorStore()
            >>> store.add_documents(docs, embeddings)
            >>> print(f"Stored: {store.get_stats()['total_chunks']} chunks")
        """
        
        print(f"\n[ADD DOCUMENTS TO VECTOR STORE]")
        print(f"Adding {len(documents)} chunks...")
        
        if len(documents) == 0:
            print("⚠️  No documents to add!")
            return
        
        # Prepare lists for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings = []
        
        # Process each document-embedding pair
        for doc, emb in zip(documents, embeddings_list):
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Store metadata
            metadatas.append(doc.metadata)
            
            # Store text
            documents_text.append(doc.page_content)
            
            # Convert embedding to list
            if hasattr(emb, 'tolist'):
                embeddings.append(emb.tolist())
            else:
                embeddings.append(emb)
        
        try:
            # Add all to ChromaDB at once (faster than one-by-one)
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            total = self.collection.count()
            
            print(f"✅ Success!")
            print(f"   Added: {len(documents)} chunks")
            print(f"   Total in database: {total}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    # ========================================================================
    # SEARCH VECTOR STORE
    # ========================================================================
    
    def search(self, query_embedding, top_k: int = 5) -> List[Dict]:
        """
        Search for documents similar to query embedding.
        
        PROCESS:
        1. Take query embedding (vector)
        2. ChromaDB finds most similar document embeddings
        3. Return top_k results with similarity scores
        
        DISTANCE vs SIMILARITY:
        ChromaDB returns "distance" (how different):
        - 0 = identical
        - 1 = different
        
        We convert to "similarity":
        - 1 = identical
        - 0 = different
        
        Args:
            query_embedding: Query vector from EmbeddingGenerator
                            Shape: (384,)
            
            top_k: How many results to return (default: 5)
        
        Returns:
            List of dictionaries:
            {
                'document': str - the text
                'metadata': dict - source info
                'distance': float - 0 to 1 (lower = more similar)
                'similarity': float - 0 to 1 (higher = more similar)
                'id': str - unique ID
            }
        
        EXAMPLE:
            >>> embedder = EmbeddingGenerator()
            >>> store = VectorStore()
            >>> 
            >>> query = "How to reset password?"
            >>> query_emb = embedder.embed_text(query)
            >>> results = store.search(query_emb, top_k=5)
            >>> 
            >>> for r in results:
            ...     print(f"{r['similarity']:.3f} - {r['document'][:80]}")
        """
        
        # Convert numpy to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
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
    
    # ========================================================================
    # GET STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics.
        
        Returns:
            Dict with total chunks, path, collection name
        
        EXAMPLE:
            >>> store = VectorStore()
            >>> stats = store.get_stats()
            >>> print(stats['total_chunks'])
            24
        """
        count = self.collection.count()
        
        return {
            'total_chunks': count,
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
        print(f"Database path: {stats['db_path']}")
        print(f"Collection: {stats['collection_name']}")
        
        if stats['total_chunks'] > 0:
            print(f"Status: ✅ Ready for search\n")
        else:
            print(f"Status: ⚠️  No documents yet\n")
    
    # ========================================================================
    # CLEAR ALL
    # ========================================================================
    
    def clear(self) -> None:
        """
        Delete all documents from vector store.
        
        ⚠️  WARNING: Deletes everything!
        
        Use for: Testing, rebuilding index
        
        EXAMPLE:
            >>> store = VectorStore()
            >>> store.clear()
        """
        
        print("\n⚠️  Clearing vector store...")
        
        # Delete and recreate collection
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ Cleared!")


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("VECTOR STORE - BASIC TESTS")
    print("="*70)
    
    # Test 1: Initialize
    print("\n[TEST 1] Initialize vector store")
    print("-"*70)
    
    try:
        store = VectorStore("data/chroma_db")
        store.print_stats()
        
        print("✅ Vector store initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Test 2: Show persistence
    print("\n[TEST 2] Persistence")
    print("-"*70)
    
    print("\n📁 Vector store saves to disk at: data/chroma_db/")
    print("\nData persists across program restarts:")
    print("  1. Add documents")
    print("  2. Program ends")
    print("  3. Restart program")
    print("  4. Documents still there! ✅")
    
    print("\n✅ Ready for integration!")
