"""
SRC/RETRIEVER.PY
================

Main retriever component combining all RAG parts.
Located in src/ folder as per project structure.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
from typing import List, Dict
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearcher


# ============================================================================
# RETRIEVER CLASS
# ============================================================================

class Retriever:
    """
    Unified RAG retriever with relevance thresholding.
    
    Combines:
    - DocumentLoader (load files)
    - EmbeddingGenerator (create vectors)
    - VectorStore (semantic search)
    - KeywordSearcher (exact matches)
    
    Into one interface with hybrid search.
    """
    
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3,
                 min_relevance_score: float = 0.35):
        """
        Initialize retriever.
        
        Args:
            vector_weight: Weight for vector search (0.7 = 70%)
            keyword_weight: Weight for keyword search (0.3 = 30%)
            min_relevance_score: Minimum combined score to return results
        """
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.min_relevance_score = min_relevance_score
        
        print("\n" + "="*70)
        print("RETRIEVER - INITIALIZATION")
        print("="*70)
        print(f"\nInitializing components...")
        
        try:
            self.document_loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
            self.embeddings_generator = EmbeddingGenerator()
            self.vector_store = VectorStore()
            self.keyword_searcher = KeywordSearcher()
            
            self.indexed = False
            
            print(f"\n✅ All components initialized")
            print(f"   Vector weight: {vector_weight*100:.0f}%")
            print(f"   Keyword weight: {keyword_weight*100:.0f}%")
            print(f"   Min relevance score: {min_relevance_score}")
            
        except Exception as e:
            print(f"\n❌ Error initializing retriever: {e}")
            raise
    
    # ========================================================================
    # LOAD AND INDEX
    # ========================================================================
    
    def load_and_index_documents(self, file_path: str) -> int:
        """
        Load documents from file and index.
        
        PROCESS:
        1. Load document chunks
        2. Generate embeddings
        3. Index in vector store
        4. Index for keyword search
        
        Args:
            file_path: Path to document (e.g., "data/password_guide.txt")
        
        Returns:
            Number of chunks indexed
        
        EXAMPLE:
            >>> retriever = Retriever()
            >>> num = retriever.load_and_index_documents("data/doc.txt")
            >>> print(f"Indexed {num} chunks")
        """
        
        print(f"\n" + "="*70)
        print("[LOAD AND INDEX DOCUMENTS]")
        print("="*70)
        
        # Step 1: Load documents
        print(f"\n[STEP 1] Load documents from: {file_path}")
        print("-"*70)
        
        try:
            docs = self.document_loader.load_from_file(file_path)
            print(f"✅ Loaded {len(docs)} chunks")
            
            # Debug: show sample chunks
            for i, doc in enumerate(docs[:3]):
                print(f"   Chunk {i+1}: {len(doc.page_content)} chars | "
                      f"metadata: {doc.metadata}")
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        # Step 2: Create embeddings
        print(f"\n[STEP 2] Generate embeddings")
        print("-"*70)
        
        try:
            chunk_texts = [doc.page_content for doc in docs]
            embeddings = self.embeddings_generator.embed_texts(chunk_texts)
            print(f"✅ Created embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        # Step 3: Index in vector store
        print(f"\n[STEP 3] Index in vector store")
        print("-"*70)
        
        try:
            self.vector_store.add_documents(docs, embeddings)
            print(f"✅ Indexed in vector store")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        # Step 4: Index for keyword search
        print(f"\n[STEP 4] Index for keyword search")
        print("-"*70)
        
        try:
            self.keyword_searcher.index_documents(chunk_texts)
            print(f"✅ Indexed for keyword search")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
        
        self.indexed = True
        
        print(f"\n✅ INDEXING COMPLETE!")
        print(f"   Total chunks: {len(docs)}")
        
        return len(docs)
    
    # ========================================================================
    # RETRIEVE (HYBRID)
    # ========================================================================
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using hybrid search with relevance threshold.
        
        PROCESS:
        1. Embed query
        2. Vector search (semantic)
        3. Keyword search (exact)
        4. Normalize scores
        5. Blend: 0.7*vector + 0.3*keyword
        6. Filter by minimum relevance score
        7. Rank and return top K
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of results with scores (empty if none above threshold)
        
        EXAMPLE:
            >>> results = retriever.retrieve("How to reset password?", top_k=5)
            >>> for r in results:
            ...     print(f"{r['combined_score']:.3f} - {r['document']}")
        """
        
        if not self.indexed:
            raise ValueError("Must call load_and_index_documents() first!")
        
        print(f"\n[RETRIEVE] Query: '{query}'")
        
        # Step 1: Embed query
        query_embedding = self.embeddings_generator.embed_text(query)
        
        # Step 2: Vector search
        vector_results = self.vector_store.search(query_embedding, top_k=top_k*3)
        
        # Step 3: Keyword search
        keyword_results = self.keyword_searcher.search(query, top_k=top_k*3)
        
        # Step 4: Combine
        combined = {}
        
        # Add vector results
        for result in vector_results:
            doc = result['document']
            if doc not in combined:
                combined[doc] = {
                    'vector_score': 0,
                    'keyword_score': 0,
                    'metadata': result['metadata']
                }
            combined[doc]['vector_score'] = result['similarity']
        
        # Add keyword results (normalize)
        max_keyword = max([score for _, score in keyword_results], default=1)
        
        for doc, score in keyword_results:
            normalized_score = score / max(max_keyword, 1.0)
            
            if doc not in combined:
                combined[doc] = {
                    'vector_score': 0,
                    'keyword_score': 0,
                    'metadata': {}
                }
            combined[doc]['keyword_score'] = normalized_score
        
        # Step 5: Blend scores
        final_results = []
        
        for doc, scores in combined.items():
            combined_score = (
                self.vector_weight * scores['vector_score'] +
                self.keyword_weight * scores['keyword_score']
            )
            
            final_results.append({
                'document': doc,
                'combined_score': float(combined_score),
                'vector_score': float(scores['vector_score']),
                'keyword_score': float(scores['keyword_score']),
                'metadata': scores['metadata']
            })
        
        # Step 6: Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Step 7: FILTER by minimum relevance threshold
        filtered = [r for r in final_results if r['combined_score'] >= self.min_relevance_score]
        
        print(f"   Raw results: {len(final_results)} | After threshold ({self.min_relevance_score}): {len(filtered)}")
        
        if filtered:
            print(f"   Top result score: {filtered[0]['combined_score']:.3f}")
            print(f"   Top result name: {filtered[0]['metadata'].get('employee_name', 'N/A')}")
        
        return filtered[:top_k]
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Retrieve for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Results per query
        
        Returns:
            Dict mapping query → results
        """
        
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k=top_k)
        
        return results


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("RETRIEVER - TEST")
    print("="*70)
    
    retriever = Retriever()
    
    try:
        retriever.load_and_index_documents("data/password_guide.txt")
    except FileNotFoundError:
        print("Create data/password_guide.txt first")
        exit(1)
    
    results = retriever.retrieve("How to reset password?", top_k=5)
    
    print(f"\n\nTop 5 results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['combined_score']:.3f}")
        print(f"   Document: {result['document'][:80]}...\n")
    
    print("✅ Retriever test complete!")