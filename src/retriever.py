"""
RETRIEVER - UNIFIED RAG SYSTEM
==============================
Main retriever combining all RAG components:
- Chunking R&D
- Encoder R&D  
- Query Rewriting (Multi-LLM: Gemini → Groq → OpenRouter)
- Hybrid Search (Vector + Keyword)
- Re-ranking (Multi-LLM)

All settings read from .env file (UTF-8 encoding).
"""

import os
import numpy as np
from typing import List, Dict, Optional

from langchain_core.documents import Document

from src.config import Config
from src.document_loader import DocumentLoader, ChunkingRD
from src.embeddings import EmbeddingGenerator, EncoderRD
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearcher
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.multi_llm import MultiLLM, get_multi_llm


class Retriever:
    """
    Unified RAG Retriever with all advanced features and Multi-LLM fallback.
    
    PIPELINE:
    1. Load & Chunk documents (Chunking R&D)
    2. Generate embeddings (Encoder R&D)
    3. Index in vector store + keyword search
    4. Query rewriting (Multi-LLM: Gemini → Groq → OpenRouter)
    5. Hybrid search (Vector + Keyword)
    6. Re-ranking (Multi-LLM)
    7. Return top results
    """
    
    def __init__(self, 
                 chunking_strategy: str = None,
                 embedding_model: str = None,
                 vector_weight: float = None,
                 keyword_weight: float = None,
                 min_relevance_score: float = None,
                 enable_query_rewrite: bool = None,
                 enable_rerank: bool = None,
                 rerank_method: str = "pointwise",
                 multi_llm: MultiLLM = None):
        """
        Initialize the unified retriever with Multi-LLM.
        
        All parameters default to .env values if not specified.
        """
        
        print("\n" + "="*70)
        print("UNIFIED RAG RETRIEVER - INITIALIZATION")
        print("="*70)
        
        # Chunking R&D
        self.chunking_rd = ChunkingRD(
            strategy=chunking_strategy or Config.CHUNK_STRATEGY,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Document Loader
        self.document_loader = DocumentLoader(chunking_rd=self.chunking_rd)
        
        # Encoder R&D
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embeddings_generator = EmbeddingGenerator(model_name=self.embedding_model)
        
        # Vector Store
        self.vector_store = VectorStore()
        
        # Keyword Search
        self.keyword_searcher = KeywordSearcher()
        
        # Multi-LLM (shared across query rewriter and reranker)
        self.multi_llm = multi_llm or get_multi_llm()
        
        # Query Rewriting (uses Multi-LLM)
        self.enable_query_rewrite = enable_query_rewrite if enable_query_rewrite is not None else Config.QUERY_REWRITING_ENABLED
        self.query_rewriter = QueryRewriter(multi_llm=self.multi_llm)
        
        # Re-ranking (uses Multi-LLM)
        self.enable_rerank = enable_rerank if enable_rerank is not None else Config.RERANKER_ENABLED
        self.reranker = Reranker(multi_llm=self.multi_llm)
        self.rerank_method = rerank_method
        
        # Search weights
        self.vector_weight = vector_weight or Config.VECTOR_WEIGHT
        self.keyword_weight = keyword_weight or Config.KEYWORD_WEIGHT
        self.min_relevance_score = min_relevance_score or Config.MIN_RELEVANCE_SCORE
        
        self.indexed = False
        
        print(f"\n✅ All components initialized")
        print(f"   Chunking: {self.chunking_rd.strategy}")
        print(f"   Embedding: {self.embedding_model}")
        print(f"   Vector weight: {self.vector_weight*100:.0f}%")
        print(f"   Keyword weight: {self.keyword_weight*100:.0f}%")
        print(f"   Min relevance: {self.min_relevance_score}")
        print(f"   Query rewrite: {self.enable_query_rewrite}")
        print(f"   Re-ranking: {self.enable_rerank} ({self.rerank_method})")
        print(f"   Multi-LLM: {self.multi_llm.provider_order}")
    
    # =====================================================================
    # LOAD AND INDEX
    # =====================================================================
    
    def load_and_index_documents(self, file_path: str) -> int:
        """
        Load documents from file and index them.
        
        PROCESS:
        1. Load & chunk documents
        2. Generate embeddings
        3. Index in vector store
        4. Index for keyword search
        
        Returns:
            Number of chunks indexed
        """
        
        print(f"\n{'='*70}")
        print("[LOAD AND INDEX DOCUMENTS]")
        print(f"{'='*70}")
        
        # Step 1: Load & chunk
        print(f"\n[STEP 1] Load & Chunk: {file_path}")
        docs = self.document_loader.load_from_file(file_path)
        print(f"   ✅ Loaded {len(docs)} chunks")
        
        # Step 2: Generate embeddings
        print(f"\n[STEP 2] Generate Embeddings")
        chunk_texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings_generator.embed_texts(chunk_texts)
        
        # FIX: Convert list to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        print(f"   ✅ Created embeddings: {embeddings.shape}")
        
        # Step 3: Index in vector store
        print(f"\n[STEP 3] Index in Vector Store")
        self.vector_store.add_documents(docs, embeddings, source_file=file_path)
        
        # Step 4: Index for keyword search
        print(f"\n[STEP 4] Index for Keyword Search")
        self.keyword_searcher.index_documents(chunk_texts)
        
        self.indexed = True
        
        print(f"\n✅ INDEXING COMPLETE! Total chunks: {len(docs)}")
        return len(docs)
    
    # =====================================================================
    # RETRIEVE (FULL PIPELINE)
    # =====================================================================
    
    def retrieve(self, query: str, top_k: int = None, 
                 use_query_rewrite: bool = None,
                 use_rerank: bool = None,
                 use_hyde: bool = False) -> List[Dict]:
        """
        Full retrieval pipeline with all features and Multi-LLM fallback.
        
        PIPELINE:
        1. Query Rewriting (Multi-LLM: Gemini → Groq → OpenRouter)
        2. Hybrid Search (Vector + Keyword)
        3. Re-ranking (Multi-LLM)
        4. Return top K
        
        Args:
            query: User question
            top_k: Number of results (default from .env)
            use_query_rewrite: Override .env setting
            use_rerank: Override .env setting
            use_hyde: Use Hypothetical Document Embeddings
        
        Returns:
            List of ranked results with scores and metadata
        """
        
        if not self.indexed:
            raise ValueError("Must call load_and_index_documents() first!")
        
        top_k = top_k or Config.TOP_K
        use_query_rewrite = use_query_rewrite if use_query_rewrite is not None else self.enable_query_rewrite
        use_rerank = use_rerank if use_rerank is not None else self.enable_rerank
        
        print(f"\n{'='*70}")
        print("[RETRIEVE - FULL PIPELINE]")
        print(f"{'='*70}")
        print(f"   Original Query: '{query}'")
        
        # Step 1: Query Rewriting
        search_queries = [query]
        rewrite_metadata = {}
        
        if use_query_rewrite and self.query_rewriter.llm:
            try:
                rewrite_result = self.query_rewriter.full_pipeline(query, use_hyde=use_hyde)
                search_queries = rewrite_result["final_queries"]
                rewrite_metadata = {
                    "optimized": rewrite_result.get("optimized", query),
                    "variations": rewrite_result.get("variations", []),
                    "key_entities": rewrite_result.get("key_entities", []),
                    "hyde_document": rewrite_result.get("hyde_document")
                }
                print(f"\n   🔄 Rewritten to {len(search_queries)} queries")
                print(f"   📝 Optimized: '{rewrite_metadata.get('optimized', query)}'")
                print(f"   🔑 Key entities: {rewrite_metadata.get('key_entities', [])}")
                if rewrite_metadata.get('hyde_document'):
                    print(f"   🤖 HyDE: Generated ({len(rewrite_metadata['hyde_document'])} chars)")
            except Exception as e:
                print(f"   ⚠️ Query rewriting failed: {e}. Using original query.")
                search_queries = [query]
        
        # Step 2: Hybrid Search for each query
        all_results = {}
        
        for search_query in search_queries:
            print(f"\n   🔍 Searching: '{search_query}'")
            
            # Vector search
            query_embedding = self.embeddings_generator.embed_text(search_query)
            vector_results = self.vector_store.search(query_embedding, top_k=top_k*3)
            
            # Keyword search
            keyword_results = self.keyword_searcher.search(search_query, top_k=top_k*3)
            
            # Combine
            for result in vector_results:
                doc = result['document']
                if doc not in all_results:
                    all_results[doc] = {
                        'vector_score': 0,
                        'keyword_score': 0,
                        'metadata': result['metadata'],
                        'document': doc
                    }
                all_results[doc]['vector_score'] = max(
                    all_results[doc]['vector_score'], 
                    result['similarity']
                )
            
            max_keyword = max([score for _, score in keyword_results], default=1)
            for doc, score in keyword_results:
                if doc not in all_results:
                    all_results[doc] = {
                        'vector_score': 0,
                        'keyword_score': 0,
                        'metadata': {},
                        'document': doc
                    }
                all_results[doc]['keyword_score'] = max(
                    all_results[doc]['keyword_score'],
                    score / max(max_keyword, 1.0)
                )
        
        # Blend scores
        final_results = []
        for doc, scores in all_results.items():
            combined_score = (
                self.vector_weight * scores['vector_score'] +
                self.keyword_weight * scores['keyword_score']
            )
            final_results.append({
                'document': doc,
                'combined_score': float(combined_score),
                'vector_score': float(scores['vector_score']),
                'keyword_score': float(scores['keyword_score']),
                'metadata': scores['metadata'],
                'query_rewrite': rewrite_metadata  # Include rewrite info
            })
        
        # Sort and filter
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        filtered = [r for r in final_results if r['combined_score'] >= self.min_relevance_score]
        
        print(f"\n   📊 Raw: {len(final_results)} | Filtered (≥{self.min_relevance_score}): {len(filtered)}")
        
        # Step 3: Re-ranking
        if use_rerank and self.reranker.llm and filtered:
            try:
                print(f"\n   🎯 Re-ranking with {self.rerank_method} method...")
                filtered = self.reranker.rerank(query, filtered, method=self.rerank_method)
                print(f"   ✅ Re-ranking complete")
            except Exception as e:
                print(f"   ⚠️ Re-ranking failed: {e}. Using hybrid scores.")
        
        # Add rewrite metadata to top result for transparency
        results = filtered[:top_k]
        if results and rewrite_metadata:
            results[0]['query_rewrite'] = rewrite_metadata
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve for multiple queries."""
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k=top_k)
        return results
    
    # =====================================================================
    # R&D FEATURES
    # =====================================================================
    
    def compare_chunking_strategies(self, file_path: str) -> Dict:
        """Compare all chunking strategies on a file."""
        print(f"\n{'='*70}")
        print("CHUNKING R&D - STRATEGY COMPARISON")
        print(f"{'='*70}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        doc = Document(page_content=text, metadata={"source": file_path})
        rd = ChunkingRD(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        return rd.compare_strategies([doc])
    
    def benchmark_encoders(self, test_set: List[Dict]) -> List[Dict]:
        """Run encoder benchmark on test set."""
        print(f"\n{'='*70}")
        print("ENCODER R&D - BENCHMARK")
        print(f"{'='*70}")
        
        rd = EncoderRD(test_set=test_set)
        return rd.run_benchmark()
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "chunking_strategy": self.chunking_rd.strategy,
            "embedding_model": self.embedding_model,
            "query_rewrite_enabled": self.enable_query_rewrite,
            "rerank_enabled": self.enable_rerank,
            "multi_llm": self.multi_llm.get_provider_status()
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RETRIEVER - FULL SYSTEM TEST (Multi-LLM)")
    print("="*70)
    
    retriever = Retriever()
    
    # Create test document
    test_doc_path = "data/test_doc.txt"
    os.makedirs("data", exist_ok=True)
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write("""
Password Reset Guide
====================

How to Reset Your Password
---------------------------
If you forgot your password, follow these steps:
1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your inbox for reset link
5. Click the link and create a new password

Two-Factor Authentication
-------------------------
Enable 2FA for extra security:
1. Go to Security Settings
2. Click "Enable 2FA"
3. Scan QR code with authenticator app
4. Enter verification code
5. Save backup codes

Security Best Practices
-----------------------
- Use strong passwords (12+ characters)
- Enable two-factor authentication
- Never share your password
- Use a password manager
- Change passwords regularly
        """)
    
    try:
        # Index
        retriever.load_and_index_documents(test_doc_path)
        
        # Retrieve with query rewrite and re-ranking
        results = retriever.retrieve(
            "How do I reset my password?",
            top_k=3,
            use_query_rewrite=True,
            use_rerank=True,
            use_hyde=False
        )
        
        print(f"\n📋 Results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. Score: {r.get('final_score', r['combined_score']):.3f} - {r['document'][:100]}...")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")