"""
EMBEDDING GENERATOR & ENCODER R&D
=================================
Generate embeddings using HuggingFace models with encoder selection.
All settings read from .env file (UTF-8 encoding).

Encoder R&D: Test multiple models and select the best based on a test set.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config


class EmbeddingGenerator:
    """
    Generate embeddings using HuggingFace models.
    
    Default: BAAI/bge-base-en-v1.5 (excellent for RAG, 768-dim)
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        self.normalize = Config.EMBEDDING_NORMALIZE
        
        print(f"\n🔤 Embedding Generator initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Normalize: {self.normalize}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": self.normalize}
        )
        print(f"   ✅ Model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings."""
        return self.embeddings.embed_documents(texts)
    
    def embed_documents(self, documents: List) -> np.ndarray:
        """Embed LangChain Document objects."""
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)


class EncoderRD:
    """
    Encoder R&D - Select the best embedding model based on a test set.
    
    TEST SET FORMAT:
    [
        {
            "query": "What is machine learning?",
            "relevant_docs": ["doc1", "doc2"],
            "irrelevant_docs": ["doc3", "doc4"]
        }
    ]
    
    METRICS:
    - MRR (Mean Reciprocal Rank)
    - NDCG@K
    - Precision@K
    """
    
    CANDIDATE_MODELS = [
        "BAAI/bge-base-en-v1.5",      # 768-dim, excellent for RAG
        "BAAI/bge-small-en-v1.5",     # 384-dim, faster
        "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, popular
        "intfloat/e5-base-v2",         # 768-dim, strong performance
        "intfloat/multilingual-e5-base",  # multilingual support
    ]
    
    def __init__(self, test_set: List[Dict] = None):
        self.test_set = test_set or []
        print(f"\n🧪 Encoder R&D initialized")
        print(f"   Test set size: {len(self.test_set)} queries")
        print(f"   Candidate models: {len(self.CANDIDATE_MODELS)}")
    
    def evaluate_model(self, model_name: str, test_set: List[Dict] = None) -> Dict:
        """
        Evaluate a single embedding model on the test set.
        Returns metrics: MRR, NDCG@5, Precision@5
        """
        test_set = test_set or self.test_set
        if not test_set:
            print("⚠️ No test set provided, returning empty metrics")
            return {"mrr": 0, "ndcg@5": 0, "precision@5": 0}
        
        print(f"\n📊 Evaluating: {model_name}")
        
        try:
            # Load model
            embedder = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": Config.EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            mrr_scores = []
            ndcg_scores = []
            precision_scores = []
            
            for test_case in test_set:
                query = test_case["query"]
                relevant = set(test_case.get("relevant_docs", []))
                all_docs = test_case.get("relevant_docs", []) + test_case.get("irrelevant_docs", [])
                
                if not all_docs:
                    continue
                
                # Embed query and documents
                query_emb = embedder.embed_query(query)
                doc_embs = embedder.embed_documents(all_docs)
                
                # Calculate similarities
                similarities = cosine_similarity([query_emb], doc_embs)[0]
                
                # Rank documents
                ranked_indices = np.argsort(similarities)[::-1]
                ranked_docs = [all_docs[i] for i in ranked_indices]
                
                # MRR
                for i, doc in enumerate(ranked_docs):
                    if doc in relevant:
                        mrr_scores.append(1.0 / (i + 1))
                        break
                else:
                    mrr_scores.append(0)
                
                # NDCG@5
                k = min(5, len(ranked_docs))
                dcg = 0
                idcg = 0
                for i in range(k):
                    rel = 1 if ranked_docs[i] in relevant else 0
                    dcg += rel / np.log2(i + 2)
                
                # Ideal DCG
                ideal_rels = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
                for i, rel in enumerate(ideal_rels):
                    idcg += rel / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
                
                # Precision@5
                top5 = ranked_docs[:5]
                hits = sum(1 for d in top5 if d in relevant)
                precision_scores.append(hits / len(top5) if top5 else 0)
            
            metrics = {
                "mrr": round(np.mean(mrr_scores), 4),
                "ndcg@5": round(np.mean(ndcg_scores), 4),
                "precision@5": round(np.mean(precision_scores), 4),
                "model": model_name
            }
            
            print(f"   MRR: {metrics['mrr']:.4f}")
            print(f"   NDCG@5: {metrics['ndcg@5']:.4f}")
            print(f"   Precision@5: {metrics['precision@5']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"mrr": 0, "ndcg@5": 0, "precision@5": 0, "error": str(e)}
    
    def run_benchmark(self, test_set: List[Dict] = None, 
                      models: List[str] = None) -> List[Dict]:
        """
        Run benchmark across all candidate models.
        Returns ranked list of models with metrics.
        """
        test_set = test_set or self.test_set
        models = models or self.CANDIDATE_MODELS
        
        print(f"\n🏆 ENCODER BENCHMARK")
        print(f"   Testing {len(models)} models on {len(test_set)} queries")
        print(f"   {'='*60}")
        
        results = []
        for model in models:
            metrics = self.evaluate_model(model, test_set)
            results.append(metrics)
        
        # Sort by NDCG@5 (primary) then MRR
        results.sort(key=lambda x: (x.get("ndcg@5", 0), x.get("mrr", 0)), reverse=True)
        
        print(f"\n📈 RANKING RESULTS:")
        print(f"   {'Rank':<6} {'Model':<40} {'NDCG@5':<10} {'MRR':<10} {'P@5':<10}")
        print(f"   {'-'*76}")
        for i, r in enumerate(results, 1):
            if "error" not in r:
                print(f"   {i:<6} {r['model']:<40} {r['ndcg@5']:<10.4f} {r['mrr']:<10.4f} {r['precision@5']:<10.4f}")
        
        return results
    
    def select_best_model(self, test_set: List[Dict] = None) -> str:
        """Select the best model from benchmark results."""
        results = self.run_benchmark(test_set)
        
        # Filter out models with errors
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            print("⚠️ No valid results, falling back to default model")
            return Config.EMBEDDING_MODEL
        
        best = valid_results[0]
        print(f"\n🏆 BEST MODEL: {best['model']}")
        print(f"   NDCG@5: {best['ndcg@5']:.4f}")
        print(f"   MRR: {best['mrr']:.4f}")
        
        return best["model"]


if __name__ == "__main__":
    # Test with sample data
    test_set = [
        {
            "query": "What is machine learning?",
            "relevant_docs": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "ML algorithms improve through experience without being explicitly programmed."
            ],
            "irrelevant_docs": [
                "The capital of France is Paris.",
                "Water boils at 100 degrees Celsius."
            ]
        },
        {
            "query": "How does deep learning work?",
            "relevant_docs": [
                "Deep learning uses neural networks with many layers to learn complex patterns.",
                "Neural networks are inspired by the structure of the human brain."
            ],
            "irrelevant_docs": [
                "Python is a popular programming language.",
                "The Earth revolves around the Sun."
            ]
        }
    ]
    
    rd = EncoderRD(test_set=test_set)
    
    # Quick test with default model
    print("\n" + "="*70)
    print("QUICK TEST - Default Model")
    print("="*70)
    gen = EmbeddingGenerator()
    text = "Machine learning is a subset of artificial intelligence."
    emb = gen.embed_text(text)
    print(f"   Embedding shape: {emb.shape}")
    print(f"   Embedding (first 5): {emb[:5]}")