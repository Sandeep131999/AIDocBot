"""
SRC/EVALUATOR.PY
================

Evaluate RAG retrieval quality.
Located in src/ folder as per project structure.

Metrics:
- Hit@K: Did relevant doc appear in top K?
- Precision@K: What % of top K are relevant?
- Recall@K: Did we find all relevant documents?
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain
- F1: Harmonic mean of Precision and Recall
"""

# ============================================================================
# IMPORTS
# ============================================================================

from typing import List, Dict
import numpy as np


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class Evaluator:
    """
    Evaluate retrieval quality.
    
    Measures how well the retriever performs
    using ground truth relevant documents.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        
        print("\n" + "="*70)
        print("EVALUATOR - INITIALIZATION")
        print("="*70)
        print(f"\n✅ Evaluator ready")
    
    # ========================================================================
    # HIT@K
    # ========================================================================
    
    def hit_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> int:
        """
        Count how many relevant docs in top K.
        
        Args:
            retrieved_docs: Documents retrieved by system
            relevant_docs: Ground truth relevant documents
            k: Consider top K results
        
        Returns:
            Number of hits (0 to min(k, len(relevant)))
        """
        
        hits = 0
        top_k = retrieved_docs[:k]
        
        for doc in top_k:
            if doc in relevant_docs:
                hits += 1
        
        return hits
    
    # ========================================================================
    # PRECISION@K
    # ========================================================================
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        What % of top K results are relevant?
        
        Formula: Precision@K = (# relevant in top K) / K
        
        Range: 0 to 1
        Higher is better
        """
        
        hits = self.hit_at_k(retrieved_docs, relevant_docs, k)
        precision = hits / k
        
        return precision
    
    # ========================================================================
    # RECALL@K
    # ========================================================================
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        What % of relevant docs did we find?
        
        Formula: Recall@K = (# relevant in top K) / (# total relevant)
        
        Range: 0 to 1
        Higher is better
        """
        
        if len(relevant_docs) == 0:
            return 0.0
        
        hits = self.hit_at_k(retrieved_docs, relevant_docs, k)
        recall = hits / len(relevant_docs)
        
        return recall
    
    # ========================================================================
    # MRR (Mean Reciprocal Rank)
    # ========================================================================
    
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Rank position of first relevant document.
        
        Formula: MRR = 1 / (position of first relevant)
        
        Range: 0 to 1
        Position 1 = 1.0, Position 2 = 0.5, Position 3 = 0.33
        """
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        
        return 0.0
    
    # ========================================================================
    # NDCG@K
    # ========================================================================
    
    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Ranking quality score.
        
        Penalizes relevant docs that appear lower.
        
        Range: 0 to 1
        Considers position and relevance
        """
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_docs:
                discount = 1.0 / np.log2(i + 1)
                dcg += discount
        
        # Calculate ideal DCG
        ideal_dcg = 0.0
        num_relevant = min(len(relevant_docs), k)
        for i in range(num_relevant):
            discount = 1.0 / np.log2(i + 2)
            ideal_dcg += discount
        
        if ideal_dcg == 0:
            return 0.0
        
        ndcg = dcg / ideal_dcg
        
        return ndcg
    
    # ========================================================================
    # F1 SCORE
    # ========================================================================
    
    def f1_score_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        """
        Harmonic mean of Precision and Recall.
        
        Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Range: 0 to 1
        Balances precision and recall
        """
        
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    # ========================================================================
    # EVALUATE QUERY
    # ========================================================================
    
    def evaluate_query(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> Dict[str, float]:
        """
        Evaluate all metrics for a single query.
        
        Returns:
            Dict with all metrics
        
        EXAMPLE:
            metrics = evaluator.evaluate_query(results, relevant_docs)
            
            Results:
            {
                'hit_at_k': 3,
                'precision_at_k': 0.6,
                'recall_at_k': 1.0,
                'mrr': 0.5,
                'ndcg_at_k': 0.85,
                'f1_score_at_k': 0.75
            }
        """
        
        metrics = {
            'hit_at_k': self.hit_at_k(retrieved_docs, relevant_docs, k),
            'precision_at_k': self.precision_at_k(retrieved_docs, relevant_docs, k),
            'recall_at_k': self.recall_at_k(retrieved_docs, relevant_docs, k),
            'mrr': self.mean_reciprocal_rank(retrieved_docs, relevant_docs),
            'ndcg_at_k': self.ndcg_at_k(retrieved_docs, relevant_docs, k),
            'f1_score_at_k': self.f1_score_at_k(retrieved_docs, relevant_docs, k)
        }
        
        return metrics
    
    # ========================================================================
    # EVALUATE DATASET
    # ========================================================================
    
    def evaluate_dataset(self, queries_results: Dict, ground_truth: Dict, k: int = 5) -> Dict:
        """
        Evaluate multiple queries and aggregate.
        
        Args:
            queries_results: Dict of query → retrieved_docs
            ground_truth: Dict of query → relevant_docs
            k: Top K to evaluate
        
        Returns:
            Aggregated metrics with mean, std, min, max
        """
        
        all_metrics = []
        
        for query, retrieved in queries_results.items():
            if query not in ground_truth:
                continue
            
            relevant = ground_truth[query]
            metrics = self.evaluate_query(retrieved, relevant, k)
            all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def print_metrics(self, metrics: Dict, query: str = None) -> None:
        """Pretty print metrics."""
        
        if query:
            print(f"\nQuery: '{query}'")
        
        print("\n" + "="*70)
        print("METRICS")
        print("="*70)
        
        print(f"\nHit@K: {metrics['hit_at_k']}")
        print(f"Precision@K: {metrics['precision_at_k']:.3f}")
        print(f"Recall@K: {metrics['recall_at_k']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"NDCG@K: {metrics['ndcg_at_k']:.3f}")
        print(f"F1 Score@K: {metrics['f1_score_at_k']:.3f}")


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EVALUATOR - TEST")
    print("="*70)
    
    evaluator = Evaluator()
    
    # Test data
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc6"]
    
    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    
    # Evaluate
    metrics = evaluator.evaluate_query(retrieved, relevant, k=5)
    
    evaluator.print_metrics(metrics)
    
    print("\n✅ Evaluator test complete!")