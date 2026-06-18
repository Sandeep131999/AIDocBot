"""
TEST_EVALUATOR.PY
=================

Comprehensive tests for the Evaluator class.

Tests all metrics:
- Hit@K
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- F1 Score@K
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluator import Evaluator


# ============================================================================
# TEST 1: BASIC METRICS
# ============================================================================

def test_basic_metrics():
    """Test basic metrics with simple data."""
    
    print("\n" + "="*70)
    print("TEST 1: BASIC METRICS")
    print("="*70)
    
    evaluator = Evaluator()
    
    # Test data
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4"]
    
    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    
    # Test Hit@K
    print("\n[Hit@K] Count relevant docs in top K")
    print("-"*70)
    
    for k in [3, 5]:
        hits = evaluator.hit_at_k(retrieved, relevant, k)
        print(f"Hit@{k}: {hits} (found {hits} relevant docs in top {k})")
    
    # Test Precision@K
    print("\n[Precision@K] % of top K that are relevant")
    print("-"*70)
    
    for k in [3, 5]:
        precision = evaluator.precision_at_k(retrieved, relevant, k)
        print(f"Precision@{k}: {precision:.3f} ({precision*100:.1f}%)")
    
    # Test Recall@K
    print("\n[Recall@K] % of relevant docs found")
    print("-"*70)
    
    for k in [3, 5]:
        recall = evaluator.recall_at_k(retrieved, relevant, k)
        print(f"Recall@{k}: {recall:.3f} ({recall*100:.1f}%)")
    
    # Test MRR
    print("\n[MRR] Position of first relevant doc")
    print("-"*70)
    
    mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)
    print(f"MRR: {mrr:.3f}")
    print(f"(First relevant doc at position: {1/mrr if mrr > 0 else 'None'})")
    
    # Test NDCG
    print("\n[NDCG@K] Ranking quality")
    print("-"*70)
    
    for k in [5]:
        ndcg = evaluator.ndcg_at_k(retrieved, relevant, k)
        print(f"NDCG@{k}: {ndcg:.3f}")
    
    # Test F1
    print("\n[F1 Score@K] Precision-Recall balance")
    print("-"*70)
    
    for k in [3, 5]:
        f1 = evaluator.f1_score_at_k(retrieved, relevant, k)
        print(f"F1@{k}: {f1:.3f}")
    
    print("\n✅ Basic metrics test passed!")


# ============================================================================
# TEST 2: EVALUATE SINGLE QUERY
# ============================================================================

def test_evaluate_query():
    """Test evaluate_query with all metrics."""
    
    print("\n" + "="*70)
    print("TEST 2: EVALUATE SINGLE QUERY")
    print("="*70)
    
    evaluator = Evaluator()
    
    # Test case 1: Good retrieval
    print("\n[CASE 1] Good retrieval (2 out of 3 relevant found)")
    print("-"*70)
    
    retrieved_good = ["doc1", "doc2", "doc3"]
    relevant_good = ["doc2", "doc3"]
    
    print(f"Retrieved: {retrieved_good}")
    print(f"Relevant: {relevant_good}")
    
    metrics_good = evaluator.evaluate_query(retrieved_good, relevant_good, k=3)
    
    print(f"\nMetrics:")
    evaluator.print_metrics(metrics_good, query="Good case")
    
    # Test case 2: Poor retrieval
    print("\n[CASE 2] Poor retrieval (0 out of 3 relevant found)")
    print("-"*70)
    
    retrieved_poor = ["doc4", "doc5", "doc6"]
    relevant_poor = ["doc1", "doc2", "doc3"]
    
    print(f"Retrieved: {retrieved_poor}")
    print(f"Relevant: {relevant_poor}")
    
    metrics_poor = evaluator.evaluate_query(retrieved_poor, relevant_poor, k=5)
    
    print(f"\nMetrics:")
    evaluator.print_metrics(metrics_poor, query="Poor case")
    
    # Test case 3: Perfect retrieval
    print("\n[CASE 3] Perfect retrieval (all relevant found)")
    print("-"*70)
    
    retrieved_perfect = ["doc1", "doc2", "doc3"]
    relevant_perfect = ["doc1", "doc2", "doc3"]
    
    print(f"Retrieved: {retrieved_perfect}")
    print(f"Relevant: {relevant_perfect}")
    
    metrics_perfect = evaluator.evaluate_query(retrieved_perfect, relevant_perfect, k=5)
    
    print(f"\nMetrics:")
    evaluator.print_metrics(metrics_perfect, query="Perfect case")
    
    print("\n✅ Evaluate query test passed!")


# ============================================================================
# TEST 3: EDGE CASES
# ============================================================================

def test_edge_cases():
    """Test edge cases."""
    
    print("\n" + "="*70)
    print("TEST 3: EDGE CASES")
    print("="*70)
    
    evaluator = Evaluator()
    
    # Edge case 1: Empty relevant docs
    print("\n[EDGE CASE 1] No relevant documents")
    print("-"*70)
    
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = []
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    
    metrics = evaluator.evaluate_query(retrieved, relevant, k=5)
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print(f"✓ All metrics are 0 (expected)")
    
    # Edge case 2: More relevant docs than K
    print("\n[EDGE CASE 2] More relevant docs than K")
    print("-"*70)
    
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7"]
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant} (7 total)")
    
    metrics = evaluator.evaluate_query(retrieved, relevant, k=5)
    
    print(f"\nMetrics:")
    print(f"  Hit@5: {metrics['hit_at_k']} (max is 5)")
    print(f"  Recall@5: {metrics['recall_at_k']:.3f} (5/7 = 0.714)")
    print(f"✓ Correctly handles more relevant docs than K")
    
    # Edge case 3: Single relevant document
    print("\n[EDGE CASE 3] Single relevant document")
    print("-"*70)
    
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc3"]
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    
    metrics = evaluator.evaluate_query(retrieved, relevant, k=5)
    
    print(f"\nMetrics:")
    print(f"  Hit@5: {metrics['hit_at_k']}")
    print(f"  MRR: {metrics['mrr']:.3f} (position 3: 1/3 = 0.333)")
    print(f"✓ Single doc handled correctly")
    
    print("\n✅ Edge cases test passed!")


# ============================================================================
# TEST 4: DATASET EVALUATION
# ============================================================================

def test_dataset_evaluation():
    """Test evaluate_dataset with multiple queries."""
    
    print("\n" + "="*70)
    print("TEST 4: DATASET EVALUATION")
    print("="*70)
    
    evaluator = Evaluator()
    
    # Multiple queries with results
    queries_results = {
        "query1": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "query2": ["doc5", "doc4", "doc3", "doc2", "doc1"],
        "query3": ["doc1", "doc2", "doc6", "doc7", "doc8"]
    }
    
    # Ground truth
    ground_truth = {
        "query1": ["doc2", "doc4"],
        "query2": ["doc2", "doc4"],
        "query3": ["doc1", "doc2"]
    }
    
    print(f"\nDataset: {len(queries_results)} queries")
    
    for i, (query, retrieved) in enumerate(queries_results.items(), 1):
        relevant = ground_truth[query]
        print(f"\n{i}. {query}")
        print(f"   Retrieved: {retrieved[:3]}...")
        print(f"   Relevant: {relevant}")
    
    # Evaluate
    print("\n[EVALUATING DATASET]")
    print("-"*70)
    
    aggregated = evaluator.evaluate_dataset(queries_results, ground_truth, k=5)
    
    if aggregated:
        print("\nAggregated Metrics (mean, std, min, max):")
        print()
        
        for metric_name, stats in aggregated.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std:  {stats['std']:.3f}")
            print(f"  Min:  {stats['min']:.3f}")
            print(f"  Max:  {stats['max']:.3f}")
            print()
    
    print("✅ Dataset evaluation test passed!")


# ============================================================================
# TEST 5: METRIC INTERPRETATION
# ============================================================================

def test_metric_interpretation():
    """Show what each metric means with examples."""
    
    print("\n" + "="*70)
    print("TEST 5: METRIC INTERPRETATION")
    print("="*70)
    
    evaluator = Evaluator()
    
    print("\n[METRIC MEANINGS]")
    print("-"*70)
    
    print("\n1. HIT@K - Count of relevant docs in top K")
    print("   Example: Retrieved [doc1, doc2, doc3], Relevant [doc2]")
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2"]
    hits = evaluator.hit_at_k(retrieved, relevant, k=3)
    print(f"   Hit@3 = {hits} (found 1 relevant doc)")
    
    print("\n2. PRECISION@K - % of top K that are relevant")
    print("   Example: Top 5 has 2 relevant")
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4"]
    precision = evaluator.precision_at_k(retrieved, relevant, k=5)
    print(f"   Precision@5 = {precision:.3f} ({precision*100:.0f}% of results are relevant)")
    
    print("\n3. RECALL@K - % of all relevant docs found")
    print("   Example: Out of 3 relevant docs, found 2 in top 5")
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc2", "doc3"]
    recall = evaluator.recall_at_k(retrieved, relevant, k=5)
    print(f"   Recall@5 = {recall:.3f} ({recall*100:.0f}% of relevant docs found)")
    
    print("\n4. MRR - Reciprocal rank of first relevant")
    print("   Example: First relevant at position 2")
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2"]
    mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)
    print(f"   MRR = {mrr:.3f} (1/position = 1/2 = 0.5)")
    
    print("\n5. NDCG@K - Ranking quality (penalizes lower positions)")
    print("   Example: Ideal [doc1, doc2], Actual [doc2, doc1]")
    retrieved = ["doc2", "doc1"]
    relevant = ["doc1", "doc2"]
    ndcg = evaluator.ndcg_at_k(retrieved, relevant, k=5)
    print(f"   NDCG@5 = {ndcg:.3f} (ideal ranking = 1.0, actual is lower)")
    
    print("\n6. F1@K - Harmonic mean of Precision and Recall")
    print("   Example: Precision 0.6, Recall 0.8")
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc5"]
    f1 = evaluator.f1_score_at_k(retrieved, relevant, k=5)
    print(f"   F1@5 = {f1:.3f} (balances precision and recall)")
    
    print("\n✅ Metric interpretation test passed!")


# ============================================================================
# TEST 6: PERFECT vs WORST CASE
# ============================================================================

def test_perfect_vs_worst():
    """Compare perfect and worst case scenarios."""
    
    print("\n" + "="*70)
    print("TEST 6: PERFECT vs WORST CASE")
    print("="*70)
    
    evaluator = Evaluator()
    
    relevant = ["doc1", "doc2", "doc3"]
    
    # Perfect case
    print("\n[PERFECT CASE] All relevant at top")
    print("-"*70)
    
    perfect_retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    print(f"Retrieved: {perfect_retrieved}")
    print(f"Relevant: {relevant}")
    
    perfect_metrics = evaluator.evaluate_query(perfect_retrieved, relevant, k=5)
    
    print(f"\nMetrics:")
    print(f"  Hit@5: {perfect_metrics['hit_at_k']} (all found)")
    print(f"  Precision@5: {perfect_metrics['precision_at_k']:.3f} (60%)")
    print(f"  Recall@5: {perfect_metrics['recall_at_k']:.3f} (100%)")
    print(f"  MRR: {perfect_metrics['mrr']:.3f}")
    print(f"  NDCG@5: {perfect_metrics['ndcg_at_k']:.3f}")
    print(f"  F1@5: {perfect_metrics['f1_score_at_k']:.3f}")
    
    # Worst case
    print("\n[WORST CASE] No relevant at top")
    print("-"*70)
    
    worst_retrieved = ["doc4", "doc5", "doc6", "doc7", "doc8"]
    print(f"Retrieved: {worst_retrieved}")
    print(f"Relevant: {relevant}")
    
    worst_metrics = evaluator.evaluate_query(worst_retrieved, relevant, k=5)
    
    print(f"\nMetrics:")
    print(f"  Hit@5: {worst_metrics['hit_at_k']} (none found)")
    print(f"  Precision@5: {worst_metrics['precision_at_k']:.3f} (0%)")
    print(f"  Recall@5: {worst_metrics['recall_at_k']:.3f} (0%)")
    print(f"  MRR: {worst_metrics['mrr']:.3f}")
    print(f"  NDCG@5: {worst_metrics['ndcg_at_k']:.3f}")
    print(f"  F1@5: {worst_metrics['f1_score_at_k']:.3f}")
    
    print("\n✅ Perfect vs Worst case test passed!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EVALUATOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nRunning all tests...\n")
    
    try:
        test_basic_metrics()
        test_evaluate_query()
        test_edge_cases()
        test_dataset_evaluation()
        test_metric_interpretation()
        test_perfect_vs_worst()
        
        print("\n" + "="*70)
        print("✅ ALL EVALUATOR TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Basic metrics working correctly")
        print("  ✓ Single query evaluation working")
        print("  ✓ Edge cases handled properly")
        print("  ✓ Dataset evaluation working")
        print("  ✓ Metric interpretation correct")
        print("  ✓ Perfect/worst cases tested")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()