"""
TEST_RETRIEVER.PY
=================

Comprehensive tests for the Retriever class.

Tests:
1. Initialization
2. Load and index documents
3. Single query retrieval
4. Batch retrieval
5. Hybrid search blending
6. Performance and speed
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retriever import Retriever


# ============================================================================
# TEST 1: INITIALIZATION
# ============================================================================

def test_initialization():
    """Test retriever initialization."""
    
    print("\n" + "="*70)
    print("TEST 1: INITIALIZATION")
    print("="*70)
    
    print("\n[Test 1a] Default weights")
    print("-"*70)
    
    retriever = Retriever()
    assert retriever.vector_weight == 0.7, "Default vector weight should be 0.7"
    assert retriever.keyword_weight == 0.3, "Default keyword weight should be 0.3"
    assert retriever.indexed == False, "Should not be indexed initially"
    print("✓ Default weights correct (0.7 vector, 0.3 keyword)")
    
    print("\n[Test 1b] Custom weights")
    print("-"*70)
    
    retriever_custom = Retriever(vector_weight=0.6, keyword_weight=0.4)
    assert retriever_custom.vector_weight == 0.6
    assert retriever_custom.keyword_weight == 0.4
    print("✓ Custom weights set correctly")
    
    print("\n[Test 1c] Weight validation")
    print("-"*70)
    
    try:
        retriever_invalid = Retriever(vector_weight=0.5, keyword_weight=0.3)
        print("✗ Should have raised error for weights not summing to 1.0")
    except AssertionError:
        print("✓ Weights validation working (must sum to 1.0)")
    
    print("\n✅ Initialization test passed!")


# ============================================================================
# TEST 2: LOAD AND INDEX
# ============================================================================

def test_load_and_index():
    """Test loading and indexing documents."""
    
    print("\n" + "="*70)
    print("TEST 2: LOAD AND INDEX DOCUMENTS")
    print("="*70)
    
    retriever = Retriever()
    
    print("\n[Test 2a] Load documents")
    print("-"*70)
    
    try:
        num_chunks = retriever.load_and_index_documents("data/password_guide.txt")
        print(f"✓ Loaded and indexed {num_chunks} chunks")
        assert num_chunks > 0, "Should load at least 1 chunk"
        assert retriever.indexed == True, "Should be indexed after loading"
    except FileNotFoundError:
        print("✗ File not found: data/password_guide.txt")
        print("  Create this file from Day 2 guide")
        return False
    
    print("\n[Test 2b] Verify indexing")
    print("-"*70)
    
    assert retriever.indexed == True, "indexed flag should be True"
    print("✓ Retriever marked as indexed")
    
    print("\n✅ Load and index test passed!")
    return True


# ============================================================================
# TEST 3: SINGLE QUERY RETRIEVAL
# ============================================================================

def test_single_query_retrieval():
    """Test retrieving documents for single query."""
    
    print("\n" + "="*70)
    print("TEST 3: SINGLE QUERY RETRIEVAL")
    print("="*70)
    
    retriever = Retriever()
    
    try:
        retriever.load_and_index_documents("data/password_guide.txt")
    except FileNotFoundError:
        print("❌ Need data/password_guide.txt")
        return False
    
    # Test 1: Basic query
    print("\n[Test 3a] Basic query")
    print("-"*70)
    
    query1 = "How to reset password?"
    print(f"Query: '{query1}'")
    
    results = retriever.retrieve(query1, top_k=5)
    
    assert len(results) > 0, "Should return results"
    assert len(results) <= 5, "Should return top 5"
    print(f"✓ Retrieved {len(results)} results")
    
    # Check result structure
    print("\n[Test 3b] Result structure")
    print("-"*70)
    
    first_result = results[0]
    
    required_fields = ['document', 'combined_score', 'vector_score', 'keyword_score', 'metadata']
    for field in required_fields:
        assert field in first_result, f"Missing field: {field}"
    
    print("✓ All required fields present:")
    print(f"  - document: {first_result['document'][:60]}...")
    print(f"  - combined_score: {first_result['combined_score']:.3f}")
    print(f"  - vector_score: {first_result['vector_score']:.3f}")
    print(f"  - keyword_score: {first_result['keyword_score']:.3f}")
    
    # Test 2: Different top_k values
    print("\n[Test 3c] Different top_k values")
    print("-"*70)
    
    for k in [1, 3, 5]:
        results_k = retriever.retrieve(query1, top_k=k)
        assert len(results_k) == k, f"Should return exactly {k} results"
        print(f"✓ top_k={k}: Retrieved {len(results_k)} results")
    
    # Test 3: Scores are normalized (0-1)
    print("\n[Test 3d] Score validation")
    print("-"*70)
    
    results = retriever.retrieve(query1, top_k=5)
    
    for i, result in enumerate(results):
        combined = result['combined_score']
        vector = result['vector_score']
        keyword = result['keyword_score']
        
        assert 0 <= combined <= 1, f"combined_score should be 0-1, got {combined}"
        assert 0 <= vector <= 1, f"vector_score should be 0-1, got {vector}"
        assert 0 <= keyword <= 1, f"keyword_score should be 0-1, got {keyword}"
    
    print("✓ All scores in valid range (0-1)")
    
    # Test 4: Results are ranked by combined_score
    print("\n[Test 3e] Results ranking")
    print("-"*70)
    
    results = retriever.retrieve(query1, top_k=5)
    
    for i in range(len(results) - 1):
        assert results[i]['combined_score'] >= results[i+1]['combined_score'], \
            "Results should be sorted by combined_score"
    
    print("✓ Results properly ranked by combined_score")
    
    # Test 5: Different queries
    print("\n[Test 3f] Different query types")
    print("-"*70)
    
    test_queries = [
        "password reset",
        "2FA enable authentication",
        "security best practices",
        "account recovery procedures"
    ]
    
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        assert len(results) > 0, f"Should find results for: {query}"
        print(f"✓ '{query}': {len(results)} results, top score {results[0]['combined_score']:.3f}")
    
    print("\n✅ Single query retrieval test passed!")
    return True


# ============================================================================
# TEST 4: BATCH RETRIEVAL
# ============================================================================

def test_batch_retrieval():
    """Test batch retrieval for multiple queries."""
    
    print("\n" + "="*70)
    print("TEST 4: BATCH RETRIEVAL")
    print("="*70)
    
    retriever = Retriever()
    
    try:
        retriever.load_and_index_documents("data/password_guide.txt")
    except FileNotFoundError:
        print("❌ Need data/password_guide.txt")
        return False
    
    print("\n[Test 4a] Batch retrieval")
    print("-"*70)
    
    queries = [
        "How to reset password?",
        "Enable 2FA",
        "Account recovery"
    ]
    
    print(f"Retrieving for {len(queries)} queries...")
    
    batch_results = retriever.batch_retrieve(queries, top_k=3)
    
    assert isinstance(batch_results, dict), "Should return dict"
    assert len(batch_results) == len(queries), f"Should have {len(queries)} results"
    
    print(f"✓ Retrieved results for all {len(queries)} queries")
    
    print("\n[Test 4b] Batch results structure")
    print("-"*70)
    
    for query, results in batch_results.items():
        assert isinstance(results, list), f"Results should be list for: {query}"
        assert len(results) > 0, f"Should have results for: {query}"
        print(f"✓ '{query}': {len(results)} results")
    
    print("\n✅ Batch retrieval test passed!")
    return True


# ============================================================================
# TEST 5: HYBRID SEARCH BLENDING
# ============================================================================

def test_hybrid_blending():
    """Test hybrid search blending of vector and keyword scores."""
    
    print("\n" + "="*70)
    print("TEST 5: HYBRID SEARCH BLENDING")
    print("="*70)
    
    # Test with different weights
    print("\n[Test 5a] Different weight combinations")
    print("-"*70)
    
    test_weights = [
        (1.0, 0.0),   # 100% vector
        (0.8, 0.2),   # 80% vector, 20% keyword
        (0.7, 0.3),   # 70% vector, 30% keyword (default)
        (0.5, 0.5),   # 50/50 split
    ]
    
    query = "password reset"
    
    previous_score = None
    
    for v_weight, k_weight in test_weights:
        retriever = Retriever(vector_weight=v_weight, keyword_weight=k_weight)
        
        try:
            retriever.load_and_index_documents("data/password_guide.txt")
        except FileNotFoundError:
            print("❌ Need data/password_guide.txt")
            return False
        
        results = retriever.retrieve(query, top_k=1)
        
        if results:
            score = results[0]['combined_score']
            print(f"✓ {v_weight*100:.0f}% vector + {k_weight*100:.0f}% keyword: top score = {score:.3f}")
    
    print("\n[Test 5b] Score components")
    print("-"*70)
    
    retriever = Retriever()
    retriever.load_and_index_documents("data/password_guide.txt")
    
    results = retriever.retrieve(query, top_k=3)
    
    print(f"Query: '{query}'")
    print("\nTop 3 results (Vector score | Keyword score | Combined):")
    
    for i, result in enumerate(results, 1):
        v = result['vector_score']
        k = result['keyword_score']
        c = result['combined_score']
        expected = 0.7 * v + 0.3 * k
        
        assert abs(c - expected) < 0.001, "Combined score calculation incorrect"
        
        print(f"{i}. Vector: {v:.3f} | Keyword: {k:.3f} | Combined: {c:.3f}")
    
    print("✓ Combined scores correctly calculated")
    
    print("\n✅ Hybrid blending test passed!")
    return True


# ============================================================================
# TEST 6: PERFORMANCE
# ============================================================================

def test_performance():
    """Test retrieval performance and speed."""
    
    print("\n" + "="*70)
    print("TEST 6: PERFORMANCE")
    print("="*70)
    
    retriever = Retriever()
    
    try:
        start = time.time()
        retriever.load_and_index_documents("data/password_guide.txt")
        load_time = time.time() - start
    except FileNotFoundError:
        print("❌ Need data/password_guide.txt")
        return False
    
    print(f"\n[Test 6a] Indexing speed")
    print("-"*70)
    print(f"✓ Indexing took {load_time:.3f} seconds")
    
    # Single query performance
    print(f"\n[Test 6b] Single query speed")
    print("-"*70)
    
    query = "How to reset password?"
    
    start = time.time()
    results = retriever.retrieve(query, top_k=5)
    query_time = time.time() - start
    
    print(f"✓ Single query took {query_time*1000:.1f} ms")
    print(f"  Retrieved {len(results)} results")
    
    # Batch performance
    print(f"\n[Test 6c] Batch query speed")
    print("-"*70)
    
    queries = [
        "password reset",
        "2FA enable",
        "account recovery",
        "security best practices",
        "password strength"
    ]
    
    start = time.time()
    batch_results = retriever.batch_retrieve(queries, top_k=5)
    batch_time = time.time() - start
    
    print(f"✓ Batch of {len(queries)} queries took {batch_time*1000:.1f} ms")
    print(f"  Average per query: {batch_time/len(queries)*1000:.1f} ms")
    
    print("\n✅ Performance test passed!")
    return True


# ============================================================================
# TEST 7: ERROR HANDLING
# ============================================================================

def test_error_handling():
    """Test error handling."""
    
    print("\n" + "="*70)
    print("TEST 7: ERROR HANDLING")
    print("="*70)
    
    print("\n[Test 7a] Retrieve before indexing")
    print("-"*70)
    
    retriever = Retriever()
    
    try:
        results = retriever.retrieve("test query")
        print("✗ Should have raised error")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("\n[Test 7b] Invalid file path")
    print("-"*70)
    
    retriever = Retriever()
    
    try:
        retriever.load_and_index_documents("data/password_guide.txt")
        print("✗ Should have raised error")
    except FileNotFoundError:
        print("✓ Correctly raised FileNotFoundError")
    
    print("\n✅ Error handling test passed!")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("RETRIEVER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nRunning all tests...\n")
    
    results = []
    
    try:
        # Test 1
        test_initialization()
        results.append(("Initialization", True))
        
        # Test 2
        success = test_load_and_index()
        results.append(("Load and Index", success))
        
        if success:
            # Tests 3-7 depend on successful indexing
            try:
                test_single_query_retrieval()
                results.append(("Single Query", True))
            except Exception as e:
                print(f"❌ Single Query test failed: {e}")
                results.append(("Single Query", False))
            
            try:
                test_batch_retrieval()
                results.append(("Batch Retrieval", True))
            except Exception as e:
                print(f"❌ Batch Retrieval test failed: {e}")
                results.append(("Batch Retrieval", False))
            
            try:
                test_hybrid_blending()
                results.append(("Hybrid Blending", True))
            except Exception as e:
                print(f"❌ Hybrid Blending test failed: {e}")
                results.append(("Hybrid Blending", False))
            
            try:
                test_performance()
                results.append(("Performance", True))
            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                results.append(("Performance", False))
        
        # Test error handling
        test_error_handling()
        results.append(("Error Handling", True))
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n" + "="*70)
        print("✅ ALL RETRIEVER TESTS PASSED!")
        print("="*70 + "\n")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed\n")