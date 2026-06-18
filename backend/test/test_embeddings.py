"""
DAY 3: TEST EMBEDDINGS WITH REAL DOCUMENTS
==========================================

This script shows how embeddings integrate with document chunks.
It demonstrates semantic search on real documents.

WHAT THIS DOES:
1. Load documents using DocumentLoader (from Day 2)
2. Embed all document chunks using EmbeddingGenerator
3. Search for similar chunks using a query
4. Show statistics about embeddings
5. Test multiple queries
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src/
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings import EmbeddingGenerator
from src.document_loader import DocumentLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# TEST 1: EMBEDDINGS WITH REAL DOCUMENTS
# ============================================================================

def test_embedding_real_documents():
    """
    Test embedding system with real document chunks
    
    PROCESS:
    1. Load documents from file (Day 2 skill)
    2. Embed all chunks (Day 3 skill)
    3. Search for similar documents
    4. Show results
    """
    
    print("\n" + "="*70)
    print("TEST 1: EMBEDDINGS WITH REAL DOCUMENTS")
    print("="*70)
    
    # ====================================================================
    # STEP 1: Load documents
    # ====================================================================
    print("\n[STEP 1] Load and chunk documents")
    print("-"*70)
    
    try:
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        docs = loader.load_from_file("data/password_guide.txt")
        print(f"\n✅ Loaded {len(docs)} chunks from document")
        
    except FileNotFoundError:
        print(f"\n❌ File not found: data/password_guide.txt")
        print(f"   Please create this file first (see Day 2 guide)")
        return
    
    # ====================================================================
    # STEP 2: Initialize embedder
    # ====================================================================
    print("\n[STEP 2] Initialize embedder")
    print("-"*70)
    
    embedder = EmbeddingGenerator()
    
    # ====================================================================
    # STEP 3: Embed all document chunks
    # ====================================================================
    print("\n[STEP 3] Embed all document chunks")
    print("-"*70)
    
    # Extract text from documents
    chunk_texts = [doc.page_content for doc in docs]
    
    print(f"\nEmbedding {len(chunk_texts)} chunks...")
    print("(This might take 10-20 seconds)")
    
    # Embed all chunks at once (faster than one-by-one)
    embeddings = embedder.embed_texts(chunk_texts)
    
    print(f"\n✅ Embedding complete!")
    print(f"   Shape: {embeddings.shape}")
    print(f"   ({len(chunk_texts)} chunks × {embeddings.shape[1]} dimensions)")
    
    # ====================================================================
    # STEP 4: Search using Query 1
    # ====================================================================
    print("\n[STEP 4] SEARCH TEST 1: Password reset query")
    print("-"*70)
    
    query1 = "How do I reset my password?"
    print(f"\nQuery: '{query1}'")
    print(f"\nFinding most similar chunks...")
    
    # Embed the query
    query_embedding = embedder.embed_text(query1)
    
    # Calculate similarity to all chunks
    query_embedding_reshaped = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding_reshaped, embeddings)[0]
    
    # Get top 5 most similar
    top_5_indices = similarities.argsort()[-5:][::-1]  # Reverse to get highest first
    
    print(f"\n✅ Top 5 most similar chunks:\n")
    
    for rank, idx in enumerate(top_5_indices, 1):
        score = similarities[idx]
        chunk = chunk_texts[idx]
        
        # Visual bar
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Interpretation
        if score > 0.7:
            relevance = "✓ HIGHLY RELEVANT"
        elif score > 0.5:
            relevance = "◐ RELEVANT"
        else:
            relevance = "△ SOMEWHAT RELEVANT"
        
        print(f"{rank}. Similarity: {score:.3f} {bar} {relevance}")
        print(f"   Chunk preview: {chunk[:80]}...")
        print()
    
    # ====================================================================
    # STEP 5: Embeddings statistics
    # ====================================================================
    print("[STEP 5] STATISTICS")
    print("-"*70)
    
    print(f"\nEmbedding statistics for query-document similarity:")
    print(f"  Min similarity: {similarities.min():.3f}")
    print(f"  Max similarity: {similarities.max():.3f}")
    print(f"  Mean similarity: {similarities.mean():.3f}")
    print(f"  Median similarity: {np.median(similarities):.3f}")
    print(f"  Std deviation: {similarities.std():.3f}")
    
    # How many are "relevant" (similarity > 0.5)?
    relevant_count = sum(1 for s in similarities if s > 0.5)
    print(f"\n  Documents with similarity > 0.5: {relevant_count}/{len(similarities)}")
    
    # ====================================================================
    # STEP 6: Search using Query 2
    # ====================================================================
    print("\n[STEP 6] SEARCH TEST 2: Two-factor authentication query")
    print("-"*70)
    
    query2 = "How to enable two-factor authentication for security"
    print(f"\nQuery: '{query2}'")
    
    query_embedding2 = embedder.embed_text(query2)
    query_embedding2_reshaped = query_embedding2.reshape(1, -1)
    similarities2 = cosine_similarity(query_embedding2_reshaped, embeddings)[0]
    
    # Get top 3
    top_3_indices = similarities2.argsort()[-3:][::-1]
    
    print(f"\n✅ Top 3 most similar chunks:\n")
    
    for rank, idx in enumerate(top_3_indices, 1):
        score = similarities2[idx]
        chunk = chunk_texts[idx]
        
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{rank}. Score: {score:.3f} {bar}")
        print(f"   {chunk[:85]}...\n")
    
    # ====================================================================
    # STEP 7: Search using Query 3 (different query)
    # ====================================================================
    print("[STEP 7] SEARCH TEST 3: Account recovery query")
    print("-"*70)
    
    query3 = "I lost access to my account, how do I recover it?"
    print(f"\nQuery: '{query3}'")
    
    query_embedding3 = embedder.embed_text(query3)
    query_embedding3_reshaped = query_embedding3.reshape(1, -1)
    similarities3 = cosine_similarity(query_embedding3_reshaped, embeddings)[0]
    
    best_idx = similarities3.argmax()
    best_score = similarities3[best_idx]
    best_chunk = chunk_texts[best_idx]
    
    bar_length = int(best_score * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)
    
    print(f"\nBest match (similarity: {best_score:.3f}):")
    print(f"Score: {best_score:.3f} {bar}")
    print(f"Content: {best_chunk[:120]}...\n")
    
    # ====================================================================
    # STEP 8: Verify embedding consistency
    # ====================================================================
    print("[STEP 8] EMBEDDING CONSISTENCY CHECK")
    print("-"*70)
    
    print("\nEmbedding same text twice should give same result:")
    
    test_text = "Password security is important"
    emb1 = embedder.embed_text(test_text)
    emb2 = embedder.embed_text(test_text)
    
    # Check if they're exactly the same
    difference = np.sum(np.abs(emb1 - emb2))
    
    print(f"\nText: '{test_text}'")
    print(f"Embedding 1 (first 5 dims): {emb1[:5]}")
    print(f"Embedding 2 (first 5 dims): {emb2[:5]}")
    print(f"Total difference: {difference:.10f}")
    
    if difference < 0.0001:
        print("✅ Embeddings are consistent (identical)")
    else:
        print("⚠ Small differences due to floating point precision (expected)")


# ============================================================================
# TEST 2: UNDERSTAND EMBEDDING DIMENSIONS
# ============================================================================

def test_embedding_dimensions():
    """
    Understand what embedding dimensions represent
    
    Each dimension captures something about semantic meaning:
    - Some capture topic (password, security, etc.)
    - Some capture sentiment (positive, negative)
    - Some capture formality level
    - etc.
    """
    
    print("\n" + "="*70)
    print("TEST 2: UNDERSTANDING EMBEDDING DIMENSIONS")
    print("="*70)
    
    embedder = EmbeddingGenerator()
    
    # Different types of texts
    texts = {
        "Password topic": "How to reset password account security",
        "Security topic": "Enable two-factor authentication encryption",
        "Different domain": "How to cook pasta recipe ingredients",
        "Another domain": "How to build a house construction materials"
    }
    
    print(f"\nEmbedding different text types...")
    embeddings_dict = {}
    
    for name, text in texts.items():
        emb = embedder.embed_text(text)
        embeddings_dict[name] = emb
        print(f"✓ {name}: {emb.shape}")
    
    # Compare embeddings
    print(f"\n[COMPARISONS] How different types of texts relate:")
    print("-"*70)
    
    # Same domain comparisons
    print("\n1. SAME DOMAIN (both about passwords/security):")
    emb_pwd = embeddings_dict["Password topic"]
    emb_sec = embeddings_dict["Security topic"]
    
    sim_same_domain = cosine_similarity(
        emb_pwd.reshape(1, -1),
        emb_sec.reshape(1, -1)
    )[0][0]
    
    bar = "█" * int(sim_same_domain * 20)
    print(f"   Password topic vs Security topic")
    print(f"   Similarity: {sim_same_domain:.3f} {bar}")
    print(f"   (Should be HIGH because related topics)")
    
    # Different domain comparisons
    print("\n2. DIFFERENT DOMAIN (passwords vs cooking):")
    emb_cook = embeddings_dict["Different domain"]
    
    sim_diff_domain = cosine_similarity(
        emb_pwd.reshape(1, -1),
        emb_cook.reshape(1, -1)
    )[0][0]
    
    bar = "█" * max(1, int(sim_diff_domain * 20))
    print(f"   Password topic vs How to cook")
    print(f"   Similarity: {sim_diff_domain:.3f} {bar}")
    print(f"   (Should be LOW because unrelated topics)")
    
    # Show the difference
    print(f"\n3. DIFFERENCE:")
    difference = sim_same_domain - sim_diff_domain
    print(f"   Same domain - Different domain = {difference:.3f}")
    print(f"   Shows embeddings capture semantic meaning!")
    
    # Statistical analysis
    print(f"\n4. EMBEDDING STATISTICS:")
    all_embeddings = np.array(list(embeddings_dict.values()))
    print(f"   Min value across all embeddings: {all_embeddings.min():.4f}")
    print(f"   Max value across all embeddings: {all_embeddings.max():.4f}")
    print(f"   Mean value across all embeddings: {all_embeddings.mean():.4f}")
    print(f"   Values range from -1 to 1 (normalized)")


# ============================================================================
# TEST 3: BATCH EMBEDDING SPEED
# ============================================================================

def test_embedding_speed():
    """
    Show that batch embedding is much faster than embedding one-by-one
    """
    
    print("\n" + "="*70)
    print("TEST 3: EMBEDDING SPEED COMPARISON")
    print("="*70)
    
    embedder = EmbeddingGenerator()
    
    # Create test texts
    test_texts = [
        "How to reset password",
        "Enable two-factor authentication",
        "Account security best practices",
        "Password change procedures",
        "Recovery codes and backup methods"
    ]
    
    print(f"\nEmbedding {len(test_texts)} texts...")
    
    # Method 1: Batch embedding (recommended)
    print(f"\nMethod 1: Batch embedding (all at once)")
    print(f"Code: embeddings = embedder.embed_texts(texts)")
    
    import time
    start = time.time()
    batch_embeddings = embedder.embed_texts(test_texts)
    batch_time = time.time() - start
    
    print(f"✅ Result shape: {batch_embeddings.shape}")
    print(f"   Time taken: {batch_time:.4f} seconds")
    
    # Method 2: Individual embedding (slower)
    print(f"\nMethod 2: Individual embeddings (one-by-one)")
    print(f"Code: embeddings = [embedder.embed_text(t) for t in texts]")
    
    start = time.time()
    individual_embeddings = [embedder.embed_text(t) for t in test_texts]
    individual_time = time.time() - start
    
    print(f"✅ Result count: {len(individual_embeddings)}")
    print(f"   Time taken: {individual_time:.4f} seconds")
    
    # Comparison
    print(f"\n[COMPARISON]")
    print(f"Batch method: {batch_time:.4f}s")
    print(f"Individual method: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.1f}x faster with batch!")
    print(f"\n✓ Always use batch embedding for multiple texts!")


# ============================================================================
# MAIN - RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DAY 3: COMPLETE EMBEDDINGS TEST SUITE")
    print("="*70)
    
    try:
        # Run all tests
        test_embedding_real_documents()
        test_embedding_dimensions()
        test_embedding_speed()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📚 WHAT YOU LEARNED TODAY:")
        print("\n1. HOW EMBEDDINGS WORK")
        print("   • Convert text to 384-dimensional vectors")
        print("   • Similar texts have similar vectors")
        print("   • Neural networks learn semantic relationships")
        
        print("\n2. SEMANTIC SEARCH")
        print("   • Query embedded and compared to all documents")
        print("   • Cosine similarity measures vector angle")
        print("   • Most similar documents ranked first")
        
        print("\n3. PRACTICAL SKILLS")
        print("   • embed_text() for single texts")
        print("   • embed_texts() for multiple (faster!)")
        print("   • similarity() for comparing two texts")
        print("   • similarities() for searching (RAG core operation)")
        
        print("\n🎯 INTERVIEW STRENGTH:")
        print("   You can now explain semantic search from first principles!")
        print("   And code it from scratch!")
        
        print("\n📅 NEXT: Day 4 - Vector Database (ChromaDB)")
        print("   Store embeddings for fast retrieval")
        print("   Integrate with Day 2 document chunks")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
