"""
DAY 4: COMPLETE INTEGRATION TEST
================================

Tests the full RAG pipeline:
1. DocumentLoader (Day 2) - Load documents
2. EmbeddingGenerator (Day 3) - Create embeddings
3. VectorStore (Day 4) - Store and search

This brings everything together!
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Import from your src folder
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


# ============================================================================
# TEST 1: COMPLETE PIPELINE
# ============================================================================

def test_complete_pipeline():
    """Test the complete RAG pipeline end-to-end"""
    
    print("\n" + "="*70)
    print("DAY 4: COMPLETE RAG PIPELINE TEST")
    print("="*70)
    
    # ====================================================================
    # STEP 1: Load documents (Day 2)
    # ====================================================================
    print("\n[STEP 1] Load documents using DocumentLoader")
    print("-"*70)
    
    try:
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        docs = loader.load_from_file("data/password_guide.txt")
        print(f"\n✅ Loaded {len(docs)} document chunks")
        
    except FileNotFoundError:
        print(f"\n❌ File not found: data/password_guide.txt")
        print(f"   Please create this file first (see Day 2 guide)")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 2: Generate embeddings (Day 3)
    # ====================================================================
    print("\n[STEP 2] Generate embeddings using EmbeddingGenerator")
    print("-"*70)
    
    try:
        embedder = EmbeddingGenerator()
        
        chunk_texts = [doc.page_content for doc in docs]
        print(f"\nEmbedding {len(chunk_texts)} chunks...")
        
        embeddings = embedder.embed_texts(chunk_texts)
        print(f"✅ Created embeddings")
        print(f"   Shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 3: Store in vector database (Day 4)
    # ====================================================================
    print("\n[STEP 3] Store in vector database using VectorStore")
    print("-"*70)
    
    try:
        store = VectorStore("data/chroma_db")
        
        print(f"\nAdding documents to vector store...")
        store.add_documents(docs, embeddings)
        
        store.print_stats()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 4: Search Test 1 - Password Reset
    # ====================================================================
    print("\n[STEP 4] SEARCH TEST 1: Password reset query")
    print("-"*70)
    
    query1 = "How do I reset my password?"
    print(f"\nQuery: '{query1}'")
    
    try:
        query_embedding1 = embedder.embed_text(query1)
        results1 = store.search(query_embedding1, top_k=5)
        
        print(f"\n✅ Top 5 results:\n")
        
        for i, result in enumerate(results1, 1):
            similarity = result['similarity']
            doc = result['document']
            
            # Create visual bar
            bar_length = int(similarity * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Relevance interpretation
            if similarity > 0.7:
                relevance = "✓ HIGHLY RELEVANT"
            elif similarity > 0.5:
                relevance = "◐ RELEVANT"
            else:
                relevance = "△ SOMEWHAT RELEVANT"
            
            print(f"{i}. Similarity: {similarity:.3f} {bar} {relevance}")
            print(f"   Document: {doc[:80]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 5: Search Test 2 - 2FA
    # ====================================================================
    print("[STEP 5] SEARCH TEST 2: Two-factor authentication")
    print("-"*70)
    
    query2 = "How to enable two-factor authentication"
    print(f"\nQuery: '{query2}'")
    
    try:
        query_embedding2 = embedder.embed_text(query2)
        results2 = store.search(query_embedding2, top_k=3)
        
        print(f"\n✅ Top 3 results:\n")
        
        for i, result in enumerate(results2, 1):
            similarity = result['similarity']
            doc = result['document']
            
            bar = "█" * int(similarity * 20)
            
            print(f"{i}. Score: {similarity:.3f} {bar}")
            print(f"   {doc[:100]}...\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 6: Search Test 3 - Account Recovery
    # ====================================================================
    print("[STEP 6] SEARCH TEST 3: Account recovery")
    print("-"*70)
    
    query3 = "I lost access to my account, how do I recover it?"
    print(f"\nQuery: '{query3}'")
    
    try:
        query_embedding3 = embedder.embed_text(query3)
        results3 = store.search(query_embedding3, top_k=3)
        
        print(f"\n✅ Top 3 results:\n")
        
        for i, result in enumerate(results3, 1):
            similarity = result['similarity']
            doc = result['document']
            
            bar = "█" * int(similarity * 20)
            
            print(f"{i}. Score: {similarity:.3f} {bar}")
            print(f"   {doc[:100]}...\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ====================================================================
    # STEP 7: Show metadata
    # ====================================================================
    print("[STEP 7] Metadata from first result")
    print("-"*70)
    
    first_result = results1[0]
    
    print(f"\nMetadata:")
    for key, value in first_result['metadata'].items():
        print(f"  • {key}: {value}")
    
    print(f"\nDocument ID: {first_result['id']}")
    print(f"Distance: {first_result['distance']:.3f}")
    print(f"Similarity: {first_result['similarity']:.3f}")
    
    # ====================================================================
    # STEP 8: Persistence
    # ====================================================================
    print("\n[STEP 8] Persistence demonstration")
    print("-"*70)
    
    print("\n📁 Vector store saved to disk!")
    print(f"   Location: data/chroma_db/")
    print("\nIf you run this script again:")
    print("  1. Documents already loaded in database")
    print("  2. No need to re-embed them")
    print("  3. Just query the existing index!")


# ============================================================================
# TEST 2: CLEAR AND REBUILD
# ============================================================================

def test_clear_and_rebuild():
    """Test clearing and rebuilding the database"""
    
    print("\n" + "="*70)
    print("TEST: CLEAR AND REBUILD DATABASE")
    print("="*70)
    
    try:
        store = VectorStore("data/chroma_db")
        
        print("\nBefore clear:")
        store.print_stats()
        
        print("\nClearing database...")
        store.clear()
        
        print("After clear:")
        store.print_stats()
        
        print("\n✅ Database cleared and ready for new data!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Run main test
    test_complete_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print("✅ DAY 4 INTEGRATION TEST COMPLETE!")
    print("="*70)
    
    print("\n📚 WHAT YOU ACCOMPLISHED:")
    print("\n1. ✅ Loaded documents (DocumentLoader)")
    print("2. ✅ Generated embeddings (EmbeddingGenerator)")
    print("3. ✅ Stored in vector database (VectorStore)")
    print("4. ✅ Searched for similar documents")
    print("5. ✅ Retrieved metadata")
    print("6. ✅ Data persisted to disk")
    
    print("\n🎯 YOU NOW HAVE:")
    print("   • A working semantic search system")
    print("   • Persistent vector database")
    print("   • Complete RAG pipeline (Days 2-4)")
    
    print("\n📅 NEXT STEP (Day 5):")
    print("   Build keyword search (BM25)")
    print("   Create hybrid retriever")
    print("   Combine vector + keyword search")
    
    print("\n" + "="*70 + "\n")
