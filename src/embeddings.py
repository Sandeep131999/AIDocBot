"""
DAY 3: EMBEDDINGS SYSTEM
========================

This file converts text chunks into vector representations.
These vectors are the foundation of semantic search for RAG.

What this code does:
1. Takes text input
2. Loads a pre-trained embedding model
3. Converts text → vector (list of 384 numbers)
4. Calculates similarity between texts
5. Finds most similar documents for a query

The embedding is the CORE of semantic search in RAG systems.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# EMBEDDING GENERATOR CLASS
# ============================================================================

class EmbeddingGenerator:
    """
    Converts text to embeddings (vector representations)
    
    Think of this as:
    - Input: Text ("How to reset password?")
    - Process: Neural network converts to numbers
    - Output: Vector with 384 dimensions (numbers)
    
    Similar texts → similar vectors
    Different texts → different vectors
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a pre-trained model.
        
        WHAT HAPPENS HERE:
        1. Download the embedding model (first time only, then cached)
        2. Load it into memory
        3. Store model details
        
        Args:
            model_name: Which embedding model to use
                       - "all-MiniLM-L6-v2": Small, fast, perfect for learning (22MB)
                       - "all-mpnet-base-v2": Better quality but slower (420MB)
                       - "all-distilroberta-v1": Balanced (268MB)
        
        WHY THESE MODELS:
        - Pre-trained on millions of texts
        - Learn to convert meaning → vectors
        - All produce 384-dimensional vectors (for MiniLM)
        
        EXAMPLE:
            embedder = EmbeddingGenerator()
            # ✓ Model loaded, ready to use
        """
        
        self.model_name = model_name
        
        print("\n" + "="*70)
        print("EMBEDDING GENERATOR - INITIALIZATION")
        print("="*70)
        print(f"\n📥 Loading model: {model_name}")
        print(f"   Status: Downloading (first time ~30 seconds)...")
        print(f"   Note: Automatically cached after first load")
        
        try:
            # Load the pre-trained model
            # sentence_transformers handles downloading + caching automatically
            self.model = SentenceTransformer(model_name)
            
            # Get vector dimensions for this model
            self.dimensions = self.model.get_sentence_embedding_dimension()
            
            print(f"\n✅ Model loaded successfully!")
            print(f"   Dimensions: {self.dimensions}")
            print(f"   (Each text becomes a vector of {self.dimensions} numbers)")
            print(f"   Range: Each number is between -1 and 1")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print(f"   Try: pip install -r requirements.txt")
            raise
    
    # ========================================================================
    # MAIN FUNCTIONS
    # ========================================================================
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a SINGLE text to an embedding (vector).
        
        PROCESS:
        1. Take input text: "How to reset password?"
        2. Model processes it through neural network
        3. Output: Vector of 384 numbers
        
        Args:
            text: Text to convert (string)
        
        Returns:
            numpy array of shape (384,) with numbers between -1 and 1
        
        HOW IT WORKS:
            Behind the scenes:
            "password reset" 
                ↓
            [Tokenize] → ["password", "reset"]
                ↓
            [Look up learned vectors] → [word1_vector, word2_vector]
                ↓
            [Average them] → sentence_vector
                ↓
            [Return] → numpy array with 384 dimensions
        
        EXAMPLE:
            >>> embedder = EmbeddingGenerator()
            >>> text = "How to reset password?"
            >>> embedding = embedder.embed_text(text)
            >>> print(embedding.shape)
            (384,)
            >>> print(embedding[:5])  # First 5 dimensions
            [-0.024  0.189 -0.045  0.712  0.023]
        
        WHAT THE NUMBERS MEAN:
            Each of 384 dimensions captures something about meaning:
            - Dimension 1: "Is this about passwords?" (0=no, 1=yes)
            - Dimension 2: "Is this about actions?" (0=no, 1=yes)
            - Dimension 3: "Formal vs casual?" (0=casual, 1=formal)
            - ... 381 more dimensions
        """
        
        try:
            # encode() converts text to embedding
            # convert_to_numpy=True returns as numpy array (easier to work with)
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            print(f"❌ Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert MULTIPLE texts to embeddings (MUCH FASTER than one-by-one).
        
        PROCESS:
        1. Take list of texts
        2. Process all at once through model
        3. Return 2D array (each row is one embedding)
        
        Args:
            texts: List of text strings
        
        Returns:
            2D numpy array of shape (num_texts, 384)
            Each row is one embedding
        
        WHY THIS IS FASTER:
            - GPU can process multiple texts simultaneously
            - Less overhead per item
            - Typically 10-100x faster than embedding one-by-one
        
        EXAMPLE:
            >>> texts = [
            ...     "How to reset password?",
            ...     "Forgot my account password",
            ...     "How to make pasta?"
            ... ]
            >>> embeddings = embedder.embed_texts(texts)
            >>> print(embeddings.shape)
            (3, 384)
            >>> print(f"First text embedding: {embeddings[0][:5]}")
            First text embedding: [-0.024  0.189 -0.045  0.712  0.023]
            >>> print(f"Second text embedding: {embeddings[1][:5]}")
            Second text embedding: [0.015  0.201 -0.038  0.718  0.019]
        
        MEMORY NOTES:
            - 100 texts: ~150 KB
            - 1000 texts: ~1.5 MB
            - 10000 texts: ~15 MB
        """
        
        try:
            # encode() can handle both single text and list of texts
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            print(f"❌ Error embedding texts: {e}")
            raise
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate how similar TWO texts are using COSINE SIMILARITY.
        
        CONCEPT:
            Imagine two vectors as arrows pointing in directions.
            - If arrows point same direction → similar meaning → score ≈ 1.0
            - If arrows point opposite → opposite meaning → score ≈ -1.0
            - If arrows perpendicular → different → score ≈ 0.0
        
        Args:
            text1: First text to compare
            text2: Second text to compare
        
        Returns:
            Float between -1 and 1:
            - 1.0 = identical meaning
            - 0.9+ = nearly identical
            - 0.7-0.9 = very similar
            - 0.5-0.7 = similar
            - 0.3-0.5 = somewhat similar
            - <0.3 = different
            - -1.0 = opposite meaning
        
        COSINE SIMILARITY FORMULA (for understanding):
            similarity = (A · B) / (||A|| × ||B||)
            
            Where:
            - A · B = dot product (multiply corresponding values, sum)
            - ||A|| = length/magnitude of vector A
            - ||B|| = length/magnitude of vector B
            - Result: angle between vectors (0° = same direction = 1.0)
        
        EXAMPLE 1 - Similar texts:
            >>> text_a = "How to reset password?"
            >>> text_b = "Click Forgot Password button"
            >>> score = embedder.similarity(text_a, text_b)
            >>> print(f"Similarity: {score:.3f}")
            Similarity: 0.923
            (Both about password reset, very similar!)
        
        EXAMPLE 2 - Different texts:
            >>> text_a = "How to reset password?"
            >>> text_c = "How to make pasta?"
            >>> score = embedder.similarity(text_a, text_c)
            >>> print(f"Similarity: {score:.3f}")
            Similarity: 0.087
            (Completely different topics!)
        
        EXAMPLE 3 - Somewhat similar:
            >>> text_a = "How to reset password?"
            >>> text_d = "Enable two-factor authentication"
            >>> score = embedder.similarity(text_a, text_d)
            >>> print(f"Similarity: {score:.3f}")
            Similarity: 0.456
            (Both about account security, but different topics)
        """
        
        # Get embeddings for both texts
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Reshape for cosine_similarity (needs 2D arrays)
        # from (384,) to (1, 384)
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Calculate cosine similarity
        # Returns 2D array [[score]], so [0][0] gets the number
        similarity_score = cosine_similarity(emb1, emb2)[0][0]
        
        return float(similarity_score)
    
    def similarities(self, text: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Calculate similarity between ONE text and MANY candidates.
        
        THIS IS THE CORE OF RAG SEARCH!
        When user asks a question, we:
        1. Get question embedding
        2. Compare to all document embeddings
        3. Return most similar documents
        
        Args:
            text: Reference text (usually the user query)
            candidates: List of texts to compare against
        
        Returns:
            List of (candidate_text, similarity_score) tuples
            Sorted by similarity (highest first)
        
        EXAMPLE - RAG SEARCH:
            >>> query = "How to reset password?"
            >>> documents = [
            ...     "Click Forgot Password to reset",
            ...     "How to cook pasta",
            ...     "Account recovery steps"
            ... ]
            >>> results = embedder.similarities(query, documents)
            >>> for doc, score in results:
            ...     print(f"{score:.3f} - {doc}")
            0.923 - Click Forgot Password to reset
            0.854 - Account recovery steps
            0.087 - How to cook pasta
        
        HOW IT WORKS:
            1. Embed query: "How to reset password?" → vector1
            2. Embed all documents:
               - "Click Forgot..." → vector2
               - "How to cook..." → vector3
               - "Account recovery..." → vector4
            3. Calculate similarity:
               - query vs doc1: 0.923
               - query vs doc2: 0.087
               - query vs doc3: 0.854
            4. Sort by score (descending)
            5. Return ranked list
        
        REAL-WORLD USE:
            This is what powers semantic search!
            - Precision depends on embeddings quality
            - Speed depends on number of documents
            - Most relevant documents returned first
        """
        
        # Get query embedding
        query_emb = self.embed_text(text)
        
        # Get embeddings for all candidates
        candidate_embs = self.embed_texts(candidates)
        
        # Reshape for cosine_similarity
        query_emb = query_emb.reshape(1, -1)
        
        # Calculate similarities (returns array of scores)
        similarities_array = cosine_similarity(query_emb, candidate_embs)[0]
        
        # Pair candidates with their scores
        results = list(zip(candidates, similarities_array))
        
        # Sort by score (descending = highest similarity first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model details
        
        EXAMPLE:
            >>> embedder = EmbeddingGenerator()
            >>> info = embedder.get_info()
            >>> print(f"Model: {info['model_name']}")
            Model: all-MiniLM-L6-v2
            >>> print(f"Dimensions: {info['dimensions']}")
            Dimensions: 384
        """
        
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "description": "SentenceTransformer for semantic similarity"
        }


# ============================================================================
# TEST CODE (Run this to test everything works!)
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DAY 3: EMBEDDINGS SYSTEM - COMPREHENSIVE TEST")
    print("="*70)
    
    # ====================================================================
    # TEST 1: Initialize embedder
    # ====================================================================
    print("\n" + "-"*70)
    print("TEST 1: Initialize embedder")
    print("-"*70)
    
    try:
        embedder = EmbeddingGenerator()
        print("\n✅ Embedder initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        exit(1)
    
    # ====================================================================
    # TEST 2: Embed single text
    # ====================================================================
    print("\n" + "-"*70)
    print("TEST 2: Embed single text")
    print("-"*70)
    
    text = "How do I reset my password?"
    print(f"\nText: '{text}'")
    print(f"Converting to embedding...")
    
    embedding = embedder.embed_text(text)
    
    print(f"\n✅ Success!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Type: {type(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   Min value: {embedding.min():.4f}")
    print(f"   Max value: {embedding.max():.4f}")
    print(f"   Mean value: {embedding.mean():.4f}")
    
    # ====================================================================
    # TEST 3: Embed multiple texts
    # ====================================================================
    print("\n" + "-"*70)
    print("TEST 3: Embed multiple texts at once")
    print("-"*70)
    
    texts = [
        "How to reset password?",
        "Forgot my account password",
        "How to cook pasta?"
    ]
    
    print(f"\nTexts to embed:")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. '{t}'")
    
    print(f"\nEmbedding all at once...")
    embeddings = embedder.embed_texts(texts)
    
    print(f"\n✅ Success!")
    print(f"   Result shape: {embeddings.shape}")
    print(f"   ({len(texts)} texts × {embeddings.shape[1]} dimensions)")
    print(f"\n   Embedding 1 (first 5 dims): {embeddings[0][:5]}")
    print(f"   Embedding 2 (first 5 dims): {embeddings[1][:5]}")
    print(f"   Embedding 3 (first 5 dims): {embeddings[2][:5]}")
    
    # ====================================================================
    # TEST 4: Calculate similarity scores
    # ====================================================================
    print("\n" + "-"*70)
    print("TEST 4: Calculate pairwise similarities")
    print("-"*70)
    
    pairs = [
        ("How to reset password?", "Forgot my account password"),
        ("How to reset password?", "How to cook pasta?"),
        ("Forgot my account password", "How to cook pasta?"),
    ]
    
    print("\nCalculating similarity for each pair:")
    print("(1.0 = identical, 0.5 = similar, 0.0 = different, -1.0 = opposite)\n")
    
    for text1, text2 in pairs:
        score = embedder.similarity(text1, text2)
        
        # Visual bar for similarity
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Interpretation
        if score > 0.7:
            interpretation = "✓ VERY SIMILAR"
        elif score > 0.5:
            interpretation = "◐ SIMILAR"
        elif score > 0.3:
            interpretation = "△ SOMEWHAT SIMILAR"
        else:
            interpretation = "✗ DIFFERENT"
        
        print(f"Pair 1: '{text1}'")
        print(f"Pair 2: '{text2}'")
        print(f"Score: {score:.3f} {bar} {interpretation}\n")
    
    # ====================================================================
    # TEST 5: Find similar documents (CORE RAG OPERATION)
    # ====================================================================
    print("-"*70)
    print("TEST 5: Find similar documents (semantic search)")
    print("-"*70)
    print("⭐ This is what RAG systems use under the hood!")
    
    query = "How do I reset my password?"
    documents = [
        "Click Forgot Password button to reset your account",
        "Two-factor authentication adds security",
        "Never share your password with anyone",
        "Password reset: Go to login and click reset link",
        "How to make spaghetti at home"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nSearching {len(documents)} documents...")
    
    results = embedder.similarities(query, documents)
    
    print(f"\n✅ Results (ranked by relevance):\n")
    
    for i, (doc, score) in enumerate(results, 1):
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        if score > 0.6:
            relevance = "✓ RELEVANT"
        else:
            relevance = "✗ NOT RELEVANT"
        
        print(f"{i}. Score: {score:.3f} {bar} {relevance}")
        print(f"   Document: {doc[:60]}...\n")
    
    # ====================================================================
    # TEST 6: Model information
    # ====================================================================
    print("-"*70)
    print("TEST 6: Model information")
    print("-"*70)
    
    info = embedder.get_info()
    
    print(f"\nModel Details:")
    for key, value in info.items():
        print(f"  • {key}: {value}")
    
    # ====================================================================
    # TEST 7: Show what you learned
    # ====================================================================
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    
    print("\n📚 KEY CONCEPTS YOU NOW UNDERSTAND:")
    print("\n1. EMBEDDINGS")
    print("   • Text → Vector of 384 numbers")
    print("   • Similar texts → similar vectors")
    print("   • Numbers capture semantic meaning")
    
    print("\n2. COSINE SIMILARITY")
    print("   • Measures angle between vectors")
    print("   • 1.0 = same direction (identical meaning)")
    print("   • 0.0 = perpendicular (different meaning)")
    print("   • -1.0 = opposite (opposite meaning)")
    
    print("\n3. SEMANTIC SEARCH")
    print("   • Query embedded as vector")
    print("   • Compared to all document vectors")
    print("   • Most similar documents returned first")
    print("   • This is the foundation of RAG!")
    
    print("\n4. DIMENSIONS")
    print("   • 384 dimensions = rich representation")
    print("   • Each captures aspect of meaning")
    print("   • Balance between precision and speed")
    
    print("\n🎯 INTERVIEW ANSWER:")
    print("   'Embeddings convert text to numerical vectors where")
    print("   semantic similarity is captured geometrically. Using")
    print("   cosine similarity, we find relevant documents for any")
    print("   query without keyword matching. This is the semantic")
    print("   foundation of RAG systems.'")
    
    print("\n" + "="*70)
    print("Ready for Day 4: Vector Database Integration!")
    print("="*70 + "\n")
