"""
KEYWORD SEARCH - BM25 IMPLEMENTATION
===================================

Fast keyword-based document search using BM25 algorithm.

BM25 = Best-matching (25) ranking algorithm
Good at: Exact terminology, technical terms, proper nouns
Speed: Very fast (no embeddings needed)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from rank_bm25 import BM25Okapi
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')


# ============================================================================
# KEYWORD SEARCHER CLASS
# ============================================================================

class KeywordSearcher:
    """
    Implements BM25 keyword search.
    
    WHAT IT DOES:
    1. Index documents by keywords
    2. Search by exact word matches
    3. Rank by word frequency + importance
    4. Return top K results
    
    GOOD FOR:
    - "SMTP port 587" → finds exact mention
    - "two-factor authentication" → finds exact phrase
    - Technical terminology
    - Proper nouns
    """
    
    def __init__(self):
        """Initialize BM25 searcher."""
        
        self.bm25 = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        
        print("\n" + "="*70)
        print("KEYWORD SEARCHER - INITIALIZATION")
        print("="*70)
        print(f"\n✅ BM25 initialized")
        print(f"   Stop words loaded: {len(self.stop_words)} common words")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and clean text.
        
        PROCESS:
        1. Lowercase
        2. Split into words
        3. Remove stop words (the, a, is, etc.)
        4. Remove short words (<2 chars)
        5. Return cleaned tokens
        
        Example:
            "How to reset your password?"
            → ["reset", "password"]
        """
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Clean: remove stop words, keep alphanumeric, length > 1
        cleaned = [
            token for token in tokens 
            if token.isalnum() and token not in self.stop_words and len(token) > 1
        ]
        
        return cleaned
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of text strings to index
        
        PROCESS:
        1. Store documents
        2. Tokenize all
        3. Build BM25 index
        """
        
        print(f"\n[INDEX DOCUMENTS]")
        print(f"Indexing {len(documents)} documents...")
        
        if len(documents) == 0:
            print("⚠️  No documents!")
            return
        
        self.documents = documents
        
        # Tokenize all documents
        tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✅ Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        """
        
        if self.bm25 is None:
            raise ValueError("Must call index_documents() first!")
        
        # Tokenize query
        query_tokens = self._preprocess_text(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top K indices
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Create results
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        
        return results


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("KEYWORD SEARCH - TESTS")
    print("="*70)
    
    # Initialize
    searcher = KeywordSearcher()
    
    # Test documents
    docs = [
        "How to reset password: click Forgot Password button",
        "Enable two-factor authentication for security",
        "Never share your password with anyone",
        "Password reset: go to login page and click reset link",
        "How to make spaghetti at home"
    ]
    
    # Index
    print("\n[TEST 1] Index documents")
    print("-"*70)
    searcher.index_documents(docs)
    
    # Search
    print("\n[TEST 2] Search for password reset")
    print("-"*70)
    
    query = "password reset"
    results = searcher.search(query, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print(f"\nResults:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.2f} - {doc[:60]}...")
    
    print("\n✅ Keyword search working!")