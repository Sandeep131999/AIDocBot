"""
KEYWORD SEARCH - BM25 IMPLEMENTATION
=====================================
Fast keyword-based document search using BM25 algorithm.
All settings read from .env file (UTF-8 encoding).
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class KeywordSearcher:
    """Implements BM25 keyword search for exact terminology matching."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        print(f"\n🔍 Keyword Searcher initialized (BM25)")
        print(f"   Stop words: {len(self.stop_words)}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned = [
            token for token in tokens 
            if token.isalnum() and token not in self.stop_words and len(token) > 1
        ]
        return cleaned
    
    def index_documents(self, documents: List[str]) -> None:
        """Build BM25 index from documents."""
        print(f"\n[INDEX] Indexing {len(documents)} documents...")
        
        if len(documents) == 0:
            print("⚠️  No documents!")
            return
        
        self.documents = documents
        tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✅ Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for documents using BM25."""
        if self.bm25 is None:
            raise ValueError("Must call index_documents() first!")
        
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        
        return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("KEYWORD SEARCH TEST")
    print("="*70)
    
    searcher = KeywordSearcher()
    docs = [
        "How to reset password: click Forgot Password button",
        "Enable two-factor authentication for security",
        "Never share your password with anyone",
        "Password reset: go to login page and click reset link",
        "How to make spaghetti at home"
    ]
    
    searcher.index_documents(docs)
    results = searcher.search("password reset", top_k=5)
    
    print(f"\nQuery: 'password reset'")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.2f} - {doc[:60]}...")