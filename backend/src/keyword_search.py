"""
KEYWORD SEARCH - DISK-PERSISTENT BM25 INDEX
===========================================
Persists keyword index to disk so it survives restarts.
Uses pickle to save/load BM25 index and document corpus.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
import nltk

from src.config import Config

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class KeywordSearcher:
    """
    BM25-based keyword search with disk persistence.
    
    FIX: Saves index to disk so it survives application restarts.
    """
    
    def __init__(self, index_path: str = None):
        self.index_path = index_path or os.path.join(
            os.path.dirname(Config.VECTOR_DB_PATH),
            "keyword_index.pkl"
        )
        
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        self.stop_words = set(stopwords.words('english'))
        
        # Try to load existing index
        self._load_index()
        
        if self.bm25:
            print(f"   📚 Loaded keyword index with {len(self.corpus)} documents")
        else:
            print(f"   ⚠️  No keyword index found — will build on first index_documents()")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text for BM25."""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in self.stop_words and len(t) > 1]
    
    def _save_index(self):
        """Save BM25 index and corpus to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            data = {
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
            }
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"   💾 Saved keyword index to {self.index_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to save keyword index: {e}")
    
    def _load_index(self):
        """Load BM25 index and corpus from disk."""
        if not os.path.exists(self.index_path):
            return
        
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            self.corpus = data.get('corpus', [])
            self.tokenized_corpus = data.get('tokenized_corpus', [])
            
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                print(f"   ✅ Restored keyword index from {self.index_path}")
            else:
                self.bm25 = None
        except Exception as e:
            print(f"   ⚠️  Failed to load keyword index: {e}")
            self.bm25 = None
            self.corpus = []
            self.tokenized_corpus = []
    
    def index_documents(self, documents: List[str]):
        """
        Add documents to keyword index.
        
        FIX: Appends to existing index and persists to disk.
        """
        if not documents:
            return
        
        print(f"\n[KEYWORD INDEX] Adding {len(documents)} documents...")
        
        # Add new documents
        for doc in documents:
            if doc not in self.corpus:  # Avoid duplicates
                self.corpus.append(doc)
                self.tokenized_corpus.append(self._tokenize(doc))
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Persist
        self._save_index()
        
        print(f"   ✅ Keyword index: {len(self.corpus)} total documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents matching query keywords.
        
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or not self.corpus:
            return []
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.corpus[idx], float(scores[idx])))
        
        return results
    
    def clear(self):
        """Clear the keyword index."""
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        print("🗑️  Keyword index cleared")


if __name__ == "__main__":
    searcher = KeywordSearcher()
    
    # Test
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Python is a great programming language",
        "Machine learning is fascinating"
    ]
    
    searcher.index_documents(docs)
    results = searcher.search("quick fox", top_k=2)
    
    print("\nResults:")
    for doc, score in results:
        print(f"  {score:.3f}: {doc}")