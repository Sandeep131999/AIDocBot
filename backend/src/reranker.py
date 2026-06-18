"""
RE-RANKING
===========
Use LLM to sub-select and re-rank RAG results.
Uses Multi-LLM fallback system (Gemini → Groq → OpenRouter).
All settings read from .env file (UTF-8 encoding).
"""

from typing import List, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import numpy as np

from src.config import Config
from src.multi_llm import MultiLLM, get_multi_llm


class Reranker:
    """
    LLM-based Re-ranking for RAG results with Multi-LLM fallback.
    
    APPROACHES:
    1. Point-wise: Score each document individually
    2. Pair-wise: Compare documents pairwise
    3. List-wise: Rank entire list at once
    
    Default: Point-wise (fastest, good quality)
    """
    
    POINTWISE_PROMPT = """You are an expert document relevance evaluator for a RAG system.

Evaluate how relevant the following document is to answering the user's query.

Query: {query}

Document:
{document}

Score the relevance on a scale of 0-10:
- 0-3: Not relevant or only tangentially related
- 4-6: Partially relevant, contains some useful information
- 7-8: Highly relevant, directly answers parts of the query
- 9-10: Perfect match, completely answers the query

Also provide a brief explanation (1 sentence) for your score.

Return as JSON:
{{
    "score": 0-10,
    "explanation": "..."
}}
"""
    
    LISTWISE_PROMPT = """You are an expert document ranking system for RAG.

Given a user query and a list of retrieved documents, re-rank them from most to least relevant.

Query: {query}

Documents:
{documents}

Return the re-ranked document indices (0-based) as JSON:
{{
    "ranking": [2, 0, 1, 3],
    "explanations": [
        "Document 2 is most relevant because...",
        "Document 0 is second because..."
    ]
}}
"""
    
    def __init__(self, multi_llm: MultiLLM = None,
                 top_n: int = None, enabled: bool = None):
        """
        Initialize Re-ranker with Multi-LLM.
        
        Args:
            multi_llm: MultiLLM instance (auto-created if None)
            top_n: Number of top results to return
            enabled: Whether re-ranking is enabled
        """
        self.multi_llm = multi_llm or get_multi_llm()
        self.top_n = top_n or Config.RERANKER_TOP_N
        self.enabled = enabled if enabled is not None else Config.RERANKER_ENABLED
        
        # Check if any LLM is available
        health = self.multi_llm.health_check()
        available = [name for name, ok in health.items() if ok]
        
        if not available:
            print("⚠️  No LLM providers available. Re-ranking will be disabled.")
            print("   Set at least one API key in .env:")
            print("   GEMINI_API_KEY=... or GROQ_API_KEY=... or OPENROUTER_API_KEY=...")
        else:
            print(f"\n🎯 Re-ranker initialized")
            print(f"   Available providers: {available}")
            print(f"   Top-N: {self.top_n}")
            print(f"   Enabled: {self.enabled}")
    
    @property
    def llm(self):
        """Check if any LLM is available."""
        health = self.multi_llm.health_check()
        return any(health.values())
    
    def pointwise_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Point-wise re-ranking: Score each document individually.
        
        Args:
            query: User query
            results: List of result dicts with 'document' and 'combined_score'
        
        Returns:
            Re-ranked results with LLM scores
        """
        if not self.llm or not self.enabled:
            return results
        
        print(f"\n🎯 Point-wise Re-ranking {len(results)} documents...")
        
        prompt = PromptTemplate.from_template(self.POINTWISE_PROMPT)
        
        scored_results = []
        
        for i, result in enumerate(results):
            doc_text = result['document']
            # Truncate long documents
            if len(doc_text) > 1000:
                doc_text = doc_text[:1000] + "..."
            
            try:
                response = self.multi_llm.invoke(
                    prompt.format(query=query, document=doc_text),
                    system_prompt="You are a document relevance evaluator. Return only valid JSON."
                )
                
                parser = JsonOutputParser()
                score_result = parser.parse(response.content)
                
                llm_score = score_result.get("score", 5) / 10.0  # Normalize to 0-1
                explanation = score_result.get("explanation", "")
                
                # Combine original score with LLM score
                original_score = result.get('combined_score', 0.5)
                final_score = 0.6 * llm_score + 0.4 * original_score
                
                scored_results.append({
                    **result,
                    "llm_score": llm_score,
                    "final_score": final_score,
                    "explanation": explanation,
                    "llm_provider": self.multi_llm.get_last_successful_provider()
                })
                
                print(f"   Doc {i+1}: LLM={llm_score:.2f} | Final={final_score:.3f} | {explanation[:50]}...")
                
            except Exception as e:
                print(f"   Doc {i+1}: Error - {e}")
                scored_results.append({
                    **result,
                    "llm_score": 0.5,
                    "final_score": result.get('combined_score', 0.5),
                    "explanation": "Scoring failed",
                    "llm_provider": "failed"
                })
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_results[:self.top_n]
    
    def listwise_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        List-wise re-ranking: Rank all documents at once.
        More accurate but slower and more expensive.
        """
        if not self.llm or not self.enabled or len(results) <= 1:
            return results
        
        print(f"\n🎯 List-wise Re-ranking {len(results)} documents...")
        
        # Format documents for prompt
        docs_text = ""
        for i, result in enumerate(results):
            doc_text = result['document']
            if len(doc_text) > 500:
                doc_text = doc_text[:500] + "..."
            docs_text += f"\n[{i}] {doc_text}\n"
        
        prompt = PromptTemplate.from_template(self.LISTWISE_PROMPT)
        
        try:
            response = self.multi_llm.invoke(
                prompt.format(query=query, documents=docs_text),
                system_prompt="You are a document ranking expert. Return only valid JSON."
            )
            
            parser = JsonOutputParser()
            ranking_result = parser.parse(response.content)
            
            ranking = ranking_result.get("ranking", list(range(len(results))))
            explanations = ranking_result.get("explanations", [])
            
            # Reorder results
            reordered = []
            for idx in ranking:
                if 0 <= idx < len(results):
                    result = results[idx]
                    result['final_score'] = 1.0 - (len(reordered) * 0.1)  # Decreasing scores
                    result['llm_rank'] = len(reordered) + 1
                    if len(reordered) < len(explanations):
                        result['explanation'] = explanations[len(reordered)]
                    result['llm_provider'] = self.multi_llm.get_last_successful_provider()
                    reordered.append(result)
            
            print(f"   ✅ Re-ranked: {ranking}")
            print(f"   Provider: {self.multi_llm.get_last_successful_provider()}")
            return reordered[:self.top_n]
            
        except Exception as e:
            print(f"   ⚠️  List-wise re-ranking failed: {e}")
            return results[:self.top_n]
    
    def rerank(self, query: str, results: List[Dict], 
               method: str = "pointwise") -> List[Dict]:
        """
        Main re-ranking interface.
        
        Args:
            query: User query
            results: Results from hybrid search
            method: 'pointwise' or 'listwise'
        
        Returns:
            Top-N re-ranked results
        """
        if not results:
            return []
        
        print(f"\n{'='*70}")
        print("RE-RANKING")
        print(f"{'='*70}")
        print(f"   Query: {query}")
        print(f"   Method: {method}")
        print(f"   Input results: {len(results)}")
        print(f"   Target Top-N: {self.top_n}")
        
        if method == "pointwise":
            reranked = self.pointwise_rerank(query, results)
        elif method == "listwise":
            reranked = self.listwise_rerank(query, results)
        else:
            reranked = results[:self.top_n]
        
        print(f"\n📊 Re-ranking Complete:")
        print(f"   Output: {len(reranked)} documents")
        for i, r in enumerate(reranked, 1):
            score = r.get('final_score', r.get('combined_score', 0))
            provider = r.get('llm_provider', 'N/A')
            print(f"   {i}. Score: {score:.3f} [{provider}] - {r['document'][:80]}...")
        
        return reranked


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RE-RANKING TEST (Multi-LLM)")
    print("="*70)
    
    reranker = Reranker()
    
    if reranker.llm:
        # Mock results
        mock_results = [
            {"document": "Password reset guide: Click forgot password on login page.", "combined_score": 0.85},
            {"document": "Two-factor authentication setup instructions.", "combined_score": 0.72},
            {"document": "How to change your password in settings.", "combined_score": 0.68},
            {"document": "Security best practices for online accounts.", "combined_score": 0.55}
        ]
        
        query = "How do I reset my password?"
        reranked = reranker.rerank(query, mock_results, method="pointwise")
    else:
        print("\n⚠️  Skipping test - no API keys configured")