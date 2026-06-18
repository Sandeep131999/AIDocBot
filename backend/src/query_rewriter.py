"""
QUERY REWRITING
================
Use LLM to convert user questions into optimized RAG queries.
Uses Multi-LLM fallback system (Gemini → Groq → OpenRouter).
All settings read from .env file (UTF-8 encoding).
"""

from typing import List, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import Config
from src.multi_llm import MultiLLM, get_multi_llm


class QueryRewriter:
    """
    Query Rewriting using Multi-LLM with automatic fallback.
    
    Converts vague/conversational user questions into:
    1. Optimized search queries
    2. Multiple query variations
    3. Hypothetical answers (for HyDE)
    
    APPROACHES:
    - rewrite: Simple query optimization
    - expand: Generate multiple query variations
    - hyde: Hypothetical Document Embeddings
    - decompose: Break complex queries into sub-queries
    """
    
    DEFAULT_PROMPT = """You are a query optimization expert for a Retrieval-Augmented Generation (RAG) system.

Your task is to convert the user's question into an optimized search query that will retrieve the most relevant documents from a knowledge base.

Original Question: {query}

Instructions:
1. Identify the key concepts and entities in the question
2. Remove conversational filler words ("can you", "please", "I want to know", etc.)
3. Add relevant synonyms or alternative phrasings
4. Make the query specific and information-dense
5. Keep the query concise but complete

Return ONLY the optimized query as a JSON object:
{{
    "original_query": "...",
    "optimized_query": "...",
    "key_entities": ["entity1", "entity2"],
    "query_type": "factual|analytical|procedural"
}}
"""
    
    EXPAND_PROMPT = """You are a query expansion expert for a RAG system.

Given a user question, generate multiple search queries that cover different aspects and phrasings of the same information need.

Original Question: {query}

Generate 3-5 different query variations that:
1. Use different keywords and synonyms
2. Cover different angles of the question
3. Include both broad and specific phrasings

Return as JSON:
{{
    "original_query": "...",
    "variations": [
        "variation 1",
        "variation 2",
        "variation 3"
    ],
    "combined_query": "..."
}}
"""
    
    HYDE_PROMPT = """You are an expert at generating hypothetical documents for retrieval.

Given a user question, write a short hypothetical document (2-3 sentences) that would be the perfect answer to this question. This document will be used to find similar real documents in the knowledge base.

Question: {query}

Write a concise, factual hypothetical answer:
"""
    
    def __init__(self, multi_llm: MultiLLM = None):
        """
        Initialize Query Rewriter with Multi-LLM.
        
        Args:
            multi_llm: MultiLLM instance (auto-created if None)
        """
        self.multi_llm = multi_llm or get_multi_llm()
        
        # Check if any LLM is available
        health = self.multi_llm.health_check()
        available = [name for name, ok in health.items() if ok]
        
        if not available:
            print("⚠️  No LLM providers available. Query rewriting will be disabled.")
            print("   Set at least one API key in .env:")
            print("   GEMINI_API_KEY=... or GROQ_API_KEY=... or OPENROUTER_API_KEY=...")
        else:
            print(f"\n📝 Query Rewriter initialized")
            print(f"   Available providers: {available}")
            print(f"   Fallback order: {self.multi_llm.provider_order}")
    
    @property
    def llm(self):
        """Check if any LLM is available."""
        health = self.multi_llm.health_check()
        return any(health.values())
    
    def rewrite(self, query: str) -> Dict:
        """
        Rewrite a single query into an optimized form.
        
        Returns:
            {
                "original_query": str,
                "optimized_query": str,
                "key_entities": List[str],
                "query_type": str
            }
        """
        if not self.llm:
            return {
                "original_query": query,
                "optimized_query": query,
                "key_entities": [],
                "query_type": "unknown"
            }
        
        print(f"\n🔄 Rewriting query: '{query}'")
        
        prompt = PromptTemplate.from_template(self.DEFAULT_PROMPT)
        
        try:
            # Use Multi-LLM with fallback
            response = self.multi_llm.invoke(
                prompt.format(query=query),
                system_prompt="You are a query optimization expert. Return only valid JSON."
            )
            
            # Parse JSON response
            parser = JsonOutputParser()
            result = parser.parse(response.content)
            
            print(f"   ✅ Optimized: '{result.get('optimized_query', query)}'")
            print(f"   Entities: {result.get('key_entities', [])}")
            print(f"   Provider: {self.multi_llm.get_last_successful_provider()}")
            return result
            
        except Exception as e:
            print(f"   ⚠️  Rewrite failed: {e}")
            return {
                "original_query": query,
                "optimized_query": query,
                "key_entities": [],
                "query_type": "unknown"
            }
    
    def expand(self, query: str) -> List[str]:
        """
        Expand a query into multiple variations.
        
        Returns:
            List of query strings
        """
        if not self.llm:
            return [query]
        
        print(f"\n📤 Expanding query: '{query}'")
        
        prompt = PromptTemplate.from_template(self.EXPAND_PROMPT)
        
        try:
            response = self.multi_llm.invoke(
                prompt.format(query=query),
                system_prompt="You are a query expansion expert. Return only valid JSON."
            )
            
            parser = JsonOutputParser()
            result = parser.parse(response.content)
            
            variations = result.get("variations", [query])
            print(f"   ✅ Generated {len(variations)} variations")
            print(f"   Provider: {self.multi_llm.get_last_successful_provider()}")
            for i, v in enumerate(variations, 1):
                print(f"      {i}. {v}")
            return variations
            
        except Exception as e:
            print(f"   ⚠️  Expansion failed: {e}")
            return [query]
    
    def hyde(self, query: str) -> str:
        """
        Generate Hypothetical Document Embedding (HyDE).
        
        Returns:
            Hypothetical document text
        """
        if not self.llm:
            return query
        
        print(f"\n🤖 Generating HyDE for: '{query}'")
        
        try:
            response = self.multi_llm.invoke(
                self.HYDE_PROMPT.format(query=query),
                system_prompt="You are an expert at generating hypothetical documents for retrieval."
            )
            
            hypothetical_doc = response.content.strip()
            print(f"   ✅ HyDE generated ({len(hypothetical_doc)} chars)")
            print(f"   Provider: {self.multi_llm.get_last_successful_provider()}")
            return hypothetical_doc
            
        except Exception as e:
            print(f"   ⚠️  HyDE failed: {e}")
            return query
    
    def full_pipeline(self, query: str, use_hyde: bool = False) -> Dict:
        """
        Full query rewriting pipeline.
        
        Returns:
            {
                "original": str,
                "optimized": str,
                "variations": List[str],
                "hyde_document": str (optional),
                "final_queries": List[str]  # all queries to search with
            }
        """
        print(f"\n{'='*70}")
        print("QUERY REWRITING PIPELINE")
        print(f"{'='*70}")
        
        # Step 1: Rewrite
        rewrite_result = self.rewrite(query)
        optimized = rewrite_result.get("optimized_query", query)
        
        # Step 2: Expand
        variations = self.expand(optimized)
        
        # Step 3: HyDE (optional)
        hyde_doc = None
        if use_hyde:
            hyde_doc = self.hyde(query)
        
        # Combine all queries
        all_queries = list(set([query, optimized] + variations))
        
        result = {
            "original": query,
            "optimized": optimized,
            "variations": variations,
            "hyde_document": hyde_doc,
            "final_queries": all_queries,
            "key_entities": rewrite_result.get("key_entities", [])
        }
        
        print(f"\n📋 Final Queries ({len(all_queries)}):")
        for i, q in enumerate(all_queries, 1):
            print(f"   {i}. {q}")
        
        return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUERY REWRITING TEST (Multi-LLM)")
    print("="*70)
    
    rewriter = QueryRewriter()
    
    if rewriter.llm:
        test_query = "Hey, can you tell me how to reset my password? I forgot it."
        result = rewriter.full_pipeline(test_query, use_hyde=True)
        print(f"\n📊 Final Result:")
        print(f"   Original: {result['original']}")
        print(f"   Optimized: {result['optimized']}")
        print(f"   Variations: {len(result['variations'])}")
        print(f"   HyDE: {result['hyde_document'][:100]}..." if result['hyde_document'] else "   HyDE: None")
    else:
        print("\n⚠️  Skipping test - no API keys configured")