"""
src/evaluator.py
=================
Evaluate RAG retrieval quality for AIDocBot.
(Hit@K, Precision@K, Recall@K, MRR,
NDCG@K, F1@K) exactly as-is, and adds two integration points:

1. `evaluate_retriever(_dataset)` — runs queries through the *live*
   Chroma retriever from embeddings.py, so you can score real vector
   search results instead of hand-typed id lists.
2. `evaluate_graph_run` — scaffolding for once the LangGraph nodes
   (generate_query_or_respond / retrieve / grade_documents / ...) are
   wired up. See the note on that function before using it.

Document identity = the `source` field set in document_loader.py'
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from langchain_core.documents import Document

# from retriever import retriver


def _doc_id(doc: Document) -> str:
    """Identity used to match a retrieved chunk against ground truth."""
    return doc.metadata.get("source", doc.page_content[:50])


class Evaluator:
    """Evaluate retrieval quality using ground-truth relevant documents."""

    def __init__(self):
        print("Evaluator ready")

    # ------------------------------------------------------------------
    # Core metrics — unchanged from your original implementation
    # ------------------------------------------------------------------

    def hit_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> int:
        top_k = retrieved_docs[:k]
        return sum(1 for doc in top_k if doc in relevant_docs)

    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        return self.hit_at_k(retrieved_docs, relevant_docs, k) / k

    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        if not relevant_docs:
            return 0.0
        return self.hit_at_k(retrieved_docs, relevant_docs, k) / len(relevant_docs)

    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0

    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        dcg = sum(
            1.0 / np.log2(i + 1)
            for i, doc in enumerate(retrieved_docs[:k], 1)
            if doc in relevant_docs
        )
        num_relevant = min(len(relevant_docs), k)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        return dcg / ideal_dcg if ideal_dcg else 0.0

    def f1_score_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> float:
        precision = self.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.recall_at_k(retrieved_docs, relevant_docs, k)
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def evaluate_query(self, retrieved_docs: List[str], relevant_docs: List[str], k: int = 5) -> Dict[str, float]:
        return {
            "hit_at_k": self.hit_at_k(retrieved_docs, relevant_docs, k),
            "precision_at_k": self.precision_at_k(retrieved_docs, relevant_docs, k),
            "recall_at_k": self.recall_at_k(retrieved_docs, relevant_docs, k),
            "mrr": self.mean_reciprocal_rank(retrieved_docs, relevant_docs),
            "ndcg_at_k": self.ndcg_at_k(retrieved_docs, relevant_docs, k),
            "f1_score_at_k": self.f1_score_at_k(retrieved_docs, relevant_docs, k),
        }

    def evaluate_dataset(self, queries_results: Dict, ground_truth: Dict, k: int = 5) -> Dict:
        all_metrics = []
        for query, retrieved in queries_results.items():
            if query not in ground_truth:
                continue
            all_metrics.append(self.evaluate_query(retrieved, ground_truth[query], k))

        if not all_metrics:
            return {}

        return {
            key: {
                "mean": np.mean([m[key] for m in all_metrics]),
                "std": np.std([m[key] for m in all_metrics]),
                "min": np.min([m[key] for m in all_metrics]),
                "max": np.max([m[key] for m in all_metrics]),
            }
            for key in all_metrics[0]
        }

    def print_metrics(self, metrics: Dict, query: str = None) -> None:
        if query:
            print(f"\nQuery: '{query}'")
        print(f"Hit@K: {metrics['hit_at_k']}")
        print(f"Precision@K: {metrics['precision_at_k']:.3f}")
        print(f"Recall@K: {metrics['recall_at_k']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"NDCG@K: {metrics['ndcg_at_k']:.3f}")
        print(f"F1 Score@K: {metrics['f1_score_at_k']:.3f}")

    # ------------------------------------------------------------------
    # LangChain integration — evaluate the real retriever, today
    # ------------------------------------------------------------------

    def evaluate_retriever(self, query: str, relevant_docs: List[str], k: int = 5) -> Dict[str, float]:
        """Run `query` through the live Chroma retriever and score the results."""
        results = get_retriever(k=k).invoke(query)
        retrieved_docs = [_doc_id(doc) for doc in results]
        return self.evaluate_query(retrieved_docs, relevant_docs, k)

    def evaluate_retriever_dataset(self, ground_truth: Dict[str, List[str]], k: int = 5) -> Dict:
        """
        ground_truth: {query: [relevant_doc_id, ...], ...}
        Runs every query through the live retriever — no pre-fetched results needed.
        """
        queries_results = {
            query: [_doc_id(doc) for doc in get_retriever(k=k).invoke(query)]
            for query in ground_truth
        }
        return self.evaluate_dataset(queries_results, ground_truth, k)

    # ------------------------------------------------------------------
    # LangGraph integration — for once the graph nodes exist
    # ------------------------------------------------------------------

    def evaluate_graph_run(self, graph, question: str, relevant_docs: List[str], k: int = 5) -> Dict[str, float]:
        """
        Score one end-to-end LangGraph run against ground truth.

        CAVEAT: the reference `retrieve_blog_posts` tool returns retrieved
        chunks as one joined string, which loses per-doc source metadata —
        there's nothing here to match against `relevant_docs`. For this to
        work, the retriever tool (or a node right after ToolNode) needs to
        stash the retrieved Documents' metadata somewhere state-visible,
        e.g. add a `retrieved_docs: list[Document]` key to your graph
        state and populate it alongside the ToolMessage. Once that's in
        place, swap the line below for however you read that state key.
        """
        result = graph.invoke({"messages": [{"role": "user", "content": question}]})
        retrieved = result.get("retrieved_docs", [])  # populate this in your graph state
        retrieved_docs = [_doc_id(doc) for doc in retrieved]
        return self.evaluate_query(retrieved_docs, relevant_docs, k)

