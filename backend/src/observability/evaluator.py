"""
src/evaluator.py — Enterprise RAG Evaluator
─────────────────────────────────────────────
Two evaluation tracks:

Track 1 — Retrieval metrics (offline, no LLM cost):
  Hit@K, Precision@K, Recall@K, MRR, NDCG@K, F1@K
  Works with live Chroma retriever or pre-fetched results.

Track 2 — RAGAS end-to-end metrics (LLM-based):
  faithfulness, answer_relevancy, context_precision, context_recall
  Uses ragas library with the configured LLM.

Also exposes:
  • golden_dataset_eval()  — run against a curated ground-truth dataset
  • evaluate_graph_run()   — end-to-end graph evaluation (now with retrieved_docs in state)
"""
from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document

import asyncio
from src.retrieval.vector_store import get_retriever

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _doc_id(doc: Document) -> str:
    """Return the source record identity used by evaluation labels."""
    return (
        doc.metadata.get("document_id")
        or doc.metadata.get("url")
        or doc.metadata.get("label")
        or doc.metadata.get("file_hash")
        or doc.metadata.get("source")
        or doc.page_content[:80]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Track 1 — Classical retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Classical retrieval quality metrics with live Chroma + LangGraph integration."""

    def __init__(self) -> None:
        logger.info("[Evaluator] Initialized")

    # ── Core metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def _relevant_count(retrieved: List[str], relevant: List[str], k: int) -> int:
        relevant_ids = set(relevant)
        unique_retrieved = dict.fromkeys(retrieved[:k])
        return sum(document_id in relevant_ids for document_id in unique_retrieved)

    def hit_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> int:
        """Return 1 when any relevant document appears in the top-k, else 0."""
        return int(k > 0 and self._relevant_count(retrieved, relevant, k) > 0)

    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        return self._relevant_count(retrieved, relevant, k) / k if k > 0 else 0.0

    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        relevant_count = len(set(relevant))
        if relevant_count == 0:
            return 0.0
        return self._relevant_count(retrieved, relevant, k) / relevant_count

    def mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        relevant_ids = set(relevant)
        for rank, document_id in enumerate(dict.fromkeys(retrieved), 1):
            if document_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        relevant_ids = set(relevant)
        unique_retrieved = list(dict.fromkeys(retrieved))
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, doc in enumerate(unique_retrieved[:k], 1)
            if doc in relevant_ids
        )
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        return float(dcg / ideal) if ideal else 0.0

    def f1_score_at_k(self, retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def evaluate_query(self, retrieved: List[str], relevant: List[str], k: int = 5) -> Dict[str, float]:
        return {
            "hit_at_k":        float(self.hit_at_k(retrieved, relevant, k)),
            "precision_at_k":  self.precision_at_k(retrieved, relevant, k),
            "recall_at_k":     self.recall_at_k(retrieved, relevant, k),
            "mrr":             self.mean_reciprocal_rank(retrieved, relevant),
            "ndcg_at_k":       self.ndcg_at_k(retrieved, relevant, k),
            "f1_score_at_k":   self.f1_score_at_k(retrieved, relevant, k),
        }

    def evaluate_dataset(
        self,
        queries_results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        all_metrics = []
        for query, retrieved in queries_results.items():
            if query not in ground_truth:
                continue
            all_metrics.append(self.evaluate_query(retrieved, ground_truth[query], k))

        if not all_metrics:
            return {}

        return {
            key: {
                "mean": float(np.mean([m[key] for m in all_metrics])),
                "std":  float(np.std([m[key] for m in all_metrics])),
                "min":  float(np.min([m[key] for m in all_metrics])),
                "max":  float(np.max([m[key] for m in all_metrics])),
            }
            for key in all_metrics[0]
        }

    def print_metrics(self, metrics: Dict[str, Any], query: str = "") -> None:
        if query:
            logger.info(f"Query: '{query}'")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")

    # ── Live retriever integration ────────────────────────────────────────────

    def evaluate_retriever(
        self, query: str, relevant_docs: List[str], k: int = 5
    ) -> Dict[str, float]:
        """Run query through live Chroma EnterpriseRetriever and score."""
        results = get_retriever(k=k).invoke(query)
        retrieved = [_doc_id(doc) for doc in results]
        return self.evaluate_query(retrieved, relevant_docs, k)

    def evaluate_retriever_dataset(
        self, ground_truth: Dict[str, List[str]], k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Batch evaluate retriever against a ground truth dict."""
        queries_results = {
            query: [_doc_id(doc) for doc in get_retriever(k=k).invoke(query)]
            for query in ground_truth
        }
        return self.evaluate_dataset(queries_results, ground_truth, k)

    # ── LangGraph integration (now with retrieved_docs in AgentState) ─────────

    def evaluate_graph_run(
        self,
        graph,
        question: str,
        relevant_docs: List[str],
        k: int = 5,
        thread_id: str = "eval",
    ) -> Dict[str, float]:
        """
        Score one end-to-end LangGraph run.
        requires AgentState.retrieved_docs (Annotated[List[Document]]).
        """
        import asyncio
        config = {"configurable": {"thread_id": thread_id}}
        result = asyncio.run(graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "question": question},
            config=config,
        ))
        retrieved_doc_objects = result.get("retrieved_docs", [])
        retrieved = [_doc_id(doc) for doc in retrieved_doc_objects]
        metrics = self.evaluate_query(retrieved, relevant_docs, k)
        logger.info(f"[Evaluator] Graph run metrics: {metrics}")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Track 2 — RAGAS end-to-end evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    RAGAS evaluation for faithfulness, answer relevancy,
    context precision, and context recall.

    Args:
        questions:    List of user questions
        answers:      List of generated answers (one per question)
        contexts:     List of context lists (retrieved docs per question)
        ground_truths: Optional reference answers for recall calculation

    Returns:
        Dict of metric_name → score (0-1)
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as exc:
        logger.error("[RAGAS] Evaluation dependencies unavailable: %s", exc)
        return {"_error": f"RAGAS dependencies unavailable: {exc}"}

    if not questions or len(questions) != len(answers) or len(questions) != len(contexts):
        return {"_error": "questions, answers, and contexts must have equal non-zero lengths"}

    data: Dict[str, Any] = {
        "question":  questions,
        "answer":    answers,
        "contexts":  contexts,
    }
    metrics_to_run = [faithfulness, answer_relevancy, context_precision]

    if ground_truths and len(ground_truths) == len(questions) and all(ground_truths):
        data["ground_truth"] = ground_truths
        metrics_to_run.append(context_recall)

    try:
        dataset = Dataset.from_dict(data)
        # Use our configured LLM for RAGAS evaluation
        from src.routing.multi_llm import get_ragas_llm
        from src.retrieval.embeddings import get_embeddings

        result = await asyncio.to_thread(
            ragas_evaluate,
            dataset=dataset,
            metrics=metrics_to_run,
            llm=get_ragas_llm(),
            embeddings=get_embeddings(),
        )
        scores = result.to_pandas().mean().to_dict()
        logger.info("[RAGAS] Scores: %s", scores)
        return {
            str(name): float(value)
            for name, value in scores.items()
            if isinstance(value, (int, float))
        }
    except Exception as exc:
        logger.exception("[RAGAS] Evaluation failed")
        return {"_error": str(exc)}


async def golden_dataset_eval(
    dataset: List[Dict[str, Any]],
    graph=None,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run RAGAS + retrieval metrics over a golden dataset.

    Dataset format:
        [{"question": str, "ground_truth": str, "relevant_doc_ids": [str]}, ...]

    Returns combined metrics dict.
    """
    questions = [item["question"] for item in dataset]
    ground_truths = [item.get("ground_truth", "") for item in dataset]
    relevant_ids = [item.get("relevant_doc_ids", []) for item in dataset]

    # Run through live graph if provided
    answers: List[str] = []
    contexts: List[List[str]] = []
    retrieval_eval = Evaluator()
    retrieval_metrics_list = []

    if graph:
        for i, question in enumerate(questions):
            try:
                import asyncio
                config = {"configurable": {"thread_id": f"golden-eval-{i}"}}
                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": question}], "question": question},
                    config=config,
                )
                answer = result.get("answer", "")
                retrieved_docs = result.get("retrieved_docs", [])
                context_texts = [doc.page_content for doc in retrieved_docs]
                answers.append(str(answer))
                contexts.append(context_texts)

                if relevant_ids[i]:
                    m = retrieval_eval.evaluate_query(
                        [_doc_id(d) for d in retrieved_docs],
                        relevant_ids[i],
                        k=k,
                    )
                    retrieval_metrics_list.append(m)
            except Exception as e:
                logger.error(f"[GoldenEval] Error on question {i}: {e}")
                answers.append("")
                contexts.append([])

    # RAGAS metrics
    ragas_result = await run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths if all(ground_truths) else None,
    ) if graph is not None else {"_error": "A graph is required for RAGAS evaluation"}
    ragas_error = ragas_result.pop("_error", None)

    # Aggregate retrieval metrics
    retrieval_agg = {}
    if retrieval_metrics_list:
        import numpy as np
        for key in retrieval_metrics_list[0]:
            vals = [m[key] for m in retrieval_metrics_list]
            retrieval_agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
            }

    return {
        "ragas": ragas_result,
        "ragas_error": ragas_error,
        "retrieval": retrieval_agg,
        "num_questions": len(questions),
        "k": k,
    }
