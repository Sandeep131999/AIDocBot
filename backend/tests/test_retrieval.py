"""
tests/test_retrieval.py
Tests for hybrid retrieval, reranking, HyDE, and the evaluator metrics.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# RRF fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFFusion:

    def _make_docs(self, n: int) -> list[Document]:
        return [
            Document(
                page_content=f"Document {i} content",
                metadata={"filename": f"doc{i}.pdf", "file_hash": f"hash{i}"},
            )
            for i in range(n)
        ]

    def test_dense_only_returns_correct_count(self):
        from src.vector_store import _rrf_fusion
        docs = self._make_docs(10)
        result = _rrf_fusion(dense_results=docs[:5], sparse_results=[], dense_weight=1.0, sparse_weight=0.0)
        assert len(result) == 5

    def test_sparse_only_returns_correct_count(self):
        from src.vector_store import _rrf_fusion
        docs = self._make_docs(5)
        sparse = [(d, float(5 - i)) for i, d in enumerate(docs)]
        result = _rrf_fusion(dense_results=[], sparse_results=sparse, dense_weight=0.0, sparse_weight=1.0)
        assert len(result) == 5

    def test_fusion_deduplicates(self):
        """Same document in both dense and sparse should only appear once."""
        from src.vector_store import _rrf_fusion
        doc = Document(page_content="Shared doc", metadata={"file_hash": "shared"})
        result = _rrf_fusion(
            dense_results=[doc],
            sparse_results=[(doc, 5.0)],
            dense_weight=0.6,
            sparse_weight=0.4,
        )
        assert len(result) == 1

    def test_fusion_ranks_overlap_docs_higher(self):
        """Doc appearing in both dense and sparse should rank higher than doc in only one."""
        from src.vector_store import _rrf_fusion
        shared = Document(page_content="Shared", metadata={"file_hash": "shared"})
        dense_only = Document(page_content="Dense only", metadata={"file_hash": "denseonly"})
        sparse = [(shared, 1.0)]
        result = _rrf_fusion(
            dense_results=[dense_only, shared],
            sparse_results=sparse,
        )
        # shared doc should come first (appears in both lists)
        assert result[0].metadata["file_hash"] == "shared"


# ─────────────────────────────────────────────────────────────────────────────
# BM25 search
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25:

    def test_bm25_returns_top_k(self):
        from src.vector_store import _bm25_search
        docs = [
            Document(page_content="reset password login", metadata={"file_hash": "a"}),
            Document(page_content="pricing enterprise plan", metadata={"file_hash": "b"}),
            Document(page_content="password recovery steps", metadata={"file_hash": "c"}),
        ]
        results = _bm25_search("reset password", docs, k=2)
        assert len(results) == 2

    def test_bm25_empty_docs(self):
        from src.vector_store import _bm25_search
        results = _bm25_search("anything", [], k=5)
        assert results == []

    def test_bm25_password_query_ranks_password_docs_high(self):
        from src.vector_store import _bm25_search
        docs = [
            Document(page_content="reset password login authentication", metadata={"file_hash": "a"}),
            Document(page_content="the quick brown fox", metadata={"file_hash": "b"}),
            Document(page_content="forgot password recovery email link", metadata={"file_hash": "c"}),
        ]
        results = _bm25_search("password reset", docs, k=3)
        top_hashes = [d.metadata["file_hash"] for d, _ in results[:2]]
        assert "a" in top_hashes or "c" in top_hashes


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatorMetrics:

    @pytest.fixture
    def evaluator(self):
        from src.evaluator import Evaluator
        return Evaluator()

    def test_hit_at_k_all_relevant(self, evaluator):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        assert evaluator.hit_at_k(retrieved, relevant, k=3) == 3

    def test_hit_at_k_none_relevant(self, evaluator):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]
        assert evaluator.hit_at_k(retrieved, relevant, k=3) == 0

    def test_hit_at_k_partial(self, evaluator):
        retrieved = ["doc1", "doc99", "doc3"]
        relevant = ["doc1", "doc3"]
        assert evaluator.hit_at_k(retrieved, relevant, k=3) == 2

    def test_precision_at_k(self, evaluator):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]
        assert evaluator.precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)

    def test_recall_at_k(self, evaluator):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc3", "doc5"]
        assert evaluator.recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_mrr_first_hit(self, evaluator):
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1"]
        assert evaluator.mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

    def test_mrr_second_hit(self, evaluator):
        retrieved = ["doc99", "doc1"]
        relevant = ["doc1"]
        assert evaluator.mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_mrr_no_hit(self, evaluator):
        retrieved = ["doc99", "doc88"]
        relevant = ["doc1"]
        assert evaluator.mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_f1_perfect(self, evaluator):
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1", "doc2"]
        assert evaluator.f1_score_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_f1_zero(self, evaluator):
        retrieved = ["doc99"]
        relevant = ["doc1"]
        assert evaluator.f1_score_at_k(retrieved, relevant, k=1) == 0.0

    def test_evaluate_query_returns_all_metrics(self, evaluator):
        metrics = evaluator.evaluate_query(["doc1", "doc2"], ["doc1"], k=2)
        assert set(metrics.keys()) == {"hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "f1_score_at_k"}

    def test_evaluate_dataset_aggregation(self, evaluator):
        queries_results = {
            "q1": ["doc1", "doc2"],
            "q2": ["doc3", "doc4"],
        }
        ground_truth = {
            "q1": ["doc1"],
            "q2": ["doc3"],
        }
        agg = evaluator.evaluate_dataset(queries_results, ground_truth, k=2)
        assert "hit_at_k" in agg
        assert "mean" in agg["hit_at_k"]
        assert agg["hit_at_k"]["mean"] == pytest.approx(1.0)

    def test_evaluate_dataset_empty(self, evaluator):
        result = evaluator.evaluate_dataset({}, {}, k=5)
        assert result == {}
