"""
test_metrics.py — Tests unitaires pour les métriques d'évaluation RAG
"""

from __future__ import annotations

from evaluation.metrics import evaluate_retrieval_batch, mrr_at_k, ndcg_at_k, recall_at_k


def test_retrieval_metrics_perfect_ranking():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d2"}

    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert mrr_at_k(retrieved, relevant, k=10) == 1.0
    assert ndcg_at_k(retrieved, relevant, k=5) == 1.0


def test_retrieval_metrics_partial_and_miss():
    retrieved = ["x1", "x2", "d1", "x3", "x4"]
    relevant = {"d1", "d2"}

    assert recall_at_k(retrieved, relevant, k=5) == 0.5
    assert mrr_at_k(retrieved, relevant, k=10) == pytest_approx(1 / 3)
    assert 0.0 < ndcg_at_k(retrieved, relevant, k=5) < 1.0

    batch = evaluate_retrieval_batch([(retrieved, relevant)])
    assert batch["recall@5"] == 0.5


def pytest_approx(val: float, tol: float = 1e-6) -> float:
    import pytest

    return pytest.approx(val, abs=tol)
