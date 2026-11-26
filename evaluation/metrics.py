"""
metrics.py — Métriques d'évaluation RAG pour MaintiDoc
=======================================================

Implémente les métriques standards de recherche d'information :
  - Recall@K (ex: Recall@5)
  - nDCG@K   (ex: nDCG@5)
  - MRR@K    (ex: MRR@10)
  - Citation Validity Rate
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 5) -> float:
    """
    Proportion des documents pertinents retrouvés dans le top-k.
    Retourne 1.0 si relevant_ids est vide.
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    """
    Mean Reciprocal Rank sur le top-k : 1 / rang du premier document pertinent.
    Retourne 0.0 si aucun document pertinent dans le top-k.
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain @ K (pertinence binaire 0/1).
    """
    if not relevant_ids:
        return 1.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_batch(
    queries_results: list[tuple[Sequence[str], set[str]]],
) -> dict[str, float]:
    """
    Calcule Recall@5, nDCG@5 et MRR@10 moyens sur un jeu de requêtes.
    """
    if not queries_results:
        return {"recall@5": 0.0, "ndcg@5": 0.0, "mrr@10": 0.0}

    n = len(queries_results)
    recall_5 = sum(recall_at_k(r, rel, k=5) for r, rel in queries_results) / n
    ndcg_5 = sum(ndcg_at_k(r, rel, k=5) for r, rel in queries_results) / n
    mrr_10 = sum(mrr_at_k(r, rel, k=10) for r, rel in queries_results) / n

    return {
        "recall@5": round(recall_5, 4),
        "ndcg@5": round(ndcg_5, 4),
        "mrr@10": round(mrr_10, 4),
    }
