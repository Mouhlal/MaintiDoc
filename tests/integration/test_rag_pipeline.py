"""
test_rag_pipeline.py — Tests d'intégration du pipeline RAG (RRF, ContextBuilder, Citations)
"""

from __future__ import annotations

from backend.rag.retriever import SemanticRetriever


def test_reciprocal_rank_fusion_merges_text_and_visual():
    text_hits = [
        {"id": "a", "score": 0.9, "payload": {"content": "A"}},
        {"id": "b", "score": 0.8, "payload": {"content": "B"}},
    ]
    visual_hits = [
        {"id": "b", "score": 0.95, "payload": {"content": "B"}},
        {"id": "c", "score": 0.7, "payload": {"content": "C"}},
    ]

    merged = SemanticRetriever._reciprocal_rank_fusion(text_hits, visual_hits, k=60)
    # "b" apparaît dans les deux listes (rang 2 et rang 1) → doit être premier !
    assert merged[0]["id"] == "b"
    assert len(merged) == 3
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]
