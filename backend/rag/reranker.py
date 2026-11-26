"""
reranker.py — Reranking sémantique via CrossEncoder
===================================================

Ré-ordonne les chunks candidats retournés par Qdrant à l'aide
d'un modèle CrossEncoder léger (~90 Mo) qui évalue conjointement
la requête et chaque passage.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from backend.config import settings


class Reranker:
    """Reranker CrossEncoder avec chargement paresseux."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model: Any = None

    def _get_model(self) -> Any:
        """Charge le modèle CrossEncoder au premier appel."""
        if self._model is None:
            logger.info(f"Chargement du reranker : {self.model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Attribue un `rerank_score` à chaque résultat et retourne le top_k trié.
        """
        if not results:
            return []

        k = top_k or settings.rag_top_k_reranked

        # Extraire uniquement les résultats disposant d'un contenu textuel
        pairs: list[tuple[str, str]] = []
        valid_indices: list[int] = []
        for idx, res in enumerate(results):
            payload = res.get("payload", {})
            text = payload.get("content") or payload.get("content_preview") or ""
            if text.strip():
                pairs.append((query, text))
                valid_indices.append(idx)

        if not pairs:
            return results[:k]

        model = await asyncio.to_thread(self._get_model)
        scores = await asyncio.to_thread(model.predict, pairs)

        scored_results: list[dict[str, Any]] = []
        for idx, score in zip(valid_indices, scores):
            item = dict(results[idx])
            item["rerank_score"] = float(score)
            scored_results.append(item)

        scored_results.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        logger.debug(f"Rerank terminé : {len(results)} → top {min(k, len(scored_results))}")
        return scored_results[:k]
