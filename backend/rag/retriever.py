"""
retriever.py — Recherche sémantique et multimodale (RRF)
========================================================

Fournit trois modes de recherche dans Qdrant :
  - retrieve()            : recherche textuelle (BGE-M3)
  - retrieve_visual()     : recherche croisée texte → image (CLIP)
  - retrieve_multimodal() : fusion texte + image via Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from backend.config import settings
from backend.rag.embeddings import EmbeddingManager
from backend.rag.qdrant_store import QdrantStore


class SemanticRetriever:
    """Moteur de recherche sémantique et multimodale sur Qdrant."""

    def __init__(
        self,
        embeddings: EmbeddingManager | None = None,
        qdrant: QdrantStore | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingManager()
        self.qdrant = qdrant or QdrantStore()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recherche sémantique pure sur les chunks textuels."""
        k = top_k or settings.rag_top_k
        query_vec = await asyncio.to_thread(self.embeddings.encode_text_single, query)
        results = await asyncio.to_thread(
            self.qdrant.search,
            settings.qdrant_collection_text,
            query_vec,
            k,
            filters,
        )
        logger.debug(f"retrieve() : {len(results)} résultat(s) pour '{query[:60]}'")
        return results

    async def retrieve_visual(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recherche d'images pertinentes à partir d'une requête textuelle (CLIP)."""
        k = top_k or settings.rag_top_k
        query_vec = await asyncio.to_thread(self.embeddings.encode_text_for_image_search, query)
        results = await asyncio.to_thread(
            self.qdrant.search,
            settings.qdrant_collection_visual,
            query_vec,
            k,
            filters,
        )
        logger.debug(f"retrieve_visual() : {len(results)} image(s) pour '{query[:60]}'")
        return results

    async def retrieve_multimodal(
        self,
        query: str,
        image_bytes: bytes | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche multimodale combinant texte et image via Reciprocal Rank Fusion.
        """
        k = top_k or settings.rag_top_k
        text_results = await self.retrieve(query=query, top_k=k, filters=filters)

        if not image_bytes:
            return text_results

        image_vec = await asyncio.to_thread(self.embeddings.encode_image, image_bytes)
        visual_results = await asyncio.to_thread(
            self.qdrant.search,
            settings.qdrant_collection_visual,
            image_vec,
            k,
            filters,
        )
        cross_modal_results = await self.retrieve_visual(query=query, top_k=k, filters=filters)

        merged = self._reciprocal_rank_fusion(text_results, visual_results, cross_modal_results)
        return merged[:k]

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Fusionne plusieurs listes classées avec la formule RRF :
            score(doc) = sum(1 / (k + rank))
        où rank est 1-based.
        """
        rrf_scores: dict[str, float] = {}
        items_by_id: dict[str, dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, item in enumerate(result_list, start=1):
                item_id = str(item["id"])
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))
                if item_id not in items_by_id:
                    items_by_id[item_id] = dict(item)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        merged: list[dict[str, Any]] = []
        for item_id in sorted_ids:
            entry = dict(items_by_id[item_id])
            entry["rrf_score"] = round(rrf_scores[item_id], 6)
            merged.append(entry)

        return merged
