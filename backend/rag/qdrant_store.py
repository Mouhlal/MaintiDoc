"""
qdrant_store.py — Wrapper Qdrant pour MaintiDoc
===============================================

Encapsule les opérations d'insertion, de recherche filtrée et de suppression
dans les collections Qdrant (texte et visuel).
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.config import settings


class QdrantStore:
    """Interface haut niveau avec la base vectorielle Qdrant."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key if api_key is not None else settings.qdrant_api_key,
            timeout=15,
            check_compatibility=False,
        )

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Crée la collection si elle n'existe pas encore."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection Qdrant créée : {collection} (dim={vector_size})")

    def upsert_chunks(self, collection: str, chunks_with_vectors: list[dict[str, Any]]) -> int:
        """
        Insère ou met à jour une liste de points dans Qdrant.

        Chaque dict doit contenir : {"id": str, "vector": list[float], "payload": dict}.
        Retourne le nombre de points insérés.
        """
        if not chunks_with_vectors:
            return 0

        points = [
            PointStruct(
                id=str(item["id"]),
                vector=item["vector"],
                payload=item.get("payload", {}),
            )
            for item in chunks_with_vectors
        ]

        batch_size = 64
        total = 0
        for idx in range(0, len(points), batch_size):
            batch = points[idx : idx + batch_size]
            self.client.upsert(collection_name=collection, points=batch, wait=True)
            total += len(batch)

        logger.info(f"Upsert de {total} points dans Qdrant collection '{collection}'")
        return total

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche les *top_k* vecteurs les plus proches avec filtres optionnels.

        Retourne une liste de dicts : [{"id": ..., "score": ..., "payload": ...}].
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": dict(point.payload or {}),
                }
            )
        return results

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        """Convertit un dictionnaire simple en filtre Qdrant FieldCondition."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )
        return Filter(must=conditions) if conditions else None

    def delete_document(self, collection: str, document_id: str) -> int:
        """Supprime tous les points appartenant à *document_id* dans *collection*."""
        doc_filter = self._build_filter({"document_id": document_id})
        if doc_filter is None:
            return 0

        count_before = self.client.count(
            collection_name=collection,
            count_filter=doc_filter,
            exact=True,
        ).count

        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=doc_filter),
            wait=True,
        )
        logger.info(f"Supprimé {count_before} points pour document_id={document_id} dans {collection}")
        return int(count_before)

    def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Retourne les informations de statut d'une collection Qdrant."""
        info = self.client.get_collection(collection)
        return {
            "name": collection,
            "points_count": info.points_count,
            "status": str(info.status),
        }
