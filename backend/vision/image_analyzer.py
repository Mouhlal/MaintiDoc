"""
image_analyzer.py — Analyse combinée VLM + RAG visuel
====================================================

Combine l'analyse visuelle par VLM (LLaVA) avec la recherche
de figures et schémas similaires dans Qdrant (CLIP).
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from backend.llm.prompts import IMAGE_ANALYSIS_PROMPT
from backend.rag.retriever import SemanticRetriever
from backend.vision.vlm_client import VLMClient, VLMResponse, vlm_client


class ImageAnalyzer:
    """Orchestre l'analyse d'une photo d'équipement et son croisement documentaire."""

    def __init__(
        self,
        vlm: VLMClient | None = None,
        retriever: SemanticRetriever | None = None,
    ) -> None:
        self.vlm = vlm or vlm_client
        self.retriever = retriever or SemanticRetriever()

    async def analyze(self, image_bytes: bytes, question: str | None = None) -> dict[str, Any]:
        """
        Analyse l'image avec le VLM puis cherche des documents visuels proches.
        """
        prompt = IMAGE_ANALYSIS_PROMPT
        if question and question.strip():
            prompt = f"{IMAGE_ANALYSIS_PROMPT}\n\nQUESTION SPÉCIFIQUE DU TECHNICIEN : {question.strip()}"

        vlm_response = await asyncio.to_thread(self.vlm.analyze_image, image_bytes, prompt)

        visual_results: list[dict[str, Any]] = []
        try:
            search_query = question or vlm_response.description[:300]
            visual_results = await self.retriever.retrieve_multimodal(
                query=search_query,
                image_bytes=image_bytes,
                top_k=5,
            )
        except Exception as exc:
            logger.warning(f"Recherche visuelle Qdrant ignorée : {exc}")

        return self._enrich_with_docs(vlm_response, visual_results)

    @staticmethod
    def _enrich_with_docs(
        vlm_response: VLMResponse,
        visual_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Fusionne la sortie VLM avec les métadonnées des documents similaires."""
        similar_docs = [
            {
                "id": res.get("id"),
                "score": res.get("rrf_score", res.get("score", 0.0)),
                "document_id": res.get("payload", {}).get("document_id", ""),
                "page_number": res.get("payload", {}).get("page_number", 0),
                "storage_path": res.get("payload", {}).get("storage_path", ""),
            }
            for res in visual_results
        ]

        return {
            "vlm_description": vlm_response.description,
            "detected_issues": vlm_response.detected_issues,
            "similar_docs": similar_docs,
            "confidence": vlm_response.confidence,
            "raw_vlm_response": vlm_response,
        }
