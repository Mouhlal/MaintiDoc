"""
context_builder.py — Construction du contexte et des références pour le LLM
===========================================================================

Transforme les résultats rerankés en :
  1. Une chaîne de contexte numérotée [Source N - Document: X, Page: Y]
  2. Une liste de ContextItem pour la vérification des citations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextItem:
    """Élément de contexte sourcé transmis au LLM et au CitationChecker."""

    chunk_id: str
    content: str
    page_number: int
    document_id: str
    chunk_type: str
    score: float
    rerank_score: float | None = None
    filename: str = ""


class ContextBuilder:
    """Formate les passages récupérés dans un budget de tokens borné."""

    def build(
        self,
        query: str,
        results: list[dict[str, Any]],
        max_tokens: int = 3000,
    ) -> tuple[str, list[ContextItem]]:
        """
        Construit le contexte formaté et la liste des items conservés.
        """
        del query  # conservé pour signature uniforme et extensions futures
        if not results:
            return "", []

        items: list[ContextItem] = []
        for res in results:
            payload = res.get("payload", {})
            content = payload.get("content") or payload.get("content_preview") or ""
            if not content.strip():
                continue

            items.append(
                ContextItem(
                    chunk_id=str(res.get("id", "")),
                    content=content.strip(),
                    page_number=int(payload.get("page_number", 0) or 0),
                    document_id=str(payload.get("document_id", "inconnu")),
                    chunk_type=str(payload.get("chunk_type", "text")),
                    score=float(res.get("score", 0.0)),
                    rerank_score=(
                        float(res["rerank_score"]) if res.get("rerank_score") is not None else None
                    ),
                    filename=str(payload.get("filename", "")),
                )
            )

        formatted = self._format_context(items)
        truncated = self._truncate_to_tokens(formatted, max_tokens=max_tokens)
        return truncated, items

    def _format_context(self, items: list[ContextItem]) -> str:
        """Formate chaque passage avec son en-tête de citation numéroté."""
        blocks: list[str] = []
        for idx, item in enumerate(items, start=1):
            doc_label = item.filename or item.document_id
            header = f"[Source {idx} - Document: {doc_label}, Page: {item.page_number}]"
            blocks.append(f"{header}\n{item.content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Tronque approximativement à max_tokens * 4 caractères."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit("\n", 1)[0] + "\n..."
