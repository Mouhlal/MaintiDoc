"""
citation_checker.py — Vérification des citations et du risque d'hallucination
==============================================================================

Vérifie que :
  1. Chaque [Source N - ...] cité existe bien dans `context_items`.
  2. Le contenu de la réponse est ancré (grounded) dans le vocabulaire des sources.
"""

from __future__ import annotations

import re
from typing import Any


_SOURCE_NUM_PATTERN = re.compile(r"\[Source\s+(\d+)([^\]]*)\]", re.IGNORECASE)
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "en",
    "à", "au", "aux", "pour", "par", "avec", "sur", "dans", "est", "sont",
    "ce", "cette", "ces", "qui", "que", "il", "elle", "nous", "vous", "ne",
    "pas", "plus", "selon", "source", "page", "document",
}


class CitationChecker:
    """Vérifie la fidélité des citations et l'ancrage factuel des réponses."""

    def verify(self, response: str, context_items: list[Any]) -> dict[str, Any]:
        """
        Analyse les citations d'une réponse par rapport aux context_items fournis.
        """
        text = (response or "").strip()
        matches = _SOURCE_NUM_PATTERN.findall(text)

        verified: list[str] = []
        unverified: list[str] = []

        for num_str, rest in matches:
            label = f"[Source {num_str}{rest}]"
            if self._check_citation_exists(num_str, context_items):
                verified.append(label)
            else:
                unverified.append(label)

        grounding_score = self._check_content_grounded(text, context_items)
        hallucination = self.hallucination_risk(grounding_score) if context_items else True
        is_valid = len(unverified) == 0 and not hallucination

        return {
            "valid": is_valid,
            "verified_citations": verified,
            "unverified_citations": unverified,
            "grounding_score": round(grounding_score, 3),
            "hallucination_risk": hallucination,
        }

    @staticmethod
    def _check_citation_exists(source_ref: str, context_items: list[Any]) -> bool:
        """Vérifie que l'indice de source (1-based) existe dans context_items."""
        try:
            idx = int(source_ref)
        except ValueError:
            return False
        return 1 <= idx <= len(context_items)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_-]{3,}", text.lower())
        return {t for t in tokens if t not in _STOPWORDS_FR}

    def _check_content_grounded(self, answer: str, context_items: list[Any]) -> float:
        """
        Calcule la proportion des phrases de la réponse qui partagent
        des termes significatifs avec le contexte documentaire.
        """
        if not answer or not context_items:
            return 0.0

        context_text = " ".join(
            getattr(item, "content", "") if not isinstance(item, dict) else item.get("content", "")
            for item in context_items
        )
        context_tokens = self._tokenize(context_text)
        if not context_tokens:
            return 0.0

        # Découper en phrases non vides (en ignorant les citations seules)
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip()) > 15]
        if not sentences:
            answer_tokens = self._tokenize(answer)
            if not answer_tokens:
                return 0.0
            return len(answer_tokens & context_tokens) / len(answer_tokens)

        grounded_sentences = 0
        for sent in sentences:
            sent_tokens = self._tokenize(sent)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & context_tokens)
            if overlap >= 2 or (overlap / len(sent_tokens)) >= 0.2:
                grounded_sentences += 1

        return grounded_sentences / len(sentences)

    @staticmethod
    def hallucination_risk(grounding_score: float) -> bool:
        """Retourne True si l'ancrage dans le contexte est inférieur à 0.3."""
        return grounding_score < 0.3
