"""
response_parser.py — Analyse et structuration des réponses LLM
===============================================================

Extrait les citations [Source N - ...], détecte les marqueurs
d'incertitude, identifie les propositions d'action (ex: création ticket)
et estime un score de confiance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\s*[^\]]*\]", re.IGNORECASE)

_UNCERTAINTY_PHRASES = (
    "je ne sais pas",
    "incertain",
    "je ne trouve pas",
    "pourrait être",
    "possible que",
    "pas d'information",
    "aucune information",
    "information insuffisante",
)

_TICKET_ACTION_PHRASES = (
    "créer un ticket",
    "ouvrir un ticket",
    "ouvrir une demande",
    "déclarer une panne",
    "intervention requise",
)


@dataclass
class ParsedResponse:
    """Réponse structurée après parsing."""

    answer: str
    sources_cited: list[str] = field(default_factory=list)
    has_uncertainty: bool = False
    suggested_action: str | None = None
    confidence_score: float = 0.0


class ResponseParser:
    """Analyse la réponse brute du LLM et en déduit des métadonnées structurées."""

    def parse(self, raw_response: str, context_items: list[Any]) -> ParsedResponse:
        text = (raw_response or "").strip()
        citations = self._extract_citations(text)
        has_uncertainty = self._detect_uncertainty(text)
        confidence = self._estimate_confidence(text, citations, context_items, has_uncertainty)
        suggested_action = self._extract_action(text)

        return ParsedResponse(
            answer=text,
            sources_cited=citations,
            has_uncertainty=has_uncertainty,
            suggested_action=suggested_action,
            confidence_score=round(confidence, 3),
        )

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        """Extrait toutes les références [Source N - ...] de la réponse."""
        full_matches = re.findall(r"\[Source\s+\d+[^\]]*\]", text, flags=re.IGNORECASE)
        # Déduplication en préservant l'ordre
        seen: set[str] = set()
        unique: list[str] = []
        for match in full_matches:
            if match not in seen:
                seen.add(match)
                unique.append(match)
        return unique

    @staticmethod
    def _detect_uncertainty(text: str) -> bool:
        """Détecte si le modèle exprime un doute ou un manque d'information."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in _UNCERTAINTY_PHRASES)

    @staticmethod
    def _estimate_confidence(
        text: str,
        citations: list[str],
        context_items: list[Any],
        has_uncertainty: bool,
    ) -> float:
        """
        Estime un score de confiance entre 0.0 et 1.0 :
          - +0.4 si des citations valides sont présentes
          - +0.3 si aucune phrase d'incertitude n'est détectée
          - +0.3 si la réponse a une longueur raisonnable et qu'un contexte existe
        """
        if not text or not context_items:
            return 0.0

        score = 0.0
        if citations:
            score += 0.4
        if not has_uncertainty:
            score += 0.3
        if len(text) >= 40:
            score += 0.3

        return min(1.0, max(0.0, score))

    @staticmethod
    def _extract_action(text: str) -> str | None:
        """Détecte si la réponse suggère la création d'un ticket."""
        lowered = text.lower()
        if any(phrase in lowered for phrase in _TICKET_ACTION_PHRASES):
            return "create_ticket"
        return None
