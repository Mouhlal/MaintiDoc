"""
vlm_client.py — Client VLM (Vision-Language Model) via Ollama
==============================================================

Permet d'analyser des photographies de panneaux de commande et d'équipements
avec un modèle multimodal léger (llava:7b-v1.6-mistral-q4_0) chargé à la demande.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings
from backend.llm.prompts import IMAGE_ANALYSIS_PROMPT


class VLMError(Exception):
    """Erreur levée par VLMClient."""


@dataclass
class VLMResponse:
    """Résultat structuré de l'analyse d'image par le VLM."""

    description: str
    detected_issues: list[str] = field(default_factory=list)
    confidence: float = 0.7
    model: str = ""
    latency_ms: float = 0.0


_ISSUE_KEYWORDS = (
    "fuite",
    "usure",
    "corrosion",
    "dégradation",
    "anomalie",
    "fissure",
    "surchauffe",
    "oxydation",
    "alarme",
    "voyant rouge",
    "défaut",
    "cassé",
)


class VLMClient:
    """Client HTTP synchrone vers l'API multimodale d'Ollama."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.vlm_model
        self.base_url = settings.ollama_base_url.rstrip("/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )
    def analyze_image(self, image_bytes: bytes, prompt: str | None = None) -> VLMResponse:
        """
        Envoie une image encodée en base64 à Ollama /api/generate.
        """
        if not image_bytes:
            raise VLMError("Aucune image fournie pour l'analyse VLM")

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        full_prompt = prompt or IMAGE_ANALYSIS_PROMPT

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        start = time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"Erreur HTTP VLM Ollama : {exc}")
            raise VLMError(f"Échec appel VLM : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        description = (data.get("response") or "").strip()
        issues = self._extract_issues(description)
        confidence = 0.8 if len(description) > 80 else 0.5

        logger.info(
            f"VLM analyse terminée en {latency_ms:.0f}ms ({len(issues)} anomalie(s) détectée(s))"
        )
        return VLMResponse(
            description=description,
            detected_issues=issues,
            confidence=confidence,
            model=self.model,
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _extract_issues(text: str) -> list[str]:
        """Heuristique simple d'extraction de mots-clés d'anomalies."""
        lowered = text.lower()
        return [kw for kw in _ISSUE_KEYWORDS if kw in lowered]

    def is_available(self) -> bool:
        """Vérifie si Ollama tourne et possède le modèle VLM configuré."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                base_model = self.model.split(":")[0]
                return any(self.model == m or m.startswith(base_model) for m in models)
        except Exception:
            return False


vlm_client = VLMClient()
