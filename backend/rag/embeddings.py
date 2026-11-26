"""
embeddings.py — Gestionnaire d'embeddings textuels et visuels
=============================================================

Charge paresseusement (lazy loading) les modèles d'embeddings pour
économiser la RAM sur Apple M2 (8 Go) :
  - Texte : BAAI/bge-m3 (dimension 1024, multilingue FR/EN)
  - Visuel : clip-ViT-B-32-multilingual-v1 (dimension 512, cross-modal)
"""

from __future__ import annotations

import io
from typing import Any

from loguru import logger
from PIL import Image

from backend.config import settings


class EmbeddingError(Exception):
    """Erreur levée lors du chargement ou du calcul d'embeddings."""


class EmbeddingManager:
    """
    Gestionnaire unifié des embeddings texte (BGE-M3) et image (CLIP).
    Les modèles ne sont chargés en mémoire qu'au premier appel.
    """

    def __init__(self) -> None:
        self._text_model: Any = None
        self._visual_model: Any = None
        self.device = self._detect_device()
        logger.debug(f"EmbeddingManager initialisé (device={self.device})")

    @staticmethod
    def _detect_device() -> str:
        """Détecte le meilleur device disponible (MPS sur Apple Silicon, sinon CPU)."""
        try:
            import torch

            if torch.backends.mps.is_available():
                logger.info("Accélération Apple MPS disponible pour PyTorch")
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _get_text_model(self) -> Any:
        """Charge BGE-M3 à la demande (FlagEmbedding ou fallback SentenceTransformer)."""
        if self._text_model is None:
            logger.info(f"Chargement du modèle d'embeddings texte : {settings.embedding_model_text}")
            try:
                from FlagEmbedding import FlagModel

                self._text_model = (
                    "flag",
                    FlagModel(
                        settings.embedding_model_text,
                        use_fp16=False,  # fp32 plus sûr sur CPU/MPS M2
                    ),
                )
            except Exception as exc:
                logger.warning(f"FlagEmbedding indisponible ({exc}), fallback SentenceTransformer")
                from sentence_transformers import SentenceTransformer

                self._text_model = (
                    "st",
                    SentenceTransformer(settings.embedding_model_text, device="cpu"),
                )
        return self._text_model

    def _get_visual_model(self) -> Any:
        """Charge le modèle CLIP multilingue à la demande."""
        if self._visual_model is None:
            logger.info(f"Chargement du modèle d'embeddings visuels : {settings.embedding_model_visual}")
            from sentence_transformers import SentenceTransformer

            self._visual_model = SentenceTransformer(
                settings.embedding_model_visual,
                device="cpu",  # CPU fiable pour CLIP 400 Mo sur 8 Go RAM
            )
        return self._visual_model

    def encode_text(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Encode une liste de textes en vecteurs denses (dim=1024 pour BGE-M3).
        """
        if not texts:
            return []

        try:
            backend_type, model = self._get_text_model()
            if backend_type == "flag":
                vectors = model.encode(texts, batch_size=batch_size)
                # FlagModel.encode peut renvoyer un dict {"dense_vecs": ...} ou un ndarray
                if isinstance(vectors, dict):
                    vectors = vectors["dense_vecs"]
                return [vec.tolist() for vec in vectors]

            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [vec.tolist() for vec in vectors]
        except Exception as exc:
            logger.error(f"Erreur encodage texte : {exc}")
            raise EmbeddingError(f"Échec encodage texte : {exc}") from exc

    def encode_text_single(self, text: str) -> list[float]:
        """Encode une seule chaîne de texte."""
        vectors = self.encode_text([text])
        return vectors[0]

    def encode_image(self, image_bytes: bytes) -> list[float]:
        """
        Encode une image brute (bytes) en vecteur CLIP (dim=512).
        """
        if not image_bytes:
            raise EmbeddingError("Image vide fournie à encode_image()")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            model = self._get_visual_model()
            vector = model.encode([image], normalize_embeddings=True, show_progress_bar=False)[0]
            return vector.tolist()
        except Exception as exc:
            logger.error(f"Erreur encodage image : {exc}")
            raise EmbeddingError(f"Échec encodage image : {exc}") from exc

    def encode_text_for_image_search(self, text: str) -> list[float]:
        """
        Encode une requête textuelle dans l'espace vectoriel CLIP
        pour rechercher des images similaires (cross-modal).
        """
        if not text or not text.strip():
            raise EmbeddingError("Requête textuelle vide pour recherche visuelle")

        try:
            model = self._get_visual_model()
            vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            return vector.tolist()
        except Exception as exc:
            logger.error(f"Erreur encodage texte pour recherche visuelle : {exc}")
            raise EmbeddingError(f"Échec encodage texte CLIP : {exc}") from exc
