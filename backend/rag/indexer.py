"""
indexer.py — Pipeline d'indexation documentaire de MaintiDoc
============================================================

Orchestre l'ingestion complète d'un PDF :
  1. Chargement PDF (PyMuPDF) + OCR si pages scannées
  2. Chunking intelligent avec chevauchement
  3. Extraction des figures et upload MinIO
  4. Construction des métadonnées
  5. Calcul des embeddings texte (BGE-M3) et image (CLIP)
  6. Upsert dans les collections Qdrant texte et visuel
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import fitz
from loguru import logger

from backend.config import settings
from backend.ingestion.chunker import Chunk, TextChunker
from backend.ingestion.image_extractor import ImageExtractor
from backend.ingestion.metadata_builder import MetadataBuilder
from backend.ingestion.ocr_processor import OCRProcessor
from backend.ingestion.pdf_loader import PDFLoader
from backend.rag.embeddings import EmbeddingManager
from backend.rag.qdrant_store import QdrantStore


class DocumentIndexer:
    """Orchestrateur du pipeline PDF → Qdrant + MinIO."""

    def __init__(self) -> None:
        self.pdf_loader = PDFLoader()
        self.ocr_processor = OCRProcessor()
        self.chunker = TextChunker()
        self.image_extractor = ImageExtractor()
        self.metadata_builder = MetadataBuilder()
        self.embeddings = EmbeddingManager()
        self.qdrant = QdrantStore()

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Génère un UUID déterministe (uuid5) pour garantir l'idempotence."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}-chunk-{chunk_index}"))

    def index_document(
        self,
        file_path: str | Path,
        document_id: str,
        equipment_type: str | None = None,
        doc_type: str | None = None,
        version: str | None = None,
        use_ocr: bool = True,
        save_images_minio: bool = True,
    ) -> dict[str, Any]:
        """
        Indexe un PDF complet dans Qdrant (texte + images).

        Retourne un dictionnaire récapitulatif avec les compteurs et métadonnées.
        """
        file_path = Path(file_path)
        logger.info(f"Début d'indexation : {file_path.name} (document_id={document_id})")

        # 1. Chargement PDF
        extracted_doc = self.pdf_loader.load(file_path)

        # 1b. OCR optionnel sur les pages sans texte natif
        if use_ocr:
            pages_needing_ocr = [p for p in extracted_doc.pages if self.ocr_processor.should_ocr(p.text)]
            if pages_needing_ocr:
                logger.info(f"{len(pages_needing_ocr)} page(s) nécessitent un OCR")
                with fitz.open(str(file_path)) as fitz_doc:
                    for page in pages_needing_ocr:
                        ocr_text = self.ocr_processor.process_page(fitz_doc[page.page_number - 1])
                        if ocr_text.strip():
                            page.text = f"{page.text}\n{ocr_text}".strip()
                            page.has_text = True

        # 2. Chunking
        chunks: list[Chunk] = self.chunker.chunk_document(extracted_doc)

        # 3. Extraction des images
        with fitz.open(str(file_path)) as fitz_doc:
            extracted_images = self.image_extractor.extract_from_document(str(file_path), fitz_doc)

        if save_images_minio and extracted_images:
            extracted_images = self.image_extractor.save_to_minio(extracted_images, document_id)

        # 4. Métadonnées document
        doc_metadata = self.metadata_builder.build_document_metadata(
            file_path=str(file_path),
            doc=extracted_doc,
            document_id=document_id,
            equipment_type=equipment_type,
            doc_type=doc_type,
            version=version,
        )

        # 5. Embeddings texte par batch de 32 + Upsert Qdrant
        self.qdrant.ensure_collection(settings.qdrant_collection_text, settings.embedding_dim_text)
        text_points: list[dict[str, Any]] = []

        if chunks:
            texts = [c.content for c in chunks]
            vectors = self.embeddings.encode_text(texts, batch_size=32)
            for chunk, vec in zip(chunks, vectors):
                payload = self.metadata_builder.build_chunk_metadata(chunk, document_id, doc_metadata)
                text_points.append(
                    {
                        "id": self._generate_chunk_id(document_id, chunk.chunk_index),
                        "vector": vec,
                        "payload": payload,
                    }
                )
            self.qdrant.upsert_chunks(settings.qdrant_collection_text, text_points)

        # 6. Embeddings visuels + Upsert Qdrant
        visual_points: list[dict[str, Any]] = []
        if extracted_images:
            self.qdrant.ensure_collection(
                settings.qdrant_collection_visual, settings.embedding_dim_visual
            )
            for img in extracted_images:
                try:
                    vec = self.embeddings.encode_image(img.image_bytes)
                    payload = self.metadata_builder.build_image_metadata(img, document_id, doc_metadata)
                    visual_points.append(
                        {
                            "id": img.image_id,
                            "vector": vec,
                            "payload": payload,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Image {img.image_id} ignorée à l'indexation : {exc}")

            if visual_points:
                self.qdrant.upsert_chunks(settings.qdrant_collection_visual, visual_points)

        summary = {
            "document_id": document_id,
            "filename": extracted_doc.filename,
            "total_pages": extracted_doc.total_pages,
            "chunks_indexed": len(text_points),
            "images_indexed": len(visual_points),
            "metadata": doc_metadata,
        }
        logger.info(
            f"Indexation terminée : {summary['chunks_indexed']} chunks, "
            f"{summary['images_indexed']} images"
        )
        return summary
