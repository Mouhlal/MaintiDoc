"""
Metadata Builder for MaintiDoc.

Constructs structured, JSON-serialisable metadata dictionaries for
documents, text chunks, and images — ready for storage in Qdrant or
PostgreSQL.
"""

from __future__ import annotations

import datetime
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.ingestion.pdf_loader import ExtractedDocument
    from backend.ingestion.chunker import Chunk
    from backend.ingestion.image_extractor import ExtractedImage


class MetadataBuilder:
    """
    Builds standardised metadata payloads for documents, chunks, and images.

    All returned dicts are JSON-serialisable (no bytes, no non-primitive types).

    Example::

        builder = MetadataBuilder()
        doc_meta = builder.build_document_metadata(
            file_path="manual.pdf",
            doc=extracted_doc,
            document_id="DOC-001",
            equipment_type="HVAC",
        )
    """

    # ------------------------------------------------------------------
    # Document-level metadata
    # ------------------------------------------------------------------

    def build_document_metadata(
        self,
        file_path: str,
        doc: "ExtractedDocument",
        document_id: str,
        equipment_type: Optional[str] = None,
        doc_type: Optional[str] = None,
        version: Optional[str] = None,
    ) -> dict:
        """
        Build complete metadata for an ingested document.

        Args:
            file_path:      Original file path.
            doc:            Parsed :class:`ExtractedDocument`.
            document_id:    Unique identifier assigned to this document.
            equipment_type: Optional equipment category (e.g. "HVAC", "Pump").
            doc_type:       Optional document type (e.g. "manual", "datasheet").
            version:        Optional document version string.

        Returns:
            Flat dict with document-level metadata.
        """
        pages_with_text = sum(1 for p in doc.pages if p.has_text)
        total_images = sum(len(p.images) for p in doc.pages)

        metadata = {
            # Identifiers
            "document_id": document_id,
            "filename": doc.filename,
            "file_path": file_path,
            # Document classification
            "equipment_type": equipment_type or "",
            "doc_type": doc_type or "",
            "version": version or "",
            # PDF-extracted metadata
            "title": doc.metadata.get("title", "") or doc.filename,
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "creation_date": doc.metadata.get("creation_date", ""),
            "mod_date": doc.metadata.get("mod_date", ""),
            # Statistics
            "total_pages": doc.total_pages,
            "pages_with_text": pages_with_text,
            "total_images": total_images,
            # Ingestion timestamp
            "ingested_at": datetime.datetime.utcnow().isoformat(),
        }

        logger.debug(f"Built document metadata for '{document_id}': {list(metadata.keys())}")
        return metadata

    # ------------------------------------------------------------------
    # Chunk-level metadata (Qdrant text collection payload)
    # ------------------------------------------------------------------

    def build_chunk_metadata(
        self,
        chunk: "Chunk",
        document_id: str,
        doc_metadata: dict,
    ) -> dict:
        """
        Build the Qdrant payload dict for a text chunk.

        All fields are JSON-serialisable primitives.

        Args:
            chunk:         The :class:`Chunk` to describe.
            document_id:   Parent document identifier.
            doc_metadata:  Result of :meth:`build_document_metadata`.

        Returns:
            Dict suitable for use as a Qdrant point payload.
        """
        payload = {
            # Linking fields
            "document_id": document_id,
            "filename": doc_metadata.get("filename", ""),
            # Chunk position
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            # Quick preview (first 200 chars)
            "content_preview": chunk.content[:200],
            # Full content (stored for retrieval)
            "content": chunk.content,
            # Token stats
            "token_count": chunk.token_count,
            # Document classification passthrough
            "equipment_type": doc_metadata.get("equipment_type", ""),
            "doc_type": doc_metadata.get("doc_type", ""),
            "version": doc_metadata.get("version", ""),
        }

        logger.debug(
            f"Built chunk metadata for chunk {chunk.chunk_index} "
            f"(doc={document_id}, page={chunk.page_number})."
        )
        return payload

    # ------------------------------------------------------------------
    # Image-level metadata (Qdrant visual collection payload)
    # ------------------------------------------------------------------

    def build_image_metadata(
        self,
        image: "ExtractedImage",
        document_id: str,
        doc_metadata: dict,
    ) -> dict:
        """
        Build the Qdrant payload dict for an extracted image.

        Args:
            image:         The :class:`ExtractedImage` to describe.
            document_id:   Parent document identifier.
            doc_metadata:  Result of :meth:`build_document_metadata`.

        Returns:
            Dict suitable for use as a Qdrant point payload (visual collection).
        """
        payload = {
            # Linking fields
            "document_id": document_id,
            "filename": doc_metadata.get("filename", ""),
            "image_id": image.image_id,
            # Image position
            "page_number": image.page_number,
            "image_index": image.image_index,
            # Image properties
            "width": image.width,
            "height": image.height,
            "colorspace": image.colorspace,
            # MinIO storage location (may be None if upload failed)
            "storage_path": image.storage_path or "",
            # Document classification passthrough
            "equipment_type": doc_metadata.get("equipment_type", ""),
            "doc_type": doc_metadata.get("doc_type", ""),
        }

        logger.debug(
            f"Built image metadata for image {image.image_id} "
            f"(doc={document_id}, page={image.page_number})."
        )
        return payload
