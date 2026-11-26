"""
documents.py — Routes de gestion et d'ingestion des PDF
========================================================

Endpoints :
    POST   /api/v1/documents/upload      → Importer et indexer un PDF
    GET    /api/v1/documents/            → Lister les documents accessibles
    GET    /api/v1/documents/{doc_id}    → Détails d'un document
    DELETE /api/v1/documents/{doc_id}    → Supprimer un document (Qdrant + MinIO + PG)
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db import crud
from backend.db.database import get_db
from backend.db.models import Document
from backend.db.schemas import DocumentResponse
from backend.rag.indexer import DocumentIndexer
from backend.rag.qdrant_store import QdrantStore
from backend.security.input_validator import validate_file_size
from backend.security.permissions import check_document_access
from backend.storage.minio_client import minio_client

router = APIRouter()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    doc_type: Optional[str] = Form(default="manuel"),
    equipment_type: Optional[str] = Form(default=None),
    version: Optional[str] = Form(default="1.0"),
    db: AsyncSession = Depends(get_db),
):
    """
    Importe un fichier PDF, le stocke sur MinIO, l'indexe dans Qdrant
    et enregistre ses métadonnées dans PostgreSQL.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les fichiers PDF sont acceptés.",
        )

    file_bytes = await file.read()
    validate_file_size(len(file_bytes))

    doc_uuid = str(uuid.uuid4())
    storage_path: str | None = None

    # 1. Enregistrement initial en base
    doc = await crud.create_document(
        db,
        {
            "filename": file.filename,
            "title": title or file.filename,
            "doc_type": doc_type,
            "equipment_type": equipment_type,
            "version": version,
            "status": "processing",
            "permissions": ["technician", "supervisor", "admin"],
        },
    )
    doc_uuid = str(doc.id)

    # 2. Sauvegarde MinIO (tolérante si MinIO indisponible en test local)
    try:
        storage_path = await asyncio.to_thread(
            minio_client.upload_pdf, doc_uuid, file.filename, file_bytes
        )
        doc.storage_path = storage_path
    except Exception as exc:
        logger.warning(f"Upload MinIO ignoré ({exc}) — sauvegarde locale dans data/raw")
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        local_path = settings.raw_dir / f"{doc_uuid}_{file.filename}"
        local_path.write_bytes(file_bytes)
        doc.storage_path = str(local_path)

    # 3. Indexation PDF → Qdrant
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        indexer = DocumentIndexer()
        summary = await asyncio.to_thread(
            indexer.index_document,
            tmp_path,
            doc_uuid,
            equipment_type,
            doc_type,
            version,
            True,
            bool(storage_path),
        )
        tmp_path.unlink(missing_ok=True)

        doc.total_pages = summary["total_pages"]
        doc.total_chunks = summary["chunks_indexed"]
        doc.status = "indexed"
        await db.commit()
        await db.refresh(doc)
        return doc
    except Exception as exc:
        logger.error(f"Erreur indexation PDF '{file.filename}' : {exc}")
        await crud.update_document_status(db, doc_uuid, "error", error_msg=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec de l'indexation du document : {exc}",
        )


@router.get("/", response_model=list[DocumentResponse], summary="Lister les documents")
async def list_documents(
    user_role: str = "technician",
    db: AsyncSession = Depends(get_db),
):
    """Retourne les documents autorisés pour le rôle courant."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = list(result.scalars().all())
    return [d for d in docs if check_document_access(user_role, d.permissions)]


@router.get("/{document_id}", response_model=DocumentResponse, summary="Détails d'un document")
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="document_id invalide")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_200_OK, summary="Supprimer un document")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="document_id invalide")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document introuvable")

    # Suppression Qdrant + MinIO + PostgreSQL
    try:
        store = QdrantStore()
        await asyncio.to_thread(store.delete_document, settings.qdrant_collection_text, document_id)
        await asyncio.to_thread(store.delete_document, settings.qdrant_collection_visual, document_id)
    except Exception as exc:
        logger.warning(f"Suppression Qdrant partielle : {exc}")

    try:
        await asyncio.to_thread(minio_client.delete_document, document_id)
    except Exception as exc:
        logger.warning(f"Suppression MinIO partielle : {exc}")

    await db.delete(doc)
    await db.commit()
    return {"deleted": True, "document_id": document_id}
