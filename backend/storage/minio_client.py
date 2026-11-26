"""
minio_client.py — Client MinIO (stockage objet S3) pour MaintiDoc
================================================================

Gère le stockage des fichiers PDF originaux et des images extraites
dans MinIO. Compatible avec l'API S3.

Buckets utilisés :
  - maintidoc-documents : PDF originaux
  - maintidoc-images    : figures et schémas extraits
"""

from __future__ import annotations

import io
from datetime import timedelta
from functools import lru_cache
from typing import BinaryIO

from loguru import logger
from minio import Minio
from minio.error import S3Error

from backend.config import settings


class MinIOClient:
    """Wrapper autour du client officiel MinIO pour MaintiDoc."""

    def __init__(self) -> None:
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=settings.minio_secure,
        )
        self.bucket_documents = settings.minio_bucket_documents
        self.bucket_images = settings.minio_bucket_images

    def ensure_buckets(self) -> None:
        """Crée les buckets documents, images et mlflow s'ils n'existent pas."""
        buckets = [
            self.bucket_documents,
            self.bucket_images,
            "maintidoc-mlflow",
        ]
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Bucket MinIO créé : {bucket}")
            except S3Error as exc:
                logger.error(f"Erreur création bucket MinIO '{bucket}' : {exc}")
                raise

    def upload_pdf(self, document_id: str, filename: str, file_bytes: bytes) -> str:
        """Upload un PDF dans le bucket documents et retourne son object_path."""
        self.ensure_buckets()
        object_name = f"{document_id}/{filename}"
        self.client.put_object(
            bucket_name=self.bucket_documents,
            object_name=object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type="application/pdf",
        )
        logger.info(f"PDF uploadé sur MinIO : {self.bucket_documents}/{object_name}")
        return object_name

    def upload_image(
        self,
        document_id: str,
        image_id: str,
        image_bytes: bytes,
        extension: str = "png",
    ) -> str:
        """Upload une image dans le bucket images et retourne son object_path."""
        self.ensure_buckets()
        object_name = f"{document_id}/images/{image_id}.{extension}"
        self.put_object(
            object_name=object_name,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=f"image/{extension}",
        )
        return object_name

    def put_object(
        self,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
        bucket_name: str | None = None,
    ) -> str:
        """Méthode générique utilisée notamment par ImageExtractor."""
        target_bucket = bucket_name or self.bucket_images
        if not self.client.bucket_exists(target_bucket):
            self.client.make_bucket(target_bucket)
        self.client.put_object(
            bucket_name=target_bucket,
            object_name=object_name,
            data=data,
            length=length,
            content_type=content_type,
        )
        return object_name

    def get_pdf_url(self, object_path: str, expires: int = 3600) -> str:
        """Génère une URL pré-signée temporaire pour télécharger un PDF."""
        return self.client.presigned_get_object(
            bucket_name=self.bucket_documents,
            object_name=object_path,
            expires=timedelta(seconds=expires),
        )

    def get_image_bytes(self, object_path: str) -> bytes:
        """Télécharge les octets d'une image depuis le bucket images."""
        response = self.client.get_object(
            bucket_name=self.bucket_images,
            object_name=object_path,
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_document(self, document_id: str) -> None:
        """Supprime tous les objets associés à un document dans les deux buckets."""
        prefix = f"{document_id}/"
        for bucket in (self.bucket_documents, self.bucket_images):
            try:
                if not self.client.bucket_exists(bucket):
                    continue
                objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
                for obj in objects:
                    self.client.remove_object(bucket, obj.object_name)
                logger.info(f"Objets supprimés dans {bucket} pour préfixe '{prefix}'")
            except S3Error as exc:
                logger.warning(f"Erreur suppression MinIO ({bucket}/{prefix}) : {exc}")


@lru_cache(maxsize=1)
def get_minio_client() -> MinIOClient:
    """Retourne l'instance singleton de MinIOClient."""
    return MinIOClient()


minio_client = get_minio_client()
