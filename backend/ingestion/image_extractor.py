"""
Image Extractor for MaintiDoc.

Extracts embedded images from PDF documents using PyMuPDF and optionally
uploads them to MinIO object storage.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ExtractedImage:
    """Metadata and raw bytes for a single image extracted from a PDF."""

    image_id: str
    """Unique identifier for this image (UUID)."""

    page_number: int
    """1-based page number where the image was found."""

    image_index: int
    """0-based index of the image within the page."""

    image_bytes: bytes
    """Raw image data (PNG/JPEG/etc.)."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    colorspace: str
    """Colorspace string (e.g. 'DeviceRGB', 'DeviceGray')."""

    storage_path: Optional[str] = None
    """MinIO object path after upload; None until saved."""


# ---------------------------------------------------------------------------
# Image Extractor
# ---------------------------------------------------------------------------

class ImageExtractor:
    """
    Extracts meaningful (non-decorative) images from PDF documents.

    Images smaller than 100 × 100 pixels are discarded as decorative.

    Example::

        extractor = ImageExtractor()
        images = extractor.extract_from_document("manual.pdf", fitz_doc)
    """

    MIN_SIZE = 100  # minimum width AND height in pixels

    def extract_from_document(
        self,
        doc_path: str,
        fitz_doc: fitz.Document,
    ) -> list[ExtractedImage]:
        """
        Extract all meaningful images from a PDF document.

        Args:
            doc_path:  File path of the PDF (used for logging).
            fitz_doc:  An already-open :class:`fitz.Document`.

        Returns:
            List of :class:`ExtractedImage` objects, one per qualifying image.
        """
        all_images: list[ExtractedImage] = []

        for page_index in range(len(fitz_doc)):
            try:
                page = fitz_doc[page_index]
                page_images = self._extract_page_images(page, fitz_doc, page_index + 1)
                all_images.extend(page_images)
            except Exception as exc:
                logger.warning(
                    f"Failed to extract images from page {page_index + 1} "
                    f"of '{doc_path}': {exc}"
                )

        logger.info(
            f"Extracted {len(all_images)} meaningful images from '{doc_path}'."
        )
        return all_images

    def _extract_page_images(
        self,
        page: fitz.Page,
        fitz_doc: fitz.Document,
        page_num: int,
    ) -> list[ExtractedImage]:
        """
        Extract images from a single page.

        Args:
            page:      The fitz.Page to process.
            fitz_doc:  Parent document (needed for extract_image()).
            page_num:  1-based page number.

        Returns:
            List of :class:`ExtractedImage` objects for this page.
        """
        page_images: list[ExtractedImage] = []

        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                img_dict = fitz_doc.extract_image(xref)
            except Exception as exc:
                logger.debug(f"Could not extract image xref={xref} on page {page_num}: {exc}")
                continue

            if not img_dict:
                continue

            width: int = img_dict.get("width", 0)
            height: int = img_dict.get("height", 0)
            colorspace: str = img_dict.get("colorspace", "")
            image_bytes: bytes = img_dict.get("image", b"")

            # Normalise colorspace (can be int or string from fitz)
            if isinstance(colorspace, int):
                _cs_map = {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}
                colorspace = _cs_map.get(colorspace, f"cs{colorspace}")

            extracted = ExtractedImage(
                image_id=str(uuid.uuid4()),
                page_number=page_num,
                image_index=img_index,
                image_bytes=image_bytes,
                width=width,
                height=height,
                colorspace=str(colorspace),
            )

            if self._is_meaningful_image(extracted):
                page_images.append(extracted)
                logger.debug(
                    f"  Page {page_num}, img {img_index}: {width}×{height} "
                    f"({colorspace}) — kept."
                )
            else:
                logger.debug(
                    f"  Page {page_num}, img {img_index}: {width}×{height} — skipped (too small)."
                )

        return page_images

    def _is_meaningful_image(self, img: ExtractedImage) -> bool:
        """
        Return True if the image is large enough to be meaningful content.

        Images below :attr:`MIN_SIZE` in either dimension are considered
        decorative (icons, bullets, borders) and are filtered out.

        Args:
            img: The :class:`ExtractedImage` to evaluate.

        Returns:
            True when width ≥ 100 and height ≥ 100.
        """
        return img.width >= self.MIN_SIZE and img.height >= self.MIN_SIZE

    def save_to_minio(
        self,
        images: list[ExtractedImage],
        document_id: str,
    ) -> list[ExtractedImage]:
        """
        Upload extracted images to MinIO and update storage_path on each image.

        Args:
            images:      List of :class:`ExtractedImage` objects to upload.
            document_id: Document identifier used as the MinIO folder prefix.

        Returns:
            The same list with ``storage_path`` set on successfully uploaded images.
        """
        from backend.storage.minio_client import get_minio_client  # local import

        try:
            minio_client = get_minio_client()
        except Exception as exc:
            logger.error(f"Cannot connect to MinIO, images not saved: {exc}")
            return images

        for img in images:
            object_name = f"{document_id}/images/{img.image_id}.png"
            try:
                minio_client.put_object(
                    object_name=object_name,
                    data=io.BytesIO(img.image_bytes),
                    length=len(img.image_bytes),
                    content_type="image/png",
                )
                img.storage_path = object_name
                logger.debug(f"Uploaded image {img.image_id} to MinIO at '{object_name}'.")
            except Exception as exc:
                logger.warning(f"Failed to upload image {img.image_id} to MinIO: {exc}")

        return images
