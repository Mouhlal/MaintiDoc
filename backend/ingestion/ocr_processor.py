"""
OCR Processor for MaintiDoc.

Uses EasyOCR to extract text from scanned / image-heavy PDF pages.
The EasyOCR Reader is initialized lazily to avoid loading GPU models
until they are actually needed.
"""

from __future__ import annotations

import io
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from loguru import logger
from PIL import Image


class OCRProcessor:
    """
    Runs OCR on individual images or full PDF pages using EasyOCR.

    The underlying ``easyocr.Reader`` is created on first use (lazy init)
    so that importing this module does not trigger a costly model download /
    GPU allocation.

    Args:
        languages: ISO 639-1 language codes passed to EasyOCR.
                   Defaults to French + English.

    Example::

        processor = OCRProcessor(languages=["fr", "en"])
        text = processor.process_page(fitz_page)
    """

    def __init__(self, languages: list[str] = None) -> None:
        self.languages: list[str] = languages or ["fr", "en"]
        self._reader = None  # lazily initialised

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_reader(self):
        """
        Return (and if necessary create) the EasyOCR Reader.

        The Reader is kept as an instance attribute so it is reused across
        multiple calls without re-loading the model weights.
        """
        if self._reader is None:
            try:
                import easyocr  # local import keeps startup fast
                logger.info(
                    f"Initialising EasyOCR Reader for languages: {self.languages}"
                )
                self._reader = easyocr.Reader(self.languages, gpu=False)
                logger.info("EasyOCR Reader ready.")
            except ImportError as exc:
                raise RuntimeError(
                    "easyocr is not installed. Run: pip install easyocr"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to initialise EasyOCR Reader: {exc}") from exc

        return self._reader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image_bytes: bytes) -> str:
        """
        Run OCR on raw image bytes and return all recognised text.

        Args:
            image_bytes: Raw bytes of an image file (PNG, JPEG, …).

        Returns:
            Concatenated OCR text, one recognition result per line.
        """
        if not image_bytes:
            return ""

        try:
            reader = self._get_reader()
            # EasyOCR can accept a numpy array
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image)

            results = reader.readtext(image_np, detail=0, paragraph=True)
            text = "\n".join(str(r) for r in results)
            logger.debug(f"OCR produced {len(text)} chars from image bytes.")
            return text
        except Exception as exc:
            logger.warning(f"OCR failed on image bytes: {exc}")
            return ""

    def process_page(self, page: fitz.Page) -> str:
        """
        Render a PDF page as a 200 DPI image and run OCR on it.

        Args:
            page: An open fitz.Page object.

        Returns:
            OCR-extracted text string.
        """
        try:
            # 200 DPI → scale factor ≈ 200/72 ≈ 2.78
            zoom = 200 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            image_bytes: bytes = pix.tobytes("png")
            logger.debug(
                f"Rendered page {page.number + 1} to {pix.width}×{pix.height} px for OCR."
            )
            return self.process_image(image_bytes)
        except Exception as exc:
            logger.warning(f"Failed to render page {page.number + 1} for OCR: {exc}")
            return ""

    def should_ocr(self, page_text: str) -> bool:
        """
        Decide whether OCR is needed for a page based on its extracted text.

        A page is considered *scanned* (and thus requires OCR) when the
        directly extracted text is shorter than 100 characters.

        Args:
            page_text: Text already extracted from the PDF layer.

        Returns:
            True if OCR should be run; False otherwise.
        """
        return len(page_text.strip()) < 100
