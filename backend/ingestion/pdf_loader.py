"""
PDF Loader module for MaintiDoc.

Loads PDF files using PyMuPDF (fitz), extracting text, images, tables,
and document metadata from each page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF
from loguru import logger


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPage:
    """Represents a single page extracted from a PDF document."""

    page_number: int
    """1-based page number."""

    text: str
    """Plain text extracted from the page."""

    images: list[dict]
    """List of image metadata dicts (xref, width, height, colorspace, etc.)."""

    tables: list[str]
    """List of text blocks that likely represent tabular data."""

    has_text: bool
    """True when the page contains meaningful extractable text (> 50 chars)."""


@dataclass
class ExtractedDocument:
    """Represents the full content extracted from a PDF file."""

    file_path: str
    """Absolute path to the source PDF."""

    filename: str
    """Base filename (without directory)."""

    total_pages: int
    """Total number of pages in the document."""

    pages: list[ExtractedPage]
    """Ordered list of extracted pages."""

    metadata: dict = field(default_factory=dict)
    """Document-level metadata (title, author, creation date, etc.)."""


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Loads and parses PDF files using PyMuPDF.

    Usage::

        loader = PDFLoader()
        doc = loader.load("manual.pdf")
        for page in doc.pages:
            print(page.page_number, page.text[:200])
    """

    def load(self, file_path: Union[str, Path]) -> ExtractedDocument:
        """
        Open and parse a PDF file, extracting text, images, and table blocks.

        Args:
            file_path: Path to the PDF file (str or Path object).

        Returns:
            ExtractedDocument containing all parsed pages and metadata.

        Raises:
            PDFLoadError: If the file cannot be opened or processed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PDFLoadError(f"PDF file not found: {file_path}")

        if not file_path.is_file():
            raise PDFLoadError(f"Path is not a file: {file_path}")

        logger.info(f"Loading PDF: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")

        try:
            fitz_doc = fitz.open(str(file_path))
        except Exception as exc:
            raise PDFLoadError(f"Failed to open PDF '{file_path}': {exc}") from exc

        try:
            metadata = self.extract_metadata(fitz_doc)
            pages: list[ExtractedPage] = []

            for page_index in range(len(fitz_doc)):
                try:
                    page = fitz_doc[page_index]
                    extracted_page = self._extract_page(page, page_index + 1)
                    pages.append(extracted_page)
                except Exception as exc:
                    logger.warning(
                        f"Could not extract page {page_index + 1} from '{file_path.name}': {exc}"
                    )
                    # Insert a placeholder so total_pages stays consistent
                    pages.append(
                        ExtractedPage(
                            page_number=page_index + 1,
                            text="",
                            images=[],
                            tables=[],
                            has_text=False,
                        )
                    )

            logger.info(
                f"Loaded '{file_path.name}': {len(pages)} pages, "
                f"{sum(1 for p in pages if p.has_text)} with text, "
                f"{sum(len(p.images) for p in pages)} images total."
            )

            return ExtractedDocument(
                file_path=str(file_path.resolve()),
                filename=file_path.name,
                total_pages=len(fitz_doc),
                pages=pages,
                metadata=metadata,
            )

        finally:
            fitz_doc.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_page(self, page: fitz.Page, page_number: int) -> ExtractedPage:
        """
        Extract all content from a single fitz.Page.

        Args:
            page: The fitz.Page object.
            page_number: 1-based page number for display/storage.

        Returns:
            Populated ExtractedPage dataclass.
        """
        # --- Text ---
        text: str = page.get_text("text") or ""

        # --- Images ---
        images: list[dict] = []
        try:
            for img_info in page.get_images(full=True):
                # img_info tuple: (xref, smask, width, height, bpc, colorspace, ...)
                images.append(
                    {
                        "xref": img_info[0],
                        "smask": img_info[1],
                        "width": img_info[2],
                        "height": img_info[3],
                        "bpc": img_info[4],
                        "colorspace": img_info[5],
                        "alt_colorspace": img_info[6],
                        "name": img_info[7],
                        "filter": img_info[8],
                    }
                )
        except Exception as exc:
            logger.debug(f"Image extraction skipped on page {page_number}: {exc}")

        # --- Tables (heuristic: text blocks with ≥3 lines of short text) ---
        tables: list[str] = []
        try:
            blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
            for block in blocks:
                if len(block) >= 5 and block[6] == 0:  # block_type 0 = text
                    block_text: str = block[4]
                    lines = [l for l in block_text.splitlines() if l.strip()]
                    # Heuristic: many short lines ≈ table row content
                    if len(lines) >= 3 and all(len(l) < 80 for l in lines):
                        tables.append(block_text)
        except Exception as exc:
            logger.debug(f"Table extraction skipped on page {page_number}: {exc}")

        has_text = len(text.strip()) > 50

        return ExtractedPage(
            page_number=page_number,
            text=text,
            images=images,
            tables=tables,
            has_text=has_text,
        )

    def extract_metadata(self, doc: fitz.Document) -> dict:
        """
        Extract document-level metadata from a fitz.Document.

        Args:
            doc: An open fitz.Document.

        Returns:
            Dictionary with keys: title, author, subject, creator, producer,
            creation_date, mod_date, page_count.
        """
        raw: dict = doc.metadata or {}

        return {
            "title": raw.get("title", ""),
            "author": raw.get("author", ""),
            "subject": raw.get("subject", ""),
            "creator": raw.get("creator", ""),
            "producer": raw.get("producer", ""),
            "creation_date": raw.get("creationDate", ""),
            "mod_date": raw.get("modDate", ""),
            "page_count": len(doc),
        }
