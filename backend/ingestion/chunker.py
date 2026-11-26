"""
Text Chunker for MaintiDoc.

Splits extracted document text into overlapping, sentence-aware chunks
suitable for embedding and storage in a vector database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.ingestion.pdf_loader import ExtractedDocument

# Sentence boundary pattern — splits after ". ", ".\n", "! ", "? "
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk ready for embedding and indexing."""

    content: str
    """The actual text content of this chunk."""

    chunk_index: int
    """Global index of this chunk within the document (0-based)."""

    page_number: int
    """1-based page number where this chunk originates."""

    chunk_type: str
    """Category of the chunk: 'text', 'table', or 'ocr'."""

    token_count: int
    """Rough estimated token count for this chunk."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary extra metadata (e.g. document_id, equipment_type)."""


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Splits an :class:`~backend.ingestion.pdf_loader.ExtractedDocument` into
    overlapping text chunks while respecting sentence boundaries.

    Args:
        chunk_size:    Target maximum number of *words* per chunk (used as a
                       proxy for tokens).  Defaults to ``settings.rag_chunk_size``.
        chunk_overlap: Number of *words* to overlap between consecutive chunks.
                       Defaults to ``settings.rag_chunk_overlap``.

    Example::

        chunker = TextChunker()
        chunks = chunker.chunk_document(extracted_doc)
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        from backend.config import settings

        self.chunk_size: int = chunk_size if chunk_size is not None else settings.rag_chunk_size
        self.chunk_overlap: int = (
            chunk_overlap if chunk_overlap is not None else settings.rag_chunk_overlap
        )
        logger.debug(
            f"TextChunker initialised: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: "ExtractedDocument") -> list[Chunk]:
        """
        Chunk all pages of a document.

        Args:
            doc: A fully parsed :class:`ExtractedDocument`.

        Returns:
            Ordered list of :class:`Chunk` objects across all pages.
        """
        all_chunks: list[Chunk] = []
        global_index = 0

        for page in doc.pages:
            if page.text and page.has_text:
                page_chunks = self._chunk_text(
                    page.text, page.page_number, global_index, chunk_type="text"
                )
                all_chunks.extend(page_chunks)
                global_index += len(page_chunks)

            # Also chunk table blocks separately
            for table_text in page.tables:
                table_chunks = self._chunk_text(
                    table_text, page.page_number, global_index, chunk_type="table"
                )
                all_chunks.extend(table_chunks)
                global_index += len(table_chunks)

        logger.info(
            f"Chunked document '{doc.filename}': "
            f"{len(doc.pages)} pages → {len(all_chunks)} chunks."
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        start_index: int,
        chunk_type: str = "text",
    ) -> list[Chunk]:
        """
        Split *text* into overlapping chunks, trying to honour sentence ends.

        Args:
            text:        Raw text to split.
            page_number: Source page number (1-based).
            start_index: Global chunk index offset for this batch.
            chunk_type:  Semantic type label ('text', 'table', 'ocr').

        Returns:
            List of :class:`Chunk` objects.
        """
        # Split into sentences first
        sentences = _SENTENCE_BOUNDARY.split(text.strip())
        # Flatten further on newlines within each sentence fragment
        sentences = [s.replace("\n", " ").strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_words: list[str] = []
        local_index = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            if not sentence_words:
                continue

            # If adding this sentence would exceed chunk_size, flush first
            if current_words and len(current_words) + len(sentence_words) > self.chunk_size:
                chunk_text = " ".join(current_words)
                if len(chunk_text) >= 50:
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            chunk_index=start_index + local_index,
                            page_number=page_number,
                            chunk_type=chunk_type,
                            token_count=self._estimate_tokens(chunk_text),
                        )
                    )
                    local_index += 1

                # Keep overlap: last `chunk_overlap` words carry forward
                if self.chunk_overlap > 0:
                    current_words = current_words[-self.chunk_overlap:]
                else:
                    current_words = []

            current_words.extend(sentence_words)

        # Flush remaining words
        if current_words:
            chunk_text = " ".join(current_words)
            if len(chunk_text) >= 50:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_index=start_index + local_index,
                        page_number=page_number,
                        chunk_type=chunk_type,
                        token_count=self._estimate_tokens(chunk_text),
                    )
                )

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token-count estimate.

        Uses the common heuristic: ``word_count * 1.3`` (accounts for
        sub-word tokenisation in BPE-based models).

        Args:
            text: Input string.

        Returns:
            Estimated integer token count.
        """
        return int(len(text.split()) * 1.3)
