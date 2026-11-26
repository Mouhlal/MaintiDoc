"""
test_chunker.py — Tests unitaires pour TextChunker et PDFLoader
"""

from __future__ import annotations

import fitz

from backend.ingestion.chunker import TextChunker
from backend.ingestion.pdf_loader import ExtractedDocument, ExtractedPage, PDFLoader


def test_chunker_splits_long_text_with_overlap():
    sentences = [
        f"La procédure numéro {i} impose de vérifier le joint mécanique et la pression d'aspiration."
        for i in range(1, 25)
    ]
    full_text = " ".join(sentences)
    doc = ExtractedDocument(
        file_path="/tmp/test.pdf",
        filename="test.pdf",
        total_pages=1,
        pages=[ExtractedPage(page_number=1, text=full_text, images=[], tables=[], has_text=True)],
        metadata={},
    )

    chunker = TextChunker(chunk_size=60, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.chunk_type == "text" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_filters_short_fragments():
    doc = ExtractedDocument(
        file_path="/tmp/short.pdf",
        filename="short.pdf",
        total_pages=1,
        pages=[ExtractedPage(page_number=1, text="Trop court.", images=[], tables=[], has_text=True)],
        metadata={},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk_document(doc) == []


def test_pdf_loader_reads_generated_pdf(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 72),
        "Manuel technique pompe centrifuge P-101 : ouvrir la vanne de purge avant démarrage. " * 3,
    )
    doc.save(str(pdf_file))
    doc.close()

    loader = PDFLoader()
    extracted = loader.load(pdf_file)

    assert extracted.total_pages == 1
    assert extracted.filename == "sample.pdf"
    assert extracted.pages[0].has_text is True
    assert "pompe centrifuge" in extracted.pages[0].text
