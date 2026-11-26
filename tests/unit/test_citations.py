"""
test_citations.py — Tests unitaires pour ResponseParser, CitationChecker et ContextBuilder
"""

from __future__ import annotations

from backend.llm.citation_checker import CitationChecker
from backend.llm.response_parser import ResponseParser
from backend.rag.context_builder import ContextBuilder, ContextItem


def test_context_builder_formats_numbered_sources():
    builder = ContextBuilder()
    results = [
        {
            "id": "c1",
            "score": 0.91,
            "rerank_score": 0.95,
            "payload": {
                "document_id": "DOC-001",
                "filename": "manuel_p101.pdf",
                "page_number": 4,
                "chunk_type": "text",
                "content": "Ouvrir la vanne de purge V-101 avant le démarrage de la pompe.",
            },
        }
    ]
    ctx_str, items = builder.build("purge pompe", results)
    assert "[Source 1 - Document: manuel_p101.pdf, Page: 4]" in ctx_str
    assert len(items) == 1
    assert items[0].page_number == 4


def test_citation_checker_validates_grounded_response():
    items = [
        ContextItem(
            chunk_id="c1",
            content="Ouvrir la vanne de purge d'air située sur le corps de pompe P-101 avant démarrage.",
            page_number=2,
            document_id="DOC-001",
            chunk_type="text",
            score=0.9,
        )
    ]
    answer = "Selon la documentation, il faut ouvrir la vanne de purge d'air avant démarrage [Source 1 - Document: DOC-001, Page: 2]."

    checker = CitationChecker()
    report = checker.verify(answer, items)
    assert report["valid"] is True
    assert len(report["verified_citations"]) == 1
    assert report["unverified_citations"] == []
    assert report["hallucination_risk"] is False


def test_citation_checker_flags_nonexistent_source():
    items = [
        ContextItem(
            chunk_id="c1",
            content="Consigner le sectionneur S-101 avant intervention.",
            page_number=1,
            document_id="DOC-001",
            chunk_type="text",
            score=0.8,
        )
    ]
    answer = "Faire la recette de pâtes [Source 9 - Document: Inconnu, Page: 99]."

    checker = CitationChecker()
    report = checker.verify(answer, items)
    assert report["valid"] is False
    assert len(report["unverified_citations"]) == 1
    assert report["hallucination_risk"] is True


def test_response_parser_detects_uncertainty_and_ticket_action():
    parser = ResponseParser()
    items = [ContextItem("c1", "contenu", 1, "DOC-1", "text", 0.9)]

    uncertain = parser.parse("Je ne trouve pas cette information dans les documents.", items)
    assert uncertain.has_uncertainty is True
    assert uncertain.confidence_score < 0.5

    action_resp = parser.parse(
        "Le joint fuit selon [Source 1 - Document: DOC-1, Page: 1]. Je propose de créer un ticket d'intervention.",
        items,
    )
    assert action_resp.suggested_action == "create_ticket"
    assert action_resp.confidence_score >= 0.7
