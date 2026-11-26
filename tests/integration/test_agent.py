"""
test_agent.py — Tests d'intégration du workflow LangGraph et validation humaine
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agent.edges import route_after_approval, route_after_intent, route_after_verification
from backend.agent.graph import resume_ticket_approval, run_agent
from backend.llm.llm_client import LLMResponse


def test_agent_routing_rules():
    assert route_after_intent({"needs_clarification": True}) == "request_clarification"
    assert route_after_intent({"needs_image_analysis": True}) == "analyze_image"
    assert route_after_intent({"needs_mcp": True}) == "consult_mcp"
    assert route_after_intent({}) == "retrieve_documents"

    assert route_after_verification({"intent": "ticket"}) == "prepare_ticket"
    assert route_after_verification({"confidence_score": 0.1, "reranked_chunks": []}) == "handle_uncertainty"
    assert route_after_verification({"confidence_score": 0.8, "reranked_chunks": [{"id": "1"}]}) == "end"

    assert route_after_approval({"human_approved": True}) == "create_ticket"
    assert route_after_approval({"human_approved": False}) == "end"


@pytest.mark.asyncio
async def test_ticket_workflow_pauses_for_human_approval_and_resumes():
    """
    Vérifie qu'une demande de ticket :
      1. s'interrompt AVANT create_ticket (requires_human_approval=True)
      2. ne crée PAS le ticket automatiquement
      3. crée le ticket uniquement après resume_ticket_approval(approved=True)
    """
    fake_chunks = [
        {
            "id": "c1",
            "score": 0.92,
            "payload": {
                "document_id": "DOC-P101",
                "filename": "manuel_p101.pdf",
                "page_number": 4,
                "chunk_type": "text",
                "content": "Remplacer le joint mécanique Burgmann MG1-28 en cas de fuite.",
            },
        }
    ]
    fake_llm_resp = LLMResponse(
        content="Selon le manuel, il faut remplacer le joint [Source 1 - Document: manuel_p101.pdf, Page: 4].",
        model="llama3.2:3b",
        tokens_used=42,
        latency_ms=10.0,
    )

    mock_mcp = MagicMock()
    mock_mcp.get_equipment = AsyncMock(return_value={"equipment_id": "EQ-POMPE-001", "name": "Pompe P-101"})
    mock_mcp.get_history = AsyncMock(return_value=[])
    mock_mcp.get_open_tickets = AsyncMock(return_value=[])
    mock_mcp.create_ticket = AsyncMock(
        return_value={"ticket_number": "TKT-DEMO001", "status": "pending"}
    )

    with (
        patch("backend.rag.retriever.SemanticRetriever.retrieve", new=AsyncMock(return_value=fake_chunks)),
        patch("backend.rag.reranker.Reranker.rerank", new=AsyncMock(return_value=fake_chunks)),
        patch("backend.llm.llm_client.llm_client.generate", return_value=fake_llm_resp),
        patch("backend.mcp.client.MCPToolsClient", return_value=mock_mcp),
    ):
        first_run = await run_agent(
            question="Il y a une fuite sur la pompe EQ-POMPE-001, je veux créer un ticket.",
            equipment_id="EQ-POMPE-001",
            user_id="tech_martin",
            user_role="technician",
        )

        # Étape 1 : interruption avant création
        assert first_run["requires_human_approval"] is True
        assert first_run["ticket_draft"] is not None
        assert first_run["ticket_created"] is None
        mock_mcp.create_ticket.assert_not_awaited()

        # Étape 2 : approbation humaine explicite
        token = first_run["ticket_draft"]["approval_token"]
        resumed = await resume_ticket_approval(
            approval_token=token,
            approved=True,
            approver="supervisor_1",
        )

        assert resumed["status"] == "approved"
        assert resumed["ticket_created"]["ticket_number"] == "TKT-DEMO001"
        mock_mcp.create_ticket.assert_awaited_once()
