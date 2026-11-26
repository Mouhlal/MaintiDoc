"""
edges.py — Transitions conditionnelles du graphe LangGraph
==========================================================

Décide dynamiquement du prochain nœud à exécuter en fonction
de l'état courant (AgentState).
"""

from __future__ import annotations

from backend.agent.state import AgentState


def route_after_intent(state: AgentState) -> str:
    """
    Après analyse de l'intention :
      - si la question est trop vague → request_clarification
      - si une image est fournie      → analyze_image
      - si consultation MCP requise    → consult_mcp
      - sinon                          → retrieve_documents
    """
    if state.get("needs_clarification"):
        return "request_clarification"
    if state.get("needs_image_analysis"):
        return "analyze_image"
    if state.get("needs_mcp"):
        return "consult_mcp"
    return "retrieve_documents"


def route_after_image(state: AgentState) -> str:
    """Après analyse d'image, consulte MCP si besoin, sinon récupère les docs."""
    if state.get("needs_mcp"):
        return "consult_mcp"
    return "retrieve_documents"


def route_after_verification(state: AgentState) -> str:
    """
    Après vérification des citations et confiance :
      - si intent == "ticket"       → prepare_ticket
      - si confiance très faible    → handle_uncertainty
      - sinon                       → end
    """
    if state.get("intent") == "ticket":
        return "prepare_ticket"

    confidence = state.get("confidence_score", 0.0)
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    if not chunks or confidence < 0.25:
        return "handle_uncertainty"

    return "end"


def route_after_approval(state: AgentState) -> str:
    """
    Après la validation humaine :
      - si approuvé → create_ticket
      - sinon       → end (annulation propre)
    """
    if state.get("human_approved"):
        return "create_ticket"
    return "end"
