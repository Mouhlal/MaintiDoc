"""
graph.py — Assemblage du graphe d'agent LangGraph pour MaintiDoc
================================================================

Construit le StateGraph complet avec :
  - mémoire par thread (MemorySaver)
  - interruption avant `create_ticket` pour la validation humaine
  - point d'entrée simplifié `run_agent()` et `resume_ticket_approval()`
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from backend.agent import edges, nodes
from backend.agent.human_approval import approval_manager
from backend.agent.state import AgentState


def build_agent_graph(with_checkpointer: bool = True):
    """
    Construit et compile le workflow LangGraph de MaintiDoc.
    """
    workflow = StateGraph(AgentState)

    # 1. Déclaration des nœuds
    workflow.add_node("analyze_intent", nodes.analyze_intent)
    workflow.add_node("request_clarification", nodes.request_clarification)
    workflow.add_node("analyze_image", nodes.analyze_image_node)
    workflow.add_node("consult_mcp", nodes.consult_mcp)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("rerank_results", nodes.rerank_results)
    workflow.add_node("generate_response", nodes.generate_response)
    workflow.add_node("verify_citations", nodes.verify_citations)
    workflow.add_node("handle_uncertainty", nodes.handle_uncertainty)
    workflow.add_node("prepare_ticket", nodes.prepare_ticket)
    workflow.add_node("request_human_approval", nodes.request_human_approval)
    workflow.add_node("create_ticket", nodes.create_ticket)

    # 2. Transitions
    workflow.add_edge(START, "analyze_intent")

    workflow.add_conditional_edges(
        "analyze_intent",
        edges.route_after_intent,
        {
            "request_clarification": "request_clarification",
            "analyze_image": "analyze_image",
            "consult_mcp": "consult_mcp",
            "retrieve_documents": "retrieve_documents",
        },
    )

    workflow.add_edge("request_clarification", END)

    workflow.add_conditional_edges(
        "analyze_image",
        edges.route_after_image,
        {
            "consult_mcp": "consult_mcp",
            "retrieve_documents": "retrieve_documents",
        },
    )

    workflow.add_edge("consult_mcp", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "rerank_results")
    workflow.add_edge("rerank_results", "generate_response")
    workflow.add_edge("generate_response", "verify_citations")

    workflow.add_conditional_edges(
        "verify_citations",
        edges.route_after_verification,
        {
            "prepare_ticket": "prepare_ticket",
            "handle_uncertainty": "handle_uncertainty",
            "end": END,
        },
    )

    workflow.add_edge("handle_uncertainty", END)
    workflow.add_edge("prepare_ticket", "request_human_approval")

    workflow.add_conditional_edges(
        "request_human_approval",
        edges.route_after_approval,
        {
            "create_ticket": "create_ticket",
            "end": END,
        },
    )

    workflow.add_edge("create_ticket", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["request_human_approval"] if with_checkpointer else None,
    )


agent_graph = build_agent_graph(with_checkpointer=True)


async def run_agent(
    question: str,
    user_id: str = "anonymous",
    user_role: str = "technician",
    document_ids: list[str] | None = None,
    equipment_id: str | None = None,
    image_bytes: bytes | None = None,
    image_base64: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    Exécute le graphe agent pour une requête utilisateur.
    Si un brouillon de ticket est préparé, enregistre l'approbation en attente.
    """
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    initial_state: AgentState = {
        "question": question,
        "user_id": user_id,
        "user_role": user_role,
        "document_ids": document_ids,
        "equipment_id": equipment_id,
        "image_bytes": image_bytes,
        "image_base64": image_base64,
        "messages": [],
        "errors": [],
        "human_approved": False,
    }

    result = await agent_graph.ainvoke(initial_state, config=config)

    ticket_draft = result.get("ticket_draft")
    approval_token = result.get("approval_token")
    if ticket_draft and approval_token:
        approval_manager.register(
            approval_token=approval_token,
            thread_id=tid,
            ticket_draft=ticket_draft,
            requested_by=user_id,
        )
        logger.info(f"Approbation enregistrée pour thread={tid}, token={approval_token[:8]}...")

    return {
        "thread_id": tid,
        "answer": result.get("final_answer") or result.get("llm_response", ""),
        "sources": result.get("sources", []),
        "confidence": float(result.get("confidence_score", 0.0)),
        "requires_human_approval": bool(ticket_draft and not result.get("ticket_created")),
        "ticket_draft": ticket_draft,
        "ticket_created": result.get("ticket_created"),
        "errors": result.get("errors", []),
    }


async def resume_ticket_approval(
    approval_token: str,
    approved: bool,
    approver: str,
    rejection_reason: str = "",
) -> dict[str, Any]:
    """
    Reprend un workflow interrompu après décision humaine (approuver/rejeter).
    """
    record = approval_manager.get(approval_token)
    if record is None:
        raise ValueError("Demande d'approbation introuvable ou expirée.")

    if not approved:
        approval_manager.reject(approval_token, approver=approver, reason=rejection_reason)
        return {
            "status": "rejected",
            "reason": rejection_reason or "Rejeté par le superviseur",
            "ticket_created": None,
        }

    approval_manager.approve(approval_token, approver=approver)
    config = {"configurable": {"thread_id": record.thread_id}}
    await agent_graph.aupdate_state(
        config,
        {"human_approved": True},
        as_node="request_human_approval",
    )
    final_state = await agent_graph.ainvoke(None, config=config)
    return {
        "status": "approved",
        "ticket_created": final_state.get("ticket_created"),
        "answer": final_state.get("final_answer", ""),
    }
