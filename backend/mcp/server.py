"""
server.py — Serveur MCP MaintiDoc (FastMCP)
===========================================

Expose les outils métier (équipements, historique, tickets) via le protocole MCP.
Chaque outil valide les permissions côté serveur avant d'accéder à PostgreSQL.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from backend.db.database import AsyncSessionLocal
from backend.mcp import tools


mcp = FastMCP(
    name="MaintiDoc MCP Server",
    instructions=(
        "Serveur MCP pour MaintiDoc. Fournit l'accès en lecture aux équipements et "
        "historiques d'interventions, et la création de tickets avec token d'approbation."
    ),
)


@mcp.tool()
async def get_equipment(equipment_id: str, user_role: str = "technician") -> dict[str, Any]:
    """Consulte la fiche d'un équipement industriel par son identifiant (ex: EQ-POMPE-001)."""
    try:
        async with AsyncSessionLocal() as session:
            return await tools.get_equipment_info(equipment_id, session, user_role=user_role)
    except Exception as exc:
        return {"error": str(exc), "equipment_id": equipment_id}


@mcp.tool()
async def history(
    equipment_id: str,
    limit: int = 10,
    user_role: str = "technician",
) -> list[dict[str, Any]]:
    """Consulte l'historique des dernières interventions d'un équipement."""
    try:
        async with AsyncSessionLocal() as session:
            return await tools.get_equipment_history(
                equipment_id, session, limit=limit, user_role=user_role
            )
    except Exception as exc:
        return [{"error": str(exc), "equipment_id": equipment_id}]


@mcp.tool()
async def get_open_tickets(equipment_id: str, user_role: str = "technician") -> list[dict[str, Any]]:
    """Liste les tickets de maintenance encore ouverts pour un équipement."""
    try:
        async with AsyncSessionLocal() as session:
            return await tools.get_open_tickets(equipment_id, session, user_role=user_role)
    except Exception as exc:
        return [{"error": str(exc), "equipment_id": equipment_id}]


@mcp.tool()
async def create_ticket(
    equipment_id: str,
    title: str,
    description: str,
    approval_token: str,
    priority: str = "medium",
    user_role: str = "technician",
) -> dict[str, Any]:
    """
    Crée un ticket de maintenance (nécessite un approval_token issu d'une validation humaine).
    """
    try:
        async with AsyncSessionLocal() as session:
            return await tools.create_maintenance_ticket(
                equipment_id=equipment_id,
                title=title,
                description=description,
                priority=priority,
                requested_by_id=None,
                approval_token=approval_token,
                source_chunks=[],
                agent_reasoning="Créé via outil MCP",
                db=session,
                user_role=user_role,
            )
    except Exception as exc:
        return {"error": str(exc), "equipment_id": equipment_id}


if __name__ == "__main__":
    mcp.run()
