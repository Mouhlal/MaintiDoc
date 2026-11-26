"""
tools.py — Logique métier des outils MCP MaintiDoc
===================================================

Implémente les opérations consultables par l'agent et le serveur FastMCP :
  - get_equipment_info        : fiche équipement
  - get_equipment_history     : historique des interventions
  - get_open_tickets          : tickets ouverts sur un équipement
  - create_maintenance_ticket : création idempotente de ticket après validation
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import crud
from backend.mcp.permissions import check_mcp_permission


def _equipment_to_dict(eq: Any) -> dict[str, Any]:
    return {
        "id": str(eq.id),
        "equipment_id": eq.equipment_id,
        "name": eq.name,
        "equipment_type": eq.equipment_type,
        "location": eq.location,
        "building": eq.building,
        "manufacturer": eq.manufacturer,
        "model": eq.model,
        "serial_number": eq.serial_number,
        "status": eq.status,
        "last_maintenance_date": (
            eq.last_maintenance_date.isoformat() if eq.last_maintenance_date else None
        ),
        "maintenance_interval_days": eq.maintenance_interval_days,
    }


def _intervention_to_dict(item: Any) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "intervention_type": item.intervention_type,
        "title": item.title,
        "description": item.description,
        "technician_name": item.technician_name,
        "started_at": item.started_at.isoformat() if item.started_at else None,
        "completed_at": item.completed_at.isoformat() if item.completed_at else None,
        "duration_minutes": item.duration_minutes,
        "result": item.result,
        "notes": item.notes,
    }


def _ticket_to_dict(ticket: Any) -> dict[str, Any]:
    return {
        "id": str(ticket.id),
        "ticket_number": ticket.ticket_number,
        "title": ticket.title,
        "description": ticket.description,
        "priority": ticket.priority,
        "status": ticket.status,
        "idempotency_key": ticket.idempotency_key,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "approved_at": ticket.approved_at.isoformat() if ticket.approved_at else None,
    }


async def get_equipment_info(
    equipment_id: str,
    db: AsyncSession,
    user_role: str = "technician",
) -> dict[str, Any]:
    """Récupère les informations d'un équipement par son identifiant métier."""
    check_mcp_permission(user_role, "get_equipment")
    equipment = await crud.get_equipment(db, equipment_id)
    if equipment is None:
        raise ValueError(f"Équipement '{equipment_id}' introuvable.")
    return _equipment_to_dict(equipment)


async def get_equipment_history(
    equipment_id: str,
    db: AsyncSession,
    limit: int = 10,
    user_role: str = "technician",
) -> list[dict[str, Any]]:
    """Retourne les *limit* dernières interventions sur l'équipement."""
    check_mcp_permission(user_role, "get_history")
    interventions = await crud.get_equipment_history(db, equipment_id, limit=limit)
    return [_intervention_to_dict(item) for item in interventions]


async def get_open_tickets(
    equipment_id: str,
    db: AsyncSession,
    user_role: str = "technician",
) -> list[dict[str, Any]]:
    """Retourne les tickets non clôturés pour un équipement."""
    check_mcp_permission(user_role, "get_open_tickets")
    tickets = await crud.get_open_tickets(db, equipment_id)
    return [_ticket_to_dict(t) for t in tickets]


async def create_maintenance_ticket(
    equipment_id: str,
    title: str,
    description: str,
    priority: str,
    requested_by_id: str | None,
    approval_token: str,
    source_chunks: list[str],
    agent_reasoning: str,
    db: AsyncSession,
    user_role: str = "technician",
) -> dict[str, Any]:
    """
    Crée un ticket en base avec vérification du token d'approbation humaine
    et clé d'idempotence (approval_token).
    """
    check_mcp_permission(user_role, "create_ticket", approval_token=approval_token)

    equipment = await crud.get_equipment(db, equipment_id)
    if equipment is None:
        raise ValueError(f"Équipement '{equipment_id}' introuvable pour la création de ticket.")

    requester_uuid = None
    if requested_by_id:
        try:
            requester_uuid = uuid.UUID(str(requested_by_id))
        except ValueError:
            requester_uuid = None

    ticket = await crud.create_ticket(
        db,
        {
            "equipment_id": equipment.id,
            "title": title[:500],
            "description": description,
            "priority": priority if priority in {"low", "medium", "high", "critical"} else "medium",
            "status": "pending",
            "requested_by": requester_uuid,
            "idempotency_key": approval_token,
            "source_chunks": source_chunks or [],
            "agent_reasoning": agent_reasoning,
        },
    )
    logger.info(f"Ticket MCP créé : {ticket.ticket_number} pour {equipment_id}")
    return _ticket_to_dict(ticket)
