"""
tickets.py — Routes tickets, validation humaine et équipements MCP
===================================================================

Endpoints :
    GET  /api/v1/tickets/                       → Lister les tickets
    GET  /api/v1/tickets/pending-approvals      → Brouillons en attente de validation
    POST /api/v1/tickets/approve-draft          → Valider/rejeter un brouillon de l'agent
    POST /api/v1/tickets/{ticket_id}/approve    → Approuver un ticket existant
    POST /api/v1/tickets/{ticket_id}/reject     → Rejeter un ticket existant
    GET  /api/v1/tickets/equipments             → Lister les équipements
    GET  /api/v1/tickets/equipments/{eq_id}     → Détails + historique équipement
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.graph import resume_ticket_approval
from backend.agent.human_approval import approval_manager
from backend.db import crud
from backend.db.database import get_db
from backend.db.models import Equipment, Ticket
from backend.db.schemas import EquipmentResponse, TicketResponse
from backend.security.input_validator import validate_equipment_id

router = APIRouter()


class DraftApprovalRequest(BaseModel):
    approval_token: str
    approved: bool
    approver: str = "supervisor_demo"
    rejection_reason: str = ""


class TicketRejectRequest(BaseModel):
    reason: str


@router.get("/", response_model=list[TicketResponse], summary="Lister tous les tickets")
async def list_tickets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Ticket).order_by(Ticket.created_at.desc()))
    return list(result.scalars().all())


@router.get("/pending-approvals", summary="Brouillons de tickets en attente de validation humaine")
async def list_pending_approvals() -> list[dict[str, Any]]:
    pending = [
        {
            "approval_token": rec.approval_token,
            "thread_id": rec.thread_id,
            "status": rec.status,
            "requested_by": rec.requested_by,
            "created_at": rec.created_at.isoformat(),
            "ticket_draft": rec.ticket_draft,
        }
        for rec in approval_manager._pending.values()
        if rec.status == "pending"
    ]
    return pending


@router.post("/approve-draft", summary="Approuver ou rejeter un brouillon généré par l'agent")
async def approve_or_reject_draft(payload: DraftApprovalRequest):
    try:
        result = await resume_ticket_approval(
            approval_token=payload.approval_token,
            approved=payload.approved,
            approver=payload.approver,
            rejection_reason=payload.rejection_reason,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{ticket_id}/approve", response_model=TicketResponse, summary="Approuver un ticket en base")
async def approve_existing_ticket(
    ticket_id: str,
    approver_id: str = "00000000-0000-0000-0000-000000000001",
    db: AsyncSession = Depends(get_db),
):
    try:
        return await crud.approve_ticket(db, ticket_id=ticket_id, approved_by_id=approver_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{ticket_id}/reject", response_model=TicketResponse, summary="Rejeter un ticket en base")
async def reject_existing_ticket(
    ticket_id: str,
    payload: TicketRejectRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await crud.reject_ticket(db, ticket_id=ticket_id, reason=payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/equipments", response_model=list[EquipmentResponse], summary="Lister les équipements")
async def list_equipments(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Equipment).order_by(Equipment.equipment_id))
    return list(result.scalars().all())


@router.get("/equipments/{equipment_id}", summary="Fiche équipement et historique des interventions")
async def get_equipment_details(equipment_id: str, db: AsyncSession = Depends(get_db)):
    clean_id = validate_equipment_id(equipment_id)
    equipment = await crud.get_equipment(db, clean_id)
    if equipment is None:
        raise HTTPException(status_code=404, detail=f"Équipement '{clean_id}' introuvable")

    history = await crud.get_equipment_history(db, clean_id, limit=10)
    open_tickets = await crud.get_open_tickets(db, clean_id)

    return {
        "equipment": {
            "equipment_id": equipment.equipment_id,
            "name": equipment.name,
            "equipment_type": equipment.equipment_type,
            "location": equipment.location,
            "manufacturer": equipment.manufacturer,
            "model": equipment.model,
            "status": equipment.status,
            "last_maintenance_date": (
                equipment.last_maintenance_date.isoformat()
                if equipment.last_maintenance_date
                else None
            ),
        },
        "history": [
            {
                "intervention_type": h.intervention_type,
                "title": h.title,
                "description": h.description,
                "technician_name": h.technician_name,
                "started_at": h.started_at.isoformat() if h.started_at else None,
                "result": h.result,
                "notes": h.notes,
            }
            for h in history
        ],
        "open_tickets": [
            {
                "ticket_number": t.ticket_number,
                "title": t.title,
                "priority": t.priority,
                "status": t.status,
            }
            for t in open_tickets
        ],
    }
