"""
crud.py — Async CRUD operations for MaintiDoc.

All functions accept an AsyncSession and operate on the
SQLAlchemy ORM models defined in backend.db.models.
Idempotency and soft-error patterns are applied where needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, Equipment, Intervention, Ticket, User
from backend.db.schemas import UserCreate

# Password hashing context (also used by auth.py)
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    """Return the User matching *username*, or None if not found."""
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def create_user(session: AsyncSession, data: UserCreate) -> User:
    """
    Hash the plain-text password and persist a new User row.

    Raises:
        ValueError: if the username or email already exists.
    """
    # Check uniqueness
    existing = await get_user_by_username(session, data.username)
    if existing:
        raise ValueError(f"Username '{data.username}' is already taken.")

    hashed = _pwd_context.hash(data.password)
    user = User(
        id=uuid.uuid4(),
        username=data.username,
        email=data.email,
        hashed_password=hashed,
        role=data.role,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info(f"Created user: {user.username} (role={user.role})")
    return user


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def create_document(session: AsyncSession, data: dict) -> Document:
    """
    Insert a new Document row from a plain dict of fields.

    Args:
        data: mapping of Document column names to values.

    Returns:
        The persisted Document instance.
    """
    doc = Document(id=uuid.uuid4(), **data)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    logger.info(f"Created document: {doc.filename} (id={doc.id})")
    return doc


async def update_document_status(
    session: AsyncSession,
    document_id: str,
    status: str,
    error_msg: Optional[str] = None,
) -> Document:
    """
    Update the *status* (and optionally *error_message*) of a Document.

    Args:
        document_id: UUID string of the target document.
        status: New status value (e.g. "ready", "error").
        error_msg: Optional error details to store.

    Returns:
        The updated Document instance.

    Raises:
        ValueError: if the document is not found.
    """
    result = await session.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise ValueError(f"Document {document_id} not found.")

    doc.status = status
    if error_msg is not None:
        doc.error_message = error_msg
    doc.updated_at = datetime.now(tz=timezone.utc)

    await session.commit()
    await session.refresh(doc)
    logger.info(f"Document {document_id} → status={status}")
    return doc


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------


async def get_equipment(session: AsyncSession, equipment_id: str) -> Equipment | None:
    """
    Look up equipment by its human-readable *equipment_id* field (not the UUID pk).

    Args:
        equipment_id: Business identifier (e.g. "PUMP-001").

    Returns:
        Equipment instance or None.
    """
    result = await session.execute(
        select(Equipment).where(Equipment.equipment_id == equipment_id)
    )
    return result.scalar_one_or_none()


async def get_equipment_history(
    session: AsyncSession,
    equipment_id: str,
    limit: int = 10,
) -> list[Intervention]:
    """
    Return the last *limit* interventions for a piece of equipment,
    ordered by started_at DESC.

    Args:
        equipment_id: Business identifier (e.g. "PUMP-001").
        limit: Maximum number of records to return.

    Returns:
        List of Intervention instances (may be empty).
    """
    equipment = await get_equipment(session, equipment_id)
    if equipment is None:
        return []

    result = await session.execute(
        select(Intervention)
        .where(Intervention.equipment_id == equipment.id)
        .order_by(Intervention.started_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


async def get_open_tickets(session: AsyncSession, equipment_id: str) -> list[Ticket]:
    """
    Return all non-terminal tickets for a piece of equipment.

    Open statuses: draft, pending, approved, in_progress.

    Args:
        equipment_id: Business identifier string.

    Returns:
        List of Ticket instances.
    """
    equipment = await get_equipment(session, equipment_id)
    if equipment is None:
        return []

    open_statuses = ["draft", "pending", "approved", "in_progress"]
    result = await session.execute(
        select(Ticket)
        .where(Ticket.equipment_id == equipment.id)
        .where(Ticket.status.in_(open_statuses))
        .order_by(Ticket.created_at.desc())
    )
    return list(result.scalars().all())


async def create_ticket(session: AsyncSession, data: dict) -> Ticket:
    """
    Create a maintenance ticket with idempotency support.

    If *idempotency_key* is present in *data* and a ticket with that key
    already exists, the existing ticket is returned without creating a duplicate.

    Args:
        data: Mapping of Ticket column names to values.

    Returns:
        New or existing Ticket instance.
    """
    idempotency_key = data.get("idempotency_key")
    if idempotency_key:
        result = await session.execute(
            select(Ticket).where(Ticket.idempotency_key == idempotency_key)
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info(f"Idempotent ticket hit: key={idempotency_key} → {existing.ticket_number}")
            return existing

    # Generate a human-readable ticket number
    ticket_number = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    ticket = Ticket(id=uuid.uuid4(), ticket_number=ticket_number, **data)
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    logger.info(f"Created ticket: {ticket.ticket_number} (id={ticket.id})")
    return ticket


async def approve_ticket(
    session: AsyncSession,
    ticket_id: str,
    approved_by_id: str,
) -> Ticket:
    """
    Transition a ticket to *approved* status.

    Args:
        ticket_id: UUID string of the ticket.
        approved_by_id: UUID string of the approving user.

    Returns:
        Updated Ticket instance.

    Raises:
        ValueError: if the ticket is not found.
    """
    result = await session.execute(
        select(Ticket).where(Ticket.id == uuid.UUID(ticket_id))
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found.")

    ticket.status = "approved"
    ticket.approved_by = uuid.UUID(approved_by_id)
    ticket.approved_at = datetime.now(tz=timezone.utc)
    ticket.updated_at = datetime.now(tz=timezone.utc)

    await session.commit()
    await session.refresh(ticket)
    logger.info(f"Ticket {ticket.ticket_number} approved by {approved_by_id}")
    return ticket


async def reject_ticket(
    session: AsyncSession,
    ticket_id: str,
    reason: str,
) -> Ticket:
    """
    Transition a ticket to *rejected* status with a reason.

    Args:
        ticket_id: UUID string of the ticket.
        reason: Human-readable rejection explanation.

    Returns:
        Updated Ticket instance.

    Raises:
        ValueError: if the ticket is not found.
    """
    result = await session.execute(
        select(Ticket).where(Ticket.id == uuid.UUID(ticket_id))
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found.")

    ticket.status = "rejected"
    ticket.rejection_reason = reason
    ticket.updated_at = datetime.now(tz=timezone.utc)

    await session.commit()
    await session.refresh(ticket)
    logger.info(f"Ticket {ticket.ticket_number} rejected: {reason}")
    return ticket
