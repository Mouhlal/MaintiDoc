"""
schemas.py — Pydantic v2 schemas for the MaintiDoc API.

Defines request (input) and response (output) schemas for all
API endpoints: users, documents, equipment, interventions,
tickets, and chat.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """Payload for creating a new user account."""

    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(default="technician")


class UserResponse(BaseModel):
    """Public representation of a user (never includes hashed_password)."""

    id: UUID
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
    """Payload for registering an uploaded document."""

    filename: str = Field(..., max_length=500)
    title: str | None = None
    doc_type: str | None = None
    equipment_type: str | None = None
    version: str | None = None


class DocumentResponse(BaseModel):
    """Public representation of a processed document."""

    id: UUID
    filename: str
    title: str | None
    status: str
    total_pages: int | None
    total_chunks: int
    equipment_type: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Equipment schemas
# ---------------------------------------------------------------------------

class EquipmentResponse(BaseModel):
    """Public representation of a piece of equipment."""

    id: UUID
    equipment_id: str
    name: str
    equipment_type: str
    location: str | None
    status: str
    last_maintenance_date: date | None
    manufacturer: str | None
    model: str | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Intervention schemas
# ---------------------------------------------------------------------------

class InterventionResponse(BaseModel):
    """Public representation of a maintenance intervention record."""

    id: UUID
    intervention_type: str
    title: str
    description: str | None
    technician_name: str | None
    started_at: datetime | None
    completed_at: datetime | None
    result: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Ticket schemas
# ---------------------------------------------------------------------------

class TicketCreate(BaseModel):
    """Payload for creating a maintenance ticket."""

    equipment_id: str
    title: str = Field(..., max_length=500)
    description: str
    priority: str = Field(default="medium")


class TicketResponse(BaseModel):
    """Public representation of a maintenance ticket."""

    id: UUID
    ticket_number: str
    title: str
    description: str
    priority: str
    status: str
    created_at: datetime
    approved_at: datetime | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Payload sent by the front-end to the RAG/agent chat endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = None
    equipment_id: str | None = None
    image_base64: str | None = None


class ChatResponse(BaseModel):
    """Structured response from the RAG/agent chat endpoint."""

    answer: str
    sources: list[dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human_approval: bool = False
    ticket_draft: dict[str, Any] | None = None
