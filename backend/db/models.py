"""
models.py — SQLAlchemy async ORM models for MaintiDoc.

Defines the database schema for Users, Documents, Chunks,
Equipment, Interventions, and Tickets using SQLAlchemy's
DeclarativeBase with async support.
"""

import uuid
from datetime import datetime, date

from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    Text,
    DateTime,
    Date,
    JSON,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as SAUUID
from sqlalchemy.orm import relationship, DeclarativeBase


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(Base):
    """Represents a system user (technician, manager, admin)."""

    __tablename__ = "users"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="technician")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    documents = relationship("Document", back_populates="uploader", foreign_keys="Document.uploaded_by")
    interventions = relationship("Intervention", back_populates="technician", foreign_keys="Intervention.technician_id")
    requested_tickets = relationship("Ticket", back_populates="requester", foreign_keys="Ticket.requested_by")
    approved_tickets = relationship("Ticket", back_populates="approver", foreign_keys="Ticket.approved_by")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username} role={self.role}>"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(Base):
    """Represents an uploaded maintenance document (PDF, etc.)."""

    __tablename__ = "documents"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    filename = Column(String(500), nullable=False)
    title = Column(String(500), nullable=True)
    doc_type = Column(String(100), nullable=True)
    equipment_type = Column(String(100), nullable=True)
    version = Column(String(50), nullable=True)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=False, default=0)
    storage_path = Column(String(1000), nullable=True)
    status = Column(String(50), nullable=False, default="pending")  # pending | processing | ready | error
    error_message = Column(Text, nullable=True)
    uploaded_by = Column(
        SAUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    permissions = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    uploader = relationship("User", back_populates="documents", foreign_keys=[uploaded_by])
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(Base):
    """
    Represents a text or visual chunk extracted from a Document.
    Each chunk is indexed in Qdrant and linked back to the source page.
    """

    __tablename__ = "chunks"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    document_id = Column(
        SAUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    qdrant_id = Column(SAUUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    page_number = Column(Integer, nullable=False)
    chunk_type = Column(String(50), nullable=False, default="text")  # text | image
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    image_path = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} page={self.page_number} type={self.chunk_type}>"


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------

class Equipment(Base):
    """Represents a physical piece of equipment tracked in the CMMS."""

    __tablename__ = "equipments"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    equipment_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(500), nullable=False)
    equipment_type = Column(String(100), nullable=False)
    location = Column(String(500), nullable=True)
    building = Column(String(100), nullable=True)
    manufacturer = Column(String(255), nullable=True)
    model = Column(String(255), nullable=True)
    serial_number = Column(String(255), nullable=True)
    status = Column(String(100), nullable=False, default="operational")
    last_maintenance_date = Column(Date, nullable=True)
    maintenance_interval_days = Column(Integer, nullable=False, default=90)
    extra_metadata = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    interventions = relationship("Intervention", back_populates="equipment")
    tickets = relationship("Ticket", back_populates="equipment")

    def __repr__(self) -> str:
        return f"<Equipment equipment_id={self.equipment_id} name={self.name}>"


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

class Intervention(Base):
    """
    Records a completed or ongoing maintenance intervention
    performed on a piece of equipment.
    """

    __tablename__ = "interventions"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    equipment_id = Column(
        SAUUID(as_uuid=True),
        ForeignKey("equipments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    intervention_type = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    technician_id = Column(
        SAUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    technician_name = Column(String(255), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    result = Column(String(100), nullable=False, default="completed")
    parts_used = Column(JSON, nullable=False, default=list)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    equipment = relationship("Equipment", back_populates="interventions")
    technician = relationship("User", back_populates="interventions", foreign_keys=[technician_id])

    def __repr__(self) -> str:
        return f"<Intervention id={self.id} type={self.intervention_type} result={self.result}>"


# ---------------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------------

class Ticket(Base):
    """
    Represents a maintenance work-order ticket generated by the AI agent
    or submitted manually. Requires human approval before execution.
    """

    __tablename__ = "tickets"

    id = Column(
        SAUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    equipment_id = Column(
        SAUUID(as_uuid=True),
        ForeignKey("equipments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ticket_number = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(String(50), nullable=False, default="medium")  # low | medium | high | critical
    status = Column(String(100), nullable=False, default="draft")    # draft | pending | approved | rejected | in_progress | done
    requested_by = Column(
        SAUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    idempotency_key = Column(String(255), unique=True, nullable=True, index=True)
    approved_by = Column(
        SAUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    approved_at = Column(DateTime(timezone=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    source_chunks = Column(JSON, nullable=False, default=list)
    agent_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    equipment = relationship("Equipment", back_populates="tickets")
    requester = relationship("User", back_populates="requested_tickets", foreign_keys=[requested_by])
    approver = relationship("User", back_populates="approved_tickets", foreign_keys=[approved_by])

    def __repr__(self) -> str:
        return f"<Ticket ticket_number={self.ticket_number} status={self.status} priority={self.priority}>"
