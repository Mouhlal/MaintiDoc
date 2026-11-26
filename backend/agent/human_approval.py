"""
human_approval.py — Gestion de la validation humaine (Human-in-the-loop)
========================================================================

Maintient un registre sécurisé des demandes de tickets en attente
d'approbation explicite par un humain, avec vérification du token
d'approbation pour garantir l'idempotence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PendingApproval:
    """Représente un brouillon de ticket en attente de validation."""

    approval_token: str
    thread_id: str
    ticket_draft: dict[str, Any]
    requested_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    status: str = "pending"  # pending | approved | rejected
    decision_by: str | None = None
    rejection_reason: str | None = None


class ApprovalManager:
    """Registre en mémoire des approbations humaines en attente."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def register(
        self,
        approval_token: str,
        thread_id: str,
        ticket_draft: dict[str, Any],
        requested_by: str,
    ) -> PendingApproval:
        record = PendingApproval(
            approval_token=approval_token,
            thread_id=thread_id,
            ticket_draft=ticket_draft,
            requested_by=requested_by,
        )
        self._pending[approval_token] = record
        return record

    def get(self, approval_token: str) -> PendingApproval | None:
        return self._pending.get(approval_token)

    def approve(self, approval_token: str, approver: str) -> PendingApproval:
        record = self._pending.get(approval_token)
        if record is None:
            raise ValueError("Token d'approbation introuvable ou expiré.")
        if record.status != "pending":
            return record  # Idempotence
        record.status = "approved"
        record.decision_by = approver
        return record

    def reject(self, approval_token: str, approver: str, reason: str) -> PendingApproval:
        record = self._pending.get(approval_token)
        if record is None:
            raise ValueError("Token d'approbation introuvable ou expiré.")
        record.status = "rejected"
        record.decision_by = approver
        record.rejection_reason = reason
        return record


approval_manager = ApprovalManager()
