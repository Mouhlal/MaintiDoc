"""
permissions.py — Contrôle d'accès aux documents (couche Security)
=================================================================

Ré-exporte et complète les contrôles RBAC côté API FastAPI et RAG.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from backend.mcp.permissions import check_document_access, check_mcp_permission, has_role


def filter_results_by_role(results: list[dict[str, Any]], user_role: str) -> list[dict[str, Any]]:
    """
    Filtre les chunks retournés par Qdrant selon les permissions du document.
    """
    if user_role == "admin":
        return results

    allowed: list[dict[str, Any]] = []
    for res in results:
        payload = res.get("payload", {})
        perms = payload.get("permissions")
        if check_document_access(user_role, perms):
            allowed.append(res)
    return allowed


def ensure_document_access(user_role: str, document_permissions: list[str] | None) -> None:
    """Lève HTTP 403 si l'utilisateur n'a pas accès au document."""
    if not check_document_access(user_role, document_permissions):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'avez pas les permissions nécessaires pour consulter ce document.",
        )


__all__ = [
    "check_document_access",
    "check_mcp_permission",
    "ensure_document_access",
    "filter_results_by_role",
    "has_role",
]
