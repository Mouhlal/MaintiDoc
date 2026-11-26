"""
permissions.py — Contrôle d'accès aux documents et permissions MCP
===================================================================

Centralise la vérification des rôles RBAC :
  - technician : lecture documents autorisés, consultation MCP, préparation tickets
  - supervisor : + approbation / rejet de tickets
  - admin      : tous droits
"""

from __future__ import annotations

from typing import Iterable


ROLE_HIERARCHY: dict[str, int] = {
    "technician": 1,
    "supervisor": 2,
    "admin": 3,
}

MCP_TOOL_PERMISSIONS: dict[str, set[str]] = {
    "get_equipment": {"technician", "supervisor", "admin"},
    "get_history": {"technician", "supervisor", "admin"},
    "get_open_tickets": {"technician", "supervisor", "admin"},
    "create_ticket": {"technician", "supervisor", "admin"},
    "approve_ticket": {"supervisor", "admin"},
}


def has_role(user_role: str, required_roles: Iterable[str]) -> bool:
    """Vérifie si le rôle utilisateur figure parmi les rôles autorisés."""
    return user_role in set(required_roles)


def check_document_access(user_role: str, document_permissions: list[str] | None) -> bool:
    """
    Vérifie si l'utilisateur a le droit d'accéder à un document.
    Les administrateurs ont toujours accès.
    """
    if user_role == "admin":
        return True
    if not document_permissions:
        return True
    return user_role in document_permissions


def check_mcp_permission(user_role: str, tool_name: str, approval_token: str | None = None) -> None:
    """
    Contrôle les permissions côté serveur avant chaque appel d'outil MCP.
    Lève PermissionError si l'accès est refusé.
    """
    allowed_roles = MCP_TOOL_PERMISSIONS.get(tool_name)
    if allowed_roles is None:
        raise PermissionError(f"Outil MCP inconnu : '{tool_name}'")
    if user_role not in allowed_roles:
        raise PermissionError(
            f"Rôle '{user_role}' non autorisé pour l'outil MCP '{tool_name}'"
        )
    if tool_name == "create_ticket" and not (approval_token and approval_token.strip()):
        raise PermissionError(
            "Création de ticket refusée : un token d'approbation humaine est obligatoire."
        )
