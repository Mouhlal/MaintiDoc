"""
dependencies.py — Injection de dépendances FastAPI pour MaintiDoc
==================================================================
"""

from __future__ import annotations

from backend.config import Settings, get_settings
from backend.db.database import get_db
from backend.security.auth import get_current_user, require_role

__all__ = [
    "Settings",
    "get_settings",
    "get_db",
    "get_current_user",
    "require_role",
]
