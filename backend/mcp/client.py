"""
client.py — Client MCP interne pour l'agent LangGraph
=====================================================

Encapsule les appels aux outils MCP avec :
  - contrôle des permissions par rôle
  - limitation du nombre d'appels par session (rate limiting)
  - journalisation structurée sans données sensibles
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from backend.db.database import AsyncSessionLocal
from backend.mcp import tools


class MCPRateLimitError(Exception):
    """Levée lorsque le plafond d'appels aux outils MCP est dépassé."""


class MCPToolsClient:
    """Client MCP utilisé par les nœuds de l'agent LangGraph."""

    MAX_CALLS_PER_SESSION = 10

    def __init__(self, user_role: str = "technician", user_id: str | None = None) -> None:
        self.user_role = user_role
        self.user_id = user_id
        self._call_count: dict[str, int] = {}

    def _track_call(self, tool_name: str) -> None:
        total = sum(self._call_count.values())
        if total >= self.MAX_CALLS_PER_SESSION:
            raise MCPRateLimitError(
                f"Limite de {self.MAX_CALLS_PER_SESSION} appels MCP atteinte pour cette session."
            )
        self._call_count[tool_name] = self._call_count.get(tool_name, 0) + 1
        logger.info(f"[MCP Client] Appel '{tool_name}' (#{self._call_count[tool_name]})")

    async def get_equipment(self, equipment_id: str) -> dict[str, Any]:
        self._track_call("get_equipment")
        async with AsyncSessionLocal() as session:
            return await tools.get_equipment_info(
                equipment_id=equipment_id,
                db=session,
                user_role=self.user_role,
            )

    async def get_history(self, equipment_id: str, limit: int = 10) -> list[dict[str, Any]]:
        self._track_call("get_history")
        async with AsyncSessionLocal() as session:
            return await tools.get_equipment_history(
                equipment_id=equipment_id,
                db=session,
                limit=limit,
                user_role=self.user_role,
            )

    async def get_open_tickets(self, equipment_id: str) -> list[dict[str, Any]]:
        self._track_call("get_open_tickets")
        async with AsyncSessionLocal() as session:
            return await tools.get_open_tickets(
                equipment_id=equipment_id,
                db=session,
                user_role=self.user_role,
            )

    async def create_ticket(
        self,
        equipment_id: str,
        title: str,
        description: str,
        priority: str,
        approval_token: str,
        source_chunks: list[str],
        agent_reasoning: str,
    ) -> dict[str, Any]:
        self._track_call("create_ticket")
        async with AsyncSessionLocal() as session:
            return await tools.create_maintenance_ticket(
                equipment_id=equipment_id,
                title=title,
                description=description,
                priority=priority,
                requested_by_id=self.user_id,
                approval_token=approval_token,
                source_chunks=source_chunks,
                agent_reasoning=agent_reasoning,
                db=session,
                user_role=self.user_role,
            )
