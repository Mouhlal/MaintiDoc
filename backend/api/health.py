"""
health.py — Routes de vérification de l'état des services
===========================================================

Ces endpoints permettent de vérifier que tous les services
(Qdrant, PostgreSQL, MinIO, Ollama) sont disponibles.

Endpoints :
    GET /api/v1/health          → Statut global rapide
    GET /api/v1/health/detailed → Statut détaillé par service
"""

import asyncio
import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings

router = APIRouter()


# =============================================================================
# SCHÉMAS DE RÉPONSE (Pydantic)
# =============================================================================

class ServiceStatus(BaseModel):
    """Statut d'un service individuel."""
    name: str
    status: Literal["ok", "error", "unavailable"]
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Réponse complète du healthcheck."""
    status: Literal["ok", "degraded", "error"]
    app: str
    version: str
    environment: str
    services: list[ServiceStatus]


# =============================================================================
# FONCTIONS DE VÉRIFICATION
# =============================================================================

async def check_qdrant() -> ServiceStatus:
    """Vérifie la connexion à Qdrant."""
    start = time.monotonic()
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, timeout=3, check_compatibility=False)
        collections = client.get_collections().collections
        latency = round((time.monotonic() - start) * 1000, 2)
        return ServiceStatus(
            name="qdrant",
            status="ok",
            latency_ms=latency,
            detail=f"{len(collections)} collection(s)",
        )
    except Exception as e:
        return ServiceStatus(name="qdrant", status="error", detail=str(e))


async def check_postgres() -> ServiceStatus:
    """Vérifie la connexion à PostgreSQL."""
    start = time.monotonic()
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=3,
        )
        cur = conn.cursor()
        cur.execute("SELECT version()")
        row = cur.fetchone()[0]
        version = " ".join(row.split(" ")[:2]) if row else "PostgreSQL OK"
        conn.close()
        latency = round((time.monotonic() - start) * 1000, 2)
        return ServiceStatus(name="postgres", status="ok", latency_ms=latency, detail=version)
    except Exception as e:
        return ServiceStatus(name="postgres", status="error", detail=str(e))


async def check_minio() -> ServiceStatus:
    """Vérifie la connexion à MinIO."""
    start = time.monotonic()
    try:
        import urllib3
        from minio import Minio

        def _check():
            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_root_user,
                secret_key=settings.minio_root_password,
                secure=settings.minio_secure,
                http_client=urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=15.0), retries=1),
            )
            return client.list_buckets()

        buckets = await asyncio.to_thread(_check)
        latency = round((time.monotonic() - start) * 1000, 2)
        return ServiceStatus(
            name="minio",
            status="ok",
            latency_ms=latency,
            detail=f"{len(buckets)} bucket(s)",
        )
    except Exception as e:
        return ServiceStatus(name="minio", status="error", detail=str(e))


async def check_ollama() -> ServiceStatus:
    """Vérifie la connexion à Ollama."""
    start = time.monotonic()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
        latency = round((time.monotonic() - start) * 1000, 2)
        return ServiceStatus(
            name="ollama",
            status="ok",
            latency_ms=latency,
            detail=f"Modèles disponibles : {', '.join(models) if models else 'aucun'}",
        )
    except Exception as e:
        return ServiceStatus(name="ollama", status="unavailable", detail=str(e))


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/health", response_model=dict, summary="Statut rapide")
async def health_simple():
    """
    Vérification rapide de l'état de l'API.
    Utilisé par les load balancers et Docker healthchecks.
    Ne vérifie pas les services externes (rapide).
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@router.get(
    "/health/detailed",
    response_model=HealthResponse,
    summary="Statut détaillé de tous les services",
)
async def health_detailed():
    """
    Vérifie l'état de chaque service externe.
    Plus lent (fait des appels réseau réels).
    Utilisé pour le monitoring et le debugging.
    """
    # Vérification parallèle de tous les services
    services = list(
        await asyncio.gather(
            check_qdrant(),
            check_postgres(),
            check_minio(),
            check_ollama(),
        )
    )

    # Statut global
    statuses = [s.status for s in services]
    if all(s == "ok" for s in statuses):
        global_status = "ok"
    elif any(s == "error" for s in statuses):
        global_status = "degraded"
    else:
        global_status = "degraded"

    return HealthResponse(
        status=global_status,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        services=services,
    )
