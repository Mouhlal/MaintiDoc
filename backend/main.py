"""
main.py — Point d'entrée de l'API FastAPI MaintiDoc
=====================================================

Ce fichier est le "chef d'orchestre" du backend.
Il crée l'application FastAPI, configure le middleware,
enregistre les routes et gère le cycle de vie (startup/shutdown).

Lancement :
    uvicorn backend.main:app --reload --port 8000
    # ou
    make api

Documentation auto-générée :
    http://localhost:8000/docs    ← Swagger UI (interactive)
    http://localhost:8000/redoc  ← ReDoc (lisible)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings, validate_config

# Configuration du logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("maintidoc")


# =============================================================================
# CYCLE DE VIE : STARTUP & SHUTDOWN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application FastAPI.

    Le code AVANT `yield` s'exécute au démarrage (startup).
    Le code APRÈS `yield` s'exécute à l'arrêt (shutdown).

    C'est ici qu'on initialise et ferme les connexions aux services.
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info("=" * 55)
    logger.info(f"🚀 Démarrage de {settings.app_name} v{settings.app_version}")
    logger.info(f"   Environnement : {settings.app_env}")
    logger.info("=" * 55)

    # 1. Valider la configuration
    try:
        validate_config()
        logger.info("✅ Configuration validée")
    except ValueError as e:
        logger.error(f"❌ Configuration invalide : {e}")
        raise

    # 2. Vérification Qdrant (non bloquante en dev)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, timeout=5, check_compatibility=False)
        collections = [c.name for c in client.get_collections().collections]
        logger.info(f"✅ Qdrant connecté — Collections : {collections}")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant non disponible : {e} (lance 'make up')")

    logger.info(f"📡 API disponible sur : http://127.0.0.1:8001")
    logger.info(f"📚 Documentation     : http://127.0.0.1:8001/docs")

    yield  # ← L'application tourne ici

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("👋 Arrêt de MaintiDoc...")


# =============================================================================
# CRÉATION DE L'APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## MaintiDoc — Assistant documentaire pour techniciens de maintenance

### Fonctionnalités
- 📄 **Ingestion** de manuels PDF et documentation technique
- 🔍 **RAG textuel** : recherche sémantique dans les documents
- 🖼️ **RAG multimodal** : analyse d'images et schémas
- 🔧 **Outils MCP** : consultation équipements et tickets
- 🤖 **Agent LangGraph** : orchestration intelligente avec validation humaine

### Notes de sécurité
- Validation humaine **obligatoire** avant création de ticket
- Réponses sourcées avec page et document d'origine
- Refus de réponse si aucun contexte pertinent trouvé
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS — Cross-Origin Resource Sharing
# Permet à Streamlit (port 8501) de faire des requêtes à FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit local
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # Si jamais un frontend React est ajouté
    ] if settings.is_development else [],  # En production : configurer explicitement
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTES — Enregistrement des routeurs
# =============================================================================
# Les routes sont définies dans des fichiers séparés (api/)
# et enregistrées ici avec un préfixe et un tag.
#
# Architecture des URLs :
#   /api/v1/health        ← État des services
#   /api/v1/documents/    ← Gestion des PDF
#   /api/v1/chat/         ← Conversation avec l'agent
#   /api/v1/tickets/      ← Gestion des tickets
#
# Note : les routers seront ajoutés progressivement au fil des phases.
# Pour l'instant, seul le healthcheck est disponible.

from backend.api.health import router as health_router
from backend.api.auth import router as auth_router
from backend.api.documents import router as documents_router
from backend.api.chat import router as chat_router
from backend.api.tickets import router as tickets_router

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health"],
)
app.include_router(
    auth_router,
    prefix="/api/v1/auth",
    tags=["Authentification"],
)
app.include_router(
    documents_router,
    prefix="/api/v1/documents",
    tags=["Documents"],
)
app.include_router(
    chat_router,
    prefix="/api/v1/chat",
    tags=["Chat"],
)
app.include_router(
    tickets_router,
    prefix="/api/v1/tickets",
    tags=["Tickets & Équipements"],
)


# =============================================================================
# ROUTE RACINE
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Route d'accueil de l'API.
    Retourne les informations de base de l'application.
    """
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "docs": "/docs",
        "health": "/api/v1/health",
        "message": "Bienvenue dans MaintiDoc — Assistant documentaire pour techniciens de maintenance",
    }
