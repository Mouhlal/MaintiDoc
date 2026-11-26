"""
config.py — Configuration centralisée de MaintiDoc
====================================================

Ce module est le point unique de configuration de toute l'application.
Il lit les variables d'environnement depuis le fichier .env et les expose
via un objet `settings` importable partout.

Usage dans les autres modules :
    from backend.config import settings

    qdrant_url = settings.qdrant_url      # "http://localhost:6333"
    llm_model  = settings.llm_model       # "llama3.2:3b"
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# CLASSE DE CONFIGURATION PRINCIPALE
# =============================================================================

class Settings(BaseSettings):
    """
    Paramètres de configuration de MaintiDoc.

    Pydantic lit automatiquement les variables depuis .env.
    L'ordre de priorité est :
      1. Variables d'environnement du système (export VAR=value)
      2. Fichier .env
      3. Valeurs par défaut définies ici
    """

    # -------------------------------------------------------------------------
    # Configuration Pydantic : comment lire le .env
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        # Cherche le .env en remontant depuis ce fichier jusqu'à la racine du projet
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        # Si une variable est dans le .env ET dans l'environnement système,
        # l'environnement système a la priorité (utile en production Docker)
        case_sensitive=False,
        # Ne pas planter si des champs supplémentaires sont dans le .env
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # APPLICATION
    # -------------------------------------------------------------------------
    app_name: str = Field(default="MaintiDoc", description="Nom de l'application")
    app_version: str = Field(default="0.1.0", description="Version de l'application")
    app_env: str = Field(default="development", description="Environnement : development | production")
    app_secret_key: str = Field(default="change_this_in_production", description="Clé secrète JWT")
    log_level: str = Field(default="INFO", description="Niveau de log : DEBUG | INFO | WARNING | ERROR")

    @computed_field  # Propriété calculée automatiquement, pas lue depuis .env
    @property
    def is_development(self) -> bool:
        """Retourne True si on est en mode développement."""
        return self.app_env.lower() == "development"

    # -------------------------------------------------------------------------
    # POSTGRESQL — Base de données relationnelle
    # -------------------------------------------------------------------------
    postgres_user: str = Field(default="maintidoc")
    postgres_password: str = Field(default="maintidoc_secret_2024")
    postgres_db: str = Field(default="maintidoc_db")
    postgres_host: str = Field(default="localhost", description="'localhost' en dev, 'postgres' dans Docker")
    postgres_port: int = Field(default=5432)

    @computed_field
    @property
    def database_url(self) -> str:
        """
        Construit l'URL de connexion PostgreSQL au format SQLAlchemy.
        Format : postgresql+asyncpg://user:password@host:port/dbname
        On utilise asyncpg (driver async) pour ne pas bloquer FastAPI.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """
        URL synchrone pour les migrations Alembic (qui ne supporte pas asyncpg).
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # QDRANT — Base de données vectorielle
    # -------------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str | None = Field(default=None, description="Clé API (None si pas d'auth locale)")

    @computed_field
    @property
    def qdrant_url(self) -> str:
        """URL complète de Qdrant."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # Noms des collections Qdrant (un seul endroit à modifier si on renomme)
    qdrant_collection_text: str = Field(
        default="maintidoc_text",
        description="Collection pour les embeddings textuels (chunks de PDF)"
    )
    qdrant_collection_visual: str = Field(
        default="maintidoc_visual",
        description="Collection pour les embeddings visuels (images extraites)"
    )

    # Dimension des vecteurs (doit correspondre au modèle d'embeddings)
    # BGE-M3 produit des vecteurs de dimension 1024
    embedding_dim_text: int = Field(default=1024, description="Dimension BGE-M3")
    # CLIP ViT-B/32 produit des vecteurs de dimension 512
    embedding_dim_visual: int = Field(default=512, description="Dimension CLIP")

    # -------------------------------------------------------------------------
    # MINIO — Stockage de fichiers (compatible S3)
    # -------------------------------------------------------------------------
    minio_root_user: str = Field(default="maintidoc_admin")
    minio_root_password: str = Field(default="maintidoc_minio_2024")
    minio_host: str = Field(default="localhost")
    minio_port: int = Field(default=9000)
    minio_secure: bool = Field(default=False, description="True si HTTPS (production)")

    @computed_field
    @property
    def minio_endpoint(self) -> str:
        """Endpoint MinIO pour le client Python."""
        return f"{self.minio_host}:{self.minio_port}"

    # Noms des buckets MinIO
    minio_bucket_documents: str = Field(
        default="maintidoc-documents",
        description="Bucket pour les PDF originaux"
    )
    minio_bucket_images: str = Field(
        default="maintidoc-images",
        description="Bucket pour les images extraites des PDF"
    )

    # -------------------------------------------------------------------------
    # OLLAMA — LLM et VLM locaux
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL du serveur Ollama"
    )
    llm_model: str = Field(
        default="llama3.2:3b",
        description="Modèle LLM principal. llama3.2:3b ≈ 2Go RAM (compatible M2 8Go)"
    )
    vlm_model: str = Field(
        default="llava:7b-v1.6-mistral-q4_0",
        description="Modèle VLM pour l'analyse d'images. Chargé à la demande."
    )
    llm_temperature: float = Field(
        default=0.1,
        description="Température LLM. 0.1 = réponses déterministes (bien pour le RAG)"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Nombre max de tokens en sortie du LLM"
    )

    # -------------------------------------------------------------------------
    # MODÈLES D'EMBEDDINGS
    # -------------------------------------------------------------------------
    embedding_model_text: str = Field(
        default="BAAI/bge-m3",
        description="Modèle d'embeddings textuels multilingues. ≈ 1.1 Go RAM"
    )
    embedding_model_visual: str = Field(
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        description="Modèle d'embeddings visuels. ≈ 400 Mo RAM"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Modèle de reranking. ≈ 90 Mo RAM"
    )

    # -------------------------------------------------------------------------
    # MLFLOW — Tracking des expériences
    # -------------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI du serveur MLflow"
    )
    mlflow_experiment_name: str = Field(
        default="maintidoc-rag-evaluation",
        description="Nom de l'expérience MLflow"
    )

    # -------------------------------------------------------------------------
    # PARAMÈTRES RAG
    # -------------------------------------------------------------------------
    rag_top_k: int = Field(
        default=10,
        description="Nombre de documents récupérés avant reranking"
    )
    rag_top_k_reranked: int = Field(
        default=5,
        description="Nombre de documents conservés après reranking"
    )
    rag_chunk_size: int = Field(
        default=512,
        description="Taille des chunks en tokens"
    )
    rag_chunk_overlap: int = Field(
        default=64,
        description="Chevauchement entre chunks consécutifs (en tokens)"
    )
    rag_score_threshold: float = Field(
        default=0.5,
        description="Score minimum de similarité pour retenir un résultat"
    )

    # -------------------------------------------------------------------------
    # SÉCURITÉ ET LIMITES
    # -------------------------------------------------------------------------
    max_file_size_mb: int = Field(
        default=50,
        description="Taille maximale d'un PDF en Mo"
    )
    max_pages_per_document: int = Field(
        default=500,
        description="Nombre maximal de pages par document"
    )
    rate_limit_requests_per_minute: int = Field(
        default=30,
        description="Limite d'appels API par minute par utilisateur"
    )

    # -------------------------------------------------------------------------
    # CHEMINS LOCAUX
    # -------------------------------------------------------------------------
    @computed_field
    @property
    def project_root(self) -> Path:
        """Chemin absolu vers la racine du projet (dossier contenant backend/)."""
        return Path(__file__).parent.parent

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Dossier des données locales."""
        return self.project_root / "data"

    @computed_field
    @property
    def raw_dir(self) -> Path:
        """Dossier des PDF d'entrée."""
        return self.data_dir / "raw"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        """Dossier des données traitées."""
        return self.data_dir / "processed"


# =============================================================================
# INSTANCE SINGLETON — Point d'accès global
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retourne l'instance unique de Settings.

    @lru_cache(maxsize=1) garantit que Settings n'est instancié qu'UNE SEULE FOIS
    pendant toute la durée de vie de l'application.
    Sans ce cache, chaque import rechargerait et re-validerait le .env.

    Usage :
        from backend.config import settings   ← pour usage direct
        from backend.config import get_settings  ← pour injection FastAPI
    """
    return Settings()


# Instance globale pré-construite pour les imports directs
# C'est l'objet qu'on importera dans 99% des cas
settings: Settings = get_settings()


# =============================================================================
# VÉRIFICATION AU DÉMARRAGE
# =============================================================================

def validate_config() -> None:
    """
    Vérifie que la configuration est cohérente.
    Appelée au démarrage de l'application (dans main.py).

    Lève une ValueError si une configuration critique est invalide.
    """
    s = settings

    # Vérifier que la clé secrète a été changée en production
    if not s.is_development and s.app_secret_key == "change_this_in_production":
        raise ValueError(
            "APP_SECRET_KEY doit être défini en production. "
            "Génère une clé avec : python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    # Vérifier la cohérence chunk_size / chunk_overlap
    if s.rag_chunk_overlap >= s.rag_chunk_size:
        raise ValueError(
            f"rag_chunk_overlap ({s.rag_chunk_overlap}) doit être "
            f"inférieur à rag_chunk_size ({s.rag_chunk_size})"
        )

    # Vérifier que top_k_reranked <= top_k
    if s.rag_top_k_reranked > s.rag_top_k:
        raise ValueError(
            f"rag_top_k_reranked ({s.rag_top_k_reranked}) doit être "
            f"<= rag_top_k ({s.rag_top_k})"
        )
