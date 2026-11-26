"""
Script de test manuel pour config.py
Exécute : python scripts/test_config.py
"""
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings, validate_config

print("=" * 60)
print("🔧 TEST DE CONFIGURATION MAINTIDOC")
print("=" * 60)

# --- Infos générales ---
print(f"\n📋 Application")
print(f"   Nom         : {settings.app_name}")
print(f"   Version     : {settings.app_version}")
print(f"   Environnement: {settings.app_env}")
print(f"   Dev mode    : {settings.is_development}")
print(f"   Log level   : {settings.log_level}")

# --- PostgreSQL ---
print(f"\n🗄️  PostgreSQL")
print(f"   Host        : {settings.postgres_host}:{settings.postgres_port}")
print(f"   Database    : {settings.postgres_db}")
print(f"   User        : {settings.postgres_user}")
print(f"   URL async   : {settings.database_url}")

# --- Qdrant ---
print(f"\n🔵 Qdrant")
print(f"   URL         : {settings.qdrant_url}")
print(f"   Collection texte  : {settings.qdrant_collection_text}")
print(f"   Collection visuel : {settings.qdrant_collection_visual}")
print(f"   Dim texte   : {settings.embedding_dim_text}")
print(f"   Dim visuel  : {settings.embedding_dim_visual}")

# --- MinIO ---
print(f"\n📦 MinIO")
print(f"   Endpoint    : {settings.minio_endpoint}")
print(f"   Bucket docs : {settings.minio_bucket_documents}")
print(f"   Bucket imgs : {settings.minio_bucket_images}")

# --- LLM ---
print(f"\n🤖 Modèles")
print(f"   LLM         : {settings.llm_model}")
print(f"   VLM         : {settings.vlm_model}")
print(f"   Embeddings  : {settings.embedding_model_text}")
print(f"   CLIP        : {settings.embedding_model_visual}")
print(f"   Reranker    : {settings.reranker_model}")
print(f"   Température : {settings.llm_temperature}")

# --- RAG ---
print(f"\n🔍 Paramètres RAG")
print(f"   top_k           : {settings.rag_top_k}")
print(f"   top_k_reranked  : {settings.rag_top_k_reranked}")
print(f"   chunk_size      : {settings.rag_chunk_size} tokens")
print(f"   chunk_overlap   : {settings.rag_chunk_overlap} tokens")
print(f"   score_threshold : {settings.rag_score_threshold}")

# --- Chemins ---
print(f"\n📁 Chemins")
print(f"   Racine projet : {settings.project_root}")
print(f"   Données       : {settings.data_dir}")
print(f"   PDF bruts     : {settings.raw_dir}")

# --- Validation ---
print(f"\n✅ Validation de la configuration...")
try:
    validate_config()
    print("   Configuration valide !")
except ValueError as e:
    print(f"   ⚠️  Avertissement : {e}")

print("\n" + "=" * 60)
print("✅ config.py fonctionne correctement !")
print("=" * 60)
