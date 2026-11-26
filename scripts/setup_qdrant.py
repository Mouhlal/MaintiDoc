"""
setup_qdrant.py — Initialisation des collections Qdrant
=========================================================

Ce script crée les deux collections vectorielles de MaintiDoc dans Qdrant :

  1. maintidoc_text   → chunks textuels des PDF (BGE-M3, dim=1024)
  2. maintidoc_visual → images extraites des PDF (CLIP, dim=512)

Il est idempotent : si une collection existe déjà, elle n'est pas recréée.

Usage :
    python scripts/setup_qdrant.py
    # ou
    make setup-qdrant
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    VectorParams,
)

from backend.config import settings


def get_client() -> QdrantClient:
    """
    Crée et retourne un client Qdrant connecté.

    QdrantClient se connecte via HTTP REST à l'URL définie dans config.py.
    En local (dev), pas d'API key requise.
    """
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,  # None en développement local
        timeout=10,
    )
    return client


def create_text_collection(client: QdrantClient) -> None:
    """
    Crée la collection pour les embeddings textuels.

    Paramètres techniques expliqués :
    ----------------------------------
    - size=1024        : dimension des vecteurs BGE-M3
    - distance=COSINE  : métrique de similarité.
                         Cosine mesure l'angle entre deux vecteurs.
                         Idéal pour les embeddings textuels (indépendant de la norme).
    - hnsw_config      : paramètres de l'index HNSW (Hierarchical Navigable Small World)
                         C'est l'algorithme de recherche approximative des plus proches voisins.
      - m=16           : nombre de liens par nœud. Plus grand = plus précis mais plus de RAM.
      - ef_construct=100 : taille du beam pendant la construction. Qualité vs temps de build.
    - optimizers_config: contrôle quand Qdrant réorganise les segments sur disque.
      - indexing_threshold=20000 : indexe les vecteurs par batch de 20k (moins d'I/O)
    """
    collection_name = settings.qdrant_collection_text

    # Vérifier si la collection existe déjà
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"  ⏭️  Collection '{collection_name}' existe déjà — ignorée")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim_text,  # 1024 pour BGE-M3
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000,  # Scan exhaustif si moins de 10k vecteurs
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )
    print(f"  ✅ Collection '{collection_name}' créée (dim={settings.embedding_dim_text}, COSINE)")

    # Créer des index sur les champs de métadonnées les plus filtrés
    # Cela accélère les recherches filtrées (ex: "cherche uniquement dans le doc X")
    _create_payload_indexes(client, collection_name)


def create_visual_collection(client: QdrantClient) -> None:
    """
    Crée la collection pour les embeddings visuels (images extraites des PDF).

    Différences par rapport à la collection texte :
    - size=512     : dimension CLIP ViT-B/32 (moins de dimension = moins de RAM)
    - m=8          : index plus léger (moins d'images que de chunks texte)
    """
    collection_name = settings.qdrant_collection_visual

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"  ⏭️  Collection '{collection_name}' existe déjà — ignorée")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim_visual,  # 512 pour CLIP ViT-B/32
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=8,            # Index plus léger pour les images (moins nombreuses)
            ef_construct=100,
        ),
    )
    print(f"  ✅ Collection '{collection_name}' créée (dim={settings.embedding_dim_visual}, COSINE)")
    _create_payload_indexes(client, collection_name)


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Crée des index sur les métadonnées stockées avec chaque vecteur (le "payload").

    Sans index → Qdrant fait un scan complet des métadonnées à chaque filtre.
    Avec index → recherche O(log n) sur ces champs.

    Champs indexés :
    - document_id   : filtrer par document source
    - page_number   : filtrer par page
    - equipment_type: filtrer par type d'équipement
    - chunk_type    : filtrer par type (text, table, figure, caption)
    """
    indexes = {
        "document_id": PayloadSchemaType.KEYWORD,    # Valeur exacte (ex: "DOC-001")
        "page_number": PayloadSchemaType.INTEGER,    # Comparaison numérique (ex: page > 5)
        "equipment_type": PayloadSchemaType.KEYWORD, # Valeur exacte (ex: "pompe_centrifuge")
        "chunk_type": PayloadSchemaType.KEYWORD,     # Valeur exacte (ex: "text", "table")
    }

    for field_name, field_type in indexes.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type,
        )
    print(f"     → Index créés sur : {', '.join(indexes.keys())}")


def verify_collections(client: QdrantClient) -> None:
    """
    Vérifie que les collections sont bien créées et affiche leurs infos.
    """
    print("\n📊 Vérification des collections :")
    collections = client.get_collections().collections

    for col in collections:
        info = client.get_collection(col.name)
        vec_config = info.config.params.vectors
        print(f"  🔵 {col.name}")
        print(f"     Vecteurs : dim={vec_config.size}, distance={vec_config.distance}")
        print(f"     Points   : {info.points_count}")
        print(f"     Status   : {info.status}")


def main() -> None:
    print("=" * 55)
    print("🔵 INITIALISATION QDRANT — MaintiDoc")
    print("=" * 55)
    print(f"\n📡 Connexion à : {settings.qdrant_url}")

    # Connexion
    try:
        client = get_client()
        # Test de connexion : récupérer la liste des collections
        client.get_collections()
        print("  ✅ Connexion réussie\n")
    except Exception as e:
        print(f"  ❌ Impossible de se connecter à Qdrant : {e}")
        print(f"     → Lance Docker : docker compose up -d qdrant")
        sys.exit(1)

    # Création des collections
    print("🏗️  Création des collections :")
    create_text_collection(client)
    create_visual_collection(client)

    # Vérification
    verify_collections(client)

    print("\n" + "=" * 55)
    print("✅ Qdrant initialisé avec succès !")
    print(f"   Dashboard : {settings.qdrant_url}/dashboard")
    print("=" * 55)


if __name__ == "__main__":
    main()
