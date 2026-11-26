"""
setup_db.py — Initialisation des tables PostgreSQL
====================================================

Ce script crée toutes les tables relationnelles de MaintiDoc.

Tables créées :
  - users          → utilisateurs et leurs rôles
  - documents      → métadonnées des PDF ingérés
  - chunks         → métadonnées des chunks (lien vers Qdrant)
  - equipments     → équipements de maintenance (fictifs pour la démo)
  - interventions  → historique des interventions
  - tickets        → tickets de maintenance créés par l'agent

Ce script utilise SQLAlchemy Core (pas l'ORM) pour être indépendant
des modèles Python définis dans backend/db/models.py.

Usage :
    python scripts/setup_db.py
    # ou
    make setup-db
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from backend.config import settings


# =============================================================================
# SCHÉMA SQL — Définition de toutes les tables
# =============================================================================

# Chaque CREATE TABLE est commenté pour expliquer chaque colonne.
# Conventions :
#   - UUID comme clé primaire (pas d'entier auto-incrémenté → pas de collision)
#   - created_at / updated_at sur chaque table pour l'audit
#   - VARCHAR avec longueur limitée pour éviter les abus

SCHEMA_SQL = """
-- ============================================================
-- TABLE : users
-- Rôle : stocker les techniciens qui utilisent MaintiDoc
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username        VARCHAR(100) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,

    -- Rôle détermine les permissions :
    --   'technician'  → lecture documents + création tickets
    --   'supervisor'  → + validation tickets
    --   'admin'       → tout
    role            VARCHAR(50)  NOT NULL DEFAULT 'technician',
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);


-- ============================================================
-- TABLE : documents
-- Rôle : métadonnées des PDF ingérés dans MaintiDoc
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename        VARCHAR(500) NOT NULL,
    title           VARCHAR(500),

    -- Type de document technique
    -- Ex: 'manuel_utilisateur', 'procedure_maintenance', 'schema_electrique'
    doc_type        VARCHAR(100),

    -- Type d'équipement concerné
    -- Ex: 'pompe_centrifuge', 'compresseur', 'moteur_electrique'
    equipment_type  VARCHAR(100),

    -- Version du document (ex: "v2.3", "REV-05")
    version         VARCHAR(50),

    total_pages     INTEGER,
    total_chunks    INTEGER DEFAULT 0,

    -- Chemin MinIO où le PDF original est stocké
    -- Format : 'maintidoc-documents/DOC-001/manuel.pdf'
    storage_path    VARCHAR(1000),

    -- Statut du traitement du document
    -- 'pending' → en attente d'ingestion
    -- 'processing' → ingestion en cours
    -- 'indexed' → prêt pour la recherche
    -- 'error' → échec de l'ingestion
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message   TEXT,

    -- Qui a importé ce document
    uploaded_by     UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Permissions : liste des rôles qui peuvent accéder à ce document
    -- Ex: '["technician", "supervisor"]'
    permissions     JSONB NOT NULL DEFAULT '["technician", "supervisor", "admin"]'::jsonb,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_equipment_type ON documents(equipment_type);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);


-- ============================================================
-- TABLE : chunks
-- Rôle : métadonnées des chunks indexés dans Qdrant
--
-- Chaque chunk dans Qdrant a un vecteur + un ID.
-- Cette table stocke les métadonnées textuelles complètes
-- pour construire les citations et afficher les sources.
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- ID du document parent
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- ID du vecteur dans Qdrant (même valeur que l'ID dans Qdrant)
    qdrant_id       UUID NOT NULL UNIQUE,

    -- Numéro de page source dans le PDF
    page_number     INTEGER NOT NULL,

    -- Type de contenu du chunk
    -- 'text'    → paragraphe de texte
    -- 'table'   → tableau (converti en texte structuré)
    -- 'figure'  → description d'une figure
    -- 'caption' → légende d'une figure
    chunk_type      VARCHAR(50) NOT NULL DEFAULT 'text',

    -- Contenu textuel complet du chunk
    content         TEXT NOT NULL,

    -- Numéro d'ordre du chunk dans le document (pour reconstruire le contexte)
    chunk_index     INTEGER NOT NULL,

    -- Référence à l'image dans MinIO si chunk_type = 'figure'
    image_path      VARCHAR(1000),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_id);


-- ============================================================
-- TABLE : equipments
-- Rôle : catalogue des équipements de maintenance
--        Données fictives pour la démo
-- ============================================================
CREATE TABLE IF NOT EXISTS equipments (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identifiant métier (ex: "EQ-POMPE-001")
    equipment_id    VARCHAR(100) UNIQUE NOT NULL,

    name            VARCHAR(500) NOT NULL,
    equipment_type  VARCHAR(100) NOT NULL,  -- Ex: 'pompe_centrifuge'

    -- Localisation physique
    location        VARCHAR(500),           -- Ex: "Atelier B, Ligne 3"
    building        VARCHAR(100),

    -- Infos constructeur
    manufacturer    VARCHAR(255),
    model           VARCHAR(255),
    serial_number   VARCHAR(255),

    -- Statut opérationnel
    -- 'operational', 'maintenance', 'hors_service', 'inspection'
    status          VARCHAR(100) NOT NULL DEFAULT 'operational',

    -- Date de la dernière maintenance
    last_maintenance_date DATE,

    -- Fréquence de maintenance recommandée en jours
    maintenance_interval_days INTEGER DEFAULT 90,

    -- Métadonnées supplémentaires (flexible)
    metadata        JSONB DEFAULT '{}'::jsonb,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_equipments_equipment_id ON equipments(equipment_id);
CREATE INDEX IF NOT EXISTS idx_equipments_status ON equipments(status);
CREATE INDEX IF NOT EXISTS idx_equipments_type ON equipments(equipment_type);


-- ============================================================
-- TABLE : interventions
-- Rôle : historique des interventions de maintenance
-- ============================================================
CREATE TABLE IF NOT EXISTS interventions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id    UUID NOT NULL REFERENCES equipments(id) ON DELETE CASCADE,

    -- Type d'intervention
    -- 'preventive', 'corrective', 'inspection', 'remplacement'
    intervention_type VARCHAR(100) NOT NULL,

    title           VARCHAR(500) NOT NULL,
    description     TEXT,

    -- Technicien ayant réalisé l'intervention
    technician_id   UUID REFERENCES users(id) ON DELETE SET NULL,
    technician_name VARCHAR(255),  -- Dénormalisé pour l'historique

    -- Dates
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    duration_minutes INTEGER,

    -- Résultat
    -- 'completed', 'partial', 'failed', 'deferred'
    result          VARCHAR(100) DEFAULT 'completed',

    -- Pièces remplacées, coût, notes
    parts_used      JSONB DEFAULT '[]'::jsonb,
    notes           TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interventions_equipment_id ON interventions(equipment_id);
CREATE INDEX IF NOT EXISTS idx_interventions_started_at ON interventions(started_at DESC);


-- ============================================================
-- TABLE : tickets
-- Rôle : tickets de maintenance créés par l'agent MaintiDoc
--        Toujours après validation humaine explicite.
-- ============================================================
CREATE TABLE IF NOT EXISTS tickets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id    UUID NOT NULL REFERENCES equipments(id) ON DELETE CASCADE,

    -- Identifiant métier lisible (ex: "TKT-2024-001")
    ticket_number   VARCHAR(100) UNIQUE NOT NULL,

    title           VARCHAR(500) NOT NULL,
    description     TEXT NOT NULL,

    -- Priorité : 'low', 'medium', 'high', 'critical'
    priority        VARCHAR(50) NOT NULL DEFAULT 'medium',

    -- Statut du ticket
    -- 'draft'      → préparé par l'agent, pas encore validé
    -- 'pending'    → en attente de validation humaine
    -- 'approved'   → validé par un superviseur
    -- 'rejected'   → rejeté
    -- 'in_progress'→ en cours de traitement
    -- 'completed'  → terminé
    -- 'cancelled'  → annulé
    status          VARCHAR(100) NOT NULL DEFAULT 'draft',

    -- Qui a demandé la création (technicien via l'agent)
    requested_by    UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Token d'idempotence : empêche la création du même ticket deux fois
    -- Généré par l'agent, vérifié avant insertion
    idempotency_key VARCHAR(255) UNIQUE,

    -- Validation humaine
    approved_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    approved_at     TIMESTAMPTZ,
    rejection_reason TEXT,

    -- Contexte : sources documentaires qui ont justifié la création du ticket
    source_chunks   JSONB DEFAULT '[]'::jsonb,  -- Liste des chunk_ids utilisés

    -- Contenu généré par l'agent (pour audit)
    agent_reasoning TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickets_equipment_id ON tickets(equipment_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_requested_by ON tickets(requested_by);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at DESC);
"""


# =============================================================================
# DONNÉES DE DÉMONSTRATION (fictives)
# =============================================================================

DEMO_DATA_SQL = """
-- Utilisateur admin de démonstration
INSERT INTO users (id, username, email, hashed_password, role)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'admin',
    'admin@maintidoc.local',
    -- Hash bcrypt de 'admin_password_demo' (NE PAS utiliser en production)
    '$2b$12$demo_hash_placeholder_not_real',
    'admin'
) ON CONFLICT (username) DO NOTHING;

-- Technicien de démonstration
INSERT INTO users (id, username, email, hashed_password, role)
VALUES (
    '00000000-0000-0000-0000-000000000002',
    'tech_martin',
    'martin@maintidoc.local',
    '$2b$12$demo_hash_placeholder_not_real',
    'technician'
) ON CONFLICT (username) DO NOTHING;

-- Équipements fictifs de démonstration
INSERT INTO equipments (equipment_id, name, equipment_type, location, manufacturer, model, status, last_maintenance_date)
VALUES
    ('EQ-POMPE-001', 'Pompe centrifuge P-101', 'pompe_centrifuge',
     'Atelier A - Ligne 1', 'Grundfos', 'CM5-6', 'operational', '2024-09-15'),

    ('EQ-POMPE-002', 'Pompe de transfert P-202', 'pompe_centrifuge',
     'Atelier A - Ligne 2', 'Grundfos', 'CM10-6', 'maintenance', '2024-11-01'),

    ('EQ-COMPRESSEUR-001', 'Compresseur air C-301', 'compresseur_air',
     'Salle machines B', 'Atlas Copco', 'GA15', 'operational', '2024-10-20'),

    ('EQ-MOTEUR-001', 'Moteur électrique M-401', 'moteur_electrique',
     'Atelier C - Poste 4', 'ABB', 'M2BAX 90', 'operational', '2024-08-30'),

    ('EQ-VANNE-001', 'Vanne pneumatique V-501', 'vanne',
     'Tuyauterie principale', 'Emerson', 'Fisher 657', 'inspection', '2024-11-10')
ON CONFLICT (equipment_id) DO NOTHING;

-- Interventions fictives pour la pompe P-101
INSERT INTO interventions (equipment_id, intervention_type, title, description, technician_name, started_at, completed_at, duration_minutes, result, notes)
SELECT
    e.id,
    'preventive',
    'Maintenance préventive trimestrielle - P-101',
    'Vérification joints mécaniques, lubrification roulements, contrôle débitmètre. Usure normale détectée sur joint d''étanchéité.',
    'M. Martin',
    '2024-09-15 08:00:00+00',
    '2024-09-15 11:30:00+00',
    210,
    'completed',
    'Joint mécanique à surveiller. Prévoir remplacement dans 3 mois.'
FROM equipments e WHERE e.equipment_id = 'EQ-POMPE-001'
ON CONFLICT DO NOTHING;

INSERT INTO interventions (equipment_id, intervention_type, title, description, technician_name, started_at, completed_at, duration_minutes, result, notes)
SELECT
    e.id,
    'corrective',
    'Remplacement joint d''étanchéité - P-101',
    'Suite à détection de fuite légère. Remplacement joint mécanique type Burgmann MG1-28.',
    'M. Martin',
    '2024-11-20 14:00:00+00',
    '2024-11-20 16:00:00+00',
    120,
    'completed',
    'Fuite stoppée. Équipement opérationnel. Prochain contrôle dans 90 jours.'
FROM equipments e WHERE e.equipment_id = 'EQ-POMPE-001'
ON CONFLICT DO NOTHING;
"""


def get_connection():
    """
    Crée une connexion psycopg2 à PostgreSQL.
    Utilise les paramètres de settings (lus depuis .env).
    """
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    return conn


def main() -> None:
    print("=" * 55)
    print("🗄️  INITIALISATION POSTGRESQL — MaintiDoc")
    print("=" * 55)
    print(f"\n📡 Connexion à : {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")

    try:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        print("  ✅ Connexion réussie\n")
    except Exception as e:
        print(f"  ❌ Impossible de se connecter à PostgreSQL : {e}")
        print(f"     → Lance Docker : docker compose up -d postgres")
        sys.exit(1)

    # Création des tables
    print("🏗️  Création des tables...")
    try:
        cur.execute(SCHEMA_SQL)
        print("  ✅ Tables créées (ou déjà existantes)")
    except Exception as e:
        print(f"  ❌ Erreur création tables : {e}")
        sys.exit(1)

    # Données de démonstration
    print("\n📦 Insertion des données de démonstration...")
    try:
        cur.execute(DEMO_DATA_SQL)
        print("  ✅ Données de démonstration insérées")
    except Exception as e:
        print(f"  ⚠️  Erreur données démo (peut-être déjà insérées) : {e}")

    # Vérification
    print("\n📊 Vérification des tables :")
    tables = ["users", "documents", "chunks", "equipments", "interventions", "tickets"]
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  📋 {table:<20} → {count} ligne(s)")

    cur.close()
    conn.close()

    print("\n" + "=" * 55)
    print("✅ PostgreSQL initialisé avec succès !")
    print("=" * 55)


if __name__ == "__main__":
    main()
