-- =============================================================================
-- MaintiDoc — Script d'initialisation PostgreSQL
-- Exécuté automatiquement au PREMIER démarrage du conteneur postgres
-- =============================================================================

-- Création de la base si elle n'existe pas déjà (gérée par POSTGRES_DB)
-- Ce script sert à créer les extensions utiles

-- Extension UUID : génère des identifiants uniques (ex: document_id, ticket_id)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extension pour la recherche full-text (utile pour les métadonnées)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Schéma MLflow : MLflow crée ses propres tables dans le schéma public
-- Rien à faire ici, MLflow s'en charge au démarrage

-- Message de confirmation dans les logs Docker
DO $$
BEGIN
  RAISE NOTICE 'MaintiDoc DB initialisée avec succès.';
END $$;
