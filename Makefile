# =============================================================================
# MaintiDoc — Makefile
# =============================================================================
# Raccourcis pour les commandes les plus fréquentes du projet.
# Usage : make <commande>
# Exemples:
#   make up          → Démarrer l'infrastructure Docker
#   make test        → Lancer tous les tests
#   make ingest PDF=data/raw/manuel.pdf  → Ingérer un PDF
# =============================================================================

# Variables
PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest

# Couleurs pour les messages
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
RESET  := \033[0m

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# AIDE
# -----------------------------------------------------------------------------
.PHONY: help
help: ## Affiche cette aide
	@echo ""
	@echo "$(BLUE)╔══════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║        MaintiDoc — Commandes         ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════╝$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# ENVIRONNEMENT PYTHON
# -----------------------------------------------------------------------------
.PHONY: venv
venv: ## Créer l'environnement virtuel Python
	@echo "$(YELLOW)Création du venv...$(RESET)"
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	@echo "$(GREEN)✅ Venv créé. Active-le : source .venv/bin/activate$(RESET)"

.PHONY: install
install: ## Installer toutes les dépendances Python
	@echo "$(YELLOW)Installation des dépendances...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dépendances installées$(RESET)"

.PHONY: install-dev
install-dev: install ## Installer les dépendances + outils de dev
	$(PIP) install black isort mypy ruff
	@echo "$(GREEN)✅ Outils de dev installés$(RESET)"

# -----------------------------------------------------------------------------
# INFRASTRUCTURE DOCKER
# -----------------------------------------------------------------------------
.PHONY: up
up: ## Démarrer tous les services Docker (Qdrant, PostgreSQL, MinIO, MLflow)
	@echo "$(YELLOW)Démarrage de l'infrastructure...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)✅ Services démarrés$(RESET)"
	@echo "   Qdrant  : http://127.0.0.1:6333/dashboard"
	@echo "   MinIO   : http://127.0.0.1:9101"
	@echo "   MLflow  : http://127.0.0.1:5000"

.PHONY: down
down: ## Arrêter les services Docker (données conservées)
	@echo "$(YELLOW)Arrêt des services...$(RESET)"
	docker compose down
	@echo "$(GREEN)✅ Services arrêtés$(RESET)"

.PHONY: down-volumes
down-volumes: ## Arrêter et SUPPRIMER toutes les données Docker ⚠️
	@echo "$(YELLOW)⚠️  Suppression de toutes les données Docker...$(RESET)"
	docker compose down -v
	@echo "$(GREEN)✅ Services et volumes supprimés$(RESET)"

.PHONY: status
status: ## Vérifier l'état des services Docker
	docker compose ps

.PHONY: logs
logs: ## Afficher les logs de tous les services
	docker compose logs -f

.PHONY: logs-qdrant
logs-qdrant: ## Logs Qdrant uniquement
	docker compose logs -f qdrant

.PHONY: logs-postgres
logs-postgres: ## Logs PostgreSQL uniquement
	docker compose logs -f postgres

# -----------------------------------------------------------------------------
# INITIALISATION
# -----------------------------------------------------------------------------
.PHONY: setup
setup: ## Initialisation complète (venv + docker + DB + Qdrant)
	@echo "$(BLUE)Initialisation complète de MaintiDoc...$(RESET)"
	$(MAKE) up
	@echo "$(YELLOW)Attente que les services soient prêts (30s)...$(RESET)"
	sleep 30
	$(MAKE) setup-db
	$(MAKE) setup-qdrant
	@echo "$(GREEN)✅ MaintiDoc prêt !$(RESET)"

.PHONY: setup-db
setup-db: ## Initialiser les tables PostgreSQL
	@echo "$(YELLOW)Initialisation PostgreSQL...$(RESET)"
	$(PYTHON) scripts/setup_db.py
	@echo "$(GREEN)✅ Base de données initialisée$(RESET)"

.PHONY: setup-qdrant
setup-qdrant: ## Initialiser les collections Qdrant
	@echo "$(YELLOW)Initialisation Qdrant...$(RESET)"
	$(PYTHON) scripts/setup_qdrant.py
	@echo "$(GREEN)✅ Collections Qdrant créées$(RESET)"

# -----------------------------------------------------------------------------
# MODÈLES OLLAMA
# -----------------------------------------------------------------------------
.PHONY: pull-models
pull-models: ## Télécharger les modèles Ollama (LLM + VLM)
	@echo "$(YELLOW)Téléchargement llama3.2:3b (~2Go)...$(RESET)"
	ollama pull llama3.2:3b
	@echo "$(YELLOW)Téléchargement llava:7b-v1.6-mistral-q4_0 (~4Go)...$(RESET)"
	ollama pull llava:7b-v1.6-mistral-q4_0
	@echo "$(GREEN)✅ Modèles téléchargés$(RESET)"

# -----------------------------------------------------------------------------
# APPLICATION
# -----------------------------------------------------------------------------
.PHONY: api
api: ## Lancer le backend FastAPI (développement)
	@echo "$(YELLOW)Démarrage FastAPI sur http://127.0.0.1:8001$(RESET)"
	$(PYTHON) -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001

.PHONY: ui
ui: ## Lancer l'interface Streamlit
	@echo "$(YELLOW)Démarrage Streamlit sur http://localhost:8501$(RESET)"
	$(PYTHON) -m streamlit run frontend/app.py

.PHONY: mcp
mcp: ## Lancer le serveur MCP
	@echo "$(YELLOW)Démarrage serveur MCP...$(RESET)"
	$(PYTHON) mcp_server/main.py

# -----------------------------------------------------------------------------
# INGESTION
# -----------------------------------------------------------------------------
.PHONY: ingest
ingest: ## Ingérer un PDF : make ingest PDF=data/raw/manuel.pdf
	@if [ -z "$(PDF)" ]; then \
		echo "$(YELLOW)Usage : make ingest PDF=chemin/vers/fichier.pdf$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Ingestion de : $(PDF)$(RESET)"
	$(PYTHON) scripts/ingest_sample.py --file "$(PDF)"

.PHONY: ingest-sample
ingest-sample: ## Ingérer les documents de démonstration
	@echo "$(YELLOW)Ingestion des documents de démonstration...$(RESET)"
	$(PYTHON) scripts/ingest_sample.py --sample
	@echo "$(GREEN)✅ Documents de démonstration ingérés$(RESET)"

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
.PHONY: test
test: ## Lancer tous les tests
	@echo "$(YELLOW)Lancement des tests...$(RESET)"
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-unit
test-unit: ## Tests unitaires uniquement
	$(PYTEST) tests/unit/ -v --tb=short

.PHONY: test-integration
test-integration: ## Tests d'intégration uniquement
	$(PYTEST) tests/integration/ -v --tb=short

.PHONY: test-cov
test-cov: ## Tests avec rapport de couverture
	$(PYTEST) tests/ --cov=backend --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Rapport : htmlcov/index.html$(RESET)"

.PHONY: test-config
test-config: ## Tester la configuration
	$(PYTHON) scripts/test_config.py

# -----------------------------------------------------------------------------
# QUALITÉ DU CODE
# -----------------------------------------------------------------------------
.PHONY: lint
lint: ## Vérifier le style du code (ruff)
	$(PYTHON) -m ruff check backend/ tests/

.PHONY: format
format: ## Formater le code (black + isort)
	$(PYTHON) -m black backend/ tests/ frontend/
	$(PYTHON) -m isort backend/ tests/ frontend/

.PHONY: typecheck
typecheck: ## Vérification des types (mypy)
	$(PYTHON) -m mypy backend/ --ignore-missing-imports

# -----------------------------------------------------------------------------
# ÉVALUATION
# -----------------------------------------------------------------------------
.PHONY: evaluate
evaluate: ## Lancer l'évaluation RAG (Recall@5, nDCG@5, MRR@10)
	@echo "$(YELLOW)Lancement de l'évaluation...$(RESET)"
	$(PYTHON) evaluation/run_evaluation.py

# -----------------------------------------------------------------------------
# NETTOYAGE
# -----------------------------------------------------------------------------
.PHONY: clean
clean: ## Supprimer les fichiers temporaires Python
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(RESET)"

.PHONY: clean-data
clean-data: ## Supprimer les données traitées (garde les PDF bruts)
	rm -rf data/processed/*
	@echo "$(GREEN)✅ Données traitées supprimées$(RESET)"
