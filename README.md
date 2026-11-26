# 🔧 MaintiDoc

> **Assistant documentaire multimodal et agentique pour techniciens de maintenance**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)](https://qdrant.tech)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Description

MaintiDoc est un assistant IA destiné aux techniciens de maintenance industrielle.  
Il permet de **poser des questions en français** sur des manuels techniques, **analyser des photos d'équipements**, **consulter l'historique des interventions** et **préparer des tickets de maintenance** avec validation humaine obligatoire.

> ⚠️ MaintiDoc est une **aide documentaire**. Il ne pilote aucun équipement et ne remplace pas les procédures de sécurité ni un technicien habilité.

---

## ✨ Fonctionnalités

| Fonctionnalité | Statut |
|---|---|
| Ingestion de documents PDF | 🔄 Phase 2 |
| RAG textuel avec citations vérifiables | 🔄 Phase 3-4 |
| Analyse d'images et schémas (VLM) | 🔄 Phase 5 |
| Consultation équipements via MCP | 🔄 Phase 6 |
| Agent intelligent LangGraph | 🔄 Phase 7 |
| Validation humaine des tickets | 🔄 Phase 8 |
| Sécurité et contrôle d'accès | 🔄 Phase 9 |
| Évaluation et fine-tuning | 🔄 Phase 10 |
| Interface Streamlit | 🔄 Phase 11 |
| Déploiement Docker complet | 🔄 Phase 12 |

---

## 🏗️ Architecture

```
Technicien (Streamlit)
       ↓
  FastAPI (API Gateway)
       ↓
  Agent LangGraph
  ├── RAG Textuel (Qdrant + BGE-M3)
  ├── RAG Visuel  (Qdrant + CLIP)
  ├── VLM         (LLaVA via Ollama)
  └── MCP Tools   (équipements, tickets)
       ↓
  PostgreSQL | MinIO | MLflow
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Backend API | FastAPI + Pydantic |
| Interface | Streamlit |
| Orchestration agent | LangGraph + LangChain |
| Base vectorielle | Qdrant |
| Embeddings texte | BAAI/bge-m3 |
| Embeddings image | CLIP multilingue |
| LLM local | Llama 3.2:3b via Ollama |
| VLM | LLaVA 7B via Ollama |
| Protocole outils | MCP (FastMCP) |
| Base relationnelle | PostgreSQL 16 |
| Stockage fichiers | MinIO |
| MLOps | MLflow |
| Infrastructure | Docker Compose |

---

## 🚀 Installation rapide

### Prérequis

- Python 3.11+
- Docker & Docker Compose
- Ollama ([installation](https://ollama.com))
- 8 Go de RAM minimum (optimisé pour Apple M2)

### 1. Cloner le projet

```bash
git clone https://github.com/Mouhlal/MaintiDoc.git
cd MaintiDoc/maintidoc
```

### 2. Configurer l'environnement

```bash
cp .env.example .env
# Édite .env avec tes valeurs
```

### 3. Lancer l'infrastructure

```bash
docker compose up -d
docker compose ps  # Vérifier que tout est healthy
```

### 4. Installer les dépendances Python

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Télécharger les modèles Ollama

```bash
ollama pull llama3.2:3b
ollama pull llava:7b-v1.6-mistral-q4_0
```

### 6. Lancer l'application

```bash
# Backend
uvicorn backend.main:app --reload

# Frontend (autre terminal)
streamlit run frontend/app.py
```

---

## 📁 Structure du projet

```
maintidoc/
├── backend/
│   ├── api/          # Routes FastAPI
│   ├── ingestion/    # Traitement PDF
│   ├── rag/          # RAG textuel et visuel
│   ├── llm/          # Génération LLM
│   ├── vision/       # Analyse VLM
│   ├── mcp/          # Serveur et client MCP
│   ├── agent/        # Workflow LangGraph
│   ├── db/           # Modèles et CRUD PostgreSQL
│   └── security/     # Auth et permissions
├── frontend/         # Interface Streamlit
├── evaluation/       # Métriques RAG
├── fine_tuning/      # Fine-tuning reranker
├── tests/            # Tests unitaires et intégration
├── scripts/          # Utilitaires
├── docker-compose.yml
└── requirements.txt
```

---

## 📊 Phases de développement

| Phase | Description | Branche Git |
|---|---|---|
| Phase 1 | Architecture & Environnement | `phase/01-architecture` |
| Phase 2 | Ingestion PDF | `phase/02-ingestion` |
| Phase 3 | RAG Textuel | `phase/03-rag-textuel` |
| Phase 4 | Réponses LLM & Citations | `phase/04-llm-citations` |
| Phase 5 | RAG Multimodal | `phase/05-rag-multimodal` |
| Phase 6 | Serveur MCP | `phase/06-mcp` |
| Phase 7 | Agent LangGraph | `phase/07-agent` |
| Phase 8 | Tickets & Validation humaine | `phase/08-tickets` |
| Phase 9 | Sécurité | `phase/09-securite` |
| Phase 10 | Évaluation & Fine-tuning | `phase/10-evaluation` |
| Phase 11 | Interface Streamlit | `phase/11-interface` |
| Phase 12 | Docker & Démo finale | `phase/12-deploiement` |

---

## 🔒 Sécurité

- Validation humaine **obligatoire** avant toute création de ticket
- Contrôle d'accès par rôle sur les documents
- Protection contre les injections de prompt
- Journalisation sans données sensibles

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)

---

## 👤 Auteur

**Mouhlal** — Étudiant en ingénierie informatique, spécialité Data & IA

---

*Projet réalisé dans le cadre d'une formation pratique en Generative AI, RAG et MCP.*
