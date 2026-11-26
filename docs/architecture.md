# Architecture Technique — MaintiDoc

## Vue d'ensemble

MaintiDoc suit une architecture modulaire découpée en couches clairement séparées :

1. **Infrastructure (`docker-compose.yml`)** :
   - **Qdrant** : stockage vectoriel double collection (`maintidoc_text` dim=1024 BGE-M3, `maintidoc_visual` dim=512 CLIP).
   - **PostgreSQL 16** : métadonnées documentaires, catalogue équipements, historique des interventions, tickets et utilisateurs.
   - **MinIO** : stockage objet S3 pour PDF bruts et figures extraites.
   - **MLflow** : suivi des expérimentations RAG et fine-tuning.

2. **Ingestion (`backend/ingestion/`)** :
   - `PDFLoader` (PyMuPDF) extrait texte natif, tableaux et références d'images.
   - `OCRProcessor` (EasyOCR FR/EN) traite uniquement les pages scannées (`len < 100`).
   - `TextChunker` découpe en respectant les phrases avec chevauchement configurable (`512/64`).
   - `ImageExtractor` filtre les icônes (`< 100x100`) et pousse les schémas vers MinIO.

3. **RAG Multimodal (`backend/rag/` & `backend/vision/`)** :
   - Recherche dense (`BGE-M3`) + recherche visuelle (`CLIP`) fusionnées par **Reciprocal Rank Fusion (RRF)**.
   - Reranking CrossEncoder (`ms-marco-MiniLM-L-6-v2`) pour maximiser la précision avant génération.
   - `VLMClient` (LLaVA Q4 via Ollama) analyse les photos d'équipements à la demande.

4. **MCP & Agent LangGraph (`backend/mcp/` & `backend/agent/`)** :
   - Serveur **FastMCP** exposant `get_equipment`, `history`, `get_open_tickets`, `create_ticket`.
   - Graphe `StateGraph` avec **interruption humaine (`interrupt_after=["request_human_approval"]`)** empêchant toute création de ticket sans validation explicite et clé d'idempotence.

5. **Sécurité (`backend/security/`)** :
   - JWT (`HS256`) + RBAC (`technician`, `supervisor`, `admin`).
   - Protection contre les injections de prompt et filtrage par permissions sur Qdrant.
