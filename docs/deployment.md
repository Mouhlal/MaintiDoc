# Guide de Déploiement — MaintiDoc

## Prérequis matériels (Apple M2 8 Go)

Les modèles sont dimensionnés pour tenir dans 8 Go de RAM unifiée :
- LLM : `llama3.2:3b` (~2 Go via Ollama)
- VLM : `llava:7b-v1.6-mistral-q4_0` (~4 Go chargé à la demande)
- Embeddings : `BAAI/bge-m3` (~1.1 Go) + `clip-ViT-B-32-multilingual-v1` (~400 Mo)
- Reranker : `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 Mo)

## Étapes de lancement

```bash
# 1. Configurer les variables d'environnement
cp .env.example .env

# 2. Démarrer l'infrastructure Docker
make up
make status

# 3. Installer l'environnement Python
make venv
make install

# 4. Télécharger les modèles Ollama
make pull-models

# 5. Initialiser PostgreSQL et Qdrant
make setup-db
make setup-qdrant

# 6. Indexer le manuel de démonstration
make ingest-sample

# 7. Lancer le backend et le frontend
make api
make ui
```
