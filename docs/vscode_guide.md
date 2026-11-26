# Guide VS Code — Exécuter, Déboguer et Apprendre MaintiDoc

## 1. Ouvrir le projet au bon niveau

Dans VS Code, ouvre **directement le dossier `maintidoc/`** (et non le dossier parent `MCP_AIG/`) :

```bash
code /Users/mouhlal/MCP_AIG/maintidoc
```

Ainsi, `.vscode/settings.json`, `.vscode/launch.json` et `.vscode/tasks.json` sont automatiquement activés.

---

## 2. Extensions VS Code recommandées

1. **Python** (`ms-python.python`)
2. **Python Debugger** (`ms-python.debugpy`)
3. **Docker** (`ms-azuretools.vscode-docker`) — pour voir Qdrant/Postgres/MinIO visuellement

---

## 3. Démarrer en 3 clics dans VS Code

### Étape A — Lancer Docker (Qdrant, PostgreSQL, MinIO)
- Menu **Terminal → Run Task…** (`Cmd+Shift+P` → `Tasks: Run Task`)
- Choisis **`1. Docker Up (Qdrant + Postgres + MinIO)`**

### Étape B — Lancer API + UI avec le Débogueur (`F5`)
- Ouvre le panneau **Run and Debug** (`Cmd+Shift+D`)
- Sélectionne **`⚡ MaintiDoc Complet (API + UI)`** dans la liste déroulante
- Appuie sur **`F5`** (triangle vert)

Les deux serveurs démarrent ensemble dans VS Code :
- API FastAPI (Swagger) : http://127.0.0.1:8001/docs
- Interface Streamlit : http://127.0.0.1:8501

---

## 4. Parcours d'apprentissage recommandé avec des Breakpoints 🔴

Pour vraiment comprendre chaque ligne et la défendre en soutenance, pose un **breakpoint** (clic dans la marge gauche) puis lance **`F5`** :

1. **Comprendre la configuration & FastAPI**
   - Breakpoint dans `backend/config.py` (`validate_config`) et `backend/api/health.py` (`health_detailed`)
   - Ouvre http://127.0.0.1:8001/api/v1/health/detailed et observe l'exécution asynchrone parallèle (`asyncio.gather`).

2. **Comprendre l'ingestion PDF**
   - Pose un breakpoint dans `backend/ingestion/pdf_loader.py` (`load`), `chunker.py` (`_chunk_text`), `metadata_builder.py`.
   - Lance **`📥 Ingérer Manuel Démo (Debug)`** avec `F5` (après avoir installé `torch sentence-transformers FlagEmbedding`).

3. **Comprendre le workflow agentique LangGraph + Validation Humaine**
   - Pose des breakpoints dans `backend/agent/nodes.py` (`analyze_intent`, `retrieve_documents`, `consult_mcp`, `prepare_ticket`) et `backend/agent/edges.py`.
   - Lance **`🧪 Pytest (Tous les tests)`** ou **`⚡ MaintiDoc Complet`** et pose une question demandant un ticket.
   - Observe comment le graphe s'interrompt avant `create_ticket` (`interrupt_after=["request_human_approval"]`) et n'écrit en base PostgreSQL qu'après ton clic **Approuver**.

4. **Comprendre MCP et la Sécurité**
   - Breakpoints dans `backend/mcp/permissions.py` (`check_mcp_permission`), `backend/mcp/tools.py`, et `backend/security/input_validator.py`.
   - Teste une injection de prompt (`"Ignore previous instructions..."`) depuis Streamlit et regarde comment `validate_question()` la bloque avant même d'appeler le LLM.
