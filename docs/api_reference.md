# Référence API — MaintiDoc

Base URL : `http://localhost:8000/api/v1`

## Health
- `GET /health` : statut rapide.
- `GET /health/detailed` : latence et état de Qdrant, PostgreSQL, MinIO, Ollama.

## Authentification
- `POST /auth/token` : obtient un token JWT (`OAuth2PasswordRequestForm`).
- `POST /auth/register` : crée un compte utilisateur.
- `GET /auth/me` : retourne l'utilisateur courant.

## Documents
- `POST /documents/upload` : upload multipart d'un PDF, sauvegarde MinIO + indexation Qdrant.
- `GET /documents/` : liste les documents accessibles selon le rôle.
- `GET /documents/{document_id}` : fiche détaillée d'un document.
- `DELETE /documents/{document_id}` : supprime le document dans Qdrant, MinIO et PostgreSQL.

## Chat Agentique
- `POST /chat/ask` : requête JSON (`question`, `document_ids`, `equipment_id`, `image_base64`).
- `POST /chat/ask-upload` : requête multipart avec photo d'équipement.

## Tickets & Équipements
- `GET /tickets/` : liste tous les tickets.
- `GET /tickets/pending-approvals` : brouillons générés par l'agent en attente de validation.
- `POST /tickets/approve-draft` : valide ou rejette un brouillon (`approval_token`, `approved`).
- `GET /tickets/equipments` : liste des équipements.
- `GET /tickets/equipments/{equipment_id}` : fiche équipement, historique et tickets ouverts.
