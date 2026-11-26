"""
app.py — Interface utilisateur Streamlit pour MaintiDoc
=========================================================

Onglets :
  1. 💬 Assistant Maintenance (question + photo + sources + validation ticket)
  2. 📄 Documents PDF (upload, indexation, liste)
  3. 🔧 Équipements & Historique (données MCP / PostgreSQL)
  4. 🎫 Validation des Tickets (Human-in-the-loop)
  5. ⚙️ État des Services (Qdrant, PostgreSQL, MinIO, Ollama)
"""

from __future__ import annotations

import os
import httpx
import streamlit as st

API_BASE_URL = os.getenv("MAINTIDOC_API_URL", "http://127.0.0.1:8001/api/v1")

st.set_page_config(
    page_title="MaintiDoc — Assistant Maintenance IA",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 MaintiDoc — Assistant Documentaire de Maintenance")
st.caption(
    "⚠️ Aide documentaire uniquement. Ne pilote aucun équipement et ne remplace pas "
    "les procédures de sécurité ni un technicien habilité."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def api_get(path: str, timeout: float = 10.0):
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{API_BASE_URL}{path}")
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, json_data: dict | None = None, timeout: float = 120.0):
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{API_BASE_URL}{path}", json=json_data)
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# BARRE LATÉRALE — CONTEXTE ÉQUIPEMENT & STATUT RAPIDE
# =============================================================================
with st.sidebar:
    st.header("🎯 Contexte d'intervention")
    selected_equipment = st.selectbox(
        "Équipement ciblé (optionnel)",
        options=[
            "",
            "EQ-POMPE-001",
            "EQ-POMPE-002",
            "EQ-COMPRESSEUR-001",
            "EQ-MOTEUR-001",
            "EQ-VANNE-001",
        ],
    )
    st.divider()
    if st.button("🔄 Rafraîchir état API"):
        try:
            health = api_get("/health")
            st.success(f"API : {health.get('status', 'ok').upper()}")
        except Exception as exc:
            st.error(f"API inaccessible : {exc}")


# =============================================================================
# ONGLETS PRINCIPAUX
# =============================================================================
tab_chat, tab_docs, tab_equip, tab_tickets, tab_health = st.tabs(
    [
        "💬 Assistant Maintenance",
        "📄 Documents PDF",
        "🔧 Équipements & Historique",
        "🎫 Validation Tickets",
        "⚙️ État des Services",
    ]
)


# -----------------------------------------------------------------------------
# TAB 1 : CHAT MULTIMODAL
# -----------------------------------------------------------------------------
with tab_chat:
    st.subheader("Poser une question technique")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources citées ({len(msg['sources'])})"):
                    for idx, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**Source {idx}** — Document `{src.get('document_id')}` "
                            f"| Page **{src.get('page_number')}** "
                            f"| Score `{src.get('score', 0):.2f}`"
                        )
                        st.caption(src.get("content_preview", ""))

    uploaded_image = st.file_uploader(
        "📷 Joindre une photo de panneau ou d'équipement (optionnel)",
        type=["png", "jpg", "jpeg"],
    )

    question = st.chat_input("Posez votre question en français...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche documentaire et analyse en cours..."):
                try:
                    if uploaded_image is not None:
                        files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                        data = {"question": question}
                        if selected_equipment:
                            data["equipment_id"] = selected_equipment
                        with httpx.Client(timeout=180.0) as client:
                            resp = client.post(f"{API_BASE_URL}/chat/ask-upload", data=data, files=files)
                            resp.raise_for_status()
                            result = resp.json()
                    else:
                        payload = {"question": question}
                        if selected_equipment:
                            payload["equipment_id"] = selected_equipment
                        result = api_post("/chat/ask", json_data=payload, timeout=180.0)

                    answer = result.get("answer", "")
                    confidence = result.get("confidence", 0.0)
                    sources = result.get("sources", [])

                    st.markdown(answer)
                    st.caption(f"Indice de confiance : **{confidence:.0%}**")

                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for idx, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Source {idx}** — Document `{src.get('document_id')}` "
                                    f"| Page **{src.get('page_number')}**"
                                )
                                st.caption(src.get("content_preview", ""))

                    if result.get("requires_human_approval") and result.get("ticket_draft"):
                        st.warning(
                            "🎫 Un brouillon de ticket a été préparé. "
                            "Rendez-vous dans l'onglet **Validation Tickets** pour l'approuver ou le rejeter."
                        )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"Erreur lors de l'appel à l'assistant : {exc}")


# -----------------------------------------------------------------------------
# TAB 2 : DOCUMENTS PDF
# -----------------------------------------------------------------------------
with tab_docs:
    st.subheader("Importer et indexer un manuel technique (PDF)")
    with st.form("upload_pdf_form"):
        pdf_file = st.file_uploader("Fichier PDF", type=["pdf"])
        col1, col2, col3 = st.columns(3)
        with col1:
            doc_title = st.text_input("Titre du document")
        with col2:
            eq_type = st.text_input("Type d'équipement", placeholder="pompe_centrifuge")
        with col3:
            doc_version = st.text_input("Version", value="1.0")
        submitted = st.form_submit_button("📥 Importer et Indexer")

    if submitted and pdf_file is not None:
        with st.spinner("Extraction PDF, chunking et indexation dans Qdrant..."):
            try:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                data = {
                    "title": doc_title or pdf_file.name,
                    "equipment_type": eq_type,
                    "version": doc_version,
                }
                with httpx.Client(timeout=180.0) as client:
                    resp = client.post(f"{API_BASE_URL}/documents/upload", data=data, files=files)
                    resp.raise_for_status()
                    doc_res = resp.json()
                st.success(
                    f"✅ Document indexé : {doc_res.get('filename')} "
                    f"({doc_res.get('total_pages')} pages, {doc_res.get('total_chunks')} chunks)"
                )
            except Exception as exc:
                st.error(f"Échec de l'import : {exc}")

    st.divider()
    st.subheader("Documents disponibles")
    try:
        docs = api_get("/documents/")
        if not docs:
            st.info("Aucun document indexé pour le moment.")
        else:
            st.dataframe(docs, use_container_width=True)
    except Exception as exc:
        st.warning(f"Impossible de charger la liste des documents : {exc}")


# -----------------------------------------------------------------------------
# TAB 3 : ÉQUIPEMENTS & HISTORIQUE MCP
# -----------------------------------------------------------------------------
with tab_equip:
    st.subheader("Catalogue des équipements & Historique des interventions")
    try:
        equipments = api_get("/tickets/equipments")
        if equipments:
            st.dataframe(equipments, use_container_width=True)
            eq_ids = [e["equipment_id"] for e in equipments]
            chosen = st.selectbox("Consulter la fiche détaillée :", eq_ids)
            if chosen:
                details = api_get(f"/tickets/equipments/{chosen}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Fiche technique**")
                    st.json(details["equipment"])
                with col_b:
                    st.markdown("**Historique des interventions**")
                    st.json(details["history"])
                    st.markdown("**Tickets ouverts**")
                    st.json(details["open_tickets"])
        else:
            st.info("Aucun équipement enregistré. Lancez `make setup-db`.")
    except Exception as exc:
        st.warning(f"Impossible de charger les équipements : {exc}")


# -----------------------------------------------------------------------------
# TAB 4 : VALIDATION HUMAINE DES TICKETS
# -----------------------------------------------------------------------------
with tab_tickets:
    st.subheader("Validation humaine obligatoire (Human-in-the-loop)")
    try:
        pending = api_get("/tickets/pending-approvals")
        if not pending:
            st.info("Aucun brouillon de ticket en attente de validation.")
        else:
            for item in pending:
                draft = item["ticket_draft"]
                token = item["approval_token"]
                with st.container(border=True):
                    st.markdown(f"### 🎫 {draft.get('title', 'Ticket')}")
                    st.write(f"**Équipement** : {draft.get('equipment_name', 'N/A')}")
                    st.write(f"**Priorité** : {draft.get('priority', 'medium')}")
                    st.text_area("Description proposée", value=draft.get("description", ""), height=140, key=f"desc_{token}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approuver et Créer", key=f"app_{token}"):
                            res = api_post(
                                "/tickets/approve-draft",
                                json_data={"approval_token": token, "approved": True, "approver": "superviseur"},
                            )
                            st.success(f"Ticket créé : {res.get('ticket_created')}")
                            st.rerun()
                    with col2:
                        if st.button("❌ Rejeter", key=f"rej_{token}"):
                            api_post(
                                "/tickets/approve-draft",
                                json_data={
                                    "approval_token": token,
                                    "approved": False,
                                    "approver": "superviseur",
                                    "rejection_reason": "Rejeté depuis l'interface Streamlit",
                                },
                            )
                            st.warning("Brouillon rejeté.")
                            st.rerun()

        st.divider()
        st.markdown("### Tickets enregistrés en base")
        tickets = api_get("/tickets/")
        st.dataframe(tickets, use_container_width=True)
    except Exception as exc:
        st.warning(f"Impossible de charger les tickets : {exc}")


# -----------------------------------------------------------------------------
# TAB 5 : ÉTAT DES SERVICES
# -----------------------------------------------------------------------------
with tab_health:
    st.subheader("Santé de l'infrastructure MaintiDoc")
    if st.button("🔍 Vérifier tous les services"):
        try:
            detailed = api_get("/health/detailed", timeout=15.0)
            st.metric("Statut global", detailed.get("status", "inconnu").upper())
            for svc in detailed.get("services", []):
                icon = "✅" if svc["status"] == "ok" else "⚠️"
                st.write(
                    f"{icon} **{svc['name'].upper()}** — `{svc['status']}` "
                    f"({svc.get('latency_ms', '-')} ms) — {svc.get('detail', '')}"
                )
        except Exception as exc:
            st.error(f"Backend FastAPI indisponible : {exc}")
