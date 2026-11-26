"""
nodes.py — Nœuds du workflow LangGraph de MaintiDoc
=====================================================

Chaque nœud est une fonction async qui reçoit l'AgentState,
effectue une action, et retourne un dict de mise à jour partielle.

Nœuds définis :
  1. analyze_intent        → Comprend ce que veut le technicien
  2. retrieve_documents    → Cherche dans la base vectorielle
  3. rerank_results        → Rerrank les résultats
  4. consult_mcp           → Consulte les outils MCP (équipements/tickets)
  5. analyze_image         → Analyse l'image avec le VLM
  6. generate_response     → Génère la réponse avec le LLM
  7. verify_citations      → Vérifie les citations de la réponse
  8. prepare_ticket        → Prépare un brouillon de ticket
  9. request_human_approval→ Interrompt le workflow pour validation humaine
  10. create_ticket        → Crée le ticket après approbation
  11. handle_uncertainty   → Gère les cas de faible confiance
"""

import base64
import secrets
import time
from loguru import logger

from backend.agent.state import AgentState
from backend.config import settings


# =============================================================================
# NŒUD 1 — ANALYSE DE L'INTENTION
# =============================================================================

async def analyze_intent(state: AgentState) -> dict:
    """
    Analyse la question pour détecter l'intention et planifier le workflow.

    Détecte :
    - Est-ce une demande de création de ticket ?
    - Y a-t-il une image à analyser ?
    - Faut-il consulter un équipement via MCP ?
    - La question est-elle trop vague ?
    """
    logger.info(f"[analyze_intent] Question: {state['question'][:100]}...")
    question = state["question"].lower()

    # Détection intention ticket
    ticket_keywords = ["ticket", "créer une demande", "ouvrir un ticket",
                       "signaler une panne", "déclarer", "intervention urgente"]
    is_ticket_request = any(kw in question for kw in ticket_keywords)

    # Détection besoin MCP
    mcp_keywords = ["historique", "interventions", "pannes précédentes",
                    "tickets ouverts", "équipement", "statut machine"]
    needs_mcp = bool(state.get("equipment_id")) or any(kw in question for kw in mcp_keywords)

    # Détection analyse image
    needs_image = state.get("image_bytes") is not None or state.get("image_base64") is not None

    # Détection question vague
    too_short = len(state["question"].strip()) < 10
    vague_keywords = ["problème", "aide", "quoi faire"]
    is_vague = too_short or (any(kw in question for kw in vague_keywords) and len(question) < 30)

    intent = "ticket" if is_ticket_request else "image_analysis" if needs_image else "unclear" if is_vague else "question"

    logger.info(f"[analyze_intent] Intent={intent}, needs_mcp={needs_mcp}, needs_image={needs_image}")

    return {
        "intent": intent,
        "needs_mcp": needs_mcp,
        "needs_image_analysis": needs_image,
        "needs_clarification": is_vague and not is_ticket_request,
        "current_step": "analyze_intent",
        "messages": [{"role": "system", "content": f"Intent détecté: {intent}"}],
    }


# =============================================================================
# NŒUD 2 — RÉCUPÉRATION DOCUMENTAIRE
# =============================================================================

async def retrieve_documents(state: AgentState) -> dict:
    """
    Recherche les documents pertinents dans Qdrant.
    Gère la recherche textuelle et/ou visuelle selon le contexte.
    """
    logger.info("[retrieve_documents] Démarrage de la recherche...")

    try:
        from backend.rag.retriever import SemanticRetriever
        retriever = SemanticRetriever()

        query = state["question"]
        filters = {}
        if state.get("document_ids"):
            # Qdrant ne supporte pas "IN" directement → on prend le premier doc
            # En production: requête multi-doc avec "should" filter
            filters["document_id"] = state["document_ids"][0]

        # Récupération textuelle
        image_bytes = state.get("image_bytes")
        if image_bytes and state.get("needs_image_analysis"):
            # Recherche multimodale : texte + image
            results = await retriever.retrieve_multimodal(
                query=query,
                image_bytes=image_bytes,
                top_k=settings.rag_top_k
            )
        else:
            results = await retriever.retrieve(
                query=query,
                top_k=settings.rag_top_k,
                filters=filters if filters else None
            )

        from backend.security.permissions import filter_results_by_role
        results = filter_results_by_role(results, state.get("user_role", "technician"))

        logger.info(f"[retrieve_documents] {len(results)} résultats trouvés")

        return {
            "retrieved_chunks": results,
            "current_step": "retrieve_documents",
            "messages": [{"role": "system", "content": f"{len(results)} chunks récupérés"}],
        }
    except Exception as e:
        logger.error(f"[retrieve_documents] Erreur: {e}")
        return {
            "retrieved_chunks": [],
            "errors": [f"Erreur retrieval: {str(e)}"],
            "current_step": "retrieve_documents",
        }


# =============================================================================
# NŒUD 3 — RERANKING
# =============================================================================

async def rerank_results(state: AgentState) -> dict:
    """
    Rerrank les résultats de Qdrant avec le CrossEncoder.
    Améliore la précision en reclassant par pertinence réelle.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"reranked_chunks": [], "current_step": "rerank_results"}

    logger.info(f"[rerank_results] Reranking de {len(chunks)} chunks...")
    try:
        from backend.rag.reranker import Reranker
        from backend.rag.context_builder import ContextBuilder

        reranker = Reranker()
        reranked = await reranker.rerank(
            query=state["question"],
            results=chunks,
            top_k=settings.rag_top_k_reranked
        )

        builder = ContextBuilder()
        context_string, context_items = builder.build(state["question"], reranked)

        logger.info(f"[rerank_results] {len(reranked)} chunks après reranking")
        return {
            "reranked_chunks": reranked,
            "context_string": context_string,
            "context_items": context_items,
            "current_step": "rerank_results",
        }
    except Exception as e:
        logger.error(f"[rerank_results] Erreur: {e}")
        # Fallback : utiliser les chunks non rerankés
        from backend.rag.context_builder import ContextBuilder
        builder = ContextBuilder()
        context_string, context_items = builder.build(state["question"], chunks[:5])
        return {
            "reranked_chunks": chunks[:5],
            "context_string": context_string,
            "context_items": context_items,
            "errors": [f"Reranking échoué (fallback): {str(e)}"],
            "current_step": "rerank_results",
        }


# =============================================================================
# NŒUD 4 — CONSULTATION MCP
# =============================================================================

async def consult_mcp(state: AgentState) -> dict:
    """
    Consulte les outils MCP pour récupérer les infos de l'équipement.
    N'est exécuté que si needs_mcp=True et equipment_id est fourni.
    """
    equipment_id = state.get("equipment_id")
    if not equipment_id:
        logger.info("[consult_mcp] Pas d'equipment_id → skip")
        return {"current_step": "consult_mcp"}

    logger.info(f"[consult_mcp] Consultation équipement: {equipment_id}")
    try:
        from backend.mcp.client import MCPToolsClient
        client = MCPToolsClient(
            user_role=state.get("user_role", "technician"),
            user_id=state.get("user_id"),
        )

        equipment_info = await client.get_equipment(equipment_id)
        history = await client.get_history(equipment_id, limit=5)
        open_tickets = await client.get_open_tickets(equipment_id)

        logger.info(f"[consult_mcp] Équipement trouvé: {equipment_info.get('name', 'inconnu')}")
        return {
            "equipment_info": equipment_info,
            "equipment_history": history,
            "open_tickets": open_tickets,
            "current_step": "consult_mcp",
            "messages": [{"role": "system", "content": f"Équipement consulté: {equipment_id}"}],
        }
    except Exception as e:
        logger.error(f"[consult_mcp] Erreur MCP: {e}")
        return {
            "equipment_info": None,
            "errors": [f"MCP error: {str(e)}"],
            "current_step": "consult_mcp",
        }


# =============================================================================
# NŒUD 5 — ANALYSE D'IMAGE
# =============================================================================

async def analyze_image_node(state: AgentState) -> dict:
    """
    Analyse l'image fournie par le technicien avec le VLM (LLaVA).
    """
    image_bytes = state.get("image_bytes")
    image_b64 = state.get("image_base64")

    if not image_bytes and not image_b64:
        return {"current_step": "analyze_image"}

    logger.info("[analyze_image] Analyse de l'image avec VLM...")
    try:
        from backend.vision.image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer()

        # Convertir base64 → bytes si nécessaire
        if not image_bytes and image_b64:
            image_bytes = base64.b64decode(image_b64)

        result = await analyzer.analyze(
            image_bytes=image_bytes,
            question=state["question"]
        )

        logger.info(f"[analyze_image] Analyse terminée. Issues: {result.get('detected_issues', [])}")
        return {
            "image_analysis": result,
            "current_step": "analyze_image",
            "messages": [{"role": "system", "content": "Image analysée par VLM"}],
        }
    except Exception as e:
        logger.error(f"[analyze_image] Erreur VLM: {e}")
        return {
            "image_analysis": {"error": str(e), "description": "Analyse d'image indisponible"},
            "errors": [f"VLM error: {str(e)}"],
            "current_step": "analyze_image",
        }


# =============================================================================
# NŒUD 6 — GÉNÉRATION DE LA RÉPONSE
# =============================================================================

async def generate_response(state: AgentState) -> dict:
    """
    Génère la réponse finale en appelant le LLM avec le contexte assemblé.
    Intègre les résultats documentaires, MCP et VLM dans le prompt.
    """
    logger.info("[generate_response] Génération de la réponse...")
    try:
        from backend.llm.llm_client import llm_client
        from backend.llm.prompts import SYSTEM_PROMPT, format_rag_prompt

        context = state.get("context_string", "")

        # Enrichir le contexte avec les infos MCP
        equipment_info = state.get("equipment_info")
        if equipment_info:
            eq_text = f"\n\n=== INFORMATIONS ÉQUIPEMENT ===\n"
            eq_text += f"Nom: {equipment_info.get('name', 'N/A')}\n"
            eq_text += f"Statut: {equipment_info.get('status', 'N/A')}\n"
            eq_text += f"Type: {equipment_info.get('equipment_type', 'N/A')}\n"
            history = state.get("equipment_history", [])
            if history:
                eq_text += f"\nDernières interventions:\n"
                for h in history[:3]:
                    eq_text += f"  - {h.get('title', 'N/A')} ({h.get('result', 'N/A')})\n"
            context += eq_text

        # Enrichir avec l'analyse VLM
        image_analysis = state.get("image_analysis")
        if image_analysis and not image_analysis.get("error"):
            vision_text = f"\n\n=== ANALYSE VISUELLE DE L'IMAGE ===\n"
            vision_text += image_analysis.get("vlm_description", "")
            issues = image_analysis.get("detected_issues", [])
            if issues:
                vision_text += f"\nProblèmes visuels détectés: {', '.join(issues)}"
            context += vision_text

        if not context.strip():
            # Aucun contexte trouvé → réponse d'incertitude
            return {
                "llm_response": "Je ne trouve pas d'information pertinente dans les documents disponibles pour répondre à votre question. Veuillez vérifier que les documents relatifs à cet équipement ont bien été importés.",
                "confidence_score": 0.0,
                "current_step": "generate_response",
            }

        prompt = format_rag_prompt(context=context, question=state["question"])
        llm_resp = llm_client.generate(prompt=prompt, system=SYSTEM_PROMPT)

        logger.info(f"[generate_response] Réponse générée ({llm_resp.tokens_used} tokens, {llm_resp.latency_ms:.0f}ms)")
        return {
            "llm_response": llm_resp.content,
            "current_step": "generate_response",
        }
    except Exception as e:
        logger.error(f"[generate_response] Erreur LLM: {e}")
        return {
            "llm_response": "Désolé, le service de génération est temporairement indisponible. Veuillez réessayer ou consulter directement la documentation.",
            "errors": [f"LLM error: {str(e)}"],
            "current_step": "generate_response",
        }


# =============================================================================
# NŒUD 7 — VÉRIFICATION DES CITATIONS
# =============================================================================

async def verify_citations(state: AgentState) -> dict:
    """
    Vérifie que les citations dans la réponse correspondent aux sources réelles.
    Calcule le score de confiance final.
    """
    logger.info("[verify_citations] Vérification des citations...")
    try:
        from backend.llm.citation_checker import CitationChecker
        from backend.llm.response_parser import ResponseParser

        checker = CitationChecker()
        parser = ResponseParser()

        raw_response = state.get("llm_response", "")
        context_items = state.get("context_items", [])

        citation_result = checker.verify(raw_response, context_items)
        parsed = parser.parse(raw_response, context_items)

        # Construire les sources pour l'UI
        sources = []
        for item in context_items:
            sources.append({
                "document_id": getattr(item, "document_id", ""),
                "page_number": getattr(item, "page_number", 0),
                "chunk_type": getattr(item, "chunk_type", "text"),
                "content_preview": getattr(item, "content", "")[:200],
                "score": getattr(item, "score", 0.0),
            })

        logger.info(f"[verify_citations] Confiance: {parsed.confidence_score:.2f}, Hallucination: {citation_result.get('hallucination_risk', False)}")

        return {
            "citation_check": citation_result,
            "parsed_response": parsed,
            "confidence_score": parsed.confidence_score,
            "final_answer": raw_response,
            "sources": sources,
            "current_step": "verify_citations",
        }
    except Exception as e:
        logger.error(f"[verify_citations] Erreur: {e}")
        return {
            "citation_check": {"valid": False, "error": str(e)},
            "confidence_score": 0.5,
            "final_answer": state.get("llm_response", ""),
            "sources": [],
            "current_step": "verify_citations",
        }


# =============================================================================
# NŒUD 8 — PRÉPARATION DU TICKET
# =============================================================================

async def prepare_ticket(state: AgentState) -> dict:
    """
    Prépare un brouillon de ticket de maintenance basé sur la conversation.
    NE CRÉE PAS le ticket — attend la validation humaine.
    """
    logger.info("[prepare_ticket] Préparation du brouillon de ticket...")
    try:
        from backend.llm.llm_client import llm_client
        from backend.llm.prompts import format_ticket_prompt

        equipment_info = state.get("equipment_info") or {}
        problem = state.get("question", "")
        sources = state.get("sources", [])
        source_labels = [
            f"Doc {s.get('document_id', '')}, Page {s.get('page_number', 0)}"
            if isinstance(s, dict)
            else str(s)
            for s in sources
        ]

        prompt = format_ticket_prompt(
            equipment_info=equipment_info,
            problem=problem,
            sources=source_labels,
        )

        # Générer le contenu du ticket (avec fallback si Ollama indisponible)
        try:
            ticket_response = llm_client.generate(prompt=prompt)
            ticket_description = ticket_response.content
        except Exception as llm_exc:
            logger.warning(f"[prepare_ticket] LLM indisponible ({llm_exc}), fallback structuré")
            ticket_description = (
                f"Problème signalé : {problem}\n"
                f"Équipement : {equipment_info.get('name', 'Non spécifié')}\n"
                f"Sources : {', '.join(source_labels) if source_labels else 'Aucune'}"
            )

        # Token unique pour l'idempotence (empêche double création)
        approval_token = secrets.token_urlsafe(32)

        ticket_draft = {
            "equipment_id": state.get("equipment_id", ""),
            "equipment_name": equipment_info.get("name", "Équipement non spécifié"),
            "title": f"Intervention requise - {equipment_info.get('name', 'Équipement')}",
            "description": ticket_description,
            "priority": "medium",
            "source_chunks": [s.get("document_id", "") if isinstance(s, dict) else str(s) for s in sources[:3]],
            "agent_reasoning": state.get("final_answer", "")[:500],
            "approval_token": approval_token,
        }

        logger.info(f"[prepare_ticket] Brouillon créé, token: {approval_token[:8]}...")
        return {
            "ticket_draft": ticket_draft,
            "approval_token": approval_token,
            "human_approved": False,
            "current_step": "prepare_ticket",
            "messages": [{"role": "assistant", "content": "Brouillon de ticket préparé. Validation humaine requise."}],
        }
    except Exception as e:
        logger.error(f"[prepare_ticket] Erreur: {e}")
        return {
            "errors": [f"Erreur préparation ticket: {str(e)}"],
            "current_step": "prepare_ticket",
        }


# =============================================================================
# NŒUD 9 — DEMANDE DE VALIDATION HUMAINE (INTERRUPTION)
# =============================================================================

async def request_human_approval(state: AgentState) -> dict:
    """
    Nœud d'interruption : suspend le workflow pour validation humaine.

    Dans LangGraph, ce nœud utilise `interrupt()` pour suspendre l'exécution.
    Le workflow reprend quand le superviseur appelle l'API d'approbation.

    En pratique : l'état est sauvegardé, l'UI affiche le brouillon,
    le technicien clique "Approuver" ou "Rejeter".
    """
    logger.info("[human_approval] ⏸️  Interruption pour validation humaine")

    ticket = state.get("ticket_draft", {})
    logger.info(f"[human_approval] Ticket en attente: {ticket.get('title', 'N/A')}")

    return {
        "current_step": "request_human_approval",
        "messages": [{
            "role": "assistant",
            "content": (
                f"⚠️ **Validation humaine requise**\n\n"
                f"Je souhaite créer le ticket suivant :\n"
                f"**Titre** : {ticket.get('title', 'N/A')}\n"
                f"**Équipement** : {ticket.get('equipment_name', 'N/A')}\n"
                f"**Priorité** : {ticket.get('priority', 'medium')}\n\n"
                f"**Description** :\n{ticket.get('description', '')[:500]}...\n\n"
                f"Voulez-vous **Approuver** ou **Rejeter** cette demande ?"
            )
        }],
    }


# =============================================================================
# NŒUD 10 — CRÉATION DU TICKET (après approbation)
# =============================================================================

async def create_ticket(state: AgentState) -> dict:
    """
    Crée le ticket dans PostgreSQL après validation humaine.
    Vérifie que l'approbation est bien présente avant d'agir.
    """
    if not state.get("human_approved"):
        logger.warning("[create_ticket] Tentative de création sans approbation !")
        return {
            "errors": ["Création de ticket refusée : validation humaine manquante"],
            "current_step": "create_ticket",
        }

    logger.info("[create_ticket] ✅ Création du ticket après approbation...")
    try:
        from backend.mcp.client import MCPToolsClient
        client = MCPToolsClient(
            user_role=state.get("user_role", "technician"),
            user_id=state.get("user_id"),
        )
        draft = state.get("ticket_draft", {})

        ticket = await client.create_ticket(
            equipment_id=draft.get("equipment_id", ""),
            title=draft.get("title", ""),
            description=draft.get("description", ""),
            priority=draft.get("priority", "medium"),
            approval_token=draft.get("approval_token", ""),
            source_chunks=draft.get("source_chunks", []),
            agent_reasoning=draft.get("agent_reasoning", ""),
        )

        logger.info(f"[create_ticket] Ticket créé: {ticket.get('ticket_number', 'N/A')}")
        return {
            "ticket_created": ticket,
            "current_step": "create_ticket",
            "final_answer": f"✅ Ticket **{ticket.get('ticket_number', 'N/A')}** créé avec succès.\n\nIl est maintenant en attente d'assignation.",
            "messages": [{"role": "assistant", "content": f"Ticket {ticket.get('ticket_number')} créé."}],
        }
    except Exception as e:
        logger.error(f"[create_ticket] Erreur: {e}")
        return {
            "errors": [f"Erreur création ticket: {str(e)}"],
            "current_step": "create_ticket",
        }


# =============================================================================
# NŒUD 11 — GESTION DE L'INCERTITUDE
# =============================================================================

async def handle_uncertainty(state: AgentState) -> dict:
    """
    Gère les cas où la confiance est trop faible ou où le contexte manque.
    Formule une réponse honnête sur les limites de l'assistant.
    """
    logger.info(f"[handle_uncertainty] Confiance faible: {state.get('confidence_score', 0):.2f}")

    question = state.get("question", "")
    answer = (
        f"⚠️ **Information insuffisante pour répondre avec certitude.**\n\n"
        f"Je n'ai pas trouvé de documentation suffisamment pertinente pour votre question : "
        f"*\"{question}\"*\n\n"
        f"**Recommandations :**\n"
        f"- Vérifiez que les manuels de l'équipement concerné ont été importés\n"
        f"- Consultez directement le technicien habilité\n"
        f"- Référez-vous aux procédures de sécurité officielles\n\n"
        f"*MaintiDoc ne peut pas confirmer un diagnostic sans source documentaire vérifiable.*"
    )

    return {
        "final_answer": answer,
        "confidence_score": 0.0,
        "current_step": "handle_uncertainty",
    }


# =============================================================================
# NŒUD 12 — DEMANDE DE CLARIFICATION
# =============================================================================

async def request_clarification(state: AgentState) -> dict:
    """
    Demande une clarification au technicien si la question est trop vague.
    """
    question = state.get("question", "")
    answer = (
        f"🔍 **Votre question est trop vague pour que je puisse vous aider efficacement.**\n\n"
        f"Question reçue : *\"{question}\"*\n\n"
        f"Pourriez-vous préciser :\n"
        f"- **L'équipement** concerné (ex: pompe P-101, compresseur C-301)\n"
        f"- **Le symptôme** observé (ex: fuite, bruit anormal, vibration)\n"
        f"- **Ce que vous avez déjà essayé**\n\n"
        f"Exemple : *\"La pompe centrifuge P-101 fait un bruit anormal depuis ce matin, que dit la procédure de diagnostic ?\"*"
    )

    return {
        "final_answer": answer,
        "needs_clarification": True,
        "current_step": "request_clarification",
    }
