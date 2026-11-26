"""
state.py — État du graphe LangGraph pour MaintiDoc
====================================================

Définit le TypedDict qui représente l'état complet de l'agent
à chaque étape du workflow.

LangGraph passe cet état entre chaque nœud du graphe.
Chaque nœud lit l'état, effectue son travail, et retourne
un dict de mise à jour partielle (pas besoin de tout retourner).
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict, total=False):
    """
    État complet du workflow agent MaintiDoc.

    Convention LangGraph :
    - Annotated[list, operator.add] → les mises à jour s'AJOUTENT à la liste (accumulation)
    - Sans annotation → la valeur est REMPLACÉE à chaque update
    """

    # -------------------------------------------------------------------------
    # ENTRÉE UTILISATEUR
    # -------------------------------------------------------------------------
    question: str                        # Question originale du technicien
    image_bytes: bytes | None            # Photo d'équipement (optionnelle)
    image_base64: str | None             # Version base64 pour le LLM
    user_id: str                         # ID de l'utilisateur authentifié
    user_role: str                       # Rôle : technician, supervisor, admin
    document_ids: list[str] | None       # Filtrer sur des documents spécifiques
    equipment_id: str | None             # ID équipement pour consultation MCP

    # -------------------------------------------------------------------------
    # ANALYSE DE L'INTENTION
    # -------------------------------------------------------------------------
    intent: str                          # detected intent: "question", "ticket", "image_analysis", "unclear"
    needs_mcp: bool                      # L'agent doit-il consulter un équipement ?
    needs_image_analysis: bool           # Y a-t-il une image à analyser ?
    needs_clarification: bool            # La question est-elle trop vague ?
    clarification_question: str | None   # Question de clarification à poser

    # -------------------------------------------------------------------------
    # RÉSULTATS DE RECHERCHE
    # -------------------------------------------------------------------------
    retrieved_chunks: list[dict]         # Résultats bruts de Qdrant
    reranked_chunks: list[dict]          # Résultats après reranking
    context_string: str                  # Contexte formaté pour le LLM
    context_items: list[Any]             # ContextItem objects pour citations

    # -------------------------------------------------------------------------
    # RÉSULTATS MCP
    # -------------------------------------------------------------------------
    equipment_info: dict | None          # Infos de l'équipement consulté
    equipment_history: list[dict]        # Historique des interventions
    open_tickets: list[dict]             # Tickets ouverts existants

    # -------------------------------------------------------------------------
    # ANALYSE D'IMAGE (VLM)
    # -------------------------------------------------------------------------
    image_analysis: dict | None          # Résultat de l'analyse VLM

    # -------------------------------------------------------------------------
    # GÉNÉRATION DE LA RÉPONSE
    # -------------------------------------------------------------------------
    llm_response: str                    # Réponse brute du LLM
    parsed_response: Any                 # ParsedResponse object
    final_answer: str                    # Réponse finale formatée

    # -------------------------------------------------------------------------
    # CITATIONS ET VÉRIFICATION
    # -------------------------------------------------------------------------
    citation_check: dict | None          # Résultat de CitationChecker
    confidence_score: float              # Score de confiance 0.0-1.0
    sources: list[dict]                  # Sources formatées pour l'UI

    # -------------------------------------------------------------------------
    # GESTION DES TICKETS
    # -------------------------------------------------------------------------
    ticket_draft: dict | None            # Brouillon du ticket préparé par l'agent
    ticket_created: dict | None          # Ticket créé après validation
    approval_token: str | None           # Token unique pour l'approbation humaine
    human_approved: bool                 # L'humain a-t-il validé la création du ticket ?
    human_rejection_reason: str | None   # Raison du refus (si rejeté)

    # -------------------------------------------------------------------------
    # CONTRÔLE DU WORKFLOW
    # -------------------------------------------------------------------------
    # Accumulation des messages via operator.add (liste append)
    messages: Annotated[list[dict], operator.add]  # Historique de conversation
    errors: Annotated[list[str], operator.add]     # Erreurs accumulées
    should_stop: bool                    # Signal d'arrêt du workflow
    current_step: str                    # Étape courante (pour debug/monitoring)
