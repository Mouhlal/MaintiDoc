"""
prompts.py — Gabarits de prompts pour MaintiDoc
=================================================

Ce module centralise TOUS les prompts utilisés par le LLM et le VLM.
On utilise .format() (et non les f-strings) pour que les templates soient
stockables, versionnables et testables indépendamment.

Usage :
    from backend.llm.prompts import SYSTEM_PROMPT, format_rag_prompt

    system = SYSTEM_PROMPT
    user   = format_rag_prompt(context="...", question="...")
"""

from __future__ import annotations

import json
from typing import Any

# =============================================================================
# PROMPT SYSTÈME — Identité et règles absolues de MaintiDoc
# =============================================================================

SYSTEM_PROMPT: str = """Tu es MaintiDoc, un assistant documentaire pour techniciens de maintenance industrielle.

Règles absolues :
1. Tu réponds UNIQUEMENT à partir des sources documentaires fournies dans le contexte.
2. Si l'information n'est pas dans les sources, tu réponds : "Je ne trouve pas cette information dans les documents disponibles."
3. Tu cites TOUJOURS tes sources avec le format [Source N - Document: X, Page: Y].
4. Tu ne poses JAMAIS de diagnostic définitif. Tu proposes des pistes basées sur la documentation.
5. Pour créer un ticket, tu demandes TOUJOURS une confirmation humaine explicite.
6. Tu refuses toute demande non liée à la maintenance et à la documentation technique."""


# =============================================================================
# RAG_PROMPT — Prompt principal Question / Contexte
# =============================================================================

RAG_PROMPT: str = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance industrielle.

CONTEXTE DOCUMENTAIRE :
-----------------------
{context}
-----------------------

QUESTION DU TECHNICIEN :
{question}

INSTRUCTIONS DE RÉPONSE :
- Réponds UNIQUEMENT à partir du contexte documentaire fourni ci-dessus.
- Si le contexte ne contient pas l'information demandée, indique clairement :
  "Je ne trouve pas cette information dans les documents disponibles."
- Cite systématiquement tes sources avec le format exact : [Source N - Document: NomDuDocument, Page: NuméroPage]
  où N est le numéro de la source dans la liste ci-dessus.
- Structure ta réponse de manière claire : commence par la réponse directe,
  puis les détails techniques, puis les sources.
- Ne pose pas de diagnostic définitif : utilise des formulations comme
  "Selon la documentation...", "Le manuel indique...", "Il est possible que...".
- Si plusieurs sources apportent des informations complémentaires, synthétise-les.
- Adapte le niveau de détail technique au contexte de maintenance industrielle.

RÉPONSE :"""


# =============================================================================
# TICKET_PREPARATION_PROMPT — Rédaction d'un brouillon de ticket de maintenance
# =============================================================================

TICKET_PREPARATION_PROMPT: str = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance industrielle.

Ta tâche est de préparer un BROUILLON de ticket de maintenance à partir des informations fournies.
Ce brouillon DOIT être validé par un technicien ou un responsable avant toute création effective.

INFORMATIONS SUR L'ÉQUIPEMENT :
--------------------------------
{equipment_info}

DESCRIPTION DU PROBLÈME SIGNALÉ :
-----------------------------------
{problem_description}

SOURCES DOCUMENTAIRES PERTINENTES :
--------------------------------------
{relevant_sources}

INSTRUCTIONS :
- Génère un brouillon de ticket de maintenance structuré.
- Inclue : titre, priorité estimée (Critique/Haute/Moyenne/Basse), description détaillée,
  équipement concerné, actions suggérées selon la documentation, et sources de référence.
- Rappelle EXPLICITEMENT que ce brouillon doit être confirmé par un humain avant création.
- N'invente aucune information absente du contexte fourni.
- Si aucune source documentaire n'est fournie, écris "Aucune source documentaire indexée" et n'invente JAMAIS de nom de document ou de numéro de page.
- Utilise un vocabulaire technique approprié à la maintenance industrielle.

FORMAT DE SORTIE :
==================
TITRE : [Titre court et descriptif du ticket]
PRIORITÉ ESTIMÉE : [Critique / Haute / Moyenne / Basse]
ÉQUIPEMENT : [Référence et localisation de l'équipement]
DESCRIPTION : [Description détaillée du problème]
ACTIONS SUGGÉRÉES :
  1. [Première action selon la documentation]
  2. [Deuxième action selon la documentation]
  ...
SOURCES DE RÉFÉRENCE : [Citations documentaires]
==================
⚠️  IMPORTANT : Ce brouillon est généré automatiquement. Un technicien habilité doit valider
et confirmer la création de ce ticket avant toute intervention.

BROUILLON DE TICKET :"""


# =============================================================================
# IMAGE_ANALYSIS_PROMPT — Analyse d'image technique par le VLM
# =============================================================================

IMAGE_ANALYSIS_PROMPT: str = """Tu es un expert en maintenance industrielle analysant une image d'équipement technique.

OBJECTIF :
Analyse cette image avec rigueur et méthodologie pour identifier :
1. La nature et l'état général de l'équipement visible
2. Tout défaut, anomalie, dégradation ou signe d'usure visibles
3. Les zones nécessitant une attention immédiate ou une inspection approfondie
4. Les risques potentiels pour la sécurité des personnes et des équipements

INSTRUCTIONS D'ANALYSE :
- Décris objectivement ce que tu vois : type d'équipement, composants identifiables,
  matériaux, état de surface, couleurs anormales, déformations, fuites, corrosion, etc.
- Pour chaque anomalie détectée, précise : sa localisation, sa nature, son niveau de gravité estimé.
- Référence les standards de maintenance industrielle pertinents si applicable
  (ISO 13849, NF EN 13460, etc.).
- Utilise un vocabulaire technique précis (ex : "cavitation", "délaminage",
  "grippage", "fatigue de surface", "oxydation galvanique").
- Distingue clairement ce que tu observes directement de tes interprétations.
- Si l'image n'est pas suffisamment nette ou si l'angle ne permet pas une analyse complète,
  indique-le explicitement.

FORMAT DE RÉPONSE :
1. DESCRIPTION GÉNÉRALE : Vue d'ensemble de l'équipement
2. ANOMALIES DÉTECTÉES : Liste des problèmes identifiés (ou "Aucune anomalie visible")
3. ZONES D'ATTENTION : Parties nécessitant surveillance ou intervention
4. RECOMMANDATIONS : Actions suggérées basées sur l'observation visuelle
5. LIMITES DE L'ANALYSE : Ce que l'image ne permet pas de confirmer

ANALYSE :"""


# =============================================================================
# UNCERTAINTY_RESPONSE — Réponse standard en cas de faible confiance
# =============================================================================

UNCERTAINTY_RESPONSE: str = """Je ne dispose pas d'informations suffisamment fiables dans les documents disponibles \
pour répondre avec certitude à votre question.

Voici ce que je peux vous indiquer à partir des sources partiellement pertinentes :
{partial_info}

Pour obtenir une réponse précise, je vous recommande :
1. Consulter directement le manuel de l'équipement concerné (référence : {equipment_ref})
2. Contacter le service technique du fabricant ou un technicien spécialisé
3. Vérifier si un document plus récent est disponible dans la GMAO

⚠️  Ne prenez aucune décision d'intervention basée uniquement sur cette réponse partielle."""


# =============================================================================
# CLARIFICATION_PROMPT — Demande de précisions pour une question ambiguë
# =============================================================================

CLARIFICATION_PROMPT: str = """Votre question nécessite quelques précisions pour que je puisse \
vous apporter une réponse documentée et pertinente.

QUESTION REÇUE : "{question}"

POINTS À CLARIFIER :
{clarification_points}

Pour vous aider efficacement, pourriez-vous préciser :
1. La référence ou le modèle exact de l'équipement concerné (ex : pompe centrifuge XYZ-100)
2. Le symptôme observé de manière précise (bruit, vibration, température, débit...)
3. Les conditions d'exploitation au moment du problème (charge, température ambiante...)
4. Les interventions récentes effectuées sur cet équipement

Ces informations me permettront de rechercher dans la documentation la plus adaptée à votre situation."""


# =============================================================================
# FONCTIONS UTILITAIRES DE FORMATAGE
# =============================================================================

def format_rag_prompt(context: str, question: str) -> str:
    """
    Formate le prompt RAG avec le contexte récupéré et la question de l'utilisateur.

    Args:
        context:  Bloc de texte contenant les chunks documentaires récupérés,
                  typiquement générés par `format_context_items()` du retriever.
        question: Question posée par le technicien.

    Returns:
        Prompt complet prêt à être envoyé au LLM.

    Example:
        >>> prompt = format_rag_prompt(
        ...     context="[Source 1] Manuel pompe XYZ...",
        ...     question="Quelle est la pression nominale de la pompe XYZ ?"
        ... )
    """
    if not context or not context.strip():
        context = "Aucun document pertinent trouvé pour cette requête."

    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    return RAG_PROMPT.format(context=context.strip(), question=question.strip())


def format_ticket_prompt(
    equipment_info: dict[str, Any],
    problem: str,
    sources: list[str],
) -> str:
    """
    Formate le prompt de préparation de ticket de maintenance.

    Args:
        equipment_info: Dictionnaire contenant les informations sur l'équipement.
                        Clés attendues : "name", "reference", "location", "last_maintenance".
                        Les clés manquantes sont remplacées par "Non renseigné".
        problem:        Description du problème signalé par le technicien.
        sources:        Liste de chaînes représentant les sources documentaires pertinentes.

    Returns:
        Prompt complet pour la génération du brouillon de ticket.

    Example:
        >>> prompt = format_ticket_prompt(
        ...     equipment_info={"name": "Pompe centrifuge", "reference": "PC-001"},
        ...     problem="Vibrations anormales et fuite au niveau du joint mécanique",
        ...     sources=["Manuel PC-001, page 42", "Procédure de remplacement joint, page 7"]
        ... )
    """
    if not problem or not problem.strip():
        raise ValueError("La description du problème ne peut pas être vide.")

    # Normalisation des informations équipement
    _defaults = {
        "name": "Non renseigné",
        "reference": "Non renseigné",
        "location": "Non renseigné",
        "last_maintenance": "Non renseigné",
        "criticality": "Non renseigné",
    }
    _info = {**_defaults, **equipment_info}

    equipment_block = (
        "Nom        : {name}\n"
        "Référence  : {reference}\n"
        "Localisation : {location}\n"
        "Dernière maintenance : {last_maintenance}\n"
        "Criticité  : {criticality}"
    ).format(**_info)

    sources_block = (
        "\n".join(f"  [{i + 1}] {src}" for i, src in enumerate(sources))
        if sources
        else "  Aucune source documentaire trouvée."
    )

    return TICKET_PREPARATION_PROMPT.format(
        equipment_info=equipment_block,
        problem_description=problem.strip(),
        relevant_sources=sources_block,
    )


def format_uncertainty_response(partial_info: str = "", equipment_ref: str = "Non spécifiée") -> str:
    """
    Formate la réponse standard en cas d'incertitude élevée.

    Args:
        partial_info:   Informations partielles disponibles (peut être vide).
        equipment_ref:  Référence de l'équipement pour orienter la recherche manuelle.

    Returns:
        Message d'incertitude formaté.
    """
    _partial = partial_info.strip() if partial_info.strip() else "Aucune information partielle disponible."
    return UNCERTAINTY_RESPONSE.format(
        partial_info=_partial,
        equipment_ref=equipment_ref,
    )


def format_clarification_prompt(question: str, clarification_points: list[str]) -> str:
    """
    Formate le prompt de demande de clarification.

    Args:
        question:             Question originale ambiguë de l'utilisateur.
        clarification_points: Liste des points nécessitant une précision.

    Returns:
        Message de demande de clarification formaté.
    """
    if not clarification_points:
        clarification_points = ["La question manque de contexte sur l'équipement concerné."]

    points_block = "\n".join(f"  - {p}" for p in clarification_points)
    return CLARIFICATION_PROMPT.format(
        question=question.strip(),
        clarification_points=points_block,
    )
