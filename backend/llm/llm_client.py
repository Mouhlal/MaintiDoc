"""
llm_client.py — Client LLM pour MaintiDoc (Ollama + fallback optionnel Gemini)
================================================================================

Ce module expose un client HTTP synchrone vers Ollama pour la génération de texte.
Il est conçu pour être utilisé comme singleton via `llm_client`.

Architecture :
    LLMClient
    ├── generate()              ← appel principal (Ollama /api/generate)
    ├── generate_with_context() ← raccourci RAG (formate le prompt puis génère)
    └── is_available()          ← vérification de santé Ollama

Retry : tenacity (3 tentatives, backoff exponentiel 1→4s).
Transport : httpx (synchrone) — pas de dépendance à asyncio pour la simplicité.

Usage :
    from backend.llm.llm_client import llm_client

    response = llm_client.generate(
        prompt="Qu'est-ce que la cavitation ?",
        system="Tu es un assistant maintenance."
    )
    print(response.content)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings
from backend.llm.prompts import SYSTEM_PROMPT, format_rag_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS PERSONNALISÉES
# =============================================================================

class LLMError(Exception):
    """Erreur générique levée par LLMClient."""


class LLMConnectionError(LLMError):
    """Impossible de joindre le serveur Ollama."""


class LLMTimeoutError(LLMError):
    """Le serveur Ollama n'a pas répondu dans les délais."""


class LLMParseError(LLMError):
    """La réponse Ollama n'a pas pu être parsée."""


# =============================================================================
# DATACLASS DE RÉPONSE
# =============================================================================

@dataclass
class LLMResponse:
    """
    Résultat structuré d'un appel au LLM.

    Attributes:
        content:     Texte généré par le modèle.
        model:       Nom du modèle Ollama utilisé.
        tokens_used: Nombre de tokens de sortie (eval_count dans la réponse Ollama).
        latency_ms:  Temps de réponse total en millisecondes.
    """
    content: str
    model: str
    tokens_used: int
    latency_ms: float

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.content[:80].replace("\n", " ")
        return (
            f"LLMResponse(model={self.model!r}, tokens={self.tokens_used}, "
            f"latency={self.latency_ms:.0f}ms, content={preview!r}...)"
        )


# =============================================================================
# CLIENT LLM PRINCIPAL
# =============================================================================

class LLMClient:
    """
    Client HTTP synchrone vers Ollama pour la génération de texte.

    Le client gère :
    - La construction du prompt complet (system + user)
    - L'appel HTTP avec timeout configuré
    - Le retry automatique via tenacity
    - La mesure de latence
    - La gestion d'erreurs avec exceptions typées

    Example:
        client = LLMClient()
        if client.is_available():
            resp = client.generate("Explique la procédure de purge.")
            print(resp.content)
    """

    # Timeouts HTTP en secondes
    _CONNECT_TIMEOUT: float = 10.0
    _READ_TIMEOUT: float = 120.0  # Les LLM locaux peuvent être lents

    def __init__(self) -> None:
        """Initialise le client avec les paramètres de configuration."""
        self._model: str = settings.llm_model
        self._base_url: str = settings.ollama_base_url.rstrip("/")
        self._temperature: float = settings.llm_temperature
        self._max_tokens: int = settings.llm_max_tokens

        logger.info(
            "LLMClient initialisé — modèle: %s, base_url: %s",
            self._model,
            self._base_url,
        )

    # -------------------------------------------------------------------------
    # MÉTHODE PRINCIPALE : generate()
    # -------------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((LLMConnectionError, LLMTimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Génère une réponse texte via l'API Ollama /api/generate.

        Args:
            prompt:      Texte d'entrée (prompt utilisateur ou prompt RAG complet).
            system:      Prompt système (instructions de rôle). Si None, utilise
                         SYSTEM_PROMPT de MaintiDoc.
            temperature: Température de génération (0.0 = déterministe, 1.0 = créatif).
                         Si None, utilise settings.llm_temperature.
            max_tokens:  Nombre maximum de tokens à générer.
                         Si None, utilise settings.llm_max_tokens.

        Returns:
            LLMResponse avec le texte généré et les méta-données.

        Raises:
            LLMConnectionError: Ollama inaccessible.
            LLMTimeoutError:    Pas de réponse dans le délai imparti.
            LLMParseError:      Réponse Ollama malformée.
            LLMError:           Toute autre erreur LLM.
        """
        _system = system if system is not None else SYSTEM_PROMPT
        _temperature = temperature if temperature is not None else self._temperature
        _max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        # Ollama attend un prompt unique — on préfixe avec le system prompt
        full_prompt = f"{_system}\n\n{prompt}" if _system else prompt

        payload: dict = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": _temperature,
                "num_predict": _max_tokens,
            },
        }

        url = f"{self._base_url}/api/generate"
        t_start = time.monotonic()

        try:
            with httpx.Client(
                timeout=httpx.Timeout(
                    connect=self._CONNECT_TIMEOUT,
                    read=self._READ_TIMEOUT,
                    write=10.0,
                    pool=5.0,
                )
            ) as client:
                http_response = client.post(url, json=payload)
                http_response.raise_for_status()

        except httpx.ConnectError as exc:
            logger.error("Impossible de joindre Ollama à %s : %s", url, exc)
            raise LLMConnectionError(
                f"Impossible de joindre Ollama à {url}. "
                "Vérifiez que le service Ollama est démarré."
            ) from exc

        except httpx.TimeoutException as exc:
            logger.error("Timeout lors de l'appel Ollama (%s) : %s", url, exc)
            raise LLMTimeoutError(
                f"Ollama n'a pas répondu dans les {self._READ_TIMEOUT}s imparties."
            ) from exc

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Erreur HTTP %d depuis Ollama : %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise LLMError(
                f"Ollama a retourné une erreur HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        latency_ms = (time.monotonic() - t_start) * 1000.0

        # Parsing de la réponse JSON
        try:
            data: dict = http_response.json()
            content: str = data["response"]
            tokens_used: int = data.get("eval_count", 0)
        except (KeyError, ValueError) as exc:
            logger.error("Impossible de parser la réponse Ollama : %s", http_response.text[:500])
            raise LLMParseError(
                f"La réponse Ollama n'a pas pu être parsée : {exc}"
            ) from exc

        logger.info(
            "LLM generate OK — tokens: %d, latency: %.0fms",
            tokens_used,
            latency_ms,
        )

        return LLMResponse(
            content=content.strip(),
            model=self._model,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    # RACCOURCI RAG : generate_with_context()
    # -------------------------------------------------------------------------

    def generate_with_context(self, question: str, context: str) -> LLMResponse:
        """
        Formate le prompt RAG (contexte + question) puis appelle generate().

        C'est le point d'entrée principal pour le pipeline RAG de MaintiDoc.

        Args:
            question: Question posée par le technicien.
            context:  Contexte documentaire récupéré par le retriever.

        Returns:
            LLMResponse avec la réponse sourcée.

        Raises:
            ValueError:   Si question ou context sont vides.
            LLMError:     En cas d'erreur de génération.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")
        if not context:
            context = "Aucun document pertinent trouvé."

        rag_prompt = format_rag_prompt(context=context, question=question)

        logger.debug(
            "generate_with_context — question: %s... | context_len: %d chars",
            question[:60],
            len(context),
        )

        # On passe system=None pour utiliser SYSTEM_PROMPT par défaut
        return self.generate(prompt=rag_prompt, system=None)

    # -------------------------------------------------------------------------
    # VÉRIFICATION DE SANTÉ : is_available()
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Vérifie si le serveur Ollama est accessible et opérationnel.

        Effectue un GET sur /api/tags (endpoint de listing des modèles).
        Ne lève pas d'exception : retourne simplement False si indisponible.

        Returns:
            True si Ollama répond avec HTTP 200, False sinon.
        """
        url = f"{self._base_url}/api/tags"
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=10.0)) as client:
                response = client.get(url)
                available = response.status_code == 200
                if not available:
                    logger.warning(
                        "Ollama /api/tags a retourné HTTP %d", response.status_code
                    )
                return available
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("Ollama inaccessible (%s) : %s", url, exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur inattendue lors du health-check Ollama : %s", exc)
            return False

    def get_available_models(self) -> list[str]:
        """
        Retourne la liste des modèles disponibles sur Ollama.

        Returns:
            Liste de noms de modèles (ex: ["llama3.2:3b", "llava:7b"]).
            Liste vide si Ollama est inaccessible.
        """
        url = f"{self._base_url}/api/tags"
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=10.0)) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.debug("Modèles Ollama disponibles : %s", models)
                return models
        except Exception as exc:  # noqa: BLE001
            logger.warning("Impossible de récupérer les modèles Ollama : %s", exc)
            return []


# =============================================================================
# SINGLETON
# =============================================================================

#: Instance unique du client LLM — à importer directement dans les autres modules.
#: Usage : from backend.llm.llm_client import llm_client
llm_client: LLMClient = LLMClient()
