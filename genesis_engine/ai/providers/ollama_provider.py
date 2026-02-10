"""
Ollama Provider — Local-First LLM Integration
================================================

Implements the ``AIProvider`` interface to call a local Ollama instance,
enabling 100% offline operation without any API keys.

Supported models:
- **llama3.1:8b** (default) — general-purpose reasoning
- **nomic-embed-text** — embedding generation (future)

The provider communicates with Ollama via its REST API (default:
``http://localhost:11434``). If Ollama is unreachable, the provider
gracefully falls back to the deterministic ``LocalProvider``.

Design Principle:
    This provider is the *default* when no external API key is detected,
    ensuring the Genesis Engine can operate entirely offline.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from genesis_engine.core.ai_provider import (
    AIProvider,
    Candidate,
    LocalProvider,
    Perspective,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    """Configuration for the Ollama provider.

    Sprint 11.5 updates:
    - ``timeout`` increased to 600s for 100-round simulations on 12B models.
    - ``num_ctx`` set to 128_000 for large legislative PDF ingestion.
    - ``role`` selects task-specific model from SOCRATIC_MODEL_MAP.
    """

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout: int = 600  # seconds — increased for 100-round sims on 12B models
    temperature: float = 0.7
    max_tokens: int = 1024
    num_ctx: int = 128_000  # 128K context for legislative PDF ingestion
    role: str = ""  # Socratic Role (thinker, builder, sentry)

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/api/tags"


# ---------------------------------------------------------------------------
# Prompt templates for each perspective
# ---------------------------------------------------------------------------

_PERSPECTIVE_PROMPTS: dict[Perspective, str] = {
    Perspective.CAUSALITY: (
        "You are an ethical systems analyst using CAUSAL reasoning.\n"
        "Analyse the following system description and its relationships "
        "(morphisms). Trace the cause-effect chains that lead to harm.\n"
        "Identify the root cause of any disharmony.\n\n"
        "System description: {source_text}\n\n"
        "Relationships (morphisms):\n{morphisms_text}\n\n"
        "Entities (objects):\n{objects_text}\n\n"
        "Respond in this JSON format:\n"
        '{{"content": "<your causal analysis>", '
        '"reasoning": "<your reasoning process>", '
        '"confidence": <0.0-1.0>}}'
    ),
    Perspective.CONTRADICTION: (
        "You are an ethical systems analyst using CONTRADICTION detection.\n"
        "Analyse the following system and find tensions between stated "
        "values and actual behaviour (morphisms).\n"
        "Identify where the system contradicts its own principles.\n\n"
        "System description: {source_text}\n\n"
        "Relationships (morphisms):\n{morphisms_text}\n\n"
        "Entities (objects):\n{objects_text}\n\n"
        "Respond in this JSON format:\n"
        '{{"content": "<your contradiction analysis>", '
        '"reasoning": "<your reasoning process>", '
        '"confidence": <0.0-1.0>}}'
    ),
    Perspective.ANALOGY: (
        "You are an ethical systems analyst using ANALOGICAL reasoning.\n"
        "Map the following system to known ethical patterns or historical "
        "analogues. Identify what archetype this system resembles.\n\n"
        "System description: {source_text}\n\n"
        "Relationships (morphisms):\n{morphisms_text}\n\n"
        "Entities (objects):\n{objects_text}\n\n"
        "Respond in this JSON format:\n"
        '{{"content": "<your analogical analysis>", '
        '"reasoning": "<your reasoning process>", '
        '"confidence": <0.0-1.0>}}'
    ),
}


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

class OllamaProvider(AIProvider):
    """AI provider using a local Ollama instance for LLM inference.

    Falls back to ``LocalProvider`` (deterministic) if:
    - Ollama is not running / unreachable
    - The configured model is not available
    - Any request-level error occurs

    This ensures the Genesis Engine always produces candidates, even
    in fully offline environments without Ollama installed.
    """

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()
        self._fallback = LocalProvider()
        self._available: bool | None = None  # lazy-checked

    @property
    def provider_name(self) -> str:
        if self._is_available():
            return f"OllamaProvider ({self._config.model})"
        return f"OllamaProvider (fallback→LocalProvider)"

    @property
    def config(self) -> OllamaConfig:
        return self._config

    # -- AIProvider interface -----------------------------------------------

    def generate_candidates(
        self,
        context: dict[str, Any],
        perspectives: list[Perspective] | None = None,
    ) -> list[Candidate]:
        """Generate reasoning candidates via Ollama or fallback.

        Parameters
        ----------
        context : dict
            Must contain ``source_text``, ``morphisms``, ``objects``.
        perspectives : list[Perspective] | None
            Which perspectives to generate. Defaults to all three.

        Returns
        -------
        list[Candidate]
        """
        perspectives = perspectives or list(Perspective)

        if not self._is_available():
            logger.info(
                "Ollama not available at %s — using LocalProvider fallback.",
                self._config.base_url,
            )
            return self._fallback.generate_candidates(context, perspectives)

        candidates: list[Candidate] = []
        for perspective in perspectives:
            candidate = self._generate_one(context, perspective)
            candidates.append(candidate)

        return candidates

    # -- Ollama communication -----------------------------------------------

    def _is_available(self) -> bool:
        """Check if Ollama is reachable (cached after first check)."""
        if self._available is not None:
            return self._available

        try:
            req = urllib.request.Request(
                self._config.health_url,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            self._available = False

        return self._available

    def reset_availability(self) -> None:
        """Force re-check of Ollama availability on next call."""
        self._available = None

    def _generate_one(
        self, context: dict[str, Any], perspective: Perspective,
    ) -> Candidate:
        """Generate a single candidate for one perspective via Ollama."""
        prompt_template = _PERSPECTIVE_PROMPTS[perspective]

        # Format morphisms and objects for the prompt
        morphisms = context.get("morphisms", [])
        objects = context.get("objects", [])

        morphisms_text = "\n".join(
            f"  - {m.get('label', '?')}: {m.get('source', '?')} → {m.get('target', '?')} "
            f"[tags: {', '.join(m.get('tags', []))}]"
            for m in morphisms
        ) or "  (none)"

        objects_text = "\n".join(
            f"  - {o.get('label', '?')} [tags: {', '.join(o.get('tags', []))}]"
            for o in objects
        ) or "  (none)"

        prompt = prompt_template.format(
            source_text=context.get("source_text", ""),
            morphisms_text=morphisms_text,
            objects_text=objects_text,
        )

        try:
            response_text = self._call_ollama(prompt)
            return self._parse_response(response_text, perspective)
        except Exception as exc:
            logger.warning(
                "Ollama request failed for %s perspective: %s. "
                "Falling back to LocalProvider for this candidate.",
                perspective.value, exc,
            )
            return self._fallback.generate_candidates(
                context, [perspective],
            )[0]

    def _call_ollama(self, prompt: str) -> str:
        """Send a generate request to Ollama and return the response text."""
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
                "num_ctx": self._config.num_ctx,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._config.generate_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")

    def _parse_response(self, text: str, perspective: Perspective) -> Candidate:
        """Parse the LLM response into a Candidate.

        Attempts to parse JSON; falls back to treating the entire
        response as content if JSON parsing fails.
        """
        # Try to extract JSON from the response
        try:
            # Find the JSON object in the response
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])

            return Candidate(
                perspective=perspective,
                content=data.get("content", text),
                reasoning=data.get("reasoning", ""),
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
                metadata={"provider": "ollama", "model": self._config.model},
            )
        except (ValueError, json.JSONDecodeError):
            # If we can't parse JSON, use the raw text
            return Candidate(
                perspective=perspective,
                content=text.strip() or "No response generated.",
                reasoning="Raw LLM output (JSON parsing failed).",
                confidence=0.5,
                metadata={"provider": "ollama", "model": self._config.model, "raw": True},
            )


# ---------------------------------------------------------------------------
# Provider factory — selects the best available provider
# ---------------------------------------------------------------------------

def get_default_provider() -> AIProvider:
    """Return the best available AI provider.

    Selection logic:
    1. If an external API key is detected (ANTHROPIC_API_KEY or
       OPENAI_API_KEY), return LocalProvider (future: cloud provider).
    2. Otherwise, return OllamaProvider — which itself falls back to
       LocalProvider if Ollama is not running.

    This ensures 100% offline operation when no API key is configured.
    """
    has_api_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if has_api_key:
        # Future: return AnthropicProvider() or OpenAIProvider()
        # For now, still prefer local-first operation
        logger.info("API key detected, but using local-first provider.")

    # Default to Ollama (which falls back to LocalProvider if unavailable)
    return OllamaProvider()
