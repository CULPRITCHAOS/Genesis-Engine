"""
AI Provider Interface — Ollama-Ready Abstraction Layer

Defines the ``AIProvider`` protocol and a deterministic ``LocalProvider``
for Sprint 4.  The interface is designed so that an ``OllamaProvider``
(or any LLM backend) can be dropped in without changing downstream code.

Every provider must implement ``generate_candidates``, which takes a
context dict and returns a list of candidate dicts with perspective
metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Perspective types used by the multi-perspective Dream Engine
# ---------------------------------------------------------------------------

class Perspective(Enum):
    """Manus-style reasoning perspectives for candidate generation."""

    CAUSALITY = "causality"          # A caused B
    CONTRADICTION = "contradiction"  # A disproves B
    ANALOGY = "analogy"              # A is like B


# ---------------------------------------------------------------------------
# Candidate — a single generated response with metadata
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single reasoning candidate produced by an AI provider."""

    perspective: Perspective
    content: str
    reasoning: str
    confidence: float = 0.5  # 0.0 – 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "perspective": self.perspective.value,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AIProvider Protocol
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """Abstract base for AI inference providers.

    Implementations:
    - ``LocalProvider``   — deterministic, rule-based (Sprint 4)
    - ``OllamaProvider``  — local LLM via Ollama API (future)
    - ``AnthropicProvider`` — cloud LLM via Anthropic API (future)
    """

    @abstractmethod
    def generate_candidates(
        self,
        context: dict[str, Any],
        perspectives: list[Perspective] | None = None,
    ) -> list[Candidate]:
        """Generate reasoning candidates from *context*.

        Parameters
        ----------
        context : dict
            Must contain at minimum ``source_text`` and ``morphisms``.
        perspectives : list[Perspective] | None
            Which perspectives to generate.  Defaults to all three.

        Returns
        -------
        list[Candidate]
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this provider."""
        ...


# ---------------------------------------------------------------------------
# LocalProvider — deterministic rule-based candidate generation
# ---------------------------------------------------------------------------

class LocalProvider(AIProvider):
    """Deterministic candidate generator using neuro-symbolic rules.

    This provider analyses morphism tags and graph structure to produce
    candidates from three perspectives without requiring an LLM.
    """

    @property
    def provider_name(self) -> str:
        return "LocalProvider (deterministic)"

    def generate_candidates(
        self,
        context: dict[str, Any],
        perspectives: list[Perspective] | None = None,
    ) -> list[Candidate]:
        perspectives = perspectives or list(Perspective)
        source_text = context.get("source_text", "")
        morphisms = context.get("morphisms", [])
        objects = context.get("objects", [])

        candidates: list[Candidate] = []
        for p in perspectives:
            if p == Perspective.CAUSALITY:
                candidates.append(self._causality_candidate(source_text, morphisms, objects))
            elif p == Perspective.CONTRADICTION:
                candidates.append(self._contradiction_candidate(source_text, morphisms, objects))
            elif p == Perspective.ANALOGY:
                candidates.append(self._analogy_candidate(source_text, morphisms, objects))

        return candidates

    # -- perspective generators ---------------------------------------------

    def _causality_candidate(
        self, source_text: str, morphisms: list[dict], objects: list[dict],
    ) -> Candidate:
        """Causality: trace cause-effect chains in the morphism graph."""
        harmful = [m for m in morphisms if self._is_harmful(m)]
        if harmful:
            chain_parts = []
            for m in harmful:
                src = self._label_for_id(m.get("source", ""), objects)
                tgt = self._label_for_id(m.get("target", ""), objects)
                chain_parts.append(
                    f"{src}'s {m.get('label', '?')} causes harm to {tgt}"
                )
            chain = "; ".join(chain_parts)
            reasoning = (
                f"Causal analysis: {chain}. "
                f"Root cause: structural power imbalance where actors "
                f"extract value from vulnerable entities."
            )
            content = (
                f"The disharmony originates from a causal chain: {chain}. "
                f"Breaking this chain requires transforming the power "
                f"relationship at its root."
            )
            confidence = 0.85
        else:
            reasoning = "No harmful causal chains detected in the graph."
            content = "The system exhibits healthy causal relationships."
            confidence = 0.9

        return Candidate(
            perspective=Perspective.CAUSALITY,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _contradiction_candidate(
        self, source_text: str, morphisms: list[dict], objects: list[dict],
    ) -> Candidate:
        """Contradiction: find tensions between stated goals and morphisms."""
        protective_objects = [
            o for o in objects
            if "protective" in o.get("tags", []) or "vulnerable" in o.get("tags", [])
        ]
        harmful = [m for m in morphisms if self._is_harmful(m)]

        if harmful and protective_objects:
            protected_labels = [o.get("label", "?") for o in protective_objects]
            harmful_labels = [m.get("label", "?") for m in harmful]
            reasoning = (
                f"Contradiction detected: the system claims to value "
                f"{', '.join(protected_labels)} but simultaneously "
                f"employs {', '.join(harmful_labels)} morphisms that "
                f"undermine those very values."
            )
            content = (
                f"A fundamental contradiction exists: "
                f"{', '.join(harmful_labels)} directly contradicts the "
                f"stated commitment to {', '.join(protected_labels)}. "
                f"The system cannot coherently maintain both positions."
            )
            confidence = 0.9
        elif harmful:
            reasoning = "Harmful morphisms present but no contradiction with stated values."
            content = "The system is openly extractive without internal contradiction."
            confidence = 0.7
        else:
            reasoning = "No contradictions detected between values and morphisms."
            content = "The system is internally consistent and aligned."
            confidence = 0.9

        return Candidate(
            perspective=Perspective.CONTRADICTION,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _analogy_candidate(
        self, source_text: str, morphisms: list[dict], objects: list[dict],
    ) -> Candidate:
        """Analogy: map the situation to known ethical patterns."""
        harmful = [m for m in morphisms if self._is_harmful(m)]

        _ANALOGIES = {
            "extraction": (
                "resource colony",
                "Like a colony extracting resources from indigenous land, "
                "the system takes value from vulnerable entities without "
                "reciprocity or consent."
            ),
            "exploitation": (
                "indentured labour",
                "Like an indentured system that traps workers in debt, "
                "the system exploits power asymmetry for one-sided benefit."
            ),
            "coercion": (
                "panopticon",
                "Like a panopticon where the mere possibility of being "
                "watched controls behaviour, the system uses coercion to "
                "suppress autonomy."
            ),
            "neglect": (
                "abandoned commons",
                "Like a commons left without stewardship, vulnerable "
                "entities are neglected while value is extracted elsewhere."
            ),
        }

        if harmful:
            all_tags: set[str] = set()
            for m in harmful:
                all_tags.update(t.lower() for t in m.get("tags", []))

            matched_tag = None
            for tag in ("exploitation", "extraction", "coercion", "neglect"):
                if tag in all_tags:
                    matched_tag = tag
                    break

            if matched_tag and matched_tag in _ANALOGIES:
                archetype, description = _ANALOGIES[matched_tag]
                reasoning = f"Analogical mapping: system resembles a '{archetype}' pattern."
                content = description
                confidence = 0.8
            else:
                reasoning = "Harmful patterns present but no strong analogy found."
                content = "The system exhibits disharmony without a clear historical analogue."
                confidence = 0.6
        else:
            reasoning = "No harmful patterns; system resembles a healthy ecosystem."
            content = (
                "Like a healthy forest ecosystem where each organism "
                "contributes to the whole, the system's relationships "
                "are mutually beneficial."
            )
            confidence = 0.85

        return Candidate(
            perspective=Perspective.ANALOGY,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _is_harmful(morphism: dict[str, Any]) -> bool:
        harmful_tags = {"extraction", "exploitation", "coercion", "neglect", "division"}
        return bool(set(t.lower() for t in morphism.get("tags", [])) & harmful_tags)

    @staticmethod
    def _label_for_id(obj_id: str, objects: list[dict]) -> str:
        for o in objects:
            if o.get("id") == obj_id:
                return o.get("label", obj_id)
        return obj_id
