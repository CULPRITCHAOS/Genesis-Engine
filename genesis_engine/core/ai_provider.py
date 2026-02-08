"""
AI Provider Interface — Ollama-First Local Sovereignty Layer

Defines the ``AIProvider`` protocol, a deterministic ``LocalProvider``,
an ``OllamaProvider`` for local-first LLM inference (Sprint 9), and
an ``OffloadSkeleton`` for anonymized 100-round simulation offloading
that protects the local soul file from hardware bottlenecks.

The system defaults to Ollama when available, falling back to the
deterministic LocalProvider when Ollama is not reachable.

Every provider must implement ``generate_candidates``, which takes a
context dict and returns a list of candidate dicts with perspective
metadata.
"""

from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------------------
# OllamaProvider — Local-first LLM inference (Sprint 9)
# ---------------------------------------------------------------------------

class OllamaProvider(AIProvider):
    """Local-first LLM provider via Ollama API.

    Defaults to the ``llama3.2`` model on ``localhost:11434``.  Falls back
    to the deterministic ``LocalProvider`` when Ollama is not reachable.

    Parameters
    ----------
    model : str
        Ollama model name (default ``llama3.2``).
    base_url : str
        Ollama API base URL.
    timeout : float
        Connection timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._fallback = LocalProvider()
        self._available: bool | None = None

    @property
    def provider_name(self) -> str:
        return f"OllamaProvider ({self._model})"

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self._timeout):
                self._available = True
        except Exception:
            self._available = False
        return self._available

    def generate_candidates(
        self,
        context: dict[str, Any],
        perspectives: list[Perspective] | None = None,
    ) -> list[Candidate]:
        """Generate candidates via Ollama, falling back to LocalProvider.

        If Ollama is not available, transparently delegates to the
        deterministic LocalProvider to ensure the system always works.
        """
        if not self.is_available():
            return self._fallback.generate_candidates(context, perspectives)

        perspectives = perspectives or list(Perspective)
        source_text = context.get("source_text", "")
        morphisms = context.get("morphisms", [])

        candidates: list[Candidate] = []
        for p in perspectives:
            prompt = self._build_prompt(p, source_text, morphisms)
            try:
                response = self._query_ollama(prompt)
                candidates.append(Candidate(
                    perspective=p,
                    content=response,
                    reasoning=f"Generated via Ollama ({self._model})",
                    confidence=0.75,
                    metadata={"provider": "ollama", "model": self._model},
                ))
            except Exception:
                # Fall back to deterministic for this perspective
                fallback_candidates = self._fallback.generate_candidates(
                    context, [p],
                )
                candidates.extend(fallback_candidates)

        return candidates

    def _build_prompt(
        self,
        perspective: Perspective,
        source_text: str,
        morphisms: list[dict],
    ) -> str:
        """Build a perspective-specific prompt for Ollama."""
        morph_desc = "; ".join(
            f"{m.get('label', '?')} ({m.get('source', '?')} -> {m.get('target', '?')})"
            for m in morphisms[:5]
        )
        base = f"Analyze this system: {source_text[:200]}. Morphisms: {morph_desc}."

        if perspective == Perspective.CAUSALITY:
            return f"{base} What are the cause-effect chains that create harm?"
        elif perspective == Perspective.CONTRADICTION:
            return f"{base} What contradictions exist between stated values and actual mechanisms?"
        else:
            return f"{base} What historical or ethical patterns does this resemble?"

    def _query_ollama(self, prompt: str) -> str:
        """Send a query to the Ollama API and return the response text."""
        import urllib.request

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")


# ---------------------------------------------------------------------------
# OffloadSkeleton — Anonymized simulation offloading (Sprint 9)
# ---------------------------------------------------------------------------

@dataclass
class OffloadPacket:
    """An anonymized packet for offloading 100-round simulations.

    All identifying information is stripped — no soul IDs, no user data,
    no timestamps.  Only the structural parameters needed for the
    simulation are included.
    """

    morphism_tag_histogram: dict[str, int]
    object_count: int
    morphism_count: int
    vulnerable_node_count: int
    extraction_ratio: float
    simulation_rounds: int = 100
    packet_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "offloadPacket": {
                "morphismTagHistogram": self.morphism_tag_histogram,
                "objectCount": self.object_count,
                "morphismCount": self.morphism_count,
                "vulnerableNodeCount": self.vulnerable_node_count,
                "extractionRatio": round(self.extraction_ratio, 4),
                "simulationRounds": self.simulation_rounds,
                "packetHash": self.packet_hash,
            }
        }


class OffloadSkeleton:
    """Lightweight anonymized offload for 100-round simulations.

    Protects the local soul file from hardware bottlenecks by packaging
    only anonymized structural parameters (tag histograms, node counts)
    for external simulation.  No PII, no soul IDs, no timestamps.

    The offload produces an ``OffloadPacket`` that could be sent to
    a remote simulation service.  Results are reintegrated locally.

    Usage::

        skeleton = OffloadSkeleton()
        packet = skeleton.prepare(graph)
        # Send packet to remote service...
        # result = remote_service.simulate(packet)
        # skeleton.reintegrate(result, soul)
    """

    @staticmethod
    def prepare(
        graph_dict: dict[str, Any],
        simulation_rounds: int = 100,
    ) -> OffloadPacket:
        """Prepare an anonymized offload packet from a graph.

        Parameters
        ----------
        graph_dict : dict
            The graph's ``as_dict()`` output (objects + morphisms).
        simulation_rounds : int
            Number of rounds for the offloaded simulation.

        Returns
        -------
        OffloadPacket
            Anonymized packet ready for offloading.
        """
        objects = graph_dict.get("objects", [])
        morphisms = graph_dict.get("morphisms", [])

        # Build tag histogram (anonymized — no labels, no IDs)
        tag_hist: dict[str, int] = {}
        for m in morphisms:
            for tag in m.get("tags", []):
                tag_hist[tag] = tag_hist.get(tag, 0) + 1

        # Count vulnerable nodes
        vulnerable_count = sum(
            1 for o in objects if "vulnerable" in o.get("tags", [])
        )

        # Compute extraction ratio
        harmful_tags = {"extraction", "exploitation", "coercion", "neglect"}
        harmful_count = sum(
            1 for m in morphisms
            if set(t.lower() for t in m.get("tags", [])) & harmful_tags
        )
        extraction_ratio = harmful_count / max(len(morphisms), 1)

        # Compute packet hash for integrity
        content = json.dumps(tag_hist, sort_keys=True) + f"|{len(objects)}|{len(morphisms)}"
        packet_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        return OffloadPacket(
            morphism_tag_histogram=tag_hist,
            object_count=len(objects),
            morphism_count=len(morphisms),
            vulnerable_node_count=vulnerable_count,
            extraction_ratio=extraction_ratio,
            simulation_rounds=simulation_rounds,
            packet_hash=packet_hash,
        )

    @staticmethod
    def reintegrate(
        result: dict[str, Any],
        sustainability_score: float,
    ) -> dict[str, Any]:
        """Reintegrate a remote simulation result locally.

        Parameters
        ----------
        result : dict
            The remote simulation result.
        sustainability_score : float
            The sustainability score from the remote simulation.

        Returns
        -------
        dict
            Reintegrated result ready for the local pipeline.
        """
        return {
            "offloadResult": {
                "sustainabilityScore": round(sustainability_score, 4),
                "source": "offloaded_simulation",
                "remoteData": result,
            }
        }


def get_default_provider() -> AIProvider:
    """Get the default AI provider, preferring Ollama when available.

    Returns OllamaProvider if Ollama is reachable, otherwise LocalProvider.
    This is the Sprint 9 local-first handover entry point.
    """
    ollama = OllamaProvider()
    if ollama.is_available():
        return ollama
    return LocalProvider()
