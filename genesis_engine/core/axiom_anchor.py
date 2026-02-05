"""
Module 2.1 — The Axiom Anchor

Holds the Global Prime Directive and acts as a Validator gate that all
downstream logic must pass through.

The Prime Directive encodes:
    "Does this serve Love?"
    where Love := recognition of Unity and the drive for benevolent,
    coherent outcomes.

Design notes
------------
* Every validatable artefact in the system implements the ``Validatable``
  protocol (a ``dict`` with at least a ``"type"`` key).
* The Anchor exposes a single ``validate`` entry-point that returns a
  ``ValidationResult`` — a typed dict carrying the boolean verdict,
  a coherence score, and human-readable reasoning.
* Validation rules are expressed as *axiom predicates* so new rules can
  be composed without modifying the Anchor itself.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Prime Directive
# ---------------------------------------------------------------------------

class DirectivePrinciple(enum.Enum):
    """The three irreducible pillars of the Prime Directive."""

    UNITY = "unity"
    COMPASSION = "compassion"
    COHERENCE = "coherence"


@dataclass(frozen=True)
class PrimeDirective:
    """Immutable representation of the Global Prime Directive.

    Parameters
    ----------
    statement : str
        The human-readable directive.
    principles : tuple[DirectivePrinciple, ...]
        The foundational pillars against which all artefacts are measured.
    """

    statement: str = "Does this serve Love?"
    principles: tuple[DirectivePrinciple, ...] = (
        DirectivePrinciple.UNITY,
        DirectivePrinciple.COMPASSION,
        DirectivePrinciple.COHERENCE,
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "principles": [p.value for p in self.principles],
        }


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of passing an artefact through the Axiom Anchor."""

    is_aligned: bool
    coherence_score: float  # 0.0 – 1.0
    principle_scores: dict[str, float] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "isAligned": self.is_aligned,
            "coherenceScore": round(self.coherence_score, 4),
            "principleScores": {k: round(v, 4) for k, v in self.principle_scores.items()},
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Axiom predicate protocol
# ---------------------------------------------------------------------------

class AxiomPredicate(Protocol):
    """A callable that scores an artefact against a single principle.

    Returns a float in [0.0, 1.0] where 1.0 means fully aligned.
    """

    def __call__(self, artefact: dict[str, Any]) -> float: ...


# ---------------------------------------------------------------------------
# Built-in predicates
# ---------------------------------------------------------------------------

def _unity_predicate(artefact: dict[str, Any]) -> float:
    """Score how well *artefact* preserves Unity.

    Heuristic (neuro-symbolic): examines morphism tags for
    extractive / divisive intent.
    """
    disharmony_tags = {"extraction", "exploitation", "division", "coercion", "neglect"}
    harmony_tags = {"service", "protection", "collaboration", "empowerment", "care"}

    tags: set[str] = set()
    for m in artefact.get("morphisms", []):
        tags.add(m.get("label", "").lower())
        tags.update(t.lower() for t in m.get("tags", []))

    negative_hits = len(tags & disharmony_tags)
    positive_hits = len(tags & harmony_tags)
    total = negative_hits + positive_hits

    if total == 0:
        return 0.5  # neutral / unknown
    return positive_hits / total


def _compassion_predicate(artefact: dict[str, Any]) -> float:
    """Score how well *artefact* upholds Compassion.

    Checks whether vulnerable entities receive protective morphisms.
    """
    objects = artefact.get("objects", [])
    morphisms = artefact.get("morphisms", [])

    vulnerable_ids = {
        o["id"] for o in objects
        if "vulnerable" in (t.lower() for t in o.get("tags", []))
    }
    if not vulnerable_ids:
        return 1.0  # no vulnerable parties — default aligned

    protective_labels = {"protection", "care", "service", "empowerment"}
    protected: set[str] = set()
    for m in morphisms:
        label_match = m.get("label", "").lower() in protective_labels
        tag_match = bool(
            set(t.lower() for t in m.get("tags", [])) & protective_labels
        )
        if label_match or tag_match:
            protected.add(m.get("target", ""))
            protected.update(m.get("targets", []))

    ratio = len(vulnerable_ids & protected) / len(vulnerable_ids)
    return ratio


def _coherence_predicate(artefact: dict[str, Any]) -> float:
    """Score the compositional coherence of the categorical graph.

    A graph is coherent when every object participates in at least one
    morphism (no orphan nodes) and every morphism has valid endpoints.
    """
    objects = artefact.get("objects", [])
    morphisms = artefact.get("morphisms", [])

    if not objects:
        return 0.0

    obj_ids = {o["id"] for o in objects}

    referenced: set[str] = set()
    dangling = 0
    for m in morphisms:
        src = m.get("source", "")
        tgt = m.get("target", "")
        referenced.update({src, tgt})
        if src not in obj_ids or tgt not in obj_ids:
            dangling += 1

    if not morphisms:
        return 0.0

    participation = len(obj_ids & referenced) / len(obj_ids)
    integrity = 1.0 - (dangling / len(morphisms)) if morphisms else 1.0
    return (participation + integrity) / 2.0


# ---------------------------------------------------------------------------
# Axiom Anchor
# ---------------------------------------------------------------------------

class AxiomAnchor:
    """Central validation gate for the Genesis Engine.

    Every downstream component must submit its artefacts through
    ``validate`` before the system acts on them.
    """

    def __init__(
        self,
        directive: PrimeDirective | None = None,
        alignment_threshold: float = 0.5,
    ) -> None:
        self.directive = directive or PrimeDirective()
        self.alignment_threshold = alignment_threshold

        # Map principle → predicate.  Extensible by callers via
        # ``register_predicate``.
        self._predicates: dict[str, Callable[[dict[str, Any]], float]] = {
            DirectivePrinciple.UNITY.value: _unity_predicate,
            DirectivePrinciple.COMPASSION.value: _compassion_predicate,
            DirectivePrinciple.COHERENCE.value: _coherence_predicate,
        }

    # -- public API ---------------------------------------------------------

    def register_predicate(
        self,
        principle: str,
        predicate: Callable[[dict[str, Any]], float],
    ) -> None:
        """Register (or override) a predicate for *principle*."""
        self._predicates[principle] = predicate

    def validate(self, artefact: dict[str, Any]) -> ValidationResult:
        """Run *artefact* through all axiom predicates.

        Returns a ``ValidationResult`` with per-principle scores and
        a composite coherence score.
        """
        scores: dict[str, float] = {}
        reasoning: list[str] = []

        for principle in self.directive.principles:
            pred = self._predicates.get(principle.value)
            if pred is None:
                scores[principle.value] = 0.5
                reasoning.append(
                    f"[{principle.value}] No predicate registered — scored neutral."
                )
                continue

            score = pred(artefact)
            scores[principle.value] = score

            if score < self.alignment_threshold:
                reasoning.append(
                    f"[{principle.value}] MISALIGNED (score={score:.2f}): "
                    f"artefact fails the {principle.value} axiom."
                )
            else:
                reasoning.append(
                    f"[{principle.value}] ALIGNED (score={score:.2f})"
                )

        composite = sum(scores.values()) / len(scores) if scores else 0.0
        is_aligned = all(s >= self.alignment_threshold for s in scores.values())

        return ValidationResult(
            is_aligned=is_aligned,
            coherence_score=composite,
            principle_scores=scores,
            reasoning=reasoning,
        )
