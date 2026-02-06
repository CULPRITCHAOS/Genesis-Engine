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

# ---------------------------------------------------------------------------
# Incentive Stability Predicate (Module 2.1 Extension)
# ---------------------------------------------------------------------------

class IncentiveStabilityPredicate:
    """Detects Legal Gravity Wells — specifically Shareholder Primacy patterns.

    Identifies directed morphisms from a Corporation object to a Shareholder
    sink node tagged with ``maximize_value``, ``fiduciary_duty``, or
    ``profit_priority``.

    Scoring (raw 0–10 scale, normalised to 0.0–1.0 for predicate protocol):

    * **Base score**: 10
    * **-7** if a *Primacy Pattern* is detected (Corporation → Shareholder
      morphism carrying a primacy tag, or a Corporation object tagged with
      ``shareholder_primacy_risk``).
    * **+5** if counter-tags ``benefit_corporation`` or ``cooperative`` are
      present anywhere in the graph.
    * Clamped to [0, 10].
    * Any raw score **< 5** triggers the ``incentive_instability`` flag in
      the Disharmony Report.
    """

    PRIMACY_TAGS: set[str] = {"maximize_value", "fiduciary_duty", "profit_priority"}
    COUNTER_TAGS: set[str] = {"benefit_corporation", "cooperative"}

    # -- predicate protocol (returns 0.0 – 1.0) ----------------------------

    def __call__(self, artefact: dict[str, Any]) -> float:
        raw_score, _ = self.evaluate(artefact)
        return min(1.0, max(0.0, raw_score / 10.0))

    # -- detailed evaluation ------------------------------------------------

    def evaluate(self, artefact: dict[str, Any]) -> tuple[float, bool]:
        """Return ``(raw_score, has_incentive_instability)``.

        ``raw_score`` is on a 0–10 scale.
        ``has_incentive_instability`` is True when ``raw_score < 5``.
        """
        score = 10.0

        objects = artefact.get("objects", [])
        morphisms = artefact.get("morphisms", [])

        # Collect Corporation and Shareholder node IDs.
        corp_ids: set[str] = set()
        shareholder_ids: set[str] = set()
        all_tags: set[str] = set()

        for obj in objects:
            obj_tags = {t.lower() for t in obj.get("tags", [])}
            label = obj.get("label", "").lower()
            all_tags.update(obj_tags)

            if "corporation" in label or "corp" in label:
                corp_ids.add(obj["id"])
            if "shareholder" in label:
                shareholder_ids.add(obj["id"])

        # Collect morphism-level tags.
        for m in morphisms:
            all_tags.update(t.lower() for t in m.get("tags", []))

        # --- Primacy Pattern detection ------------------------------------
        primacy_detected = False

        # Path 1: Directed Corporation → Shareholder morphism with a
        #          primacy tag.
        for m in morphisms:
            m_tags = {t.lower() for t in m.get("tags", [])}
            if (m.get("source") in corp_ids
                    and m.get("target") in shareholder_ids
                    and m_tags & self.PRIMACY_TAGS):
                primacy_detected = True
                break

        # Path 2: Corporation object explicitly tagged with
        #          ``shareholder_primacy_risk`` (set by AxiomLogix).
        if not primacy_detected:
            for obj in objects:
                obj_tags = {t.lower() for t in obj.get("tags", [])}
                if "shareholder_primacy_risk" in obj_tags:
                    primacy_detected = True
                    break

        if primacy_detected:
            score -= 7.0

        # --- Counter-tag bonus --------------------------------------------
        if all_tags & self.COUNTER_TAGS:
            score += 5.0

        score = max(0.0, min(10.0, score))
        has_instability = score < 5.0

        return score, has_instability


class AxiomAnchorFrozenError(Exception):
    """Raised when attempting to modify a sealed AxiomAnchor."""


class AxiomAnchor:
    """Central validation gate for the Genesis Engine.

    Every downstream component must submit its artefacts through
    ``validate`` before the system acts on them.

    The Anchor is the objective "Ground Truth". Once sealed via
    ``seal()``, its predicates and directive become immutable.
    Human overrides are recorded in the Override Log but NEVER
    alter the Anchor's predicates — preserving the integrity of
    the axiom system.
    """

    def __init__(
        self,
        directive: PrimeDirective | None = None,
        alignment_threshold: float = 0.5,
    ) -> None:
        self.directive = directive or PrimeDirective()
        self.alignment_threshold = alignment_threshold
        self._sealed = False

        # Map principle → predicate.  Extensible by callers via
        # ``register_predicate``.
        self._predicates: dict[str, Callable[[dict[str, Any]], float]] = {
            DirectivePrinciple.UNITY.value: _unity_predicate,
            DirectivePrinciple.COMPASSION.value: _compassion_predicate,
            DirectivePrinciple.COHERENCE.value: _coherence_predicate,
        }

    # -- immutability -------------------------------------------------------

    @property
    def is_sealed(self) -> bool:
        """True if the Anchor has been sealed against modification."""
        return self._sealed

    def seal(self) -> None:
        """Seal the Anchor, preventing further predicate registration.

        Once sealed, ``register_predicate`` will raise
        ``AxiomAnchorFrozenError``. This enforces the invariant that
        human overrides never alter the Ground Truth.
        """
        self._sealed = True

    # -- public API ---------------------------------------------------------

    def register_predicate(
        self,
        principle: str,
        predicate: Callable[[dict[str, Any]], float],
    ) -> None:
        """Register (or override) a predicate for *principle*.

        Raises ``AxiomAnchorFrozenError`` if the Anchor has been sealed.
        """
        if self._sealed:
            raise AxiomAnchorFrozenError(
                "Cannot modify a sealed AxiomAnchor. "
                "Human overrides are recorded in the Override Log, "
                "not in the Axiom Anchor predicates. "
                "The Anchor remains the immutable Ground Truth."
            )
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
