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
import math
import random
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


# ---------------------------------------------------------------------------
# Sustainability Predicate (Module 2.1 Extension — Sprint 6.1)
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloProjection:
    """Result of a single Monte Carlo temporal simulation run."""

    timesteps: int
    survival_probability: float  # 0.0–1.0
    mean_health: float           # average graph health across the run
    collapse_step: int | None    # step at which collapse occurred, or None


@dataclass
class SustainabilityResult:
    """Full sustainability evaluation output."""

    sustainability_score: float       # (TemporalViability * EcologicalHarmony) / (FragilityFactor + 1)
    temporal_viability: float         # 0.0–1.0
    ecological_harmony: float         # 0.0–1.0
    fragility_factor: float           # 0.0+
    regenerative_loops: list[dict[str, Any]]
    depletion_morphisms: list[dict[str, Any]]
    projections: list[MonteCarloProjection]
    is_sustainable: bool              # True when score >= 5.0 on 0–10 scale


class SustainabilityPredicate:
    """Evaluates long-arc survival and ecological harmony of a categorical graph.

    Implements:
    - **TemporalViability**: Monte Carlo stub simulating graph evolution over
      t=10, 50, 100 steps with Bayesian priors for uncertainty handling.
    - **EcologicalHarmony**: Detects "Regenerative Loops" where morphisms
      ensure replenishment rather than pure depletion.
    - **FragilityFactor**: Measures graph vulnerability to single-point failures.

    Scoring formula:
        SustainabilityScore = (TemporalViability * EcologicalHarmony) / (FragilityFactor + 1)
        Normalised to a 0–10 scale.
    """

    # Tags indicating regenerative/replenishment morphisms
    REGENERATIVE_TAGS: set[str] = {
        "care", "protection", "service", "empowerment",
        "collaboration", "sustainability", "cooperative",
        "benefit_corporation", "stewardship", "replenishment",
    }

    # Tags indicating extractive/depleting morphisms
    DEPLETION_TAGS: set[str] = {
        "extraction", "exploitation", "coercion", "neglect",
        "division", "maximize_value", "profit_priority",
        "fiduciary_duty",
    }

    # Bayesian prior: base belief that any graph is moderately viable
    PRIOR_VIABILITY: float = 0.5
    PRIOR_STRENGTH: float = 2.0  # equivalent "pseudo-observations"

    SIMULATION_HORIZONS: tuple[int, ...] = (10, 50, 100)
    MONTE_CARLO_RUNS: int = 50  # runs per horizon

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # -- predicate protocol (returns 0.0–1.0) -------------------------------

    def __call__(self, artefact: dict[str, Any]) -> float:
        result = self.evaluate(artefact)
        return min(1.0, max(0.0, result.sustainability_score / 10.0))

    # -- detailed evaluation ------------------------------------------------

    def evaluate(self, artefact: dict[str, Any]) -> SustainabilityResult:
        """Full sustainability evaluation returning all sub-scores."""
        objects = artefact.get("objects", [])
        morphisms = artefact.get("morphisms", [])

        # 1. Ecological Harmony — detect regenerative loops vs depletion
        regen_loops, depletion_morphs = self._detect_loops(objects, morphisms)
        ecological_harmony = self._compute_ecological_harmony(
            regen_loops, depletion_morphs, morphisms,
        )

        # 2. Fragility Factor — single-point-of-failure vulnerability
        fragility = self._compute_fragility(objects, morphisms)

        # 3. Temporal Viability — Monte Carlo simulation with Bayesian priors
        projections = self._monte_carlo_simulate(
            objects, morphisms, ecological_harmony, fragility,
        )
        temporal_viability = self._compute_temporal_viability(projections)

        # 4. Composite score
        raw = (temporal_viability * ecological_harmony) / (fragility + 1.0)
        sustainability_score = min(10.0, max(0.0, raw * 10.0))

        return SustainabilityResult(
            sustainability_score=sustainability_score,
            temporal_viability=temporal_viability,
            ecological_harmony=ecological_harmony,
            fragility_factor=fragility,
            regenerative_loops=regen_loops,
            depletion_morphisms=depletion_morphs,
            projections=projections,
            is_sustainable=sustainability_score >= 5.0,
        )

    # -- Ecological Harmony -------------------------------------------------

    def _detect_loops(
        self,
        objects: list[dict[str, Any]],
        morphisms: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Detect regenerative loops and depletion morphisms.

        A regenerative loop exists when there is a cycle of morphisms
        where at least one leg carries regenerative tags — ensuring
        replenishment flows back to source nodes.
        """
        regenerative_loops: list[dict[str, Any]] = []
        depletion_morphisms: list[dict[str, Any]] = []

        # Build adjacency for cycle detection
        obj_ids = {o["id"] for o in objects}
        adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {oid: [] for oid in obj_ids}

        for m in morphisms:
            src, tgt = m.get("source", ""), m.get("target", "")
            if src in obj_ids and tgt in obj_ids:
                adjacency[src].append((tgt, m))

        # Classify individual morphisms
        for m in morphisms:
            m_tags = {t.lower() for t in m.get("tags", [])}

            if m_tags & self.DEPLETION_TAGS:
                depletion_morphisms.append({
                    "morphism_id": m.get("id", ""),
                    "label": m.get("label", ""),
                    "source": m.get("source", ""),
                    "target": m.get("target", ""),
                    "depletion_tags": sorted(m_tags & self.DEPLETION_TAGS),
                })

        # Detect regenerative loops: find cycles where at least one morphism
        # has regenerative tags (simplified: check for reciprocal edges)
        visited_pairs: set[tuple[str, str]] = set()
        for src in adjacency:
            for tgt, m_forward in adjacency[src]:
                if (src, tgt) in visited_pairs:
                    continue
                # Check for return path tgt -> src
                for back_tgt, m_back in adjacency.get(tgt, []):
                    if back_tgt == src:
                        fwd_tags = {t.lower() for t in m_forward.get("tags", [])}
                        bck_tags = {t.lower() for t in m_back.get("tags", [])}
                        combined = fwd_tags | bck_tags
                        if combined & self.REGENERATIVE_TAGS:
                            regenerative_loops.append({
                                "cycle": [src, tgt, src],
                                "forward_morphism": m_forward.get("id", ""),
                                "return_morphism": m_back.get("id", ""),
                                "regenerative_tags": sorted(combined & self.REGENERATIVE_TAGS),
                            })
                            visited_pairs.add((src, tgt))
                            visited_pairs.add((tgt, src))

        return regenerative_loops, depletion_morphisms

    def _compute_ecological_harmony(
        self,
        regen_loops: list[dict[str, Any]],
        depletion_morphs: list[dict[str, Any]],
        morphisms: list[dict[str, Any]],
    ) -> float:
        """Compute ecological harmony score (0.0–1.0).

        Higher when regenerative loops dominate over depletion morphisms.
        """
        if not morphisms:
            return 0.5  # neutral / unknown

        # Count morphisms with regenerative tags
        regen_count = 0
        deplete_count = len(depletion_morphs)
        for m in morphisms:
            m_tags = {t.lower() for t in m.get("tags", [])}
            if m_tags & self.REGENERATIVE_TAGS:
                regen_count += 1

        # Bonus for actual loop structures (cycles = strong replenishment)
        loop_bonus = min(0.3, len(regen_loops) * 0.1)

        total = regen_count + deplete_count
        if total == 0:
            return 0.5 + loop_bonus

        base_harmony = regen_count / total
        return min(1.0, base_harmony + loop_bonus)

    # -- Fragility Factor ---------------------------------------------------

    def _compute_fragility(
        self,
        objects: list[dict[str, Any]],
        morphisms: list[dict[str, Any]],
    ) -> float:
        """Compute fragility factor (0.0+, lower is better).

        Measures vulnerability to single-point-of-failure by checking
        node degree distribution and connectivity concentration.
        """
        if not objects or not morphisms:
            return 1.0  # maximum fragility for empty graphs

        obj_ids = {o["id"] for o in objects}
        degree: dict[str, int] = {oid: 0 for oid in obj_ids}

        for m in morphisms:
            src, tgt = m.get("source", ""), m.get("target", "")
            if src in degree:
                degree[src] += 1
            if tgt in degree:
                degree[tgt] += 1

        degrees = list(degree.values())
        if not degrees:
            return 1.0

        max_degree = max(degrees)
        mean_degree = sum(degrees) / len(degrees)

        # Fragility is high when one node holds disproportionate connections
        # (concentration risk) and low when connections are distributed
        if mean_degree == 0:
            return 1.0

        concentration = max_degree / (mean_degree * len(degrees))
        orphan_ratio = sum(1 for d in degrees if d == 0) / len(degrees)

        return min(2.0, concentration + orphan_ratio)

    # -- Monte Carlo Temporal Viability ------------------------------------

    def _monte_carlo_simulate(
        self,
        objects: list[dict[str, Any]],
        morphisms: list[dict[str, Any]],
        ecological_harmony: float,
        fragility: float,
    ) -> list[MonteCarloProjection]:
        """Run Monte Carlo simulations over multiple time horizons.

        Uses Bayesian priors to handle uncertainty: starts with a prior
        belief about viability and updates based on graph structure.
        """
        projections: list[MonteCarloProjection] = []

        # Bayesian prior update: posterior viability belief
        # prior = Beta(alpha, beta) where alpha ~ successes, beta ~ failures
        prior_alpha = self.PRIOR_STRENGTH * self.PRIOR_VIABILITY
        prior_beta = self.PRIOR_STRENGTH * (1.0 - self.PRIOR_VIABILITY)

        # Update with observed evidence from graph structure
        evidence_positive = ecological_harmony * 10.0
        evidence_negative = fragility * 5.0
        posterior_alpha = prior_alpha + evidence_positive
        posterior_beta = prior_beta + evidence_negative
        posterior_viability = posterior_alpha / (posterior_alpha + posterior_beta)

        for horizon in self.SIMULATION_HORIZONS:
            survivals = 0
            total_health = 0.0
            collapse_steps: list[int | None] = []

            for _ in range(self.MONTE_CARLO_RUNS):
                health = posterior_viability
                collapsed = False
                collapse_at: int | None = None
                run_health_sum = 0.0

                for step in range(1, horizon + 1):
                    # Each step: health degrades or improves stochastically
                    # Regenerative harmony sustains; depletion erodes
                    shock = self._rng.gauss(0, 0.05)
                    regen_boost = ecological_harmony * 0.02
                    fragility_drain = fragility * 0.015
                    health += regen_boost - fragility_drain + shock

                    health = max(0.0, min(1.0, health))
                    run_health_sum += health

                    if health < 0.1:
                        collapsed = True
                        collapse_at = step
                        break

                if not collapsed:
                    survivals += 1
                    collapse_steps.append(None)
                else:
                    collapse_steps.append(collapse_at)

                total_health += run_health_sum / horizon if not collapsed else run_health_sum / (collapse_at or 1)

            survival_prob = survivals / self.MONTE_CARLO_RUNS
            mean_health = total_health / self.MONTE_CARLO_RUNS

            # Find most common collapse step (mode) for reporting
            real_collapses = [s for s in collapse_steps if s is not None]
            mode_collapse = min(real_collapses) if real_collapses else None

            projections.append(MonteCarloProjection(
                timesteps=horizon,
                survival_probability=survival_prob,
                mean_health=mean_health,
                collapse_step=mode_collapse,
            ))

        return projections

    def _compute_temporal_viability(
        self,
        projections: list[MonteCarloProjection],
    ) -> float:
        """Aggregate Monte Carlo projections into a single viability score.

        Weights longer horizons more heavily (long-arc survival matters more).
        """
        if not projections:
            return self.PRIOR_VIABILITY

        # Weighted average: longer horizons get exponentially more weight
        weights = [math.log2(p.timesteps + 1) for p in projections]
        total_weight = sum(weights)
        if total_weight == 0:
            return self.PRIOR_VIABILITY

        weighted_sum = sum(
            w * (p.survival_probability * 0.7 + p.mean_health * 0.3)
            for w, p in zip(weights, projections)
        )
        return weighted_sum / total_weight


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
