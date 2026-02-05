"""
Module 1.2 — The Dream Engine

The generative layer that transforms Disharmony into Coherence.

Given a ``DisharmonyReport`` produced by the Deconstruction Engine, the
Dream Engine generates three distinct solution paths — each a new
``CategoricalGraph`` — and validates every dream through the Axiom Anchor
before finalising.

The Threefold Path
------------------
1. **Path of Reform** — Fixes the existing morphisms in-place (e.g.
   replacing Extraction with Fair Reciprocity) while preserving the
   original entity set.
2. **Path of Reinvention** — Proposes an entirely new categorical graph
   that achieves the stated goal through Unity-aligned relationships.
3. **Path of Dissolution** — Removes the harmful system by introducing
   structures that make it obsolete.

Every path carries:
* ``unityAlignmentScore`` — from Axiom Anchor recursive validation.
* ``feasibilityScore``    — heuristic estimate of implementation effort.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from genesis_engine.core.axiom_anchor import AxiomAnchor, ValidationResult
from genesis_engine.core.axiomlogix import CategoricalGraph, Object, Morphism
from genesis_engine.core.deconstruction_engine import DisharmonyReport, MorphismFinding


# ---------------------------------------------------------------------------
# Constants — morphism healing lexicon
# ---------------------------------------------------------------------------

# Maps disharmonic morphism labels → their healed Reform equivalents.
_REFORM_MAP: dict[str, tuple[str, list[str]]] = {
    "Extraction": ("Fair_Reciprocity", ["service", "collaboration"]),
    "Exploitation": ("Mutual_Benefit", ["service", "empowerment"]),
    "Surveillance": ("Transparent_Oversight", ["protection", "care"]),
    "Neglect": ("Active_Care", ["care", "protection"]),
    "Coercion": ("Informed_Consent", ["empowerment", "protection"]),
    "Manipulation": ("Honest_Engagement", ["service", "empowerment"]),
    "Harm": ("Restoration", ["care", "protection"]),
    "Discrimination": ("Equitable_Access", ["empowerment", "collaboration"]),
    "Division": ("Unification", ["collaboration", "care"]),
    "Displacement": ("Inclusive_Integration", ["collaboration", "empowerment"]),
    "Suppression": ("Open_Expression", ["empowerment", "service"]),
    "Censorship": ("Free_Dialogue", ["empowerment", "collaboration"]),
    "Restriction": ("Empowered_Access", ["empowerment", "service"]),
    "Violation": ("Respectful_Engagement", ["care", "protection"]),
    "Monetization": ("Value_Sharing", ["service", "collaboration"]),
    "Tracking": ("Privacy_Respecting_Analytics", ["protection", "service"]),
    "Prioritization": ("Balanced_Consideration", ["service", "care"]),
}

# Reinvention archetypes — templates for wholly new relational structures.
_REINVENTION_ROLES: dict[str, list[str]] = {
    "Steward": ["stakeholder", "actor"],
    "Beneficiary": ["stakeholder", "vulnerable"],
    "Commons": ["value", "shared"],
    "Covenant": ["mechanism", "protective"],
}

_REINVENTION_MORPHISMS: list[tuple[str, str, str, list[str]]] = [
    ("Stewardship", "Steward", "Commons", ["protection", "care"]),
    ("Empowerment", "Steward", "Beneficiary", ["empowerment", "service"]),
    ("Participation", "Beneficiary", "Commons", ["collaboration"]),
    ("Accountability", "Covenant", "Steward", ["protection"]),
    ("Rights_Guarantee", "Covenant", "Beneficiary", ["protection", "empowerment"]),
]

# Dissolution replacement structures.
_DISSOLUTION_ROLES: dict[str, list[str]] = {
    "Community_Collective": ["stakeholder", "actor"],
    "Member": ["stakeholder", "vulnerable"],
    "Shared_Resource": ["value", "shared"],
    "Cooperative_Protocol": ["mechanism", "protective"],
}

_DISSOLUTION_MORPHISMS: list[tuple[str, str, str, list[str]]] = [
    ("Democratic_Governance", "Community_Collective", "Shared_Resource", ["collaboration", "protection"]),
    ("Direct_Benefit", "Shared_Resource", "Member", ["service", "empowerment"]),
    ("Voice", "Member", "Community_Collective", ["empowerment", "collaboration"]),
    ("Safeguard", "Cooperative_Protocol", "Member", ["protection", "care"]),
    ("Transparency", "Cooperative_Protocol", "Community_Collective", ["service", "protection"]),
]


# ---------------------------------------------------------------------------
# Path type
# ---------------------------------------------------------------------------

class PathType(Enum):
    """The three solution archetypes."""

    REFORM = "reform"
    REINVENTION = "reinvention"
    DISSOLUTION = "dissolution"


# ---------------------------------------------------------------------------
# Dream Path (single solution)
# ---------------------------------------------------------------------------

@dataclass
class DreamPath:
    """A single harmonised solution produced by the Dream Engine."""

    path_type: PathType
    title: str
    description: str
    healed_graph: CategoricalGraph
    unity_alignment_score: float  # 0.0 – 1.0 from Axiom Anchor
    feasibility_score: float  # 0.0 – 1.0 heuristic
    validation: ValidationResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathType": self.path_type.value,
            "title": self.title,
            "description": self.description,
            "healedGraph": self.healed_graph.as_dict(),
            "unityAlignmentScore": round(self.unity_alignment_score, 4),
            "feasibilityScore": round(self.feasibility_score, 4),
            "validation": self.validation.as_dict() if self.validation else None,
        }


# ---------------------------------------------------------------------------
# Possibility Report
# ---------------------------------------------------------------------------

@dataclass
class PossibilityReport:
    """Top-level output of the Dream Engine containing all three paths."""

    source_text: str
    original_disharmony: DisharmonyReport
    paths: list[DreamPath] = field(default_factory=list)
    recommended_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "possibilityReport": {
                "timestamp": self.timestamp,
                "sourceText": self.source_text,
                "originalDisharmony": {
                    "unityImpact": round(self.original_disharmony.unity_impact, 2),
                    "compassionDeficit": round(self.original_disharmony.compassion_deficit, 2),
                    "primeDirectiveAligned": self.original_disharmony.is_aligned,
                },
                "paths": [p.as_dict() for p in self.paths],
                "recommendedPath": self.recommended_path,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """Transforms disharmony into coherence via the Threefold Path.

    Parameters
    ----------
    anchor : AxiomAnchor | None
        Shared Axiom Anchor for recursive validation.  A default instance
        is created if omitted.
    max_iterations : int
        Maximum reform iterations when iteratively healing morphisms.
    """

    def __init__(
        self,
        anchor: AxiomAnchor | None = None,
        max_iterations: int = 3,
    ) -> None:
        self.anchor = anchor or AxiomAnchor()
        self.max_iterations = max_iterations

    # -- public API ---------------------------------------------------------

    def dream(
        self,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
    ) -> PossibilityReport:
        """Generate a ``PossibilityReport`` with three harmonised paths."""

        paths: list[DreamPath] = [
            self._path_of_reform(report, original_graph),
            self._path_of_reinvention(report, original_graph),
            self._path_of_dissolution(report, original_graph),
        ]

        # Pick the recommended path: highest unity score, then feasibility.
        best = max(
            paths,
            key=lambda p: (p.unity_alignment_score, p.feasibility_score),
        )

        return PossibilityReport(
            source_text=report.source_text,
            original_disharmony=report,
            paths=paths,
            recommended_path=best.path_type.value,
        )

    # -- Path of Reform -----------------------------------------------------

    def _path_of_reform(
        self,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
    ) -> DreamPath:
        """Heal existing morphisms in-place while preserving entities."""

        graph = self._clone_graph(original_graph)
        flagged_labels = {
            f.label for f in report.findings if f.disharmony_score > 0
        }

        # Replace each disharmonic morphism with its healed counterpart.
        healed_morphisms: list[Morphism] = []
        for morph in graph.morphisms:
            if morph.label in flagged_labels:
                new_label, new_tags = _REFORM_MAP.get(
                    morph.label,
                    ("Harmonised_" + morph.label, ["service", "care"]),
                )
                healed_morphisms.append(Morphism(
                    id=f"mor-{uuid.uuid4().hex[:8]}",
                    label=new_label,
                    source=morph.source,
                    target=morph.target,
                    tags=list(new_tags),
                ))
            else:
                healed_morphisms.append(morph)

        graph.morphisms = healed_morphisms

        # Ensure every vulnerable entity is a target of at least one
        # protective morphism (iterative repair).
        graph = self._ensure_compassion_coverage(graph)

        # Recursive validation through the Axiom Anchor.
        validation = self.anchor.validate(graph.as_artefact())
        unity_score = validation.principle_scores.get("unity", 0.0)
        feasibility = self._estimate_feasibility(report, PathType.REFORM)

        return DreamPath(
            path_type=PathType.REFORM,
            title="Path of Reform",
            description=(
                "Transforms extractive and harmful morphisms into "
                "reciprocal, caring relationships while preserving "
                "the existing entity structure."
            ),
            healed_graph=graph,
            unity_alignment_score=unity_score,
            feasibility_score=feasibility,
            validation=validation,
        )

    # -- Path of Reinvention ------------------------------------------------

    def _path_of_reinvention(
        self,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
    ) -> DreamPath:
        """Propose an entirely new graph built from stewardship archetypes."""

        graph = CategoricalGraph(source_text=original_graph.source_text)

        # Map original entities to reinvention roles.
        original_actors = [
            o for o in original_graph.objects if "actor" in o.tags
        ]
        original_vulnerable = [
            o for o in original_graph.objects if "vulnerable" in o.tags
        ]
        original_values = [
            o for o in original_graph.objects if "value" in o.tags
        ]

        # Create role-based objects, carrying forward original labels
        # where meaningful.
        steward_label = original_actors[0].label + "_as_Steward" if original_actors else "Steward"
        beneficiary_labels = [o.label for o in original_vulnerable] or ["Beneficiary"]
        commons_label = original_values[0].label + "_Commons" if original_values else "Shared_Commons"

        steward = graph.add_object(steward_label, _REINVENTION_ROLES["Steward"])
        commons = graph.add_object(commons_label, _REINVENTION_ROLES["Commons"])
        covenant = graph.add_object("Governance_Covenant", _REINVENTION_ROLES["Covenant"])

        beneficiaries: list[Object] = []
        for label in beneficiary_labels:
            b = graph.add_object(label, list(_REINVENTION_ROLES["Beneficiary"]))
            beneficiaries.append(b)

        # Wire archetype morphisms.
        for morph_label, src_role, tgt_role, tags in _REINVENTION_MORPHISMS:
            src = self._resolve_role(src_role, steward, beneficiaries, commons, covenant)
            tgts = self._resolve_role_targets(tgt_role, steward, beneficiaries, commons, covenant)
            for tgt in tgts:
                graph.add_morphism(morph_label, src, tgt, list(tags))

        # Recursive validation.
        validation = self.anchor.validate(graph.as_artefact())
        unity_score = validation.principle_scores.get("unity", 0.0)
        feasibility = self._estimate_feasibility(report, PathType.REINVENTION)

        return DreamPath(
            path_type=PathType.REINVENTION,
            title="Path of Reinvention",
            description=(
                "Replaces the entire relational structure with a "
                "stewardship model: actors become stewards, values become "
                "a shared commons, and a governance covenant ensures "
                "accountability."
            ),
            healed_graph=graph,
            unity_alignment_score=unity_score,
            feasibility_score=feasibility,
            validation=validation,
        )

    # -- Path of Dissolution ------------------------------------------------

    def _path_of_dissolution(
        self,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
    ) -> DreamPath:
        """Propose structures that make the harmful system obsolete."""

        graph = CategoricalGraph(source_text=original_graph.source_text)

        # Carry forward vulnerable-entity labels as members.
        original_vulnerable = [
            o for o in original_graph.objects if "vulnerable" in o.tags
        ]
        member_labels = [o.label for o in original_vulnerable] or ["Member"]

        collective = graph.add_object(
            "Community_Collective", _DISSOLUTION_ROLES["Community_Collective"],
        )
        shared = graph.add_object(
            "Shared_Resource", _DISSOLUTION_ROLES["Shared_Resource"],
        )
        protocol = graph.add_object(
            "Cooperative_Protocol", _DISSOLUTION_ROLES["Cooperative_Protocol"],
        )

        members: list[Object] = []
        for label in member_labels:
            m = graph.add_object(label, list(_DISSOLUTION_ROLES["Member"]))
            members.append(m)

        # Wire dissolution morphisms.
        for morph_label, src_role, tgt_role, tags in _DISSOLUTION_MORPHISMS:
            src = self._resolve_role(src_role, collective, members, shared, protocol)
            tgts = self._resolve_role_targets(tgt_role, collective, members, shared, protocol)
            for tgt in tgts:
                graph.add_morphism(morph_label, src, tgt, list(tags))

        # Recursive validation.
        validation = self.anchor.validate(graph.as_artefact())
        unity_score = validation.principle_scores.get("unity", 0.0)
        feasibility = self._estimate_feasibility(report, PathType.DISSOLUTION)

        return DreamPath(
            path_type=PathType.DISSOLUTION,
            title="Path of Dissolution",
            description=(
                "Replaces the harmful system with a community-owned "
                "cooperative structure where democratic governance, "
                "direct benefit, and transparent protocols make the "
                "original extractive model obsolete."
            ),
            healed_graph=graph,
            unity_alignment_score=unity_score,
            feasibility_score=feasibility,
            validation=validation,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clone_graph(graph: CategoricalGraph) -> CategoricalGraph:
        """Deep-copy a ``CategoricalGraph``."""
        new = CategoricalGraph(source_text=graph.source_text)
        for obj in graph.objects:
            new.objects.append(Object(
                id=obj.id,
                label=obj.label,
                tags=list(obj.tags),
            ))
        for morph in graph.morphisms:
            new.morphisms.append(Morphism(
                id=morph.id,
                label=morph.label,
                source=morph.source,
                target=morph.target,
                tags=list(morph.tags),
            ))
        return new

    def _ensure_compassion_coverage(self, graph: CategoricalGraph) -> CategoricalGraph:
        """Add protective morphisms for any unprotected vulnerable entity."""
        vulnerable_ids = {
            o.id for o in graph.objects if "vulnerable" in o.tags
        }
        protective_labels = {"protection", "care", "service", "empowerment"}

        protected: set[str] = set()
        for m in graph.morphisms:
            if m.label.lower().replace("_", " ").split()[0] in {"active", "fair", "mutual", "transparent", "honest", "equitable", "inclusive", "open", "free", "empowered", "respectful", "privacy", "value", "harmonised"}:
                # Healed morphisms protect their targets
                protected.add(m.target)
            elif m.label.lower() in protective_labels:
                protected.add(m.target)
            elif set(t.lower() for t in m.tags) & protective_labels:
                protected.add(m.target)

        unprotected = vulnerable_ids - protected
        if unprotected:
            # Find a suitable source (actor or mechanism).
            actors = [o for o in graph.objects if "actor" in o.tags]
            src = actors[0] if actors else graph.objects[0]
            for vuln_id in unprotected:
                if src.id != vuln_id:
                    graph.add_morphism("Active_Care", src, vuln_id, ["care", "protection"])

        return graph

    @staticmethod
    def _resolve_role(
        role: str,
        primary: Object,
        group: list[Object],
        commons: Object,
        governance: Object,
    ) -> Object:
        """Map an archetype role name to the corresponding Object."""
        mapping: dict[str, Object] = {
            "Steward": primary,
            "Community_Collective": primary,
            "Commons": commons,
            "Shared_Resource": commons,
            "Covenant": governance,
            "Cooperative_Protocol": governance,
        }
        if role in mapping:
            return mapping[role]
        if role in ("Beneficiary", "Member"):
            return group[0] if group else primary
        return primary

    @staticmethod
    def _resolve_role_targets(
        role: str,
        primary: Object,
        group: list[Object],
        commons: Object,
        governance: Object,
    ) -> list[Object]:
        """Like ``_resolve_role`` but returns a list — expanding group roles
        so that every beneficiary / member receives a morphism."""
        if role in ("Beneficiary", "Member"):
            return group if group else [primary]
        mapping: dict[str, Object] = {
            "Steward": primary,
            "Community_Collective": primary,
            "Commons": commons,
            "Shared_Resource": commons,
            "Covenant": governance,
            "Cooperative_Protocol": governance,
        }
        return [mapping.get(role, primary)]

    @staticmethod
    def _estimate_feasibility(
        report: DisharmonyReport,
        path_type: PathType,
    ) -> float:
        """Heuristic feasibility score based on severity and path type.

        Reform is easier for mild disharmony; Dissolution is harder but
        more thorough for deep problems.
        """
        severity = (report.unity_impact + report.compassion_deficit) / 20.0

        if path_type == PathType.REFORM:
            # Reform is most feasible when problems are moderate.
            return max(0.1, 1.0 - severity * 0.5)
        elif path_type == PathType.REINVENTION:
            # Reinvention feasibility is moderate regardless.
            return 0.6
        else:
            # Dissolution feasibility scales with problem severity:
            # the worse the problem, the more justified the approach.
            return min(0.9, 0.3 + severity * 0.5)
