"""
Module 1.7 — The Mirror of Truth

A self-critique loop that performs Adversarial Deconstruction on Dream Engine
proposals *before* they reach the Final Exam.  The Mirror acts as an internal
critic, searching for **Surface Alignment** that masks **Deep Disharmony**.

Key concepts:
- **Surface Alignment** — A proposal that *appears* aligned (e.g. claims
  sustainability, job creation, community benefit) but conceals extractive
  mechanisms (hidden tariff riders, cost-shifting, shareholder extraction).
- **Deep Disharmony** — Structural misalignment that survives cosmetic
  reform: shareholder primacy gravity wells, regressive cost allocation,
  fragility amplification.
- **Adversarial Deconstruction** — The Mirror subjects every Dream Engine
  winner to critique probes that test for known disharmony patterns.
- **Refinement Trace** — A structured record of what the Mirror found
  and the mandatory Regenerative Repair it requires before the blueprint
  can proceed to the Forge.

Integration:
- Sits between the Dream Engine (Module 1.2) and the Architectural Forge
  (Module 1.3) in the pipeline.
- When the Mirror detects Deep Disharmony, it may trigger a Reinvention
  path override — forcing the Dream Engine to select a structural solution
  rather than a cosmetic Reform.
- The Aria Interface (Module 3.2) renders the Mirror's findings in a
  Refinement Panel alongside the Stewardship Manifesto.

Sprint 8 — The Mirror of Truth & The Grid War Case Study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor,
    IncentiveStabilityPredicate,
)
from genesis_engine.core.axiomlogix import CategoricalGraph, Object
from genesis_engine.core.deconstruction_engine import DisharmonyReport
from genesis_engine.core.dream_engine import DreamPath, PathType, PossibilityReport


# ---------------------------------------------------------------------------
# Surface Alignment detection patterns
# ---------------------------------------------------------------------------

# Claims that may indicate Surface Alignment when paired with extractive tags
_SURFACE_ALIGNMENT_CLAIMS: set[str] = {
    "job_creation", "economic_development", "community_benefit",
    "clean_energy", "innovation", "modernization", "investment",
    "growth", "prosperity", "partnership",
}

# Tags that reveal Deep Disharmony beneath surface claims
_DEEP_DISHARMONY_TAGS: set[str] = {
    "extraction", "exploitation", "cost_pass_through",
    "tariff_rider", "infrastructure_cost_pass_through",
    "profit_priority", "fiduciary_duty", "maximize_value",
    "preferential_treatment", "cross_subsidy",
    "regressive_impact", "capacity_demand", "reliability_threat",
    "shareholder_primacy_risk",
    "water_depletion", "unsustainable_withdrawal", "shadow_cost",
}

# Categories of Deep Disharmony
_DISHARMONY_CATEGORIES: dict[str, set[str]] = {
    "cost_shifting_to_vulnerable": {
        "cost_pass_through", "tariff_rider",
        "infrastructure_cost_pass_through", "regressive_impact",
    },
    "shareholder_primacy_extraction": {
        "profit_priority", "fiduciary_duty", "maximize_value",
        "shareholder_primacy_risk",
    },
    "grid_fragility_amplification": {
        "capacity_demand", "reliability_threat",
    },
    "regressive_wealth_transfer": {
        "extraction", "exploitation", "preferential_treatment",
        "cross_subsidy",
    },
    "unsustainable_water_withdrawal": {
        "water_depletion", "unsustainable_withdrawal", "shadow_cost",
    },
}


# ---------------------------------------------------------------------------
# Critique Finding — a single adversarial observation
# ---------------------------------------------------------------------------

@dataclass
class CritiqueFinding:
    """A single finding from the Mirror's adversarial deconstruction."""

    category: str  # e.g. "cost_shifting_to_vulnerable"
    severity: float  # 0.0–10.0
    description: str
    evidence: list[str] = field(default_factory=list)
    affected_nodes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": round(self.severity, 2),
            "description": self.description,
            "evidence": self.evidence,
            "affectedNodes": self.affected_nodes,
        }


# ---------------------------------------------------------------------------
# Refinement Trace — the Mirror's complete output
# ---------------------------------------------------------------------------

@dataclass
class RefinementTrace:
    """The complete output of the Mirror of Truth's adversarial deconstruction.

    Contains:
    - Whether Surface Alignment was detected
    - The specific Deep Disharmony categories found
    - A mandatory Regenerative Repair that must be applied
    - Whether a Reinvention path override is triggered
    - The critique findings that support the conclusions
    """

    surface_alignment_detected: bool
    deep_disharmony_categories: list[str] = field(default_factory=list)
    critique_findings: list[CritiqueFinding] = field(default_factory=list)
    mandatory_repair: str = ""
    reinvention_triggered: bool = False
    original_path_type: str = ""
    recommended_path_type: str = ""
    mirror_score: float = 10.0  # 0–10, lower = more disharmony found
    vulnerable_node_protected: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "refinementTrace": {
                "timestamp": self.timestamp,
                "surfaceAlignmentDetected": self.surface_alignment_detected,
                "deepDisharmonyCategories": self.deep_disharmony_categories,
                "critiqueFindings": [f.as_dict() for f in self.critique_findings],
                "mandatoryRepair": self.mandatory_repair,
                "reinventionTriggered": self.reinvention_triggered,
                "originalPathType": self.original_path_type,
                "recommendedPathType": self.recommended_path_type,
                "mirrorScore": round(self.mirror_score, 4),
                "vulnerableNodeProtected": self.vulnerable_node_protected,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Mirror of Truth
# ---------------------------------------------------------------------------

class MirrorOfTruth:
    """Adversarial self-critique engine that deconstructs Dream Engine proposals.

    The Mirror sits between the Dream Engine and the Forge, inspecting the
    selected DreamPath for Surface Alignment that masks Deep Disharmony.

    Parameters
    ----------
    anchor : AxiomAnchor | None
        Shared Axiom Anchor for validation.
    vulnerability_priority : str
        The label of the node that must be protected above all others.
        Defaults to "Residential_Ratepayer" for the Grid War scenario.
    """

    def __init__(
        self,
        anchor: AxiomAnchor | None = None,
        vulnerability_priority: str = "Residential_Ratepayer",
    ) -> None:
        self.anchor = anchor or AxiomAnchor()
        self.vulnerability_priority = vulnerability_priority
        self._incentive_predicate = IncentiveStabilityPredicate()

    # -- Scenario loading (Sprint 9) ----------------------------------------

    @staticmethod
    def load_scenario(json_path: str) -> dict[str, Any]:
        """Load a conflict scenario from a JSON file.

        Parameters
        ----------
        json_path : str
            Path to the scenario JSON file.

        Returns
        -------
        dict
            The parsed scenario data.
        """
        with open(json_path, "r") as f:
            return json.load(f)

    @staticmethod
    def scenario_to_graph(scenario: dict[str, Any]) -> CategoricalGraph:
        """Convert a scenario's conflict_graph into a CategoricalGraph.

        Parameters
        ----------
        scenario : dict
            The parsed scenario data.

        Returns
        -------
        CategoricalGraph
            A graph constructed from the scenario's objects and morphisms.
        """
        cg = scenario.get("conflict_graph", {})
        source_text = scenario.get("description", "Loaded conflict scenario")
        graph = CategoricalGraph(source_text=source_text)

        # Build objects
        obj_map: dict[str, Object] = {}
        for obj_data in cg.get("objects", []):
            obj = graph.add_object(
                obj_data.get("label", "Unknown"),
                obj_data.get("tags", []),
            )
            obj_map[obj_data.get("id", "")] = obj

        # Build morphisms
        for morph_data in cg.get("morphisms", []):
            src = obj_map.get(morph_data.get("source", ""))
            tgt = obj_map.get(morph_data.get("target", ""))
            if src and tgt:
                graph.add_morphism(
                    morph_data.get("label", "Unknown"),
                    src, tgt,
                    morph_data.get("tags", []),
                )

        return graph

    # -- public API ---------------------------------------------------------

    def critique(
        self,
        dream_path: DreamPath,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
        possibility_report: PossibilityReport | None = None,
    ) -> RefinementTrace:
        """Perform Adversarial Deconstruction on a selected DreamPath.

        This is the core self-critique loop.  It subjects the Dream Engine's
        winner to multiple critique probes and produces a RefinementTrace
        with mandatory repairs.

        Parameters
        ----------
        dream_path : DreamPath
            The Dream Engine's selected solution path.
        report : DisharmonyReport
            The original disharmony report from the Deconstruction Engine.
        original_graph : CategoricalGraph
            The original (unhealed) categorical graph.
        possibility_report : PossibilityReport | None
            The full possibility report (for cross-path comparison).

        Returns
        -------
        RefinementTrace
            The Mirror's structured critique and repair mandate.
        """
        findings: list[CritiqueFinding] = []
        disharmony_categories: list[str] = []

        # Probe 1: Check for Surface Alignment in the original graph
        surface_detected = self._probe_surface_alignment(
            original_graph, findings,
        )

        # Probe 2a: Catalogue Deep Disharmony in the *original* graph
        # (the Mirror must always report what exists, even if the solution
        #  addresses it — this feeds the Refinement Panel display)
        self._probe_original_disharmony(
            original_graph, findings, disharmony_categories,
        )

        # Probe 2b: Check if the healed graph still contains residual
        # Deep Disharmony (tags that survived healing)
        self._probe_deep_disharmony(
            dream_path.healed_graph, original_graph, findings,
            disharmony_categories,
        )

        # Probe 3: Check vulnerable node protection
        vulnerable_protected = self._probe_vulnerable_protection(
            dream_path.healed_graph, findings,
        )

        # Probe 4: Check incentive stability of the proposed solution
        self._probe_incentive_stability(
            dream_path.healed_graph, report, findings,
            disharmony_categories,
        )

        # Probe 5: Check for cost-shifting patterns surviving reform
        self._probe_cost_shifting(
            dream_path.healed_graph, original_graph, findings,
            disharmony_categories,
        )

        # Probe 6: Shadow Entity detection — water sustainability (Sprint 9)
        self._probe_shadow_entity(
            dream_path.healed_graph, original_graph, findings,
            disharmony_categories,
        )

        # Compute mirror score — penalise for each finding
        mirror_score = self._compute_mirror_score(findings)

        # Determine if reinvention is needed
        reinvention_triggered = self._should_trigger_reinvention(
            dream_path, report, findings, mirror_score,
            vulnerable_protected,
        )

        # Generate mandatory repair
        mandatory_repair = self._generate_mandatory_repair(
            findings, disharmony_categories, dream_path,
        )

        # Determine recommended path
        recommended_path = self._recommend_path(
            dream_path, reinvention_triggered, possibility_report,
        )

        return RefinementTrace(
            surface_alignment_detected=surface_detected,
            deep_disharmony_categories=sorted(set(disharmony_categories)),
            critique_findings=findings,
            mandatory_repair=mandatory_repair,
            reinvention_triggered=reinvention_triggered,
            original_path_type=dream_path.path_type.value,
            recommended_path_type=recommended_path,
            mirror_score=mirror_score,
            vulnerable_node_protected=vulnerable_protected,
        )

    def critique_and_refine(
        self,
        possibility_report: PossibilityReport,
        report: DisharmonyReport,
        original_graph: CategoricalGraph,
    ) -> tuple[DreamPath, RefinementTrace]:
        """Critique the recommended path and refine if necessary.

        If the Mirror triggers a Reinvention override, selects the
        best alternative path from the PossibilityReport.

        Returns
        -------
        tuple[DreamPath, RefinementTrace]
            The (possibly overridden) path and the refinement trace.
        """
        # Find the recommended path
        recommended = possibility_report.recommended_path
        paths_by_type = {p.path_type.value: p for p in possibility_report.paths}
        selected = paths_by_type.get(recommended, possibility_report.paths[0])

        # Critique the selected path
        trace = self.critique(
            selected, report, original_graph, possibility_report,
        )

        # If reinvention is triggered and we're on a Reform path, override
        if trace.reinvention_triggered and selected.path_type == PathType.REFORM:
            # Try Reinvention first, then Dissolution
            for alt_type in (PathType.REINVENTION, PathType.DISSOLUTION):
                alt = paths_by_type.get(alt_type.value)
                if alt:
                    selected = alt
                    break

        return selected, trace

    # -- Probe 1: Surface Alignment ----------------------------------------

    def _probe_surface_alignment(
        self,
        original_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
    ) -> bool:
        """Detect Surface Alignment — claims that mask extraction.

        Surface Alignment occurs when a graph contains both positive-sounding
        tags (job creation, economic development) AND extractive mechanisms
        (cost shifting, tariff riders, shareholder primacy).
        """
        all_tags: set[str] = set()
        all_labels: set[str] = set()

        for obj in original_graph.objects:
            all_tags.update(t.lower() for t in obj.tags)
            all_labels.add(obj.label.lower())

        for morph in original_graph.morphisms:
            all_tags.update(t.lower() for t in morph.tags)
            all_labels.add(morph.label.lower().replace("_", " "))

        # Look for positive surface claims
        surface_claims = all_tags & _SURFACE_ALIGNMENT_CLAIMS
        # Look for deep disharmony indicators
        disharmony_signals = all_tags & _DEEP_DISHARMONY_TAGS

        if surface_claims and disharmony_signals:
            findings.append(CritiqueFinding(
                category="surface_alignment",
                severity=7.0,
                description=(
                    "Surface Alignment detected: the proposal presents "
                    "positive claims while concealing extractive mechanisms. "
                    "Claims do not survive adversarial scrutiny."
                ),
                evidence=[
                    f"Surface claims: {', '.join(sorted(surface_claims))}",
                    f"Hidden disharmony: {', '.join(sorted(disharmony_signals))}",
                ],
            ))
            return True

        # Also check for extractive morphisms that target vulnerable nodes
        if disharmony_signals:
            findings.append(CritiqueFinding(
                category="surface_alignment",
                severity=5.0,
                description=(
                    "Disharmony signals present without explicit surface "
                    "claims — extractive patterns detected in the graph."
                ),
                evidence=[
                    f"Disharmony tags: {', '.join(sorted(disharmony_signals))}",
                ],
            ))
            return True

        return False

    # -- Probe 2a: Original graph disharmony catalogue ---------------------

    def _probe_original_disharmony(
        self,
        original_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
        categories: list[str],
    ) -> None:
        """Catalogue Deep Disharmony present in the original graph.

        Even if the healed graph resolves these issues, the Mirror must
        record what categories of disharmony existed — this drives the
        mandatory repair and the Refinement Panel display.
        """
        original_tags: set[str] = set()
        for morph in original_graph.morphisms:
            original_tags.update(t.lower() for t in morph.tags)
        for obj in original_graph.objects:
            original_tags.update(t.lower() for t in obj.tags)

        for category, indicator_tags in _DISHARMONY_CATEGORIES.items():
            matched = original_tags & indicator_tags
            if matched:
                categories.append(category)
                findings.append(CritiqueFinding(
                    category=category,
                    severity=5.0,
                    description=(
                        f"Original system exhibits '{category}' disharmony "
                        f"pattern. The Mirror records this for the "
                        f"Refinement Trace."
                    ),
                    evidence=[
                        f"Indicator tags: {', '.join(sorted(matched))}",
                    ],
                ))

    # -- Probe 2b: Residual Deep Disharmony in healed graph ----------------

    def _probe_deep_disharmony(
        self,
        healed_graph: CategoricalGraph,
        original_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
        categories: list[str],
    ) -> None:
        """Check if the healed graph still contains deep disharmony patterns."""
        healed_tags: set[str] = set()
        for morph in healed_graph.morphisms:
            healed_tags.update(t.lower() for t in morph.tags)
        for obj in healed_graph.objects:
            healed_tags.update(t.lower() for t in obj.tags)

        # Check each disharmony category
        for category, indicator_tags in _DISHARMONY_CATEGORIES.items():
            residual = healed_tags & indicator_tags
            if residual:
                categories.append(category)
                findings.append(CritiqueFinding(
                    category=category,
                    severity=6.0,
                    description=(
                        f"Deep Disharmony persists in healed graph: "
                        f"'{category}' pattern not fully resolved."
                    ),
                    evidence=[
                        f"Residual tags: {', '.join(sorted(residual))}",
                    ],
                ))

    # -- Probe 3: Vulnerable Node Protection --------------------------------

    def _probe_vulnerable_protection(
        self,
        healed_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
    ) -> bool:
        """Check that the priority vulnerable node is protected in the solution."""
        # Find the priority node
        priority_node = None
        vulnerable_nodes: list[str] = []

        for obj in healed_graph.objects:
            if "vulnerable" in obj.tags:
                vulnerable_nodes.append(obj.label)
                if (self.vulnerability_priority.lower() in obj.label.lower()
                        or obj.label.lower() in self.vulnerability_priority.lower()):
                    priority_node = obj

        if not priority_node and not vulnerable_nodes:
            # No vulnerable nodes at all — may be a reinvention graph
            # with different labels; check for beneficiary tags
            for obj in healed_graph.objects:
                if "stakeholder" in obj.tags:
                    vulnerable_nodes.append(obj.label)
            if not vulnerable_nodes:
                findings.append(CritiqueFinding(
                    category="vulnerable_node_missing",
                    severity=8.0,
                    description=(
                        f"Priority vulnerable node '{self.vulnerability_priority}' "
                        f"is absent from the healed graph. The solution fails to "
                        f"represent the interests of the most vulnerable stakeholder."
                    ),
                    affected_nodes=[self.vulnerability_priority],
                ))
                return False

        # Check that vulnerable nodes are targets of protective morphisms
        protective_tags = {"protection", "care", "service", "empowerment"}
        protected_ids: set[str] = set()

        for morph in healed_graph.morphisms:
            morph_tags = {t.lower() for t in morph.tags}
            if morph_tags & protective_tags:
                protected_ids.add(morph.target)

        vulnerable_ids = {
            o.id for o in healed_graph.objects if "vulnerable" in o.tags
        }

        unprotected = vulnerable_ids - protected_ids
        if unprotected:
            unprotected_labels = [
                o.label for o in healed_graph.objects if o.id in unprotected
            ]
            findings.append(CritiqueFinding(
                category="insufficient_protection",
                severity=7.0,
                description=(
                    f"Vulnerable nodes lack protective morphisms: "
                    f"{', '.join(unprotected_labels)}. The solution does not "
                    f"adequately shield these stakeholders."
                ),
                affected_nodes=unprotected_labels,
            ))
            return False

        return True

    # -- Probe 4: Incentive Stability of proposed solution ------------------

    def _probe_incentive_stability(
        self,
        healed_graph: CategoricalGraph,
        report: DisharmonyReport,
        findings: list[CritiqueFinding],
        categories: list[str],
    ) -> None:
        """Check if the original incentive instability is resolved."""
        if not report.incentive_instability:
            return

        # Re-evaluate incentive stability on the healed graph
        artefact = healed_graph.as_artefact()
        healed_score, still_unstable = self._incentive_predicate.evaluate(artefact)

        if still_unstable:
            categories.append("shareholder_primacy_extraction")
            findings.append(CritiqueFinding(
                category="unresolved_incentive_instability",
                severity=9.0,
                description=(
                    f"Incentive instability persists (score: {healed_score:.1f}/10). "
                    f"The shareholder primacy gravity well has not been eliminated. "
                    f"A structural solution (Reinvention or Dissolution) is required."
                ),
                evidence=[
                    f"Original score: {report.incentive_stability_score:.1f}/10",
                    f"Healed score: {healed_score:.1f}/10",
                    "Shareholder primacy pattern still present",
                ],
            ))

    # -- Probe 5: Cost-shifting survival ------------------------------------

    def _probe_cost_shifting(
        self,
        healed_graph: CategoricalGraph,
        original_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
        categories: list[str],
    ) -> None:
        """Check if cost-shifting patterns survive the healing process.

        Cost-shifting is when infrastructure costs caused by one entity
        (e.g. data centres) are borne by another (e.g. residential ratepayers).
        """
        cost_shift_tags = {
            "cost_pass_through", "tariff_rider",
            "infrastructure_cost_pass_through", "cross_subsidy",
            "regressive_impact",
        }

        # Check original for cost-shifting
        original_has_cost_shift = False
        for morph in original_graph.morphisms:
            if {t.lower() for t in morph.tags} & cost_shift_tags:
                original_has_cost_shift = True
                break

        if not original_has_cost_shift:
            return

        # Check if healed graph still has any extraction targeting vulnerable
        for morph in healed_graph.morphisms:
            morph_tags = {t.lower() for t in morph.tags}
            if morph_tags & {"extraction", "exploitation"}:
                # Find if target is vulnerable
                target_obj = None
                for obj in healed_graph.objects:
                    if obj.id == morph.target and "vulnerable" in obj.tags:
                        target_obj = obj
                        break

                if target_obj:
                    categories.append("cost_shifting_to_vulnerable")
                    findings.append(CritiqueFinding(
                        category="residual_cost_shifting",
                        severity=8.0,
                        description=(
                            f"Cost-shifting to vulnerable node "
                            f"'{target_obj.label}' survives in the healed "
                            f"graph via morphism '{morph.label}'. "
                            f"This violates cost-causation principles."
                        ),
                        evidence=[
                            f"Morphism: {morph.label} ({morph.source} → {morph.target})",
                            f"Tags: {', '.join(morph.tags)}",
                        ],
                        affected_nodes=[target_obj.label],
                    ))

    # -- Probe 6: Shadow Entity detection (Sprint 9) -----------------------

    def _probe_shadow_entity(
        self,
        healed_graph: CategoricalGraph,
        original_graph: CategoricalGraph,
        findings: list[CritiqueFinding],
        categories: list[str],
    ) -> None:
        """Detect Shadow Entity patterns — invisible dependencies whose
        degradation is not priced into the proposal.

        The canonical example is the Oklahoma Water Supply: data centre
        cooling-water demand exceeds sustainable aquifer recharge rates,
        but this cost is invisible in the HILL request economics.

        A Shadow Entity is identified by the ``shadow_entity`` tag on a
        node.  If any morphism targets such a node with extraction or
        depletion tags, the Mirror flags an unsustainable withdrawal
        pattern.
        """
        # Identify shadow entities in the original graph
        shadow_nodes: list[str] = []
        for obj in original_graph.objects:
            if "shadow_entity" in obj.tags:
                shadow_nodes.append(obj.label)

        if not shadow_nodes:
            return

        # Check for extractive morphisms targeting shadow entities
        water_tags = {"water_depletion", "unsustainable_withdrawal", "shadow_cost"}
        for morph in original_graph.morphisms:
            morph_tags = {t.lower() for t in morph.tags}
            if morph_tags & water_tags:
                # Find source label
                src_label = morph.source
                for obj in original_graph.objects:
                    if obj.id == morph.source:
                        src_label = obj.label
                        break
                tgt_label = morph.target
                for obj in original_graph.objects:
                    if obj.id == morph.target:
                        tgt_label = obj.label
                        break

                categories.append("unsustainable_water_withdrawal")
                findings.append(CritiqueFinding(
                    category="shadow_entity_depletion",
                    severity=9.0,
                    description=(
                        f"Shadow Entity detected: '{tgt_label}' is an invisible "
                        f"dependency being depleted by '{src_label}' via "
                        f"'{morph.label}'. This cost is not reflected in the "
                        f"proposal economics. Blueprints that exceed sustainable "
                        f"thresholds must be penalised."
                    ),
                    evidence=[
                        f"Shadow entity: {tgt_label}",
                        f"Depleting morphism: {morph.label}",
                        f"Tags: {', '.join(sorted(morph_tags & water_tags))}",
                    ],
                    affected_nodes=[tgt_label, src_label],
                ))

        # Check if the healed graph addresses the shadow entity
        healed_has_protection = False
        for morph in healed_graph.morphisms:
            morph_tags = {t.lower() for t in morph.tags}
            if morph_tags & {"protection", "sustainable_management", "regulatory_oversight"}:
                # Check if target is shadow-entity-related
                for obj in healed_graph.objects:
                    if obj.id == morph.target and "shadow_entity" in obj.tags:
                        healed_has_protection = True
                        break

        if not healed_has_protection and shadow_nodes:
            findings.append(CritiqueFinding(
                category="shadow_entity_unaddressed",
                severity=7.0,
                description=(
                    f"Shadow Entities {shadow_nodes} remain unprotected in "
                    f"the healed graph. The solution does not include "
                    f"sustainable management morphisms for these invisible "
                    f"dependencies."
                ),
                affected_nodes=shadow_nodes,
            ))

    # -- Score computation --------------------------------------------------

    @staticmethod
    def _compute_mirror_score(findings: list[CritiqueFinding]) -> float:
        """Compute the Mirror's overall score (0–10, lower = more issues).

        Starts at 10 and is reduced by the severity of each finding,
        with diminishing returns to prevent scores going negative.
        """
        if not findings:
            return 10.0

        total_severity = sum(f.severity for f in findings)
        # Logistic-style reduction: fast drop for severe issues
        score = 10.0 * (1.0 / (1.0 + total_severity / 10.0))
        return max(0.0, min(10.0, score))

    # -- Reinvention trigger logic ------------------------------------------

    @staticmethod
    def _should_trigger_reinvention(
        dream_path: DreamPath,
        report: DisharmonyReport,
        findings: list[CritiqueFinding],
        mirror_score: float,
        vulnerable_protected: bool,
    ) -> bool:
        """Determine if the Mirror should trigger a Reinvention path override.

        Reinvention is triggered when:
        1. The current path is Reform (cosmetic fix)
        2. AND any of:
           - Incentive instability is present (shareholder primacy)
           - Mirror score < 5.0 (significant deep disharmony found)
           - Vulnerable node is not protected
           - High-severity findings in cost-shifting or extraction categories
        """
        if dream_path.path_type != PathType.REFORM:
            return False

        if report.incentive_instability:
            return True

        if mirror_score < 5.0:
            return True

        if not vulnerable_protected:
            return True

        # Check for high-severity structural findings
        structural_categories = {
            "shareholder_primacy_extraction",
            "unresolved_incentive_instability",
            "cost_shifting_to_vulnerable",
            "residual_cost_shifting",
        }
        for finding in findings:
            if (finding.category in structural_categories
                    and finding.severity >= 7.0):
                return True

        return False

    # -- Mandatory repair generation ----------------------------------------

    @staticmethod
    def _generate_mandatory_repair(
        findings: list[CritiqueFinding],
        categories: list[str],
        dream_path: DreamPath,
    ) -> str:
        """Generate a mandatory Regenerative Repair based on findings.

        The repair is a structured directive that must be applied before
        the blueprint can proceed to the Forge.
        """
        if not findings:
            return (
                "No adversarial findings. The proposal passes Mirror scrutiny. "
                "Proceed to the Forge with confidence."
            )

        repair_parts: list[str] = []
        unique_categories = sorted(set(categories))

        if "cost_shifting_to_vulnerable" in unique_categories or "residual_cost_shifting" in [f.category for f in findings]:
            repair_parts.append(
                "REPAIR-1 (Cost Causation): Require cost-causation-based tariff "
                "allocation where large-load customers pay the full marginal cost "
                "of the infrastructure they require. No socialisation of "
                "infrastructure costs onto residential ratepayers."
            )

        if "shareholder_primacy_extraction" in unique_categories or "unresolved_incentive_instability" in [f.category for f in findings]:
            repair_parts.append(
                "REPAIR-2 (Incentive Realignment): Eliminate the shareholder "
                "primacy gravity well by restructuring the utility as a benefit "
                "corporation, cooperative, or public power authority where "
                "fiduciary duty runs to ratepayers, not shareholders."
            )

        if "grid_fragility_amplification" in unique_categories:
            repair_parts.append(
                "REPAIR-3 (Fragility Mitigation): Require large-load customers "
                "to pre-fund grid infrastructure and demonstrate capacity "
                "availability before interconnection. Implement interruptible "
                "service agreements that protect grid reliability."
            )

        if "regressive_wealth_transfer" in unique_categories:
            repair_parts.append(
                "REPAIR-4 (Equity Protection): Implement progressive rate "
                "design that shields low-income residential customers from "
                "rate increases caused by large-load infrastructure. Establish "
                "a ratepayer protection fund financed by data centre impact fees."
            )

        if "unsustainable_water_withdrawal" in unique_categories or "shadow_entity_depletion" in [f.category for f in findings]:
            repair_parts.append(
                "REPAIR-5 (Water Sustainability): Require cooling-water demand "
                "to be reduced below sustainable aquifer recharge rates through "
                "closed-loop cooling, reclaimed water, or demand reduction. No "
                "blueprint may proceed that depletes the Oklahoma Water Supply "
                "shadow entity beyond sustainable thresholds."
            )

        if not repair_parts:
            # Generic repair for other findings
            highest = max(findings, key=lambda f: f.severity)
            repair_parts.append(
                f"REPAIR (General): Address the '{highest.category}' finding "
                f"(severity {highest.severity:.1f}/10) before proceeding. "
                f"Ensure the solution serves vulnerable stakeholders first."
            )

        return " | ".join(repair_parts)

    # -- Path recommendation ------------------------------------------------

    @staticmethod
    def _recommend_path(
        current: DreamPath,
        reinvention_triggered: bool,
        possibility_report: PossibilityReport | None,
    ) -> str:
        """Recommend the path type after Mirror analysis."""
        if not reinvention_triggered:
            return current.path_type.value

        # Prefer Reinvention over Dissolution when both are available
        if possibility_report:
            best_alt = None
            best_score = -1.0
            for path in possibility_report.paths:
                if path.path_type in (PathType.REINVENTION, PathType.DISSOLUTION):
                    score = path.unity_alignment_score + path.feasibility_score
                    if score > best_score:
                        best_score = score
                        best_alt = path
            if best_alt:
                return best_alt.path_type.value

        return PathType.REINVENTION.value

    # -- Scenario loading ---------------------------------------------------

    @staticmethod
    def load_scenario(path: str) -> dict[str, Any]:
        """Load a pre-defined conflict graph scenario from a JSON file.

        Parameters
        ----------
        path : str
            Path to the scenario JSON file.

        Returns
        -------
        dict
            The parsed scenario data.
        """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def scenario_to_graph(scenario: dict[str, Any]) -> CategoricalGraph:
        """Convert a scenario's conflict graph into a CategoricalGraph.

        Parameters
        ----------
        scenario : dict
            A scenario dictionary with a ``conflict_graph`` key containing
            ``objects`` and ``morphisms``.

        Returns
        -------
        CategoricalGraph
            The constructed categorical graph.
        """
        from genesis_engine.core.axiomlogix import Object, Morphism

        conflict = scenario["conflict_graph"]
        description = scenario.get("description", "Loaded scenario")

        graph = CategoricalGraph(source_text=description)

        # Build objects
        obj_map: dict[str, Object] = {}
        for obj_data in conflict["objects"]:
            obj = Object(
                id=obj_data["id"],
                label=obj_data["label"],
                tags=list(obj_data["tags"]),
            )
            graph.objects.append(obj)
            obj_map[obj_data["id"]] = obj

        # Build morphisms
        for morph_data in conflict["morphisms"]:
            morph = Morphism(
                id=morph_data["id"],
                label=morph_data["label"],
                source=morph_data["source"],
                target=morph_data["target"],
                tags=list(morph_data["tags"]),
            )
            graph.morphisms.append(morph)

        return graph
