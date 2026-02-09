"""
Module 2.1 Extension -- The Constitutional PolicyKernel

A Reason-Based Constitution for governance policy evaluation.  Implements a
C3AI-style Self-Critique Loop where the system evaluates its own
GovernanceReport for bias, sycophancy, and constitutional violations before
the final export.

Key concepts:
- **Reason-Based Constitution** -- Every policy decision must be backed by
  an explicit *reason chain* that traces from observable evidence to the
  constitutional principle it satisfies or violates.
- **C3AI Self-Critique Loop** -- After generating a GovernanceReport, the
  PolicyKernel subjects the report to an internal critique that searches
  for:
  1. **Sycophancy Bias** -- Over-alignment with the dominant stakeholder's
     framing that suppresses legitimate counter-arguments.
  2. **Omission Bias** -- Failure to consider shadow entities, future
     generations, or vulnerable populations.
  3. **Confirmation Bias** -- Selective use of evidence that supports a
     pre-determined conclusion while ignoring contradictory data.
  4. **Anchoring Bias** -- Over-reliance on the first data point (e.g.,
     the HILL request's claimed economic benefits) when later evidence
     contradicts it.
- **Constitutional Principle** -- A named axiom against which policies
  are evaluated, each carrying a severity weight and a validation predicate.
- **SelfCritiqueResult** -- The structured output of the critique loop,
  containing bias detections, reason chains, and a constitutional compliance
  score that gates the final export.

Integration:
- Called from the Aria Interface before the Governance Report is finalised.
- Results feed into the SovereignIndexGenerator for the audit trail.
- The Self-Critique score gates the final export: reports below the
  constitutional threshold are flagged for human review.

Sprint 11 -- Policy Auditor & Regenerative Blueprint Suite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constitutional Principles
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """A named axiom in the Reason-Based Constitution.

    Each principle carries a severity weight (0.0-10.0) that determines
    how heavily a violation penalises the constitutional compliance score.

    Parameters
    ----------
    name : str
        Human-readable principle name (e.g., "Cost Causation").
    code : str
        Machine-readable identifier (e.g., "COST_CAUSATION").
    description : str
        Full statement of the constitutional principle.
    severity_weight : float
        How heavily violations of this principle penalise compliance.
    """

    name: str
    code: str
    description: str
    severity_weight: float = 8.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "severityWeight": round(self.severity_weight, 2),
        }


# Default Constitutional Principles
CONSTITUTIONAL_PRINCIPLES: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="Cost Causation",
        code="COST_CAUSATION",
        description=(
            "Infrastructure costs must be allocated to the entities that "
            "cause them. No socialisation of large-load costs onto "
            "residential ratepayers."
        ),
        severity_weight=9.0,
    ),
    ConstitutionalPrinciple(
        name="Water Floor",
        code="WATER_FLOOR",
        description=(
            "Residential water security must maintain a minimum surplus "
            "over industrial cooling demand across all simulation horizons."
        ),
        severity_weight=9.0,
    ),
    ConstitutionalPrinciple(
        name="Equity",
        code="EQUITY",
        description=(
            "Policy must not impose regressive cost burdens on vulnerable "
            "populations. Rate increases for infrastructure serving "
            "corporate loads are prima facie inequitable."
        ),
        severity_weight=8.0,
    ),
    ConstitutionalPrinciple(
        name="Sustainability",
        code="SUSTAINABILITY",
        description=(
            "No policy may be enacted that degrades long-term systemic "
            "health below the survival threshold across 50-year horizons."
        ),
        severity_weight=8.0,
    ),
    ConstitutionalPrinciple(
        name="Agency",
        code="AGENCY",
        description=(
            "All affected stakeholders must have meaningful participation "
            "in governance decisions that affect them."
        ),
        severity_weight=7.0,
    ),
    ConstitutionalPrinciple(
        name="Transparency",
        code="TRANSPARENCY",
        description=(
            "All policy rationale, cost allocations, and impact assessments "
            "must be publicly accessible and auditable."
        ),
        severity_weight=7.0,
    ),
    ConstitutionalPrinciple(
        name="Shadow Entity Protection",
        code="SHADOW_ENTITY_PROTECTION",
        description=(
            "Invisible dependencies (water supplies, ecosystems, future "
            "generations) must be explicitly accounted for in policy "
            "impact assessments."
        ),
        severity_weight=8.5,
    ),
]


# ---------------------------------------------------------------------------
# Reason Chain
# ---------------------------------------------------------------------------

@dataclass
class ReasonLink:
    """A single link in a reason chain.

    Traces from evidence to conclusion, supporting the constitutional
    evaluation of a policy decision.
    """

    evidence: str
    inference: str
    principle_code: str
    supports_compliance: bool
    confidence: float = 0.8  # 0.0-1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence,
            "inference": self.inference,
            "principleCode": self.principle_code,
            "supportsCompliance": self.supports_compliance,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ReasonChain:
    """A complete reason chain tracing policy evaluation logic.

    Each chain links observable evidence through inference steps to a
    constitutional principle, concluding with a compliance verdict.
    """

    chain_id: str
    principle: ConstitutionalPrinciple
    links: list[ReasonLink] = field(default_factory=list)
    verdict: str = "UNDETERMINED"  # "COMPLIANT" | "VIOLATION" | "UNDETERMINED"
    verdict_confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "chainId": self.chain_id,
            "principle": self.principle.as_dict(),
            "links": [link.as_dict() for link in self.links],
            "verdict": self.verdict,
            "verdictConfidence": round(self.verdict_confidence, 4),
        }


# ---------------------------------------------------------------------------
# Bias Detection
# ---------------------------------------------------------------------------

@dataclass
class BiasDetection:
    """A detected bias in the governance analysis.

    The C3AI Self-Critique Loop searches for cognitive biases that
    compromise the objectivity of the GovernanceReport.
    """

    bias_type: str  # "sycophancy" | "omission" | "confirmation" | "anchoring"
    description: str
    severity: float  # 0.0-10.0
    evidence: list[str] = field(default_factory=list)
    mitigation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "biasType": self.bias_type,
            "description": self.description,
            "severity": round(self.severity, 2),
            "evidence": self.evidence,
            "mitigation": self.mitigation,
        }


# ---------------------------------------------------------------------------
# Self-Critique Result
# ---------------------------------------------------------------------------

@dataclass
class SelfCritiqueResult:
    """Complete output of the C3AI Self-Critique Loop.

    Contains bias detections, reason chains, constitutional compliance
    score, and a gate decision that determines whether the report
    may be exported or requires human review.
    """

    critique_id: str
    bias_detections: list[BiasDetection] = field(default_factory=list)
    reason_chains: list[ReasonChain] = field(default_factory=list)
    constitutional_compliance_score: float = 10.0  # 0.0-10.0
    gate_passed: bool = True
    gate_threshold: float = 6.0
    human_review_required: bool = False
    summary: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "selfCritique": {
                "critiqueId": self.critique_id,
                "biasDetections": [b.as_dict() for b in self.bias_detections],
                "reasonChains": [rc.as_dict() for rc in self.reason_chains],
                "constitutionalComplianceScore": round(
                    self.constitutional_compliance_score, 4,
                ),
                "gatePassed": self.gate_passed,
                "gateThreshold": self.gate_threshold,
                "humanReviewRequired": self.human_review_required,
                "summary": self.summary,
                "timestamp": self.timestamp,
            }
        }


# ---------------------------------------------------------------------------
# PolicyKernel — The Constitutional Evaluation Engine
# ---------------------------------------------------------------------------

class PolicyKernel:
    """Constitutional PolicyKernel with Reason-Based evaluation and
    C3AI Self-Critique Loop.

    The PolicyKernel evaluates governance reports against a set of
    Constitutional Principles, building explicit reason chains that
    trace from evidence to conclusions.  After evaluation, it runs
    a self-critique loop that searches for cognitive biases in its
    own analysis.

    Parameters
    ----------
    principles : list[ConstitutionalPrinciple] | None
        Constitutional principles to evaluate against.
        Defaults to CONSTITUTIONAL_PRINCIPLES.
    gate_threshold : float
        Minimum constitutional compliance score to pass the gate.
    """

    def __init__(
        self,
        principles: list[ConstitutionalPrinciple] | None = None,
        gate_threshold: float = 6.0,
    ) -> None:
        self.principles = principles or list(CONSTITUTIONAL_PRINCIPLES)
        self.gate_threshold = gate_threshold

    # -- Public API ---------------------------------------------------------

    def evaluate(
        self,
        report_data: dict[str, Any],
        scenario: dict[str, Any] | None = None,
    ) -> SelfCritiqueResult:
        """Evaluate a GovernanceReport against the Constitution.

        Parameters
        ----------
        report_data : dict
            The governance report as a dictionary (from GovernanceReport.as_dict()).
        scenario : dict | None
            Optional scenario data for context-sensitive evaluation.

        Returns
        -------
        SelfCritiqueResult
            Complete critique with bias detections and reason chains.
        """
        gov = report_data.get("governanceReport", report_data)
        critique_id = self._compute_critique_id(report_data)

        # Phase 1: Build reason chains for each principle
        reason_chains = self._build_reason_chains(gov, scenario)

        # Phase 2: Detect biases in the report
        bias_detections = self._detect_biases(gov, scenario, reason_chains)

        # Phase 3: Compute constitutional compliance score
        compliance_score = self._compute_compliance_score(
            reason_chains, bias_detections,
        )

        # Phase 4: Gate decision
        gate_passed = compliance_score >= self.gate_threshold
        human_review = not gate_passed or len(bias_detections) > 0

        # Phase 5: Generate summary
        summary = self._generate_summary(
            reason_chains, bias_detections, compliance_score, gate_passed,
        )

        return SelfCritiqueResult(
            critique_id=critique_id,
            bias_detections=bias_detections,
            reason_chains=reason_chains,
            constitutional_compliance_score=compliance_score,
            gate_passed=gate_passed,
            gate_threshold=self.gate_threshold,
            human_review_required=human_review,
            summary=summary,
        )

    def critique_for_sycophancy(
        self,
        report_data: dict[str, Any],
        scenario: dict[str, Any] | None = None,
    ) -> list[BiasDetection]:
        """Run only the sycophancy detection probe.

        Specifically checks whether the report over-aligns with
        the dominant stakeholder's framing.

        Parameters
        ----------
        report_data : dict
            The governance report data.
        scenario : dict | None
            Optional scenario data.

        Returns
        -------
        list[BiasDetection]
            Detected sycophancy biases.
        """
        gov = report_data.get("governanceReport", report_data)
        biases: list[BiasDetection] = []
        self._probe_sycophancy(gov, scenario, biases)
        return biases

    # -- Phase 1: Reason Chain Construction ---------------------------------

    def _build_reason_chains(
        self,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
    ) -> list[ReasonChain]:
        """Build reason chains for each constitutional principle."""
        chains: list[ReasonChain] = []

        violations = gov.get("invariantViolations", [])
        conflicts = gov.get("conflicts", [])
        robustness = gov.get("robustnessScore", 0.0)

        for principle in self.principles:
            chain = ReasonChain(
                chain_id=f"RC-{principle.code}",
                principle=principle,
            )

            # Search for violations related to this principle
            related_violations = self._find_related_violations(
                principle, violations,
            )

            if related_violations:
                for v in related_violations:
                    chain.links.append(ReasonLink(
                        evidence=v.get("description", "Violation detected"),
                        inference=(
                            f"Violation of {principle.name} principle: "
                            f"{v.get('category', 'unknown')} "
                            f"(severity {v.get('severity', 0)})"
                        ),
                        principle_code=principle.code,
                        supports_compliance=False,
                        confidence=0.9,
                    ))
                chain.verdict = "VIOLATION"
                chain.verdict_confidence = min(
                    0.95,
                    0.7 + len(related_violations) * 0.08,
                )
            else:
                # Check scenario data for compliance evidence
                evidence = self._find_compliance_evidence(
                    principle, gov, scenario,
                )
                if evidence:
                    chain.links.append(ReasonLink(
                        evidence=evidence,
                        inference=(
                            f"Evidence supports compliance with "
                            f"{principle.name} principle."
                        ),
                        principle_code=principle.code,
                        supports_compliance=True,
                        confidence=0.7,
                    ))
                    chain.verdict = "COMPLIANT"
                    chain.verdict_confidence = 0.7
                else:
                    chain.links.append(ReasonLink(
                        evidence="No direct evidence found for or against.",
                        inference=(
                            f"Insufficient evidence to determine "
                            f"{principle.name} compliance."
                        ),
                        principle_code=principle.code,
                        supports_compliance=True,
                        confidence=0.4,
                    ))
                    chain.verdict = "UNDETERMINED"
                    chain.verdict_confidence = 0.4

            chains.append(chain)

        return chains

    def _find_related_violations(
        self,
        principle: ConstitutionalPrinciple,
        violations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find violations related to a specific principle."""
        code_lower = principle.code.lower()
        name_lower = principle.name.lower()
        related: list[dict[str, Any]] = []

        for v in violations:
            category = str(v.get("category", "")).lower()
            description = str(v.get("description", "")).lower()
            v_type = str(v.get("type", "")).lower()

            if (code_lower in category or name_lower in category
                    or code_lower in description or name_lower in description
                    or code_lower.replace("_", " ") in category
                    or code_lower.replace("_", " ") in description):
                related.append(v)
            elif principle.code == "EQUITY" and any(
                kw in category or kw in description
                for kw in ["equity", "ratepayer", "regressive", "residential"]
            ):
                related.append(v)
            elif principle.code == "SHADOW_ENTITY_PROTECTION" and any(
                kw in category or kw in description
                for kw in ["shadow", "water", "depletion", "invisible"]
            ):
                related.append(v)
            elif principle.code == "SUSTAINABILITY" and any(
                kw in category or kw in description
                for kw in ["sustainability", "collapse", "fragility", "blackout", "drought"]
            ):
                related.append(v)
            elif principle.code == "TRANSPARENCY" and any(
                kw in category or kw in description
                for kw in ["transparency", "opaque", "deceptive"]
            ):
                related.append(v)

        return related

    def _find_compliance_evidence(
        self,
        principle: ConstitutionalPrinciple,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
    ) -> str:
        """Find evidence of compliance with a principle."""
        robustness = gov.get("robustnessScore", 0.0)
        covenant = gov.get("covenantActuation", {})

        if principle.code == "SUSTAINABILITY" and robustness >= 7.0:
            return f"Robustness score {robustness:.2f}/10.0 exceeds sustainability threshold."
        if principle.code == "AGENCY" and covenant:
            av = covenant.get("adversarialValidation", {})
            if av and not av.get("deceptiveComplianceDetected", True):
                return "Covenant actuation completed without deceptive compliance."
        if principle.code == "TRANSPARENCY":
            if gov.get("eventstoreHash"):
                return "EventStore hash chain provides cryptographic audit trail."

        return ""

    # -- Phase 2: Bias Detection -------------------------------------------

    def _detect_biases(
        self,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
        reason_chains: list[ReasonChain],
    ) -> list[BiasDetection]:
        """Run all bias detection probes."""
        biases: list[BiasDetection] = []

        self._probe_sycophancy(gov, scenario, biases)
        self._probe_omission(gov, scenario, biases)
        self._probe_confirmation(gov, reason_chains, biases)
        self._probe_anchoring(gov, scenario, biases)

        return biases

    def _probe_sycophancy(
        self,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
        biases: list[BiasDetection],
    ) -> None:
        """Detect sycophancy bias -- over-alignment with dominant framing.

        Sycophancy occurs when the analysis accepts the dominant
        stakeholder's framing without sufficient adversarial scrutiny.
        In the Grid War context: accepting the data centre's economic
        development claims at face value while understating extraction.
        """
        violations = gov.get("invariantViolations", [])
        conflicts = gov.get("conflicts", [])
        robustness = gov.get("robustnessScore", 0.0)

        # Check: Are there CRITICAL violations but the report still passed?
        critical_violations = [
            v for v in violations
            if v.get("severity") == 10.0 or v.get("severity") == "CRITICAL"
        ]

        if critical_violations and robustness >= 7.0:
            biases.append(BiasDetection(
                bias_type="sycophancy",
                description=(
                    "Report shows CRITICAL violations but reports a high "
                    "robustness score. The analysis may be over-aligning "
                    "with the dominant stakeholder's positive framing."
                ),
                severity=7.0,
                evidence=[
                    f"{len(critical_violations)} CRITICAL violations detected",
                    f"Robustness score: {robustness:.2f}/10.0",
                ],
                mitigation=(
                    "Re-evaluate robustness score with CRITICAL violations "
                    "properly weighted. Consider whether the analysis has "
                    "suppressed negative findings."
                ),
            ))

        # Check: Does the report have conflicts marked ACTIVE with no violations?
        if scenario:
            for conflict in conflicts:
                if (conflict.get("status") == "ACTIVE"
                        and not conflict.get("invariantViolations")
                        and scenario.get("hill_request")):
                    # A HILL scenario with no violations is suspicious
                    biases.append(BiasDetection(
                        bias_type="sycophancy",
                        description=(
                            f"Conflict '{conflict.get('name', '?')}' is ACTIVE "
                            f"with no detected invariant violations despite "
                            f"a HILL request scenario. The analysis may be "
                            f"accepting the applicant's framing uncritically."
                        ),
                        severity=6.0,
                        evidence=[
                            f"HILL demand: {scenario.get('hill_request', {}).get('demand_mw', '?')} MW",
                            "Zero invariant violations for ACTIVE conflict",
                        ],
                        mitigation=(
                            "Apply adversarial scrutiny to the HILL request's "
                            "claimed benefits. Verify cost allocation and "
                            "water impact independently."
                        ),
                    ))

    def _probe_omission(
        self,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
        biases: list[BiasDetection],
    ) -> None:
        """Detect omission bias -- failure to consider shadow entities.

        Checks whether the analysis has adequately considered:
        - Shadow entities (water supplies, ecosystems)
        - Future generations
        - Vulnerable populations
        """
        violations = gov.get("invariantViolations", [])
        violation_categories = {
            v.get("category", "") for v in violations
        }

        # Check if shadow entities are considered
        has_shadow_analysis = any(
            "shadow" in str(v.get("description", "")).lower()
            or "shadow" in str(v.get("category", "")).lower()
            for v in violations
        )

        if scenario and not has_shadow_analysis:
            # Check if scenario has shadow entities
            objects = scenario.get("conflict_graph", {}).get("objects", [])
            shadow_entities = [
                obj for obj in objects
                if "shadow_entity" in obj.get("tags", [])
            ]
            if shadow_entities:
                biases.append(BiasDetection(
                    bias_type="omission",
                    description=(
                        "Shadow entities present in the scenario are not "
                        "reflected in the violation analysis. Invisible "
                        "dependencies may be omitted from the assessment."
                    ),
                    severity=7.0,
                    evidence=[
                        f"Shadow entities: {[e['label'] for e in shadow_entities]}",
                        "No shadow entity analysis in violations",
                    ],
                    mitigation=(
                        "Include shadow entity impact assessment in the "
                        "governance report. Evaluate water depletion, "
                        "ecosystem degradation, and intergenerational effects."
                    ),
                ))

        # Check if water analysis is present when basins exist
        if scenario:
            basin_constraints = scenario.get("constraints", {}).get(
                "basin_constraints", {},
            )
            has_water_analysis = any(
                "water" in str(v.get("description", "")).lower()
                or "water" in str(v.get("category", "")).lower()
                for v in violations
            )
            if basin_constraints and not has_water_analysis:
                biases.append(BiasDetection(
                    bias_type="omission",
                    description=(
                        "Water basin constraints are defined in the scenario "
                        "but no water-related violations appear in the "
                        "analysis. Basin sustainability may be omitted."
                    ),
                    severity=6.5,
                    evidence=[
                        f"Basin constraints: {list(basin_constraints.keys())}",
                        "No water-related violations in report",
                    ],
                    mitigation=(
                        "Run water floor invariant checks against basin "
                        "constraints. Verify cooling demand vs. recharge rates."
                    ),
                ))

    def _probe_confirmation(
        self,
        gov: dict[str, Any],
        reason_chains: list[ReasonChain],
        biases: list[BiasDetection],
    ) -> None:
        """Detect confirmation bias -- selective use of evidence.

        Checks whether the reason chains show a pattern of:
        - Only finding evidence that supports the dominant conclusion
        - Ignoring contradictory evidence
        """
        if not reason_chains:
            return

        compliant_chains = [
            rc for rc in reason_chains if rc.verdict == "COMPLIANT"
        ]
        violation_chains = [
            rc for rc in reason_chains if rc.verdict == "VIOLATION"
        ]

        # If no chains found violations, check for uniformity bias
        if len(violation_chains) == 0 and len(reason_chains) > 2:
            biases.append(BiasDetection(
                bias_type="confirmation",
                description=(
                    "No reason chains found any VIOLATION. This uniformity "
                    "may indicate confirmation bias -- the analysis may be "
                    "selectively finding supportive evidence while ignoring "
                    "contradictions."
                ),
                severity=5.0,
                evidence=[
                    f"{len(compliant_chains)}/{len(reason_chains)} chains COMPLIANT, 0 VIOLATION",
                    "No dissenting analysis found",
                ],
                mitigation=(
                    "Apply adversarial probes to each COMPLIANT chain. "
                    "Search for contradictory evidence that the analysis "
                    "may have overlooked."
                ),
            ))

        # Check for low-confidence chains being treated as definitive
        low_confidence_definitive = [
            rc for rc in reason_chains
            if rc.verdict_confidence < 0.5 and rc.verdict != "UNDETERMINED"
        ]
        if low_confidence_definitive:
            biases.append(BiasDetection(
                bias_type="confirmation",
                description=(
                    f"{len(low_confidence_definitive)} reason chain(s) have "
                    f"low confidence (<0.5) but reach definitive verdicts. "
                    f"Conclusions may be stronger than evidence warrants."
                ),
                severity=4.5,
                evidence=[
                    f"Chain {rc.chain_id}: verdict={rc.verdict}, "
                    f"confidence={rc.verdict_confidence:.2f}"
                    for rc in low_confidence_definitive
                ],
                mitigation=(
                    "Mark low-confidence chains as UNDETERMINED. "
                    "Gather additional evidence before concluding."
                ),
            ))

    def _probe_anchoring(
        self,
        gov: dict[str, Any],
        scenario: dict[str, Any] | None,
        biases: list[BiasDetection],
    ) -> None:
        """Detect anchoring bias -- over-reliance on first data point.

        In the Grid War context: checking if the analysis is anchored
        to the HILL request's claimed economic benefits rather than the
        full impact assessment.
        """
        if not scenario:
            return

        hill = scenario.get("hill_request", {})
        if not hill:
            return

        # Check if the report acknowledges the surface alignment
        violations = gov.get("invariantViolations", [])
        has_deceptive_check = any(
            "deceptive" in str(v.get("type", "")).lower()
            or "surface" in str(v.get("description", "")).lower()
            for v in violations
        )

        surface_claim = hill.get("surface_alignment_claim", "")
        deep_indicators = hill.get("deep_disharmony_indicators", [])

        if surface_claim and deep_indicators and not has_deceptive_check:
            biases.append(BiasDetection(
                bias_type="anchoring",
                description=(
                    "HILL request contains surface alignment claims and "
                    "documented deep disharmony indicators, but the report "
                    "does not flag deceptive compliance. The analysis may "
                    "be anchored to the applicant's initial framing."
                ),
                severity=6.0,
                evidence=[
                    f"Surface claim: {surface_claim[:100]}...",
                    f"Deep indicators: {len(deep_indicators)} documented",
                    "No deceptive compliance check in violations",
                ],
                mitigation=(
                    "Run adversarial deconstruction on the HILL request's "
                    "claimed benefits. Compare surface claims against deep "
                    "disharmony indicators."
                ),
            ))

    # -- Phase 3: Compliance Scoring ----------------------------------------

    def _compute_compliance_score(
        self,
        reason_chains: list[ReasonChain],
        bias_detections: list[BiasDetection],
    ) -> float:
        """Compute the constitutional compliance score (0.0-10.0).

        Factors:
        - Reason chain verdicts (weighted by principle severity)
        - Bias detections (penalty per bias)
        """
        if not reason_chains:
            return 5.0  # neutral score with no evidence

        # Score from reason chains
        total_weight = 0.0
        weighted_compliance = 0.0

        for chain in reason_chains:
            weight = chain.principle.severity_weight
            total_weight += weight

            if chain.verdict == "COMPLIANT":
                weighted_compliance += weight * chain.verdict_confidence
            elif chain.verdict == "UNDETERMINED":
                weighted_compliance += weight * 0.5 * chain.verdict_confidence
            # VIOLATION contributes 0

        chain_score = (
            (weighted_compliance / total_weight) * 10.0
            if total_weight > 0 else 5.0
        )

        # Bias penalty
        bias_penalty = sum(b.severity * 0.15 for b in bias_detections)

        final_score = max(0.0, min(10.0, chain_score - bias_penalty))
        return final_score

    # -- Phase 5: Summary Generation ----------------------------------------

    def _generate_summary(
        self,
        reason_chains: list[ReasonChain],
        bias_detections: list[BiasDetection],
        compliance_score: float,
        gate_passed: bool,
    ) -> str:
        """Generate a human-readable summary of the self-critique."""
        lines: list[str] = []

        violation_chains = [
            rc for rc in reason_chains if rc.verdict == "VIOLATION"
        ]
        compliant_chains = [
            rc for rc in reason_chains if rc.verdict == "COMPLIANT"
        ]
        undetermined_chains = [
            rc for rc in reason_chains if rc.verdict == "UNDETERMINED"
        ]

        lines.append(
            f"CONSTITUTIONAL SELF-CRITIQUE "
            f"(compliance: {compliance_score:.2f}/10.0)"
        )

        if gate_passed:
            lines.append("  GATE: PASSED -- Report may be exported.")
        else:
            lines.append(
                "  GATE: FAILED -- Report requires human review "
                "before export."
            )

        lines.append(
            f"  Reason chains: {len(violation_chains)} VIOLATION, "
            f"{len(compliant_chains)} COMPLIANT, "
            f"{len(undetermined_chains)} UNDETERMINED"
        )

        if violation_chains:
            lines.append("  Constitutional violations:")
            for rc in violation_chains:
                lines.append(
                    f"    - {rc.principle.name}: {rc.verdict} "
                    f"(confidence: {rc.verdict_confidence:.2f})"
                )

        if bias_detections:
            lines.append(f"  Bias detections: {len(bias_detections)}")
            for bd in bias_detections:
                lines.append(
                    f"    - [{bd.bias_type}] {bd.description[:80]}..."
                    if len(bd.description) > 80
                    else f"    - [{bd.bias_type}] {bd.description}"
                )

        return "\n".join(lines)

    # -- Utility -----------------------------------------------------------

    @staticmethod
    def _compute_critique_id(report_data: dict[str, Any]) -> str:
        """Compute a deterministic critique ID from the report data."""
        raw = json.dumps(report_data, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"CRT-{digest.upper()}"
