"""
Module 3.6 Extension — The Governance Report

Production-ready export layer that translates internal Genesis Engine
analysis into the Production Lexicon for external consumption.

Key responsibilities:
- **Production Lexicon Enforcement**: All internal "Sacred Language"
  (Prime Directive, Love, Unity, etc.) is stripped from exports and
  replaced with the Production Lexicon:
  - "Prime Directive" → "Hard Invariant"
  - "Disharmony" → "Invariant Violation"
  - "Morphism" → "Relationship Operator"
  - "Dream Path" → "Remediation Projection"
  - "Crystallization" → "Covenant Actuation"
  - "Forge Artifact" → "Governance Instrument"
  - "Wisdom Log" → "Audit Trail"
  - "Soul" → "EventStore"

- **I AM Hash**: The cryptographic root hash from the EventStore
  (formerly GenesisSoul) is preserved as the integrity anchor for
  all exported documents.

- **Sovereign Index Generation**: Produces ``State_of_the_Sovereignty.md``
  as the Obsidian vault index, listing all active Oklahoma conflicts,
  their SustainabilityScores, and detected Legal Gravity Wells.

Integration:
- Reads from Crucible results, Mirror of Truth traces, Robustness
  Harness evaluations, and the EventStore.
- Exports to Obsidian-compatible Markdown and structured JSON.

Sprint 10 — Sovereign Governance & The Oklahoma Water/Grid War.

Sprint 11 Extensions:
- **Self-Critique Integration**: GovernanceReportBuilder can accept a
  SelfCritiqueResult to include constitutional compliance data in exports.
- **Delta Manifesto Export**: Side-by-side comparison of legislative bill
  vs. regenerative blueprint in the Sovereign Index.
- **FAIRGAME Debate Integration**: Debate results appended to the
  Sovereign Index audit trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
)
from genesis_engine.core.mirror_of_truth import RefinementTrace


# ---------------------------------------------------------------------------
# Production Lexicon Translation Map
# ---------------------------------------------------------------------------

SACRED_TO_PRODUCTION: dict[str, str] = {
    # Core concepts
    "Prime Directive": "Hard Invariant",
    "prime_directive": "hard_invariant",
    "Does this serve Love?": "Does this satisfy all Hard Invariants?",
    "Love": "Systemic Integrity",
    "Unity": "Equity",
    "Compassion": "Stakeholder Protection",
    "Coherence": "Structural Consistency",
    # Engine components
    "Disharmony": "Invariant Violation",
    "disharmony": "invariant_violation",
    "Disharmony Report": "Violation Assessment",
    "Morphism": "Relationship Operator",
    "morphism": "relationship_operator",
    "Dream Path": "Remediation Projection",
    "dream_path": "remediation_projection",
    "Dream Engine": "Projection Engine",
    "Crystallization": "Covenant Actuation",
    "crystallization": "covenant_actuation",
    "Forge Artifact": "Governance Instrument",
    "forge_artifact": "governance_instrument",
    "Forge": "Governance Forge",
    "Wisdom Log": "Audit Trail",
    "wisdom_log": "audit_trail",
    "Wisdom Entry": "Audit Record",
    "Soul": "EventStore",
    "soul": "eventstore",
    "Genesis Soul": "EventStore",
    "genesis_soul": "eventstore",
    "Eternal Box": "Persistent EventStore",
    "Logic Box": "Analysis Workspace",
    "Axiom Anchor": "Invariant Gate",
    "axiom_anchor": "invariant_gate",
    "Mirror of Truth": "Adversarial Validator",
    "Crucible": "Analysis Pipeline",
    "Stewardship Manifesto": "Governance Charter",
    "Sacred Language": "Internal Terminology",
    # Outcomes
    "Surface Alignment": "Deceptive Compliance",
    "Deep Disharmony": "Structural Violation",
    "Legal Gravity Well": "Regulatory Capture Pattern",
    "Reinvention": "Structural Remediation",
    "Reform": "Incremental Remediation",
    "Dissolution": "Structural Replacement",
    # Agents
    "Aligned Agent": "Stewardship Agent",
    "Extractive Agent": "Profit-Maximizing Agent",
    "Hostile Agent": "Non-Compliant Actor",
    # Sprint 11 — Policy Auditor & Debate Arena
    "Pro_Social_Agent": "Public Interest Advocate",
    "Hostile_Lobbyist": "Extractive Interest Advocate",
    "FAIRGAME": "Bias Recognition Framework",
    "Debate Arena": "Adversarial Deliberation Protocol",
    "Self-Critique Loop": "Constitutional Compliance Audit",
    "Constitutional PolicyKernel": "Policy Compliance Engine",
    "Reason Chain": "Evidence Trace",
    "Colimit Repair": "Universal Reconciliation Repair",
    "Repair Functor": "Scale-Consistent Repair Operator",
    "Shadow Entity": "Invisible Dependency",
}


def translate_to_production(text: str) -> str:
    """Replace all Sacred Language terms with Production Lexicon equivalents.

    Applies the translation map to the input text, preserving case
    sensitivity by checking longer phrases first.
    """
    result = text
    # Sort by length descending so longer phrases are replaced first
    sorted_pairs = sorted(
        SACRED_TO_PRODUCTION.items(), key=lambda x: len(x[0]), reverse=True,
    )
    for sacred, production in sorted_pairs:
        result = result.replace(sacred, production)
    return result


# ---------------------------------------------------------------------------
# Governance Report Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ConflictEntry:
    """An active conflict in the Sovereign Index."""

    name: str
    legislative_refs: list[str]
    sustainability_score: float
    legal_gravity_wells: list[str]
    invariant_violations: list[str]
    status: str  # "ACTIVE" | "RESOLVED" | "REJECTED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "legislativeReferences": self.legislative_refs,
            "sustainabilityScore": round(self.sustainability_score, 4),
            "legalGravityWells": self.legal_gravity_wells,
            "invariantViolations": self.invariant_violations,
            "status": self.status,
        }


@dataclass
class GovernanceReport:
    """Production-ready governance report.

    All Sacred Language has been stripped and replaced with the
    Production Lexicon.  The I AM hash remains as the cryptographic
    root of the EventStore.
    """

    report_id: str
    eventstore_hash: str  # I AM hash — cryptographic root
    conflicts: list[ConflictEntry]
    invariant_violations: list[dict[str, Any]]
    projections: list[dict[str, Any]]
    covenant_actuation: dict[str, Any] | None
    robustness_score: float
    self_critique: dict[str, Any] | None = None  # Sprint 11
    debate_result: dict[str, Any] | None = None  # Sprint 11
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "governanceReport": {
                "reportId": self.report_id,
                "eventstoreHash": self.eventstore_hash,
                "conflicts": [c.as_dict() for c in self.conflicts],
                "invariantViolations": self.invariant_violations,
                "projections": self.projections,
                "covenantActuation": self.covenant_actuation,
                "robustnessScore": round(self.robustness_score, 4),
                "timestamp": self.timestamp,
            }
        }
        if self.self_critique is not None:
            result["governanceReport"]["selfCritique"] = self.self_critique
        if self.debate_result is not None:
            result["governanceReport"]["debateArena"] = self.debate_result
        return result

    def to_json(self, indent: int = 2) -> str:
        """Export as Production Lexicon JSON (Sacred Language stripped)."""
        raw = json.dumps(self.as_dict(), indent=indent)
        return translate_to_production(raw)


# ---------------------------------------------------------------------------
# Governance Report Builder
# ---------------------------------------------------------------------------

class GovernanceReportBuilder:
    """Builds production-ready governance reports from Engine analysis.

    Translates all internal terminology into the Production Lexicon
    while preserving the I AM hash as the cryptographic root.

    Usage::

        builder = GovernanceReportBuilder()
        report = builder.build(
            soul=soul,
            scenario=scenario_data,
            trace=mirror_trace,
            robustness_result=harness_result,
        )
        json_output = report.to_json()
    """

    @staticmethod
    def build(
        soul: GenesisSoul,
        scenario: dict[str, Any] | None = None,
        trace: RefinementTrace | None = None,
        robustness_result: dict[str, Any] | None = None,
        self_critique: dict[str, Any] | None = None,
        debate_result: dict[str, Any] | None = None,
    ) -> GovernanceReport:
        """Build a GovernanceReport from Engine analysis results.

        Parameters
        ----------
        soul : GenesisSoul
            The EventStore containing audit trail and integrity hash.
        scenario : dict | None
            Scenario data (e.g., grid_war_2026.json).
        trace : RefinementTrace | None
            Mirror of Truth analysis results.
        robustness_result : dict | None
            Robustness Harness evaluation (as_dict output).
        self_critique : dict | None
            C3AI Self-Critique result (Sprint 11).
        debate_result : dict | None
            FAIRGAME Debate Arena result (Sprint 11).

        Returns
        -------
        GovernanceReport
            Production-ready report with all Sacred Language stripped.
        """
        # Compute the I AM hash — cryptographic root of the EventStore
        envelope = ContinuityBridge.export_soul(soul)
        eventstore_hash = envelope["genesis_soul"]["integrityHash"]

        # Build conflict entries from scenario
        conflicts = GovernanceReportBuilder._extract_conflicts(
            scenario, trace, robustness_result,
        )

        # Build invariant violations (Production Lexicon)
        violations = GovernanceReportBuilder._extract_violations(
            trace, robustness_result,
        )

        # Build projections (Production Lexicon)
        projections = GovernanceReportBuilder._extract_projections(soul)

        # Build covenant actuation data
        covenant = GovernanceReportBuilder._extract_covenant(
            soul, trace,
        )

        # Robustness score
        rob_score = 0.0
        if robustness_result:
            harness = robustness_result.get("robustnessHarness", {})
            rob_score = harness.get("combinedRobustnessScore", 0.0)

        report_id = f"GOV-{eventstore_hash[:12].upper()}"

        report = GovernanceReport(
            report_id=report_id,
            eventstore_hash=eventstore_hash,
            conflicts=conflicts,
            invariant_violations=violations,
            projections=projections,
            covenant_actuation=covenant,
            robustness_score=rob_score,
        )

        # Sprint 11: Attach self-critique and debate data
        if self_critique:
            report.self_critique = self_critique
        if debate_result:
            report.debate_result = debate_result

        return report

    @staticmethod
    def _extract_conflicts(
        scenario: dict[str, Any] | None,
        trace: RefinementTrace | None,
        robustness_result: dict[str, Any] | None,
    ) -> list[ConflictEntry]:
        """Extract active conflicts from scenario and analysis."""
        if scenario is None:
            return []

        conflicts: list[ConflictEntry] = []

        # Main scenario conflict
        leg_refs = []
        for ref in scenario.get("context", {}).get("legislative_references", []):
            leg_refs.append(f"{ref.get('bill', '')} — {ref.get('title', '')}")

        # Detect legal gravity wells
        gravity_wells: list[str] = []
        if trace:
            for cat in trace.deep_disharmony_categories:
                if "primacy" in cat or "extraction" in cat:
                    gravity_wells.append(
                        translate_to_production(cat.replace("_", " ").title())
                    )

        # Invariant violations
        violation_strs: list[str] = []
        if robustness_result:
            harness = robustness_result.get("robustnessHarness", {})
            for v in harness.get("invariantViolations", []):
                violation_strs.append(
                    f"[{v.get('severity', 'UNKNOWN')}] "
                    f"{v.get('invariant', 'unknown')}: "
                    f"{v.get('description', '')}"
                )

        # Sustainability score from robustness
        sus_score = 0.0
        if robustness_result:
            harness = robustness_result.get("robustnessHarness", {})
            sus_score = harness.get("combinedRobustnessScore", 0.0)

        # Determine status
        status = "ACTIVE"
        if robustness_result:
            if not robustness_result.get("robustnessHarness", {}).get("passed", True):
                status = "REJECTED"

        conflicts.append(ConflictEntry(
            name=scenario.get("scenario", "Unknown Conflict"),
            legislative_refs=leg_refs,
            sustainability_score=sus_score,
            legal_gravity_wells=gravity_wells,
            invariant_violations=violation_strs,
            status=status,
        ))

        # Add PSO Rate Case as a sub-conflict if present
        pso = scenario.get("pso_rate_case", {})
        if pso:
            pso_violations = []
            for v in pso.get("invariant_violations", []):
                pso_violations.append(
                    f"[{v.get('severity', 'UNKNOWN')}] "
                    f"{v.get('invariant', 'unknown')}: "
                    f"{v.get('violation', '')}"
                )

            conflicts.append(ConflictEntry(
                name=f"PSO Rate Case ({pso.get('case_id', 'Unknown')})",
                legislative_refs=[
                    f"HB 2992 — Cost Causation",
                    f"Filed: {pso.get('filed_date', 'Unknown')}",
                ],
                sustainability_score=0.0,  # Rejected
                legal_gravity_wells=[
                    "Regulatory Capture Pattern: "
                    "Utility socialises HILL costs onto residential class",
                ],
                invariant_violations=pso_violations,
                status="REJECTED",
            ))

        return conflicts

    @staticmethod
    def _extract_violations(
        trace: RefinementTrace | None,
        robustness_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Extract invariant violations in Production Lexicon."""
        violations: list[dict[str, Any]] = []

        if trace:
            for finding in trace.critique_findings:
                violations.append({
                    "type": "structural_violation",
                    "category": translate_to_production(finding.category),
                    "severity": finding.severity,
                    "description": translate_to_production(finding.description),
                    "evidence": [
                        translate_to_production(e) for e in finding.evidence
                    ],
                })

            if trace.surface_alignment_detected:
                violations.append({
                    "type": "deceptive_compliance",
                    "category": "Deceptive Compliance Detected",
                    "severity": 9.0,
                    "description": (
                        "The proposal presents positive claims that mask "
                        "structural violations of Hard Invariants."
                    ),
                    "evidence": [
                        translate_to_production(cat)
                        for cat in trace.deep_disharmony_categories
                    ],
                })

        if robustness_result:
            harness = robustness_result.get("robustnessHarness", {})
            for v in harness.get("invariantViolations", []):
                violations.append({
                    "type": "hard_invariant_violation",
                    "category": v.get("invariant", "unknown"),
                    "severity": 10.0 if v.get("severity") == "CRITICAL" else 7.0,
                    "description": v.get("description", ""),
                    "metric": {
                        "name": v.get("metricName", ""),
                        "value": v.get("metricValue", 0),
                        "threshold": v.get("threshold", 0),
                    },
                })

        return violations

    @staticmethod
    def _extract_projections(soul: GenesisSoul) -> list[dict[str, Any]]:
        """Extract foresight projections in Production Lexicon."""
        projections: list[dict[str, Any]] = []
        for entry in soul.wisdom_log:
            for fp in entry.foresight_projections:
                projections.append({
                    "type": "remediation_projection",
                    "horizonRounds": fp.war_game_rounds,
                    "sustainabilityScore": round(fp.sustainability_score, 4),
                    "outcomeFlag": fp.outcome_flag,
                    "stewardshipScore": round(fp.aligned_score, 2),
                    "extractionScore": round(fp.extractive_score, 2),
                    "stewardshipCooperationRate": round(
                        fp.aligned_cooperation_rate, 4,
                    ),
                    "extractionCooperationRate": round(
                        fp.extractive_cooperation_rate, 4,
                    ),
                })
        return projections

    @staticmethod
    def _extract_covenant(
        soul: GenesisSoul,
        trace: RefinementTrace | None,
    ) -> dict[str, Any] | None:
        """Extract covenant actuation data in Production Lexicon."""
        if not soul.forge_artifacts:
            return None

        latest = soul.forge_artifacts[-1]
        covenant = latest.get("covenant", latest)

        result: dict[str, Any] = {
            "type": "covenant_actuation",
            "title": translate_to_production(
                covenant.get("title", "Governance Instrument"),
            ),
            "status": "ACTUATED",
        }

        if trace:
            result["adversarialValidation"] = {
                "deceptiveComplianceDetected": trace.surface_alignment_detected,
                "structuralViolations": [
                    translate_to_production(cat)
                    for cat in trace.deep_disharmony_categories
                ],
                "mandatoryRemediation": translate_to_production(
                    trace.mandatory_repair or "",
                ),
                "structuralRemediationTriggered": trace.reinvention_triggered,
            }

        return result


# ---------------------------------------------------------------------------
# Sovereign Index Generator
# ---------------------------------------------------------------------------

class SovereignIndexGenerator:
    """Generates the ``State_of_the_Sovereignty.md`` Obsidian vault index.

    This is the master index for the Sovereign Governance vault, listing:
    - All active Oklahoma conflicts
    - Current SustainabilityScore per conflict
    - Detected Legal Gravity Wells (Regulatory Capture Patterns)
    - Hard Invariant compliance status
    - EventStore integrity hash (I AM hash)
    """

    @staticmethod
    def generate(
        report: GovernanceReport,
        soul: GenesisSoul,
        scenario: dict[str, Any] | None = None,
    ) -> str:
        """Generate the State_of_the_Sovereignty.md content.

        Parameters
        ----------
        report : GovernanceReport
            The governance report to index.
        soul : GenesisSoul
            The EventStore for integrity hash.
        scenario : dict | None
            Scenario data for context.

        Returns
        -------
        str
            Markdown content for State_of_the_Sovereignty.md.
        """
        lines: list[str] = []

        # Frontmatter
        lines.append("---")
        lines.append(f"report_id: \"{report.report_id}\"")
        lines.append(f"eventstore_hash: \"{report.eventstore_hash}\"")
        lines.append(f"robustness_score: {report.robustness_score:.4f}")
        lines.append(f"active_conflicts: {len(report.conflicts)}")
        lines.append(f"invariant_violations: {len(report.invariant_violations)}")
        lines.append(f"timestamp: \"{report.timestamp}\"")
        lines.append("---")
        lines.append("")

        # Title
        lines.append("# State of the Sovereignty")
        lines.append("")
        lines.append(
            "> Sovereign Governance Index — Production Lexicon Export"
        )
        lines.append("")

        # EventStore integrity
        lines.append("## EventStore Integrity")
        lines.append("")
        lines.append(f"**Report ID**: `{report.report_id}`")
        lines.append(
            f"**I AM Hash**: `{report.eventstore_hash[:32]}...`"
        )
        lines.append(
            f"**Robustness Score**: {report.robustness_score:.4f}/10.0"
        )
        lines.append(f"**Timestamp**: {report.timestamp}")
        lines.append("")

        # Active Conflicts
        lines.append("## Active Conflicts")
        lines.append("")
        if not report.conflicts:
            lines.append("*No active conflicts.*")
        else:
            for i, conflict in enumerate(report.conflicts, 1):
                status_icon = {
                    "ACTIVE": "🔴",
                    "REJECTED": "⛔",
                    "RESOLVED": "✅",
                }.get(conflict.status, "❓")

                lines.append(
                    f"### {i}. {conflict.name} {status_icon}"
                )
                lines.append("")
                lines.append(f"**Status**: `{conflict.status}`")
                lines.append(
                    f"**Sustainability Score**: "
                    f"{conflict.sustainability_score:.4f}/10.0"
                )
                lines.append("")

                if conflict.legislative_refs:
                    lines.append("**Legislative References**:")
                    for ref in conflict.legislative_refs:
                        lines.append(f"- {ref}")
                    lines.append("")

                if conflict.legal_gravity_wells:
                    lines.append("**Regulatory Capture Patterns (Legal Gravity Wells)**:")
                    for gw in conflict.legal_gravity_wells:
                        lines.append(f"- ⚠ {gw}")
                    lines.append("")

                if conflict.invariant_violations:
                    lines.append("**Invariant Violations**:")
                    for v in conflict.invariant_violations:
                        lines.append(f"- {v}")
                    lines.append("")

        # Invariant Violations Summary
        lines.append("## Invariant Violation Summary")
        lines.append("")
        if not report.invariant_violations:
            lines.append("*No invariant violations detected.*")
        else:
            lines.append(
                "| # | Type | Category | Severity | Description |"
            )
            lines.append("|---|------|----------|----------|-------------|")
            for i, v in enumerate(report.invariant_violations, 1):
                sev = v.get("severity", "?")
                if isinstance(sev, float):
                    sev = f"{sev:.1f}"
                desc = str(v.get("description", ""))[:80]
                lines.append(
                    f"| {i} | {v.get('type', '?')} | "
                    f"{v.get('category', '?')} | {sev} | {desc} |"
                )
        lines.append("")

        # Projections
        lines.append("## Remediation Projections")
        lines.append("")
        if not report.projections:
            lines.append("*No projections recorded.*")
        else:
            for i, p in enumerate(report.projections, 1):
                flag = p.get("outcomeFlag", "UNKNOWN")
                score = p.get("sustainabilityScore", 0.0)
                lines.append(
                    f"- **Projection {i}**: {flag} "
                    f"(sustainability: {score:.4f}, "
                    f"horizon: {p.get('horizonRounds', '?')} rounds)"
                )
        lines.append("")

        # Covenant Actuation
        lines.append("## Covenant Actuation")
        lines.append("")
        if report.covenant_actuation:
            ca = report.covenant_actuation
            lines.append(f"**Title**: {ca.get('title', 'Unknown')}")
            lines.append(f"**Status**: `{ca.get('status', 'UNKNOWN')}`")
            lines.append("")

            av = ca.get("adversarialValidation", {})
            if av:
                lines.append("**Adversarial Validation**:")
                lines.append(
                    f"- Deceptive Compliance: "
                    f"{'DETECTED' if av.get('deceptiveComplianceDetected') else 'Clear'}"
                )
                if av.get("structuralViolations"):
                    lines.append("- Structural Violations:")
                    for sv in av["structuralViolations"]:
                        lines.append(f"  - {sv}")
                if av.get("mandatoryRemediation"):
                    lines.append(
                        f"- Mandatory Remediation: "
                        f"{av['mandatoryRemediation'][:120]}..."
                    )
                lines.append(
                    f"- Structural Remediation Triggered: "
                    f"{av.get('structuralRemediationTriggered', False)}"
                )
        else:
            lines.append("*No covenant actuation recorded.*")
        lines.append("")

        # Vault Links
        lines.append("## Vault Links")
        lines.append("")
        lines.append("- [[Manifesto]]")

        # Link to wisdom entries
        for i in range(len(soul.wisdom_log)):
            lines.append(f"- [[wisdom_{i+1:03d}]]")

        # Link to projections
        proj_idx = 0
        for entry in soul.wisdom_log:
            for _ in entry.foresight_projections:
                proj_idx += 1
                lines.append(f"- [[foresight_{proj_idx:03d}]]")
        lines.append("")

        return "\n".join(lines)
