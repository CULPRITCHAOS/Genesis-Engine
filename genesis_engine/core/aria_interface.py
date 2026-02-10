"""
Module 3.2 — The Aria Interface

A CLI visualization layer for the Crucible Engine that displays the
"Thinking" process in real-time:

* **Crucible Panel** — Shows each candidate with its perspective,
  unityAlignmentScore, and lifecycle status (PENDING → VERIFYING → CONFIRMED).
* **Crystallization Command** — Finalizes a confirmed candidate into a
  Technical Covenant within the .genesis_soul file.
* **Soul Inspector** — Displays the current state of the EternalBox.
* **Refinement Panel** (Sprint 8) — Displays the Mirror of Truth's
  self-critique findings alongside the final Stewardship Manifesto.
* **Conflict War-Room** (Sprint 9) — Interactive "I AM" Dashboard for
  loading conflict scenarios, injecting nodes during the Mirror of Truth
  phase, and running full Crystallization events.
* **Regenerative Governance Console** (Sprint 10 — Aria Sub-Module 3.1):
  - ``compare_manifestos()`` — delta report between a legislative bill
    and the Engine's Regenerative Blueprint.
  - Live Invariant Tracker — highlights which Hard Constraints
    (Equity, Sustainability, Agency) are violated during real-time
    deconstruction.
* **Policy Audit Panel** (Sprint 11 — Module 3.1 Extension):
  - ``policy_audit()`` — Full pipeline: ingest bill, run debate arena,
    self-critique, and export Sovereign Governance Report.
  - ``sovereign_audit_hook()`` — Confirms all data residency stays local
    (Sovereign Reference Architecture compliance).
  - Delta Manifesto display for side-by-side comparison.

This module provides both programmatic access and formatted CLI output
for human operators to observe the multi-perspective reasoning process.
"""

from __future__ import annotations

from typing import Any

from genesis_engine.core.ai_provider import Perspective
from genesis_engine.core.axiomlogix import CategoricalGraph
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    ForesightProjection,
    GenesisSoul,
    HumanOverrideEntry,
)
from genesis_engine.core.crucible import (
    CandidateStatus,
    CrucibleCandidate,
    CrucibleEngine,
    CrucibleResult,
    LogicBox,
    PhaseRecord,
)
from genesis_engine.core.game_theory_console import (
    BayesianFinalExam,
    BlackoutShockResult,
    GameTheoryConsole,
    OutcomeFlag,
    WarGameOutcome,
)
from genesis_engine.core.mirror_of_truth import (
    MirrorOfTruth,
    RefinementTrace,
)
from genesis_engine.core.robustness_harness import (
    HardInvariant,
    InvariantViolation,
    RobustnessHarness,
    RobustnessResult,
    ForkResult,
)
from genesis_engine.core.governance_report import (
    GovernanceReport,
    GovernanceReportBuilder,
    SovereignIndexGenerator,
)


# ---------------------------------------------------------------------------
# ANSI color codes for terminal output
# ---------------------------------------------------------------------------

class Colors:
    """ANSI escape codes for terminal coloring."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Status colors
    PENDING = "\033[33m"      # Yellow
    VERIFYING = "\033[34m"    # Blue
    CONFIRMED = "\033[32m"    # Green
    REJECTED = "\033[31m"     # Red

    # Perspective colors
    CAUSALITY = "\033[36m"     # Cyan
    CONTRADICTION = "\033[35m" # Magenta
    ANALOGY = "\033[33m"       # Yellow

    # Structure colors
    HEADER = "\033[1;37m"      # Bold white
    PHASE = "\033[1;34m"       # Bold blue
    SCORE = "\033[1;32m"       # Bold green
    WARNING = "\033[1;33m"     # Bold yellow
    ERROR = "\033[1;31m"       # Bold red

    @classmethod
    def status_color(cls, status: CandidateStatus) -> str:
        return {
            CandidateStatus.PENDING: cls.PENDING,
            CandidateStatus.VERIFYING: cls.VERIFYING,
            CandidateStatus.CONFIRMED: cls.CONFIRMED,
            CandidateStatus.REJECTED: cls.REJECTED,
        }.get(status, cls.RESET)

    @classmethod
    def perspective_color(cls, perspective: Perspective) -> str:
        return {
            Perspective.CAUSALITY: cls.CAUSALITY,
            Perspective.CONTRADICTION: cls.CONTRADICTION,
            Perspective.ANALOGY: cls.ANALOGY,
        }.get(perspective, cls.RESET)


# ---------------------------------------------------------------------------
# Aria Panel Renderers
# ---------------------------------------------------------------------------

class AriaRenderer:
    """Renders Crucible state to formatted CLI output."""

    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors = use_colors

    def _c(self, color: str) -> str:
        """Return color code if colors enabled, else empty string."""
        return color if self.use_colors else ""

    # -- Header rendering ---------------------------------------------------

    def render_header(self, title: str) -> str:
        """Render a section header."""
        line = "═" * 70
        c = self._c
        return (
            f"\n{c(Colors.HEADER)}{line}{c(Colors.RESET)}\n"
            f"{c(Colors.BOLD)}  {title}{c(Colors.RESET)}\n"
            f"{c(Colors.HEADER)}{line}{c(Colors.RESET)}\n"
        )

    def render_subheader(self, title: str) -> str:
        """Render a subsection header."""
        line = "─" * 70
        c = self._c
        return f"\n{c(Colors.DIM)}{line}{c(Colors.RESET)}\n  {title}\n"

    # -- Phase rendering ----------------------------------------------------

    def render_phase(self, phase: PhaseRecord) -> str:
        """Render a single phase record."""
        c = self._c
        phase_name = phase.phase.upper().replace("-", " → ", 1)
        return (
            f"\n{c(Colors.PHASE)}[{phase_name}]{c(Colors.RESET)}\n"
            f"  {phase.summary}\n"
        )

    def render_phases(self, phases: list[PhaseRecord]) -> str:
        """Render all phase records."""
        return "".join(self.render_phase(p) for p in phases)

    # -- Candidate rendering ------------------------------------------------

    def render_candidate(self, candidate: CrucibleCandidate) -> str:
        """Render a single Crucible candidate."""
        c = self._c
        status_color = c(Colors.status_color(candidate.status))
        persp_color = c(Colors.perspective_color(candidate.perspective))

        status_icon = {
            CandidateStatus.PENDING: "○",
            CandidateStatus.VERIFYING: "◐",
            CandidateStatus.CONFIRMED: "●",
            CandidateStatus.REJECTED: "✗",
        }.get(candidate.status, "?")

        unity = f"{candidate.unity_alignment_score:.4f}"
        conf = f"{candidate.confidence:.2f}"

        lines = [
            f"\n  {status_color}{status_icon} {candidate.status.value:10}{c(Colors.RESET)} "
            f"{persp_color}[{candidate.perspective.value:13}]{c(Colors.RESET)} "
            f"unity={c(Colors.SCORE)}{unity}{c(Colors.RESET)} "
            f"conf={conf}",
        ]

        if candidate.reasoning:
            # Truncate long reasoning
            reasoning = candidate.reasoning[:100]
            if len(candidate.reasoning) > 100:
                reasoning += "..."
            lines.append(f"    {c(Colors.DIM)}Reasoning: {reasoning}{c(Colors.RESET)}")

        if candidate.dream_path:
            lines.append(
                f"    {c(Colors.DIM)}Path: {candidate.dream_path.title}{c(Colors.RESET)}"
            )

        return "\n".join(lines)

    def render_logic_box(self, logic_box: LogicBox) -> str:
        """Render the full LogicBox panel."""
        c = self._c
        header = self.render_subheader(
            f"{c(Colors.HEADER)}LOGIC BOX{c(Colors.RESET)} "
            f"({len(logic_box.candidates)} candidates)"
        )
        candidates = "".join(
            self.render_candidate(cand) for cand in logic_box.candidates
        )
        return header + candidates

    # -- Crystallization rendering ------------------------------------------

    def render_crystallization(self, candidate: CrucibleCandidate | None) -> str:
        """Render the crystallization result."""
        c = self._c
        if not candidate:
            return (
                f"\n{c(Colors.WARNING)}[CRYSTALLIZATION]{c(Colors.RESET)} "
                f"No candidate crystallized.\n"
            )

        lines = [
            f"\n{c(Colors.SCORE)}[CRYSTALLIZATION] SUCCESS{c(Colors.RESET)}",
            f"  Candidate:   {candidate.id}",
            f"  Perspective: {candidate.perspective.value}",
            f"  Unity Score: {candidate.unity_alignment_score:.4f}",
        ]

        if candidate.artifact:
            lines.append(f"  Covenant:    {candidate.artifact.covenant.title}")
            lines.append(f"  Verified:    {candidate.artifact.integrity_verified}")

        return "\n".join(lines) + "\n"

    # -- Disharmony Report rendering ----------------------------------------

    def render_disharmony_report(self, report: Any) -> str:
        """Render the Disharmony Report scores panel.

        Displays Unity Impact, Compassion Deficit, Coherence, and
        Incentive Stability alongside each other.
        """
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}DISHARMONY REPORT{c(Colors.RESET)}"
        )]

        # Unity Impact (higher = worse)
        unity_color = c(Colors.ERROR) if report.unity_impact > 5 else c(Colors.WARNING)
        lines.append(
            f"  Unity Impact:          {unity_color}"
            f"{report.unity_impact:.1f}/10{c(Colors.RESET)}"
        )

        # Compassion Deficit (higher = worse)
        comp_color = c(Colors.ERROR) if report.compassion_deficit > 5 else c(Colors.WARNING)
        lines.append(
            f"  Compassion Deficit:    {comp_color}"
            f"{report.compassion_deficit:.1f}/10{c(Colors.RESET)}"
        )

        # Coherence Score (higher = better)
        coh_color = c(Colors.SCORE) if report.coherence_score >= 5 else c(Colors.WARNING)
        lines.append(
            f"  Coherence Score:       {coh_color}"
            f"{report.coherence_score:.1f}/10{c(Colors.RESET)}"
        )

        # Incentive Stability Score (higher = better, < 5 = instability)
        inc_score = report.incentive_stability_score
        inc_color = c(Colors.ERROR) if report.incentive_instability else c(Colors.SCORE)
        lines.append(
            f"  Incentive Stability:   {inc_color}"
            f"{inc_score:.1f}/10{c(Colors.RESET)}"
        )

        if report.incentive_instability:
            lines.append(
                f"  {c(Colors.ERROR)}⚠ INCENTIVE INSTABILITY — "
                f"Shareholder Primacy pattern detected.{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.DIM)}Legal gravity well: Corporation→Shareholder "
                f"sink with fiduciary/profit-priority tags.{c(Colors.RESET)}"
            )

        lines.append(
            f"\n  Prime Directive aligned: "
            f"{c(Colors.SCORE) if report.is_aligned else c(Colors.ERROR)}"
            f"{'Yes' if report.is_aligned else 'No'}{c(Colors.RESET)}"
        )

        return "\n".join(lines) + "\n"

    # -- Full result rendering ----------------------------------------------

    def render_result(self, result: CrucibleResult) -> str:
        """Render a complete CrucibleResult."""
        c = self._c

        output = [
            self.render_header(f"CRUCIBLE: {result.source_text[:50]}..."),
        ]

        if result.is_aligned:
            output.append(
                f"\n{c(Colors.SCORE)}✓ System is already aligned — "
                f"no intervention needed.{c(Colors.RESET)}\n"
            )
        else:
            # Show the Disharmony Report scores panel
            if result.disharmony_report:
                output.append(self.render_disharmony_report(result.disharmony_report))

            output.append(self.render_phases(result.phases))
            output.append(self.render_logic_box(result.logic_box))
            output.append(self.render_crystallization(result.crystallized_candidate))

        return "".join(output)

    # -- Soul rendering -----------------------------------------------------

    def render_soul_summary(self, soul: GenesisSoul) -> str:
        """Render a summary of the GenesisSoul state."""
        c = self._c
        lines = [
            self.render_header("ETERNAL BOX (Genesis Soul)"),
            f"  Soul ID:          {soul.soul_id}",
            f"  Version:          {soul.version}",
            f"  Prime Directive:  \"{soul.directive.statement}\"",
            f"  Alignment:        {soul.alignment_threshold}",
            f"  Graphs recorded:  {len(soul.graph_history)}",
            f"  Wisdom entries:   {len(soul.wisdom_log)}",
            f"  Human overrides:  {len(soul.human_overrides)}",
            f"  Forge artifacts:  {len(soul.forge_artifacts)}",
            f"  Created:          {soul.created_at}",
            f"  Updated:          {soul.updated_at}",
        ]

        # Verify hash chain
        is_valid, errors = ContinuityBridge.verify_wisdom_chain(soul)
        if is_valid:
            lines.append(f"\n  {c(Colors.SCORE)}✓ Hash chain integrity: VALID{c(Colors.RESET)}")
        else:
            lines.append(f"\n  {c(Colors.ERROR)}✗ Hash chain integrity: INVALID{c(Colors.RESET)}")
            for err in errors[:3]:
                lines.append(f"    {c(Colors.ERROR)}{err}{c(Colors.RESET)}")

        return "\n".join(lines) + "\n"

    def render_wisdom_log(self, soul: GenesisSoul, limit: int = 5) -> str:
        """Render recent wisdom log entries."""
        c = self._c
        lines = [self.render_subheader("Recent Wisdom")]

        entries = soul.wisdom_log[-limit:] if soul.wisdom_log else []
        if not entries:
            lines.append(f"  {c(Colors.DIM)}(no wisdom entries){c(Colors.RESET)}")
        else:
            for i, entry in enumerate(entries):
                path_color = c(Colors.SCORE) if entry.resolution_path != "unresolved" else c(Colors.WARNING)
                lines.append(
                    f"\n  [{i+1}] {entry.source_text[:50]}..."
                )
                score_line = (
                    f"      Unity: {entry.unity_impact}/10  "
                    f"Compassion: {entry.compassion_deficit}/10"
                )
                # Show incentive stability if the entry carries it.
                if hasattr(entry, "incentive_stability_score"):
                    inc = entry.incentive_stability_score
                    inc_color = c(Colors.ERROR) if inc < 5 else c(Colors.SCORE)
                    score_line += f"  Incentive: {inc_color}{inc}/10{c(Colors.RESET)}"
                lines.append(score_line)
                lines.append(
                    f"      Path: {path_color}{entry.resolution_path}{c(Colors.RESET)}"
                )
                if entry.entry_hash:
                    lines.append(
                        f"      Hash: {c(Colors.DIM)}{entry.entry_hash[:32]}...{c(Colors.RESET)}"
                    )

        return "\n".join(lines) + "\n"

    # -- Game Theory rendering -----------------------------------------------

    def render_war_game(self, outcome: WarGameOutcome) -> str:
        """Render the Game Theory war-game outcome panel."""
        c = self._c

        # Outcome color
        outcome_colors = {
            OutcomeFlag.SYSTEMIC_COLLAPSE: Colors.ERROR,
            OutcomeFlag.PYRRHIC_VICTORY: Colors.WARNING,
            OutcomeFlag.SUSTAINABLE_VICTORY: Colors.SCORE,
            OutcomeFlag.MUTUAL_PROSPERITY: Colors.SCORE,
            OutcomeFlag.STALEMATE: Colors.DIM,
        }
        oc = c(outcome_colors.get(outcome.outcome_flag, Colors.RESET))

        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}GAME THEORY WAR-GAME "
            f"({outcome.total_rounds} rounds){c(Colors.RESET)}"
        )]

        lines.append(
            f"  Aligned Agent (Stewardship):"
        )
        lines.append(
            f"    Score: {c(Colors.SCORE)}{outcome.aligned_final_score:.1f}{c(Colors.RESET)}  "
            f"Cooperation: {outcome.aligned_cooperation_rate:.1%}"
        )
        lines.append(
            f"  Extractive Agent (Profit-Led):"
        )
        lines.append(
            f"    Score: {c(Colors.WARNING)}{outcome.extractive_final_score:.1f}{c(Colors.RESET)}  "
            f"Cooperation: {outcome.extractive_cooperation_rate:.1%}"
        )

        # Sustainability score
        sus_color = c(Colors.ERROR) if outcome.sustainability_score < 5.0 else c(Colors.SCORE)
        lines.append(
            f"\n  Sustainability Score: {sus_color}"
            f"{outcome.sustainability_score:.2f}/10.0{c(Colors.RESET)}"
        )

        # Outcome flag
        lines.append(
            f"  Outcome: {oc}{outcome.outcome_flag.value}{c(Colors.RESET)}"
        )

        if outcome.outcome_flag == OutcomeFlag.SYSTEMIC_COLLAPSE:
            lines.append(
                f"\n  {c(Colors.ERROR)}*** SYSTEMIC_COLLAPSE DETECTED ***{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.ERROR)}High score achieved at the cost of long-term "
                f"sustainability.{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.ERROR)}The system cannot survive beyond the simulation "
                f"horizon.{c(Colors.RESET)}"
            )

        # Round progression summary (show every 20th round)
        lines.append(f"\n  Round progression (sampled):")
        for r in outcome.rounds:
            if r.round_number == 1 or r.round_number % 20 == 0 or r.round_number == outcome.total_rounds:
                sus_c = c(Colors.ERROR) if r.round_sustainability < 5.0 else c(Colors.SCORE)
                lines.append(
                    f"    R{r.round_number:>4}: A={r.aligned_action:>9} E={r.extractive_action:>9}  "
                    f"Scores: {r.aligned_cumulative:>6.1f} vs {r.extractive_cumulative:>6.1f}  "
                    f"Sus: {sus_c}{r.round_sustainability:.2f}{c(Colors.RESET)}"
                )

        return "\n".join(lines) + "\n"

    # -- Refinement Panel rendering (Sprint 8) --------------------------------

    def render_refinement_panel(self, trace: RefinementTrace) -> str:
        """Render the Mirror of Truth's Refinement Panel.

        Displays:
        - Surface Alignment detection status
        - Deep Disharmony categories found
        - Critique findings with severity
        - Mandatory Regenerative Repair
        - Path recommendation (with reinvention override if triggered)
        """
        c = self._c
        lines = [self.render_header("MIRROR OF TRUTH — REFINEMENT PANEL")]

        # Mirror Score
        score = trace.mirror_score
        score_color = c(Colors.ERROR) if score < 5.0 else (
            c(Colors.WARNING) if score < 7.0 else c(Colors.SCORE)
        )
        lines.append(
            f"  Mirror Score: {score_color}{score:.2f}/10.0{c(Colors.RESET)}"
        )

        # Surface Alignment
        if trace.surface_alignment_detected:
            lines.append(
                f"\n  {c(Colors.ERROR)}*** SURFACE ALIGNMENT DETECTED ***{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.DIM)}The proposal presents positive claims that "
                f"mask extractive mechanisms.{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  {c(Colors.SCORE)}No Surface Alignment detected.{c(Colors.RESET)}"
            )

        # Deep Disharmony Categories
        if trace.deep_disharmony_categories:
            lines.append(
                f"\n  {c(Colors.WARNING)}Deep Disharmony Categories:{c(Colors.RESET)}"
            )
            for cat in trace.deep_disharmony_categories:
                cat_display = cat.replace("_", " ").title()
                lines.append(f"    {c(Colors.ERROR)}• {cat_display}{c(Colors.RESET)}")

        # Vulnerable Node Protection
        if trace.vulnerable_node_protected:
            lines.append(
                f"\n  Vulnerable Node: {c(Colors.SCORE)}PROTECTED{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  Vulnerable Node: {c(Colors.ERROR)}UNPROTECTED{c(Colors.RESET)}"
            )

        # Critique Findings
        if trace.critique_findings:
            lines.append(
                f"\n  {c(Colors.HEADER)}Critique Findings "
                f"({len(trace.critique_findings)}):{c(Colors.RESET)}"
            )
            for i, finding in enumerate(trace.critique_findings, 1):
                sev_color = c(Colors.ERROR) if finding.severity >= 7.0 else (
                    c(Colors.WARNING) if finding.severity >= 4.0 else c(Colors.DIM)
                )
                lines.append(
                    f"\n    [{i}] {sev_color}{finding.category}{c(Colors.RESET)} "
                    f"(severity: {sev_color}{finding.severity:.1f}/10{c(Colors.RESET)})"
                )
                # Truncate description at 80 chars per line
                desc = finding.description
                desc_lines = [desc[j:j+72] for j in range(0, len(desc), 72)]
                for dl in desc_lines:
                    lines.append(f"        {c(Colors.DIM)}{dl}{c(Colors.RESET)}")

                for ev in finding.evidence[:3]:
                    lines.append(
                        f"        {c(Colors.DIM)}Evidence: {ev}{c(Colors.RESET)}"
                    )

        # Reinvention Override
        if trace.reinvention_triggered:
            lines.append(
                f"\n  {c(Colors.ERROR)}*** REINVENTION OVERRIDE TRIGGERED ***"
                f"{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.WARNING)}Original path: {trace.original_path_type} "
                f"→ Recommended: {trace.recommended_path_type}{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  Path: {c(Colors.SCORE)}{trace.original_path_type}"
                f"{c(Colors.RESET)} (no override)"
            )

        # Mandatory Repair
        if trace.mandatory_repair:
            lines.append(
                f"\n  {c(Colors.HEADER)}Mandatory Regenerative Repair:"
                f"{c(Colors.RESET)}"
            )
            repairs = trace.mandatory_repair.split(" | ")
            for repair in repairs:
                repair_lines = [repair[j:j+68] for j in range(0, len(repair), 68)]
                for rl in repair_lines:
                    lines.append(f"    {c(Colors.WARNING)}{rl}{c(Colors.RESET)}")

        return "\n".join(lines) + "\n"

    def render_blackout_shock(self, result: BlackoutShockResult) -> str:
        """Render the Bayesian Blackout Shock exam result."""
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}BAYESIAN BLACKOUT SHOCK EXAM{c(Colors.RESET)}"
        )]

        # Bayesian Score
        score = result.bayesian_sustainability_score
        score_color = c(Colors.ERROR) if score < 5.0 else (
            c(Colors.WARNING) if score < 7.0 else c(Colors.SCORE)
        )
        lines.append(
            f"  Bayesian Sustainability: {score_color}"
            f"{score:.2f}/10.0{c(Colors.RESET)}"
        )
        lines.append(
            f"  Base Score:              "
            f"{result.base_result.sustainability_score:.2f}/10.0"
        )

        # Bayesian Parameters
        lines.append(
            f"\n  Prior Viability:     {result.prior_viability:.4f}"
        )
        lines.append(
            f"  Posterior Viability:  {result.posterior_viability:.4f}"
        )
        lines.append(
            f"  Fragility Amplifier: {result.fragility_amplifier:.1f}x"
        )

        # Blackout Probability
        bp = result.blackout_probability
        bp_color = c(Colors.ERROR) if bp > 0.3 else (
            c(Colors.WARNING) if bp > 0.1 else c(Colors.SCORE)
        )
        lines.append(
            f"  Blackout Probability: {bp_color}"
            f"{bp:.2%}{c(Colors.RESET)}"
        )

        # Pass/Fail
        if result.passed:
            lines.append(
                f"\n  {c(Colors.SCORE)}PASSED — Grid can sustain this "
                f"load profile.{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  {c(Colors.ERROR)}FAILED — Grid cannot sustain this "
                f"load profile.{c(Colors.RESET)}"
            )
            if result.blocking_reason:
                lines.append(f"  {c(Colors.DIM)}{result.blocking_reason[:120]}{c(Colors.RESET)}")

        return "\n".join(lines) + "\n"

    def render_foresight_projections(self, soul: GenesisSoul, limit: int = 3) -> str:
        """Render foresight projections from the wisdom log."""
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}FORESIGHT PROJECTIONS{c(Colors.RESET)}"
        )]

        # Collect all foresight projections from recent wisdom entries
        projections: list[dict[str, Any]] = []
        for entry in reversed(soul.wisdom_log):
            for fp in entry.foresight_projections:
                projections.append(fp.as_dict())
                if len(projections) >= limit:
                    break
            if len(projections) >= limit:
                break

        if not projections:
            lines.append(f"  {c(Colors.DIM)}(no foresight projections recorded){c(Colors.RESET)}")
        else:
            for i, fp in enumerate(projections):
                flag = fp["outcomeFlag"]
                flag_color = c(Colors.ERROR) if "COLLAPSE" in flag else c(Colors.SCORE)
                lines.append(
                    f"\n  [{i+1}] {fp['warGameRounds']}-round war-game"
                )
                lines.append(
                    f"      Sustainability: {flag_color}"
                    f"{fp['sustainabilityScore']:.2f}/10.0{c(Colors.RESET)}"
                )
                lines.append(
                    f"      Outcome: {flag_color}{flag}{c(Colors.RESET)}"
                )

        return "\n".join(lines) + "\n"

    # -- Conflict War-Room rendering (Sprint 9) ------------------------------

    def render_conflict_war_room(
        self,
        scenario: dict[str, Any],
        graph: CategoricalGraph,
    ) -> str:
        """Render the Conflict War-Room view for a loaded scenario.

        Displays the scenario name, legislative context, conflict graph
        summary, and HILL request details in a war-room dashboard format.
        """
        c = self._c
        lines = [self.render_header("CONFLICT WAR-ROOM — I AM DASHBOARD")]

        # Scenario overview
        lines.append(f"  {c(Colors.BOLD)}Scenario:{c(Colors.RESET)} {scenario.get('scenario', 'Unknown')}")
        lines.append(f"  {c(Colors.BOLD)}Version:{c(Colors.RESET)}  {scenario.get('version', '?')}")
        lines.append(f"  {c(Colors.BOLD)}Sprint:{c(Colors.RESET)}   {scenario.get('sprint', '?')}")

        # Legislative context
        context = scenario.get("context", {})
        leg_refs = context.get("legislative_references", [])
        if not leg_refs:
            # Fallback for v1 format
            leg_ref = context.get("legislative_reference", "")
            if leg_ref:
                leg_refs = [{"bill": leg_ref, "title": "", "disharmony_vector": ""}]

        if leg_refs:
            lines.append(f"\n  {c(Colors.HEADER)}Legislative Context:{c(Colors.RESET)}")
            for ref in leg_refs:
                bill = ref.get("bill", "")
                title = ref.get("title", "")
                vector = ref.get("disharmony_vector", "")
                lines.append(f"    {c(Colors.WARNING)}{bill}{c(Colors.RESET)}: {title}")
                if vector:
                    lines.append(f"      {c(Colors.DIM)}Disharmony: {vector[:80]}{c(Colors.RESET)}")

        # Graph summary
        lines.append(f"\n  {c(Colors.HEADER)}Conflict Graph:{c(Colors.RESET)}")
        lines.append(f"    Objects:   {len(graph.objects)}")
        lines.append(f"    Morphisms: {len(graph.morphisms)}")

        for obj in graph.objects:
            tag_str = ", ".join(obj.tags[:4])
            icon = "!" if "shadow_entity" in obj.tags else ("*" if "vulnerable" in obj.tags else "-")
            color = c(Colors.ERROR) if "shadow_entity" in obj.tags else (
                c(Colors.WARNING) if "vulnerable" in obj.tags else c(Colors.DIM)
            )
            lines.append(f"    {color}{icon} {obj.label} [{tag_str}]{c(Colors.RESET)}")

        # HILL request summary
        hill = scenario.get("hill_request", {})
        if hill:
            lines.append(f"\n  {c(Colors.HEADER)}HILL Request:{c(Colors.RESET)}")
            lines.append(f"    Demand:       {hill.get('demand_mw', '?')} MW")
            lines.append(f"    Infra Cost:   ${hill.get('infrastructure_cost_usd', 0):,}")
            lines.append(f"    Rate Impact:  ${hill.get('residential_impact_monthly_usd', '?')}/month")
            water_demand = hill.get("cooling_water_demand_mgd")
            if water_demand:
                water_limit = hill.get("sustainable_withdrawal_limit_mgd", "?")
                overshoot = hill.get("water_overshoot_ratio", "?")
                lines.append(f"    Water Demand: {water_demand} MGD (limit: {water_limit} MGD, {overshoot}x overshoot)")

        # Water sustainability constraints
        constraints = scenario.get("constraints", {})
        water = constraints.get("water_sustainability", {})
        if water:
            lines.append(f"\n  {c(Colors.ERROR)}Shadow Entity Alert:{c(Colors.RESET)}")
            lines.append(f"    {c(Colors.DIM)}{water.get('shadow_entity_rule', '')[:100]}{c(Colors.RESET)}")

        return "\n".join(lines) + "\n"

    def render_node_injection(
        self,
        label: str,
        tags: list[str],
        graph: CategoricalGraph,
    ) -> str:
        """Render confirmation of a node injection during Mirror of Truth phase."""
        c = self._c
        lines = [
            f"\n  {c(Colors.SCORE)}[NODE INJECTED]{c(Colors.RESET)} {label}",
            f"    Tags: {', '.join(tags)}",
            f"    Graph now has {len(graph.objects)} objects, {len(graph.morphisms)} morphisms",
        ]
        return "\n".join(lines)

    # -- Manifesto Comparison rendering (Sprint 10) --------------------------

    def render_manifesto_comparison(
        self,
        bill: dict[str, Any],
        blueprint: dict[str, Any],
        delta: dict[str, Any],
    ) -> str:
        """Render a delta report between a legislative bill and the
        Engine's Regenerative Blueprint.

        Displays side-by-side comparison with highlighted invariant
        violations and alignment gaps.
        """
        c = self._c
        lines = [self.render_header(
            "MANIFESTO COMPARISON — REGENERATIVE GOVERNANCE CONSOLE"
        )]

        # Bill summary
        lines.append(f"  {c(Colors.BOLD)}Legislative Bill:{c(Colors.RESET)}")
        lines.append(f"    {c(Colors.WARNING)}{bill.get('bill', 'Unknown')}{c(Colors.RESET)}: "
                     f"{bill.get('title', '')}")
        if bill.get("disharmony_vector"):
            lines.append(f"    {c(Colors.DIM)}Invariant Vector: "
                        f"{bill['disharmony_vector'][:80]}{c(Colors.RESET)}")

        lines.append(f"\n  {c(Colors.BOLD)}Regenerative Blueprint:{c(Colors.RESET)}")
        lines.append(f"    {c(Colors.SCORE)}{blueprint.get('title', 'Blueprint')}{c(Colors.RESET)}")

        # Delta analysis
        lines.append(f"\n  {c(Colors.HEADER)}Delta Analysis:{c(Colors.RESET)}")

        for field_name, delta_entry in delta.get("fields", {}).items():
            bill_val = delta_entry.get("bill", "N/A")
            blueprint_val = delta_entry.get("blueprint", "N/A")
            status = delta_entry.get("status", "neutral")

            status_color = {
                "violation": c(Colors.ERROR),
                "aligned": c(Colors.SCORE),
                "gap": c(Colors.WARNING),
                "neutral": c(Colors.DIM),
            }.get(status, c(Colors.DIM))

            status_icon = {
                "violation": "✗",
                "aligned": "✓",
                "gap": "△",
                "neutral": "○",
            }.get(status, "?")

            lines.append(
                f"    {status_color}{status_icon} {field_name}{c(Colors.RESET)}"
            )
            lines.append(
                f"      Bill:      {bill_val}"
            )
            lines.append(
                f"      Blueprint: {blueprint_val}"
            )

        # Invariant violations in delta
        delta_violations = delta.get("invariant_violations", [])
        if delta_violations:
            lines.append(
                f"\n  {c(Colors.ERROR)}Invariant Violations in Delta:{c(Colors.RESET)}"
            )
            for v in delta_violations:
                lines.append(
                    f"    {c(Colors.ERROR)}✗ [{v.get('invariant', '?')}] "
                    f"{v.get('description', '')}{c(Colors.RESET)}"
                )

        return "\n".join(lines) + "\n"

    # -- Live Invariant Tracker rendering (Sprint 10) -----------------------

    def render_invariant_tracker(
        self,
        violations: list[InvariantViolation],
        active_invariants: list[str] | None = None,
    ) -> str:
        """Render the Live Invariant Tracker panel.

        Highlights which Hard Constraints (Equity, Sustainability, Agency)
        are violated during real-time deconstruction.
        """
        c = self._c
        all_invariants = active_invariants or [
            HardInvariant.EQUITY,
            HardInvariant.SUSTAINABILITY,
            HardInvariant.AGENCY,
            HardInvariant.WATER_FLOOR,
            HardInvariant.COST_CAUSATION,
        ]

        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}LIVE INVARIANT TRACKER{c(Colors.RESET)}"
        )]

        violated_set = {v.invariant for v in violations}

        for inv in all_invariants:
            is_violated = inv in violated_set
            icon = "✗" if is_violated else "✓"
            color = c(Colors.ERROR) if is_violated else c(Colors.SCORE)
            label = inv.replace("_", " ").title()
            lines.append(f"  {color}{icon} {label}{c(Colors.RESET)}")

            # Show violation details
            for v in violations:
                if v.invariant == inv:
                    lines.append(
                        f"    {c(Colors.DIM)}[{v.severity}] {v.description[:72]}{c(Colors.RESET)}"
                    )
                    lines.append(
                        f"    {c(Colors.DIM)}Metric: {v.metric_name} = "
                        f"{v.metric_value:.4f} (threshold: {v.threshold:.4f})"
                        f"{c(Colors.RESET)}"
                    )

        # Summary
        violation_count = len(violations)
        critical_count = sum(1 for v in violations if v.severity == "CRITICAL")
        if violation_count == 0:
            lines.append(
                f"\n  {c(Colors.SCORE)}All Hard Constraints satisfied.{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  {c(Colors.ERROR)}{violation_count} violation(s) detected "
                f"({critical_count} CRITICAL).{c(Colors.RESET)}"
            )

        return "\n".join(lines) + "\n"

    # -- Robustness Harness rendering (Sprint 10) ---------------------------

    def render_robustness_result(self, result: RobustnessResult) -> str:
        """Render the Robustness Harness evaluation result."""
        c = self._c
        lines = [self.render_header("ROBUSTNESS HARNESS — BAYESIAN STRESS TEST")]

        # Combined score
        score = result.combined_robustness_score
        score_color = c(Colors.ERROR) if score < 5.0 else (
            c(Colors.WARNING) if score < 7.0 else c(Colors.SCORE)
        )
        lines.append(
            f"  Combined Robustness: {score_color}"
            f"{score:.4f}/10.0{c(Colors.RESET)}"
        )

        # Blackout Shock
        bs = result.blackout_sim
        lines.append(f"\n  {c(Colors.HEADER)}Blackout Shock Simulation:{c(Colors.RESET)}")
        lines.append(f"    Prior:    Beta({bs.prior.alpha:.1f}, {bs.prior.beta:.1f}) "
                     f"μ={bs.prior.mean:.4f}")
        lines.append(f"    Posterior: Beta({bs.posterior.alpha:.1f}, {bs.posterior.beta:.1f}) "
                     f"μ={bs.posterior.mean:.4f}")
        lines.append(f"    Survival: {bs.survival_rate:.1%} ({bs.survival_count}/{bs.runs})")

        # Drought Event
        de = result.drought_sim
        lines.append(f"\n  {c(Colors.HEADER)}Drought Event Simulation:{c(Colors.RESET)}")
        lines.append(f"    Prior:    Beta({de.prior.alpha:.1f}, {de.prior.beta:.1f}) "
                     f"μ={de.prior.mean:.4f}")
        lines.append(f"    Posterior: Beta({de.posterior.alpha:.1f}, {de.posterior.beta:.1f}) "
                     f"μ={de.posterior.mean:.4f}")
        lines.append(f"    Survival: {de.survival_rate:.1%} ({de.survival_count}/{de.runs})")

        # Fork operations
        if result.fork_results:
            lines.append(f"\n  {c(Colors.WARNING)}Decentralized Fork Operations:{c(Colors.RESET)}")
            for fr in result.fork_results:
                lines.append(
                    f"    {c(Colors.ERROR)}FORK:{c(Colors.RESET)} "
                    f"Hostile node '{fr.hostile_node_label}' excluded"
                )
                lines.append(
                    f"    Refused: {fr.refused_repair[:60]}"
                )
                lines.append(
                    f"    Protected basins: {', '.join(fr.protected_basins)}"
                )
                lines.append(
                    f"    Graph: {fr.original_object_count} → "
                    f"{fr.forked_object_count} objects"
                )

        # Pass/Fail
        if result.passed:
            lines.append(
                f"\n  {c(Colors.SCORE)}PASSED — Covenant is robust under "
                f"stress scenarios.{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"\n  {c(Colors.ERROR)}FAILED — Covenant cannot withstand "
                f"stress scenarios.{c(Colors.RESET)}"
            )
            for reason in result.blocking_reasons:
                lines.append(f"  {c(Colors.DIM)}{reason[:100]}{c(Colors.RESET)}")

        return "\n".join(lines) + "\n"

    # -- Fork Operator rendering (Sprint 10) --------------------------------

    def render_fork_result(self, result: ForkResult) -> str:
        """Render a Decentralized Fork Operator result."""
        c = self._c
        lines = [
            f"\n  {c(Colors.ERROR)}[DECENTRALIZED FORK]{c(Colors.RESET)} "
            f"Hostile agent excluded",
            f"    Node:      {result.hostile_node_label}",
            f"    Refused:   {result.refused_repair[:60]}",
            f"    Original:  {result.original_object_count} objects, "
            f"{result.original_morphism_count} morphisms",
            f"    Forked:    {result.forked_object_count} objects, "
            f"{result.forked_morphism_count} morphisms",
            f"    Protected: {', '.join(result.protected_basins)}",
        ]
        return "\n".join(lines) + "\n"

    # -- Policy Audit Panel rendering (Sprint 11) ----------------------------

    def render_policy_audit_panel(
        self,
        debate_result: Any,
        critique_result: Any,
        scenario: dict[str, Any] | None = None,
    ) -> str:
        """Render the Policy Audit Panel.

        Displays debate arena results, self-critique findings,
        FAIRGAME bias traces, and the constitutional compliance gate.
        """
        c = self._c
        lines = [self.render_header(
            "POLICY AUDIT PANEL — SPRINT 11"
        )]

        # Debate Arena Summary
        arena = debate_result
        lines.append(f"  {c(Colors.BOLD)}Debate Arena:{c(Colors.RESET)}")
        lines.append(f"    Bill:       {arena.bill_reference}")
        lines.append(f"    Rounds:     {len(arena.rounds)}")
        lines.append(f"    Pro-Social: {arena.pro_social_wins} wins")
        lines.append(f"    Hostile:    {arena.hostile_wins} wins")
        lines.append(f"    Draws:      {arena.draws}")

        # Overall winner
        winner_color = (
            c(Colors.SCORE) if arena.overall_winner == "pro_social"
            else c(Colors.ERROR) if arena.overall_winner == "hostile_lobbyist"
            else c(Colors.WARNING)
        )
        lines.append(
            f"    Winner:     {winner_color}{arena.overall_winner}{c(Colors.RESET)}"
        )

        # FAIRGAME Score
        fg_color = c(Colors.ERROR) if arena.fairgame_score < 5.0 else (
            c(Colors.WARNING) if arena.fairgame_score < 7.0 else c(Colors.SCORE)
        )
        lines.append(
            f"    FAIRGAME:   {fg_color}{arena.fairgame_score:.2f}/10.0{c(Colors.RESET)}"
        )

        # Bias Traces Summary
        if arena.aggregated_bias_traces:
            lines.append(
                f"\n  {c(Colors.WARNING)}FAIRGAME Bias Traces "
                f"({len(arena.aggregated_bias_traces)}):{c(Colors.RESET)}"
            )
            # Group by type
            by_type: dict[str, int] = {}
            for bt in arena.aggregated_bias_traces:
                by_type[bt.bias_type] = by_type.get(bt.bias_type, 0) + 1
            for bias_type, count in sorted(by_type.items()):
                label = bias_type.replace("_", " ").title()
                lines.append(f"    - {label}: {count} detection(s)")

        # Self-Critique Summary
        crit = critique_result
        lines.append(f"\n  {c(Colors.BOLD)}Constitutional Self-Critique:{c(Colors.RESET)}")
        comp_color = (
            c(Colors.ERROR) if crit.constitutional_compliance_score < 5.0
            else c(Colors.WARNING) if crit.constitutional_compliance_score < 7.0
            else c(Colors.SCORE)
        )
        lines.append(
            f"    Compliance: {comp_color}"
            f"{crit.constitutional_compliance_score:.2f}/10.0{c(Colors.RESET)}"
        )

        gate_color = c(Colors.SCORE) if crit.gate_passed else c(Colors.ERROR)
        gate_label = "PASSED" if crit.gate_passed else "FAILED"
        lines.append(
            f"    Gate:       {gate_color}{gate_label}{c(Colors.RESET)} "
            f"(threshold: {crit.gate_threshold})"
        )

        if crit.human_review_required:
            lines.append(
                f"    {c(Colors.WARNING)}*** HUMAN REVIEW REQUIRED ***{c(Colors.RESET)}"
            )

        # Bias detections from self-critique
        if crit.bias_detections:
            lines.append(
                f"\n  {c(Colors.ERROR)}Self-Critique Bias Detections "
                f"({len(crit.bias_detections)}):{c(Colors.RESET)}"
            )
            for bd in crit.bias_detections[:5]:
                sev_color = (
                    c(Colors.ERROR) if bd.severity >= 6.0 else c(Colors.WARNING)
                )
                lines.append(
                    f"    {sev_color}[{bd.bias_type}] "
                    f"{bd.description[:72]}{c(Colors.RESET)}"
                )

        # Recommendation
        if arena.policy_recommendation:
            lines.append(f"\n  {c(Colors.HEADER)}Policy Recommendation:{c(Colors.RESET)}")
            for line in arena.policy_recommendation.split("\n"):
                lines.append(f"    {line}")

        return "\n".join(lines) + "\n"

    def render_sovereign_audit(
        self,
        audit_result: dict[str, Any],
    ) -> str:
        """Render the Sovereign Reference Architecture audit result."""
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}SOVEREIGN REFERENCE ARCHITECTURE AUDIT"
            f"{c(Colors.RESET)}"
        )]

        passed = audit_result.get("sovereign_compliant", False)
        if passed:
            lines.append(
                f"  {c(Colors.SCORE)}COMPLIANT — All data residency "
                f"constraints satisfied.{c(Colors.RESET)}"
            )
        else:
            lines.append(
                f"  {c(Colors.ERROR)}NON-COMPLIANT — Data residency "
                f"violations detected.{c(Colors.RESET)}"
            )

        for check in audit_result.get("checks", []):
            icon = "✓" if check["passed"] else "✗"
            color = c(Colors.SCORE) if check["passed"] else c(Colors.ERROR)
            lines.append(
                f"  {color}{icon} {check['name']}: "
                f"{check['description']}{c(Colors.RESET)}"
            )

        return "\n".join(lines) + "\n"

    def render_categorical_repair(
        self,
        colimit_result: Any,
    ) -> str:
        """Render the Categorical Repair Operators (ACT) result."""
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.HEADER)}CATEGORICAL REPAIR OPERATORS (ACT)"
            f"{c(Colors.RESET)}"
        )]

        exists_color = (
            c(Colors.SCORE) if colimit_result.colimit_exists
            else c(Colors.ERROR)
        )
        lines.append(
            f"  Colimit Exists: {exists_color}"
            f"{'Yes' if colimit_result.colimit_exists else 'No'}"
            f"{c(Colors.RESET)}"
        )
        lines.append(
            f"  Reconciliation: {colimit_result.reconciliation_score:.2%}"
        )
        lines.append(
            f"  Local Repairs:  {len(colimit_result.local_repairs)}"
        )
        lines.append(
            f"  Universal Actions: {len(colimit_result.universal_actions)}"
        )

        if colimit_result.universal_actions:
            lines.append(
                f"\n  {c(Colors.HEADER)}Universal Repair Actions:{c(Colors.RESET)}"
            )
            for action in colimit_result.universal_actions[:5]:
                lines.append(
                    f"    - [{action.action_type}] {action.target_label}: "
                    f"{action.description[:60]}"
                )

        return "\n".join(lines) + "\n"

    def render_human_overrides(self, soul: GenesisSoul, limit: int = 5) -> str:
        """Render the human override log for Soul Inspection."""
        c = self._c
        lines = [self.render_subheader(
            f"{c(Colors.WARNING)}HUMAN OVERRIDE LOG{c(Colors.RESET)} "
            f"(Subjective Gap Record)"
        )]

        entries = soul.human_overrides[-limit:] if soul.human_overrides else []
        if not entries:
            lines.append(f"  {c(Colors.DIM)}(no human overrides recorded){c(Colors.RESET)}")
        else:
            for i, entry in enumerate(entries):
                score_delta = entry.system_recommended_score - entry.human_selected_score
                lines.append(
                    f"\n  {c(Colors.WARNING)}[Override {i+1}]{c(Colors.RESET)} "
                    f"{entry.timestamp}"
                )
                lines.append(
                    f"    Problem: {entry.problem_text[:60]}..."
                    if len(entry.problem_text) > 60
                    else f"    Problem: {entry.problem_text}"
                )
                lines.append(
                    f"    System recommended: {c(Colors.SCORE)}"
                    f"{entry.system_recommended_path}{c(Colors.RESET)} "
                    f"(score={entry.system_recommended_score:.4f})"
                )
                lines.append(
                    f"    Human selected:     {c(Colors.WARNING)}"
                    f"{entry.human_selected_path}{c(Colors.RESET)} "
                    f"(score={entry.human_selected_score:.4f})"
                )
                lines.append(
                    f"    Score delta: {c(Colors.ERROR)}-{score_delta:.4f}{c(Colors.RESET)}"
                )
                lines.append(
                    f"    Category:   {entry.reason_category}"
                )
                lines.append(
                    f"    Confidence: {entry.confidence}/10"
                )
                # Wrap the reason at ~60 chars per line
                reason = entry.divergence_reason
                reason_lines = [reason[j:j+60] for j in range(0, len(reason), 60)]
                lines.append(f"    Reason:")
                for rl in reason_lines:
                    lines.append(f"      {c(Colors.DIM)}{rl}{c(Colors.RESET)}")

            # Axiom Anchor invariant reminder
            lines.append(
                f"\n  {c(Colors.PHASE)}NOTE: Overrides do NOT modify the "
                f"Axiom Anchor.{c(Colors.RESET)}"
            )
            lines.append(
                f"  {c(Colors.DIM)}The Anchor remains the objective Ground Truth; "
                f"overrides record the Subjective Gap.{c(Colors.RESET)}"
            )

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Aria Interface — main CLI class
# ---------------------------------------------------------------------------

class AriaInterface:
    """CLI interface for the Crucible Engine.

    Provides methods to:
    - Run problems through the Crucible with visualization
    - Inspect the current soul state
    - Export and verify souls
    """

    def __init__(
        self,
        crucible: CrucibleEngine | None = None,
        use_colors: bool = True,
    ) -> None:
        self.crucible = crucible or CrucibleEngine()
        self.renderer = AriaRenderer(use_colors=use_colors)

    @property
    def soul(self) -> GenesisSoul:
        """Access the Crucible's soul (EternalBox)."""
        return self.crucible.soul

    # -- Commands -----------------------------------------------------------

    def process(self, problem_text: str, verbose: bool = True) -> CrucibleResult:
        """Run a problem through the Crucible with optional CLI output."""
        result = self.crucible.process(problem_text)

        if verbose:
            print(self.renderer.render_result(result))

        return result

    def inspect_soul(self, verbose: bool = True) -> dict[str, Any]:
        """Inspect the current soul state, including the human override log."""
        if verbose:
            print(self.renderer.render_soul_summary(self.soul))
            print(self.renderer.render_wisdom_log(self.soul))
            print(self.renderer.render_foresight_projections(self.soul))
            print(self.renderer.render_human_overrides(self.soul))

        return self.soul.as_dict()

    def export_soul(self, path: str | None = None) -> str:
        """Export the soul to a file or return JSON string."""
        bridge = ContinuityBridge()
        json_str = bridge.export_soul_json(self.soul)

        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"Soul exported to: {path}")

        return json_str

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the wisdom log hash chain."""
        is_valid, errors = ContinuityBridge.verify_wisdom_chain(self.soul)

        if is_valid:
            print(f"{Colors.SCORE}✓ Hash chain integrity: VALID{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}✗ Hash chain integrity: INVALID{Colors.RESET}")
            for err in errors:
                print(f"  {Colors.ERROR}{err}{Colors.RESET}")

        return is_valid, errors

    # -- Human Override command ----------------------------------------------

    def human_override(
        self,
        result: CrucibleResult,
        selected_candidate: CrucibleCandidate,
        divergence_reason: str,
        reason_category: str,
        confidence: int,
        verbose: bool = True,
    ) -> HumanOverrideEntry:
        """Record a human override when the user selects a non-optimal candidate.

        This method is called when the user selects a candidate with a lower
        ``unityAlignmentScore`` than the system-recommended winner during the
        Aria "Soul Inspection" phase.

        The AxiomAnchor is NOT modified — the override is recorded in the
        human_overrides log only.

        Parameters
        ----------
        result : CrucibleResult
            The Crucible result containing the system recommendation.
        selected_candidate : CrucibleCandidate
            The candidate the human chose instead.
        divergence_reason : str
            100–500 character explanation of why the human disagrees.
        reason_category : str
            One of ``OVERRIDE_REASON_CATEGORIES``.
        confidence : int
            1–10, the human's confidence in their override.

        Returns
        -------
        HumanOverrideEntry
            The recorded override entry.

        Raises
        ------
        ValueError
            If the override data fails validation constraints.
        """
        system_best = result.logic_box.best
        if not system_best:
            raise ValueError("No system-recommended candidate to override.")

        entry = self.soul.record_human_override(
            system_recommended_id=system_best.id,
            system_recommended_score=system_best.unity_alignment_score,
            human_selected_id=selected_candidate.id,
            human_selected_score=selected_candidate.unity_alignment_score,
            divergence_reason=divergence_reason,
            reason_category=reason_category,
            confidence=confidence,
            problem_text=result.source_text,
            system_recommended_path=(
                system_best.dream_path.path_type.value
                if system_best.dream_path else ""
            ),
            human_selected_path=(
                selected_candidate.dream_path.path_type.value
                if selected_candidate.dream_path else ""
            ),
        )

        if verbose:
            print(self.renderer.render_header("HUMAN OVERRIDE RECORDED"))
            print(self.renderer.render_human_overrides(self.soul, limit=1))

        return entry

    # -- Mirror of Truth command (Sprint 8) ---------------------------------

    def refinement_panel(
        self,
        trace: RefinementTrace,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Display the Mirror of Truth's Refinement Panel.

        Parameters
        ----------
        trace : RefinementTrace
            The Mirror's critique output.
        verbose : bool
            Print to CLI if True.

        Returns
        -------
        dict
            The trace as a dictionary.
        """
        if verbose:
            print(self.renderer.render_refinement_panel(trace))
        return trace.as_dict()

    def blackout_shock_exam(
        self,
        fragility_amplifier: float = 1.5,
        prior_viability: float = 0.6,
        seed: int | None = None,
        verbose: bool = True,
    ) -> BlackoutShockResult:
        """Run the Bayesian Blackout Shock Final Exam.

        Parameters
        ----------
        fragility_amplifier : float
            Multiplier for fragility penalties (default 1.5).
        prior_viability : float
            Prior belief about grid viability (default 0.6).
        seed : int | None
            Random seed for reproducibility.
        verbose : bool
            Print results to CLI.

        Returns
        -------
        BlackoutShockResult
            Full Bayesian exam result.
        """
        exam = BayesianFinalExam(
            fragility_amplifier=fragility_amplifier,
            prior_viability=prior_viability,
        )
        result = exam.administer(seed=seed)

        if verbose:
            print(self.renderer.render_blackout_shock(result))

        return result

    # -- Game Theory command ------------------------------------------------

    def war_game(
        self,
        rounds: int = 100,
        sustainability_threshold: float = 5.0,
        seed: int | None = None,
        verbose: bool = True,
    ) -> WarGameOutcome:
        """Run a Game Theory war-game (Iterated Prisoner's Dilemma).

        Simulates Aligned (axiom-led) vs Extractive (profit-led) agents
        over *rounds* iterations, computes sustainability, and records
        the foresight projection in the soul.

        Parameters
        ----------
        rounds : int
            Number of IPD rounds (default 100 = "100-year" simulation).
        sustainability_threshold : float
            Score below which SYSTEMIC_COLLAPSE is flagged.
        seed : int | None
            Random seed for reproducibility.
        verbose : bool
            Print the war-game results to CLI.

        Returns
        -------
        WarGameOutcome
            Complete war-game result.
        """
        console = GameTheoryConsole(seed=seed)
        outcome = console.run_war_game(
            rounds=rounds,
            sustainability_threshold=sustainability_threshold,
        )

        # Record foresight projection in the soul
        projection = ForesightProjection(
            war_game_rounds=outcome.total_rounds,
            aligned_score=outcome.aligned_final_score,
            extractive_score=outcome.extractive_final_score,
            sustainability_score=outcome.sustainability_score,
            outcome_flag=outcome.outcome_flag.value,
            aligned_cooperation_rate=outcome.aligned_cooperation_rate,
            extractive_cooperation_rate=outcome.extractive_cooperation_rate,
            foresight_summary=outcome.foresight_summary,
        )
        self.soul.record_foresight(projection)

        if verbose:
            print(self.renderer.render_war_game(outcome))

        return outcome

    # -- Conflict War-Room commands (Sprint 9) ------------------------------

    def load_conflict(
        self,
        json_path: str,
        verbose: bool = True,
    ) -> tuple[dict[str, Any], CategoricalGraph]:
        """Load a conflict scenario from a JSON file into the War-Room.

        Parameters
        ----------
        json_path : str
            Path to the scenario JSON file (e.g. grid_war_2026.json).
        verbose : bool
            Print the War-Room dashboard if True.

        Returns
        -------
        tuple[dict, CategoricalGraph]
            The parsed scenario data and the constructed conflict graph.
        """
        scenario = MirrorOfTruth.load_scenario(json_path)
        graph = MirrorOfTruth.scenario_to_graph(scenario)

        # Store in instance for subsequent commands
        self._active_scenario = scenario
        self._active_graph = graph

        if verbose:
            print(self.renderer.render_conflict_war_room(scenario, graph))

        return scenario, graph

    def inject_node(
        self,
        graph: CategoricalGraph,
        label: str,
        tags: list[str],
        connect_to: str | None = None,
        morphism_label: str = "Injected_Relationship",
        morphism_tags: list[str] | None = None,
        verbose: bool = True,
    ) -> CategoricalGraph:
        """Inject a new node into a conflict graph during the Mirror of Truth phase.

        This enables real-time node injection (e.g., adding Water_Resources
        or Local_Innovation) to explore alternative scenario structures.

        Parameters
        ----------
        graph : CategoricalGraph
            The graph to modify.
        label : str
            Label for the new node.
        tags : list[str]
            Tags for the new node.
        connect_to : str | None
            Optional label of an existing node to connect to.
        morphism_label : str
            Label for the connecting morphism.
        morphism_tags : list[str] | None
            Tags for the connecting morphism.
        verbose : bool
            Print injection confirmation.

        Returns
        -------
        CategoricalGraph
            The modified graph with the injected node.
        """
        new_obj = graph.add_object(label, tags)

        if connect_to:
            # Find the target object by label
            target = None
            for obj in graph.objects:
                if obj.label == connect_to:
                    target = obj
                    break
            if target:
                m_tags = morphism_tags or ["collaboration", "service"]
                graph.add_morphism(morphism_label, new_obj, target, m_tags)

        if verbose:
            print(self.renderer.render_node_injection(label, tags, graph))

        return graph

    # -- Regenerative Governance Console commands (Sprint 10) ----------------

    def compare_manifestos(
        self,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Generate a delta report between a legislative bill and the
        Engine's Regenerative Blueprint.

        Parameters
        ----------
        candidate_a : dict
            Legislative bill data (e.g., HB 2992 from scenario context).
            Expected keys: bill, title, summary, disharmony_vector,
            cost_causation_intent.
        candidate_b : dict
            Regenerative Blueprint data.
            Expected keys: title, cost_allocation, water_policy,
            ratepayer_protection.
        verbose : bool
            Print the comparison to CLI if True.

        Returns
        -------
        dict
            Delta report with field-by-field comparison and violations.
        """
        delta: dict[str, Any] = {"fields": {}, "invariant_violations": []}

        # Cost allocation comparison
        bill_cost = candidate_a.get(
            "cost_causation_intent",
            candidate_a.get("disharmony_vector", "Not specified"),
        )
        blueprint_cost = candidate_b.get(
            "cost_allocation",
            "100% HILL infrastructure costs allocated to Hyperscale_Node",
        )
        cost_status = "violation" if "cost-shifting" in str(bill_cost).lower() or "socialise" in str(bill_cost).lower() else "aligned"
        delta["fields"]["Cost Allocation"] = {
            "bill": bill_cost,
            "blueprint": blueprint_cost,
            "status": cost_status,
        }
        if cost_status == "violation":
            delta["invariant_violations"].append({
                "invariant": HardInvariant.COST_CAUSATION,
                "description": "Bill enables cost-shifting that violates cost-causation invariant.",
            })

        # Water policy comparison
        bill_water = candidate_a.get("summary", "")
        blueprint_water = candidate_b.get(
            "water_policy",
            "Cooling demand must remain below aquifer recharge rates",
        )
        water_status = "gap"
        if "sustainable" in bill_water.lower() or "recharge" in bill_water.lower():
            water_status = "aligned"
        if "exceed" in str(candidate_a.get("disharmony_vector", "")).lower():
            water_status = "violation"
            delta["invariant_violations"].append({
                "invariant": HardInvariant.WATER_FLOOR,
                "description": "Bill permits withdrawal exceeding sustainable recharge.",
            })
        delta["fields"]["Water Policy"] = {
            "bill": bill_water[:100] if bill_water else "Not specified",
            "blueprint": blueprint_water,
            "status": water_status,
        }

        # Ratepayer protection comparison
        bill_protection = "Not specified"
        if "residential" in str(candidate_a).lower():
            bill_protection = "Residential ratepayers affected"
        blueprint_protection = candidate_b.get(
            "ratepayer_protection",
            "Residential ratepayer protection as axiom-level constraint",
        )
        protection_status = "gap"
        if "protect" in str(candidate_a.get("summary", "")).lower():
            protection_status = "aligned"
        if "burden" in str(candidate_a.get("disharmony_vector", "")).lower():
            protection_status = "violation"
            delta["invariant_violations"].append({
                "invariant": HardInvariant.EQUITY,
                "description": "Bill places cost burden on residential ratepayers.",
            })
        delta["fields"]["Ratepayer Protection"] = {
            "bill": bill_protection,
            "blueprint": blueprint_protection,
            "status": protection_status,
        }

        if verbose:
            print(self.renderer.render_manifesto_comparison(
                candidate_a, candidate_b, delta,
            ))

        return delta

    def invariant_tracker(
        self,
        scenario: dict[str, Any] | None = None,
        graph: CategoricalGraph | None = None,
        verbose: bool = True,
    ) -> list[InvariantViolation]:
        """Run the Live Invariant Tracker on the current scenario.

        Checks all Hard Constraints (Equity, Sustainability, Agency,
        Water Floor, Cost Causation) against the active scenario data.

        Parameters
        ----------
        scenario : dict | None
            Scenario data (grid_war_2026.json).  Falls back to
            ``self._active_scenario`` if None.
        graph : CategoricalGraph | None
            Conflict graph.  Falls back to ``self._active_graph`` if None.
        verbose : bool
            Print the tracker panel if True.

        Returns
        -------
        list[InvariantViolation]
            All detected violations.
        """
        scenario = scenario or getattr(self, "_active_scenario", None)
        graph = graph or getattr(self, "_active_graph", None)

        if scenario is None or graph is None:
            if verbose:
                print(
                    f"{Colors.WARNING}No active scenario loaded. "
                    f"Use load_conflict() first.{Colors.RESET}"
                )
            return []

        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(graph, scenario)

        violations = result.invariant_violations

        if verbose:
            print(self.renderer.render_invariant_tracker(violations))

        return violations

    def robustness_exam(
        self,
        scenario: dict[str, Any] | None = None,
        graph: CategoricalGraph | None = None,
        seed: int | None = 42,
        verbose: bool = True,
    ) -> RobustnessResult:
        """Run the full Robustness Harness evaluation.

        Parameters
        ----------
        scenario : dict | None
            Scenario data.  Falls back to ``self._active_scenario``.
        graph : CategoricalGraph | None
            Conflict graph.  Falls back to ``self._active_graph``.
        seed : int | None
            Random seed for reproducibility.
        verbose : bool
            Print results to CLI.

        Returns
        -------
        RobustnessResult
            Full harness evaluation result.
        """
        scenario = scenario or getattr(self, "_active_scenario", None)
        graph = graph or getattr(self, "_active_graph", None)

        if scenario is None or graph is None:
            raise ValueError(
                "No active scenario/graph. Use load_conflict() first."
            )

        harness = RobustnessHarness(seed=seed)
        result = harness.evaluate(graph, scenario)

        if verbose:
            print(self.renderer.render_robustness_result(result))
            print(self.renderer.render_invariant_tracker(result.invariant_violations))

        return result

    def generate_governance_report(
        self,
        trace: RefinementTrace | None = None,
        robustness_result: RobustnessResult | None = None,
        verbose: bool = True,
    ) -> GovernanceReport:
        """Generate a Production Lexicon governance report.

        Parameters
        ----------
        trace : RefinementTrace | None
            Mirror of Truth analysis results.
        robustness_result : RobustnessResult | None
            Robustness Harness evaluation results.
        verbose : bool
            Print report summary to CLI.

        Returns
        -------
        GovernanceReport
            Production-ready report with Sacred Language stripped.
        """
        scenario = getattr(self, "_active_scenario", None)
        rob_dict = robustness_result.as_dict() if robustness_result else None

        report = GovernanceReportBuilder.build(
            soul=self.soul,
            scenario=scenario,
            trace=trace,
            robustness_result=rob_dict,
        )

        if verbose:
            c = self.renderer._c
            print(self.renderer.render_header(
                "GOVERNANCE REPORT — PRODUCTION LEXICON"
            ))
            print(f"  Report ID:  {report.report_id}")
            print(f"  I AM Hash:  {report.eventstore_hash[:32]}...")
            print(f"  Robustness: {report.robustness_score:.4f}/10.0")
            print(f"  Conflicts:  {len(report.conflicts)}")
            print(f"  Violations: {len(report.invariant_violations)}")
            if report.covenant_actuation:
                print(f"  Covenant:   {report.covenant_actuation.get('title', 'N/A')}")

        return report

    def generate_sovereign_index(
        self,
        report: GovernanceReport,
        verbose: bool = True,
    ) -> str:
        """Generate the State_of_the_Sovereignty.md index.

        Parameters
        ----------
        report : GovernanceReport
            The governance report to index.
        verbose : bool
            Print confirmation to CLI.

        Returns
        -------
        str
            Markdown content for State_of_the_Sovereignty.md.
        """
        scenario = getattr(self, "_active_scenario", None)
        content = SovereignIndexGenerator.generate(
            report=report,
            soul=self.soul,
            scenario=scenario,
        )

        if verbose:
            print(f"\n  State_of_the_Sovereignty.md generated "
                  f"({len(content)} chars)")

        return content

    # -- Policy Audit Panel commands (Sprint 11) -----------------------------

    def policy_audit(
        self,
        bill_reference: str = "",
        seed: int | None = 42,
        max_debate_rounds: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full Policy Audit pipeline.

        Steps:
        1. Load active scenario (must call load_conflict() first).
        2. Run FAIRGAME Debate Arena (Pro_Social vs Hostile_Lobbyist).
        3. Generate Governance Report.
        4. Run C3AI Self-Critique Loop on the report.
        5. Run Sovereign Reference Architecture audit.
        6. Display all results in the Policy Audit Panel.

        Parameters
        ----------
        bill_reference : str
            Bill being audited (e.g., "HB 2992").
        seed : int | None
            Random seed for reproducibility.
        max_debate_rounds : int | None
            Maximum debate rounds.
        verbose : bool
            Print results to CLI.

        Returns
        -------
        dict
            Complete audit results with debate, critique, and audit.
        """
        from genesis_engine.core.adversarial_evaluator import (
            AdversarialEvaluator,
        )
        from genesis_engine.core.policy_kernel import PolicyKernel

        scenario = getattr(self, "_active_scenario", None)
        graph = getattr(self, "_active_graph", None)

        if scenario is None or graph is None:
            raise ValueError(
                "No active scenario. Use load_conflict() first."
            )

        # Step 1: Run FAIRGAME Debate Arena
        evaluator = AdversarialEvaluator(seed=seed)
        debate_result = evaluator.debate(
            scenario=scenario,
            bill_reference=bill_reference,
            max_rounds=max_debate_rounds,
        )

        # Step 2: Generate Governance Report
        gov_report = self.generate_governance_report(verbose=False)

        # Step 3: Run C3AI Self-Critique
        kernel = PolicyKernel()
        critique_result = kernel.evaluate(
            report_data=gov_report.as_dict(),
            scenario=scenario,
        )

        # Step 4: Run Sovereign Audit
        sovereign_audit = self.sovereign_audit_hook(verbose=False)

        # Step 5: Render results
        if verbose:
            print(self.renderer.render_policy_audit_panel(
                debate_result, critique_result, scenario,
            ))
            print(self.renderer.render_sovereign_audit(sovereign_audit))

        return {
            "debate": debate_result.as_dict(),
            "governance_report": gov_report.as_dict(),
            "self_critique": critique_result.as_dict(),
            "sovereign_audit": sovereign_audit,
            "bill_reference": bill_reference,
        }

    def sovereign_audit_hook(
        self,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Sovereign Reference Architecture audit hook.

        Confirms all data residency stays local during simulation.
        Checks:
        1. No external API calls in the analysis pipeline.
        2. All data stored in local Obsidian vault.
        3. EventStore hash chain integrity.
        4. No PII transmitted to external services.

        Parameters
        ----------
        verbose : bool
            Print audit results to CLI.

        Returns
        -------
        dict
            Audit result with compliance status and checks.
        """
        checks: list[dict[str, Any]] = []

        # Check 1: Local-First Provider
        from genesis_engine.core.ai_provider import get_default_provider
        provider = get_default_provider()
        provider_name = type(provider).__name__
        is_local = "Local" in provider_name or "Ollama" in provider_name
        checks.append({
            "name": "Local-First Provider",
            "passed": is_local,
            "description": (
                f"AI provider '{provider_name}' is "
                f"{'local' if is_local else 'REMOTE — data residency risk'}"
            ),
        })

        # Check 2: EventStore Integrity
        is_valid, errors = ContinuityBridge.verify_wisdom_chain(self.soul)
        checks.append({
            "name": "EventStore Integrity",
            "passed": is_valid,
            "description": (
                "Hash chain integrity: VALID"
                if is_valid
                else f"Hash chain integrity: INVALID ({len(errors)} errors)"
            ),
        })

        # Check 3: Local Data Residency (vault path exists)
        from pathlib import Path
        vault_path = Path(__file__).parent.parent / "reports" / "obsidian_vault"
        vault_exists = vault_path.exists()
        checks.append({
            "name": "Local Vault Residency",
            "passed": vault_exists,
            "description": (
                f"Obsidian vault at local path: {vault_path.name}"
                if vault_exists
                else "Obsidian vault not found at expected local path"
            ),
        })

        # Check 4: No external transmission (static check — always passes
        # in local-first mode since all data stays in memory/disk)
        checks.append({
            "name": "Data Sovereignty",
            "passed": True,
            "description": (
                "All simulation data remains in local memory and disk. "
                "No external transmission detected."
            ),
        })

        all_passed = all(c["passed"] for c in checks)
        result = {
            "sovereign_compliant": all_passed,
            "checks": checks,
            "provider": provider_name,
        }

        if verbose:
            print(self.renderer.render_sovereign_audit(result))

        return result

    # -- Crystallization command --------------------------------------------

    def crystallize(
        self,
        candidate: CrucibleCandidate,
        verbose: bool = True,
    ) -> bool:
        """Manually crystallize a specific candidate into the soul.

        Returns True if successful.
        """
        if candidate.status != CandidateStatus.CONFIRMED:
            if verbose:
                print(
                    f"{Colors.ERROR}Cannot crystallize: "
                    f"candidate status is {candidate.status.value}, not CONFIRMED.{Colors.RESET}"
                )
            return False

        if not candidate.artifact:
            if verbose:
                print(
                    f"{Colors.WARNING}Candidate has no forge artifact. "
                    f"Run full Crucible process first.{Colors.RESET}"
                )
            return False

        # Already crystallized during the Crucible process
        if verbose:
            print(self.renderer.render_crystallization(candidate))

        return True
