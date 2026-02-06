"""
Module 3.2 — The Aria Interface

A CLI visualization layer for the Crucible Engine that displays the
"Thinking" process in real-time:

* **Crucible Panel** — Shows each candidate with its perspective,
  unityAlignmentScore, and lifecycle status (PENDING → VERIFYING → CONFIRMED).
* **Crystallization Command** — Finalizes a confirmed candidate into a
  Technical Covenant within the .genesis_soul file.
* **Soul Inspector** — Displays the current state of the EternalBox.

This module provides both programmatic access and formatted CLI output
for human operators to observe the multi-perspective reasoning process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from genesis_engine.core.ai_provider import Perspective
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    ForesightProjection,
    GenesisSoul,
    HumanOverrideEntry,
    OVERRIDE_REASON_CATEGORIES,
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
    GameTheoryConsole,
    OutcomeFlag,
    WarGameOutcome,
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
