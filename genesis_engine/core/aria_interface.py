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
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul
from genesis_engine.core.crucible import (
    CandidateStatus,
    CrucibleCandidate,
    CrucibleEngine,
    CrucibleResult,
    LogicBox,
    PhaseRecord,
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
                lines.append(
                    f"      Unity: {entry.unity_impact}/10  "
                    f"Compassion: {entry.compassion_deficit}/10"
                )
                lines.append(
                    f"      Path: {path_color}{entry.resolution_path}{c(Colors.RESET)}"
                )
                if entry.entry_hash:
                    lines.append(
                        f"      Hash: {c(Colors.DIM)}{entry.entry_hash[:32]}...{c(Colors.RESET)}"
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
        """Inspect the current soul state."""
        if verbose:
            print(self.renderer.render_soul_summary(self.soul))
            print(self.renderer.render_wisdom_log(self.soul))

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
