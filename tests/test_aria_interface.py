"""Tests for Module 3.2 — The Aria Interface."""

from genesis_engine.core.ai_provider import Perspective
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer, Colors
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul
from genesis_engine.core.crucible import (
    CandidateStatus,
    CrucibleCandidate,
    CrucibleEngine,
    LogicBox,
    PhaseRecord,
)


# ---------------------------------------------------------------------------
# AriaRenderer tests
# ---------------------------------------------------------------------------

class TestAriaRenderer:
    def test_render_header(self):
        renderer = AriaRenderer(use_colors=False)
        output = renderer.render_header("Test Header")
        assert "Test Header" in output
        assert "═" in output

    def test_render_subheader(self):
        renderer = AriaRenderer(use_colors=False)
        output = renderer.render_subheader("Sub Header")
        assert "Sub Header" in output
        assert "─" in output

    def test_render_phase(self):
        renderer = AriaRenderer(use_colors=False)
        phase = PhaseRecord(phase="1-ingest", summary="Translated 5 objects")
        output = renderer.render_phase(phase)
        assert "1 → INGEST" in output
        assert "Translated 5 objects" in output

    def test_render_candidate_pending(self):
        renderer = AriaRenderer(use_colors=False)
        cand = CrucibleCandidate(
            perspective=Perspective.CAUSALITY,
            status=CandidateStatus.PENDING,
            unity_alignment_score=0.5,
        )
        output = renderer.render_candidate(cand)
        assert "PENDING" in output
        assert "causality" in output

    def test_render_candidate_confirmed(self):
        renderer = AriaRenderer(use_colors=False)
        cand = CrucibleCandidate(
            perspective=Perspective.CONTRADICTION,
            status=CandidateStatus.CONFIRMED,
            unity_alignment_score=1.0,
        )
        output = renderer.render_candidate(cand)
        assert "CONFIRMED" in output
        assert "●" in output

    def test_render_logic_box(self):
        renderer = AriaRenderer(use_colors=False)
        box = LogicBox()
        box.add(CrucibleCandidate(perspective=Perspective.CAUSALITY))
        box.add(CrucibleCandidate(perspective=Perspective.ANALOGY))
        output = renderer.render_logic_box(box)
        assert "LOGIC BOX" in output
        assert "2 candidates" in output

    def test_render_crystallization_success(self):
        renderer = AriaRenderer(use_colors=False)
        cand = CrucibleCandidate(
            perspective=Perspective.CONTRADICTION,
            status=CandidateStatus.CONFIRMED,
            unity_alignment_score=1.0,
        )
        output = renderer.render_crystallization(cand)
        assert "CRYSTALLIZATION" in output
        assert "SUCCESS" in output

    def test_render_crystallization_none(self):
        renderer = AriaRenderer(use_colors=False)
        output = renderer.render_crystallization(None)
        assert "No candidate crystallized" in output

    def test_render_soul_summary(self):
        renderer = AriaRenderer(use_colors=False)
        soul = GenesisSoul()
        output = renderer.render_soul_summary(soul)
        assert "ETERNAL BOX" in output
        assert soul.soul_id in output
        assert "Does this serve Love?" in output

    def test_colors_disabled(self):
        renderer = AriaRenderer(use_colors=False)
        output = renderer.render_header("Test")
        # Should not contain ANSI escape codes
        assert "\033[" not in output

    def test_colors_enabled(self):
        renderer = AriaRenderer(use_colors=True)
        output = renderer.render_header("Test")
        # Should contain ANSI escape codes
        assert "\033[" in output


# ---------------------------------------------------------------------------
# Colors helper tests
# ---------------------------------------------------------------------------

class TestColors:
    def test_status_color_mapping(self):
        assert Colors.status_color(CandidateStatus.PENDING) == Colors.PENDING
        assert Colors.status_color(CandidateStatus.CONFIRMED) == Colors.CONFIRMED
        assert Colors.status_color(CandidateStatus.REJECTED) == Colors.REJECTED

    def test_perspective_color_mapping(self):
        assert Colors.perspective_color(Perspective.CAUSALITY) == Colors.CAUSALITY
        assert Colors.perspective_color(Perspective.CONTRADICTION) == Colors.CONTRADICTION
        assert Colors.perspective_color(Perspective.ANALOGY) == Colors.ANALOGY


# ---------------------------------------------------------------------------
# AriaInterface tests
# ---------------------------------------------------------------------------

class TestAriaInterface:
    def test_creates_with_defaults(self):
        aria = AriaInterface()
        assert aria.crucible is not None
        assert aria.renderer is not None

    def test_soul_property(self):
        aria = AriaInterface()
        assert aria.soul is not None
        assert aria.soul.soul_id.startswith("soul-")

    def test_process_returns_result(self):
        aria = AriaInterface(use_colors=False)
        result = aria.process("A corporate policy that prioritizes profit.", verbose=False)
        assert result is not None

    def test_inspect_soul_returns_dict(self):
        aria = AriaInterface(use_colors=False)
        data = aria.inspect_soul(verbose=False)
        assert "soulId" in data
        assert "axiomAnchorState" in data

    def test_export_soul_returns_json(self):
        aria = AriaInterface(use_colors=False)
        json_str = aria.export_soul()
        assert "genesis_soul" in json_str

    def test_verify_chain_empty_log(self):
        aria = AriaInterface(use_colors=False)
        is_valid, errors = aria.verify_chain()
        assert is_valid is True
        assert len(errors) == 0

    def test_crystallize_confirmed(self):
        aria = AriaInterface(use_colors=False)
        # Process a problem to get a confirmed candidate
        result = aria.process("Corporation exploits workers.", verbose=False)

        if result.crystallized_candidate:
            success = aria.crystallize(result.crystallized_candidate, verbose=False)
            assert success is True

    def test_crystallize_rejected_fails(self):
        aria = AriaInterface(use_colors=False)
        cand = CrucibleCandidate(status=CandidateStatus.REJECTED)
        success = aria.crystallize(cand, verbose=False)
        assert success is False


# ---------------------------------------------------------------------------
# End-to-end Aria tests
# ---------------------------------------------------------------------------

class TestAriaEndToEnd:
    def test_full_workflow(self, capsys):
        aria = AriaInterface(use_colors=False)

        # Process a problem
        result = aria.process("A corporate policy that prioritizes profit over user safety.")

        # Inspect the soul
        aria.inspect_soul()

        # Verify the chain
        is_valid, _ = aria.verify_chain()
        assert is_valid

        # Export the soul
        json_str = aria.export_soul()
        assert "genesis_soul" in json_str

    def test_multiple_problems_workflow(self):
        aria = AriaInterface(use_colors=False)

        result1 = aria.process("Corporation exploits workers.", verbose=False)
        result2 = aria.process("AI neglects privacy.", verbose=False)

        assert len(aria.soul.wisdom_log) >= 2
        assert len(aria.soul.graph_history) >= 2
