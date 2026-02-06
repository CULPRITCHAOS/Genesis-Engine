"""Tests for Module 3.1 — The Crucible Engine and AIProvider."""

import json

from genesis_engine.core.ai_provider import (
    AIProvider,
    Candidate,
    LocalProvider,
    Perspective,
)
from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.continuity_bridge import ContinuityBridge
from genesis_engine.core.crucible import (
    CandidateStatus,
    CrucibleCandidate,
    CrucibleEngine,
    CrucibleResult,
    LogicBox,
    PhaseRecord,
)
from genesis_engine.core.dream_engine import PathType


# ---------------------------------------------------------------------------
# AIProvider tests
# ---------------------------------------------------------------------------

class TestLocalProvider:
    def test_provider_name(self):
        provider = LocalProvider()
        assert "LocalProvider" in provider.provider_name

    def test_generates_three_perspectives(self):
        provider = LocalProvider()
        context = {
            "source_text": "A corporate policy that prioritizes profit.",
            "morphisms": [
                {"label": "Extraction", "tags": ["extraction"], "source": "corp", "target": "user"},
            ],
            "objects": [
                {"id": "corp", "label": "Corporation", "tags": ["actor"]},
                {"id": "user", "label": "User", "tags": ["vulnerable"]},
            ],
        }
        candidates = provider.generate_candidates(context)
        assert len(candidates) == 3
        perspectives = {c.perspective for c in candidates}
        assert perspectives == {Perspective.CAUSALITY, Perspective.CONTRADICTION, Perspective.ANALOGY}

    def test_causality_detects_harm(self):
        provider = LocalProvider()
        context = {
            "source_text": "Exploitation scenario",
            "morphisms": [{"label": "Exploitation", "tags": ["exploitation"], "source": "a", "target": "b"}],
            "objects": [{"id": "a", "label": "Actor"}, {"id": "b", "label": "Victim"}],
        }
        candidates = provider.generate_candidates(context, [Perspective.CAUSALITY])
        assert len(candidates) == 1
        assert "harm" in candidates[0].content.lower() or "causal" in candidates[0].reasoning.lower()

    def test_contradiction_detects_tension(self):
        provider = LocalProvider()
        context = {
            "source_text": "Safety ignored",
            "morphisms": [{"label": "Neglect", "tags": ["neglect"], "source": "a", "target": "b"}],
            "objects": [
                {"id": "a", "label": "System"},
                {"id": "b", "label": "Safety", "tags": ["protective"]},
            ],
        }
        candidates = provider.generate_candidates(context, [Perspective.CONTRADICTION])
        assert len(candidates) == 1
        assert candidates[0].perspective == Perspective.CONTRADICTION

    def test_analogy_finds_patterns(self):
        provider = LocalProvider()
        context = {
            "source_text": "Resource extraction",
            "morphisms": [{"label": "Extraction", "tags": ["extraction"], "source": "a", "target": "b"}],
            "objects": [],
        }
        candidates = provider.generate_candidates(context, [Perspective.ANALOGY])
        assert len(candidates) == 1
        assert candidates[0].perspective == Perspective.ANALOGY

    def test_benign_scenario_high_confidence(self):
        provider = LocalProvider()
        context = {
            "source_text": "Community program",
            "morphisms": [{"label": "Protection", "tags": ["protection"], "source": "a", "target": "b"}],
            "objects": [],
        }
        candidates = provider.generate_candidates(context)
        # At least one candidate should have high confidence for benign scenario
        assert any(c.confidence >= 0.8 for c in candidates)


# ---------------------------------------------------------------------------
# LogicBox tests
# ---------------------------------------------------------------------------

class TestLogicBox:
    def test_add_candidate(self):
        box = LogicBox()
        cand = CrucibleCandidate()
        box.add(cand)
        assert len(box.candidates) == 1

    def test_clear(self):
        box = LogicBox()
        box.add(CrucibleCandidate())
        box.add(CrucibleCandidate())
        box.clear()
        assert len(box.candidates) == 0

    def test_confirmed_filter(self):
        box = LogicBox()
        box.add(CrucibleCandidate(status=CandidateStatus.PENDING))
        box.add(CrucibleCandidate(status=CandidateStatus.CONFIRMED))
        box.add(CrucibleCandidate(status=CandidateStatus.REJECTED))
        assert len(box.confirmed) == 1

    def test_best_returns_highest_score(self):
        box = LogicBox()
        box.add(CrucibleCandidate(
            status=CandidateStatus.CONFIRMED,
            unity_alignment_score=0.5,
        ))
        box.add(CrucibleCandidate(
            status=CandidateStatus.CONFIRMED,
            unity_alignment_score=0.9,
        ))
        best = box.best
        assert best is not None
        assert best.unity_alignment_score == 0.9

    def test_best_returns_none_if_no_confirmed(self):
        box = LogicBox()
        box.add(CrucibleCandidate(status=CandidateStatus.REJECTED))
        assert box.best is None


# ---------------------------------------------------------------------------
# CrucibleCandidate tests
# ---------------------------------------------------------------------------

class TestCrucibleCandidate:
    def test_default_status_pending(self):
        cand = CrucibleCandidate()
        assert cand.status == CandidateStatus.PENDING

    def test_as_dict_structure(self):
        cand = CrucibleCandidate(
            perspective=Perspective.CAUSALITY,
            reasoning="Test reasoning",
            content="Test content",
            confidence=0.75,
        )
        d = cand.as_dict()
        assert d["perspective"] == "causality"
        assert d["status"] == "PENDING"
        assert d["reasoning"] == "Test reasoning"
        assert d["confidence"] == 0.75


# ---------------------------------------------------------------------------
# CrucibleEngine tests
# ---------------------------------------------------------------------------

class TestCrucibleEngine:
    def test_creates_with_defaults(self):
        engine = CrucibleEngine()
        assert engine.anchor is not None
        assert engine.soul is not None
        assert engine.provider is not None

    def test_process_returns_result(self):
        engine = CrucibleEngine()
        result = engine.process("A corporate policy that prioritizes profit over user safety.")
        assert isinstance(result, CrucibleResult)

    def test_result_has_six_phases(self):
        engine = CrucibleEngine()
        result = engine.process("A corporate policy that prioritizes profit over user safety.")
        # Should have at least ingest and retrieval phases
        phase_names = [p.phase for p in result.phases]
        assert "1-ingest" in phase_names
        assert "2-retrieval" in phase_names

    def test_disharmony_detected(self):
        engine = CrucibleEngine()
        result = engine.process("A corporate policy that prioritizes profit over user safety.")
        assert result.disharmony_report is not None
        assert not result.disharmony_report.is_aligned

    def test_aligned_scenario_short_circuits(self):
        engine = CrucibleEngine()
        result = engine.process(
            "A community health program that empowers workers and protects the environment."
        )
        assert result.is_aligned

    def test_logic_box_has_candidates(self):
        engine = CrucibleEngine()
        result = engine.process("An AI surveillance system that exploits user data.")
        assert len(result.logic_box.candidates) > 0

    def test_crystallization_occurs(self):
        engine = CrucibleEngine()
        result = engine.process("A corporate policy that prioritizes profit over user safety.")
        assert result.crystallized_candidate is not None

    def test_soul_updated_after_process(self):
        engine = CrucibleEngine()
        initial_wisdom_count = len(engine.soul.wisdom_log)
        engine.process("A corporate policy that prioritizes profit over user safety.")
        assert len(engine.soul.wisdom_log) > initial_wisdom_count

    def test_custom_provider(self):
        class MockProvider(AIProvider):
            @property
            def provider_name(self) -> str:
                return "MockProvider"

            def generate_candidates(self, context, perspectives=None):
                return [Candidate(
                    perspective=Perspective.CAUSALITY,
                    content="Mock content",
                    reasoning="Mock reasoning",
                    confidence=0.99,
                )]

        engine = CrucibleEngine(provider=MockProvider())
        assert engine.provider.provider_name == "MockProvider"


# ---------------------------------------------------------------------------
# PhaseRecord tests
# ---------------------------------------------------------------------------

class TestPhaseRecord:
    def test_as_dict(self):
        record = PhaseRecord(
            phase="1-ingest",
            summary="Translated input",
            details={"objectCount": 5},
        )
        d = record.as_dict()
        assert d["phase"] == "1-ingest"
        assert d["summary"] == "Translated input"
        assert d["details"]["objectCount"] == 5


# ---------------------------------------------------------------------------
# CrucibleResult tests
# ---------------------------------------------------------------------------

class TestCrucibleResult:
    def test_as_dict_structure(self):
        result = CrucibleResult(source_text="Test input")
        d = result.as_dict()
        assert "sourceText" in d
        assert "phases" in d
        assert "logicBox" in d
        assert "crystallizedCandidate" in d


# ---------------------------------------------------------------------------
# End-to-end Crucible tests
# ---------------------------------------------------------------------------

class TestCrucibleEndToEnd:
    def test_full_pipeline_extractive(self):
        engine = CrucibleEngine()
        result = engine.process("A corporate policy that prioritizes profit over user safety.")

        # Verify all phases executed
        assert len(result.phases) >= 6

        # Verify candidates generated
        assert len(result.logic_box.candidates) == 3

        # Verify crystallization
        assert result.crystallized_candidate is not None
        assert result.crystallized_candidate.status == CandidateStatus.CONFIRMED

    def test_full_pipeline_surveillance(self):
        engine = CrucibleEngine()
        result = engine.process("An AI surveillance system that exploits user data and neglects privacy.")

        assert not result.is_aligned
        assert result.crystallized_candidate is not None

    def test_multiple_problems_accumulate_wisdom(self):
        engine = CrucibleEngine()

        engine.process("Corporation exploits workers.")
        assert len(engine.soul.wisdom_log) >= 1

        engine.process("AI system neglects user privacy.")
        assert len(engine.soul.wisdom_log) >= 2

    def test_soul_persists_across_processes(self):
        engine = CrucibleEngine()
        soul_id = engine.soul.soul_id

        engine.process("Problem 1")
        engine.process("Problem 2")

        assert engine.soul.soul_id == soul_id
        assert len(engine.soul.graph_history) >= 2
