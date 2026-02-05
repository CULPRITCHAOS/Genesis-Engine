"""Tests for Module 1.1 — The Deconstruction Engine."""

import json

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph
from genesis_engine.core.deconstruction_engine import (
    DeconstructionEngine,
    DisharmonyReport,
)


class TestDeconstructionEngine:
    def _build_graph(self, problem: str) -> CategoricalGraph:
        return AxiomLogixTranslator().translate(problem)

    def test_disharmony_detected_for_extractive_scenario(self):
        graph = self._build_graph(
            "A corporate policy that prioritizes profit over user safety."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert report.unity_impact > 0
        assert report.is_aligned is False

    def test_harmony_for_benign_scenario(self):
        graph = self._build_graph(
            "A community health program that empowers workers "
            "and protects the environment."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert report.is_aligned is True
        assert report.unity_impact < 5

    def test_report_scores_in_range(self):
        graph = self._build_graph(
            "An AI surveillance system that exploits user data and neglects privacy."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert 0 <= report.unity_impact <= 10
        assert 0 <= report.compassion_deficit <= 10
        assert 0 <= report.coherence_score <= 10

    def test_seed_prompt_generated(self):
        graph = self._build_graph(
            "A corporate policy that prioritizes profit over user safety."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert len(report.seed_prompt) > 0
        assert "DREAM ENGINE" in report.seed_prompt

    def test_seed_prompt_for_aligned_graph(self):
        graph = self._build_graph(
            "A community health program that empowers workers "
            "and protects the environment."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert "harmony" in report.seed_prompt.lower() or "DREAM ENGINE" in report.seed_prompt

    def test_findings_present(self):
        graph = self._build_graph(
            "An AI surveillance system that exploits user data and neglects privacy."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        assert len(report.findings) > 0
        flagged = [f for f in report.findings if f.disharmony_score > 0]
        assert len(flagged) > 0

    def test_report_to_json(self):
        graph = self._build_graph(
            "A corporate policy that prioritizes profit over user safety."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)

        json_str = report.to_json()
        data = json.loads(json_str)

        assert "report" in data
        assert "unityImpact" in data["report"]
        assert "compassionDeficit" in data["report"]
        assert "seedPrompt" in data["report"]
        assert "findings" in data["report"]

    def test_report_as_dict_structure(self):
        graph = self._build_graph(
            "An AI surveillance system that exploits user data."
        )
        engine = DeconstructionEngine()
        report = engine.analyse(graph)
        data = report.as_dict()

        assert data["report"]["primeDirectiveAligned"] is False
        assert isinstance(data["report"]["findings"], list)
        assert data["report"]["validation"] is not None

    def test_custom_anchor_passed_through(self):
        anchor = AxiomAnchor(alignment_threshold=0.0)
        engine = DeconstructionEngine(anchor=anchor)
        graph = self._build_graph("Users exploited by corporations.")
        report = engine.analyse(graph)

        # With threshold 0.0 everything should appear aligned
        assert report.is_aligned is True


class TestEndToEndPipeline:
    """Integration tests running the full Translator → Anchor → Engine pipeline."""

    def test_full_pipeline_extractive(self):
        translator = AxiomLogixTranslator()
        anchor = AxiomAnchor()
        engine = DeconstructionEngine(anchor=anchor)

        graph = translator.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        report = engine.analyse(graph)

        assert not report.is_aligned
        assert report.unity_impact > 0
        assert "DREAM ENGINE" in report.seed_prompt

    def test_full_pipeline_benign(self):
        translator = AxiomLogixTranslator()
        anchor = AxiomAnchor()
        engine = DeconstructionEngine(anchor=anchor)

        graph = translator.translate(
            "A community health program that empowers workers "
            "and protects the environment."
        )
        report = engine.analyse(graph)

        assert report.is_aligned
        assert report.compassion_deficit < 5
