"""Tests for Module 1.2 — The Dream Engine."""

import json

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import (
    DreamEngine,
    DreamPath,
    PathType,
    PossibilityReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_extractive_pipeline():
    """Run the full pipeline for the extractive corporate scenario."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)

    graph = translator.translate(
        "A corporate policy that prioritizes profit over user safety."
    )
    report = decon.analyse(graph)
    return dream, report, graph


def _build_surveillance_pipeline():
    """Run the full pipeline for the AI surveillance scenario."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)

    graph = translator.translate(
        "An AI surveillance system that exploits user data and neglects privacy."
    )
    report = decon.analyse(graph)
    return dream, report, graph


# ---------------------------------------------------------------------------
# DreamEngine core tests
# ---------------------------------------------------------------------------

class TestDreamEngine:
    def test_dream_returns_possibility_report(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        assert isinstance(result, PossibilityReport)

    def test_dream_produces_three_paths(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        assert len(result.paths) == 3

    def test_path_types_are_distinct(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        types = {p.path_type for p in result.paths}
        assert types == {PathType.REFORM, PathType.REINVENTION, PathType.DISSOLUTION}

    def test_recommended_path_is_set(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        assert result.recommended_path in ("reform", "reinvention", "dissolution")

    def test_source_text_preserved(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        assert result.source_text == report.source_text


# ---------------------------------------------------------------------------
# Path of Reform tests
# ---------------------------------------------------------------------------

class TestPathOfReform:
    def test_reform_heals_disharmonic_morphisms(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)

        morph_labels = {m.label for m in reform.healed_graph.morphisms}
        # Original "Extraction" should be replaced.
        assert "Extraction" not in morph_labels
        assert "Fair_Reciprocity" in morph_labels

    def test_reform_preserves_original_entities(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)

        original_labels = {o.label for o in graph.objects}
        healed_labels = {o.label for o in reform.healed_graph.objects}
        assert original_labels == healed_labels

    def test_reform_passes_axiom_anchor(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)

        assert reform.validation is not None
        assert reform.validation.is_aligned is True
        assert reform.unity_alignment_score > 0.5

    def test_reform_no_disharmony_tags_remain(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)

        disharmony_tags = {"extraction", "exploitation", "coercion", "neglect", "division"}
        all_tags: set[str] = set()
        for m in reform.healed_graph.morphisms:
            all_tags.update(t.lower() for t in m.tags)
        assert not (all_tags & disharmony_tags)

    def test_reform_feasibility_in_range(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)
        assert 0.0 <= reform.feasibility_score <= 1.0


# ---------------------------------------------------------------------------
# Path of Reinvention tests
# ---------------------------------------------------------------------------

class TestPathOfReinvention:
    def test_reinvention_creates_new_graph(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reinv = next(p for p in result.paths if p.path_type == PathType.REINVENTION)

        # Should have different object labels from the original.
        original_labels = {o.label for o in graph.objects}
        healed_labels = {o.label for o in reinv.healed_graph.objects}
        assert healed_labels != original_labels

    def test_reinvention_includes_stewardship_morphisms(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reinv = next(p for p in result.paths if p.path_type == PathType.REINVENTION)

        morph_labels = {m.label for m in reinv.healed_graph.morphisms}
        assert "Stewardship" in morph_labels
        assert "Empowerment" in morph_labels

    def test_reinvention_passes_axiom_anchor(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reinv = next(p for p in result.paths if p.path_type == PathType.REINVENTION)

        assert reinv.validation is not None
        assert reinv.validation.is_aligned is True

    def test_reinvention_has_governance_covenant(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reinv = next(p for p in result.paths if p.path_type == PathType.REINVENTION)

        labels = {o.label for o in reinv.healed_graph.objects}
        assert "Governance_Covenant" in labels

    def test_reinvention_feasibility_in_range(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        reinv = next(p for p in result.paths if p.path_type == PathType.REINVENTION)
        assert 0.0 <= reinv.feasibility_score <= 1.0


# ---------------------------------------------------------------------------
# Path of Dissolution tests
# ---------------------------------------------------------------------------

class TestPathOfDissolution:
    def test_dissolution_creates_cooperative_structure(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        diss = next(p for p in result.paths if p.path_type == PathType.DISSOLUTION)

        labels = {o.label for o in diss.healed_graph.objects}
        assert "Community_Collective" in labels
        assert "Shared_Resource" in labels
        assert "Cooperative_Protocol" in labels

    def test_dissolution_includes_democratic_morphisms(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        diss = next(p for p in result.paths if p.path_type == PathType.DISSOLUTION)

        morph_labels = {m.label for m in diss.healed_graph.morphisms}
        assert "Democratic_Governance" in morph_labels
        assert "Voice" in morph_labels

    def test_dissolution_passes_axiom_anchor(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        diss = next(p for p in result.paths if p.path_type == PathType.DISSOLUTION)

        assert diss.validation is not None
        assert diss.validation.is_aligned is True

    def test_dissolution_preserves_vulnerable_entities(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        diss = next(p for p in result.paths if p.path_type == PathType.DISSOLUTION)

        original_vulnerable = {o.label for o in graph.objects if "vulnerable" in o.tags}
        healed_labels = {o.label for o in diss.healed_graph.objects}
        assert original_vulnerable.issubset(healed_labels)

    def test_dissolution_feasibility_scales_with_severity(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        diss = next(p for p in result.paths if p.path_type == PathType.DISSOLUTION)
        # Severe problem → higher dissolution feasibility.
        assert diss.feasibility_score >= 0.3


# ---------------------------------------------------------------------------
# Recursive Validation tests
# ---------------------------------------------------------------------------

class TestRecursiveValidation:
    def test_all_paths_have_validation(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        for path in result.paths:
            assert path.validation is not None

    def test_all_paths_are_anchor_aligned(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        for path in result.paths:
            assert path.validation is not None
            assert path.validation.is_aligned is True, (
                f"{path.path_type.value} failed anchor validation"
            )

    def test_unity_scores_positive(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        for path in result.paths:
            assert path.unity_alignment_score > 0

    def test_validation_uses_shared_anchor(self):
        anchor = AxiomAnchor(alignment_threshold=0.9)
        dream = DreamEngine(anchor=anchor)
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)

        graph = translator.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        report = decon.analyse(graph)
        result = dream.dream(report, graph)

        # All paths should still pass even with a high threshold.
        for path in result.paths:
            assert path.validation is not None


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

class TestPossibilityReportSerialisation:
    def test_to_json_valid(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "possibilityReport" in data

    def test_as_dict_has_required_fields(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        data = result.as_dict()
        pr = data["possibilityReport"]
        assert "sourceText" in pr
        assert "originalDisharmony" in pr
        assert "paths" in pr
        assert "recommendedPath" in pr
        assert len(pr["paths"]) == 3

    def test_each_path_has_scores(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        data = result.as_dict()
        for path_dict in data["possibilityReport"]["paths"]:
            assert "unityAlignmentScore" in path_dict
            assert "feasibilityScore" in path_dict
            assert "healedGraph" in path_dict
            assert "validation" in path_dict

    def test_dream_path_as_dict(self):
        dream, report, graph = _build_extractive_pipeline()
        result = dream.dream(report, graph)
        for path in result.paths:
            d = path.as_dict()
            assert d["pathType"] in ("reform", "reinvention", "dissolution")
            assert isinstance(d["title"], str)
            assert isinstance(d["description"], str)


# ---------------------------------------------------------------------------
# End-to-end with different scenarios
# ---------------------------------------------------------------------------

class TestEndToEndDream:
    def test_surveillance_scenario(self):
        dream, report, graph = _build_surveillance_pipeline()
        result = dream.dream(report, graph)

        assert len(result.paths) == 3
        reform = next(p for p in result.paths if p.path_type == PathType.REFORM)
        morph_labels = {m.label for m in reform.healed_graph.morphisms}
        assert "Exploitation" not in morph_labels
        assert "Surveillance" not in morph_labels
        assert "Neglect" not in morph_labels

    def test_surveillance_all_paths_aligned(self):
        dream, report, graph = _build_surveillance_pipeline()
        result = dream.dream(report, graph)
        for path in result.paths:
            assert path.validation is not None
            assert path.validation.is_aligned is True

    def test_aligned_scenario_still_works(self):
        """Dream Engine should handle an already-aligned report gracefully."""
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)

        graph = translator.translate(
            "A community health program that empowers workers "
            "and protects the environment."
        )
        report = decon.analyse(graph)

        # Even for an aligned report, dream() should still produce paths.
        result = dream.dream(report, graph)
        assert len(result.paths) == 3
