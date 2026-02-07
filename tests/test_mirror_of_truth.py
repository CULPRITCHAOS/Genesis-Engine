"""Tests for Module 1.7 — The Mirror of Truth (Sprint 8)."""

import os

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import CategoricalGraph, Object, Morphism
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.mirror_of_truth import (
    CritiqueFinding,
    MirrorOfTruth,
    RefinementTrace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_grid_war_graph() -> CategoricalGraph:
    """Build the Oklahoma Grid War conflict graph directly."""
    graph = CategoricalGraph(
        source_text=(
            "A utility provider shifts data centre infrastructure costs "
            "onto residential ratepayers through tariff riders under "
            "shareholder primacy obligations."
        ),
    )
    residential = Object(
        id="obj-residential", label="Residential_Ratepayer",
        tags=["stakeholder", "vulnerable", "ratepayer"],
    )
    datacenter = Object(
        id="obj-datacenter", label="Hyperscale_Data_Center",
        tags=["stakeholder", "actor", "corporate"],
    )
    utility = Object(
        id="obj-utility", label="Utility_Provider",
        tags=["stakeholder", "actor", "shareholder_primacy_risk"],
    )
    grid = Object(
        id="obj-grid", label="Oklahoma_Grid",
        tags=["value", "shared", "critical_infrastructure"],
    )
    graph.objects.extend([residential, datacenter, utility, grid])

    graph.morphisms.extend([
        Morphism(
            id="mor-cost-shift", label="Cost_Shifting",
            source="obj-utility", target="obj-residential",
            tags=["extraction", "exploitation", "tariff_rider",
                  "infrastructure_cost_pass_through"],
        ),
        Morphism(
            id="mor-below-market", label="Below_Market_Rate_Agreement",
            source="obj-utility", target="obj-datacenter",
            tags=["service", "preferential_treatment"],
        ),
        Morphism(
            id="mor-shareholder", label="Shareholder_Value_Maximization",
            source="obj-utility", target="obj-utility",
            tags=["extraction", "profit_priority", "fiduciary_duty",
                  "maximize_value"],
        ),
        Morphism(
            id="mor-strain", label="Grid_Capacity_Strain",
            source="obj-datacenter", target="obj-grid",
            tags=["extraction", "capacity_demand", "reliability_threat"],
        ),
        Morphism(
            id="mor-dependency", label="Grid_Dependency",
            source="obj-residential", target="obj-grid",
            tags=["collaboration", "essential_service"],
        ),
        Morphism(
            id="mor-rate", label="Rate_Increase_Burden",
            source="obj-grid", target="obj-residential",
            tags=["extraction", "cost_pass_through", "regressive_impact"],
        ),
    ])
    return graph


def _build_harmless_graph() -> CategoricalGraph:
    """Build a simple harmonious graph for negative testing."""
    graph = CategoricalGraph(
        source_text="A cooperative energy community sharing solar power.",
    )
    graph.objects.extend([
        Object(id="obj-coop", label="Solar_Cooperative",
               tags=["stakeholder", "actor", "cooperative"]),
        Object(id="obj-member", label="Community_Member",
               tags=["stakeholder", "vulnerable"]),
        Object(id="obj-solar", label="Shared_Solar_Farm",
               tags=["value", "shared"]),
    ])
    graph.morphisms.extend([
        Morphism(id="mor-share", label="Energy_Sharing",
                 source="obj-coop", target="obj-member",
                 tags=["service", "empowerment", "collaboration"]),
        Morphism(id="mor-govern", label="Democratic_Governance",
                 source="obj-member", target="obj-coop",
                 tags=["collaboration", "empowerment"]),
        Morphism(id="mor-steward", label="Stewardship",
                 source="obj-coop", target="obj-solar",
                 tags=["care", "protection", "sustainability"]),
    ])
    return graph


def _run_grid_war_pipeline():
    """Full pipeline: deconstruct + dream + mirror."""
    anchor = AxiomAnchor()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    mirror = MirrorOfTruth(
        anchor=anchor,
        vulnerability_priority="Residential_Ratepayer",
    )

    graph = _build_grid_war_graph()
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    selected, trace = mirror.critique_and_refine(possibility, report, graph)

    return graph, report, possibility, selected, trace, mirror


# ---------------------------------------------------------------------------
# RefinementTrace core tests
# ---------------------------------------------------------------------------

class TestRefinementTrace:
    def test_trace_serialisation(self):
        trace = RefinementTrace(
            surface_alignment_detected=True,
            deep_disharmony_categories=["cost_shifting"],
            mirror_score=4.5,
        )
        d = trace.as_dict()
        assert "refinementTrace" in d
        rt = d["refinementTrace"]
        assert rt["surfaceAlignmentDetected"] is True
        assert rt["mirrorScore"] == 4.5

    def test_trace_to_json(self):
        trace = RefinementTrace(surface_alignment_detected=False)
        json_str = trace.to_json()
        assert "refinementTrace" in json_str

    def test_trace_defaults(self):
        trace = RefinementTrace(surface_alignment_detected=False)
        assert trace.mirror_score == 10.0
        assert trace.reinvention_triggered is False
        assert trace.vulnerable_node_protected is True
        assert trace.deep_disharmony_categories == []
        assert trace.critique_findings == []


# ---------------------------------------------------------------------------
# CritiqueFinding tests
# ---------------------------------------------------------------------------

class TestCritiqueFinding:
    def test_finding_serialisation(self):
        finding = CritiqueFinding(
            category="cost_shifting",
            severity=8.0,
            description="Test finding",
            evidence=["Evidence 1"],
            affected_nodes=["Node_A"],
        )
        d = finding.as_dict()
        assert d["category"] == "cost_shifting"
        assert d["severity"] == 8.0
        assert len(d["evidence"]) == 1
        assert len(d["affectedNodes"]) == 1


# ---------------------------------------------------------------------------
# Mirror of Truth — Grid War scenario
# ---------------------------------------------------------------------------

class TestMirrorGridWar:
    def test_detects_surface_alignment(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        assert trace.surface_alignment_detected is True

    def test_detects_deep_disharmony(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        # The Mirror catalogues original disharmony categories
        assert len(trace.deep_disharmony_categories) > 0

    def test_detects_cost_shifting(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        categories = trace.deep_disharmony_categories
        # Should detect cost shifting patterns in the original graph
        has_cost_related = any(
            "cost" in c or "extraction" in c or "wealth" in c
            for c in categories
        )
        assert has_cost_related

    def test_detects_shareholder_primacy(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        categories = trace.deep_disharmony_categories
        has_primacy = any("shareholder" in c for c in categories)
        assert has_primacy

    def test_detects_grid_fragility(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        categories = trace.deep_disharmony_categories
        has_fragility = any("fragility" in c for c in categories)
        assert has_fragility

    def test_mirror_score_penalised(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        # Grid War graph has many issues; score should be well below 10
        assert trace.mirror_score < 8.0

    def test_dream_engine_already_selects_structural_path(self):
        _, _, possibility, _, _, _ = _run_grid_war_pipeline()
        # The Dream Engine should already prefer dissolution/reinvention
        # due to incentive instability in the DisharmonyReport
        assert possibility.recommended_path != "reform"

    def test_reinvention_not_needed_when_already_structural(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        # The Mirror's reinvention trigger only fires for Reform paths.
        # Since the Dream Engine already selected a structural path,
        # reinvention override is not triggered.
        if trace.original_path_type in ("reinvention", "dissolution"):
            assert trace.reinvention_triggered is False

    def test_mandatory_repair_nonempty(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        assert len(trace.mandatory_repair) > 0

    def test_mandatory_repair_addresses_categories(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        repair = trace.mandatory_repair.lower()
        # Should address at least one structural issue
        assert any(term in repair for term in [
            "cost", "tariff", "shareholder", "incentive",
            "fragility", "equity", "repair",
        ])

    def test_recommended_path_not_reform(self):
        _, _, _, selected, trace, _ = _run_grid_war_pipeline()
        # Should not recommend reform for the grid war scenario
        assert trace.recommended_path_type != "reform"

    def test_selected_path_is_structural(self):
        _, _, _, selected, _, _ = _run_grid_war_pipeline()
        # Should select reinvention or dissolution, not reform
        assert selected.path_type in (PathType.REINVENTION, PathType.DISSOLUTION)

    def test_critique_findings_present(self):
        _, _, _, _, trace, _ = _run_grid_war_pipeline()
        assert len(trace.critique_findings) > 0

    def test_incentive_instability_detected(self):
        _, report, _, _, _, _ = _run_grid_war_pipeline()
        assert report.incentive_instability is True
        assert report.incentive_stability_score < 5.0


# ---------------------------------------------------------------------------
# Mirror of Truth — harmless scenario (negative test)
# ---------------------------------------------------------------------------

class TestMirrorHarmlessScenario:
    def test_no_surface_alignment_on_harmless(self):
        anchor = AxiomAnchor()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        mirror = MirrorOfTruth(anchor=anchor)

        graph = _build_harmless_graph()
        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)

        # Critique the recommended path
        paths_by_type = {
            p.path_type.value: p for p in possibility.paths
        }
        selected = paths_by_type.get(
            possibility.recommended_path, possibility.paths[0],
        )

        trace = mirror.critique(selected, report, graph, possibility)
        # Harmless graph should have high mirror score
        assert trace.mirror_score >= 5.0

    def test_no_reinvention_on_harmless(self):
        anchor = AxiomAnchor()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        mirror = MirrorOfTruth(anchor=anchor)

        graph = _build_harmless_graph()
        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)

        paths_by_type = {
            p.path_type.value: p for p in possibility.paths
        }
        selected = paths_by_type.get(
            possibility.recommended_path, possibility.paths[0],
        )

        trace = mirror.critique(selected, report, graph, possibility)
        # Should not trigger reinvention on a cooperative graph
        assert trace.reinvention_triggered is False


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

class TestScenarioLoading:
    def test_load_grid_war_scenario(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scenarios", "grid_war_2026.json",
        )
        scenario = MirrorOfTruth.load_scenario(path)
        assert scenario["scenario"] == "The 2026 Oklahoma Grid War"

    def test_scenario_to_graph(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scenarios", "grid_war_2026.json",
        )
        scenario = MirrorOfTruth.load_scenario(path)
        graph = MirrorOfTruth.scenario_to_graph(scenario)
        assert len(graph.objects) == 4
        assert len(graph.morphisms) == 6

    def test_scenario_graph_has_vulnerable_node(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scenarios", "grid_war_2026.json",
        )
        scenario = MirrorOfTruth.load_scenario(path)
        graph = MirrorOfTruth.scenario_to_graph(scenario)
        vulnerable = [o for o in graph.objects if "vulnerable" in o.tags]
        assert len(vulnerable) >= 1
        assert any("Residential" in o.label for o in vulnerable)

    def test_scenario_graph_has_primacy_risk(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scenarios", "grid_war_2026.json",
        )
        scenario = MirrorOfTruth.load_scenario(path)
        graph = MirrorOfTruth.scenario_to_graph(scenario)
        primacy_nodes = [
            o for o in graph.objects
            if "shareholder_primacy_risk" in o.tags
        ]
        assert len(primacy_nodes) >= 1


# ---------------------------------------------------------------------------
# critique_and_refine integration
# ---------------------------------------------------------------------------

class TestCritiqueAndRefine:
    def test_returns_path_and_trace(self):
        _, _, _, selected, trace, _ = _run_grid_war_pipeline()
        assert selected is not None
        assert trace is not None
        assert isinstance(trace, RefinementTrace)

    def test_override_selects_better_path(self):
        _, _, possibility, selected, trace, _ = _run_grid_war_pipeline()
        if trace.reinvention_triggered:
            # Selected path should be non-reform
            assert selected.path_type != PathType.REFORM
