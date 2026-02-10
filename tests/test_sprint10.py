"""
Sprint 10 — Sovereign Governance & The Oklahoma Water/Grid War

Comprehensive test suite covering:
- Robustness Harness (Beta-distribution priors, Monte Carlo, Fork Operator)
- Governance Report (Production Lexicon, Sacred Language stripping)
- Aria Interface extensions (compare_manifestos, invariant tracker)
- Grid War 2026 scenario (v3.0.0 with Tulsa/Moore basin nodes)
- Sovereign Index generation
- Full Covenant Actuation pipeline
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from genesis_engine.core.axiomlogix import CategoricalGraph, Object, Morphism
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
)
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer
from genesis_engine.core.mirror_of_truth import (
    MirrorOfTruth,
    RefinementTrace,
    CritiqueFinding,
)
from genesis_engine.core.robustness_harness import (
    BetaPrior,
    DecentralizedForkOperator,
    ForkResult,
    HardInvariant,
    InvariantViolation,
    MonteCarloSimResult,
    RobustnessHarness,
    RobustnessResult,
)
from genesis_engine.core.governance_report import (
    ConflictEntry,
    GovernanceReport,
    GovernanceReportBuilder,
    SovereignIndexGenerator,
    SACRED_TO_PRODUCTION,
    translate_to_production,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_PATH = str(Path(__file__).parent.parent / "scenarios" / "grid_war_2026.json")


@pytest.fixture
def scenario_data() -> dict[str, Any]:
    """Load the Grid War 2026 scenario."""
    with open(SCENARIO_PATH) as f:
        return json.load(f)


@pytest.fixture
def conflict_graph(scenario_data: dict[str, Any]) -> CategoricalGraph:
    """Build a CategoricalGraph from the scenario."""
    return MirrorOfTruth.scenario_to_graph(scenario_data)


@pytest.fixture
def soul() -> GenesisSoul:
    """Create a fresh GenesisSoul."""
    return ContinuityBridge.create_soul()


@pytest.fixture
def aria(soul: GenesisSoul) -> AriaInterface:
    """Create an AriaInterface with no colors for testing."""
    interface = AriaInterface(use_colors=False)
    return interface


@pytest.fixture
def sample_trace() -> RefinementTrace:
    """Create a sample RefinementTrace for testing."""
    return RefinementTrace(
        mirror_score=2.5,
        surface_alignment_detected=True,
        deep_disharmony_categories=[
            "cost_shifting_to_vulnerable",
            "shareholder_primacy_extraction",
            "unsustainable_water_withdrawal",
        ],
        vulnerable_node_protected=False,
        critique_findings=[
            CritiqueFinding(
                category="cost_shifting_to_vulnerable",
                description="Infrastructure costs socialised onto residential ratepayers",
                severity=9.0,
                evidence=["$47/month rate increase", "62% allocated to residential"],
            ),
            CritiqueFinding(
                category="unsustainable_water_withdrawal",
                description="Water demand exceeds aquifer recharge by 2.33x",
                severity=8.5,
                evidence=["42 MGD demand vs 18 MGD recharge"],
            ),
        ],
        reinvention_triggered=True,
        original_path_type="reform",
        recommended_path_type="reinvention",
        mandatory_repair=(
            "Cost-causation tariff allocation | "
            "Cooling water reduction below 18 MGD"
        ),
    )


# ---------------------------------------------------------------------------
# Beta Prior Tests
# ---------------------------------------------------------------------------

class TestBetaPrior:
    """Tests for the BetaPrior class."""

    def test_creation(self) -> None:
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        assert prior.alpha == 2.0
        assert prior.beta == 8.0
        assert prior.label == "Test"

    def test_mean(self) -> None:
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        assert abs(prior.mean - 0.2) < 0.001

    def test_variance(self) -> None:
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        expected = (2.0 * 8.0) / (10.0 ** 2 * 11.0)
        assert abs(prior.variance - expected) < 0.001

    def test_update(self) -> None:
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        posterior = prior.update(successes=3, failures=1)
        assert posterior.alpha == 5.0
        assert posterior.beta == 9.0
        assert posterior.label == "Test"

    def test_sample_in_range(self) -> None:
        import random
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        rng = random.Random(42)
        for _ in range(100):
            sample = prior.sample(rng)
            assert 0.0 <= sample <= 1.0

    def test_as_dict(self) -> None:
        prior = BetaPrior(alpha=2.0, beta=8.0, label="Test")
        d = prior.as_dict()
        assert d["label"] == "Test"
        assert d["alpha"] == 2.0
        assert d["beta"] == 8.0
        assert "mean" in d
        assert "variance" in d


# ---------------------------------------------------------------------------
# Invariant Violation Tests
# ---------------------------------------------------------------------------

class TestInvariantViolation:
    """Tests for the InvariantViolation class."""

    def test_creation(self) -> None:
        v = InvariantViolation(
            invariant=HardInvariant.EQUITY,
            description="Test violation",
            severity="CRITICAL",
            metric_name="test_metric",
            metric_value=0.0,
            threshold=1.0,
        )
        assert v.invariant == "equity"
        assert v.severity == "CRITICAL"

    def test_as_dict(self) -> None:
        v = InvariantViolation(
            invariant=HardInvariant.WATER_FLOOR,
            description="Water floor violated",
            severity="CRITICAL",
            metric_name="surplus_pct",
            metric_value=-133.0,
            threshold=25.0,
        )
        d = v.as_dict()
        assert d["invariant"] == "water_floor"
        assert d["severity"] == "CRITICAL"
        assert d["metricValue"] == -133.0

    def test_hard_invariant_constants(self) -> None:
        assert HardInvariant.EQUITY == "equity"
        assert HardInvariant.SUSTAINABILITY == "sustainability"
        assert HardInvariant.AGENCY == "agency"
        assert HardInvariant.WATER_FLOOR == "water_floor"
        assert HardInvariant.COST_CAUSATION == "cost_causation"


# ---------------------------------------------------------------------------
# Decentralized Fork Operator Tests
# ---------------------------------------------------------------------------

class TestDecentralizedForkOperator:
    """Tests for the DecentralizedForkOperator."""

    def test_fork_hostile_node(self, conflict_graph: CategoricalGraph) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Switch to treated wastewater",
        )
        assert fork_result.hostile_node_label == "Hyperscale_Data_Center"
        assert fork_result.forked_object_count < fork_result.original_object_count
        assert fork_result.forked_morphism_count < fork_result.original_morphism_count
        assert len(fork_result.protected_basins) > 0

    def test_fork_preserves_community_nodes(
        self, conflict_graph: CategoricalGraph,
    ) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Switch to closed-loop cooling",
        )
        forked_labels = {
            obj.label for obj in fork_result.forked_graph.objects
        }
        assert "Residential_Ratepayer" in forked_labels
        assert "Oklahoma_Water_Supply" in forked_labels
        assert "Hyperscale_Data_Center" not in forked_labels

    def test_fork_preserves_protective_morphisms(
        self, conflict_graph: CategoricalGraph,
    ) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Refuse wastewater switch",
        )
        forked_labels = {
            m.label for m in fork_result.forked_graph.morphisms
        }
        # Community dependency and protective morphisms should survive
        assert "Community_Water_Dependency" in forked_labels

    def test_fork_nonexistent_node(self, conflict_graph: CategoricalGraph) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "NonExistent_Node",
            "Some repair",
        )
        # Should return original graph unchanged
        assert fork_result.forked_object_count == fork_result.original_object_count

    def test_fork_result_as_dict(self, conflict_graph: CategoricalGraph) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Switch to treated wastewater",
        )
        d = fork_result.as_dict()
        assert "forkOperator" in d
        assert d["forkOperator"]["hostileNodeLabel"] == "Hyperscale_Data_Center"

    def test_fork_protects_tulsa_basin(
        self, conflict_graph: CategoricalGraph,
    ) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Refuse wastewater switch",
        )
        assert "Tulsa_Water_Basin" in fork_result.protected_basins

    def test_fork_protects_moore_basin(
        self, conflict_graph: CategoricalGraph,
    ) -> None:
        fork_result = DecentralizedForkOperator.fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Refuse wastewater switch",
        )
        assert "Moore_Water_Basin" in fork_result.protected_basins


# ---------------------------------------------------------------------------
# Robustness Harness Tests
# ---------------------------------------------------------------------------

class TestRobustnessHarness:
    """Tests for the RobustnessHarness."""

    def test_evaluate_returns_result(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        assert isinstance(result, RobustnessResult)
        assert 0.0 <= result.combined_robustness_score <= 10.0

    def test_blackout_sim_uses_beta_prior(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(
            blackout_prior=BetaPrior(2.0, 8.0, "Blackout Shock"),
            seed=42,
            monte_carlo_runs=50,
        )
        result = harness.evaluate(conflict_graph, scenario_data)
        bs = result.blackout_sim
        assert bs.prior.label == "Blackout Shock"
        assert bs.prior.alpha == 2.0
        # Posterior should be updated with graph evidence
        assert bs.posterior.alpha != bs.prior.alpha or bs.posterior.beta != bs.prior.beta

    def test_drought_sim_uses_beta_prior(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(
            drought_prior=BetaPrior(3.0, 7.0, "Drought Event"),
            seed=42,
            monte_carlo_runs=50,
        )
        result = harness.evaluate(conflict_graph, scenario_data)
        de = result.drought_sim
        assert de.prior.label == "Drought Event"
        assert de.posterior.alpha > de.prior.alpha or de.posterior.beta > de.prior.beta

    def test_detects_cost_causation_violation(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        cost_violations = [
            v for v in result.invariant_violations
            if v.invariant == HardInvariant.COST_CAUSATION
        ]
        assert len(cost_violations) > 0
        assert cost_violations[0].severity == "CRITICAL"

    def test_detects_equity_violation(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        equity_violations = [
            v for v in result.invariant_violations
            if v.invariant == HardInvariant.EQUITY
        ]
        assert len(equity_violations) > 0

    def test_triggers_fork_for_hostile_agent(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        assert len(result.fork_results) > 0
        assert result.fork_results[0].hostile_node_label == "Hyperscale_Data_Center"

    def test_manual_fork_trigger(
        self, conflict_graph: CategoricalGraph,
    ) -> None:
        harness = RobustnessHarness(seed=42)
        fork_result = harness.trigger_fork(
            conflict_graph,
            "Hyperscale_Data_Center",
            "Refuse wastewater treatment",
        )
        assert fork_result.hostile_node_label == "Hyperscale_Data_Center"
        assert fork_result.forked_object_count < fork_result.original_object_count

    def test_result_as_dict(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        d = result.as_dict()
        assert "robustnessHarness" in d
        assert "blackoutShock" in d["robustnessHarness"]
        assert "droughtEvent" in d["robustnessHarness"]
        assert "forkResults" in d["robustnessHarness"]

    def test_scenario_rejection(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        """The Grid War scenario should FAIL the robustness harness."""
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)
        assert result.passed is False
        assert len(result.blocking_reasons) > 0


# ---------------------------------------------------------------------------
# Production Lexicon Tests
# ---------------------------------------------------------------------------

class TestProductionLexicon:
    """Tests for the Sacred Language → Production Lexicon translation."""

    def test_translate_prime_directive(self) -> None:
        result = translate_to_production("Prime Directive violated")
        assert "Hard Invariant" in result
        assert "Prime Directive" not in result

    def test_translate_disharmony(self) -> None:
        result = translate_to_production("Disharmony detected in graph")
        assert "Invariant Violation" in result

    def test_translate_morphism(self) -> None:
        result = translate_to_production("Morphism from A to B")
        assert "Relationship Operator" in result

    def test_translate_dream_path(self) -> None:
        result = translate_to_production("Dream Path: reform")
        assert "Remediation Projection" in result

    def test_translate_crystallization(self) -> None:
        result = translate_to_production("Crystallization complete")
        assert "Covenant Actuation" in result

    def test_translate_soul(self) -> None:
        result = translate_to_production("Genesis Soul exported")
        assert "EventStore" in result

    def test_translate_forge_artifact(self) -> None:
        result = translate_to_production("Forge Artifact produced")
        assert "Governance Instrument" in result

    def test_translate_wisdom_log(self) -> None:
        result = translate_to_production("Wisdom Log entry added")
        assert "Audit Trail" in result

    def test_translate_surface_alignment(self) -> None:
        result = translate_to_production("Surface Alignment detected")
        assert "Deceptive Compliance" in result

    def test_translate_legal_gravity_well(self) -> None:
        result = translate_to_production("Legal Gravity Well found")
        assert "Regulatory Capture Pattern" in result

    def test_translate_preserves_non_sacred_text(self) -> None:
        result = translate_to_production("Regular text with no sacred terms")
        assert result == "Regular text with no sacred terms"

    def test_all_sacred_terms_have_translations(self) -> None:
        """Every key in the translation map should produce a different output."""
        for sacred, production in SACRED_TO_PRODUCTION.items():
            assert sacred != production


# ---------------------------------------------------------------------------
# Governance Report Tests
# ---------------------------------------------------------------------------

class TestGovernanceReport:
    """Tests for the GovernanceReport and GovernanceReportBuilder."""

    def test_build_report(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        assert isinstance(report, GovernanceReport)
        assert report.report_id.startswith("GOV-")
        assert len(report.eventstore_hash) == 64

    def test_report_has_conflicts(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        assert len(report.conflicts) >= 1
        assert report.conflicts[0].name == "The 2026 Oklahoma Grid War"

    def test_report_has_pso_conflict(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        pso_conflicts = [
            c for c in report.conflicts if "PSO" in c.name
        ]
        assert len(pso_conflicts) == 1
        assert pso_conflicts[0].status == "REJECTED"

    def test_report_detects_legal_gravity_wells(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        main_conflict = report.conflicts[0]
        assert len(main_conflict.legal_gravity_wells) > 0

    def test_report_to_json_strips_sacred_language(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        json_output = report.to_json()
        assert "Prime Directive" not in json_output
        assert "Disharmony" not in json_output or "Invariant Violation" in json_output

    def test_report_preserves_iam_hash(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        envelope = ContinuityBridge.export_soul(soul)
        expected_hash = envelope["genesis_soul"]["integrityHash"]
        assert report.eventstore_hash == expected_hash

    def test_report_as_dict(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        d = report.as_dict()
        assert "governanceReport" in d
        assert "eventstoreHash" in d["governanceReport"]
        assert "conflicts" in d["governanceReport"]


# ---------------------------------------------------------------------------
# Conflict Entry Tests
# ---------------------------------------------------------------------------

class TestConflictEntry:
    """Tests for the ConflictEntry data structure."""

    def test_creation(self) -> None:
        entry = ConflictEntry(
            name="Test Conflict",
            legislative_refs=["HB 2992"],
            sustainability_score=3.5,
            legal_gravity_wells=["Shareholder Primacy"],
            invariant_violations=["Cost Causation"],
            status="REJECTED",
        )
        assert entry.name == "Test Conflict"
        assert entry.status == "REJECTED"

    def test_as_dict(self) -> None:
        entry = ConflictEntry(
            name="Test",
            legislative_refs=["HB 2992"],
            sustainability_score=5.0,
            legal_gravity_wells=[],
            invariant_violations=[],
            status="ACTIVE",
        )
        d = entry.as_dict()
        assert d["name"] == "Test"
        assert d["sustainabilityScore"] == 5.0


# ---------------------------------------------------------------------------
# Sovereign Index Tests
# ---------------------------------------------------------------------------

class TestSovereignIndex:
    """Tests for the SovereignIndexGenerator."""

    def test_generate_returns_markdown(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert isinstance(md, str)
        assert "# State of the Sovereignty" in md

    def test_index_has_frontmatter(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert md.startswith("---")
        assert "report_id:" in md
        assert "eventstore_hash:" in md

    def test_index_lists_conflicts(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert "Active Conflicts" in md
        assert "2026 Oklahoma Grid War" in md

    def test_index_lists_pso_case(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert "PSO Rate Case" in md
        assert "REJECTED" in md

    def test_index_has_iam_hash(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert "I AM Hash" in md

    def test_index_has_vault_links(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        report = GovernanceReportBuilder.build(soul=soul, scenario=scenario_data)
        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert "[[Manifesto]]" in md


# ---------------------------------------------------------------------------
# Grid War 2026 Scenario Tests (v3.0.0)
# ---------------------------------------------------------------------------

class TestGridWar2026V3:
    """Tests for the updated Grid War 2026 scenario."""

    def test_scenario_version(self, scenario_data: dict[str, Any]) -> None:
        assert scenario_data["version"] == "4.0.0"
        assert scenario_data["sprint"] == 11

    def test_tulsa_basin_node_exists(self, scenario_data: dict[str, Any]) -> None:
        objects = scenario_data["conflict_graph"]["objects"]
        labels = {obj["label"] for obj in objects}
        assert "Tulsa_Water_Basin" in labels

    def test_moore_basin_node_exists(self, scenario_data: dict[str, Any]) -> None:
        objects = scenario_data["conflict_graph"]["objects"]
        labels = {obj["label"] for obj in objects}
        assert "Moore_Water_Basin" in labels

    def test_tulsa_basin_tagged_shadow_entity(
        self, scenario_data: dict[str, Any],
    ) -> None:
        objects = scenario_data["conflict_graph"]["objects"]
        tulsa = next(o for o in objects if o["label"] == "Tulsa_Water_Basin")
        assert "shadow_entity" in tulsa["tags"]
        assert "water_basin" in tulsa["tags"]

    def test_moore_basin_tagged_shadow_entity(
        self, scenario_data: dict[str, Any],
    ) -> None:
        objects = scenario_data["conflict_graph"]["objects"]
        moore = next(o for o in objects if o["label"] == "Moore_Water_Basin")
        assert "shadow_entity" in moore["tags"]
        assert "water_basin" in moore["tags"]

    def test_tulsa_withdrawal_morphism(
        self, scenario_data: dict[str, Any],
    ) -> None:
        morphisms = scenario_data["conflict_graph"]["morphisms"]
        tulsa_withdrawal = next(
            m for m in morphisms if m["label"] == "Tulsa_Basin_Withdrawal"
        )
        assert tulsa_withdrawal["source"] == "obj-datacenter"
        assert tulsa_withdrawal["target"] == "obj-tulsa-basin"
        assert "unsustainable_withdrawal" in tulsa_withdrawal["tags"]

    def test_moore_withdrawal_morphism(
        self, scenario_data: dict[str, Any],
    ) -> None:
        morphisms = scenario_data["conflict_graph"]["morphisms"]
        moore_withdrawal = next(
            m for m in morphisms if m["label"] == "Moore_Basin_Withdrawal"
        )
        assert moore_withdrawal["source"] == "obj-datacenter"
        assert moore_withdrawal["target"] == "obj-moore-basin"
        assert "unsustainable_withdrawal" in moore_withdrawal["tags"]

    def test_basin_protection_morphisms(
        self, scenario_data: dict[str, Any],
    ) -> None:
        morphisms = scenario_data["conflict_graph"]["morphisms"]
        protection_morphisms = [
            m for m in morphisms if "basin_protection" in m["tags"]
        ]
        assert len(protection_morphisms) == 2  # Tulsa + Moore

    def test_water_floor_invariant_defined(
        self, scenario_data: dict[str, Any],
    ) -> None:
        constraints = scenario_data["constraints"]
        wf = constraints["water_floor_invariant"]
        assert wf["residential_surplus_pct"] == 25
        assert wf["simulation_horizon_years"] == 50

    def test_cost_causation_invariant_defined(
        self, scenario_data: dict[str, Any],
    ) -> None:
        constraints = scenario_data["constraints"]
        cc = constraints["cost_causation_invariant"]
        assert cc["hill_cost_allocation_pct"] == 100

    def test_basin_constraints_tulsa(
        self, scenario_data: dict[str, Any],
    ) -> None:
        tulsa = scenario_data["constraints"]["basin_constraints"]["tulsa_basin"]
        assert tulsa["sustainable_withdrawal_mgd"] == 12.0
        assert tulsa["projected_datacenter_demand_mgd"] == 28.0
        assert tulsa["residential_population_served"] == 600000

    def test_basin_constraints_moore(
        self, scenario_data: dict[str, Any],
    ) -> None:
        moore = scenario_data["constraints"]["basin_constraints"]["moore_basin"]
        assert moore["sustainable_withdrawal_mgd"] == 6.0
        assert moore["sole_source_aquifer"] is True
        assert moore["residential_population_served"] == 120000

    def test_pso_rate_case_included(
        self, scenario_data: dict[str, Any],
    ) -> None:
        pso = scenario_data["pso_rate_case"]
        assert pso["case_id"] == "PSO-2026-GR-001"
        assert pso["revenue_increase_requested_usd"] == 380000000
        assert pso["hill_cost_allocated_to_hill_pct"] == 0
        assert pso["hill_cost_allocated_to_residential_pct"] == 62
        assert len(pso["invariant_violations"]) == 3

    def test_8_objects_in_graph(
        self, scenario_data: dict[str, Any],
    ) -> None:
        objects = scenario_data["conflict_graph"]["objects"]
        assert len(objects) == 8  # Original 6 + Tulsa + Moore

    def test_16_morphisms_in_graph(
        self, scenario_data: dict[str, Any],
    ) -> None:
        morphisms = scenario_data["conflict_graph"]["morphisms"]
        assert len(morphisms) == 16  # Original 10 + 6 basin morphisms

    def test_three_legislative_references(
        self, scenario_data: dict[str, Any],
    ) -> None:
        refs = scenario_data["context"]["legislative_references"]
        assert len(refs) == 3
        bills = {ref["bill"] for ref in refs}
        assert "HB 2992" in bills
        assert "SB 1488" in bills
        assert "PSO July 2026 Rate Case" in bills


# ---------------------------------------------------------------------------
# Aria Interface Sprint 10 Extension Tests
# ---------------------------------------------------------------------------

class TestAriaInterfaceSprint10:
    """Tests for Sprint 10 Aria Interface extensions."""

    def test_compare_manifestos(self, aria: AriaInterface) -> None:
        bill = {
            "bill": "HB 2992",
            "title": "Oklahoma HILL Act",
            "summary": "Cost allocation framework",
            "disharmony_vector": "Enables cost-shifting to residential ratepayers",
        }
        blueprint = {
            "title": "Regenerative Grid Covenant",
            "cost_allocation": "100% to Hyperscale_Node",
            "water_policy": "Below aquifer recharge",
            "ratepayer_protection": "Zero cost-shifting",
        }
        delta = aria.compare_manifestos(bill, blueprint, verbose=False)
        assert "fields" in delta
        assert "Cost Allocation" in delta["fields"]
        assert len(delta["invariant_violations"]) > 0

    def test_invariant_tracker_without_scenario(
        self, aria: AriaInterface,
    ) -> None:
        violations = aria.invariant_tracker(verbose=False)
        assert violations == []

    def test_invariant_tracker_with_scenario(
        self,
        aria: AriaInterface,
        scenario_data: dict[str, Any],
        conflict_graph: CategoricalGraph,
    ) -> None:
        aria._active_scenario = scenario_data
        aria._active_graph = conflict_graph
        violations = aria.invariant_tracker(verbose=False)
        assert isinstance(violations, list)

    def test_robustness_exam(
        self,
        aria: AriaInterface,
        scenario_data: dict[str, Any],
        conflict_graph: CategoricalGraph,
    ) -> None:
        aria._active_scenario = scenario_data
        aria._active_graph = conflict_graph
        result = aria.robustness_exam(seed=42, verbose=False)
        assert isinstance(result, RobustnessResult)
        assert 0.0 <= result.combined_robustness_score <= 10.0

    def test_generate_governance_report(
        self,
        aria: AriaInterface,
        scenario_data: dict[str, Any],
        conflict_graph: CategoricalGraph,
        sample_trace: RefinementTrace,
    ) -> None:
        aria._active_scenario = scenario_data
        aria._active_graph = conflict_graph
        report = aria.generate_governance_report(
            trace=sample_trace, verbose=False,
        )
        assert isinstance(report, GovernanceReport)
        assert report.report_id.startswith("GOV-")

    def test_generate_sovereign_index(
        self,
        aria: AriaInterface,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        aria._active_scenario = scenario_data
        report = GovernanceReportBuilder.build(
            soul=aria.soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        md = aria.generate_sovereign_index(report, verbose=False)
        assert "# State of the Sovereignty" in md


# ---------------------------------------------------------------------------
# Renderer Tests
# ---------------------------------------------------------------------------

class TestAriaRendererSprint10:
    """Tests for Sprint 10 AriaRenderer methods."""

    def test_render_manifesto_comparison(self) -> None:
        renderer = AriaRenderer(use_colors=False)
        bill = {"bill": "HB 2992", "title": "HILL Act", "disharmony_vector": "cost-shifting"}
        blueprint = {"title": "Regenerative Covenant"}
        delta = {
            "fields": {
                "Cost Allocation": {
                    "bill": "Socialised",
                    "blueprint": "100% to HILL",
                    "status": "violation",
                },
            },
            "invariant_violations": [
                {"invariant": "cost_causation", "description": "Test"},
            ],
        }
        output = renderer.render_manifesto_comparison(bill, blueprint, delta)
        assert "MANIFESTO COMPARISON" in output
        assert "HB 2992" in output
        assert "violation" not in output.lower() or "Cost Allocation" in output

    def test_render_invariant_tracker_no_violations(self) -> None:
        renderer = AriaRenderer(use_colors=False)
        output = renderer.render_invariant_tracker([])
        assert "LIVE INVARIANT TRACKER" in output
        assert "All Hard Constraints satisfied" in output

    def test_render_invariant_tracker_with_violations(self) -> None:
        renderer = AriaRenderer(use_colors=False)
        violations = [
            InvariantViolation(
                invariant=HardInvariant.COST_CAUSATION,
                description="0% allocated to HILL",
                severity="CRITICAL",
                metric_name="allocation_pct",
                metric_value=0.0,
                threshold=100.0,
            ),
        ]
        output = renderer.render_invariant_tracker(violations)
        assert "Cost Causation" in output
        assert "CRITICAL" in output
        assert "1 violation" in output

    def test_render_robustness_result(self) -> None:
        renderer = AriaRenderer(use_colors=False)
        prior = BetaPrior(2.0, 8.0, "Test")
        sim = MonteCarloSimResult(
            scenario="blackout_shock",
            prior=prior,
            posterior=prior.update(5, 3),
            runs=100,
            survival_count=60,
            collapse_count=40,
            mean_survival_probability=0.6,
            event_probability=0.3,
        )
        result = RobustnessResult(
            blackout_sim=sim,
            drought_sim=sim,
            combined_robustness_score=4.5,
            invariant_violations=[],
            fork_results=[],
            passed=False,
            blocking_reasons=["Score too low"],
        )
        output = renderer.render_robustness_result(result)
        assert "ROBUSTNESS HARNESS" in output
        assert "Blackout Shock" in output
        assert "FAILED" in output

    def test_render_fork_result(self) -> None:
        renderer = AriaRenderer(use_colors=False)
        graph = CategoricalGraph()
        result = ForkResult(
            hostile_node_label="Test_Node",
            refused_repair="Test repair",
            original_object_count=5,
            original_morphism_count=8,
            forked_object_count=4,
            forked_morphism_count=5,
            protected_basins=["Basin_A", "Basin_B"],
            forked_graph=graph,
        )
        output = renderer.render_fork_result(result)
        assert "DECENTRALIZED FORK" in output
        assert "Test_Node" in output
        assert "Basin_A" in output


# ---------------------------------------------------------------------------
# Integration Tests — Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end integration tests for the Sprint 10 pipeline."""

    def test_full_covenant_actuation(
        self,
        scenario_data: dict[str, Any],
    ) -> None:
        """Test the complete covenant actuation pipeline."""
        # Step 1: Load scenario
        aria = AriaInterface(use_colors=False)
        scenario, graph = aria.load_conflict(SCENARIO_PATH, verbose=False)

        # Step 2: Compare manifestos
        hb_2992 = scenario["context"]["legislative_references"][0]
        delta = aria.compare_manifestos(
            hb_2992,
            {"title": "Regenerative Grid Covenant"},
            verbose=False,
        )
        assert "fields" in delta

        # Step 3: Run invariant tracker
        violations = aria.invariant_tracker(verbose=False)
        assert isinstance(violations, list)

        # Step 4: Run robustness exam
        result = aria.robustness_exam(seed=42, verbose=False)
        assert isinstance(result, RobustnessResult)

        # Step 5: Generate governance report
        report = aria.generate_governance_report(
            robustness_result=result, verbose=False,
        )
        assert isinstance(report, GovernanceReport)
        assert report.report_id.startswith("GOV-")

        # Step 6: Generate sovereign index
        md = aria.generate_sovereign_index(report, verbose=False)
        assert "State of the Sovereignty" in md
        assert "2026 Oklahoma Grid War" in md

    def test_pso_rate_case_rejected(
        self, scenario_data: dict[str, Any],
    ) -> None:
        """PSO Rate Case must be REJECTED for invariant violations."""
        pso = scenario_data["pso_rate_case"]
        assert pso["hill_cost_allocated_to_hill_pct"] == 0
        assert pso["hill_cost_allocated_to_residential_pct"] == 62
        assert len(pso["invariant_violations"]) >= 3

        # Verify specific violations
        violations = pso["invariant_violations"]
        invariant_names = {v["invariant"] for v in violations}
        assert "Cost Causation (HB 2992)" in invariant_names
        assert "Water Floor" in invariant_names

    def test_production_lexicon_in_report_json(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
        sample_trace: RefinementTrace,
    ) -> None:
        """Governance report JSON must use Production Lexicon only."""
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario_data,
            trace=sample_trace,
        )
        json_output = report.to_json()

        # These Sacred terms should NOT appear in production output
        # (some may appear as part of longer translated phrases, so
        # we check for standalone usage patterns)
        assert "Prime Directive" not in json_output
        assert "Dream Path" not in json_output
        assert "Forge Artifact" not in json_output
        assert "Wisdom Log" not in json_output

    def test_iam_hash_preserved_across_pipeline(
        self,
        soul: GenesisSoul,
        scenario_data: dict[str, Any],
    ) -> None:
        """The I AM hash must be consistent across all exports."""
        envelope = ContinuityBridge.export_soul(soul)
        expected_hash = envelope["genesis_soul"]["integrityHash"]

        report = GovernanceReportBuilder.build(
            soul=soul, scenario=scenario_data,
        )
        assert report.eventstore_hash == expected_hash

        md = SovereignIndexGenerator.generate(report, soul, scenario_data)
        assert expected_hash[:32] in md

    def test_water_floor_25_pct_surplus_invariant(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        """Water Floor: residential security must maintain 25% surplus."""
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)

        water_violations = [
            v for v in result.invariant_violations
            if v.invariant == HardInvariant.WATER_FLOOR
        ]
        # The scenario's 42 MGD demand vs 18 MGD recharge should violate
        assert len(water_violations) > 0

    def test_cost_causation_100_pct_invariant(
        self,
        conflict_graph: CategoricalGraph,
        scenario_data: dict[str, Any],
    ) -> None:
        """Cost Causation: 100% of HILL costs must go to Hyperscale_Node."""
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(conflict_graph, scenario_data)

        cost_violations = [
            v for v in result.invariant_violations
            if v.invariant == HardInvariant.COST_CAUSATION
        ]
        assert len(cost_violations) > 0
        assert cost_violations[0].severity == "CRITICAL"
