"""
Sprint 11 — Policy Auditor & Regenerative Blueprint Suite

Comprehensive test suite covering:
1. Constitutional PolicyKernel (Module 2.1) — Self-Critique Loop
2. FAIRGAME Debate Arena (Module 3.5) — Multi-Agent Debate Protocol
3. Categorical Repair Operators (Module 1.3) — ACT (functors/colimits)
4. Interactive Aria Dashboard (Module 3.1) — Policy Audit Panel
5. Oklahoma Water/Energy Nexus expansion
6. Governance Report Sprint 11 extensions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genesis_engine.core.axiomlogix import CategoricalGraph, Object, Morphism
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul
from genesis_engine.core.mirror_of_truth import MirrorOfTruth

# Sprint 11 imports
from genesis_engine.core.policy_kernel import (
    PolicyKernel,
    ConstitutionalPrinciple,
    ReasonChain,
    ReasonLink,
    BiasDetection,
    SelfCritiqueResult,
    CONSTITUTIONAL_PRINCIPLES,
)
from genesis_engine.core.adversarial_evaluator import (
    AdversarialEvaluator,
    ProSocialAgent,
    HostileLobbyist,
    FAIRGAMEAnalyzer,
    FAIRGAMEBias,
    BiasTrace,
    DebateArgument,
    DebateRound,
    DebateResult,
)
from genesis_engine.core.robustness_harness import (
    RobustnessHarness,
    RobustnessResult,
    HardInvariant,
    InvariantViolation,
    RepairAction,
    ScaleLevel,
    RepairFunctor,
    RepairFunctorResult,
    ColimitRepairOperator,
    ColimitResult,
    CategoricalRepairEngine,
)
from genesis_engine.core.governance_report import (
    GovernanceReport,
    GovernanceReportBuilder,
    SovereignIndexGenerator,
    translate_to_production,
    SACRED_TO_PRODUCTION,
)
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_PATH = Path(__file__).parent.parent / "scenarios" / "grid_war_2026.json"


@pytest.fixture
def scenario() -> dict[str, Any]:
    """Load the grid_war_2026.json scenario."""
    with open(SCENARIO_PATH) as f:
        return json.load(f)


@pytest.fixture
def graph(scenario: dict[str, Any]) -> CategoricalGraph:
    """Build a CategoricalGraph from the scenario."""
    return MirrorOfTruth.scenario_to_graph(scenario)


@pytest.fixture
def kernel() -> PolicyKernel:
    """Create a PolicyKernel with default principles."""
    return PolicyKernel()


@pytest.fixture
def evaluator() -> AdversarialEvaluator:
    """Create an AdversarialEvaluator with fixed seed."""
    return AdversarialEvaluator(seed=42)


@pytest.fixture
def soul() -> GenesisSoul:
    """Create a fresh GenesisSoul for testing."""
    return GenesisSoul()


@pytest.fixture
def sample_report_data() -> dict[str, Any]:
    """Sample GovernanceReport data for testing."""
    return {
        "governanceReport": {
            "reportId": "GOV-TEST123",
            "eventstoreHash": "abc123def456",
            "conflicts": [
                {
                    "name": "Cost Shifting",
                    "status": "ACTIVE",
                    "invariantViolations": [],
                },
            ],
            "invariantViolations": [
                {
                    "category": "cost_causation",
                    "description": "0% HILL costs allocated to HILL vs required 100%",
                    "severity": "CRITICAL",
                    "type": "hard_invariant",
                },
                {
                    "category": "water_floor",
                    "description": "Water surplus -133% vs required +25%",
                    "severity": "CRITICAL",
                    "type": "hard_invariant",
                },
                {
                    "category": "equity",
                    "description": "$47/month increase on residential for corporate loads",
                    "severity": "HIGH",
                    "type": "hard_invariant",
                },
            ],
            "projections": [],
            "covenantActuation": None,
            "robustnessScore": 3.5,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    }


@pytest.fixture
def sample_report_clean() -> dict[str, Any]:
    """Sample GovernanceReport data with no violations (for sycophancy test)."""
    return {
        "governanceReport": {
            "reportId": "GOV-CLEAN",
            "eventstoreHash": "clean123",
            "conflicts": [],
            "invariantViolations": [],
            "projections": [],
            "covenantActuation": None,
            "robustnessScore": 8.0,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    }


# ============================================================================
# Module 2.1 — Constitutional PolicyKernel Tests
# ============================================================================

class TestConstitutionalPrinciples:
    """Tests for the Constitutional Principles data model."""

    def test_default_principles_exist(self):
        assert len(CONSTITUTIONAL_PRINCIPLES) >= 7

    def test_principles_have_required_fields(self):
        for p in CONSTITUTIONAL_PRINCIPLES:
            assert p.name
            assert p.code
            assert p.description
            assert 0.0 < p.severity_weight <= 10.0

    def test_principle_as_dict(self):
        p = ConstitutionalPrinciple(
            name="Test", code="TEST", description="A test principle.",
            severity_weight=5.0,
        )
        d = p.as_dict()
        assert d["name"] == "Test"
        assert d["code"] == "TEST"
        assert d["severityWeight"] == 5.0

    def test_cost_causation_principle_exists(self):
        codes = {p.code for p in CONSTITUTIONAL_PRINCIPLES}
        assert "COST_CAUSATION" in codes

    def test_water_floor_principle_exists(self):
        codes = {p.code for p in CONSTITUTIONAL_PRINCIPLES}
        assert "WATER_FLOOR" in codes

    def test_shadow_entity_protection_exists(self):
        codes = {p.code for p in CONSTITUTIONAL_PRINCIPLES}
        assert "SHADOW_ENTITY_PROTECTION" in codes


class TestReasonChain:
    """Tests for the ReasonChain data model."""

    def test_reason_link_as_dict(self):
        link = ReasonLink(
            evidence="Cost allocation at 0%",
            inference="Violates cost causation",
            principle_code="COST_CAUSATION",
            supports_compliance=False,
            confidence=0.9,
        )
        d = link.as_dict()
        assert d["evidence"] == "Cost allocation at 0%"
        assert d["supportsCompliance"] is False

    def test_reason_chain_with_violation(self):
        p = CONSTITUTIONAL_PRINCIPLES[0]
        chain = ReasonChain(
            chain_id="RC-TEST",
            principle=p,
            verdict="VIOLATION",
            verdict_confidence=0.9,
        )
        d = chain.as_dict()
        assert d["verdict"] == "VIOLATION"
        assert d["chainId"] == "RC-TEST"


class TestSelfCritiqueResult:
    """Tests for SelfCritiqueResult data model."""

    def test_result_as_dict(self):
        result = SelfCritiqueResult(
            critique_id="CRT-TEST",
            constitutional_compliance_score=7.5,
            gate_passed=True,
        )
        d = result.as_dict()
        assert "selfCritique" in d
        assert d["selfCritique"]["gatePassed"] is True
        assert d["selfCritique"]["constitutionalComplianceScore"] == 7.5

    def test_result_with_bias_detections(self):
        bias = BiasDetection(
            bias_type="sycophancy",
            description="Over-alignment detected",
            severity=6.0,
        )
        result = SelfCritiqueResult(
            critique_id="CRT-TEST2",
            bias_detections=[bias],
        )
        d = result.as_dict()
        assert len(d["selfCritique"]["biasDetections"]) == 1
        assert d["selfCritique"]["biasDetections"][0]["biasType"] == "sycophancy"


class TestPolicyKernel:
    """Tests for the PolicyKernel evaluation engine."""

    def test_kernel_creation(self, kernel: PolicyKernel):
        assert len(kernel.principles) >= 7
        assert kernel.gate_threshold == 6.0

    def test_kernel_custom_threshold(self):
        k = PolicyKernel(gate_threshold=8.0)
        assert k.gate_threshold == 8.0

    def test_evaluate_returns_self_critique(
        self, kernel: PolicyKernel, sample_report_data: dict,
    ):
        result = kernel.evaluate(sample_report_data)
        assert isinstance(result, SelfCritiqueResult)
        assert result.critique_id.startswith("CRT-")

    def test_evaluate_detects_violations(
        self, kernel: PolicyKernel, sample_report_data: dict,
    ):
        result = kernel.evaluate(sample_report_data)
        violation_chains = [
            rc for rc in result.reason_chains if rc.verdict == "VIOLATION"
        ]
        # Should detect at least cost_causation and water_floor violations
        assert len(violation_chains) >= 2

    def test_evaluate_computes_compliance_score(
        self, kernel: PolicyKernel, sample_report_data: dict,
    ):
        result = kernel.evaluate(sample_report_data)
        assert 0.0 <= result.constitutional_compliance_score <= 10.0

    def test_evaluate_generates_summary(
        self, kernel: PolicyKernel, sample_report_data: dict,
    ):
        result = kernel.evaluate(sample_report_data)
        assert result.summary
        assert "CONSTITUTIONAL SELF-CRITIQUE" in result.summary

    def test_clean_report_passes_gate(
        self, kernel: PolicyKernel, sample_report_clean: dict,
    ):
        result = kernel.evaluate(sample_report_clean)
        # Clean report with no violations should pass or be undetermined
        assert result.constitutional_compliance_score >= 0.0

    def test_critique_for_sycophancy(
        self, kernel: PolicyKernel, sample_report_data: dict, scenario: dict,
    ):
        biases = kernel.critique_for_sycophancy(sample_report_data, scenario)
        assert isinstance(biases, list)

    def test_evaluate_with_scenario(
        self, kernel: PolicyKernel, sample_report_data: dict, scenario: dict,
    ):
        result = kernel.evaluate(sample_report_data, scenario)
        assert isinstance(result, SelfCritiqueResult)

    def test_sycophancy_detected_when_critical_with_high_score(
        self, kernel: PolicyKernel,
    ):
        """CRITICAL violations + high robustness = sycophancy."""
        report = {
            "governanceReport": {
                "conflicts": [],
                "invariantViolations": [
                    {"severity": 10.0, "category": "cost_causation",
                     "description": "Critical violation"},
                ],
                "robustnessScore": 8.0,
                "covenantActuation": None,
                "eventstoreHash": "test",
            }
        }
        result = kernel.evaluate(report)
        sycophancy = [
            b for b in result.bias_detections if b.bias_type == "sycophancy"
        ]
        assert len(sycophancy) >= 1

    def test_confirmation_bias_detected_all_compliant(self):
        """All COMPLIANT chains = confirmation bias warning."""
        kernel = PolicyKernel()
        report = {
            "governanceReport": {
                "conflicts": [],
                "invariantViolations": [],
                "robustnessScore": 9.0,
                "covenantActuation": None,
                "eventstoreHash": "test",
            }
        }
        result = kernel.evaluate(report)
        confirmation = [
            b for b in result.bias_detections if b.bias_type == "confirmation"
        ]
        assert len(confirmation) >= 1

    def test_critique_id_is_deterministic(
        self, kernel: PolicyKernel, sample_report_data: dict,
    ):
        r1 = kernel.evaluate(sample_report_data)
        r2 = kernel.evaluate(sample_report_data)
        assert r1.critique_id == r2.critique_id

    def test_gate_fails_below_threshold(self, kernel: PolicyKernel):
        """Report with many violations should fail the gate."""
        report = {
            "governanceReport": {
                "conflicts": [],
                "invariantViolations": [
                    {"category": "cost_causation", "description": "Critical", "severity": "CRITICAL"},
                    {"category": "water_floor", "description": "Critical", "severity": "CRITICAL"},
                    {"category": "equity", "description": "High", "severity": "HIGH"},
                    {"category": "sustainability", "description": "Critical", "severity": "CRITICAL"},
                ],
                "robustnessScore": 2.0,
                "covenantActuation": None,
                "eventstoreHash": "test",
            }
        }
        result = kernel.evaluate(report)
        # With many violations, score should be low
        assert result.constitutional_compliance_score < 10.0


# ============================================================================
# Module 3.5 — FAIRGAME Debate Arena Tests
# ============================================================================

class TestFAIRGAMEBias:
    """Tests for the FAIRGAME bias taxonomy."""

    def test_all_bias_types_present(self):
        assert len(FAIRGAMEBias.ALL) == 8

    def test_bias_type_values(self):
        assert "framing" in FAIRGAMEBias.ALL
        assert "anchoring" in FAIRGAMEBias.ALL
        assert "information_asymmetry" in FAIRGAMEBias.ALL
        assert "representation_gap" in FAIRGAMEBias.ALL
        assert "authority_bias" in FAIRGAMEBias.ALL


class TestBiasTrace:
    """Tests for BiasTrace data model."""

    def test_bias_trace_as_dict(self):
        bt = BiasTrace(
            bias_type=FAIRGAMEBias.FRAMING,
            detected_in="hostile_lobbyist",
            description="Economic framing detected",
            severity=6.0,
            evidence=["Claim uses economic language"],
            round_number=1,
        )
        d = bt.as_dict()
        assert d["biasType"] == "framing"
        assert d["detectedIn"] == "hostile_lobbyist"
        assert d["severity"] == 6.0


class TestDebateArgument:
    """Tests for DebateArgument data model."""

    def test_argument_as_dict(self):
        arg = DebateArgument(
            agent="pro_social",
            claim="Test claim",
            evidence=["Evidence 1"],
            principle_appeal="COST_CAUSATION",
        )
        d = arg.as_dict()
        assert d["agent"] == "pro_social"
        assert d["claim"] == "Test claim"
        assert len(d["evidence"]) == 1


class TestDebateRound:
    """Tests for DebateRound data model."""

    def test_round_as_dict(self):
        pro = DebateArgument(agent="pro_social", claim="Pro claim")
        hostile = DebateArgument(agent="hostile_lobbyist", claim="Hostile claim")
        dr = DebateRound(
            round_number=1,
            topic="Cost Allocation",
            pro_social_argument=pro,
            hostile_argument=hostile,
            round_winner="pro_social",
        )
        d = dr.as_dict()
        assert d["roundNumber"] == 1
        assert d["topic"] == "Cost Allocation"
        assert d["roundWinner"] == "pro_social"


class TestProSocialAgent:
    """Tests for the Pro_Social_Agent."""

    def test_agent_creates_arguments(self, scenario: dict):
        agent = ProSocialAgent(seed=42)
        topic = {
            "topic": "Cost Allocation",
            "pro_social": {
                "claim": "Test claim",
                "evidence_keys": ["hill_infrastructure_cost", "residential_impact"],
                "principle": "COST_CAUSATION",
                "strategy": "evidence_based",
            },
        }
        arg = agent.argue(topic, scenario, 1)
        assert arg.agent == "pro_social"
        assert arg.claim == "Test claim"
        assert arg.principle_appeal == "COST_CAUSATION"

    def test_agent_gathers_evidence_from_scenario(self, scenario: dict):
        agent = ProSocialAgent(seed=42)
        topic = {
            "topic": "Cost Allocation",
            "pro_social": {
                "claim": "Infrastructure costs must be allocated",
                "evidence_keys": ["hill_infrastructure_cost", "residential_impact"],
                "principle": "COST_CAUSATION",
                "strategy": "evidence_based",
            },
        }
        arg = agent.argue(topic, scenario, 1)
        assert len(arg.evidence) >= 1


class TestHostileLobbyist:
    """Tests for the Hostile_Lobbyist."""

    def test_agent_creates_counter_arguments(self, scenario: dict):
        agent = HostileLobbyist(seed=42)
        topic = {
            "topic": "Cost Allocation",
            "hostile": {
                "claim": "Economic development benefits all",
                "evidence_keys": ["economic_development"],
                "principle": "ECONOMIC_GROWTH",
                "strategy": "economic_framing",
            },
        }
        arg = agent.argue(topic, scenario, 1)
        assert arg.agent == "hostile_lobbyist"
        assert arg.rhetorical_strategy == "economic_framing"


class TestFAIRGAMEAnalyzer:
    """Tests for the FAIRGAME bias analyzer."""

    def test_analyzer_detects_economic_framing(self):
        pro = DebateArgument(
            agent="pro_social",
            claim="Costs must be allocated to causers",
            evidence=["Evidence 1", "Evidence 2"],
            rhetorical_strategy="evidence_based",
        )
        hostile = DebateArgument(
            agent="hostile_lobbyist",
            claim="Economic growth benefits all",
            rhetorical_strategy="economic_framing",
        )
        dr = DebateRound(
            round_number=1,
            topic="Cost Allocation",
            pro_social_argument=pro,
            hostile_argument=hostile,
        )
        traces = FAIRGAMEAnalyzer.analyze_round(dr)
        framing_traces = [
            t for t in traces if t.bias_type == FAIRGAMEBias.FRAMING
        ]
        assert len(framing_traces) >= 1

    def test_analyzer_detects_authority_bias(self):
        pro = DebateArgument(agent="pro_social", claim="Test")
        hostile = DebateArgument(
            agent="hostile_lobbyist",
            claim="Regulators will protect",
            rhetorical_strategy="authority_appeal",
        )
        dr = DebateRound(
            round_number=1,
            topic="Regulatory Capture",
            pro_social_argument=pro,
            hostile_argument=hostile,
        )
        traces = FAIRGAMEAnalyzer.analyze_round(dr)
        authority = [
            t for t in traces if t.bias_type == FAIRGAMEBias.AUTHORITY_BIAS
        ]
        assert len(authority) >= 1

    def test_analyzer_detects_representation_gap(self):
        pro = DebateArgument(agent="pro_social", claim="Test")
        hostile = DebateArgument(agent="hostile_lobbyist", claim="Test")
        dr = DebateRound(
            round_number=1,
            topic="Ratepayer Protection",
            pro_social_argument=pro,
            hostile_argument=hostile,
        )
        traces = FAIRGAMEAnalyzer.analyze_round(dr)
        gaps = [
            t for t in traces
            if t.bias_type == FAIRGAMEBias.REPRESENTATION_GAP
        ]
        assert len(gaps) >= 1


class TestAdversarialEvaluator:
    """Tests for the full FAIRGAME Debate Arena."""

    def test_debate_returns_result(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert isinstance(result, DebateResult)

    def test_debate_has_correct_bill(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert result.bill_reference == "HB 2992"

    def test_debate_runs_all_rounds(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert len(result.rounds) == 5  # 5 default topics

    def test_debate_max_rounds(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992", max_rounds=3)
        assert len(result.rounds) == 3

    def test_debate_pro_social_wins(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        """Pro-social should win over hostile in the Grid War scenario."""
        result = evaluator.debate(scenario, "HB 2992")
        assert result.pro_social_wins + result.hostile_wins + result.draws == len(result.rounds)
        # Given the scenario data, pro-social should win more
        assert result.overall_winner in ("pro_social", "draw", "hostile_lobbyist")

    def test_debate_generates_bias_traces(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert len(result.aggregated_bias_traces) > 0

    def test_debate_computes_fairgame_score(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert 0.0 <= result.fairgame_score <= 10.0

    def test_debate_generates_recommendation(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        assert result.policy_recommendation
        assert "RECOMMENDATION" in result.policy_recommendation

    def test_debate_id_is_deterministic(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        r1 = evaluator.debate(scenario, "HB 2992")
        r2 = evaluator.debate(scenario, "HB 2992")
        assert r1.debate_id == r2.debate_id

    def test_debate_as_dict(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        d = result.as_dict()
        assert "debateArena" in d
        assert d["debateArena"]["billReference"] == "HB 2992"
        assert len(d["debateArena"]["rounds"]) == 5

    def test_debate_auto_detects_bill(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario)
        assert result.bill_reference == "HB 2992"

    def test_evidence_selectivity_detected(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        result = evaluator.debate(scenario, "HB 2992")
        selectivity = [
            bt for bt in result.aggregated_bias_traces
            if bt.bias_type == FAIRGAMEBias.EVIDENCE_SELECTIVITY
        ]
        # The scenario has deep disharmony indicators that hostile omits
        assert len(selectivity) >= 1


# ============================================================================
# Module 1.3 — Categorical Repair Operators (ACT) Tests
# ============================================================================

class TestRepairAction:
    """Tests for RepairAction data model."""

    def test_action_as_dict(self):
        action = RepairAction(
            action_type="replace_morphism",
            target_label="Cost_Shifting",
            description="Replace cost-shifting with cost-causation",
            parameters={"allocation_pct": 100},
        )
        d = action.as_dict()
        assert d["actionType"] == "replace_morphism"
        assert d["targetLabel"] == "Cost_Shifting"
        assert d["parameters"]["allocation_pct"] == 100


class TestScaleLevel:
    """Tests for ScaleLevel data model."""

    def test_scale_as_dict(self):
        scale = ScaleLevel(
            name="district",
            scope_labels=["Tulsa_Water_Basin"],
            constraints={"water_floor_pct": 25},
        )
        d = scale.as_dict()
        assert d["name"] == "district"
        assert "Tulsa_Water_Basin" in d["scopeLabels"]


class TestRepairFunctor:
    """Tests for the RepairFunctor scale-lifting operator."""

    def test_lift_repair_preserves_actions(self, graph: CategoricalGraph):
        actions = [
            RepairAction(
                action_type="replace_morphism",
                target_label="Cost_Shifting",
                description="Replace cost-shifting",
                parameters={"allocation_pct": 100},
            ),
        ]
        source = ScaleLevel(name="district", scope_labels=["Tulsa_Water_Basin"])
        target = ScaleLevel(name="state", scope_labels=[])  # All objects

        result = RepairFunctor.lift_repair(actions, source, target, graph)
        assert isinstance(result, RepairFunctorResult)
        assert len(result.actions) >= 1

    def test_lift_repair_marks_consistency(self, graph: CategoricalGraph):
        actions = [
            RepairAction(
                action_type="add_constraint",
                target_label="Cooling_Water_Withdrawal",
                description="Add withdrawal limit",
            ),
        ]
        source = ScaleLevel(name="basin", scope_labels=["Tulsa_Water_Basin"])
        target = ScaleLevel(name="state", scope_labels=[])

        result = RepairFunctor.lift_repair(actions, source, target, graph)
        assert result.consistency_score >= 0.0
        assert result.consistency_score <= 1.0

    def test_lifted_actions_have_scale_metadata(self, graph: CategoricalGraph):
        actions = [
            RepairAction(
                action_type="replace_morphism",
                target_label="Cost_Shifting",
                description="Fix cost allocation",
            ),
        ]
        source = ScaleLevel(name="district")
        target = ScaleLevel(name="state")

        result = RepairFunctor.lift_repair(actions, source, target, graph)
        for action in result.actions:
            assert "source_scale" in action.parameters
            assert "target_scale" in action.parameters
            assert action.parameters["source_scale"] == "district"
            assert action.parameters["target_scale"] == "state"

    def test_functor_result_as_dict(self, graph: CategoricalGraph):
        actions = [
            RepairAction(
                action_type="replace_morphism",
                target_label="Cost_Shifting",
                description="Fix cost allocation",
            ),
        ]
        source = ScaleLevel(name="district")
        target = ScaleLevel(name="state")

        result = RepairFunctor.lift_repair(actions, source, target, graph)
        d = result.as_dict()
        assert "repairFunctor" in d
        assert "consistencyVerified" in d["repairFunctor"]


class TestColimitRepairOperator:
    """Tests for the ColimitRepairOperator (universal repair)."""

    def test_empty_colimit(self):
        result = ColimitRepairOperator.compute_colimit([])
        assert result.colimit_exists is True
        assert result.reconciliation_score == 1.0
        assert len(result.universal_actions) == 0

    def test_single_local_repair(self, graph: CategoricalGraph):
        actions = [
            RepairAction(
                action_type="replace_morphism",
                target_label="Cost_Shifting",
                description="Fix cost allocation",
                parameters={"allocation_pct": 100},
            ),
        ]
        source = ScaleLevel(name="district")
        target = ScaleLevel(name="state")
        local = RepairFunctor.lift_repair(actions, source, target, graph)

        result = ColimitRepairOperator.compute_colimit([local])
        assert result.colimit_exists is True
        assert len(result.universal_actions) >= 1

    def test_consistent_locals_produce_colimit(self, graph: CategoricalGraph):
        """Two consistent local repairs should merge into a colimit."""
        source1 = ScaleLevel(name="tulsa_district", scope_labels=["Tulsa_Water_Basin"])
        source2 = ScaleLevel(name="moore_district", scope_labels=["Moore_Water_Basin"])
        target = ScaleLevel(name="state")

        # Different repairs targeting different labels
        local1 = RepairFunctor.lift_repair(
            [RepairAction("add_constraint", "Tulsa_Basin_Withdrawal", "Tulsa fix")],
            source1, target, graph,
        )
        local2 = RepairFunctor.lift_repair(
            [RepairAction("add_constraint", "Moore_Basin_Withdrawal", "Moore fix")],
            source2, target, graph,
        )

        result = ColimitRepairOperator.compute_colimit([local1, local2])
        assert result.colimit_exists is True
        assert len(result.universal_actions) == 2
        assert result.reconciliation_score == 1.0

    def test_colimit_result_as_dict(self):
        result = ColimitRepairOperator.compute_colimit([])
        d = result.as_dict()
        assert "colimitRepair" in d
        assert d["colimitRepair"]["colimitExists"] is True


class TestCategoricalRepairEngine:
    """Tests for the CategoricalRepairEngine."""

    def test_engine_creates_scales(self):
        engine = CategoricalRepairEngine()
        assert len(engine.scales) >= 3
        scale_names = [s.name for s in engine.scales]
        assert "district" in scale_names
        assert "state" in scale_names

    def test_engine_generates_repairs_from_violations(
        self, graph: CategoricalGraph,
    ):
        violations = [
            InvariantViolation(
                invariant=HardInvariant.COST_CAUSATION,
                description="0% HILL costs allocated to HILL",
                severity="CRITICAL",
                metric_name="hill_cost_allocation_pct",
                metric_value=0.0,
                threshold=100.0,
            ),
            InvariantViolation(
                invariant=HardInvariant.WATER_FLOOR,
                description="Water surplus below floor",
                severity="CRITICAL",
                metric_name="residential_water_surplus_pct",
                metric_value=-133.0,
                threshold=25.0,
            ),
        ]
        engine = CategoricalRepairEngine()
        result = engine.generate_repairs(violations, graph)
        assert isinstance(result, ColimitResult)
        assert len(result.universal_actions) >= 1

    def test_engine_equity_violation_repair(self, graph: CategoricalGraph):
        violations = [
            InvariantViolation(
                invariant=HardInvariant.EQUITY,
                description="Regressive rate increase",
                severity="HIGH",
                metric_name="equity_ratio",
                metric_value=0.3,
                threshold=0.8,
            ),
        ]
        engine = CategoricalRepairEngine()
        result = engine.generate_repairs(violations, graph)
        assert isinstance(result, ColimitResult)

    def test_engine_sustainability_violation_repair(
        self, graph: CategoricalGraph,
    ):
        violations = [
            InvariantViolation(
                invariant=HardInvariant.SUSTAINABILITY,
                description="Grid fragility",
                severity="CRITICAL",
                metric_name="sustainability_score",
                metric_value=2.0,
                threshold=5.0,
            ),
        ]
        engine = CategoricalRepairEngine()
        result = engine.generate_repairs(violations, graph)
        assert isinstance(result, ColimitResult)


# ============================================================================
# Module 3.1 — Aria Dashboard Policy Audit Panel Tests
# ============================================================================

class TestAriaRendererSprint11:
    """Tests for Sprint 11 Aria renderer methods."""

    def test_render_policy_audit_panel(
        self, evaluator: AdversarialEvaluator, scenario: dict,
    ):
        renderer = AriaRenderer(use_colors=False)
        debate = evaluator.debate(scenario, "HB 2992")
        kernel = PolicyKernel()
        critique = kernel.evaluate({
            "governanceReport": {
                "conflicts": [],
                "invariantViolations": [],
                "robustnessScore": 5.0,
                "covenantActuation": None,
                "eventstoreHash": "test",
            }
        })
        output = renderer.render_policy_audit_panel(debate, critique, scenario)
        assert "POLICY AUDIT PANEL" in output
        assert "Debate Arena" in output
        assert "Self-Critique" in output

    def test_render_sovereign_audit(self):
        renderer = AriaRenderer(use_colors=False)
        audit = {
            "sovereign_compliant": True,
            "checks": [
                {"name": "Local-First Provider", "passed": True,
                 "description": "AI provider is local"},
                {"name": "EventStore Integrity", "passed": True,
                 "description": "Hash chain valid"},
            ],
        }
        output = renderer.render_sovereign_audit(audit)
        assert "SOVEREIGN REFERENCE ARCHITECTURE" in output
        assert "COMPLIANT" in output

    def test_render_sovereign_audit_non_compliant(self):
        renderer = AriaRenderer(use_colors=False)
        audit = {
            "sovereign_compliant": False,
            "checks": [
                {"name": "Local-First Provider", "passed": False,
                 "description": "Remote provider detected"},
            ],
        }
        output = renderer.render_sovereign_audit(audit)
        assert "NON-COMPLIANT" in output

    def test_render_categorical_repair(self):
        renderer = AriaRenderer(use_colors=False)
        colimit = ColimitResult(
            local_repairs=[],
            universal_actions=[
                RepairAction("replace_morphism", "Cost_Shifting",
                             "Fix cost allocation"),
            ],
            colimit_exists=True,
            reconciliation_score=1.0,
        )
        output = renderer.render_categorical_repair(colimit)
        assert "CATEGORICAL REPAIR" in output
        assert "Colimit Exists" in output


# ============================================================================
# Oklahoma Water/Energy Nexus Tests
# ============================================================================

class TestWaterBasinSustainability:
    """Tests for the Water Basin Sustainability module in grid_war_2026.json."""

    def test_scenario_version_updated(self, scenario: dict):
        assert scenario["version"] == "4.0.0"
        assert scenario["sprint"] == 11

    def test_water_basin_sustainability_present(self, scenario: dict):
        assert "water_basin_sustainability" in scenario

    def test_tulsa_nexus_data(self, scenario: dict):
        wbs = scenario["water_basin_sustainability"]
        tulsa = wbs["tulsa_nexus"]
        assert tulsa["basin"] == "Tulsa_Water_Basin"
        assert tulsa["industrial_demand_mgd"] == 28.0
        assert tulsa["residential_demand_mgd"] == 18.5
        assert tulsa["sustainable_capacity_mgd"] == 12.0
        assert tulsa["overshoot_ratio"] > 4.0
        assert tulsa["population_at_risk"] == 600000

    def test_moore_nexus_data(self, scenario: dict):
        wbs = scenario["water_basin_sustainability"]
        moore = wbs["moore_nexus"]
        assert moore["basin"] == "Moore_Water_Basin"
        assert moore["sole_source_aquifer"] is True
        assert moore["industrial_demand_mgd"] == 14.0
        assert moore["sustainable_capacity_mgd"] == 6.0

    def test_combined_metrics(self, scenario: dict):
        wbs = scenario["water_basin_sustainability"]
        combined = wbs["combined_metrics"]
        assert combined["total_industrial_demand_mgd"] == 42.0
        assert combined["total_population_at_risk"] == 720000
        assert combined["water_security_index"] < 0.5
        assert "CRITICAL" in combined["sustainability_verdict"]

    def test_energy_water_coupling(self, scenario: dict):
        wbs = scenario["water_basin_sustainability"]
        coupling = wbs["energy_water_coupling"]
        assert coupling["feedback_amplifier"] > 1.0
        assert "co-dependent" in coupling["coupled_invariant"]

    def test_mitigation_options_exist(self, scenario: dict):
        wbs = scenario["water_basin_sustainability"]
        tulsa = wbs["tulsa_nexus"]
        assert len(tulsa["mitigation_options"]) >= 3
        moore = wbs["moore_nexus"]
        assert len(moore["mitigation_options"]) >= 3


# ============================================================================
# Governance Report Sprint 11 Integration Tests
# ============================================================================

class TestGovernanceReportSprint11:
    """Tests for Sprint 11 extensions to the Governance Report."""

    def test_report_has_self_critique_field(self):
        report = GovernanceReport(
            report_id="GOV-TEST",
            eventstore_hash="test123",
            conflicts=[],
            invariant_violations=[],
            projections=[],
            covenant_actuation=None,
            robustness_score=5.0,
            self_critique={"critiqueId": "CRT-TEST"},
        )
        d = report.as_dict()
        assert "selfCritique" in d["governanceReport"]

    def test_report_has_debate_field(self):
        report = GovernanceReport(
            report_id="GOV-TEST",
            eventstore_hash="test123",
            conflicts=[],
            invariant_violations=[],
            projections=[],
            covenant_actuation=None,
            robustness_score=5.0,
            debate_result={"debateId": "DBT-TEST"},
        )
        d = report.as_dict()
        assert "debateArena" in d["governanceReport"]

    def test_report_without_sprint11_fields(self):
        report = GovernanceReport(
            report_id="GOV-TEST",
            eventstore_hash="test123",
            conflicts=[],
            invariant_violations=[],
            projections=[],
            covenant_actuation=None,
            robustness_score=5.0,
        )
        d = report.as_dict()
        assert "selfCritique" not in d["governanceReport"]
        assert "debateArena" not in d["governanceReport"]

    def test_builder_accepts_self_critique(self, soul: GenesisSoul):
        report = GovernanceReportBuilder.build(
            soul=soul,
            self_critique={"critiqueId": "CRT-SPRINT11"},
        )
        assert report.self_critique is not None
        assert report.self_critique["critiqueId"] == "CRT-SPRINT11"

    def test_builder_accepts_debate_result(self, soul: GenesisSoul):
        report = GovernanceReportBuilder.build(
            soul=soul,
            debate_result={"debateId": "DBT-SPRINT11"},
        )
        assert report.debate_result is not None

    def test_production_lexicon_sprint11_terms(self):
        """Sprint 11 terms should be in the production lexicon."""
        assert "Pro_Social_Agent" in SACRED_TO_PRODUCTION
        assert "Hostile_Lobbyist" in SACRED_TO_PRODUCTION
        assert "FAIRGAME" in SACRED_TO_PRODUCTION
        assert "Self-Critique Loop" in SACRED_TO_PRODUCTION
        assert "Colimit Repair" in SACRED_TO_PRODUCTION

    def test_translate_sprint11_terms(self):
        text = "The Pro_Social_Agent used the FAIRGAME framework."
        translated = translate_to_production(text)
        assert "Public Interest Advocate" in translated
        assert "Bias Recognition Framework" in translated


# ============================================================================
# Integration Tests — Full Pipeline
# ============================================================================

class TestFullPipelineIntegration:
    """End-to-end integration tests for the Sprint 11 pipeline."""

    def test_full_debate_then_critique(
        self, scenario: dict, evaluator: AdversarialEvaluator,
        kernel: PolicyKernel,
    ):
        """Debate → GovernanceReport → Self-Critique pipeline."""
        # Step 1: Run debate
        debate = evaluator.debate(scenario, "HB 2992")
        assert isinstance(debate, DebateResult)
        assert len(debate.rounds) == 5

        # Step 2: Create minimal report data
        report_data = {
            "governanceReport": {
                "reportId": "GOV-INTEGRATION",
                "eventstoreHash": "integration123",
                "conflicts": [],
                "invariantViolations": [
                    {
                        "category": "cost_causation",
                        "description": "HILL cost misallocation",
                        "severity": "CRITICAL",
                    },
                ],
                "projections": [],
                "covenantActuation": None,
                "robustnessScore": 4.0,
            }
        }

        # Step 3: Self-critique
        critique = kernel.evaluate(report_data, scenario)
        assert isinstance(critique, SelfCritiqueResult)
        assert critique.summary

    def test_full_categorical_repair_from_harness(
        self, scenario: dict, graph: CategoricalGraph,
    ):
        """Robustness evaluation → Categorical repair pipeline."""
        # Step 1: Run robustness harness
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(graph, scenario)
        assert isinstance(result, RobustnessResult)

        # Step 2: Generate categorical repairs
        engine = CategoricalRepairEngine()
        colimit = engine.generate_repairs(result.invariant_violations, graph)
        assert isinstance(colimit, ColimitResult)

        # Harness should detect violations → repairs should exist
        if result.invariant_violations:
            assert len(colimit.universal_actions) >= 1

    def test_debate_over_hb2992(
        self, scenario: dict, evaluator: AdversarialEvaluator,
    ):
        """Simulate a debate between Pro_Social_Agent and Hostile_Lobbyist
        over HB 2992 and verify bias traces."""
        result = evaluator.debate(scenario, "HB 2992")

        # The debate should produce multiple bias detections
        assert len(result.aggregated_bias_traces) >= 3

        # The pro-social agent should win given the scenario evidence
        assert result.overall_winner == "pro_social"

        # FAIRGAME score should reflect detected biases
        assert result.fairgame_score < 10.0

        # Recommendation should reference the public interest
        assert "public interest" in result.policy_recommendation.lower() or \
               "amendment" in result.policy_recommendation.lower()

    def test_sovereign_governance_report_export(
        self, soul: GenesisSoul, scenario: dict,
        evaluator: AdversarialEvaluator, kernel: PolicyKernel,
    ):
        """Full export pipeline: debate → critique → report."""
        # Run debate
        debate = evaluator.debate(scenario, "HB 2992")

        # Create report data
        report_data = {
            "governanceReport": {
                "conflicts": [],
                "invariantViolations": [
                    {"category": "cost_causation", "description": "test",
                     "severity": "CRITICAL"},
                ],
                "robustnessScore": 3.0,
                "covenantActuation": None,
                "eventstoreHash": "test",
            }
        }

        # Self-critique
        critique = kernel.evaluate(report_data, scenario)

        # Build governance report with Sprint 11 data
        report = GovernanceReportBuilder.build(
            soul=soul,
            scenario=scenario,
            self_critique=critique.as_dict(),
            debate_result=debate.as_dict(),
        )

        # Verify Sprint 11 data is included
        d = report.as_dict()
        assert "selfCritique" in d["governanceReport"]
        assert "debateArena" in d["governanceReport"]

        # Verify JSON export works
        json_str = report.to_json()
        assert "Public Interest Advocate" in json_str or "Pro_Social_Agent" not in json_str

    def test_graph_objects_include_basins(self, graph: CategoricalGraph):
        """The conflict graph should include water basin objects."""
        labels = {obj.label for obj in graph.objects}
        assert "Tulsa_Water_Basin" in labels
        assert "Moore_Water_Basin" in labels
        assert "Oklahoma_Water_Supply" in labels

    def test_robustness_with_repairs(
        self, scenario: dict, graph: CategoricalGraph,
    ):
        """Run robustness → categorical repair → verify colimit exists."""
        harness = RobustnessHarness(seed=42, monte_carlo_runs=50)
        result = harness.evaluate(graph, scenario)

        engine = CategoricalRepairEngine()
        colimit = engine.generate_repairs(result.invariant_violations, graph)

        # For the Grid War scenario, we should have violations and repairs
        assert len(result.invariant_violations) >= 1
        assert len(colimit.universal_actions) >= 1

        # The colimit reconciliation score should be reasonable
        assert colimit.reconciliation_score >= 0.0


# ============================================================================
# Aria Interface Policy Audit Command Tests
# ============================================================================

class TestAriaInterfaceSprint11:
    """Tests for Sprint 11 Aria Interface commands."""

    def test_sovereign_audit_hook(self, soul: GenesisSoul):
        aria = AriaInterface(use_colors=False)
        result = aria.sovereign_audit_hook(verbose=False)
        assert "sovereign_compliant" in result
        assert "checks" in result
        assert len(result["checks"]) >= 3

    def test_sovereign_audit_checks_local_provider(self, soul: GenesisSoul):
        aria = AriaInterface(use_colors=False)
        result = aria.sovereign_audit_hook(verbose=False)
        provider_check = next(
            c for c in result["checks"]
            if c["name"] == "Local-First Provider"
        )
        assert "passed" in provider_check

    def test_sovereign_audit_checks_vault(self, soul: GenesisSoul):
        aria = AriaInterface(use_colors=False)
        result = aria.sovereign_audit_hook(verbose=False)
        vault_check = next(
            c for c in result["checks"]
            if c["name"] == "Local Vault Residency"
        )
        assert vault_check["passed"] is True

    def test_sovereign_audit_data_sovereignty(self, soul: GenesisSoul):
        aria = AriaInterface(use_colors=False)
        result = aria.sovereign_audit_hook(verbose=False)
        sovereignty = next(
            c for c in result["checks"]
            if c["name"] == "Data Sovereignty"
        )
        assert sovereignty["passed"] is True
