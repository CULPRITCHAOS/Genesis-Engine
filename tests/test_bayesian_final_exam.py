"""Tests for Bayesian Final Exam and Covenant Final Exam (Sprint 8)."""

from genesis_engine.core.game_theory_console import (
    BayesianFinalExam,
    BlackoutShockResult,
    CovenantFinalExam,
    CovenantExamResult,
    FinalExamResult,
)


# ---------------------------------------------------------------------------
# BayesianFinalExam core tests
# ---------------------------------------------------------------------------

class TestBayesianFinalExam:
    def test_returns_blackout_shock_result(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        assert isinstance(result, BlackoutShockResult)

    def test_base_result_present(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        assert isinstance(result.base_result, FinalExamResult)

    def test_bayesian_score_in_range(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        assert 0.0 <= result.bayesian_sustainability_score <= 10.0

    def test_posterior_viability_in_range(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        assert 0.0 <= result.posterior_viability <= 1.0

    def test_blackout_probability_in_range(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        assert 0.0 <= result.blackout_probability <= 1.0

    def test_seed_reproducibility(self):
        exam1 = BayesianFinalExam()
        exam2 = BayesianFinalExam()
        r1 = exam1.administer(seed=42)
        r2 = exam2.administer(seed=42)
        assert r1.bayesian_sustainability_score == r2.bayesian_sustainability_score
        assert r1.posterior_viability == r2.posterior_viability

    def test_fragility_amplifier_stored(self):
        exam = BayesianFinalExam(fragility_amplifier=2.0)
        result = exam.administer(seed=42)
        assert result.fragility_amplifier == 2.0

    def test_prior_viability_stored(self):
        exam = BayesianFinalExam(prior_viability=0.3)
        result = exam.administer(seed=42)
        assert result.prior_viability == 0.3


# ---------------------------------------------------------------------------
# Fragility amplifier effect
# ---------------------------------------------------------------------------

class TestFragilityAmplifier:
    def test_higher_fragility_lowers_score(self):
        low_frag = BayesianFinalExam(fragility_amplifier=1.0)
        high_frag = BayesianFinalExam(fragility_amplifier=3.0)
        r_low = low_frag.administer(seed=42)
        r_high = high_frag.administer(seed=42)
        # Higher fragility should produce a lower or equal score
        assert r_high.bayesian_sustainability_score <= r_low.bayesian_sustainability_score

    def test_very_high_fragility_may_fail(self):
        exam = BayesianFinalExam(
            fragility_amplifier=5.0,
            pass_threshold=7.0,
        )
        result = exam.administer(seed=42)
        # Very high fragility should make it very hard to pass
        assert result.bayesian_sustainability_score < 7.0


# ---------------------------------------------------------------------------
# Prior viability effect
# ---------------------------------------------------------------------------

class TestPriorViability:
    def test_low_prior_reduces_score(self):
        optimistic = BayesianFinalExam(prior_viability=0.9)
        pessimistic = BayesianFinalExam(prior_viability=0.2)
        r_opt = optimistic.administer(seed=42)
        r_pes = pessimistic.administer(seed=42)
        # Pessimistic prior should generally give lower score
        assert r_pes.bayesian_sustainability_score <= r_opt.bayesian_sustainability_score


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestBlackoutShockSerialization:
    def test_as_dict_structure(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        d = result.as_dict()
        assert "blackoutShockExam" in d
        bse = d["blackoutShockExam"]
        assert "passed" in bse
        assert "bayesianSustainabilityScore" in bse
        assert "priorViability" in bse
        assert "posteriorViability" in bse
        assert "fragilityAmplifier" in bse
        assert "blackoutProbability" in bse

    def test_as_dict_types(self):
        exam = BayesianFinalExam()
        result = exam.administer(seed=42)
        d = result.as_dict()["blackoutShockExam"]
        assert isinstance(d["passed"], bool)
        assert isinstance(d["bayesianSustainabilityScore"], float)
        assert isinstance(d["posteriorViability"], float)
        assert isinstance(d["fragilityAmplifier"], float)
        assert isinstance(d["blackoutProbability"], float)


# ---------------------------------------------------------------------------
# Beta CDF approximation
# ---------------------------------------------------------------------------

class TestBetaCDF:
    def test_cdf_at_zero(self):
        assert BayesianFinalExam._beta_cdf(0.0, 2.0, 2.0) == 0.0

    def test_cdf_at_one(self):
        assert BayesianFinalExam._beta_cdf(1.0, 2.0, 2.0) == 1.0

    def test_cdf_at_mean_is_around_half(self):
        # For a symmetric Beta(2,2), the CDF at x=0.5 should be ~0.5
        val = BayesianFinalExam._beta_cdf(0.5, 2.0, 2.0)
        assert 0.4 <= val <= 0.6

    def test_cdf_monotonic(self):
        vals = [
            BayesianFinalExam._beta_cdf(x / 10.0, 3.0, 3.0)
            for x in range(11)
        ]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


# ---------------------------------------------------------------------------
# Pass / fail logic
# ---------------------------------------------------------------------------

class TestBayesianPassFail:
    def test_very_low_threshold_passes(self):
        exam = BayesianFinalExam(
            pass_threshold=0.1,
            fragility_amplifier=1.0,
        )
        result = exam.administer(seed=42)
        assert result.passed is True
        assert result.blocking_reason == ""

    def test_blocking_reason_on_fail(self):
        exam = BayesianFinalExam(
            pass_threshold=10.0,
            fragility_amplifier=2.0,
        )
        result = exam.administer(seed=42)
        assert result.passed is False
        assert "BLOCKED (Bayesian)" in result.blocking_reason
        assert "Unity over Power" in result.blocking_reason


# ---------------------------------------------------------------------------
# CovenantFinalExam tests
# ---------------------------------------------------------------------------

class TestCovenantFinalExam:
    def test_returns_covenant_result(self):
        exam = CovenantFinalExam()
        result = exam.administer(governance_strength=0.5, seed=42)
        assert isinstance(result, CovenantExamResult)

    def test_score_in_range(self):
        exam = CovenantFinalExam()
        result = exam.administer(governance_strength=0.5, seed=42)
        assert 0.0 <= result.sustainability_score <= 10.0

    def test_governance_strength_stored(self):
        exam = CovenantFinalExam()
        result = exam.administer(governance_strength=0.8, seed=42)
        assert result.governance_strength == 0.8

    def test_high_governance_passes(self):
        exam = CovenantFinalExam(pass_threshold=7.5)
        result = exam.administer(governance_strength=0.95, seed=42)
        assert result.passed is True
        assert result.sustainability_score > 7.5

    def test_zero_governance_may_fail(self):
        exam = CovenantFinalExam(pass_threshold=7.5)
        result = exam.administer(governance_strength=0.0, seed=42)
        assert result.sustainability_score < 7.5

    def test_higher_governance_higher_score(self):
        exam = CovenantFinalExam(pass_threshold=7.5)
        r_low = exam.administer(governance_strength=0.2, seed=42)
        r_high = exam.administer(governance_strength=0.9, seed=42)
        assert r_high.sustainability_score > r_low.sustainability_score

    def test_seed_reproducibility(self):
        exam = CovenantFinalExam()
        r1 = exam.administer(governance_strength=0.5, seed=42)
        r2 = exam.administer(governance_strength=0.5, seed=42)
        assert r1.sustainability_score == r2.sustainability_score

    def test_serialisation(self):
        exam = CovenantFinalExam()
        result = exam.administer(governance_strength=0.7, seed=42)
        d = result.as_dict()
        assert "covenantExam" in d
        ce = d["covenantExam"]
        assert "passed" in ce
        assert "sustainabilityScore" in ce
        assert "governanceStrength" in ce
        assert "effectiveDefectionRate" in ce

    def test_blocking_reason_on_fail(self):
        exam = CovenantFinalExam(pass_threshold=10.0)
        result = exam.administer(governance_strength=0.1, seed=42)
        assert result.passed is False
        assert "BLOCKED" in result.blocking_reason
        assert "Unity over Power" in result.blocking_reason


class TestGovernanceStrength:
    def test_perfect_alignment(self):
        gov = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 1.0, "compassion": 1.0, "coherence": 1.0},
            governance_rule_count=5,
            repair_morphism_count=2,
        )
        assert 0.9 <= gov <= 1.0

    def test_empty_alignment(self):
        gov = CovenantFinalExam.compute_governance_strength(
            alignment_scores={},
        )
        assert gov == 0.0

    def test_rules_add_bonus(self):
        base = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 0.8},
        )
        with_rules = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 0.8},
            governance_rule_count=5,
        )
        assert with_rules > base

    def test_repair_morphisms_add_bonus(self):
        base = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 0.8},
        )
        with_repairs = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 0.8},
            repair_morphism_count=2,
        )
        assert with_repairs > base

    def test_capped_at_one(self):
        gov = CovenantFinalExam.compute_governance_strength(
            alignment_scores={"unity": 1.0, "compassion": 1.0, "coherence": 1.0},
            governance_rule_count=100,
            repair_morphism_count=100,
        )
        assert gov <= 1.0
