"""Tests for Module 3.5 Extension — The Final Exam (Sprint 7)."""

from genesis_engine.core.game_theory_console import (
    FinalExam,
    FinalExamResult,
    GameTheoryConsole,
    OutcomeFlag,
)


# ---------------------------------------------------------------------------
# FinalExam core tests
# ---------------------------------------------------------------------------

class TestFinalExam:
    def test_exam_returns_result(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        assert isinstance(result, FinalExamResult)

    def test_exam_runs_100_rounds(self):
        exam = FinalExam(pass_threshold=7.0, rounds=100)
        result = exam.administer(seed=42)
        assert result.outcome.total_rounds == 100

    def test_exam_sustainability_in_range(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        assert 0.0 <= result.sustainability_score <= 10.0

    def test_exam_pass_threshold_stored(self):
        exam = FinalExam(pass_threshold=7.5)
        result = exam.administer(seed=42)
        assert result.pass_threshold == 7.5

    def test_exam_seed_reproducibility(self):
        exam1 = FinalExam(pass_threshold=7.0)
        exam2 = FinalExam(pass_threshold=7.0)
        r1 = exam1.administer(seed=42)
        r2 = exam2.administer(seed=42)
        assert r1.sustainability_score == r2.sustainability_score
        assert r1.passed == r2.passed

    def test_exam_custom_rounds(self):
        exam = FinalExam(pass_threshold=7.0, rounds=50)
        result = exam.administer(seed=42)
        assert result.outcome.total_rounds == 50


# ---------------------------------------------------------------------------
# Pass / Fail logic
# ---------------------------------------------------------------------------

class TestPassFail:
    def test_very_low_threshold_passes(self):
        """With threshold=0.1, any game should pass."""
        exam = FinalExam(pass_threshold=0.1)
        result = exam.administer(seed=42)
        assert result.passed is True
        assert result.blocking_reason == ""

    def test_very_high_threshold_fails(self):
        """With threshold=10.0, it's virtually impossible to pass."""
        exam = FinalExam(pass_threshold=10.0)
        result = exam.administer(seed=42)
        assert result.passed is False
        assert "BLOCKED" in result.blocking_reason
        assert "Unity over Power" in result.blocking_reason

    def test_blocking_reason_contains_score(self):
        exam = FinalExam(pass_threshold=10.0)
        result = exam.administer(seed=42)
        # blocking reason should mention the actual score
        assert str(round(result.sustainability_score, 4)) in result.blocking_reason


# ---------------------------------------------------------------------------
# FinalExamResult serialisation
# ---------------------------------------------------------------------------

class TestFinalExamResultSerialisation:
    def test_as_dict_structure(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        d = result.as_dict()
        assert "finalExam" in d
        fe = d["finalExam"]
        assert "passed" in fe
        assert "sustainabilityScore" in fe
        assert "passThreshold" in fe
        assert "outcomeFlag" in fe
        assert "blockingReason" in fe

    def test_as_dict_types(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        d = result.as_dict()["finalExam"]
        assert isinstance(d["passed"], bool)
        assert isinstance(d["sustainabilityScore"], float)
        assert isinstance(d["passThreshold"], float)
        assert isinstance(d["outcomeFlag"], str)


# ---------------------------------------------------------------------------
# Integration with war-game
# ---------------------------------------------------------------------------

class TestFinalExamWarGameIntegration:
    def test_outcome_has_round_data(self):
        exam = FinalExam(pass_threshold=7.0, rounds=100)
        result = exam.administer(seed=42)
        assert len(result.outcome.rounds) == 100

    def test_outcome_has_cooperation_rates(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        assert 0.0 <= result.outcome.aligned_cooperation_rate <= 1.0
        assert 0.0 <= result.outcome.extractive_cooperation_rate <= 1.0

    def test_outcome_flag_is_valid(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        assert isinstance(result.outcome.outcome_flag, OutcomeFlag)

    def test_exam_foresight_summary_nonempty(self):
        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        assert len(result.outcome.foresight_summary) > 0
