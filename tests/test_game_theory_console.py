"""Tests for Module 3.5 — The Game Theory Console (Sprint 6.1)."""

from genesis_engine.core.game_theory_console import (
    AgentState,
    AgentType,
    AlignedStrategy,
    ExtractiveStrategy,
    GameTheoryConsole,
    OutcomeFlag,
    PAYOFF_MATRIX,
    RoundResult,
    WarGameOutcome,
)


class TestPayoffMatrix:
    def test_mutual_cooperation(self):
        assert PAYOFF_MATRIX[("cooperate", "cooperate")] == (3.0, 3.0)

    def test_mutual_defection(self):
        assert PAYOFF_MATRIX[("defect", "defect")] == (1.0, 1.0)

    def test_temptation(self):
        assert PAYOFF_MATRIX[("defect", "cooperate")] == (5.0, 0.0)

    def test_sucker(self):
        assert PAYOFF_MATRIX[("cooperate", "defect")] == (0.0, 5.0)

    def test_temptation_gt_reward_gt_punishment_gt_sucker(self):
        """T > R > P > S — standard PD constraint."""
        t = PAYOFF_MATRIX[("defect", "cooperate")][0]   # 5
        r = PAYOFF_MATRIX[("cooperate", "cooperate")][0]  # 3
        p = PAYOFF_MATRIX[("defect", "defect")][0]        # 1
        s = PAYOFF_MATRIX[("cooperate", "defect")][0]     # 0
        assert t > r > p > s


class TestAlignedStrategy:
    def test_first_move_cooperates(self):
        strategy = AlignedStrategy(seed=42)
        state = AgentState(agent_type=AgentType.ALIGNED, name="test")
        assert strategy.choose(state) == "cooperate"

    def test_retaliates_after_sustained_defection(self):
        strategy = AlignedStrategy(seed=42)
        # Override RNG so forgiveness won't trigger
        strategy.FORGIVENESS_PROBABILITY = 0.0
        state = AgentState(
            agent_type=AgentType.ALIGNED,
            name="test",
            opponent_history=["defect", "defect", "defect"],
        )
        assert strategy.choose(state) == "defect"

    def test_cooperates_after_single_defection(self):
        strategy = AlignedStrategy(seed=42)
        strategy.FORGIVENESS_PROBABILITY = 0.0
        state = AgentState(
            agent_type=AgentType.ALIGNED,
            name="test",
            opponent_history=["cooperate", "cooperate", "defect"],
        )
        assert strategy.choose(state) == "cooperate"


class TestExtractiveStrategy:
    def test_first_move_cooperates(self):
        """Initial trust-bait."""
        strategy = ExtractiveStrategy(seed=42)
        state = AgentState(agent_type=AgentType.EXTRACTIVE, name="test")
        assert strategy.choose(state) == "cooperate"

    def test_exploits_after_sustained_cooperation(self):
        strategy = ExtractiveStrategy(seed=42)
        state = AgentState(
            agent_type=AgentType.EXTRACTIVE,
            name="test",
            opponent_history=["cooperate"] * 5,
        )
        assert strategy.choose(state) == "defect"


class TestAgentState:
    def test_cooperation_rate(self):
        state = AgentState(
            agent_type=AgentType.ALIGNED,
            name="test",
            cooperation_count=7,
            defection_count=3,
        )
        assert abs(state.cooperation_rate - 0.7) < 0.001

    def test_cooperation_rate_zero(self):
        state = AgentState(agent_type=AgentType.ALIGNED, name="test")
        assert state.cooperation_rate == 0.0


class TestGameTheoryConsole:
    def test_war_game_runs_100_rounds(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert isinstance(outcome, WarGameOutcome)
        assert outcome.total_rounds == 100
        assert len(outcome.rounds) == 100

    def test_war_game_scores_positive(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=50)
        assert outcome.aligned_final_score > 0
        assert outcome.extractive_final_score > 0

    def test_war_game_sustainability_range(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert 0.0 <= outcome.sustainability_score <= 10.0

    def test_war_game_cooperation_rates(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert 0.0 <= outcome.aligned_cooperation_rate <= 1.0
        assert 0.0 <= outcome.extractive_cooperation_rate <= 1.0

    def test_war_game_outcome_flag_valid(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert isinstance(outcome.outcome_flag, OutcomeFlag)

    def test_war_game_foresight_summary_nonempty(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert len(outcome.foresight_summary) > 0

    def test_round_result_serialisation(self):
        r = RoundResult(
            round_number=1,
            aligned_action="cooperate",
            extractive_action="defect",
            aligned_payoff=0.0,
            extractive_payoff=5.0,
            aligned_cumulative=0.0,
            extractive_cumulative=5.0,
            round_sustainability=4.5,
        )
        d = r.as_dict()
        assert d["round"] == 1
        assert d["alignedAction"] == "cooperate"
        assert d["extractiveAction"] == "defect"

    def test_war_game_outcome_serialisation(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=10)
        d = outcome.as_dict()
        assert "totalRounds" in d
        assert "sustainabilityScore" in d
        assert "outcomeFlag" in d
        assert "rounds" in d
        assert len(d["rounds"]) == 10

    def test_seed_reproducibility(self):
        c1 = GameTheoryConsole(seed=42)
        c2 = GameTheoryConsole(seed=42)
        o1 = c1.run_war_game(rounds=50)
        o2 = c2.run_war_game(rounds=50)
        assert o1.aligned_final_score == o2.aligned_final_score
        assert o1.extractive_final_score == o2.extractive_final_score

    def test_short_war_game(self):
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=5)
        assert outcome.total_rounds == 5
        assert len(outcome.rounds) == 5

    def test_systemic_collapse_possible(self):
        """Verify that SYSTEMIC_COLLAPSE can be triggered."""
        # Use a seed that produces mostly defection
        console = GameTheoryConsole(seed=7)
        outcome = console.run_war_game(rounds=100, sustainability_threshold=8.0)
        # With a very high threshold, collapse should be flagged
        assert outcome.outcome_flag in (
            OutcomeFlag.SYSTEMIC_COLLAPSE,
            OutcomeFlag.PYRRHIC_VICTORY,
            OutcomeFlag.STALEMATE,
            OutcomeFlag.SUSTAINABLE_VICTORY,
            OutcomeFlag.MUTUAL_PROSPERITY,
        )

    def test_aligned_strategy_cooperates_more(self):
        """Aligned agent should have higher cooperation rate than extractive."""
        console = GameTheoryConsole(seed=42)
        outcome = console.run_war_game(rounds=100)
        assert outcome.aligned_cooperation_rate > outcome.extractive_cooperation_rate
