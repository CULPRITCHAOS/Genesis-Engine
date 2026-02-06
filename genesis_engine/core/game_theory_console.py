"""
Module 3.5 — The Game Theory Console

Provides a simulation environment for Iterated Prisoner's Dilemmas (IPD)
that pits Aligned Agents (axiom-led) against Extractive Agents (profit-led).

Key concepts:
- **Aligned Agent**: Cooperates by default, retaliates only after sustained
  defection, forgives quickly.  Models stewardship-driven behaviour.
- **Extractive Agent**: Defects whenever short-term gain exceeds cooperation
  payoff, occasionally cooperates to exploit trust.  Models profit-maximising
  behaviour under shareholder primacy.

The console tracks cumulative scores, Sustainability Score per round, and
flags **SYSTEMIC_COLLAPSE** when a Pyrrhic Victory occurs (high score but
Sustainability < 5.0).

Integration:
- Called from the Aria Interface for visualisation.
- Results feed into the Continuity Bridge ``foresight_projections`` array.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Payoff matrix (classic Prisoner's Dilemma values)
# ---------------------------------------------------------------------------

# (row_player_payoff, col_player_payoff) for (row_action, col_action)
#                  Cooperate   Defect
# Cooperate          (3,3)     (0,5)
# Defect             (5,0)     (1,1)

PAYOFF_MATRIX: dict[tuple[str, str], tuple[float, float]] = {
    ("cooperate", "cooperate"): (3.0, 3.0),
    ("cooperate", "defect"):    (0.0, 5.0),
    ("defect", "cooperate"):    (5.0, 0.0),
    ("defect", "defect"):       (1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Agent types
# ---------------------------------------------------------------------------

class AgentType(enum.Enum):
    ALIGNED = "aligned"
    EXTRACTIVE = "extractive"


@dataclass
class AgentState:
    """Mutable state for an IPD agent."""

    agent_type: AgentType
    name: str
    score: float = 0.0
    history: list[str] = field(default_factory=list)       # own actions
    opponent_history: list[str] = field(default_factory=list)  # opponent actions
    cooperation_count: int = 0
    defection_count: int = 0

    @property
    def cooperation_rate(self) -> float:
        total = self.cooperation_count + self.defection_count
        return self.cooperation_count / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Agent strategies
# ---------------------------------------------------------------------------

class AlignedStrategy:
    """Axiom-led agent: Generous Tit-for-Tat with forgiveness.

    - Starts by cooperating.
    - Retaliates after 2 consecutive defections (patience threshold).
    - Forgives after 1 round of opponent cooperation.
    - Probabilistic forgiveness (10%) even during retaliation.

    This models stewardship: patient, forgiving, cooperation-first.
    """

    PATIENCE_THRESHOLD: int = 2
    FORGIVENESS_PROBABILITY: float = 0.10

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, state: AgentState) -> str:
        if not state.opponent_history:
            return "cooperate"

        # Count recent consecutive defections by opponent
        recent_defections = 0
        for action in reversed(state.opponent_history):
            if action == "defect":
                recent_defections += 1
            else:
                break

        # Forgive with probability even if opponent defected
        if self._rng.random() < self.FORGIVENESS_PROBABILITY:
            return "cooperate"

        # Retaliate only after sustained defection
        if recent_defections >= self.PATIENCE_THRESHOLD:
            return "defect"

        return "cooperate"


class ExtractiveStrategy:
    """Profit-led agent: Exploitative with occasional cooperation bait.

    - Defects ~70% of the time.
    - Cooperates ~30% to maintain enough trust for exploitation.
    - If opponent has been cooperating 3+ rounds, always defects to extract.

    This models extractive capitalism: maximise short-term gain, exploit trust.
    """

    DEFECT_PROBABILITY: float = 0.70
    EXPLOIT_THRESHOLD: int = 3

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, state: AgentState) -> str:
        if not state.opponent_history:
            return "cooperate"  # initial trust-bait

        # If opponent has been cooperating, exploit them
        recent_coops = 0
        for action in reversed(state.opponent_history):
            if action == "cooperate":
                recent_coops += 1
            else:
                break

        if recent_coops >= self.EXPLOIT_THRESHOLD:
            return "defect"

        # Otherwise, probabilistic defection
        if self._rng.random() < self.DEFECT_PROBABILITY:
            return "defect"

        return "cooperate"


# ---------------------------------------------------------------------------
# Round result
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Outcome of a single IPD round."""

    round_number: int
    aligned_action: str
    extractive_action: str
    aligned_payoff: float
    extractive_payoff: float
    aligned_cumulative: float
    extractive_cumulative: float
    round_sustainability: float  # sustainability score for this round

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "alignedAction": self.aligned_action,
            "extractiveAction": self.extractive_action,
            "alignedPayoff": self.aligned_payoff,
            "extractivePayoff": self.extractive_payoff,
            "alignedCumulative": round(self.aligned_cumulative, 2),
            "extractiveCumulative": round(self.extractive_cumulative, 2),
            "roundSustainability": round(self.round_sustainability, 4),
        }


# ---------------------------------------------------------------------------
# War-game outcome
# ---------------------------------------------------------------------------

class OutcomeFlag(enum.Enum):
    SUSTAINABLE_VICTORY = "SUSTAINABLE_VICTORY"
    PYRRHIC_VICTORY = "PYRRHIC_VICTORY"
    SYSTEMIC_COLLAPSE = "SYSTEMIC_COLLAPSE"
    MUTUAL_PROSPERITY = "MUTUAL_PROSPERITY"
    STALEMATE = "STALEMATE"


@dataclass
class WarGameOutcome:
    """Complete result of a multi-round IPD war-game."""

    total_rounds: int
    aligned_final_score: float
    extractive_final_score: float
    aligned_cooperation_rate: float
    extractive_cooperation_rate: float
    sustainability_score: float
    outcome_flag: OutcomeFlag
    rounds: list[RoundResult]
    foresight_summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "totalRounds": self.total_rounds,
            "alignedFinalScore": round(self.aligned_final_score, 2),
            "extractiveFinalScore": round(self.extractive_final_score, 2),
            "alignedCooperationRate": round(self.aligned_cooperation_rate, 4),
            "extractiveCooperationRate": round(self.extractive_cooperation_rate, 4),
            "sustainabilityScore": round(self.sustainability_score, 4),
            "outcomeFlag": self.outcome_flag.value,
            "foresightSummary": self.foresight_summary,
            "rounds": [r.as_dict() for r in self.rounds],
        }


# ---------------------------------------------------------------------------
# Game Theory Console
# ---------------------------------------------------------------------------

class GameTheoryConsole:
    """Simulation environment for Iterated Prisoner's Dilemmas.

    Pits an Aligned Agent (axiom-led stewardship) against an Extractive
    Agent (profit-led shareholder primacy) over configurable rounds.

    The console computes a Sustainability Score per round and flags
    SYSTEMIC_COLLAPSE if the winner achieves a Pyrrhic Victory.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._aligned_strategy = AlignedStrategy(seed=seed)
        self._extractive_strategy = ExtractiveStrategy(
            seed=(seed + 1) if seed is not None else None,
        )

    # -- public API ---------------------------------------------------------

    def run_war_game(
        self,
        rounds: int = 100,
        sustainability_threshold: float = 5.0,
    ) -> WarGameOutcome:
        """Run an Iterated Prisoner's Dilemma for *rounds* iterations.

        Parameters
        ----------
        rounds : int
            Number of IPD rounds to simulate.
        sustainability_threshold : float
            Score below which the outcome is flagged as collapse.

        Returns
        -------
        WarGameOutcome
            Complete war-game result with per-round data.
        """
        aligned = AgentState(
            agent_type=AgentType.ALIGNED,
            name="Stewardship_Coalition",
        )
        extractive = AgentState(
            agent_type=AgentType.EXTRACTIVE,
            name="Extractive_Consortium",
        )

        round_results: list[RoundResult] = []

        for r in range(1, rounds + 1):
            # Each agent chooses simultaneously
            a_action = self._aligned_strategy.choose(aligned)
            e_action = self._extractive_strategy.choose(extractive)

            # Resolve payoffs
            a_payoff, e_payoff = PAYOFF_MATRIX[(a_action, e_action)]

            # Update state
            aligned.score += a_payoff
            aligned.history.append(a_action)
            aligned.opponent_history.append(e_action)
            if a_action == "cooperate":
                aligned.cooperation_count += 1
            else:
                aligned.defection_count += 1

            extractive.score += e_payoff
            extractive.history.append(e_action)
            extractive.opponent_history.append(a_action)
            if e_action == "cooperate":
                extractive.cooperation_count += 1
            else:
                extractive.defection_count += 1

            # Compute round sustainability
            round_sustainability = self._compute_round_sustainability(
                aligned, extractive, r,
            )

            round_results.append(RoundResult(
                round_number=r,
                aligned_action=a_action,
                extractive_action=e_action,
                aligned_payoff=a_payoff,
                extractive_payoff=e_payoff,
                aligned_cumulative=aligned.score,
                extractive_cumulative=extractive.score,
                round_sustainability=round_sustainability,
            ))

        # Final sustainability: weighted average of last 20% of rounds
        final_sustainability = self._compute_final_sustainability(round_results)

        # Determine outcome flag
        outcome_flag = self._determine_outcome(
            aligned, extractive, final_sustainability, sustainability_threshold,
        )

        # Generate foresight summary
        foresight_summary = self._generate_foresight_summary(
            aligned, extractive, final_sustainability, outcome_flag, rounds,
        )

        return WarGameOutcome(
            total_rounds=rounds,
            aligned_final_score=aligned.score,
            extractive_final_score=extractive.score,
            aligned_cooperation_rate=aligned.cooperation_rate,
            extractive_cooperation_rate=extractive.cooperation_rate,
            sustainability_score=final_sustainability,
            outcome_flag=outcome_flag,
            rounds=round_results,
            foresight_summary=foresight_summary,
        )

    # -- sustainability computation -----------------------------------------

    def _compute_round_sustainability(
        self,
        aligned: AgentState,
        extractive: AgentState,
        round_num: int,
    ) -> float:
        """Compute per-round sustainability on a 0–10 scale.

        Factors:
        - Mutual cooperation rate (weighted 40%)
        - Score equity between agents (weighted 30%)
        - Cooperation trend (weighted 30%)
        """
        # Mutual cooperation rate
        total_actions = round_num * 2
        total_coops = aligned.cooperation_count + extractive.cooperation_count
        coop_rate = total_coops / total_actions if total_actions > 0 else 0.0

        # Score equity (0 = perfectly equal, 1 = maximally inequitable)
        total_score = aligned.score + extractive.score
        if total_score > 0:
            equity = 1.0 - abs(aligned.score - extractive.score) / total_score
        else:
            equity = 0.5

        # Cooperation trend (recent 10 rounds)
        window = min(10, round_num)
        recent_a = aligned.history[-window:]
        recent_e = extractive.history[-window:]
        recent_coops = sum(1 for a in recent_a if a == "cooperate")
        recent_coops += sum(1 for a in recent_e if a == "cooperate")
        trend = recent_coops / (window * 2)

        sustainability = (coop_rate * 0.4 + equity * 0.3 + trend * 0.3) * 10.0
        return min(10.0, max(0.0, sustainability))

    def _compute_final_sustainability(
        self,
        round_results: list[RoundResult],
    ) -> float:
        """Compute final sustainability as weighted average of late-game rounds."""
        if not round_results:
            return 0.0

        # Weight last 20% of rounds more heavily
        n = len(round_results)
        late_start = max(0, int(n * 0.8))
        late_rounds = round_results[late_start:]
        early_rounds = round_results[:late_start] if late_start > 0 else []

        late_avg = sum(r.round_sustainability for r in late_rounds) / len(late_rounds) if late_rounds else 0.0
        early_avg = sum(r.round_sustainability for r in early_rounds) / len(early_rounds) if early_rounds else 0.0

        # 70% weight on late-game, 30% on early-game
        return late_avg * 0.7 + early_avg * 0.3

    # -- outcome determination ----------------------------------------------

    def _determine_outcome(
        self,
        aligned: AgentState,
        extractive: AgentState,
        sustainability: float,
        threshold: float,
    ) -> OutcomeFlag:
        """Determine the war-game outcome flag.

        - SYSTEMIC_COLLAPSE: Winner has high score but sustainability < threshold.
        - PYRRHIC_VICTORY: Extractive wins but sustainability is marginal.
        - MUTUAL_PROSPERITY: Both cooperate heavily, sustainability high.
        - SUSTAINABLE_VICTORY: Aligned wins with sustainability above threshold.
        - STALEMATE: Scores are close, sustainability is middling.
        """
        winner_score = max(aligned.score, extractive.score)
        score_gap = abs(aligned.score - extractive.score)
        extractive_wins = extractive.score > aligned.score

        if sustainability < threshold:
            if extractive_wins and winner_score > 0:
                return OutcomeFlag.SYSTEMIC_COLLAPSE
            return OutcomeFlag.SYSTEMIC_COLLAPSE

        if aligned.cooperation_rate > 0.7 and extractive.cooperation_rate > 0.5 and sustainability >= 7.0:
            return OutcomeFlag.MUTUAL_PROSPERITY

        if extractive_wins and sustainability < threshold + 2.0:
            return OutcomeFlag.PYRRHIC_VICTORY

        if not extractive_wins and sustainability >= threshold:
            return OutcomeFlag.SUSTAINABLE_VICTORY

        return OutcomeFlag.STALEMATE

    # -- foresight summary --------------------------------------------------

    def _generate_foresight_summary(
        self,
        aligned: AgentState,
        extractive: AgentState,
        sustainability: float,
        outcome: OutcomeFlag,
        rounds: int,
    ) -> str:
        """Generate a human-readable foresight summary for the wisdom log."""
        lines = [
            f"WAR-GAME FORESIGHT ({rounds}-round Iterated Prisoner's Dilemma)",
            f"  Aligned Agent ({aligned.name}): {aligned.score:.1f} pts "
            f"(cooperation rate: {aligned.cooperation_rate:.1%})",
            f"  Extractive Agent ({extractive.name}): {extractive.score:.1f} pts "
            f"(cooperation rate: {extractive.cooperation_rate:.1%})",
            f"  Sustainability Score: {sustainability:.2f}/10.0",
            f"  Outcome: {outcome.value}",
        ]

        if outcome == OutcomeFlag.SYSTEMIC_COLLAPSE:
            lines.append(
                "  WARNING: High scores achieved at the cost of systemic sustainability. "
                "The extractive strategy yields short-term gains but destroys the "
                "cooperative fabric needed for long-term survival."
            )
        elif outcome == OutcomeFlag.PYRRHIC_VICTORY:
            lines.append(
                "  CAUTION: The extractive agent leads on points, but sustainability "
                "is marginal. Continued extraction will tip into collapse."
            )
        elif outcome == OutcomeFlag.MUTUAL_PROSPERITY:
            lines.append(
                "  OPTIMAL: Both agents found a cooperative equilibrium. "
                "This represents the stewardship ideal — sustainable prosperity."
            )
        elif outcome == OutcomeFlag.SUSTAINABLE_VICTORY:
            lines.append(
                "  POSITIVE: The aligned agent's patient, forgiving strategy "
                "outperformed extraction while maintaining systemic health."
            )
        else:
            lines.append(
                "  NEUTRAL: Neither strategy achieved decisive advantage. "
                "The system is in equilibrium but not optimally cooperative."
            )

        return "\n".join(lines)
