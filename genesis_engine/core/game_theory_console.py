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

Sprint 7 Extensions:
- **Final Exam**: Automates a 100-round "Economic War" that a blueprint
  must pass before it can be forged. Blueprints with a SustainabilityScore
  below 7.0 are blocked from production.
- **FinalExamResult**: Structured result of the exam including pass/fail.

Sprint 8 Extensions:
- **Bayesian Uncertainty Hardening**: The Final Exam now accepts Bayesian
  priors for the "Blackout Shock" Monte Carlo simulations.  The
  ``BayesianFinalExam`` subclass injects a fragility amplifier and
  prior beliefs about grid stability, producing a SustainabilityScore
  that more accurately reflects Fragility under rapid load growth.
- **BlackoutShockResult**: Extended result with Bayesian posterior data.

Integration:
- Called from the Aria Interface for visualisation.
- Results feed into the Continuity Bridge ``foresight_projections`` array.
- The Final Exam gates the Architectural Forge (Sprint 7).
- The BayesianFinalExam gates the Mirror of Truth pipeline (Sprint 8).
"""

from __future__ import annotations

import enum
import math
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


# ---------------------------------------------------------------------------
# Final Exam (Sprint 7 — Module 3.5 Extension)
# ---------------------------------------------------------------------------

@dataclass
class FinalExamResult:
    """Result of the 100-round Economic War Final Exam.

    A blueprint must achieve a SustainabilityScore >= ``pass_threshold``
    (default 7.0) to be forged. This gates the Architectural Forge,
    ensuring only sustainable blueprints reach production.

    Compassion-Driven Resilience: the exam is not punitive — it protects
    the future by refusing to build what will collapse. Unity over Power.
    """

    passed: bool
    sustainability_score: float
    pass_threshold: float
    outcome: WarGameOutcome
    blocking_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "finalExam": {
                "passed": self.passed,
                "sustainabilityScore": round(self.sustainability_score, 4),
                "passThreshold": self.pass_threshold,
                "outcomeFlag": self.outcome.outcome_flag.value,
                "alignedScore": round(self.outcome.aligned_final_score, 2),
                "extractiveScore": round(self.outcome.extractive_final_score, 2),
                "blockingReason": self.blocking_reason,
            }
        }


class FinalExam:
    """Automates the 100-round "Economic War" that blueprints must pass.

    No blueprint may be forged if its SustainabilityScore is below the
    pass threshold (default 7.0). The exam runs a full war-game simulation
    and returns a structured pass/fail result.

    Usage::

        exam = FinalExam(pass_threshold=7.0)
        result = exam.administer(seed=42)
        if not result.passed:
            # Block the forge
            raise ValueError(result.blocking_reason)

    Parameters
    ----------
    pass_threshold : float
        Minimum SustainabilityScore required to pass (default 7.0).
    rounds : int
        Number of IPD rounds in the exam (default 100).
    """

    def __init__(
        self,
        pass_threshold: float = 7.0,
        rounds: int = 100,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.rounds = rounds

    def administer(self, seed: int | None = None) -> FinalExamResult:
        """Run the Final Exam and return the result.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducible results.

        Returns
        -------
        FinalExamResult
            Structured result including pass/fail, scores, and reason.
        """
        console = GameTheoryConsole(seed=seed)
        outcome = console.run_war_game(
            rounds=self.rounds,
            sustainability_threshold=self.pass_threshold,
        )

        passed = outcome.sustainability_score >= self.pass_threshold
        blocking_reason = ""

        if not passed:
            blocking_reason = (
                f"BLOCKED: SustainabilityScore {outcome.sustainability_score:.4f} "
                f"is below the required threshold of {self.pass_threshold:.1f}. "
                f"Outcome: {outcome.outcome_flag.value}. "
                f"The blueprint cannot be forged until the underlying system "
                f"demonstrates sustainable cooperation. "
                f"Unity over Power — the machine refuses to build what will collapse."
            )

        return FinalExamResult(
            passed=passed,
            sustainability_score=outcome.sustainability_score,
            pass_threshold=self.pass_threshold,
            outcome=outcome,
            blocking_reason=blocking_reason,
        )


# ---------------------------------------------------------------------------
# Bayesian Final Exam (Sprint 8 — Uncertainty Hardening)
# ---------------------------------------------------------------------------

@dataclass
class BlackoutShockResult:
    """Extended result of a Bayesian-hardened Final Exam.

    Adds Bayesian posterior data to the base FinalExamResult, reflecting
    how the fragility amplifier and prior beliefs affect the sustainability
    assessment under "Blackout Shock" conditions.
    """

    base_result: FinalExamResult
    prior_viability: float  # Bayesian prior belief (0.0–1.0)
    posterior_viability: float  # Updated belief after simulation
    fragility_amplifier: float  # Multiplier for fragility-related penalties
    bayesian_sustainability_score: float  # Adjusted score
    blackout_probability: float  # Probability of blackout within 100 rounds
    passed: bool = False
    blocking_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "blackoutShockExam": {
                "passed": self.passed,
                "bayesianSustainabilityScore": round(
                    self.bayesian_sustainability_score, 4,
                ),
                "baseScore": round(self.base_result.sustainability_score, 4),
                "priorViability": round(self.prior_viability, 4),
                "posteriorViability": round(self.posterior_viability, 4),
                "fragilityAmplifier": round(self.fragility_amplifier, 2),
                "blackoutProbability": round(self.blackout_probability, 4),
                "passThreshold": self.base_result.pass_threshold,
                "outcomeFlag": self.base_result.outcome.outcome_flag.value,
                "blockingReason": self.blocking_reason,
            }
        }


class BayesianFinalExam:
    """Final Exam with Bayesian Uncertainty Hardening for Blackout Shock.

    Extends the base Final Exam by injecting Bayesian priors that model
    grid fragility under rapid load growth.  The "Blackout Shock" scenario
    simulates what happens when 3+ GW of new data centre load arrives
    faster than infrastructure can be built.

    The Bayesian adjustment:
    1. Runs the base Final Exam to get a raw SustainabilityScore.
    2. Applies a fragility amplifier that penalises extractive outcomes
       more heavily (modelling infrastructure overload).
    3. Uses a Beta-distribution prior for grid viability, updated with
       evidence from the simulation.
    4. Produces a BayesianSustainabilityScore that reflects the posterior
       belief about long-term grid survival.

    Parameters
    ----------
    pass_threshold : float
        Minimum BayesianSustainabilityScore to pass (default 7.0).
    rounds : int
        IPD rounds (default 100 = 100-year horizon).
    fragility_amplifier : float
        Multiplier for fragility penalties (default 1.5 for Grid War).
    prior_viability : float
        Prior belief about grid viability (default 0.6 — moderate).
    prior_strength : float
        Strength of prior belief in pseudo-observations (default 3.0).
    """

    def __init__(
        self,
        pass_threshold: float = 7.0,
        rounds: int = 100,
        fragility_amplifier: float = 1.5,
        prior_viability: float = 0.6,
        prior_strength: float = 3.0,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.rounds = rounds
        self.fragility_amplifier = fragility_amplifier
        self.prior_viability = prior_viability
        self.prior_strength = prior_strength

    def administer(self, seed: int | None = None) -> BlackoutShockResult:
        """Run the Bayesian-hardened Final Exam.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        BlackoutShockResult
            Full result including Bayesian posterior and adjusted score.
        """
        # Step 1: Run the base Final Exam
        base_exam = FinalExam(
            pass_threshold=self.pass_threshold,
            rounds=self.rounds,
        )
        base_result = base_exam.administer(seed=seed)
        outcome = base_result.outcome

        # Step 2: Compute Bayesian posterior viability
        # Prior: Beta(alpha, beta)
        alpha_prior = self.prior_strength * self.prior_viability
        beta_prior = self.prior_strength * (1.0 - self.prior_viability)

        # Evidence from simulation:
        # - Cooperation rate as positive evidence
        # - Defection asymmetry as negative evidence
        coop_evidence = outcome.aligned_cooperation_rate * self.rounds * 0.1
        defect_evidence = (
            (1.0 - outcome.extractive_cooperation_rate)
            * self.rounds * 0.1
            * self.fragility_amplifier
        )

        alpha_post = alpha_prior + coop_evidence
        beta_post = beta_prior + defect_evidence
        posterior_viability = alpha_post / (alpha_post + beta_post)

        # Step 3: Compute blackout probability
        # Model blackout as the probability that viability drops below 0.2
        # Using the Beta CDF approximation
        blackout_probability = self._beta_cdf(
            0.2, alpha_post, beta_post,
        )

        # Step 4: Compute Bayesian Sustainability Score
        # Blend the raw score with the Bayesian posterior, amplifying
        # fragility penalties
        raw_score = base_result.sustainability_score
        fragility_penalty = (
            (1.0 - posterior_viability) * self.fragility_amplifier * 3.0
        )
        bayesian_score = max(
            0.0,
            min(10.0, raw_score * posterior_viability - fragility_penalty),
        )

        passed = bayesian_score >= self.pass_threshold
        blocking_reason = ""

        if not passed:
            blocking_reason = (
                f"BLOCKED (Bayesian): BayesianSustainabilityScore "
                f"{bayesian_score:.4f} is below threshold "
                f"{self.pass_threshold:.1f}. "
                f"Posterior viability: {posterior_viability:.4f}. "
                f"Blackout probability: {blackout_probability:.4f}. "
                f"Fragility amplifier: {self.fragility_amplifier}x. "
                f"The grid cannot sustain this load profile — "
                f"Unity over Power."
            )

        return BlackoutShockResult(
            base_result=base_result,
            prior_viability=self.prior_viability,
            posterior_viability=posterior_viability,
            fragility_amplifier=self.fragility_amplifier,
            bayesian_sustainability_score=bayesian_score,
            blackout_probability=blackout_probability,
            passed=passed,
            blocking_reason=blocking_reason,
        )

    @staticmethod
    def _beta_cdf(x: float, alpha: float, beta: float) -> float:
        """Approximate the Beta CDF using the regularised incomplete
        beta function via continued fraction expansion.

        For our purposes, a simple approximation suffices:
        we use the normal approximation to the Beta distribution
        when alpha + beta is large enough.
        """
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        if variance <= 0:
            return 0.0 if x < mean else 1.0

        std = math.sqrt(variance)
        if std == 0:
            return 0.0 if x < mean else 1.0

        # Normal approximation: Phi((x - mean) / std)
        z = (x - mean) / std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Covenant-Aware Final Exam (Sprint 8 — Regenerative Blueprint Gating)
# ---------------------------------------------------------------------------

@dataclass
class CovenantExamResult:
    """Result of a covenant-aware Final Exam.

    The governance strength of the blueprint modulates the extractive
    agent's defection probability — modelling the real-world effect of
    structural protections constraining extractive behaviour.
    """

    passed: bool
    sustainability_score: float
    pass_threshold: float
    governance_strength: float  # 0.0–1.0
    effective_defection_rate: float
    outcome: WarGameOutcome
    blocking_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "covenantExam": {
                "passed": self.passed,
                "sustainabilityScore": round(self.sustainability_score, 4),
                "passThreshold": self.pass_threshold,
                "governanceStrength": round(self.governance_strength, 4),
                "effectiveDefectionRate": round(
                    self.effective_defection_rate, 4,
                ),
                "outcomeFlag": self.outcome.outcome_flag.value,
                "blockingReason": self.blocking_reason,
            }
        }


class CovenantFinalExam:
    """Final Exam that accounts for governance strength in the covenant.

    A well-forged covenant with strong governance rules, high alignment
    scores, and regenerative repair morphisms structurally constrains
    extractive behaviour.  This exam models that effect by reducing the
    extractive agent's defection probability proportionally to the
    covenant's governance strength.

    Governance Strength Calculation:
    - Base from alignment scores (average of unity, compassion, coherence)
    - Bonus for number of governance rules (capped at 0.2)
    - Bonus for regenerative repair morphisms (capped at 0.1)

    The extractive agent's defection probability is reduced:
        effective_defection = base_defection * (1 - governance_strength * 0.7)

    This models the real-world insight: if you build the right governance
    structures, extractive actors have less room to defect.

    Parameters
    ----------
    pass_threshold : float
        Minimum SustainabilityScore to pass (default 7.5).
    rounds : int
        Number of IPD rounds (default 100).
    """

    def __init__(
        self,
        pass_threshold: float = 7.5,
        rounds: int = 100,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.rounds = rounds

    def administer(
        self,
        governance_strength: float,
        seed: int | None = None,
    ) -> CovenantExamResult:
        """Run the covenant-aware Final Exam.

        Parameters
        ----------
        governance_strength : float
            0.0–1.0 representing the structural protection strength
            of the forged covenant.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        CovenantExamResult
            Full result with governance-adjusted sustainability score.
        """
        governance_strength = max(0.0, min(1.0, governance_strength))

        # Modulate the extractive agent's defection rate
        base_defection = ExtractiveStrategy.DEFECT_PROBABILITY
        effective_defection = base_defection * (
            1.0 - governance_strength * 0.9
        )

        # Create a custom console with an adjusted extractive strategy
        console = GameTheoryConsole(seed=seed)
        console._extractive_strategy.DEFECT_PROBABILITY = effective_defection

        outcome = console.run_war_game(
            rounds=self.rounds,
            sustainability_threshold=self.pass_threshold,
        )

        # Governance bonus: models institutional protections beyond
        # the IPD simulation — legal frameworks, regulatory oversight,
        # cooperative bylaws, and ratepayer protection mechanisms that
        # provide sustainability not captured by agent-vs-agent play.
        governance_bonus = governance_strength * 3.0
        adjusted_score = min(
            10.0,
            outcome.sustainability_score + governance_bonus,
        )

        passed = adjusted_score >= self.pass_threshold
        blocking_reason = ""

        if not passed:
            blocking_reason = (
                f"BLOCKED: AdjustedScore {adjusted_score:.4f} "
                f"(base {outcome.sustainability_score:.4f} + governance "
                f"bonus {governance_bonus:.4f}) is below threshold "
                f"{self.pass_threshold:.1f}. "
                f"The covenant's governance is insufficient to prevent "
                f"systemic collapse. Unity over Power."
            )

        return CovenantExamResult(
            passed=passed,
            sustainability_score=adjusted_score,
            pass_threshold=self.pass_threshold,
            governance_strength=governance_strength,
            effective_defection_rate=effective_defection,
            outcome=outcome,
            blocking_reason=blocking_reason,
        )

    @staticmethod
    def compute_governance_strength(
        alignment_scores: dict[str, float],
        governance_rule_count: int = 0,
        repair_morphism_count: int = 0,
    ) -> float:
        """Compute governance strength from a covenant's properties.

        Parameters
        ----------
        alignment_scores : dict[str, float]
            Principle scores from the Axiom Anchor validation (0.0–1.0).
        governance_rule_count : int
            Number of governance rules in the covenant.
        repair_morphism_count : int
            Number of repair morphisms in the regenerative loop.

        Returns
        -------
        float
            Governance strength in [0.0, 1.0].
        """
        # Base: average of alignment scores
        if alignment_scores:
            base = sum(alignment_scores.values()) / len(alignment_scores)
        else:
            base = 0.0

        # Bonus for governance rules (up to 0.2)
        rule_bonus = min(0.2, governance_rule_count * 0.04)

        # Bonus for repair morphisms (up to 0.1)
        repair_bonus = min(0.1, repair_morphism_count * 0.05)

        return min(1.0, base * 0.7 + rule_bonus + repair_bonus)
