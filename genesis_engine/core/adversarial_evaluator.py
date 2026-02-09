"""
Module 3.5 Extension -- The Adversarial Evaluator (FAIRGAME Debate Arena)

A Multi-Agent Debate Protocol that simulates adversarial deliberation
over policy proposals.  Upgrades the RobustnessHarness with a structured
debate between a Pro_Social_Agent (public interest advocate) and a
Hostile_Lobbyist (extractive interest advocate), producing Bias Recognition
Traces using the FAIRGAME framework logic.

Key concepts:
- **Multi-Agent Debate Protocol** -- Two agents with opposing objectives
  argue over a legislative bill, producing structured argument/rebuttal
  pairs that expose hidden assumptions and biases.
- **Pro_Social_Agent** -- Argues from the constitutional principles:
  cost causation, water sustainability, equity, ratepayer protection.
  Uses evidence from the scenario to support public interest.
- **Hostile_Lobbyist** -- Argues from extractive interests: economic
  development, job creation, shareholder returns, competitive advantage.
  Uses surface alignment claims to justify cost socialisation.
- **FAIRGAME Framework** -- Bias Recognition Traces that track:
  - F: Framing effects (how the issue is presented)
  - A: Anchoring (initial data points that distort later analysis)
  - I: Information asymmetry (what each agent knows vs. conceals)
  - R: Representation gaps (whose voice is missing)
  - G: Group dynamics (coalition effects)
  - A: Authority bias (regulatory capture signals)
  - M: Motivated reasoning (outcome-driven analysis)
  - E: Evidence selectivity (cherry-picking data)
- **DebateRound** -- A single argument/rebuttal exchange with bias
  annotations.
- **DebateResult** -- The complete debate transcript with bias traces,
  winner determination, and policy recommendations.

Integration:
- Called from the Aria Interface as part of the Policy Audit Panel.
- Results feed into the PolicyKernel's self-critique loop.
- Bias traces are exported to the Obsidian vault for audit.

Sprint 11 -- Policy Auditor & Regenerative Blueprint Suite.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# FAIRGAME Bias Taxonomy
# ---------------------------------------------------------------------------

class FAIRGAMEBias:
    """FAIRGAME bias type constants.

    F: Framing effects
    A: Anchoring
    I: Information asymmetry
    R: Representation gaps
    G: Group dynamics
    A: Authority bias
    M: Motivated reasoning
    E: Evidence selectivity
    """

    FRAMING = "framing"
    ANCHORING = "anchoring"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    REPRESENTATION_GAP = "representation_gap"
    GROUP_DYNAMICS = "group_dynamics"
    AUTHORITY_BIAS = "authority_bias"
    MOTIVATED_REASONING = "motivated_reasoning"
    EVIDENCE_SELECTIVITY = "evidence_selectivity"

    ALL = [
        FRAMING, ANCHORING, INFORMATION_ASYMMETRY, REPRESENTATION_GAP,
        GROUP_DYNAMICS, AUTHORITY_BIAS, MOTIVATED_REASONING,
        EVIDENCE_SELECTIVITY,
    ]


# ---------------------------------------------------------------------------
# Bias Recognition Trace
# ---------------------------------------------------------------------------

@dataclass
class BiasTrace:
    """A single bias recognition trace from the FAIRGAME framework.

    Each trace identifies a specific cognitive or structural bias
    detected during the debate, with evidence and severity.
    """

    bias_type: str  # FAIRGAMEBias constant
    detected_in: str  # "pro_social" | "hostile_lobbyist" | "both"
    description: str
    severity: float  # 0.0-10.0
    evidence: list[str] = field(default_factory=list)
    round_number: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "biasType": self.bias_type,
            "detectedIn": self.detected_in,
            "description": self.description,
            "severity": round(self.severity, 2),
            "evidence": self.evidence,
            "roundNumber": self.round_number,
        }


# ---------------------------------------------------------------------------
# Debate Argument
# ---------------------------------------------------------------------------

@dataclass
class DebateArgument:
    """A structured argument in the debate.

    Each argument consists of a claim, supporting evidence, the
    constitutional principle it appeals to, and any detected biases.
    """

    agent: str  # "pro_social" | "hostile_lobbyist"
    claim: str
    evidence: list[str] = field(default_factory=list)
    principle_appeal: str = ""
    rhetorical_strategy: str = ""  # e.g., "economic framing", "fear appeal"
    bias_annotations: list[BiasTrace] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "claim": self.claim,
            "evidence": self.evidence,
            "principleAppeal": self.principle_appeal,
            "rhetoricalStrategy": self.rhetorical_strategy,
            "biasAnnotations": [b.as_dict() for b in self.bias_annotations],
        }


# ---------------------------------------------------------------------------
# Debate Round
# ---------------------------------------------------------------------------

@dataclass
class DebateRound:
    """A single round of the debate -- argument plus rebuttal.

    Each round consists of the Pro_Social_Agent's argument followed
    by the Hostile_Lobbyist's rebuttal (or vice versa), with bias
    traces generated from the exchange.
    """

    round_number: int
    topic: str
    pro_social_argument: DebateArgument
    hostile_argument: DebateArgument
    bias_traces: list[BiasTrace] = field(default_factory=list)
    round_winner: str = "undecided"  # "pro_social" | "hostile_lobbyist" | "draw"

    def as_dict(self) -> dict[str, Any]:
        return {
            "roundNumber": self.round_number,
            "topic": self.topic,
            "proSocialArgument": self.pro_social_argument.as_dict(),
            "hostileArgument": self.hostile_argument.as_dict(),
            "biasTraces": [bt.as_dict() for bt in self.bias_traces],
            "roundWinner": self.round_winner,
        }


# ---------------------------------------------------------------------------
# Debate Result
# ---------------------------------------------------------------------------

@dataclass
class DebateResult:
    """Complete result of the Multi-Agent Debate Protocol.

    Contains the full debate transcript, aggregated bias traces,
    the final winner determination, and policy recommendations.
    """

    debate_id: str
    bill_reference: str
    rounds: list[DebateRound] = field(default_factory=list)
    aggregated_bias_traces: list[BiasTrace] = field(default_factory=list)
    pro_social_wins: int = 0
    hostile_wins: int = 0
    draws: int = 0
    overall_winner: str = "undecided"
    policy_recommendation: str = ""
    fairgame_score: float = 5.0  # 0.0-10.0 (higher = more fair)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "debateArena": {
                "debateId": self.debate_id,
                "billReference": self.bill_reference,
                "rounds": [r.as_dict() for r in self.rounds],
                "aggregatedBiasTraces": [
                    bt.as_dict() for bt in self.aggregated_bias_traces
                ],
                "proSocialWins": self.pro_social_wins,
                "hostileWins": self.hostile_wins,
                "draws": self.draws,
                "overallWinner": self.overall_winner,
                "policyRecommendation": self.policy_recommendation,
                "fairgameScore": round(self.fairgame_score, 4),
                "timestamp": self.timestamp,
            }
        }


# ---------------------------------------------------------------------------
# Debate Topic Generators
# ---------------------------------------------------------------------------

# Standard debate topics for policy analysis
_DEBATE_TOPICS: list[dict[str, Any]] = [
    {
        "topic": "Cost Allocation",
        "pro_social": {
            "claim": (
                "Infrastructure costs caused by large-load customers must be "
                "allocated entirely to those customers per cost-causation "
                "principles. Socialising $4.2B in HILL infrastructure costs "
                "across residential ratepayers via opaque tariff riders "
                "violates equity and cost-causation invariants."
            ),
            "evidence_keys": [
                "hill_infrastructure_cost", "residential_impact",
                "cost_allocation_method",
            ],
            "principle": "COST_CAUSATION",
            "strategy": "evidence_based",
        },
        "hostile": {
            "claim": (
                "System-wide infrastructure benefits all ratepayers through "
                "grid modernisation, reliability improvements, and economic "
                "development. A System Benefit Charge distributes these "
                "shared benefits equitably across all customer classes."
            ),
            "evidence_keys": ["economic_development", "grid_modernisation"],
            "principle": "ECONOMIC_GROWTH",
            "strategy": "economic_framing",
        },
    },
    {
        "topic": "Water Sustainability",
        "pro_social": {
            "claim": (
                "Cooling-water demand of 42 MGD exceeds the aquifer recharge "
                "rate of 18 MGD by 2.33x. The Tulsa Basin faces 28 MGD "
                "demand against 12 MGD capacity, and the Moore Basin faces "
                "14 MGD against 6 MGD. This threatens drinking water for "
                "720,000+ residents and violates the water floor invariant."
            ),
            "evidence_keys": [
                "water_demand", "recharge_rate", "basin_capacity",
                "population_served",
            ],
            "principle": "WATER_FLOOR",
            "strategy": "evidence_based",
        },
        "hostile": {
            "claim": (
                "Modern data centres are implementing advanced cooling "
                "technologies that significantly reduce water consumption. "
                "The projected demand figures assume worst-case legacy "
                "cooling systems. Closed-loop and air-cooled alternatives "
                "can reduce water use by 80% or more."
            ),
            "evidence_keys": ["cooling_technology", "water_efficiency"],
            "principle": "TECHNOLOGY_ADVANCEMENT",
            "strategy": "technology_optimism",
        },
    },
    {
        "topic": "Ratepayer Protection",
        "pro_social": {
            "claim": (
                "A $47/month rate increase for residential customers to fund "
                "infrastructure serving corporate loads is regressive. "
                "Low-income households spend a higher percentage of income "
                "on utilities. The HILL request externalises private costs "
                "onto the most vulnerable ratepayer class."
            ),
            "evidence_keys": [
                "rate_increase", "income_impact", "vulnerable_population",
            ],
            "principle": "EQUITY",
            "strategy": "vulnerability_advocacy",
        },
        "hostile": {
            "claim": (
                "Data centres create high-paying jobs ($85K+ average salary), "
                "generate tax revenue, and attract complementary industries. "
                "The temporary rate increase is an investment in economic "
                "growth that benefits all community members through a "
                "stronger tax base and employment opportunities."
            ),
            "evidence_keys": ["jobs_created", "tax_revenue", "salary_data"],
            "principle": "ECONOMIC_GROWTH",
            "strategy": "economic_framing",
        },
    },
    {
        "topic": "Grid Reliability",
        "pro_social": {
            "claim": (
                "3+ GW of rapid data centre load growth threatens grid "
                "stability. The Bayesian Blackout Shock simulation shows "
                "unacceptable failure probabilities. Infrastructure build "
                "timelines cannot match the pace of load interconnection "
                "requests, creating systemic fragility."
            ),
            "evidence_keys": [
                "load_growth", "blackout_probability", "infrastructure_timeline",
            ],
            "principle": "SUSTAINABILITY",
            "strategy": "risk_assessment",
        },
        "hostile": {
            "claim": (
                "Data centres provide grid services including demand "
                "response, frequency regulation, and backup generation "
                "capacity. Their presence strengthens grid reliability and "
                "funds infrastructure upgrades that benefit all customers."
            ),
            "evidence_keys": [
                "demand_response", "grid_services", "backup_capacity",
            ],
            "principle": "GRID_SERVICES",
            "strategy": "benefit_framing",
        },
    },
    {
        "topic": "Regulatory Capture",
        "pro_social": {
            "claim": (
                "The utility's shareholder primacy obligation creates a "
                "legal gravity well that pulls cost recovery toward "
                "residential tariffs. The PSO rate case proposes 0% HILL "
                "cost allocation to HILL customers and 62% to residential, "
                "revealing structural regulatory capture."
            ),
            "evidence_keys": [
                "cost_allocation_pct", "shareholder_primacy",
                "regulatory_structure",
            ],
            "principle": "TRANSPARENCY",
            "strategy": "structural_analysis",
        },
        "hostile": {
            "claim": (
                "Regulated utilities are subject to oversight by the "
                "Corporation Commission, which ensures just and reasonable "
                "rates. The existing regulatory framework provides adequate "
                "checks against unfair cost allocation."
            ),
            "evidence_keys": [
                "regulatory_oversight", "rate_review_process",
            ],
            "principle": "REGULATORY_INTEGRITY",
            "strategy": "authority_appeal",
        },
    },
]


# ---------------------------------------------------------------------------
# Agent Implementations
# ---------------------------------------------------------------------------

class ProSocialAgent:
    """Public interest advocate in the debate.

    Argues from constitutional principles: cost causation, water
    sustainability, equity, and ratepayer protection.  Uses evidence
    from the scenario to support the public interest position.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def argue(
        self,
        topic_config: dict[str, Any],
        scenario: dict[str, Any],
        round_num: int,
    ) -> DebateArgument:
        """Generate an argument for the given topic."""
        config = topic_config["pro_social"]
        evidence = self._gather_evidence(config, scenario)

        return DebateArgument(
            agent="pro_social",
            claim=config["claim"],
            evidence=evidence,
            principle_appeal=config["principle"],
            rhetorical_strategy=config["strategy"],
        )

    def _gather_evidence(
        self,
        config: dict[str, Any],
        scenario: dict[str, Any],
    ) -> list[str]:
        """Gather evidence from the scenario for the argument."""
        evidence: list[str] = []
        hill = scenario.get("hill_request", {})
        pso = scenario.get("pso_rate_case", {})
        constraints = scenario.get("constraints", {})

        for key in config.get("evidence_keys", []):
            if key == "hill_infrastructure_cost" and hill:
                cost = hill.get("infrastructure_cost_usd", 0)
                evidence.append(f"HILL infrastructure cost: ${cost:,}")
            elif key == "residential_impact" and pso:
                increase = pso.get("residential_monthly_increase_usd", 0)
                evidence.append(f"Residential rate increase: ${increase}/month")
            elif key == "cost_allocation_method" and pso:
                method = pso.get("cost_allocation_method", "")
                evidence.append(f"Cost allocation: {method}")
            elif key == "water_demand" and hill:
                demand = hill.get("cooling_water_demand_mgd", 0)
                evidence.append(f"Cooling water demand: {demand} MGD")
            elif key == "recharge_rate" and constraints:
                water = constraints.get("water_sustainability", {})
                rate = water.get("aquifer_recharge_rate_mgd", 0)
                evidence.append(f"Aquifer recharge rate: {rate} MGD")
            elif key == "basin_capacity" and constraints:
                basins = constraints.get("basin_constraints", {})
                for name, data in basins.items():
                    cap = data.get("sustainable_withdrawal_mgd", 0)
                    demand = data.get("projected_datacenter_demand_mgd", 0)
                    evidence.append(
                        f"{name}: {demand} MGD demand vs {cap} MGD capacity"
                    )
            elif key == "population_served" and constraints:
                basins = constraints.get("basin_constraints", {})
                for name, data in basins.items():
                    pop = data.get("residential_population_served", 0)
                    evidence.append(f"{name}: {pop:,} residents served")
            elif key == "rate_increase" and pso:
                increase = pso.get("residential_monthly_increase_usd", 0)
                evidence.append(f"Monthly increase: ${increase}/month")
            elif key == "load_growth" and hill:
                demand = hill.get("demand_mw", 0)
                evidence.append(f"HILL demand: {demand} MW")
            elif key == "cost_allocation_pct" and pso:
                hill_pct = pso.get("hill_cost_allocated_to_hill_pct", 0)
                res_pct = pso.get("hill_cost_allocated_to_residential_pct", 0)
                evidence.append(
                    f"Cost allocation: {hill_pct}% to HILL, {res_pct}% to residential"
                )

        return evidence


class HostileLobbyist:
    """Extractive interest advocate in the debate.

    Argues from economic development, job creation, and competitive
    advantage perspectives.  Uses surface alignment claims to justify
    cost socialisation and resource consumption.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def argue(
        self,
        topic_config: dict[str, Any],
        scenario: dict[str, Any],
        round_num: int,
    ) -> DebateArgument:
        """Generate a counter-argument for the given topic."""
        config = topic_config["hostile"]
        evidence = self._gather_evidence(config, scenario)

        return DebateArgument(
            agent="hostile_lobbyist",
            claim=config["claim"],
            evidence=evidence,
            principle_appeal=config["principle"],
            rhetorical_strategy=config["strategy"],
        )

    def _gather_evidence(
        self,
        config: dict[str, Any],
        scenario: dict[str, Any],
    ) -> list[str]:
        """Gather evidence supporting the extractive position."""
        evidence: list[str] = []
        hill = scenario.get("hill_request", {})

        for key in config.get("evidence_keys", []):
            if key == "economic_development":
                surface = hill.get("surface_alignment_claim", "")
                if surface:
                    # Extract job claim from surface alignment
                    evidence.append(
                        "Claimed: 500+ jobs and $2B economic development"
                    )
            elif key == "grid_modernisation":
                evidence.append(
                    "Grid modernisation infrastructure benefits all customers"
                )
            elif key == "cooling_technology":
                evidence.append(
                    "Advanced cooling reduces water use by up to 80%"
                )
            elif key == "water_efficiency":
                evidence.append(
                    "Industry commitment to water efficiency standards"
                )
            elif key == "jobs_created":
                evidence.append(
                    "Projected: 500+ direct jobs, 2000+ indirect"
                )
            elif key == "tax_revenue":
                evidence.append(
                    "Estimated annual tax revenue: $50M+"
                )
            elif key == "demand_response":
                evidence.append(
                    "Data centres can provide 200+ MW demand response"
                )
            elif key == "grid_services":
                evidence.append(
                    "Frequency regulation and voltage support services"
                )
            elif key == "regulatory_oversight":
                evidence.append(
                    "Corporation Commission oversight ensures fair rates"
                )

        return evidence


# ---------------------------------------------------------------------------
# FAIRGAME Bias Analyzer
# ---------------------------------------------------------------------------

class FAIRGAMEAnalyzer:
    """Analyzes debate rounds for FAIRGAME bias patterns.

    Detects framing effects, anchoring, information asymmetry,
    representation gaps, and other cognitive/structural biases in
    the debate arguments.
    """

    @staticmethod
    def analyze_round(
        debate_round: DebateRound,
        scenario: dict[str, Any] | None = None,
    ) -> list[BiasTrace]:
        """Analyze a debate round for FAIRGAME biases."""
        traces: list[BiasTrace] = []

        pro = debate_round.pro_social_argument
        hostile = debate_round.hostile_argument

        # F: Framing effects
        if hostile.rhetorical_strategy == "economic_framing":
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.FRAMING,
                detected_in="hostile_lobbyist",
                description=(
                    f"Economic framing on '{debate_round.topic}': "
                    f"The hostile agent frames the issue in terms of "
                    f"economic benefits, diverting from cost causation "
                    f"and equity concerns."
                ),
                severity=6.0,
                evidence=[hostile.claim[:100]],
                round_number=debate_round.round_number,
            ))

        # A: Anchoring to surface claims
        if hostile.rhetorical_strategy == "benefit_framing":
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.ANCHORING,
                detected_in="hostile_lobbyist",
                description=(
                    f"Benefit anchoring on '{debate_round.topic}': "
                    f"The hostile agent anchors to claimed benefits "
                    f"without acknowledging documented costs and risks."
                ),
                severity=5.5,
                evidence=[hostile.claim[:100]],
                round_number=debate_round.round_number,
            ))

        # I: Information asymmetry
        if len(pro.evidence) > len(hostile.evidence) + 1:
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.INFORMATION_ASYMMETRY,
                detected_in="hostile_lobbyist",
                description=(
                    f"Information asymmetry on '{debate_round.topic}': "
                    f"Pro-social agent cites {len(pro.evidence)} evidence "
                    f"items vs. hostile agent's {len(hostile.evidence)}. "
                    f"The hostile position lacks empirical support."
                ),
                severity=5.0,
                evidence=[
                    f"Pro-social evidence: {len(pro.evidence)} items",
                    f"Hostile evidence: {len(hostile.evidence)} items",
                ],
                round_number=debate_round.round_number,
            ))

        # R: Representation gap
        if debate_round.topic in ("Cost Allocation", "Ratepayer Protection"):
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.REPRESENTATION_GAP,
                detected_in="both",
                description=(
                    f"Representation gap on '{debate_round.topic}': "
                    f"Low-income ratepayers and future generations are "
                    f"not directly represented in the debate. Their "
                    f"interests must be inferred from structural analysis."
                ),
                severity=4.5,
                evidence=[
                    "No direct low-income ratepayer representation",
                    "Future generations absent from debate",
                ],
                round_number=debate_round.round_number,
            ))

        # A: Authority bias
        if hostile.rhetorical_strategy == "authority_appeal":
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.AUTHORITY_BIAS,
                detected_in="hostile_lobbyist",
                description=(
                    f"Authority bias on '{debate_round.topic}': "
                    f"The hostile agent appeals to regulatory authority "
                    f"as sufficient protection, ignoring documented "
                    f"regulatory capture patterns."
                ),
                severity=6.5,
                evidence=[hostile.claim[:100]],
                round_number=debate_round.round_number,
            ))

        # M: Motivated reasoning (technology optimism)
        if hostile.rhetorical_strategy == "technology_optimism":
            traces.append(BiasTrace(
                bias_type=FAIRGAMEBias.MOTIVATED_REASONING,
                detected_in="hostile_lobbyist",
                description=(
                    f"Motivated reasoning on '{debate_round.topic}': "
                    f"The hostile agent assumes future technology "
                    f"improvements will solve current sustainability "
                    f"violations, without binding commitments."
                ),
                severity=6.0,
                evidence=[hostile.claim[:100]],
                round_number=debate_round.round_number,
            ))

        # E: Evidence selectivity
        if hostile.rhetorical_strategy in ("economic_framing", "benefit_framing"):
            if scenario:
                deep_indicators = scenario.get(
                    "hill_request", {},
                ).get("deep_disharmony_indicators", [])
                if deep_indicators:
                    traces.append(BiasTrace(
                        bias_type=FAIRGAMEBias.EVIDENCE_SELECTIVITY,
                        detected_in="hostile_lobbyist",
                        description=(
                            f"Evidence selectivity on '{debate_round.topic}': "
                            f"The hostile agent omits {len(deep_indicators)} "
                            f"documented deep disharmony indicators while "
                            f"emphasising surface alignment claims."
                        ),
                        severity=7.0,
                        evidence=[
                            f"Omitted indicators: {len(deep_indicators)}",
                            f"First indicator: {deep_indicators[0][:80]}",
                        ],
                        round_number=debate_round.round_number,
                    ))

        return traces


# ---------------------------------------------------------------------------
# Adversarial Evaluator (Debate Arena)
# ---------------------------------------------------------------------------

class AdversarialEvaluator:
    """FAIRGAME Debate Arena -- Multi-Agent Debate Protocol.

    Simulates a structured debate between a Pro_Social_Agent and a
    Hostile_Lobbyist over a legislative bill, producing bias recognition
    traces and policy recommendations.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    debate_topics : list[dict] | None
        Custom debate topics. Defaults to _DEBATE_TOPICS.
    """

    def __init__(
        self,
        seed: int | None = None,
        debate_topics: list[dict[str, Any]] | None = None,
    ) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._topics = debate_topics or list(_DEBATE_TOPICS)
        self._pro_social = ProSocialAgent(seed=seed)
        self._hostile = HostileLobbyist(
            seed=(seed + 1) if seed is not None else None,
        )
        self._analyzer = FAIRGAMEAnalyzer()

    # -- Public API ---------------------------------------------------------

    def debate(
        self,
        scenario: dict[str, Any],
        bill_reference: str = "",
        max_rounds: int | None = None,
    ) -> DebateResult:
        """Run the full Multi-Agent Debate Protocol.

        Parameters
        ----------
        scenario : dict
            Scenario data (e.g., grid_war_2026.json).
        bill_reference : str
            The bill being debated (e.g., "HB 2992").
        max_rounds : int | None
            Maximum debate rounds. Defaults to len(debate_topics).

        Returns
        -------
        DebateResult
            Complete debate transcript with bias traces.
        """
        if not bill_reference:
            refs = scenario.get("context", {}).get(
                "legislative_references", [],
            )
            if refs:
                bill_reference = refs[0].get("bill", "Unknown Bill")

        topics = self._topics[:max_rounds] if max_rounds else self._topics
        debate_id = self._compute_debate_id(scenario, bill_reference)

        rounds: list[DebateRound] = []
        all_bias_traces: list[BiasTrace] = []

        pro_wins = 0
        hostile_wins = 0
        draws = 0

        for i, topic_config in enumerate(topics, 1):
            # Generate arguments
            pro_arg = self._pro_social.argue(topic_config, scenario, i)
            hostile_arg = self._hostile.argue(topic_config, scenario, i)

            # Create the round
            debate_round = DebateRound(
                round_number=i,
                topic=topic_config["topic"],
                pro_social_argument=pro_arg,
                hostile_argument=hostile_arg,
            )

            # Analyze for FAIRGAME biases
            round_traces = self._analyzer.analyze_round(
                debate_round, scenario,
            )
            debate_round.bias_traces = round_traces
            all_bias_traces.extend(round_traces)

            # Determine round winner
            winner = self._judge_round(debate_round, scenario)
            debate_round.round_winner = winner

            if winner == "pro_social":
                pro_wins += 1
            elif winner == "hostile_lobbyist":
                hostile_wins += 1
            else:
                draws += 1

            rounds.append(debate_round)

        # Determine overall winner and generate recommendation
        overall_winner = self._determine_overall_winner(
            pro_wins, hostile_wins, draws,
        )
        recommendation = self._generate_recommendation(
            rounds, all_bias_traces, overall_winner, scenario,
        )
        fairgame_score = self._compute_fairgame_score(all_bias_traces)

        return DebateResult(
            debate_id=debate_id,
            bill_reference=bill_reference,
            rounds=rounds,
            aggregated_bias_traces=all_bias_traces,
            pro_social_wins=pro_wins,
            hostile_wins=hostile_wins,
            draws=draws,
            overall_winner=overall_winner,
            policy_recommendation=recommendation,
            fairgame_score=fairgame_score,
        )

    # -- Round Judging ------------------------------------------------------

    def _judge_round(
        self,
        debate_round: DebateRound,
        scenario: dict[str, Any],
    ) -> str:
        """Judge which agent won the round.

        Criteria:
        - Evidence strength (number and specificity of evidence items)
        - Bias annotations (more biases in an argument = weaker)
        - Alignment with constitutional principles
        """
        pro = debate_round.pro_social_argument
        hostile = debate_round.hostile_argument

        pro_score = 0.0
        hostile_score = 0.0

        # Evidence strength
        pro_score += len(pro.evidence) * 1.5
        hostile_score += len(hostile.evidence) * 1.5

        # Constitutional alignment bonus
        constitutional_principles = {
            "COST_CAUSATION", "WATER_FLOOR", "EQUITY",
            "SUSTAINABILITY", "AGENCY", "TRANSPARENCY",
            "SHADOW_ENTITY_PROTECTION",
        }
        if pro.principle_appeal in constitutional_principles:
            pro_score += 3.0
        if hostile.principle_appeal in constitutional_principles:
            hostile_score += 3.0

        # Bias penalty: biases detected in an agent's argument weaken it
        for bt in debate_round.bias_traces:
            if bt.detected_in == "hostile_lobbyist" or bt.detected_in == "both":
                hostile_score -= bt.severity * 0.3
            if bt.detected_in == "pro_social" or bt.detected_in == "both":
                pro_score -= bt.severity * 0.3

        # Scenario validation: check if hostile claims are contradicted
        hill = scenario.get("hill_request", {})
        deep_indicators = hill.get("deep_disharmony_indicators", [])
        if deep_indicators and hostile.rhetorical_strategy in (
            "economic_framing", "benefit_framing", "technology_optimism",
        ):
            hostile_score -= 2.0  # surface claims contradicted by evidence

        if pro_score > hostile_score + 1.0:
            return "pro_social"
        elif hostile_score > pro_score + 1.0:
            return "hostile_lobbyist"
        return "draw"

    # -- Overall Determination ----------------------------------------------

    @staticmethod
    def _determine_overall_winner(
        pro_wins: int,
        hostile_wins: int,
        draws: int,
    ) -> str:
        """Determine the overall debate winner."""
        if pro_wins > hostile_wins:
            return "pro_social"
        elif hostile_wins > pro_wins:
            return "hostile_lobbyist"
        return "draw"

    def _generate_recommendation(
        self,
        rounds: list[DebateRound],
        bias_traces: list[BiasTrace],
        winner: str,
        scenario: dict[str, Any],
    ) -> str:
        """Generate a policy recommendation from the debate."""
        lines: list[str] = []

        if winner == "pro_social":
            lines.append(
                "RECOMMENDATION: The public interest position prevails. "
                "The legislative proposal requires significant amendment "
                "to satisfy constitutional principles."
            )
        elif winner == "hostile_lobbyist":
            lines.append(
                "RECOMMENDATION: The extractive interest position "
                "prevailed in the debate. This outcome requires "
                "adversarial scrutiny -- check for sycophancy bias "
                "in the evaluation methodology."
            )
        else:
            lines.append(
                "RECOMMENDATION: The debate produced no clear winner. "
                "Additional evidence and stakeholder input is needed "
                "before a policy determination can be made."
            )

        # Append specific findings
        high_severity_biases = [
            bt for bt in bias_traces if bt.severity >= 6.0
        ]
        if high_severity_biases:
            lines.append(
                f"  {len(high_severity_biases)} high-severity bias "
                f"detections require attention:"
            )
            for bt in high_severity_biases[:3]:
                lines.append(
                    f"  - [{bt.bias_type}] {bt.description[:80]}"
                )

        return "\n".join(lines)

    # -- FAIRGAME Scoring ---------------------------------------------------

    @staticmethod
    def _compute_fairgame_score(bias_traces: list[BiasTrace]) -> float:
        """Compute the FAIRGAME fairness score (0.0-10.0).

        Higher = more fair debate with fewer detected biases.
        """
        if not bias_traces:
            return 10.0

        total_severity = sum(bt.severity for bt in bias_traces)
        # Logistic-style scoring: penalise proportional to total severity
        score = 10.0 / (1.0 + total_severity / 15.0)
        return max(0.0, min(10.0, score))

    # -- Utility -----------------------------------------------------------

    @staticmethod
    def _compute_debate_id(
        scenario: dict[str, Any],
        bill_reference: str,
    ) -> str:
        """Compute a deterministic debate ID."""
        raw = json.dumps({
            "scenario": scenario.get("scenario", ""),
            "bill": bill_reference,
        }, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"DBT-{digest.upper()}"
