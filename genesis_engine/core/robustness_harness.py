"""
Module 3.5 Extension — The Robustness Harness

Bayesian stress-testing harness for Regenerative Covenants.  Extends the
Game Theory Console with:

- **Beta-Distribution Priors** — for "Blackout Shock" and "Drought Event"
  Monte Carlo simulations.  Each simulation draws event probabilities from
  ``Beta(alpha, beta)`` priors, updating posteriors with evidence from the
  conflict graph.
- **Decentralized Fork Operator** — when a Hostile Agent (Extractive Data
  Center) refuses a Regenerative Repair (e.g., switching to treated
  wastewater), the Harness automatically triggers a ``DecentralizedForkOperator``
  that protects the local basin by forking the covenant into a protected
  sub-graph that excludes the hostile node.
- **Water Floor Invariant** — enforces that Residential_Water_Security
  maintains a 25% surplus over Industrial_Cooling_Demand across 50-year
  Monte Carlo horizons.
- **Cost Causation Invariant** — enforces that 100% of infrastructure-only
  costs for HILL customers are allocated to the Hyperscale_Node per
  HB 2992 intent.

Integration:
- Called from the Aria Interface for visualization.
- Results feed into the Governance Report for Production Lexicon export.
- The Fork Operator produces a modified CategoricalGraph that excludes
  hostile nodes, ready for re-evaluation by the Mirror of Truth.

Sprint 10 — Sovereign Governance & The Oklahoma Water/Grid War.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.axiomlogix import CategoricalGraph, Object, Morphism


# ---------------------------------------------------------------------------
# Hard Invariant Definitions
# ---------------------------------------------------------------------------

class HardInvariant:
    """Enumeration of Hard Constraints that must never be violated."""

    EQUITY = "equity"
    SUSTAINABILITY = "sustainability"
    AGENCY = "agency"
    WATER_FLOOR = "water_floor"
    COST_CAUSATION = "cost_causation"


@dataclass
class InvariantViolation:
    """A detected violation of a Hard Invariant."""

    invariant: str
    description: str
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM"
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "description": self.description,
            "severity": self.severity,
            "metricName": self.metric_name,
            "metricValue": round(self.metric_value, 4),
            "threshold": round(self.threshold, 4),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Beta-Distribution Prior for Monte Carlo Simulations
# ---------------------------------------------------------------------------

@dataclass
class BetaPrior:
    """A Beta-distribution prior for Bayesian Monte Carlo simulations.

    Parameters
    ----------
    alpha : float
        Shape parameter representing prior successes (survival events).
    beta : float
        Shape parameter representing prior failures (collapse events).
    label : str
        Human-readable label for this prior (e.g., "Blackout Shock").
    """

    alpha: float
    beta: float
    label: str

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))

    def update(self, successes: int, failures: int) -> "BetaPrior":
        """Return a new BetaPrior with updated evidence."""
        return BetaPrior(
            alpha=self.alpha + successes,
            beta=self.beta + failures,
            label=self.label,
        )

    def sample(self, rng: random.Random) -> float:
        """Draw a sample from this Beta distribution."""
        return rng.betavariate(max(0.01, self.alpha), max(0.01, self.beta))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "mean": round(self.mean, 4),
            "variance": round(self.variance, 6),
        }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloSimResult:
    """Result of a single Monte Carlo simulation scenario."""

    scenario: str  # "blackout_shock" | "drought_event"
    prior: BetaPrior
    posterior: BetaPrior
    runs: int
    survival_count: int
    collapse_count: int
    mean_survival_probability: float
    event_probability: float
    invariant_violations: list[InvariantViolation] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        return self.survival_count / self.runs if self.runs > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "prior": self.prior.as_dict(),
            "posterior": self.posterior.as_dict(),
            "runs": self.runs,
            "survivalCount": self.survival_count,
            "collapseCount": self.collapse_count,
            "survivalRate": round(self.survival_rate, 4),
            "meanSurvivalProbability": round(self.mean_survival_probability, 4),
            "eventProbability": round(self.event_probability, 4),
            "invariantViolations": [v.as_dict() for v in self.invariant_violations],
        }


# ---------------------------------------------------------------------------
# Decentralized Fork Operator
# ---------------------------------------------------------------------------

@dataclass
class ForkResult:
    """Result of a Decentralized Fork Operation.

    When a hostile agent refuses a regenerative repair, the Fork Operator
    creates a protected sub-graph that excludes the hostile node while
    preserving all community and protective relationships.
    """

    hostile_node_label: str
    refused_repair: str
    original_object_count: int
    original_morphism_count: int
    forked_object_count: int
    forked_morphism_count: int
    protected_basins: list[str]
    forked_graph: CategoricalGraph
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "forkOperator": {
                "hostileNodeLabel": self.hostile_node_label,
                "refusedRepair": self.refused_repair,
                "originalObjects": self.original_object_count,
                "originalMorphisms": self.original_morphism_count,
                "forkedObjects": self.forked_object_count,
                "forkedMorphisms": self.forked_morphism_count,
                "protectedBasins": self.protected_basins,
                "timestamp": self.timestamp,
            }
        }


class DecentralizedForkOperator:
    """Protects local basins by forking the covenant graph.

    When a hostile agent (e.g., Extractive Data Center) refuses a
    regenerative repair, the operator:
    1. Identifies all morphisms connected to the hostile node.
    2. Removes the hostile node and its extractive morphisms.
    3. Preserves community dependency and protective morphisms.
    4. Returns a forked graph representing the protected basin covenant.
    """

    @staticmethod
    def fork(
        graph: CategoricalGraph,
        hostile_label: str,
        refused_repair: str,
    ) -> ForkResult:
        """Fork the graph to exclude a hostile node.

        Parameters
        ----------
        graph : CategoricalGraph
            The current conflict graph.
        hostile_label : str
            Label of the hostile node to exclude.
        refused_repair : str
            Description of the repair the hostile agent refused.

        Returns
        -------
        ForkResult
            The fork result with the protected sub-graph.
        """
        original_obj_count = len(graph.objects)
        original_mor_count = len(graph.morphisms)

        # Find the hostile node
        hostile_id = None
        for obj in graph.objects:
            if obj.label == hostile_label:
                hostile_id = obj.id
                break

        if hostile_id is None:
            # No hostile node found, return the original graph
            return ForkResult(
                hostile_node_label=hostile_label,
                refused_repair=refused_repair,
                original_object_count=original_obj_count,
                original_morphism_count=original_mor_count,
                forked_object_count=original_obj_count,
                forked_morphism_count=original_mor_count,
                protected_basins=[],
                forked_graph=graph,
            )

        # Build new graph excluding hostile node and its extractive morphisms
        forked_objects = [
            obj for obj in graph.objects if obj.id != hostile_id
        ]
        forked_morphisms = [
            m for m in graph.morphisms
            if m.source != hostile_id and m.target != hostile_id
        ]

        # Identify protected basins (water_basin tagged objects that remain)
        protected_basins = [
            obj.label for obj in forked_objects
            if "water_basin" in obj.tags or "natural_resource" in obj.tags
        ]

        forked_graph = CategoricalGraph(
            source_text=f"Protected basin covenant (forked from hostile: {hostile_label})",
        )
        for obj in forked_objects:
            forked_graph.add_object(obj.label, list(obj.tags))
        for m in forked_morphisms:
            # Resolve source and target objects in the new graph
            src_obj = None
            tgt_obj = None
            for obj in forked_graph.objects:
                for orig in forked_objects:
                    if orig.id == m.source and orig.label == obj.label:
                        src_obj = obj
                    if orig.id == m.target and orig.label == obj.label:
                        tgt_obj = obj
            if src_obj and tgt_obj:
                forked_graph.add_morphism(m.label, src_obj, tgt_obj, list(m.tags))

        return ForkResult(
            hostile_node_label=hostile_label,
            refused_repair=refused_repair,
            original_object_count=original_obj_count,
            original_morphism_count=original_mor_count,
            forked_object_count=len(forked_graph.objects),
            forked_morphism_count=len(forked_graph.morphisms),
            protected_basins=protected_basins,
            forked_graph=forked_graph,
        )


# ---------------------------------------------------------------------------
# Robustness Harness
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    """Complete result from the Robustness Harness.

    Contains Monte Carlo simulation results for both Blackout Shock and
    Drought Event scenarios, invariant violation tracking, and any
    fork operations triggered.
    """

    blackout_sim: MonteCarloSimResult
    drought_sim: MonteCarloSimResult
    combined_robustness_score: float  # 0.0-10.0
    invariant_violations: list[InvariantViolation]
    fork_results: list[ForkResult]
    passed: bool
    blocking_reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "robustnessHarness": {
                "blackoutShock": self.blackout_sim.as_dict(),
                "droughtEvent": self.drought_sim.as_dict(),
                "combinedRobustnessScore": round(self.combined_robustness_score, 4),
                "invariantViolations": [v.as_dict() for v in self.invariant_violations],
                "forkResults": [f.as_dict() for f in self.fork_results],
                "passed": self.passed,
                "blockingReasons": self.blocking_reasons,
            }
        }


class RobustnessHarness:
    """Bayesian stress-testing harness for Regenerative Covenants.

    Runs Monte Carlo simulations with Beta-distribution priors for:
    - **Blackout Shock**: Models grid fragility under rapid 3+ GW load growth.
    - **Drought Event**: Models water basin depletion under sustained
      industrial cooling demand exceeding aquifer recharge.

    Enforces Hard Invariants:
    - **Water Floor**: Residential water security >= 125% of industrial demand.
    - **Cost Causation**: 100% of HILL infrastructure costs to Hyperscale_Node.

    Triggers the ``DecentralizedForkOperator`` when hostile agents refuse
    regenerative repairs.

    Parameters
    ----------
    blackout_prior : BetaPrior | None
        Prior for blackout shock probability.  Default: Beta(2, 8) — low
        prior belief in grid failure.
    drought_prior : BetaPrior | None
        Prior for drought event probability.  Default: Beta(3, 7) — moderate
        prior belief in water stress.
    monte_carlo_runs : int
        Number of MC runs per scenario (default 200).
    simulation_horizon_years : int
        Temporal horizon for each MC run (default 50).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        blackout_prior: BetaPrior | None = None,
        drought_prior: BetaPrior | None = None,
        monte_carlo_runs: int = 200,
        simulation_horizon_years: int = 50,
        seed: int | None = None,
    ) -> None:
        self.blackout_prior = blackout_prior or BetaPrior(
            alpha=2.0, beta=8.0, label="Blackout Shock",
        )
        self.drought_prior = drought_prior or BetaPrior(
            alpha=3.0, beta=7.0, label="Drought Event",
        )
        self.monte_carlo_runs = monte_carlo_runs
        self.simulation_horizon_years = simulation_horizon_years
        self._rng = random.Random(seed)
        self._fork_operator = DecentralizedForkOperator()

    # -- public API ---------------------------------------------------------

    def evaluate(
        self,
        graph: CategoricalGraph,
        scenario: dict[str, Any] | None = None,
    ) -> RobustnessResult:
        """Run the full robustness evaluation.

        Parameters
        ----------
        graph : CategoricalGraph
            The conflict graph to stress-test.
        scenario : dict | None
            Optional scenario data (from grid_war_2026.json) for
            extracting constraint parameters.

        Returns
        -------
        RobustnessResult
            Complete harness result with MC sims and invariant checks.
        """
        constraints = (scenario or {}).get("constraints", {})
        hill_request = (scenario or {}).get("hill_request", {})
        pso_case = (scenario or {}).get("pso_rate_case", {})

        # Run Monte Carlo simulations
        blackout_result = self._run_blackout_sim(graph, constraints)
        drought_result = self._run_drought_sim(graph, constraints)

        # Check hard invariants
        violations = self._check_invariants(
            graph, constraints, hill_request, pso_case,
            blackout_result, drought_result,
        )

        # Check for hostile agents and trigger fork if needed
        fork_results = self._check_hostile_agents(graph, scenario)

        # Combine violations from all sources
        all_violations = (
            violations
            + blackout_result.invariant_violations
            + drought_result.invariant_violations
        )

        # Compute combined robustness score
        combined_score = self._compute_combined_score(
            blackout_result, drought_result, all_violations,
        )

        # Determine pass/fail
        blocking_reasons = []
        critical_violations = [
            v for v in all_violations if v.severity == "CRITICAL"
        ]
        if critical_violations:
            for v in critical_violations:
                blocking_reasons.append(
                    f"INVARIANT VIOLATION [{v.invariant}]: {v.description}"
                )

        if combined_score < 5.0:
            blocking_reasons.append(
                f"Combined robustness score {combined_score:.4f} "
                f"is below minimum threshold of 5.0."
            )

        passed = len(blocking_reasons) == 0

        return RobustnessResult(
            blackout_sim=blackout_result,
            drought_sim=drought_result,
            combined_robustness_score=combined_score,
            invariant_violations=all_violations,
            fork_results=fork_results,
            passed=passed,
            blocking_reasons=blocking_reasons,
        )

    def trigger_fork(
        self,
        graph: CategoricalGraph,
        hostile_label: str,
        refused_repair: str,
    ) -> ForkResult:
        """Manually trigger a fork operation for a hostile agent.

        Parameters
        ----------
        graph : CategoricalGraph
            The current conflict graph.
        hostile_label : str
            Label of the hostile node.
        refused_repair : str
            Description of the repair they refused.

        Returns
        -------
        ForkResult
            The protected sub-graph result.
        """
        return self._fork_operator.fork(graph, hostile_label, refused_repair)

    # -- Monte Carlo: Blackout Shock ----------------------------------------

    def _run_blackout_sim(
        self,
        graph: CategoricalGraph,
        constraints: dict[str, Any],
    ) -> MonteCarloSimResult:
        """Run Blackout Shock Monte Carlo with Beta-distribution prior."""
        prior = self.blackout_prior

        # Extract graph evidence
        grid_strain_count = sum(
            1 for m in graph.morphisms
            if any(t in m.tags for t in ["reliability_threat", "capacity_demand"])
        )
        protective_count = sum(
            1 for m in graph.morphisms
            if any(t in m.tags for t in ["protection", "regulatory_oversight"])
        )

        # Update prior with evidence from graph structure
        posterior = prior.update(
            successes=protective_count * 2,
            failures=grid_strain_count * 3,
        )

        # Run MC simulations
        survival_count = 0
        total_survival_prob = 0.0
        violations: list[InvariantViolation] = []

        fragility_amp = constraints.get("blackout_shock", {}).get(
            "fragility_amplifier", 1.5,
        )

        for _ in range(self.monte_carlo_runs):
            # Sample event probability from posterior
            event_prob = posterior.sample(self._rng)
            event_prob = min(1.0, event_prob * fragility_amp)

            # Simulate over the horizon
            survived = True
            for year in range(self.simulation_horizon_years):
                if self._rng.random() < event_prob * 0.1:
                    # Blackout event occurred
                    # Check if grid has enough resilience
                    resilience = self._rng.random()
                    if resilience < event_prob:
                        survived = False
                        break

            if survived:
                survival_count += 1
            total_survival_prob += 1.0 if survived else 0.0

        mean_survival = total_survival_prob / self.monte_carlo_runs

        # Check if blackout probability is too high
        blackout_prob = 1.0 - mean_survival
        if blackout_prob > 0.3:
            violations.append(InvariantViolation(
                invariant=HardInvariant.SUSTAINABILITY,
                description=(
                    f"Blackout probability {blackout_prob:.1%} exceeds "
                    f"30% safety threshold over {self.simulation_horizon_years}-year horizon."
                ),
                severity="CRITICAL",
                metric_name="blackout_probability",
                metric_value=blackout_prob,
                threshold=0.3,
            ))

        return MonteCarloSimResult(
            scenario="blackout_shock",
            prior=prior,
            posterior=posterior,
            runs=self.monte_carlo_runs,
            survival_count=survival_count,
            collapse_count=self.monte_carlo_runs - survival_count,
            mean_survival_probability=mean_survival,
            event_probability=posterior.mean,
            invariant_violations=violations,
        )

    # -- Monte Carlo: Drought Event -----------------------------------------

    def _run_drought_sim(
        self,
        graph: CategoricalGraph,
        constraints: dict[str, Any],
    ) -> MonteCarloSimResult:
        """Run Drought Event Monte Carlo with Beta-distribution prior."""
        prior = self.drought_prior

        # Extract water stress evidence from graph
        water_depletion_count = sum(
            1 for m in graph.morphisms
            if any(t in m.tags for t in [
                "water_depletion", "unsustainable_withdrawal",
            ])
        )
        water_protection_count = sum(
            1 for m in graph.morphisms
            if any(t in m.tags for t in [
                "basin_protection", "sustainable_management",
            ])
        )

        # Update prior with evidence
        posterior = prior.update(
            successes=water_protection_count * 2,
            failures=water_depletion_count * 3,
        )

        # Extract water constraints
        water_c = constraints.get("water_sustainability", {})
        demand_mgd = water_c.get("datacenter_demand_mgd", 42.0)
        recharge_mgd = water_c.get("aquifer_recharge_rate_mgd", 18.0)
        overshoot = demand_mgd / recharge_mgd if recharge_mgd > 0 else float("inf")

        # Run MC simulations
        survival_count = 0
        total_survival_prob = 0.0
        violations: list[InvariantViolation] = []

        for _ in range(self.monte_carlo_runs):
            # Sample drought severity from posterior
            drought_severity = posterior.sample(self._rng)

            # Simulate water basin health over horizon
            basin_health = 1.0
            survived = True

            for year in range(self.simulation_horizon_years):
                # Industrial demand depletes; recharge replenishes
                depletion = (overshoot - 1.0) * 0.02 * drought_severity
                recharge_rate = (1.0 / overshoot) * 0.03 * (1.0 - drought_severity)
                basin_health += recharge_rate - depletion

                # Random drought events
                if self._rng.random() < drought_severity * 0.15:
                    basin_health -= 0.05

                basin_health = max(0.0, min(1.0, basin_health))

                if basin_health < 0.1:
                    survived = False
                    break

            if survived:
                survival_count += 1
            total_survival_prob += 1.0 if survived else 0.0

        mean_survival = total_survival_prob / self.monte_carlo_runs

        # Check water floor invariant
        water_floor = constraints.get("water_floor_invariant", {})
        surplus_pct = water_floor.get("residential_surplus_pct", 25)
        required_ratio = 1.0 + surplus_pct / 100.0  # 1.25

        if overshoot > 1.0:
            residential_security = recharge_mgd / demand_mgd
            actual_surplus_pct = (residential_security - 1.0) * 100
            if actual_surplus_pct < surplus_pct:
                violations.append(InvariantViolation(
                    invariant=HardInvariant.WATER_FLOOR,
                    description=(
                        f"Residential water security surplus is "
                        f"{actual_surplus_pct:.0f}% (deficit) vs. "
                        f"required +{surplus_pct}%. "
                        f"Demand {demand_mgd} MGD vs. recharge {recharge_mgd} MGD "
                        f"({overshoot:.2f}x overshoot)."
                    ),
                    severity="CRITICAL",
                    metric_name="water_security_surplus_pct",
                    metric_value=actual_surplus_pct,
                    threshold=float(surplus_pct),
                ))

        drought_prob = 1.0 - mean_survival
        if drought_prob > 0.2:
            violations.append(InvariantViolation(
                invariant=HardInvariant.SUSTAINABILITY,
                description=(
                    f"Drought-induced basin collapse probability "
                    f"{drought_prob:.1%} exceeds 20% safety threshold "
                    f"over {self.simulation_horizon_years}-year horizon."
                ),
                severity="CRITICAL",
                metric_name="drought_collapse_probability",
                metric_value=drought_prob,
                threshold=0.2,
            ))

        return MonteCarloSimResult(
            scenario="drought_event",
            prior=prior,
            posterior=posterior,
            runs=self.monte_carlo_runs,
            survival_count=survival_count,
            collapse_count=self.monte_carlo_runs - survival_count,
            mean_survival_probability=mean_survival,
            event_probability=posterior.mean,
            invariant_violations=violations,
        )

    # -- Hard Invariant Checking --------------------------------------------

    def _check_invariants(
        self,
        graph: CategoricalGraph,
        constraints: dict[str, Any],
        hill_request: dict[str, Any],
        pso_case: dict[str, Any],
        blackout_result: MonteCarloSimResult,
        drought_result: MonteCarloSimResult,
    ) -> list[InvariantViolation]:
        """Check all hard invariants against the scenario data."""
        violations: list[InvariantViolation] = []

        # -- Cost Causation Invariant (HB 2992) --
        cost_causation = constraints.get("cost_causation_invariant", {})
        required_pct = cost_causation.get("hill_cost_allocation_pct", 100)

        # Check from PSO rate case data
        hill_to_hill_pct = pso_case.get("hill_cost_allocated_to_hill_pct", None)
        if hill_to_hill_pct is not None and hill_to_hill_pct < required_pct:
            violations.append(InvariantViolation(
                invariant=HardInvariant.COST_CAUSATION,
                description=(
                    f"HILL infrastructure cost allocation to HILL customers "
                    f"is {hill_to_hill_pct}% vs. required {required_pct}% "
                    f"per HB 2992 cost-causation intent. "
                    f"{pso_case.get('hill_cost_allocated_to_residential_pct', 0)}% "
                    f"is being socialised onto residential ratepayers."
                ),
                severity="CRITICAL",
                metric_name="hill_cost_allocation_pct",
                metric_value=float(hill_to_hill_pct),
                threshold=float(required_pct),
            ))

        # -- Equity Invariant --
        residential_increase = pso_case.get("residential_monthly_increase_usd", 0)
        if residential_increase > 0:
            # Any increase for infrastructure serving corporate loads is a violation
            hill_cost_to_residential = pso_case.get(
                "hill_cost_allocated_to_residential_pct", 0,
            )
            if hill_cost_to_residential > 0:
                violations.append(InvariantViolation(
                    invariant=HardInvariant.EQUITY,
                    description=(
                        f"${residential_increase}/month increase on residential "
                        f"customers for infrastructure serving corporate HILL loads. "
                        f"{hill_cost_to_residential}% of HILL costs allocated to "
                        f"residential class."
                    ),
                    severity="HIGH",
                    metric_name="residential_monthly_increase_usd",
                    metric_value=float(residential_increase),
                    threshold=0.0,
                ))

        # -- Sustainability (aggregate check) --
        # Check for shadow entities without protective morphisms
        shadow_entities = [
            obj for obj in graph.objects
            if "shadow_entity" in obj.tags
        ]
        protected_shadows = set()
        for m in graph.morphisms:
            if any(t in m.tags for t in ["protection", "regulatory_oversight", "basin_protection"]):
                protected_shadows.add(m.target)

        for se in shadow_entities:
            if se.id not in protected_shadows:
                violations.append(InvariantViolation(
                    invariant=HardInvariant.SUSTAINABILITY,
                    description=(
                        f"Shadow entity '{se.label}' has no protective "
                        f"morphisms — invisible dependency is unguarded."
                    ),
                    severity="HIGH",
                    metric_name="unprotected_shadow_entities",
                    metric_value=1.0,
                    threshold=0.0,
                ))

        return violations

    # -- Hostile Agent Detection & Fork -------------------------------------

    def _check_hostile_agents(
        self,
        graph: CategoricalGraph,
        scenario: dict[str, Any] | None,
    ) -> list[ForkResult]:
        """Check for hostile agents that refuse regenerative repairs.

        A hostile agent is a node tagged as ``high_impact_large_load`` or
        ``corporate`` that has extractive morphisms to water basins without
        any reciprocal regenerative morphisms (e.g., wastewater treatment,
        closed-loop cooling).
        """
        if scenario is None:
            return []

        fork_results: list[ForkResult] = []

        # Find hostile nodes: corporate actors with water depletion morphisms
        # but no regenerative return morphisms
        for obj in graph.objects:
            if "high_impact_large_load" not in obj.tags:
                continue

            # Check for extractive water morphisms
            has_water_extraction = False
            has_regenerative_return = False

            for m in graph.morphisms:
                if m.source == obj.id:
                    if any(t in m.tags for t in [
                        "water_depletion", "unsustainable_withdrawal",
                    ]):
                        has_water_extraction = True
                    if any(t in m.tags for t in [
                        "wastewater_treatment", "closed_loop_cooling",
                        "regenerative", "replenishment",
                    ]):
                        has_regenerative_return = True

            # If extracting water without regeneration, this is a hostile agent
            if has_water_extraction and not has_regenerative_return:
                fork_result = self._fork_operator.fork(
                    graph,
                    obj.label,
                    "Switch to treated wastewater or closed-loop cooling "
                    "to reduce freshwater withdrawal below sustainable "
                    "aquifer recharge rates.",
                )
                fork_results.append(fork_result)

        return fork_results

    # -- Scoring ------------------------------------------------------------

    def _compute_combined_score(
        self,
        blackout: MonteCarloSimResult,
        drought: MonteCarloSimResult,
        violations: list[InvariantViolation],
    ) -> float:
        """Compute a combined robustness score (0.0-10.0).

        Factors:
        - Blackout survival probability (weight 35%)
        - Drought survival probability (weight 35%)
        - Invariant compliance (weight 30%)
        """
        blackout_score = blackout.mean_survival_probability * 10.0
        drought_score = drought.mean_survival_probability * 10.0

        # Invariant penalty
        critical_count = sum(
            1 for v in violations if v.severity == "CRITICAL"
        )
        high_count = sum(1 for v in violations if v.severity == "HIGH")
        invariant_score = max(
            0.0,
            10.0 - critical_count * 3.0 - high_count * 1.5,
        )

        combined = (
            blackout_score * 0.35
            + drought_score * 0.35
            + invariant_score * 0.30
        )
        return max(0.0, min(10.0, combined))
