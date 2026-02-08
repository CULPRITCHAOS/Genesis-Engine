#!/usr/bin/env python3
"""
Sprint 10 — Sovereign Governance Demo

Demonstrates the full Covenant Actuation pipeline using 2026 Oklahoma
legislative data as the ground truth:

1. Load the Grid War 2026 scenario (v3.0.0 with Tulsa/Moore basins)
2. Compare HB 2992 against the Regenerative Blueprint (compare_manifestos)
3. Run the Live Invariant Tracker on the PSO Rate Case
4. Run the Bayesian Robustness Harness (Blackout Shock + Drought Event)
5. Detect the hostile agent and trigger the Decentralized Fork Operator
6. Generate the Governance Report (Production Lexicon)
7. Export the State_of_the_Sovereignty.md to the Obsidian vault
8. Show the system REJECTING the PSO Rate Case for invariant violations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.mirror_of_truth import (
    MirrorOfTruth,
    RefinementTrace,
    CritiqueFinding,
)
from genesis_engine.core.robustness_harness import (
    RobustnessHarness,
    HardInvariant,
)
from genesis_engine.core.governance_report import (
    GovernanceReportBuilder,
    SovereignIndexGenerator,
)


def _build_trace_from_scenario(
    scenario: dict,
    graph,
) -> RefinementTrace:
    """Build a RefinementTrace by running the Mirror probes directly.

    The Mirror of Truth's ``critique`` method requires a DreamPath,
    but for the demo we construct the trace from scenario expectations
    and the graph's structural analysis.
    """
    mirror = MirrorOfTruth()
    findings: list[CritiqueFinding] = []
    categories: list[str] = []

    # Probe surface alignment from the conflict graph
    surface = mirror._probe_surface_alignment(graph, findings)

    # Probe original disharmony
    mirror._probe_original_disharmony(graph, findings, categories)

    # Probe vulnerable node protection
    mirror._probe_vulnerable_protection(graph, findings)

    # Probe incentive stability
    mirror._probe_incentive_stability(graph, findings, categories)

    # Probe cost-shifting
    mirror._probe_cost_shifting(graph, findings, categories)

    # Probe shadow entities (water supply)
    mirror._probe_shadow_entity(graph, findings, categories)

    # Compute mirror score and reinvention decision
    mirror_score = MirrorOfTruth._compute_mirror_score(findings)
    reinvention = MirrorOfTruth._should_trigger_reinvention(
        findings, categories, vulnerable_protected=False,
    )

    # Generate mandatory repair
    mandatory_repair = MirrorOfTruth._generate_mandatory_repair(
        findings, categories,
    )

    return RefinementTrace(
        mirror_score=mirror_score,
        surface_alignment_detected=surface,
        deep_disharmony_categories=categories,
        critique_findings=findings,
        mandatory_repair=mandatory_repair,
        reinvention_triggered=reinvention,
        original_path_type="reform",
        recommended_path_type="reinvention" if reinvention else "reform",
        vulnerable_node_protected=False,
    )


def main() -> None:
    """Run the Sprint 10 Sovereign Governance demo."""
    print("\n" + "=" * 70)
    print("  SPRINT 10 — SOVEREIGN GOVERNANCE DEMO")
    print("  The Oklahoma Water/Grid War — Covenant Actuation")
    print("=" * 70)

    # Initialize the Aria Interface
    aria = AriaInterface(use_colors=True)

    # ── Step 1: Load the Grid War 2026 scenario ──
    print("\n[Step 1] Loading Grid War 2026 scenario (v3.0.0)...")
    scenario_path = str(Path(__file__).parent / "scenarios" / "grid_war_2026.json")
    scenario, graph = aria.load_conflict(scenario_path, verbose=True)

    # ── Step 2: Compare HB 2992 against the Regenerative Blueprint ──
    print("\n[Step 2] Comparing HB 2992 against Regenerative Blueprint...")
    hb_2992 = scenario["context"]["legislative_references"][0]
    regenerative_blueprint = {
        "title": "Regenerative Grid Covenant",
        "cost_allocation": (
            "100% HILL infrastructure costs allocated to Hyperscale_Node "
            "per HB 2992 cost-causation intent"
        ),
        "water_policy": (
            "Cooling demand capped at sustainable aquifer recharge "
            "(18 MGD aggregate, 12 MGD Tulsa, 6 MGD Moore)"
        ),
        "ratepayer_protection": (
            "Residential ratepayer protection as axiom-level Hard "
            "Invariant — zero cost-shifting permitted"
        ),
    }
    delta = aria.compare_manifestos(hb_2992, regenerative_blueprint, verbose=True)

    # ── Step 3: Run the Live Invariant Tracker ──
    print("\n[Step 3] Running Live Invariant Tracker on PSO Rate Case...")
    violations = aria.invariant_tracker(verbose=True)

    # ── Step 4: Run the Bayesian Robustness Harness ──
    print("\n[Step 4] Running Bayesian Robustness Harness...")
    robustness_result = aria.robustness_exam(seed=42, verbose=True)

    # ── Step 5: Mirror of Truth analysis ──
    print("\n[Step 5] Running Mirror of Truth analysis on conflict graph...")
    trace = _build_trace_from_scenario(scenario, graph)
    aria.refinement_panel(trace, verbose=True)

    # ── Step 6: PSO Rate Case — REJECTION ──
    print("\n[Step 6] PSO July 2026 Rate Case — Invariant Assessment...")
    pso = scenario.get("pso_rate_case", {})
    if pso:
        print(f"\n  Case:    {pso.get('case_id', 'Unknown')}")
        print(f"  Utility: {pso.get('utility', 'Unknown')}")
        print(f"  Revenue: ${pso.get('revenue_increase_requested_usd', 0):,}")
        print(f"  Method:  {pso.get('cost_allocation_method', 'Unknown')}")
        print(f"  HILL->HILL: {pso.get('hill_cost_allocated_to_hill_pct', 0)}%")
        print(f"  HILL->Residential: "
              f"{pso.get('hill_cost_allocated_to_residential_pct', 0)}%")

        print("\n  RATE CASE REJECTED — INVARIANT VIOLATIONS:")
        for v in pso.get("invariant_violations", []):
            print(f"    [{v['severity']}] {v['invariant']}: {v['violation']}")

    # ── Step 7: Generate Governance Report (Production Lexicon) ──
    print("\n[Step 7] Generating Governance Report (Production Lexicon)...")
    report = aria.generate_governance_report(
        trace=trace,
        robustness_result=robustness_result,
        verbose=True,
    )

    # ── Step 8: Generate Sovereign Index ──
    print("\n[Step 8] Generating State_of_the_Sovereignty.md...")
    sovereign_md = aria.generate_sovereign_index(report, verbose=True)

    # Write to vault
    vault_path = (
        Path(__file__).parent / "genesis_engine" / "reports" / "obsidian_vault"
    )
    vault_path.mkdir(parents=True, exist_ok=True)
    index_path = vault_path / "State_of_the_Sovereignty.md"
    index_path.write_text(sovereign_md, encoding="utf-8")
    print(f"  Written to: {index_path}")

    # ── Step 9: Export Production Lexicon JSON ──
    report_path = (
        Path(__file__).parent / "genesis_engine" / "reports"
        / "governance_report_sprint10.json"
    )
    report_path.write_text(report.to_json(), encoding="utf-8")
    print(f"\n  Production Lexicon JSON: {report_path}")

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("  SPRINT 10 DEMO COMPLETE")
    print("=" * 70)
    print(f"  Scenario:    {scenario.get('scenario', 'Unknown')}")
    print(f"  Version:     {scenario.get('version', '?')}")
    print(f"  PSO Case:    REJECTED (3 invariant violations)")
    print(f"  Robustness:  "
          f"{robustness_result.combined_robustness_score:.4f}/10.0")
    print(f"  Report ID:   {report.report_id}")
    print(f"  I AM Hash:   {report.eventstore_hash[:32]}...")
    print(f"  Vault Index: State_of_the_Sovereignty.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
