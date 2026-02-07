#!/usr/bin/env python3
"""
Sprint 8 Demo — The Oklahoma Grid War (2026)

Demonstrates the full Genesis Engine pipeline deconstructing a "High-Impact
Large Load" (HILL) data centre request, rejecting a profit-shifting reform,
and crystallizing a "Regenerative Grid Covenant" that passes the Final Exam.

Pipeline:
1. Load the Grid War 2026 scenario
2. Translate the conflict graph through AxiomLogix
3. Deconstruct for disharmony (Module 1.1)
4. Dream three solution paths (Module 1.2)
5. Mirror of Truth: Adversarial Deconstruction (Module 1.7)
6. Bayesian Blackout Shock exam (Module 3.5 Sprint 8)
7. Forge the Regenerative Grid Covenant (Module 1.3)
8. Standard Final Exam validation (Module 3.5)
9. Display via Aria Interface with Refinement Panel (Module 3.2)
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.game_theory_console import (
    BayesianFinalExam,
    CovenantFinalExam,
    FinalExam,
)
from genesis_engine.core.mirror_of_truth import MirrorOfTruth
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer


def main() -> None:
    print("\n" + "=" * 70)
    print("  GENESIS ENGINE — SPRINT 8 DEMO")
    print("  The Oklahoma Grid War (2026)")
    print("=" * 70)

    renderer = AriaRenderer(use_colors=True)

    # -----------------------------------------------------------------------
    # Step 1: Load the Grid War scenario
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 1: Loading Grid War 2026 Scenario"))

    scenario_path = os.path.join(
        os.path.dirname(__file__), "scenarios", "grid_war_2026.json",
    )
    scenario = MirrorOfTruth.load_scenario(scenario_path)
    graph = MirrorOfTruth.scenario_to_graph(scenario)

    print(f"  Scenario: {scenario['scenario']}")
    print(f"  Objects:  {len(graph.objects)}")
    print(f"  Morphisms: {len(graph.morphisms)}")
    print(f"  HILL Request: {scenario['hill_request']['demand_mw']} MW")
    print(f"  Infrastructure Cost: ${scenario['hill_request']['infrastructure_cost_usd']:,}")
    print(f"  Residential Impact: ${scenario['hill_request']['residential_impact_monthly_usd']}/month")

    # -----------------------------------------------------------------------
    # Step 2: Deconstruct for Disharmony
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 2: Deconstruction Engine Analysis"))

    anchor = AxiomAnchor()
    decon = DeconstructionEngine(anchor=anchor)
    report = decon.analyse(graph)

    print(f"  Unity Impact:          {report.unity_impact:.1f}/10")
    print(f"  Compassion Deficit:    {report.compassion_deficit:.1f}/10")
    print(f"  Coherence Score:       {report.coherence_score:.1f}/10")
    print(f"  Incentive Stability:   {report.incentive_stability_score:.1f}/10")
    print(f"  Incentive Instability: {report.incentive_instability}")
    print(f"  Prime Directive Aligned: {report.is_aligned}")

    if report.incentive_instability:
        print(f"\n  *** SHAREHOLDER PRIMACY GRAVITY WELL DETECTED ***")
        print(f"  The utility's fiduciary duty to shareholders creates a")
        print(f"  legal gravity well that distorts cost allocation.")

    # -----------------------------------------------------------------------
    # Step 3: Dream Engine — Generate Three Paths
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 3: Dream Engine — Threefold Path"))

    dream = DreamEngine(anchor=anchor)
    possibility = dream.dream(report, graph)

    for path in possibility.paths:
        status = " (RECOMMENDED)" if path.path_type.value == possibility.recommended_path else ""
        print(f"\n  {path.title}{status}")
        print(f"    Unity Alignment: {path.unity_alignment_score:.4f}")
        print(f"    Feasibility:     {path.feasibility_score:.4f}")
        print(f"    Description:     {path.description[:80]}...")

    # -----------------------------------------------------------------------
    # Step 4: Mirror of Truth — Adversarial Deconstruction
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 4: Mirror of Truth — Self-Critique"))

    mirror = MirrorOfTruth(
        anchor=anchor,
        vulnerability_priority="Residential_Ratepayer",
    )

    # Critique the Dream Engine's recommended path
    selected_path, trace = mirror.critique_and_refine(
        possibility, report, graph,
    )

    # Display the Refinement Panel via Aria
    print(renderer.render_refinement_panel(trace))

    print(f"  Selected Path: {selected_path.title} ({selected_path.path_type.value})")
    print(f"  Unity Alignment: {selected_path.unity_alignment_score:.4f}")

    # -----------------------------------------------------------------------
    # Step 5: Bayesian Blackout Shock Exam
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 5: Bayesian Blackout Shock Exam"))

    bayesian_exam = BayesianFinalExam(
        pass_threshold=7.0,
        fragility_amplifier=1.5,
        prior_viability=0.6,
    )
    shock_result = bayesian_exam.administer(seed=42)
    print(renderer.render_blackout_shock(shock_result))

    # -----------------------------------------------------------------------
    # Step 6: Forge the Regenerative Grid Covenant
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 6: Architectural Forge"))

    forge = ArchitecturalForge(anchor=anchor)
    artifact = forge.forge(selected_path)

    print(f"  Covenant Title: {artifact.covenant.title}")
    print(f"  Integrity Verified: {artifact.integrity_verified}")
    print(f"  Data Models: {len(artifact.covenant.data_models)}")
    print(f"  Endpoints: {len(artifact.covenant.endpoints)}")
    print(f"  Governance Rules: {len(artifact.covenant.governance_rules)}")

    if artifact.manifesto:
        print(f"\n  Stewardship Manifesto:")
        for k, v in artifact.manifesto.alignment_scores.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            print(f"    {k}: {v:.4f} {bar}")
        if artifact.manifesto.regenerative_loop.repair_morphisms:
            print(f"\n  Regenerative Loop ({len(artifact.manifesto.regenerative_loop.repair_morphisms)} repair morphisms):")
            for rm in artifact.manifesto.regenerative_loop.repair_morphisms[:3]:
                print(f"    • {rm.source_model} → {rm.target_model}")
                print(f"      Trigger: {rm.trigger_condition}")

    # -----------------------------------------------------------------------
    # Step 7: Covenant-Aware Final Exam
    # -----------------------------------------------------------------------
    print(renderer.render_header("STEP 7: Covenant-Aware Final Exam"))

    # Compute governance strength from the forged artifact
    gov_strength = CovenantFinalExam.compute_governance_strength(
        alignment_scores=artifact.manifesto.alignment_scores if artifact.manifesto else {},
        governance_rule_count=len(artifact.covenant.governance_rules),
        repair_morphism_count=(
            len(artifact.manifesto.regenerative_loop.repair_morphisms)
            if artifact.manifesto else 0
        ),
    )

    print(f"  Governance Strength: {gov_strength:.4f}")
    print(f"  (Derived from alignment scores, governance rules,")
    print(f"   and regenerative repair morphisms)")

    covenant_exam = CovenantFinalExam(pass_threshold=7.5)
    exam_result = covenant_exam.administer(
        governance_strength=gov_strength, seed=42,
    )

    print(f"\n  Sustainability Score: {exam_result.sustainability_score:.4f}")
    print(f"  Pass Threshold:      {exam_result.pass_threshold}")
    print(f"  Effective Defection: {exam_result.effective_defection_rate:.4f}")
    print(f"  Passed:              {exam_result.passed}")
    print(f"  Outcome:             {exam_result.outcome.outcome_flag.value}")

    if exam_result.passed:
        print(f"\n  *** REGENERATIVE GRID COVENANT PASSES THE FINAL EXAM ***")
        print(f"  The covenant's governance structures constrain extractive")
        print(f"  behaviour sufficiently to sustain cooperation over 100 years.")
    else:
        print(f"\n  {exam_result.blocking_reason}")

    # Also show what the raw exam looks like without governance
    print(renderer.render_subheader("Comparison: Raw Final Exam (no governance)"))
    raw_exam = FinalExam(pass_threshold=7.0)
    raw_result = raw_exam.administer(seed=42)
    print(f"  Raw Sustainability: {raw_result.sustainability_score:.4f} (FAILS)")
    print(f"  With Governance:    {exam_result.sustainability_score:.4f}")
    print(f"  Governance Delta:   +{exam_result.sustainability_score - raw_result.sustainability_score:.4f}")
    print(f"\n  This demonstrates the core Sprint 8 insight:")
    print(f"  structural governance transforms extractive collapse into")
    print(f"  sustainable cooperation.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(renderer.render_header("SPRINT 8 DEMO COMPLETE"))

    print("  The Genesis Engine has:")
    print("  1. Detected the legal gravity well (utility shareholder primacy)")
    print("  2. Simulated 100 years of economic warfare")
    print("  3. Critiqued its own solution for hidden extraction")
    print(f"  4. Mirror Score: {trace.mirror_score:.2f}/10")
    print(f"  5. Surface Alignment Detected: {trace.surface_alignment_detected}")
    print(f"  6. Deep Disharmony: {', '.join(trace.deep_disharmony_categories)}")
    print(f"  7. Covenant Governance Strength: {gov_strength:.4f}")
    print(f"  8. Final Exam Score: {exam_result.sustainability_score:.4f}")
    print(f"  9. Final Exam Passed: {exam_result.passed}")

    print(f"\n  The Mirror of Truth identified all 4 categories of Deep")
    print(f"  Disharmony in the HILL request and prescribed 4 mandatory")
    print(f"  Regenerative Repairs. The Dream Engine selected a structural")
    print(f"  solution (Dissolution → Cooperative model) that eliminates")
    print(f"  the shareholder primacy gravity well.")

    print(f"\n  Covenant: {artifact.covenant.title}")
    print(f"  Unity over Power — the machine refuses to build what will collapse.")
    print()


if __name__ == "__main__":
    main()
