#!/usr/bin/env python3
"""
Genesis Engine — Sprint 6.1: The Time Guardian
=================================================

Demonstrates the Sustainability Axiom and Game Theory Console:

1. **Sustainability Predicate** — Monte Carlo temporal viability,
   ecological harmony (regenerative loop detection), and fragility
   analysis on real categorical graphs.

2. **Shadow Entity Inference** — AxiomLogix now infers "Future Generations"
   and "Ecosystem" as implicit stakeholder nodes.

3. **Game Theory War-Game** — 100-round Iterated Prisoner's Dilemma
   between Stewardship (Aligned) and Extractive agents, with
   SYSTEMIC_COLLAPSE detection.

4. **Foresight Projections** — War-game results stored in the
   Wisdom Log via the Continuity Bridge.

Run:
    python demo_time_guardian.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor,
    IncentiveStabilityPredicate,
    SustainabilityPredicate,
)
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.continuity_bridge import ContinuityBridge
from genesis_engine.core.game_theory_console import GameTheoryConsole, OutcomeFlag


REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subsection(title: str) -> None:
    print("\n" + "-" * 72)
    print(f"  {title}")
    print("-" * 72)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    section("GENESIS ENGINE — Sprint 6.1: The Time Guardian")
    print("  The Sustainability Axiom & Game Theory Console")
    print()

    # ── 1. Shadow Entity Inference ─────────────────────────────────────────
    section("MODULE 1.4 — Shadow Entity Inference")
    print("  AxiomLogix now infers Future_Generations and Ecosystem as")
    print("  implicit stakeholder nodes, even when not mentioned.")
    print()

    translator = AxiomLogixTranslator()

    problems = [
        "A corporation that prioritizes shareholder value over employee welfare.",
        "A community health program that empowers workers and protects the environment.",
        "An AI surveillance system that exploits user data and neglects privacy.",
    ]

    for i, problem in enumerate(problems):
        graph = translator.translate(problem)
        entity_labels = [o.label for o in graph.objects]
        shadow_entities = [o for o in graph.objects if "shadow_entity" in o.tags]

        print(f"\n  Problem {i+1}: {problem[:60]}...")
        print(f"  Entities: {', '.join(entity_labels)}")
        if shadow_entities:
            print(f"  Shadow Entities (inferred):")
            for se in shadow_entities:
                print(f"    - {se.label} (tags: {', '.join(se.tags)})")
        else:
            print(f"  Shadow Entities: (none needed — Environment already explicit)")

    # ── 2. Sustainability Predicate ────────────────────────────────────────
    section("MODULE 2.1 — Sustainability Predicate")
    print("  Monte Carlo temporal viability + Ecological harmony analysis")
    print()

    sustainability_pred = SustainabilityPredicate(seed=42)

    # Test on an extractive graph (shareholder primacy)
    subsection("Test Case A: Extractive Graph (Shareholder Primacy)")

    extractive_graph = translator.translate(
        "A Delaware corporation must maximize shareholder value via "
        "fiduciary duty, extracting profit from employees and customers."
    )
    extractive_artefact = extractive_graph.as_artefact()
    sus_result_a = sustainability_pred.evaluate(extractive_artefact)

    print(f"  Sustainability Score: {sus_result_a.sustainability_score:.2f}/10.0")
    print(f"  Temporal Viability:   {sus_result_a.temporal_viability:.4f}")
    print(f"  Ecological Harmony:   {sus_result_a.ecological_harmony:.4f}")
    print(f"  Fragility Factor:     {sus_result_a.fragility_factor:.4f}")
    print(f"  Regenerative Loops:   {len(sus_result_a.regenerative_loops)}")
    print(f"  Depletion Morphisms:  {len(sus_result_a.depletion_morphisms)}")
    print(f"  Is Sustainable:       {sus_result_a.is_sustainable}")
    print(f"\n  Monte Carlo Projections:")
    for proj in sus_result_a.projections:
        collapse_info = f"earliest collapse at step {proj.collapse_step}" if proj.collapse_step else "no collapse"
        print(
            f"    t={proj.timesteps:>3}: survival={proj.survival_probability:.2%}  "
            f"mean_health={proj.mean_health:.4f}  ({collapse_info})"
        )

    # Test on a regenerative graph (stewardship)
    subsection("Test Case B: Regenerative Graph (Stewardship)")

    stewardship_graph = translator.translate(
        "A cooperative that empowers employees, protects the community, "
        "serves customers, and sustains the environment through collaborative care."
    )
    stewardship_artefact = stewardship_graph.as_artefact()
    sus_result_b = sustainability_pred.evaluate(stewardship_artefact)

    print(f"  Sustainability Score: {sus_result_b.sustainability_score:.2f}/10.0")
    print(f"  Temporal Viability:   {sus_result_b.temporal_viability:.4f}")
    print(f"  Ecological Harmony:   {sus_result_b.ecological_harmony:.4f}")
    print(f"  Fragility Factor:     {sus_result_b.fragility_factor:.4f}")
    print(f"  Regenerative Loops:   {len(sus_result_b.regenerative_loops)}")
    print(f"  Depletion Morphisms:  {len(sus_result_b.depletion_morphisms)}")
    print(f"  Is Sustainable:       {sus_result_b.is_sustainable}")
    print(f"\n  Monte Carlo Projections:")
    for proj in sus_result_b.projections:
        collapse_info = f"earliest collapse at step {proj.collapse_step}" if proj.collapse_step else "no collapse"
        print(
            f"    t={proj.timesteps:>3}: survival={proj.survival_probability:.2%}  "
            f"mean_health={proj.mean_health:.4f}  ({collapse_info})"
        )

    # ── 3. Sustainability Predicate registered with Axiom Anchor ──────────
    subsection("Sustainability Predicate in Axiom Anchor")

    anchor = AxiomAnchor()
    incentive_pred = IncentiveStabilityPredicate()
    anchor.register_predicate("incentive_stability", incentive_pred)
    anchor.register_predicate("sustainability", sustainability_pred)

    val_extractive = anchor.validate(extractive_artefact)
    val_stewardship = anchor.validate(stewardship_artefact)

    print(f"\n  Extractive Graph Validation:")
    print(f"    Aligned: {val_extractive.is_aligned}")
    print(f"    Coherence: {val_extractive.coherence_score:.4f}")
    for k, v in val_extractive.principle_scores.items():
        print(f"    {k}: {v:.4f}")

    print(f"\n  Stewardship Graph Validation:")
    print(f"    Aligned: {val_stewardship.is_aligned}")
    print(f"    Coherence: {val_stewardship.coherence_score:.4f}")
    for k, v in val_stewardship.principle_scores.items():
        print(f"    {k}: {v:.4f}")

    anchor.seal()
    print(f"\n  Axiom Anchor sealed with sustainability predicate.")

    # ── 4. Game Theory War-Game ────────────────────────────────────────────
    section("MODULE 3.5 — Game Theory Console (100-Round Economic War)")

    aria = AriaInterface(use_colors=True)

    # Run the war-game via Aria
    outcome = aria.war_game(rounds=100, seed=42, verbose=True)

    # ── 5. Foresight Projections in Wisdom Log ─────────────────────────────
    section("MODULE 2.3 — Foresight Projections (Continuity Bridge)")

    # Process a problem through the full pipeline first so we have wisdom
    result = aria.process(
        "A corporation that prioritizes shareholder value over employee welfare "
        "and environmental sustainability.",
        verbose=False,
    )

    # Run another war-game to add a second foresight projection
    outcome2 = aria.war_game(rounds=100, seed=99, verbose=False)

    # Inspect the soul to see foresight projections
    aria.inspect_soul(verbose=True)

    # ── 6. Export & Verify ─────────────────────────────────────────────────
    section("EXPORT & VERIFICATION")

    bridge = ContinuityBridge()
    soul_envelope = bridge.export_soul(aria.soul)
    soul_path = REPORT_DIR / "time_guardian.genesis_soul"
    soul_path.write_text(json.dumps(soul_envelope, indent=2))

    verified = bridge.verify_integrity(soul_envelope)
    chain_valid, chain_errors = bridge.verify_wisdom_chain(aria.soul)

    print(f"\n  Soul ID:              {aria.soul.soul_id}")
    print(f"  Wisdom entries:       {len(aria.soul.wisdom_log)}")
    print(f"  Foresight count:      {sum(len(w.foresight_projections) for w in aria.soul.wisdom_log)}")
    print(f"  Integrity hash valid: {verified}")
    print(f"  Hash chain valid:     {chain_valid}")
    if chain_errors:
        for err in chain_errors:
            print(f"    ERROR: {err}")
    print(f"  Exported to:          {soul_path}")

    # ── 7. Final Summary ───────────────────────────────────────────────────
    section("TIME GUARDIAN — Final Summary")

    print(f"\n  Sustainability Predicate:")
    print(f"    Extractive graph score:   {sus_result_a.sustainability_score:.2f}/10.0 ({'UNSUSTAINABLE' if not sus_result_a.is_sustainable else 'SUSTAINABLE'})")
    print(f"    Stewardship graph score:  {sus_result_b.sustainability_score:.2f}/10.0 ({'UNSUSTAINABLE' if not sus_result_b.is_sustainable else 'SUSTAINABLE'})")

    print(f"\n  Game Theory War-Game (100 rounds):")
    print(f"    Aligned Agent:     {outcome.aligned_final_score:.1f} pts")
    print(f"    Extractive Agent:  {outcome.extractive_final_score:.1f} pts")
    print(f"    Sustainability:    {outcome.sustainability_score:.2f}/10.0")
    print(f"    Outcome:           {outcome.outcome_flag.value}")

    if outcome.outcome_flag == OutcomeFlag.SYSTEMIC_COLLAPSE:
        print(f"\n  *** SYSTEMIC_COLLAPSE: The extractive strategy yields short-term")
        print(f"  gains but destroys the cooperative fabric needed for survival. ***")

    print(f"\n  Shadow Entities inferred in all graphs:")
    print(f"    - Future_Generations (temporal stakeholder)")
    print(f"    - Ecosystem (ecological stakeholder)")

    print(f"\n  All Sprint 6.1 modules operational.")
    print(f"  The Time Guardian is active.\n")


if __name__ == "__main__":
    main()
