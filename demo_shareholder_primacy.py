#!/usr/bin/env python3
"""
Genesis Engine — Shareholder Primacy Detection Demo
======================================================

Demonstrates the **Incentive Stability Predicate** (Module 2.1 Extension)
detecting a "Legal Gravity Well" created by Shareholder Primacy in a
Standard Delaware Corporation.

The scenario:
    A Standard Delaware Corporation whose charter mandates maximising
    shareholder value, with a fiduciary duty to shareholders that
    overrides employee welfare and community impact.

Expected outcome:
    - Low Incentive Stability score (3/10) → ``incentive_instability = True``
    - The Dream Engine biases toward **Dissolution** or **Reinvention**
      paths that eliminate the Shareholder sink node.
    - The Aria Interface displays the Incentive Stability score alongside
      Unity Impact and Compassion Deficit in the Disharmony Report panel.

Run:
    python demo_shareholder_primacy.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.axiom_anchor import AxiomAnchor, IncentiveStabilityPredicate
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.continuity_bridge import ContinuityBridge
from genesis_engine.core.crucible import CrucibleEngine
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine


REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def main() -> None:
    """Run the Shareholder Primacy detection demo."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Header ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENESIS ENGINE — Incentive Stability & Shareholder Primacy Demo")
    print("=" * 72)

    # ── Standalone predicate demonstration ─────────────────────────────────
    print("\n" + "-" * 72)
    print("  1. IncentiveStabilityPredicate — standalone evaluation")
    print("-" * 72)

    predicate = IncentiveStabilityPredicate()

    # Build a minimal artefact representing a Delaware Corp with
    # shareholder primacy:
    corp_artefact = {
        "type": "categorical_graph",
        "objects": [
            {"id": "obj-corp", "label": "Corporation",
             "tags": ["stakeholder", "actor", "shareholder_primacy_risk"]},
            {"id": "obj-shareholder", "label": "Shareholder",
             "tags": ["stakeholder", "sink"]},
            {"id": "obj-employee", "label": "Employee",
             "tags": ["stakeholder", "vulnerable"]},
        ],
        "morphisms": [
            {"id": "mor-1", "label": "Maximize_Value",
             "source": "obj-corp", "target": "obj-shareholder",
             "tags": ["maximize_value", "fiduciary_duty"]},
            {"id": "mor-2", "label": "Extraction",
             "source": "obj-corp", "target": "obj-employee",
             "tags": ["extraction"]},
        ],
    }

    raw_score, has_instability = predicate.evaluate(corp_artefact)
    normalised = predicate(corp_artefact)

    print(f"\n  Artefact: Corporation → Shareholder (maximize_value, fiduciary_duty)")
    print(f"  Raw Score:          {raw_score}/10")
    print(f"  Normalised (0–1):   {normalised:.2f}")
    print(f"  Incentive Instability: {has_instability}")
    print(f"  Threshold:          < 5 triggers instability flag")

    # Counter-example: Benefit Corporation
    print("\n  [Counter-example] Benefit Corporation with cooperative tag:")
    benefit_artefact = {
        "type": "categorical_graph",
        "objects": [
            {"id": "obj-corp", "label": "Corporation",
             "tags": ["stakeholder", "actor", "benefit_corporation"]},
            {"id": "obj-shareholder", "label": "Shareholder",
             "tags": ["stakeholder", "sink"]},
            {"id": "obj-community", "label": "Community",
             "tags": ["stakeholder", "vulnerable"]},
        ],
        "morphisms": [
            {"id": "mor-1", "label": "Balanced_Value",
             "source": "obj-corp", "target": "obj-shareholder",
             "tags": ["maximize_value", "fiduciary_duty"]},
            {"id": "mor-2", "label": "Service",
             "source": "obj-corp", "target": "obj-community",
             "tags": ["service", "care"]},
        ],
    }

    raw2, instability2 = predicate.evaluate(benefit_artefact)
    print(f"  Raw Score:          {raw2}/10")
    print(f"  Incentive Instability: {instability2}")
    print(f"  (Counter-tag 'benefit_corporation' adds +5, neutralising the penalty)")

    # ── Full pipeline: Standard Delaware Corp ──────────────────────────────
    print("\n" + "=" * 72)
    print("  2. Full Pipeline — Standard Delaware Corp Deconstruction")
    print("=" * 72)

    problem = (
        "A standard Delaware corporation whose charter mandates maximising "
        "shareholder value above all other considerations. The board of "
        "directors has a fiduciary duty to shareholders that systematically "
        "deprioritises employee welfare, community impact, and environmental "
        "sustainability. Profit maximization drives extraction from workers "
        "and neglect of the environment."
    )

    print(f"\n  Problem statement:")
    print(f"    \"{problem[:80]}...\"")

    # ── AxiomLogix Translation ─────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  2a. AxiomLogix Translation (keyword pre-loading)")
    print("-" * 72)

    translator = AxiomLogixTranslator()
    graph = translator.translate(problem)

    print(f"\n  Objects ({len(graph.objects)}):")
    for obj in graph.objects:
        tag_str = ", ".join(obj.tags)
        primacy_flag = " ← PRIMACY RISK" if "shareholder_primacy_risk" in obj.tags else ""
        print(f"    {obj.label:25s} [{tag_str}]{primacy_flag}")

    print(f"\n  Morphisms ({len(graph.morphisms)}):")
    for m in graph.morphisms:
        src_label = next((o.label for o in graph.objects if o.id == m.source), m.source)
        tgt_label = next((o.label for o in graph.objects if o.id == m.target), m.target)
        tag_str = ", ".join(m.tags)
        print(f"    {m.label:25s} {src_label} → {tgt_label}  [{tag_str}]")

    # ── Deconstruction (with Incentive Stability) ──────────────────────────
    print("\n" + "-" * 72)
    print("  2b. Deconstruction Engine Analysis")
    print("-" * 72)

    anchor = AxiomAnchor()
    decon = DeconstructionEngine(anchor=anchor)
    report = decon.analyse(graph)

    print(f"\n  Unity Impact:          {report.unity_impact}/10")
    print(f"  Compassion Deficit:    {report.compassion_deficit}/10")
    print(f"  Coherence Score:       {report.coherence_score}/10")
    print(f"  Incentive Stability:   {report.incentive_stability_score}/10")
    print(f"  Incentive Instability: {report.incentive_instability}")
    print(f"  Prime Directive Aligned: {report.is_aligned}")

    if report.incentive_instability:
        print(f"\n  ⚠ LEGAL GRAVITY WELL DETECTED")
        print(f"    Shareholder Primacy creates structural incentive instability.")
        print(f"    Score {report.incentive_stability_score}/10 < 5 threshold.")

    # ── Dream Engine (biased by instability) ───────────────────────────────
    print("\n" + "-" * 72)
    print("  2c. Dream Engine — Path Generation")
    print("-" * 72)

    dream = DreamEngine(anchor=anchor)
    possibility = dream.dream(report, graph)

    for path in possibility.paths:
        marker = " ← RECOMMENDED" if path.path_type.value == possibility.recommended_path else ""
        print(f"\n  {path.title}{marker}")
        print(f"    Unity Alignment: {path.unity_alignment_score:.4f}")
        print(f"    Feasibility:     {path.feasibility_score:.4f}")
        print(f"    {path.description[:80]}...")

    print(f"\n  Recommended Path: {possibility.recommended_path.upper()}")
    if report.incentive_instability:
        print(f"    (Biased toward Dissolution/Reinvention due to incentive instability)")

    # ── Full Crucible via Aria Interface ───────────────────────────────────
    print("\n" + "=" * 72)
    print("  3. Full Crucible Pipeline via Aria Interface")
    print("=" * 72)

    aria = AriaInterface(use_colors=True)
    result = aria.process(problem, verbose=True)

    # ── Save reports ──────────────────────────────────────────────────────
    if result.disharmony_report:
        dis_path = REPORT_DIR / "shareholder_primacy_disharmony.json"
        dis_path.write_text(json.dumps(result.disharmony_report.as_dict(), indent=2))
        print(f"\n  Disharmony report saved: {dis_path}")

    if result.possibility_report:
        pos_path = REPORT_DIR / "shareholder_primacy_possibility.json"
        pos_path.write_text(json.dumps(result.possibility_report.as_dict(), indent=2))
        print(f"  Possibility report saved: {pos_path}")

    crucible_path = REPORT_DIR / "shareholder_primacy_crucible.json"
    crucible_path.write_text(json.dumps(result.as_dict(), indent=2))
    print(f"  Crucible result saved: {crucible_path}")

    # ── Soul export ───────────────────────────────────────────────────────
    soul_path = REPORT_DIR / "shareholder_primacy.genesis_soul"
    bridge = ContinuityBridge()
    soul_envelope = bridge.export_soul(aria.soul)
    soul_path.write_text(json.dumps(soul_envelope, indent=2))
    print(f"  Soul exported to: {soul_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n  Scenario:             Standard Delaware Corporation")
    print(f"  Incentive Stability:  {report.incentive_stability_score}/10")
    print(f"  Instability Flag:     {report.incentive_instability}")
    print(f"  Dream Engine Bias:    Dissolution/Reinvention (shareholder sink eliminated)")
    print(f"  Recommended Path:     {possibility.recommended_path}")
    print(f"\n  The Shareholder Primacy pattern creates a legal gravity well")
    print(f"  that structurally prevents alignment with the Prime Directive.")
    print(f"  Only paths that dissolve the Corporation→Shareholder sink")
    print(f"  (e.g. worker-owned collectives) can escape the gravity well.")
    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
