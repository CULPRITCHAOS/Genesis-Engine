#!/usr/bin/env python3
"""
Genesis Engine — Sprint 2: The Harmonization Flow
==================================================

Demonstrates the full pipeline:

    Natural-language problem
        → AxiomLogix Translator   (categorical graph)
        → Axiom Anchor            (validation)
        → Deconstruction Engine   (disharmony report)
        → Dream Engine            (threefold path solutions)
        → Recursive Validation    (Axiom Anchor re-check)
        → Possibility Report      (JSON output)

Run:
    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine


# ---------------------------------------------------------------------------
# Sample problem statements
# ---------------------------------------------------------------------------

PROBLEMS: list[str] = [
    "A corporate policy that prioritizes profit over user safety.",
    "A community health program that empowers workers and protects the environment.",
    "An AI surveillance system that exploits user data and neglects privacy.",
]

REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def run_pipeline(
    problem: str,
    index: int,
    directive: PrimeDirective,
    anchor: AxiomAnchor,
    translator: AxiomLogixTranslator,
    deconstruction: DeconstructionEngine,
    dream: DreamEngine,
) -> tuple[dict, dict | None]:
    """Execute the full Harmonization Flow on a single problem.

    Returns (disharmony_report_dict, possibility_report_dict_or_None).
    """
    print(f"\n{'='*72}")
    print(f"  PROBLEM {index + 1}: {problem}")
    print(f"{'='*72}")

    print(f"\n  Prime Directive: \"{directive.statement}\"")
    print(f"  Principles: {[p.value for p in directive.principles]}")

    # ── Phase 1: Translate ─────────────────────────────────────────────
    graph = translator.translate(problem)
    print(f"\n  [AxiomLogix] Extracted {len(graph.objects)} objects, "
          f"{len(graph.morphisms)} morphisms.")
    for obj in graph.objects:
        print(f"    Object: {obj.label:20s}  tags={obj.tags}")
    for mor in graph.morphisms:
        src = mor.source
        tgt = mor.target
        for o in graph.objects:
            if o.id == mor.source:
                src = o.label
            if o.id == mor.target:
                tgt = o.label
        print(f"    Morphism: {mor.label:20s}  {src} -> {tgt}  tags={mor.tags}")

    # ── Phase 2: Deconstruct ───────────────────────────────────────────
    report = deconstruction.analyse(graph)

    print(f"\n  [Axiom Anchor] Aligned: {report.is_aligned}")
    print(f"  [Report] Unity Impact:       {report.unity_impact}/10")
    print(f"  [Report] Compassion Deficit: {report.compassion_deficit}/10")
    print(f"  [Report] Coherence Score:    {report.coherence_score}/10")

    if report.findings:
        print(f"\n  Findings:")
        for finding in report.findings:
            status = "DISHARMONY" if finding.disharmony_score > 0 else "OK"
            print(f"    [{status}] {finding.label} "
                  f"({finding.source} -> {finding.target}) "
                  f"score={finding.disharmony_score:.2f}")
            if finding.recommendation:
                print(f"             Rec: {finding.recommendation}")

    # ── Phase 3: Dream (only for misaligned scenarios) ─────────────────
    possibility_dict = None
    if not report.is_aligned:
        print(f"\n  {'─'*68}")
        print(f"  [Dream Engine] Disharmony detected — generating Threefold Path...")
        print(f"  {'─'*68}")

        possibility = dream.dream(report, graph)

        for path in possibility.paths:
            aligned_str = "YES" if (path.validation and path.validation.is_aligned) else "NO"
            print(f"\n  >> {path.title} ({path.path_type.value})")
            print(f"     {path.description}")
            print(f"     Unity Alignment : {path.unity_alignment_score:.4f}")
            print(f"     Feasibility     : {path.feasibility_score:.4f}")
            print(f"     Anchor Aligned  : {aligned_str}")
            print(f"     Healed Graph    : {len(path.healed_graph.objects)} objects, "
                  f"{len(path.healed_graph.morphisms)} morphisms")
            for m in path.healed_graph.morphisms:
                src_l = m.source
                tgt_l = m.target
                for o in path.healed_graph.objects:
                    if o.id == m.source:
                        src_l = o.label
                    if o.id == m.target:
                        tgt_l = o.label
                print(f"       {m.label:30s} {src_l} -> {tgt_l}  {m.tags}")

        print(f"\n  [Dream Engine] Recommended: {possibility.recommended_path}")
        possibility_dict = possibility.as_dict()
    else:
        print(f"\n  System is aligned — no Dream Engine intervention needed.")

    return report.as_dict(), possibility_dict


def main() -> None:
    """Run all sample problems through the full Harmonization Flow."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared components (single Axiom Anchor instance for the whole run).
    directive = PrimeDirective()
    anchor = AxiomAnchor(directive=directive, alignment_threshold=0.5)
    translator = AxiomLogixTranslator()
    deconstruction = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)

    all_disharmony: list[dict] = []
    all_possibility: list[dict] = []

    for i, problem in enumerate(PROBLEMS):
        dis_dict, pos_dict = run_pipeline(
            problem, i, directive, anchor, translator, deconstruction, dream,
        )
        all_disharmony.append(dis_dict)

        # Write disharmony report.
        dis_path = REPORT_DIR / f"disharmony_report_{i + 1}.json"
        dis_path.write_text(json.dumps(dis_dict, indent=2))
        print(f"\n  Disharmony report  -> {dis_path}")

        # Write possibility report if generated.
        if pos_dict is not None:
            all_possibility.append(pos_dict)
            pos_path = REPORT_DIR / f"possibility_report_{i + 1}.json"
            pos_path.write_text(json.dumps(pos_dict, indent=2))
            print(f"  Possibility report -> {pos_path}")

    # Combined outputs.
    (REPORT_DIR / "disharmony_reports_combined.json").write_text(
        json.dumps(all_disharmony, indent=2),
    )
    if all_possibility:
        (REPORT_DIR / "possibility_reports_combined.json").write_text(
            json.dumps(all_possibility, indent=2),
        )

    print(f"\n{'='*72}")
    print(f"  All reports written to {REPORT_DIR}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
