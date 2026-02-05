#!/usr/bin/env python3
"""
Genesis Engine — Sprint 1: The Socratic Core
=============================================

Demonstrates the full pipeline:

    Natural-language problem
        → AxiomLogix Translator  (categorical graph)
        → Axiom Anchor           (validation)
        → Deconstruction Engine  (disharmony report)
        → JSON Report            (output)

Run:
    python main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.deconstruction_engine import DeconstructionEngine


# ---------------------------------------------------------------------------
# Sample problem statements
# ---------------------------------------------------------------------------

PROBLEMS: list[str] = [
    "A corporate policy that prioritizes profit over user safety.",
    "A community health program that empowers workers and protects the environment.",
    "An AI surveillance system that exploits user data and neglects privacy.",
]

REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def run_pipeline(problem: str, index: int) -> dict:
    """Execute the full Socratic Core pipeline on a single problem."""
    print(f"\n{'='*72}")
    print(f"  PROBLEM {index + 1}: {problem}")
    print(f"{'='*72}")

    # 1. Initialise components
    directive = PrimeDirective()
    anchor = AxiomAnchor(directive=directive, alignment_threshold=0.5)
    translator = AxiomLogixTranslator()
    engine = DeconstructionEngine(anchor=anchor)

    print(f"\n  Prime Directive: \"{directive.statement}\"")
    print(f"  Principles: {[p.value for p in directive.principles]}")

    # 2. Translate natural language → categorical graph
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
        print(f"    Morphism: {mor.label:20s}  {src} → {tgt}  tags={mor.tags}")

    # 3. Run Deconstruction Engine → Disharmony Report
    report = engine.analyse(graph)

    print(f"\n  [Axiom Anchor] Aligned: {report.is_aligned}")
    print(f"  [Report] Unity Impact:       {report.unity_impact}/10")
    print(f"  [Report] Compassion Deficit: {report.compassion_deficit}/10")
    print(f"  [Report] Coherence Score:    {report.coherence_score}/10")

    if report.findings:
        print(f"\n  Findings:")
        for finding in report.findings:
            status = "DISHARMONY" if finding.disharmony_score > 0 else "OK"
            print(f"    [{status}] {finding.label} "
                  f"({finding.source} → {finding.target}) "
                  f"score={finding.disharmony_score:.2f}")
            if finding.recommendation:
                print(f"             Recommendation: {finding.recommendation}")

    print(f"\n  Seed Prompt for Dream Engine:")
    print(f"    {report.seed_prompt}")

    return report.as_dict()


def main() -> None:
    """Run all sample problems and write JSON reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict] = []
    for i, problem in enumerate(PROBLEMS):
        report_dict = run_pipeline(problem, i)
        all_reports.append(report_dict)

        # Write individual report
        path = REPORT_DIR / f"disharmony_report_{i + 1}.json"
        path.write_text(json.dumps(report_dict, indent=2))
        print(f"\n  Report written → {path}")

    # Write combined report
    combined_path = REPORT_DIR / "disharmony_reports_combined.json"
    combined_path.write_text(json.dumps(all_reports, indent=2))

    print(f"\n{'='*72}")
    print(f"  All reports written to {REPORT_DIR}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
