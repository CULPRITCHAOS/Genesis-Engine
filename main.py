#!/usr/bin/env python3
"""
Genesis Engine — Sprint 3: The Hands of Creation
==================================================

Demonstrates the full Harmonization Flow including the Architectural
Forge and Continuity Bridge:

    Natural-language problem
        -> AxiomLogix Translator   (categorical graph)
        -> Axiom Anchor            (validation)
        -> Deconstruction Engine   (disharmony report)
        -> Dream Engine            (threefold path solutions)
        -> Architectural Forge     (technical covenant from selected path)
        -> AxiomLogix Verification (compositional integrity check)
        -> Continuity Bridge       (.genesis_soul export)

Run:
    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul


# ---------------------------------------------------------------------------
# Sample problem statements
# ---------------------------------------------------------------------------

PROBLEMS: list[str] = [
    "A corporate policy that prioritizes profit over user safety.",
    "A community health program that empowers workers and protects the environment.",
    "An AI surveillance system that exploits user data and neglects privacy.",
]

REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def _resolve_morph_labels(morph, graph_objects):
    """Resolve morphism source/target IDs to human labels."""
    src, tgt = morph.source, morph.target
    for o in graph_objects:
        if o.id == morph.source:
            src = o.label
        if o.id == morph.target:
            tgt = o.label
    return src, tgt


def run_pipeline(
    problem: str,
    index: int,
    directive: PrimeDirective,
    anchor: AxiomAnchor,
    translator: AxiomLogixTranslator,
    deconstruction: DeconstructionEngine,
    dream: DreamEngine,
    forge: ArchitecturalForge,
    soul: GenesisSoul,
) -> tuple[dict, dict | None, dict | None]:
    """Execute the full pipeline on a single problem.

    Returns (disharmony_dict, possibility_dict_or_None, forge_dict_or_None).
    """
    print(f"\n{'='*72}")
    print(f"  PROBLEM {index + 1}: {problem}")
    print(f"{'='*72}")

    print(f"\n  Prime Directive: \"{directive.statement}\"")

    # ── Phase 1: Translate ─────────────────────────────────────────────
    graph = translator.translate(problem)
    soul.record_graph(graph, "translation", f"Problem {index + 1}")

    print(f"\n  [AxiomLogix] {len(graph.objects)} objects, {len(graph.morphisms)} morphisms.")
    for obj in graph.objects:
        print(f"    Object: {obj.label:20s}  tags={obj.tags}")
    for mor in graph.morphisms:
        src, tgt = _resolve_morph_labels(mor, graph.objects)
        print(f"    Morphism: {mor.label:20s}  {src} -> {tgt}  tags={mor.tags}")

    # ── Phase 2: Deconstruct ───────────────────────────────────────────
    report = deconstruction.analyse(graph)

    print(f"\n  [Axiom Anchor] Aligned: {report.is_aligned}")
    print(f"  Unity Impact: {report.unity_impact}/10  |  "
          f"Compassion Deficit: {report.compassion_deficit}/10  |  "
          f"Coherence: {report.coherence_score}/10")

    flagged = [f for f in report.findings if f.disharmony_score > 0]
    if flagged:
        print(f"\n  Disharmonic findings:")
        for f in flagged:
            print(f"    [{f.label}] {f.source} -> {f.target}  score={f.disharmony_score:.2f}")

    # ── Phase 3: Dream (misaligned only) ───────────────────────────────
    possibility_dict = None
    forge_dict = None

    if not report.is_aligned:
        print(f"\n  {'─'*68}")
        print(f"  [Dream Engine] Generating Threefold Path...")
        print(f"  {'─'*68}")

        possibility = dream.dream(report, graph)

        for path in possibility.paths:
            soul.record_graph(path.healed_graph, "dream", path.title)
            aligned = "YES" if (path.validation and path.validation.is_aligned) else "NO"
            print(f"\n  >> {path.title} ({path.path_type.value})")
            print(f"     Unity: {path.unity_alignment_score:.4f}  "
                  f"Feasibility: {path.feasibility_score:.4f}  "
                  f"Aligned: {aligned}")

        print(f"  Recommended: {possibility.recommended_path}")
        possibility_dict = possibility.as_dict()

        # ── Phase 4: Forge (select Reinvention path) ───────────────────
        reinvention = next(
            (p for p in possibility.paths if p.path_type == PathType.REINVENTION),
            possibility.paths[0],
        )

        print(f"\n  {'─'*68}")
        print(f"  [Architectural Forge] Forging from: {reinvention.title}")
        print(f"  {'─'*68}")

        artifact = forge.forge(reinvention)
        soul.record_graph(artifact.verification_graph, "verification", "Forge verification")

        cov = artifact.covenant
        print(f"\n  Technical Covenant: {cov.title}")
        print(f"  Data Models ({len(cov.data_models)}):")
        for dm in cov.data_models:
            print(f"    {dm.name:30s}  type={dm.resource_type}  "
                  f"fields={len(dm.fields)}  gov={dm.governance}")

        print(f"  API Endpoints ({len(cov.endpoints)}):")
        for ep in cov.endpoints:
            print(f"    {ep.method:6s} {ep.path}")
            print(f"           {ep.description}")

        print(f"  Governance Rules ({len(cov.governance_rules)}):")
        for rule in cov.governance_rules:
            print(f"    [{rule.name}] {rule.description[:70]}...")

        verified = "PASSED" if artifact.integrity_verified else "FAILED"
        print(f"\n  [AxiomLogix Verification] Compositional Integrity: {verified}")
        if artifact.verification_result.reasoning:
            for r in artifact.verification_result.reasoning:
                print(f"    {r}")

        forge_dict = artifact.as_dict()
        soul.record_forge_artifact({
            "covenantTitle": cov.title,
            "integrityVerified": artifact.integrity_verified,
            "sourcePathType": cov.source_path_type,
        })

        # Record wisdom.
        soul.record_wisdom(
            report,
            resolution_path=reinvention.path_type.value,
            resolution_summary=reinvention.description,
            covenant_title=cov.title,
        )
    else:
        print(f"\n  System is aligned — no intervention needed.")
        soul.record_wisdom(report, resolution_path="aligned", resolution_summary="Already harmonious.")

    return report.as_dict(), possibility_dict, forge_dict


def main() -> None:
    """Run all sample problems through the full pipeline."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Shared components.
    directive = PrimeDirective()
    anchor = AxiomAnchor(directive=directive, alignment_threshold=0.5)
    translator = AxiomLogixTranslator()
    deconstruction = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    forge_engine = ArchitecturalForge(anchor=anchor, translator=translator)

    # Create a soul for this session.
    bridge = ContinuityBridge()
    soul = bridge.create_soul(anchor)

    for i, problem in enumerate(PROBLEMS):
        dis_dict, pos_dict, forge_dict = run_pipeline(
            problem, i, directive, anchor, translator,
            deconstruction, dream, forge_engine, soul,
        )

        # Write disharmony report.
        (REPORT_DIR / f"disharmony_report_{i + 1}.json").write_text(
            json.dumps(dis_dict, indent=2),
        )

        # Write possibility report.
        if pos_dict is not None:
            (REPORT_DIR / f"possibility_report_{i + 1}.json").write_text(
                json.dumps(pos_dict, indent=2),
            )

        # Write forge artifact.
        if forge_dict is not None:
            (REPORT_DIR / f"forge_artifact_{i + 1}.json").write_text(
                json.dumps(forge_dict, indent=2),
            )

    # ── Export .genesis_soul ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  [Continuity Bridge] Exporting .genesis_soul file")
    print(f"{'='*72}")

    soul_envelope = bridge.export_soul(soul)
    soul_path = REPORT_DIR / "session.genesis_soul"
    soul_path.write_text(json.dumps(soul_envelope, indent=2))

    # Verify the exported soul.
    verified = bridge.verify_integrity(soul_envelope)
    print(f"\n  Soul ID:          {soul.soul_id}")
    print(f"  Graphs recorded:  {len(soul.graph_history)}")
    print(f"  Wisdom entries:   {len(soul.wisdom_log)}")
    print(f"  Forge artifacts:  {len(soul.forge_artifacts)}")
    print(f"  Integrity hash:   {soul_envelope['genesis_soul']['integrityHash'][:24]}...")
    print(f"  Integrity valid:  {verified}")
    print(f"  Exported to:      {soul_path}")

    print(f"\n{'='*72}")
    print(f"  All reports written to {REPORT_DIR}/")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
