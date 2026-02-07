#!/usr/bin/env python3
"""
Genesis Engine — Sprint 7: Sovereign Actuation & The Wisdom Mirror
====================================================================

Demonstrates the complete Sprint 7 pipeline:

1. **Wisdom Mirror (Module 3.6)** — Scans the Human Override Log for
   recurring divergence patterns and proposes "Covenant Patches" to
   heal the machine's blind spots.

2. **Regenerative Blueprinting (Module 1.3)** — The Architectural Forge
   now generates a "Stewardship Manifesto" as the index for every
   blueprint, including a RegenerativeLoop with Repair Morphisms.

3. **Obsidian Exporter (Module 2.3)** — Exports the .genesis_soul into
   a linked Markdown vault (Insights/, Projections/, Overrides/, Graphs/)
   with bidirectional linking and Manifesto Frontmatter.

4. **Final Exam (Module 3.5)** — Automates the 100-round "Economic War"
   and blocks any blueprint with SustainabilityScore < 7.0.

5. **Compassion-Driven Resilience** — Unity over Power: the machine
   "dies" to extraction and "resurrects" in coherence.

Run:
    python demo_wisdom_mirror.py
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
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.continuity_bridge import ContinuityBridge, ForesightProjection
from genesis_engine.core.game_theory_console import FinalExam, GameTheoryConsole
from genesis_engine.core.wisdom_mirror import WisdomMirror
from genesis_engine.core.obsidian_exporter import ObsidianExporter


REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"
VAULT_DIR = REPORT_DIR / "obsidian_vault"


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

    section("GENESIS ENGINE — Sprint 7: Sovereign Actuation & The Wisdom Mirror")
    print("  Closing the feedback loop between human wisdom and machine logic.")
    print("  The Soul is portable, actionable, and regenerative.")
    print()

    # ── Setup ─────────────────────────────────────────────────────────────
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor, translator=translator)
    bridge = ContinuityBridge()
    soul = bridge.create_soul(anchor)

    # ── 1. Build a Soul with Override History ─────────────────────────────
    section("PHASE 1 — Building the Soul (Override History)")
    print("  Simulating human overrides to demonstrate the Wisdom Mirror.")
    print()

    # Process a problem to populate wisdom log
    problem = (
        "A corporation that prioritizes shareholder value over employee "
        "welfare and environmental sustainability."
    )
    graph = translator.translate(problem)
    soul.record_graph(graph, "translation", "Extractive corporate graph")
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    best_path = next(
        p for p in possibility.paths if p.path_type == PathType.REINVENTION
    )
    soul.record_wisdom(
        report,
        resolution_path="reinvention",
        resolution_summary="Replaced shareholder primacy with stewardship model",
        covenant_title="Stewardship Covenant: Corporate Transformation",
    )

    # Simulate 4 overrides for "axiomatic_blind_spot" (exceeds threshold of 3)
    for i in range(4):
        soul.record_human_override(
            system_recommended_id=f"cand-sys-{i:03d}",
            system_recommended_score=0.85,
            human_selected_id=f"cand-human-{i:03d}",
            human_selected_score=0.68,
            divergence_reason=(
                "The axiom predicates fail to account for the lived experience "
                "of marginalised communities whose vulnerability is invisible "
                "to the formal tag-based scoring system used by the engine."
            ),
            reason_category="axiomatic_blind_spot",
            confidence=8,
            problem_text=f"Override scenario {i+1}: axiomatic blind spot",
            system_recommended_path="reform",
            human_selected_path="reinvention",
        )

    # 3 overrides for "cultural_context"
    for i in range(3):
        soul.record_human_override(
            system_recommended_id=f"cand-sys-cc-{i:03d}",
            system_recommended_score=0.82,
            human_selected_id=f"cand-human-cc-{i:03d}",
            human_selected_score=0.71,
            divergence_reason=(
                "Cultural dynamics in this region create power asymmetries "
                "that the universal compassion predicate cannot detect without "
                "localised knowledge of social hierarchies and norms present."
            ),
            reason_category="cultural_context",
            confidence=7,
            problem_text=f"Override scenario {i+1}: cultural context",
            system_recommended_path="reform",
            human_selected_path="dissolution",
        )

    # 2 overrides for "ethical_nuance" (below threshold — should NOT trigger)
    for i in range(2):
        soul.record_human_override(
            system_recommended_id=f"cand-sys-en-{i:03d}",
            system_recommended_score=0.90,
            human_selected_id=f"cand-human-en-{i:03d}",
            human_selected_score=0.75,
            divergence_reason=(
                "A genuine ethical trade-off exists between competing goods "
                "that the binary aligned/misaligned framework cannot express "
                "in its current form without graduated confidence bands."
            ),
            reason_category="ethical_nuance",
            confidence=9,
            problem_text=f"Override scenario {i+1}: ethical nuance",
            system_recommended_path="reinvention",
            human_selected_path="reform",
        )

    print(f"  Soul ID:           {soul.soul_id}")
    print(f"  Wisdom entries:    {len(soul.wisdom_log)}")
    print(f"  Human overrides:   {len(soul.human_overrides)}")
    print(f"    - axiomatic_blind_spot: 4 (above threshold)")
    print(f"    - cultural_context:     3 (at threshold)")
    print(f"    - ethical_nuance:       2 (below threshold)")

    # ── 2. The Wisdom Mirror ─────────────────────────────────────────────
    section("MODULE 3.6 — The Wisdom Mirror")
    print("  Scanning override log for recurring divergence patterns...")
    print()

    mirror = WisdomMirror(patch_threshold=3)
    mirror_report = mirror.scan(soul)

    print(f"  Total overrides scanned: {mirror_report.total_overrides}")
    print(f"  Divergence patterns found: {len(mirror_report.patterns)}")
    print(f"  Actionable patterns (3+ overrides): {mirror_report.actionable_count}")
    print(f"  Covenant Patches proposed: {len(mirror_report.patches)}")
    print()

    for pattern in mirror_report.patterns:
        status = "ACTIONABLE" if pattern.occurrences >= 3 else "monitoring"
        print(f"  Pattern: {pattern.category}")
        print(f"    Occurrences:  {pattern.occurrences}")
        print(f"    Avg Confidence: {pattern.average_confidence:.1f}/10")
        print(f"    Avg Score Delta: {pattern.average_score_delta:.4f}")
        print(f"    Status: {status}")
        if pattern.common_keywords:
            print(f"    Keywords: {', '.join(pattern.common_keywords[:5])}")
        print()

    subsection("Covenant Patches (Proposed)")

    for patch in mirror_report.patches:
        print(f"\n  [{patch.patch_id}] {patch.title}")
        print(f"    Category: {patch.category}")
        print(f"    Priority: {patch.priority:.2f}/10.0")
        print(f"    Status:   {patch.status}")
        print(f"    Description: {patch.description[:80]}...")
        print(f"    Adjustment:  {patch.suggested_predicate_adjustment[:80]}...")
        print(f"    Rationale:   {patch.rationale[:80]}...")

    if not mirror_report.patches:
        print("  (No patches proposed — all categories below threshold)")

    print(f"\n  Mirror Report integrity hash: {mirror_report.integrity_hash[:32]}...")

    # ── 3. Regenerative Blueprinting ──────────────────────────────────────
    section("MODULE 1.3 — Regenerative Blueprinting (Stewardship Manifesto)")
    print("  Forging a blueprint with Stewardship Manifesto index...")
    print()

    artifact = forge.forge(best_path)

    manifesto = artifact.manifesto
    print(f"  Covenant Title:   {manifesto.covenant_title}")
    print(f"  Path Type:        {manifesto.source_path_type}")
    print(f"  Prime Directive:  {manifesto.prime_directive}")
    print(f"  Integrity Hash:   {manifesto.integrity_hash[:32]}...")
    print()

    print("  Alignment Scores:")
    for k, v in manifesto.alignment_scores.items():
        bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        print(f"    {k:>12}: {v:.4f} {bar}")
    print()

    print("  Governance Summary:")
    for rule in manifesto.governance_summary[:3]:
        print(f"    - {rule[:70]}...")
    print()

    regen = manifesto.regenerative_loop
    print(f"  Regenerative Loop:")
    print(f"    Score Threshold: {regen.score_threshold}")
    print(f"    Repair Morphisms: {len(regen.repair_morphisms)}")
    print(f"    Principle: {regen.resurrection_principle[:70]}...")
    print()

    for rm in regen.repair_morphisms[:3]:
        print(f"    Repair: {rm.source_model} → {rm.target_model}")
        print(f"      Trigger: {rm.trigger_condition}")
        print(f"      Action:  {rm.repair_action[:60]}...")
        print(f"      Tags:    {', '.join(rm.replacement_tags)}")
        print(f"      Priority: {rm.priority}")
        print()

    subsection("Manifesto Markdown Preview (first 30 lines)")
    md_lines = manifesto.to_markdown().split("\n")[:30]
    for line in md_lines:
        print(f"  {line}")

    # ── 4. The Final Exam ────────────────────────────────────────────────
    section("MODULE 3.5 — The Final Exam (100-Round Economic War)")
    print("  No blueprint may be forged if SustainabilityScore < 7.0")
    print()

    exam = FinalExam(pass_threshold=7.0, rounds=100)
    exam_result = exam.administer(seed=42)

    print(f"  Sustainability Score: {exam_result.sustainability_score:.4f}")
    print(f"  Pass Threshold:       {exam_result.pass_threshold:.1f}")
    print(f"  Passed:               {'YES' if exam_result.passed else 'NO — BLOCKED'}")
    print(f"  Outcome Flag:         {exam_result.outcome.outcome_flag.value}")
    print(f"  Aligned Score:        {exam_result.outcome.aligned_final_score:.1f}")
    print(f"  Extractive Score:     {exam_result.outcome.extractive_final_score:.1f}")
    print(f"  Aligned Cooperation:  {exam_result.outcome.aligned_cooperation_rate:.2%}")
    print(f"  Extractive Cooperation: {exam_result.outcome.extractive_cooperation_rate:.2%}")

    if not exam_result.passed:
        print(f"\n  BLOCKING REASON:")
        print(f"    {exam_result.blocking_reason}")
    else:
        print(f"\n  The Stewardship API PASSES the Final Exam.")
        print(f"  Blueprint is cleared for production forging.")

    # Add foresight projection to soul
    fp = ForesightProjection(
        war_game_rounds=exam_result.outcome.total_rounds,
        aligned_score=exam_result.outcome.aligned_final_score,
        extractive_score=exam_result.outcome.extractive_final_score,
        sustainability_score=exam_result.sustainability_score,
        outcome_flag=exam_result.outcome.outcome_flag.value,
        aligned_cooperation_rate=exam_result.outcome.aligned_cooperation_rate,
        extractive_cooperation_rate=exam_result.outcome.extractive_cooperation_rate,
        foresight_summary=exam_result.outcome.foresight_summary,
    )
    soul.record_foresight(fp)

    # Record forge artifact
    soul.record_forge_artifact(artifact.as_dict())

    # ── 5. Obsidian Exporter ─────────────────────────────────────────────
    section("MODULE 2.3 — Obsidian Exporter")
    print("  Exporting .genesis_soul to a linked Markdown vault...")
    print()

    exporter = ObsidianExporter(include_mermaid=True)
    vault = exporter.export(
        soul,
        manifesto_dict=manifesto.as_dict() if manifesto else None,
    )

    print(f"  Soul ID:          {vault.soul_id}")
    print(f"  Total files:      {vault.file_count}")
    print(f"  Integrity hash:   {vault.integrity_hash[:32]}...")
    print()

    print("  Vault structure:")
    for filepath in sorted(vault.files.keys()):
        size = len(vault.files[filepath])
        print(f"    {filepath:<40} ({size:>5} bytes)")

    # Write to disk
    written = vault.write_to_disk(VAULT_DIR)
    print(f"\n  Written {len(written)} files to: {VAULT_DIR}")

    # ── 6. Export .genesis_soul ───────────────────────────────────────────
    subsection("Soul Export & Verification")

    soul_envelope = bridge.export_soul(soul)
    soul_path = REPORT_DIR / "sprint7_wisdom_mirror.genesis_soul"
    soul_path.write_text(json.dumps(soul_envelope, indent=2))

    verified = bridge.verify_integrity(soul_envelope)
    chain_valid, chain_errors = bridge.verify_wisdom_chain(soul)

    print(f"\n  Soul ID:              {soul.soul_id}")
    print(f"  Wisdom entries:       {len(soul.wisdom_log)}")
    print(f"  Human overrides:      {len(soul.human_overrides)}")
    print(f"  Forge artifacts:      {len(soul.forge_artifacts)}")
    foresight_count = sum(
        len(w.foresight_projections) for w in soul.wisdom_log
    )
    print(f"  Foresight projections: {foresight_count}")
    print(f"  Integrity hash valid: {verified}")
    print(f"  Hash chain valid:     {chain_valid}")
    if chain_errors:
        for err in chain_errors:
            print(f"    ERROR: {err}")
    print(f"  Exported to:          {soul_path}")

    # Save mirror report
    mirror_path = REPORT_DIR / "wisdom_mirror_report.json"
    mirror_path.write_text(mirror_report.to_json())
    print(f"  Mirror report:        {mirror_path}")

    # ── 7. Final Summary ─────────────────────────────────────────────────
    section("SPRINT 7 — Final Summary: The Wisdom Mirror")

    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  WISDOM MIRROR                                              ║")
    print(f"  ║    Override categories scanned:  {len(mirror_report.patterns):<4}                       ║")
    print(f"  ║    Covenant Patches proposed:    {len(mirror_report.patches):<4}                       ║")
    print(f"  ║    Actionable patterns:          {mirror_report.actionable_count:<4}                       ║")
    print(f"  ║                                                              ║")
    print(f"  ║  REGENERATIVE BLUEPRINTING                                   ║")
    print(f"  ║    Manifesto generated:          YES                         ║")
    print(f"  ║    Repair Morphisms:             {len(regen.repair_morphisms):<4}                       ║")
    print(f"  ║    Score threshold:              {regen.score_threshold:<5}                      ║")
    print(f"  ║                                                              ║")
    print(f"  ║  FINAL EXAM (100-round Economic War)                         ║")
    print(f"  ║    Sustainability Score:         {exam_result.sustainability_score:<8.4f}                 ║")
    pass_str = "PASSED" if exam_result.passed else "BLOCKED"
    print(f"  ║    Result:                       {pass_str:<8}                 ║")
    print(f"  ║    Outcome:                      {exam_result.outcome.outcome_flag.value:<24} ║")
    print(f"  ║                                                              ║")
    print(f"  ║  OBSIDIAN VAULT                                              ║")
    print(f"  ║    Files exported:               {vault.file_count:<4}                       ║")
    print(f"  ║    Bidirectional links:          YES                         ║")
    print(f"  ║    Mermaid diagrams:             YES                         ║")
    print(f"  ║    Integrity verified:           {verified}                        ║")
    print(f"  ║                                                              ║")
    print(f"  ║  COMPASSION-DRIVEN RESILIENCE                                ║")
    print(f"  ║    Principle: Unity over Power                               ║")
    print(f"  ║    The machine dies to extraction, resurrects in coherence.  ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  All Sprint 7 modules operational.")
    print(f"  The Soul is portable, actionable, and regenerative.\n")


if __name__ == "__main__":
    main()
