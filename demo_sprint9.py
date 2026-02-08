#!/usr/bin/env python3
"""
Sprint 9 — Sovereign Synthesis & Vertical Actuation Demo

Demonstrates a full "Crystallization" event:
1. Load the Grid War 2026 scenario via load_conflict()
2. Inject a Water_Resources override node during Mirror of Truth phase
3. Run the full pipeline: Deconstruction → Dream Engine → Mirror of Truth
4. Pass the Final Exam (>7.5 via covenant-aware exam)
5. Crystallize into a new Obsidian vault with Stewardship Manifesto Frontmatter
6. Auto-sync to disk

This demo showcases the Genesis Engine as a repeatable Policy Analysis Tool,
finalized for local-first sovereignty.
"""

import tempfile
from pathlib import Path

from genesis_engine.core.ai_provider import get_default_provider, OffloadSkeleton
from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.continuity_bridge import ContinuityBridge, ForesightProjection
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine
from genesis_engine.core.game_theory_console import (
    BayesianFinalExam,
    CovenantFinalExam,
)
from genesis_engine.core.mirror_of_truth import MirrorOfTruth
from genesis_engine.core.obsidian_exporter import ObsidianExporter


def main() -> None:
    print("=" * 72)
    print("  GENESIS ENGINE — Sprint 9: Sovereign Synthesis")
    print("  Crystallization Event Demo")
    print("=" * 72)
    print()

    # -----------------------------------------------------------------------
    # 0. Initialize the system with local-first provider
    # -----------------------------------------------------------------------
    print("[0] Initializing with local-first provider...")
    provider = get_default_provider()
    print(f"    Provider: {provider.provider_name}")

    aria = AriaInterface(use_colors=True)
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    mirror = MirrorOfTruth(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor, translator=translator)
    exporter = ObsidianExporter(include_mermaid=True)
    bridge = ContinuityBridge()
    soul = bridge.create_soul(anchor)
    print()

    # -----------------------------------------------------------------------
    # 1. Load the Grid War 2026 conflict scenario
    # -----------------------------------------------------------------------
    print("[1] Loading Grid War 2026 scenario...")
    scenario, conflict_graph = aria.load_conflict(
        "scenarios/grid_war_2026.json",
        verbose=True,
    )
    print()

    # -----------------------------------------------------------------------
    # 2. Inject Water Resources override node
    # -----------------------------------------------------------------------
    print("[2] Injecting Water_Resources override node...")
    conflict_graph = aria.inject_node(
        graph=conflict_graph,
        label="Local_Innovation_Hub",
        tags=["stakeholder", "actor", "community", "innovation"],
        connect_to="Residential_Ratepayer",
        morphism_label="Community_Empowerment",
        morphism_tags=["empowerment", "collaboration", "local_benefit"],
        verbose=True,
    )
    print()

    # Record the conflict graph
    soul.record_graph(conflict_graph, "translation", "Grid War 2026 — loaded + injected")

    # -----------------------------------------------------------------------
    # 3. Run Deconstruction
    # -----------------------------------------------------------------------
    print("[3] Deconstructing conflict graph...")
    report = decon.analyse(conflict_graph)
    print(f"    Unity Impact: {report.unity_impact:.2f}")
    print(f"    Compassion Deficit: {report.compassion_deficit:.2f}")
    print(f"    Findings: {len(report.findings)}")
    print()

    # -----------------------------------------------------------------------
    # 4. Dream Engine — generate solution paths
    # -----------------------------------------------------------------------
    print("[4] Dream Engine generating solution paths...")
    possibility = dream.dream(report, conflict_graph)
    for path in possibility.paths:
        print(f"    {path.path_type.value}: unity={path.unity_alignment_score:.2f}, "
              f"feasibility={path.feasibility_score:.2f}")
    selected = possibility.paths[0]
    print(f"    Selected: {selected.path_type.value}")
    print()

    # -----------------------------------------------------------------------
    # 5. Mirror of Truth — adversarial critique
    # -----------------------------------------------------------------------
    print("[5] Mirror of Truth critique...")
    trace = mirror.critique(selected, report, conflict_graph)
    aria.refinement_panel(trace, verbose=True)
    print()

    # -----------------------------------------------------------------------
    # 6. Forge the covenant
    # -----------------------------------------------------------------------
    print("[6] Forging covenant...")
    artifact = forge.forge(selected)
    manifesto_dict = artifact.manifesto.as_dict() if artifact.manifesto else {}
    repair_morphisms = []
    if artifact.manifesto:
        for rm in artifact.manifesto.regenerative_loop.repair_morphisms:
            repair_morphisms.append(rm.as_dict())
    print(f"    Covenant: {artifact.covenant.title}")
    print(f"    Integrity Verified: {artifact.integrity_verified}")
    print(f"    Repair Morphisms: {len(repair_morphisms)}")
    print()

    # -----------------------------------------------------------------------
    # 7. Final Exam (Covenant-Aware, threshold >7.5)
    # -----------------------------------------------------------------------
    print("[7] Final Exam (Covenant-Aware, threshold >7.5)...")
    governance_strength = CovenantFinalExam.compute_governance_strength(
        alignment_scores=dict(artifact.verification_result.principle_scores),
        governance_rule_count=len(artifact.covenant.governance_rules),
        repair_morphism_count=len(repair_morphisms),
    )
    covenant_exam = CovenantFinalExam(pass_threshold=7.5, rounds=100)
    exam_result = covenant_exam.administer(
        governance_strength=governance_strength,
        seed=42,
    )
    print(f"    Governance Strength: {governance_strength:.4f}")
    print(f"    Sustainability Score: {exam_result.sustainability_score:.4f}")
    print(f"    Passed: {exam_result.passed}")
    if not exam_result.passed:
        print(f"    Blocking Reason: {exam_result.blocking_reason}")
    print()

    # Record in soul
    soul.record_wisdom(
        report,
        resolution_path=selected.path_type.value,
        resolution_summary=f"Grid War 2026 resolved via {selected.path_type.value}",
        covenant_title=artifact.covenant.title,
    )
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

    # -----------------------------------------------------------------------
    # 8. Offload Skeleton — demonstrate anonymized offloading
    # -----------------------------------------------------------------------
    print("[8] Demonstrating anonymized offload skeleton...")
    packet = OffloadSkeleton.prepare(conflict_graph.as_dict(), simulation_rounds=100)
    print(f"    Packet Hash: {packet.packet_hash}")
    print(f"    Object Count: {packet.object_count}")
    print(f"    Morphism Count: {packet.morphism_count}")
    print(f"    Extraction Ratio: {packet.extraction_ratio:.2%}")
    print(f"    Vulnerable Nodes: {packet.vulnerable_node_count}")
    print(f"    (Soul file protected — no PII in packet)")
    print()

    # -----------------------------------------------------------------------
    # 9. Crystallization — generate Obsidian vault
    # -----------------------------------------------------------------------
    print("[9] CRYSTALLIZATION EVENT...")
    crystal = exporter.crystallize(
        soul=soul,
        sustainability_score=exam_result.sustainability_score,
        fragility_index=trace.mirror_score,
        passed_exam=exam_result.passed,
        trace=trace,
        manifesto_dict=manifesto_dict,
        repair_morphisms=repair_morphisms,
    )
    print(f"    Vault files: {crystal.vault.file_count}")
    print(f"    Sustainability: {crystal.sustainability_score:.4f}")
    print(f"    Fragility Index: {crystal.fragility_index:.4f}")
    print(f"    Passed Exam: {crystal.passed_exam}")
    print(f"    Legal Gravity: {crystal.legal_gravity_detected}")
    print(f"    Regenerative Loops: {crystal.regenerative_loop_count}")
    print()

    # List vault files
    print("    Vault Structure:")
    for path in sorted(crystal.vault.files.keys()):
        print(f"      {path}")
    print()

    # -----------------------------------------------------------------------
    # 10. Auto-sync to disk
    # -----------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir) / "genesis_vault"
        written = crystal.vault.write_to_disk(vault_path)
        print(f"[10] Auto-synced to: {vault_path}")
        print(f"     Written {len(written)} files")

        # Show the Manifesto frontmatter
        manifesto_path = vault_path / "Manifesto.md"
        if manifesto_path.exists():
            content = manifesto_path.read_text()
            # Show just the frontmatter
            parts = content.split("---")
            if len(parts) >= 3:
                print("\n     Stewardship Manifesto Frontmatter:")
                print("     ---")
                for line in parts[1].strip().split("\n"):
                    print(f"     {line}")
                print("     ---")

    print()
    print("=" * 72)
    print("  CRYSTALLIZATION COMPLETE")
    print("  The Genesis Engine is now a repeatable Policy Analysis Tool,")
    print("  finalized for local-first sovereignty.")
    print("=" * 72)


if __name__ == "__main__":
    main()
