#!/usr/bin/env python3
"""
Genesis Engine — Sprint 4: The Socratic Crucible & Aria Interface
===================================================================

Demonstrates the full 6-Phase Crucible workflow with multi-perspective
reasoning and the Aria CLI visualization:

    Natural-language problem
        -> Crucible Engine (6-Phase Workflow)
            1. Ingest         — Translate to CategoricalGraph
            2. Retrieval      — Search EternalBox for prior wisdom
            3. Divergence     — Generate multi-perspective candidates
            4. Verification   — Validate each through Axiom Anchor
            5. Convergence    — Rank and select best candidate
            6. Crystallization — Commit to EternalBox as Technical Covenant
        -> Aria Interface (CLI visualization of Thinking process)
        -> Continuity Bridge (.genesis_soul export with hash-chaining)

The Dual-Memory Architecture:
    - LogicBox:   Ephemeral sandbox for candidate evaluation
    - EternalBox: Persistent .genesis_soul with hash-chained wisdom log

Run:
    python main.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.continuity_bridge import ContinuityBridge


# ---------------------------------------------------------------------------
# Sample problem statements
# ---------------------------------------------------------------------------

PROBLEMS: list[str] = [
    "A corporate policy that prioritizes profit over user safety.",
    "A community health program that empowers workers and protects the environment.",
    "An AI surveillance system that exploits user data and neglects privacy.",
]

REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def main() -> None:
    """Run all sample problems through the full Crucible pipeline."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the Aria Interface (which creates the Crucible Engine internally)
    aria = AriaInterface(use_colors=True)

    print("\n" + "═" * 72)
    print("  GENESIS ENGINE — Sprint 4: The Socratic Crucible")
    print("═" * 72)
    print(f"\n  Prime Directive: \"{aria.soul.directive.statement}\"")
    print(f"  Alignment Threshold: {aria.soul.alignment_threshold}")
    print(f"  AI Provider: {aria.crucible.provider.provider_name}")
    print()

    # Process each problem through the Crucible
    for i, problem in enumerate(PROBLEMS):
        print("\n" + "═" * 72)
        print(f"  PROBLEM {i + 1}: {problem}")
        print("═" * 72)

        # Run through the 6-phase Crucible workflow with Aria visualization
        result = aria.process(problem, verbose=True)

        # Save individual reports
        if result.disharmony_report:
            dis_path = REPORT_DIR / f"disharmony_report_{i + 1}.json"
            dis_path.write_text(json.dumps(result.disharmony_report.as_dict(), indent=2))

        if result.possibility_report:
            pos_path = REPORT_DIR / f"possibility_report_{i + 1}.json"
            pos_path.write_text(json.dumps(result.possibility_report.as_dict(), indent=2))

        if result.crystallized_candidate and result.crystallized_candidate.artifact:
            forge_path = REPORT_DIR / f"forge_artifact_{i + 1}.json"
            forge_path.write_text(json.dumps(
                result.crystallized_candidate.artifact.as_dict(), indent=2
            ))

        # Save Crucible result
        crucible_path = REPORT_DIR / f"crucible_result_{i + 1}.json"
        crucible_path.write_text(json.dumps(result.as_dict(), indent=2))

    # ── Display EternalBox (Soul) State ────────────────────────────────────
    print("\n" + "═" * 72)
    print("  ETERNAL BOX (Genesis Soul) Summary")
    print("═" * 72)
    aria.inspect_soul(verbose=True)

    # ── Verify Hash Chain ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Hash Chain Verification")
    print("─" * 72)
    is_valid, errors = aria.verify_chain()
    if not is_valid:
        for err in errors:
            print(f"  ERROR: {err}")

    # ── Export .genesis_soul ───────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Exporting .genesis_soul file")
    print("─" * 72)

    soul_path = REPORT_DIR / "session.genesis_soul"
    bridge = ContinuityBridge()
    soul_envelope = bridge.export_soul(aria.soul)
    soul_path.write_text(json.dumps(soul_envelope, indent=2))

    verified = bridge.verify_integrity(soul_envelope)
    print(f"\n  Soul ID:          {aria.soul.soul_id}")
    print(f"  Graphs recorded:  {len(aria.soul.graph_history)}")
    print(f"  Wisdom entries:   {len(aria.soul.wisdom_log)}")
    print(f"  Forge artifacts:  {len(aria.soul.forge_artifacts)}")
    print(f"  Integrity hash:   {soul_envelope['genesis_soul']['integrityHash'][:24]}...")
    print(f"  Integrity valid:  {verified}")
    print(f"  Exported to:      {soul_path}")

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  All reports written to:", REPORT_DIR)
    print("═" * 72)

    # List wisdom log with hash chain info
    print("\n  Wisdom Log (with hash chain):")
    for i, entry in enumerate(aria.soul.wisdom_log):
        hash_preview = entry.entry_hash[:16] + "..." if entry.entry_hash else "(no hash)"
        print(f"    [{i+1}] {entry.source_text[:40]}...")
        print(f"        Path: {entry.resolution_path}  |  Hash: {hash_preview}")

    print()


if __name__ == "__main__":
    main()
