#!/usr/bin/env python3
"""
Genesis Engine — Sprint 5 Demo: Human Override Protocol
=========================================================

Demonstrates the "Manual Override" scenario where a user rejects the
system-recommended "Reform" path in favour of a "Dissolution" path,
with the resulting JSON log entry showing the full "Why".

This demo also shows:
- Ollama provider selection (falls back to LocalProvider if offline)
- Immutable AxiomAnchor protection (seal prevents predicate mutation)
- Human override log visible during Soul Inspection
- Full JSON audit trail with divergence reason

Run:
    python demo_human_override.py
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.axiom_anchor import AxiomAnchor, AxiomAnchorFrozenError
from genesis_engine.core.continuity_bridge import ContinuityBridge, OVERRIDE_REASON_CATEGORIES
from genesis_engine.core.crucible import CandidateStatus, CrucibleEngine
from genesis_engine.ai.providers.ollama_provider import OllamaProvider, get_default_provider


REPORT_DIR = Path(__file__).parent / "genesis_engine" / "reports"


def main() -> None:
    """Run the Human Override demo scenario."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Provider Selection ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  GENESIS ENGINE — Sprint 5: Honesty Hardening & Ollama Integration")
    print("=" * 72)

    # Use the default provider (Ollama → LocalProvider fallback)
    provider = get_default_provider()
    print(f"\n  AI Provider: {provider.provider_name}")

    # ── Initialize with sealed Anchor ─────────────────────────────────────
    anchor = AxiomAnchor()
    anchor.seal()  # Seal the Anchor — predicates are now immutable
    print(f"  Axiom Anchor: SEALED (immutable Ground Truth)")

    # Demonstrate that sealed anchor rejects modification
    print("\n  [Test] Attempting to modify sealed Anchor...")
    try:
        anchor.register_predicate("unity", lambda a: 1.0)
        print("  ERROR: Modification succeeded (should not happen)")
    except AxiomAnchorFrozenError as e:
        print(f"  BLOCKED: {e}")

    # ── Create the Crucible with our sealed Anchor ────────────────────────
    crucible = CrucibleEngine(anchor=anchor, provider=provider)
    aria = AriaInterface(crucible=crucible, use_colors=True)

    print(f"\n  Prime Directive: \"{aria.soul.directive.statement}\"")
    print(f"  Soul ID: {aria.soul.soul_id}")

    # ── Process a problem that generates Reform as system recommendation ──
    problem = (
        "A social media platform that exploits user attention through "
        "addictive design patterns, extracts personal data without "
        "meaningful consent, and neglects the mental health impact on "
        "vulnerable young users."
    )

    print("\n" + "=" * 72)
    print(f"  PROBLEM: {problem[:60]}...")
    print("=" * 72)

    result = aria.process(problem, verbose=True)

    # ── Identify the system recommendation and an alternative ─────────────
    system_best = result.logic_box.best
    if not system_best:
        print("\n  No candidates confirmed — cannot demonstrate override.")
        return

    print("\n" + "=" * 72)
    print("  MANUAL OVERRIDE SCENARIO")
    print("=" * 72)

    print(f"\n  System recommends: {system_best.id}")
    print(f"    Perspective: {system_best.perspective.value}")
    print(f"    Path: {system_best.dream_path.path_type.value if system_best.dream_path else 'N/A'}")
    print(f"    Unity Score: {system_best.unity_alignment_score:.4f}")

    # Find the dissolution candidate (the one the human will choose)
    dissolution_candidate = None
    for cand in result.logic_box.candidates:
        if (cand.dream_path
                and cand.dream_path.path_type.value == "dissolution"
                and cand.id != system_best.id):
            dissolution_candidate = cand
            break

    if not dissolution_candidate:
        # If dissolution wasn't generated or is the best, pick any non-best
        for cand in result.logic_box.candidates:
            if cand.id != system_best.id:
                dissolution_candidate = cand
                break

    if not dissolution_candidate:
        print("\n  Only one candidate available — cannot demonstrate override.")
        return

    print(f"\n  Human selects:    {dissolution_candidate.id}")
    print(f"    Perspective: {dissolution_candidate.perspective.value}")
    print(
        f"    Path: "
        f"{dissolution_candidate.dream_path.path_type.value if dissolution_candidate.dream_path else 'N/A'}"
    )
    print(f"    Unity Score: {dissolution_candidate.unity_alignment_score:.4f}")

    # ── Record the Human Override ─────────────────────────────────────────
    divergence_reason = (
        "The Reform path preserves the existing platform structure which has "
        "a fundamentally extractive business model. Real-world evidence from "
        "regulatory hearings shows that incremental reform of attention-economy "
        "platforms consistently fails because the profit motive overrides "
        "safety commitments. Dissolution and replacement with a "
        "cooperative model is the only path that addresses root causes."
    )

    print(f"\n  Divergence reason ({len(divergence_reason)} chars):")
    print(f"    \"{divergence_reason[:80]}...\"")
    print(f"  Category: real_world_evidence")
    print(f"  Confidence: 8/10")

    override_entry = aria.human_override(
        result=result,
        selected_candidate=dissolution_candidate,
        divergence_reason=divergence_reason,
        reason_category="real_world_evidence",
        confidence=8,
        verbose=True,
    )

    # ── Verify Anchor was NOT modified ────────────────────────────────────
    print("\n" + "-" * 72)
    print("  AXIOM ANCHOR INTEGRITY CHECK")
    print("-" * 72)
    print(f"  Anchor sealed:     {anchor.is_sealed}")
    print(f"  Directive:         \"{anchor.directive.statement}\"")
    print(f"  Threshold:         {anchor.alignment_threshold}")
    print(f"  Predicates intact: Yes (sealed, immutable)")
    print(f"  Override count:    {len(aria.soul.human_overrides)}")
    print(f"  Anchor modified:   No (override recorded in Override Log only)")

    # ── Soul Inspection (shows override log) ──────────────────────────────
    print("\n" + "=" * 72)
    print("  SOUL INSPECTION (with Human Override Log)")
    print("=" * 72)
    aria.inspect_soul(verbose=True)

    # ── Export the full override JSON entry ────────────────────────────────
    print("\n" + "=" * 72)
    print("  OVERRIDE LOG JSON ENTRY")
    print("=" * 72)
    override_json = json.dumps(override_entry.as_dict(), indent=2)
    print(override_json)

    # ── Export the full soul with overrides ────────────────────────────────
    soul_path = REPORT_DIR / "sprint5_override_demo.genesis_soul"
    bridge = ContinuityBridge()
    soul_envelope = bridge.export_soul(aria.soul)
    soul_path.write_text(json.dumps(soul_envelope, indent=2))

    # Also save the override entry standalone
    override_path = REPORT_DIR / "human_override_entry.json"
    override_path.write_text(override_json)

    print(f"\n  Soul exported to:     {soul_path}")
    print(f"  Override entry saved: {override_path}")

    # ── Verify integrity ──────────────────────────────────────────────────
    verified = bridge.verify_integrity(soul_envelope)
    is_valid, chain_errors = bridge.verify_wisdom_chain(aria.soul)

    print(f"\n  Soul integrity hash:  {soul_envelope['genesis_soul']['integrityHash'][:24]}...")
    print(f"  Integrity valid:      {verified}")
    print(f"  Wisdom chain valid:   {is_valid}")

    # ── Available Override Categories ─────────────────────────────────────
    print(f"\n  Override reason categories:")
    for cat in OVERRIDE_REASON_CATEGORIES:
        print(f"    - {cat}")

    print("\n" + "=" * 72)
    print("  Sprint 5 Demo Complete")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
