"""Tests for Foresight Projections in the Continuity Bridge (Module 2.3 — Sprint 6.1)."""

import json

from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    ForesightProjection,
    GenesisSoul,
    WisdomEntry,
)
from genesis_engine.core.axiom_anchor import AxiomAnchor


class TestForesightProjection:
    def test_as_dict(self):
        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="Test summary",
        )
        d = fp.as_dict()
        assert d["warGameRounds"] == 100
        assert d["sustainabilityScore"] == 4.5
        assert d["outcomeFlag"] == "SYSTEMIC_COLLAPSE"
        assert d["foresightSummary"] == "Test summary"

    def test_timestamp_auto_set(self):
        fp = ForesightProjection(
            war_game_rounds=50,
            aligned_score=100.0,
            extractive_score=120.0,
            sustainability_score=6.0,
            outcome_flag="SUSTAINABLE_VICTORY",
            aligned_cooperation_rate=0.9,
            extractive_cooperation_rate=0.4,
            foresight_summary="Test",
        )
        assert fp.timestamp != ""


class TestWisdomEntryWithForesight:
    def test_wisdom_entry_default_empty_foresight(self):
        entry = WisdomEntry(
            source_text="test",
            disharmony_summary="none",
            unity_impact=0.0,
            compassion_deficit=0.0,
            resolution_path="unresolved",
        )
        assert entry.foresight_projections == []

    def test_wisdom_entry_with_foresight(self):
        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="War-game summary",
        )
        entry = WisdomEntry(
            source_text="test",
            disharmony_summary="none",
            unity_impact=0.0,
            compassion_deficit=0.0,
            resolution_path="unresolved",
            foresight_projections=[fp],
        )
        assert len(entry.foresight_projections) == 1

    def test_foresight_included_in_hash(self):
        """Changing foresight should change the hash."""
        entry1 = WisdomEntry(
            source_text="test",
            disharmony_summary="none",
            unity_impact=0.0,
            compassion_deficit=0.0,
            resolution_path="unresolved",
            foresight_projections=[],
        )
        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="summary",
        )
        entry2 = WisdomEntry(
            source_text="test",
            disharmony_summary="none",
            unity_impact=0.0,
            compassion_deficit=0.0,
            resolution_path="unresolved",
            foresight_projections=[fp],
            timestamp=entry1.timestamp,  # same timestamp
        )
        h1 = entry1.compute_hash("")
        h2 = entry2.compute_hash("")
        assert h1 != h2

    def test_as_dict_includes_foresight(self):
        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="summary",
        )
        entry = WisdomEntry(
            source_text="test",
            disharmony_summary="none",
            unity_impact=0.0,
            compassion_deficit=0.0,
            resolution_path="unresolved",
            foresight_projections=[fp],
        )
        d = entry.as_dict()
        assert "foresightProjections" in d
        assert len(d["foresightProjections"]) == 1
        assert d["foresightProjections"][0]["outcomeFlag"] == "SYSTEMIC_COLLAPSE"


class TestGenesisSoulForesight:
    def test_record_foresight(self):
        soul = GenesisSoul()
        # First add a wisdom entry (foresight attaches to latest wisdom)
        entry = WisdomEntry(
            source_text="test problem",
            disharmony_summary="test",
            unity_impact=3.0,
            compassion_deficit=2.0,
            resolution_path="reform",
        )
        entry.entry_hash = entry.compute_hash("")
        soul.wisdom_log.append(entry)

        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="test summary",
        )
        soul.record_foresight(fp)

        assert len(soul.wisdom_log[-1].foresight_projections) == 1
        assert soul.wisdom_log[-1].foresight_projections[0].outcome_flag == "SYSTEMIC_COLLAPSE"


class TestContinuityBridgeForesightRoundTrip:
    def test_export_import_with_foresight(self):
        """Foresight projections survive export/import."""
        anchor = AxiomAnchor()
        soul = ContinuityBridge.create_soul(anchor)

        # Add a wisdom entry with foresight
        fp = ForesightProjection(
            war_game_rounds=100,
            aligned_score=250.0,
            extractive_score=280.0,
            sustainability_score=4.5,
            outcome_flag="SYSTEMIC_COLLAPSE",
            aligned_cooperation_rate=0.8,
            extractive_cooperation_rate=0.3,
            foresight_summary="War-game result",
        )
        entry = WisdomEntry(
            source_text="test",
            disharmony_summary="test",
            unity_impact=3.0,
            compassion_deficit=2.0,
            resolution_path="reform",
            foresight_projections=[fp],
        )
        entry.entry_hash = entry.compute_hash("")
        soul.wisdom_log.append(entry)

        # Export
        envelope = ContinuityBridge.export_soul(soul)
        assert ContinuityBridge.verify_integrity(envelope)

        # Import
        imported = ContinuityBridge.import_soul(envelope)
        assert imported is not None
        assert len(imported.wisdom_log) == 1
        assert len(imported.wisdom_log[0].foresight_projections) == 1
        imported_fp = imported.wisdom_log[0].foresight_projections[0]
        assert imported_fp.war_game_rounds == 100
        assert imported_fp.sustainability_score == 4.5
        assert imported_fp.outcome_flag == "SYSTEMIC_COLLAPSE"
        assert imported_fp.foresight_summary == "War-game result"
