"""Tests for Module 2.3 — The Continuity Bridge."""

import json

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph
from genesis_engine.core.deconstruction_engine import DeconstructionEngine, DisharmonyReport
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
    GraphHistoryEntry,
    WisdomEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_populated_soul() -> GenesisSoul:
    """Create a soul with some history recorded."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor)
    bridge = ContinuityBridge()

    soul = bridge.create_soul(anchor)

    # Run extractive scenario.
    graph = translator.translate(
        "A corporate policy that prioritizes profit over user safety."
    )
    soul.record_graph(graph, "translation", "Corporate scenario")

    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    reinvention = next(p for p in possibility.paths if p.path_type == PathType.REINVENTION)
    soul.record_graph(reinvention.healed_graph, "dream", "Reinvention path")

    artifact = forge.forge(reinvention)
    soul.record_graph(artifact.verification_graph, "verification", "Forge verification")
    soul.record_forge_artifact({
        "covenantTitle": artifact.covenant.title,
        "integrityVerified": artifact.integrity_verified,
    })

    soul.record_wisdom(
        report,
        resolution_path="reinvention",
        resolution_summary="Stewardship model adopted.",
        covenant_title=artifact.covenant.title,
    )

    return soul


# ---------------------------------------------------------------------------
# GenesisSoul creation tests
# ---------------------------------------------------------------------------

class TestGenesisSoulCreation:
    def test_create_soul(self):
        bridge = ContinuityBridge()
        soul = bridge.create_soul()
        assert isinstance(soul, GenesisSoul)
        assert soul.soul_id.startswith("soul-")

    def test_soul_has_directive(self):
        bridge = ContinuityBridge()
        soul = bridge.create_soul()
        assert soul.directive.statement == "Does this serve Love?"

    def test_soul_from_custom_anchor(self):
        anchor = AxiomAnchor(alignment_threshold=0.8)
        bridge = ContinuityBridge()
        soul = bridge.create_soul(anchor)
        assert soul.alignment_threshold == 0.8

    def test_soul_starts_empty(self):
        bridge = ContinuityBridge()
        soul = bridge.create_soul()
        assert len(soul.graph_history) == 0
        assert len(soul.wisdom_log) == 0
        assert len(soul.forge_artifacts) == 0


# ---------------------------------------------------------------------------
# Recording tests
# ---------------------------------------------------------------------------

class TestSoulRecording:
    def test_record_graph(self):
        soul = GenesisSoul()
        graph = CategoricalGraph(source_text="test")
        soul.record_graph(graph, "translation", "test label")
        assert len(soul.graph_history) == 1
        assert soul.graph_history[0].phase == "translation"
        assert soul.graph_history[0].label == "test label"

    def test_record_multiple_graphs(self):
        soul = GenesisSoul()
        for i in range(5):
            soul.record_graph(CategoricalGraph(), "dream", f"graph {i}")
        assert len(soul.graph_history) == 5

    def test_record_wisdom(self):
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)

        graph = translator.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        report = decon.analyse(graph)

        soul = GenesisSoul()
        soul.record_wisdom(report, resolution_path="reform", resolution_summary="Fixed.")
        assert len(soul.wisdom_log) == 1
        assert soul.wisdom_log[0].resolution_path == "reform"
        assert soul.wisdom_log[0].unity_impact > 0

    def test_record_forge_artifact(self):
        soul = GenesisSoul()
        soul.record_forge_artifact({"title": "Test Covenant"})
        assert len(soul.forge_artifacts) == 1

    def test_updated_at_changes(self):
        soul = GenesisSoul()
        original = soul.updated_at
        soul.record_graph(CategoricalGraph(), "translation")
        assert soul.updated_at >= original


# ---------------------------------------------------------------------------
# Export / Import tests
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_produces_envelope(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)
        assert "genesis_soul" in envelope
        assert "integrityHash" in envelope["genesis_soul"]
        assert "payload" in envelope["genesis_soul"]

    def test_export_format_version(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)
        assert envelope["genesis_soul"]["format"] == "genesis_soul_v1"

    def test_integrity_hash_valid(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)
        assert bridge.verify_integrity(envelope) is True

    def test_tampered_envelope_fails(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)
        # Tamper with the payload.
        envelope["genesis_soul"]["payload"]["soulId"] = "tampered-id"
        assert bridge.verify_integrity(envelope) is False

    def test_import_restores_soul(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)

        restored = bridge.import_soul(envelope)
        assert restored is not None
        assert restored.soul_id == soul.soul_id
        assert restored.directive.statement == soul.directive.statement

    def test_import_restores_wisdom(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)

        restored = bridge.import_soul(envelope)
        assert restored is not None
        assert len(restored.wisdom_log) == len(soul.wisdom_log)
        assert restored.wisdom_log[0].resolution_path == "reinvention"

    def test_import_restores_artifacts(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)

        restored = bridge.import_soul(envelope)
        assert restored is not None
        assert len(restored.forge_artifacts) == len(soul.forge_artifacts)

    def test_import_tampered_returns_none(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        envelope = bridge.export_soul(soul)
        envelope["genesis_soul"]["payload"]["version"] = "hacked"
        assert bridge.import_soul(envelope) is None

    def test_export_json_string(self):
        soul = _create_populated_soul()
        bridge = ContinuityBridge()
        json_str = bridge.export_soul_json(soul)
        data = json.loads(json_str)
        assert "genesis_soul" in data


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

class TestSoulSerialisation:
    def test_as_dict_structure(self):
        soul = _create_populated_soul()
        data = soul.as_dict()
        assert "soulId" in data
        assert "axiomAnchorState" in data
        assert "graphHistory" in data
        assert "wisdomLog" in data
        assert "forgeArtifacts" in data
        assert "metadata" in data

    def test_metadata_counts(self):
        soul = _create_populated_soul()
        data = soul.as_dict()
        meta = data["metadata"]
        assert meta["graphCount"] == len(soul.graph_history)
        assert meta["wisdomCount"] == len(soul.wisdom_log)
        assert meta["forgeCount"] == len(soul.forge_artifacts)

    def test_to_json(self):
        soul = _create_populated_soul()
        json_str = soul.to_json()
        data = json.loads(json_str)
        assert data["soulId"] == soul.soul_id

    def test_axiom_anchor_state_preserved(self):
        soul = _create_populated_soul()
        data = soul.as_dict()
        anchor_state = data["axiomAnchorState"]
        assert anchor_state["directive"]["statement"] == "Does this serve Love?"
        assert "alignmentThreshold" in anchor_state


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

class TestEndToEndSoulLifecycle:
    def test_full_lifecycle(self):
        """Create -> populate -> export -> verify -> import -> validate."""
        anchor = AxiomAnchor()
        translator = AxiomLogixTranslator()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        bridge = ContinuityBridge()

        # Create.
        soul = bridge.create_soul(anchor)

        # Populate.
        graph = translator.translate("Users exploited by corporations.")
        soul.record_graph(graph, "translation")

        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)
        soul.record_wisdom(report, "reinvention", "Stewardship model.")

        for path in possibility.paths:
            soul.record_graph(path.healed_graph, "dream", path.title)

        # Export.
        envelope = bridge.export_soul(soul)

        # Verify.
        assert bridge.verify_integrity(envelope) is True

        # Import.
        restored = bridge.import_soul(envelope)
        assert restored is not None
        assert restored.soul_id == soul.soul_id
        assert len(restored.wisdom_log) == 1
        assert restored.wisdom_log[0].resolution_path == "reinvention"
