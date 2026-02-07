"""Tests for Module 2.3 Extension — The Obsidian Exporter (Sprint 7)."""

import tempfile
from pathlib import Path

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    ForesightProjection,
    GenesisSoul,
)
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.obsidian_exporter import ObsidianExporter, ObsidianVault


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_populated_soul() -> GenesisSoul:
    """Create a soul with wisdom log, overrides, and graph history."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    bridge = ContinuityBridge()

    soul = bridge.create_soul(anchor)

    # Add a graph and wisdom entry via the pipeline
    graph = translator.translate(
        "A corporate policy that prioritizes profit over user safety."
    )
    soul.record_graph(graph, "translation", "Extractive graph")

    report = decon.analyse(graph)
    soul.record_wisdom(
        report,
        resolution_path="reinvention",
        resolution_summary="Replaced extraction with stewardship",
        covenant_title="Stewardship Covenant: Corporate Reform",
    )

    # Add a foresight projection
    fp = ForesightProjection(
        war_game_rounds=100,
        aligned_score=245.0,
        extractive_score=198.0,
        sustainability_score=6.8,
        outcome_flag="SUSTAINABLE_VICTORY",
        aligned_cooperation_rate=0.82,
        extractive_cooperation_rate=0.35,
        foresight_summary="Aligned agent wins with sustainable cooperation.",
    )
    soul.record_foresight(fp)

    # Add a human override
    soul.record_human_override(
        system_recommended_id="cand-001",
        system_recommended_score=0.85,
        human_selected_id="cand-003",
        human_selected_score=0.72,
        divergence_reason=(
            "The system recommendation fails to account for cultural context "
            "that significantly impacts stakeholder vulnerability assessment "
            "in this particular domain of analysis"
        ),
        reason_category="cultural_context",
        confidence=8,
        problem_text="Corporate reform in emerging markets",
        system_recommended_path="reform",
        human_selected_path="reinvention",
    )

    return soul


# ---------------------------------------------------------------------------
# Vault structure tests
# ---------------------------------------------------------------------------

class TestObsidianVaultStructure:
    def test_export_returns_vault(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert isinstance(vault, ObsidianVault)

    def test_vault_has_manifesto(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert "Manifesto.md" in vault.files

    def test_vault_has_insights(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert any(f.startswith("Insights/") for f in vault.files)

    def test_vault_has_projections(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert any(f.startswith("Projections/") for f in vault.files)

    def test_vault_has_overrides(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert any(f.startswith("Overrides/") for f in vault.files)

    def test_vault_has_graphs(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert any(f.startswith("Graphs/") for f in vault.files)

    def test_file_count_matches(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        # 1 manifesto + 1 wisdom + 1 projection + 1 override + 1 graph = 5
        assert vault.file_count >= 5

    def test_empty_soul_has_manifesto_only(self):
        soul = ContinuityBridge.create_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert vault.file_count == 1
        assert "Manifesto.md" in vault.files


# ---------------------------------------------------------------------------
# Manifesto frontmatter tests
# ---------------------------------------------------------------------------

class TestManifestoFrontmatter:
    def test_manifesto_has_yaml_frontmatter(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert manifesto.startswith("---")
        # Should have closing ---
        assert manifesto.count("---") >= 2

    def test_manifesto_has_soul_id(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert soul.soul_id in manifesto

    def test_manifesto_has_integrity_hash(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "integrity_hash:" in manifesto

    def test_manifesto_has_prime_directive(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "Does this serve Love?" in manifesto


# ---------------------------------------------------------------------------
# Bidirectional linking tests
# ---------------------------------------------------------------------------

class TestBidirectionalLinking:
    def test_manifesto_links_to_insights(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "[[wisdom_001" in manifesto

    def test_manifesto_links_to_overrides(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "[[override_001" in manifesto

    def test_manifesto_links_to_graphs(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "[[graph_001" in manifesto

    def test_manifesto_links_to_projections(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        manifesto = vault.files["Manifesto.md"]
        assert "[[foresight_001" in manifesto

    def test_insight_links_back_to_manifesto(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        insight = vault.files["Insights/wisdom_001.md"]
        assert "[[Manifesto]]" in insight

    def test_override_links_back_to_manifesto(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        override = vault.files["Overrides/override_001.md"]
        assert "[[Manifesto]]" in override

    def test_graph_links_back_to_manifesto(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        graph = vault.files["Graphs/graph_001.md"]
        assert "[[Manifesto]]" in graph

    def test_projection_links_to_source_insight(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        projection = vault.files["Projections/foresight_001.md"]
        assert "[[wisdom_001" in projection


# ---------------------------------------------------------------------------
# Mermaid diagrams
# ---------------------------------------------------------------------------

class TestMermaidDiagrams:
    def test_graph_has_mermaid(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter(include_mermaid=True)
        vault = exporter.export(soul)
        graph = vault.files["Graphs/graph_001.md"]
        assert "```mermaid" in graph
        assert "graph LR" in graph

    def test_mermaid_can_be_disabled(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter(include_mermaid=False)
        vault = exporter.export(soul)
        graph = vault.files["Graphs/graph_001.md"]
        assert "```mermaid" not in graph


# ---------------------------------------------------------------------------
# Disk write tests
# ---------------------------------------------------------------------------

class TestDiskWrite:
    def test_write_to_disk(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)

        with tempfile.TemporaryDirectory() as tmpdir:
            written = vault.write_to_disk(Path(tmpdir))
            assert len(written) == vault.file_count

            # Check directory structure
            assert (Path(tmpdir) / "Manifesto.md").exists()
            assert (Path(tmpdir) / "Insights").is_dir()
            assert (Path(tmpdir) / "Projections").is_dir()
            assert (Path(tmpdir) / "Overrides").is_dir()
            assert (Path(tmpdir) / "Graphs").is_dir()


# ---------------------------------------------------------------------------
# Integrity hash tests
# ---------------------------------------------------------------------------

class TestVaultIntegrity:
    def test_vault_has_integrity_hash(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        assert len(vault.integrity_hash) == 64  # SHA-256

    def test_same_soul_produces_same_hash(self):
        """Deterministic export for the same soul data."""
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        v1 = exporter.export(soul)
        v2 = exporter.export(soul)
        assert v1.integrity_hash == v2.integrity_hash

    def test_vault_as_dict(self):
        soul = _make_populated_soul()
        exporter = ObsidianExporter()
        vault = exporter.export(soul)
        d = vault.as_dict()
        assert "obsidianVault" in d
        assert d["obsidianVault"]["soulId"] == soul.soul_id
        assert d["obsidianVault"]["fileCount"] == vault.file_count
