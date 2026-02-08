"""Tests for Sprint 9 — Sovereign Synthesis & Vertical Actuation.

Covers:
- Conflict War-Room (load_conflict, inject_node)
- Expanded Grid War scenario (water nodes, SB 1488, shadow entity)
- Mirror of Truth: Shadow Entity probe, water sustainability detection
- Obsidian Exporter: Crystallize trigger, Stewardship Manifesto Frontmatter
- AI Provider: OllamaProvider fallback, OffloadSkeleton
- Full Crystallization pipeline
"""

import json
import tempfile
from pathlib import Path

from genesis_engine.core.ai_provider import (
    LocalProvider,
    OllamaProvider,
    OffloadSkeleton,
    OffloadPacket,
    get_default_provider,
)
from genesis_engine.core.architectural_forge import ArchitecturalForge
from genesis_engine.core.aria_interface import AriaInterface
from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    ForesightProjection,
)
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine
from genesis_engine.core.game_theory_console import CovenantFinalExam
from genesis_engine.core.mirror_of_truth import MirrorOfTruth, RefinementTrace
from genesis_engine.core.obsidian_exporter import (
    CrystallizationResult,
    ObsidianExporter,
    stewardship_frontmatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCENARIO_PATH = "scenarios/grid_war_2026.json"


def _load_scenario() -> dict:
    return MirrorOfTruth.load_scenario(SCENARIO_PATH)


def _build_conflict_graph() -> CategoricalGraph:
    scenario = _load_scenario()
    return MirrorOfTruth.scenario_to_graph(scenario)


def _run_full_pipeline():
    """Run the full pipeline and return everything needed for crystallization."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    mirror = MirrorOfTruth(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor, translator=translator)
    bridge = ContinuityBridge()
    soul = bridge.create_soul(anchor)

    graph = _build_conflict_graph()
    soul.record_graph(graph, "translation", "Grid War 2026")

    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    selected = possibility.paths[0]
    trace = mirror.critique(selected, report, graph)
    artifact = forge.forge(selected)

    governance_strength = CovenantFinalExam.compute_governance_strength(
        alignment_scores=dict(artifact.verification_result.principle_scores),
        governance_rule_count=len(artifact.covenant.governance_rules),
        repair_morphism_count=len(
            artifact.manifesto.regenerative_loop.repair_morphisms
        )
        if artifact.manifesto
        else 0,
    )
    exam = CovenantFinalExam(pass_threshold=7.5, rounds=100)
    exam_result = exam.administer(governance_strength=governance_strength, seed=42)

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

    manifesto_dict = artifact.manifesto.as_dict() if artifact.manifesto else {}
    repair_morphisms = []
    if artifact.manifesto:
        for rm in artifact.manifesto.regenerative_loop.repair_morphisms:
            repair_morphisms.append(rm.as_dict())

    return {
        "soul": soul,
        "trace": trace,
        "exam_result": exam_result,
        "manifesto_dict": manifesto_dict,
        "repair_morphisms": repair_morphisms,
        "artifact": artifact,
        "graph": graph,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Grid War 2026 scenario expansion tests
# ---------------------------------------------------------------------------


class TestGridWarScenarioExpansion:
    def test_scenario_loads(self):
        scenario = _load_scenario()
        assert scenario["scenario"] == "The 2026 Oklahoma Grid War"
        assert scenario["version"] == "2.0.0"
        assert scenario["sprint"] == 9

    def test_scenario_has_legislative_references(self):
        scenario = _load_scenario()
        leg_refs = scenario["context"]["legislative_references"]
        assert len(leg_refs) == 2
        bills = [ref["bill"] for ref in leg_refs]
        assert "HB 2992" in bills
        assert "SB 1488" in bills

    def test_scenario_has_water_supply_node(self):
        scenario = _load_scenario()
        objects = scenario["conflict_graph"]["objects"]
        labels = [o["label"] for o in objects]
        assert "Oklahoma_Water_Supply" in labels

    def test_water_supply_is_shadow_entity(self):
        scenario = _load_scenario()
        objects = scenario["conflict_graph"]["objects"]
        water = next(o for o in objects if o["label"] == "Oklahoma_Water_Supply")
        assert "shadow_entity" in water["tags"]

    def test_scenario_has_water_resources_board(self):
        scenario = _load_scenario()
        objects = scenario["conflict_graph"]["objects"]
        labels = [o["label"] for o in objects]
        assert "Water_Resources_Board" in labels

    def test_scenario_has_water_morphisms(self):
        scenario = _load_scenario()
        morphisms = scenario["conflict_graph"]["morphisms"]
        labels = [m["label"] for m in morphisms]
        assert "Cooling_Water_Withdrawal" in labels
        assert "Community_Water_Dependency" in labels
        assert "Water_Withdrawal_Permit" in labels
        assert "Cooling_Impact_Assessment" in labels

    def test_water_sustainability_constraints(self):
        scenario = _load_scenario()
        water = scenario["constraints"]["water_sustainability"]
        assert water["datacenter_demand_mgd"] == 42.0
        assert water["aquifer_recharge_rate_mgd"] == 18.0

    def test_hill_request_has_water_data(self):
        scenario = _load_scenario()
        hill = scenario["hill_request"]
        assert hill["cooling_water_demand_mgd"] == 42.0
        assert hill["water_overshoot_ratio"] == 2.33

    def test_expected_findings_include_water(self):
        scenario = _load_scenario()
        cats = scenario["expected_mirror_findings"]["deep_disharmony_categories"]
        assert "unsustainable_water_withdrawal" in cats

    def test_six_objects_in_graph(self):
        scenario = _load_scenario()
        objects = scenario["conflict_graph"]["objects"]
        assert len(objects) == 6

    def test_ten_morphisms_in_graph(self):
        scenario = _load_scenario()
        morphisms = scenario["conflict_graph"]["morphisms"]
        assert len(morphisms) == 10


# ---------------------------------------------------------------------------
# Conflict War-Room tests (AriaInterface)
# ---------------------------------------------------------------------------


class TestConflictWarRoom:
    def test_load_conflict_returns_scenario_and_graph(self):
        aria = AriaInterface(use_colors=False)
        scenario, graph = aria.load_conflict(SCENARIO_PATH, verbose=False)
        assert scenario["scenario"] == "The 2026 Oklahoma Grid War"
        assert len(graph.objects) == 6
        assert len(graph.morphisms) == 10

    def test_load_conflict_stores_state(self):
        aria = AriaInterface(use_colors=False)
        aria.load_conflict(SCENARIO_PATH, verbose=False)
        assert aria._active_scenario is not None
        assert aria._active_graph is not None

    def test_inject_node_adds_object(self):
        aria = AriaInterface(use_colors=False)
        _, graph = aria.load_conflict(SCENARIO_PATH, verbose=False)
        original_count = len(graph.objects)
        graph = aria.inject_node(
            graph=graph,
            label="Local_Innovation",
            tags=["stakeholder", "community", "innovation"],
            verbose=False,
        )
        assert len(graph.objects) == original_count + 1

    def test_inject_node_with_connection(self):
        aria = AriaInterface(use_colors=False)
        _, graph = aria.load_conflict(SCENARIO_PATH, verbose=False)
        original_morph_count = len(graph.morphisms)
        graph = aria.inject_node(
            graph=graph,
            label="Water_Recycler",
            tags=["mechanism", "protective", "sustainability"],
            connect_to="Oklahoma_Water_Supply",
            morphism_label="Water_Reclamation",
            morphism_tags=["protection", "sustainable_management"],
            verbose=False,
        )
        assert len(graph.morphisms) == original_morph_count + 1

    def test_war_room_render(self, capsys):
        aria = AriaInterface(use_colors=False)
        aria.load_conflict(SCENARIO_PATH, verbose=True)
        captured = capsys.readouterr()
        assert "CONFLICT WAR-ROOM" in captured.out
        assert "HB 2992" in captured.out
        assert "SB 1488" in captured.out
        assert "Oklahoma_Water_Supply" in captured.out


# ---------------------------------------------------------------------------
# Mirror of Truth: Shadow Entity + Water probe tests
# ---------------------------------------------------------------------------


class TestMirrorShadowEntity:
    def test_scenario_to_graph(self):
        graph = _build_conflict_graph()
        assert len(graph.objects) == 6
        assert len(graph.morphisms) == 10

    def test_shadow_entity_in_graph(self):
        graph = _build_conflict_graph()
        shadow = [o for o in graph.objects if "shadow_entity" in o.tags]
        assert len(shadow) == 1
        assert shadow[0].label == "Oklahoma_Water_Supply"

    def test_mirror_detects_water_disharmony(self):
        anchor = AxiomAnchor()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        mirror = MirrorOfTruth(anchor=anchor)

        graph = _build_conflict_graph()
        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)
        selected = possibility.paths[0]

        trace = mirror.critique(selected, report, graph)
        # Should detect unsustainable_water_withdrawal category
        assert "unsustainable_water_withdrawal" in trace.deep_disharmony_categories

    def test_mirror_produces_shadow_entity_finding(self):
        anchor = AxiomAnchor()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        mirror = MirrorOfTruth(anchor=anchor)

        graph = _build_conflict_graph()
        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)
        selected = possibility.paths[0]

        trace = mirror.critique(selected, report, graph)
        shadow_findings = [
            f for f in trace.critique_findings
            if f.category == "shadow_entity_depletion"
        ]
        assert len(shadow_findings) >= 1
        assert shadow_findings[0].severity >= 8.0

    def test_mirror_repair_includes_water_sustainability(self):
        anchor = AxiomAnchor()
        decon = DeconstructionEngine(anchor=anchor)
        dream = DreamEngine(anchor=anchor)
        mirror = MirrorOfTruth(anchor=anchor)

        graph = _build_conflict_graph()
        report = decon.analyse(graph)
        possibility = dream.dream(report, graph)
        selected = possibility.paths[0]

        trace = mirror.critique(selected, report, graph)
        assert "Water Sustainability" in trace.mandatory_repair


# ---------------------------------------------------------------------------
# Obsidian Exporter: Crystallize + Frontmatter tests
# ---------------------------------------------------------------------------


class TestCrystallization:
    def test_stewardship_frontmatter_function(self):
        fm = stewardship_frontmatter(
            sustainability_score=8.5,
            fragility_index=3.2,
            regenerative_loops=4,
            legal_gravity_detected=True,
        )
        assert "---" in fm
        assert "manifesto_version: 1.0" in fm
        assert "steward: Human Arbiter" in fm
        assert "sustainability_score: 8.5000" in fm
        assert "legal_gravity_detected: True" in fm

    def test_crystallize_returns_result(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter(include_mermaid=True)
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=pipeline["exam_result"].sustainability_score,
            fragility_index=pipeline["trace"].mirror_score,
            passed_exam=pipeline["exam_result"].passed,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        assert isinstance(result, CrystallizationResult)

    def test_crystallize_vault_has_manifesto(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        assert "Manifesto.md" in result.vault.files

    def test_crystallize_vault_has_manifestos_dir(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        assert any(f.startswith("Manifestos/") for f in result.vault.files)

    def test_crystallize_vault_has_repairs_dir(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        if pipeline["repair_morphisms"]:
            assert any(f.startswith("Repairs/") for f in result.vault.files)

    def test_crystallize_vault_has_projections_dir(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        assert any(f.startswith("Projections/") for f in result.vault.files)

    def test_crystallize_manifesto_has_frontmatter(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        manifesto = result.vault.files["Manifesto.md"]
        assert manifesto.startswith("---")
        assert "manifesto_version:" in manifesto
        assert "steward:" in manifesto
        assert "pillars_aligned:" in manifesto
        assert "sustainability_score:" in manifesto
        assert "fragility_index:" in manifesto
        assert "regenerative_loops:" in manifesto
        assert "legal_gravity_detected:" in manifesto

    def test_crystallize_all_files_have_frontmatter(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        # Check that all non-legacy files start with frontmatter
        for path, content in result.vault.files.items():
            if path in ("Manifesto.md",) or path.startswith(
                ("Manifestos/", "Repairs/", "Insights/", "Projections/")
            ):
                assert content.startswith("---"), (
                    f"{path} should start with Stewardship Manifesto Frontmatter"
                )

    def test_crystallize_writes_to_disk(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            written = result.vault.write_to_disk(Path(tmpdir))
            assert len(written) == result.vault.file_count
            assert (Path(tmpdir) / "Manifesto.md").exists()
            assert (Path(tmpdir) / "Manifestos").is_dir()

    def test_crystallize_detects_legal_gravity(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        # The Grid War trace should detect shareholder primacy
        if "shareholder_primacy_extraction" in pipeline["trace"].deep_disharmony_categories:
            assert result.legal_gravity_detected is True

    def test_crystallization_result_as_dict(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
        )
        d = result.as_dict()
        assert "crystallization" in d
        assert d["crystallization"]["sustainabilityScore"] == 8.0
        assert d["crystallization"]["passedExam"] is True


# ---------------------------------------------------------------------------
# AI Provider: OllamaProvider + OffloadSkeleton tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_ollama_provider_creates(self):
        provider = OllamaProvider()
        assert provider.provider_name.startswith("OllamaProvider")

    def test_ollama_fallback_to_local(self):
        """OllamaProvider should fall back to LocalProvider when Ollama is not running."""
        provider = OllamaProvider(base_url="http://localhost:99999", timeout=0.1)
        assert provider.is_available() is False

        # Should still produce candidates via fallback
        context = {
            "source_text": "Test problem",
            "morphisms": [],
            "objects": [],
        }
        candidates = provider.generate_candidates(context)
        assert len(candidates) >= 1

    def test_get_default_provider_returns_provider(self):
        provider = get_default_provider()
        # Should return a provider (either Ollama or Local)
        assert hasattr(provider, "generate_candidates")


class TestOffloadSkeleton:
    def test_prepare_creates_packet(self):
        graph = _build_conflict_graph()
        packet = OffloadSkeleton.prepare(graph.as_dict())
        assert isinstance(packet, OffloadPacket)
        assert packet.object_count == 6
        assert packet.morphism_count == 10

    def test_packet_is_anonymized(self):
        """Packet should not contain labels, IDs, or timestamps."""
        graph = _build_conflict_graph()
        packet = OffloadSkeleton.prepare(graph.as_dict())
        packet_dict = packet.as_dict()
        packet_json = json.dumps(packet_dict)
        # No labels from the scenario should appear
        assert "Residential_Ratepayer" not in packet_json
        assert "Oklahoma_Water_Supply" not in packet_json
        assert "soul-" not in packet_json

    def test_packet_has_hash(self):
        graph = _build_conflict_graph()
        packet = OffloadSkeleton.prepare(graph.as_dict())
        assert len(packet.packet_hash) == 16

    def test_packet_has_extraction_ratio(self):
        graph = _build_conflict_graph()
        packet = OffloadSkeleton.prepare(graph.as_dict())
        assert 0.0 <= packet.extraction_ratio <= 1.0
        # Grid War has multiple extraction morphisms
        assert packet.extraction_ratio > 0.0

    def test_packet_has_vulnerable_count(self):
        graph = _build_conflict_graph()
        packet = OffloadSkeleton.prepare(graph.as_dict())
        assert packet.vulnerable_node_count >= 1

    def test_reintegrate_returns_result(self):
        result = OffloadSkeleton.reintegrate(
            result={"simulated": True},
            sustainability_score=7.8,
        )
        assert result["offloadResult"]["sustainabilityScore"] == 7.8
        assert result["offloadResult"]["source"] == "offloaded_simulation"


# ---------------------------------------------------------------------------
# End-to-end: Full Crystallization pipeline
# ---------------------------------------------------------------------------


class TestFullCrystallizationPipeline:
    def test_full_pipeline_produces_vault(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter(include_mermaid=True)
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=pipeline["exam_result"].sustainability_score,
            fragility_index=pipeline["trace"].mirror_score,
            passed_exam=pipeline["exam_result"].passed,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        # Must have at minimum: Manifesto + 1 insight + 1 projection
        assert result.vault.file_count >= 3

    def test_pipeline_manifesto_links_to_all_sections(self):
        pipeline = _run_full_pipeline()
        exporter = ObsidianExporter()
        result = exporter.crystallize(
            soul=pipeline["soul"],
            sustainability_score=8.0,
            fragility_index=3.0,
            passed_exam=True,
            trace=pipeline["trace"],
            manifesto_dict=pipeline["manifesto_dict"],
            repair_morphisms=pipeline["repair_morphisms"],
        )
        manifesto = result.vault.files["Manifesto.md"]
        # Should have wikilinks to insights
        assert "[[wisdom_001" in manifesto
        # Should have wikilinks to projections
        assert "[[foresight_001" in manifesto
