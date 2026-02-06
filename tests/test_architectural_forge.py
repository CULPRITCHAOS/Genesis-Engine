"""Tests for Module 1.3 — The Architectural Forge."""

import json

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import (
    ArchitecturalForge,
    ForgeArtifact,
    TechnicalCovenant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reinvention_path():
    """Run the full pipeline and return the Reinvention path."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)

    graph = translator.translate(
        "A corporate policy that prioritizes profit over user safety."
    )
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    return next(p for p in possibility.paths if p.path_type == PathType.REINVENTION)


def _get_dissolution_path():
    """Return the Dissolution path for the surveillance scenario."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)

    graph = translator.translate(
        "An AI surveillance system that exploits user data and neglects privacy."
    )
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    return next(p for p in possibility.paths if p.path_type == PathType.DISSOLUTION)


# ---------------------------------------------------------------------------
# Forge core tests
# ---------------------------------------------------------------------------

class TestArchitecturalForge:
    def test_forge_returns_artifact(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert isinstance(artifact, ForgeArtifact)

    def test_artifact_has_covenant(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert isinstance(artifact.covenant, TechnicalCovenant)

    def test_covenant_has_data_models(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert len(artifact.covenant.data_models) > 0

    def test_covenant_has_endpoints(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert len(artifact.covenant.endpoints) > 0

    def test_covenant_has_governance_rules(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert len(artifact.covenant.governance_rules) > 0


# ---------------------------------------------------------------------------
# Technical Covenant structure tests
# ---------------------------------------------------------------------------

class TestTechnicalCovenant:
    def test_data_models_match_graph_objects(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        model_names = {m.name for m in artifact.covenant.data_models}
        object_labels = {o.label for o in path.healed_graph.objects}
        assert model_names == object_labels

    def test_endpoints_match_graph_morphisms(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        ep_names = {e.name for e in artifact.covenant.endpoints}
        morph_labels = {m.label for m in path.healed_graph.morphisms}
        assert ep_names == morph_labels

    def test_endpoints_have_paths(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        for ep in artifact.covenant.endpoints:
            assert ep.path.startswith("/api/v1/")

    def test_endpoints_have_schemas(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        for ep in artifact.covenant.endpoints:
            assert "type" in ep.request_schema
            assert "type" in ep.response_schema

    def test_data_models_have_fields(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        for dm in artifact.covenant.data_models:
            assert len(dm.fields) >= 3  # at least id, name, created_at

    def test_stewardship_title_generated(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert "Stewardship Covenant" in artifact.covenant.title

    def test_dissolution_title_generated(self):
        path = _get_dissolution_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert "Cooperative Covenant" in artifact.covenant.title

    def test_covenant_to_json(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        json_str = artifact.covenant.to_json()
        data = json.loads(json_str)
        assert "technicalCovenant" in data

    def test_protected_entity_gets_consent_field(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        for dm in artifact.covenant.data_models:
            if dm.resource_type == "protected_entity":
                field_names = {f["name"] for f in dm.fields}
                assert "consent_status" in field_names
                assert "rights_manifest" in field_names


# ---------------------------------------------------------------------------
# AxiomLogix Verification tests
# ---------------------------------------------------------------------------

class TestAxiomLogixVerification:
    def test_verification_graph_exists(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert artifact.verification_graph is not None
        assert len(artifact.verification_graph.objects) > 0

    def test_verification_passes(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert artifact.integrity_verified is True

    def test_verification_result_aligned(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert artifact.verification_result.is_aligned is True

    def test_dissolution_also_verified(self):
        path = _get_dissolution_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert artifact.integrity_verified is True

    def test_verification_graph_has_morphisms(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        assert len(artifact.verification_graph.morphisms) > 0


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

class TestForgeArtifactSerialisation:
    def test_to_json(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        json_str = artifact.to_json()
        data = json.loads(json_str)
        assert "forgeArtifact" in data

    def test_as_dict_structure(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        data = artifact.as_dict()
        fa = data["forgeArtifact"]
        assert "covenant" in fa
        assert "verification" in fa
        assert "integrityVerified" in fa
        assert fa["integrityVerified"] is True

    def test_covenant_dict_has_all_sections(self):
        path = _get_reinvention_path()
        forge = ArchitecturalForge()
        artifact = forge.forge(path)
        data = artifact.as_dict()
        cov = data["forgeArtifact"]["covenant"]
        assert "dataModels" in cov
        assert "endpoints" in cov
        assert "governanceRules" in cov
        assert "primeDirective" in cov
