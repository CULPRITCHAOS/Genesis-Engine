"""Tests for Module 1.3 updates — Regenerative Blueprinting (Sprint 7).

Tests the Stewardship Manifesto, RegenerativeLoop, and RepairMorphism
additions to the Architectural Forge.
"""

import json

from genesis_engine.core.axiom_anchor import AxiomAnchor
from genesis_engine.core.axiomlogix import AxiomLogixTranslator
from genesis_engine.core.deconstruction_engine import DeconstructionEngine
from genesis_engine.core.dream_engine import DreamEngine, PathType
from genesis_engine.core.architectural_forge import (
    ArchitecturalForge,
    ForgeArtifact,
    RegenerativeLoop,
    RepairMorphism,
    StewardshipManifesto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reinvention_artifact() -> ForgeArtifact:
    """Run the full pipeline and return a ForgeArtifact from Reinvention path."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor, translator=translator)

    graph = translator.translate(
        "A corporate policy that prioritizes profit over user safety."
    )
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    path = next(p for p in possibility.paths if p.path_type == PathType.REINVENTION)
    return forge.forge(path)


def _get_dissolution_artifact() -> ForgeArtifact:
    """Return a ForgeArtifact from the Dissolution path."""
    anchor = AxiomAnchor()
    translator = AxiomLogixTranslator()
    decon = DeconstructionEngine(anchor=anchor)
    dream = DreamEngine(anchor=anchor)
    forge = ArchitecturalForge(anchor=anchor, translator=translator)

    graph = translator.translate(
        "An AI surveillance system that exploits user data and neglects privacy."
    )
    report = decon.analyse(graph)
    possibility = dream.dream(report, graph)
    path = next(p for p in possibility.paths if p.path_type == PathType.DISSOLUTION)
    return forge.forge(path)


# ---------------------------------------------------------------------------
# Stewardship Manifesto existence tests
# ---------------------------------------------------------------------------

class TestStewardshipManifestoExists:
    def test_artifact_has_manifesto(self):
        artifact = _get_reinvention_artifact()
        assert artifact.manifesto is not None
        assert isinstance(artifact.manifesto, StewardshipManifesto)

    def test_dissolution_has_manifesto(self):
        artifact = _get_dissolution_artifact()
        assert artifact.manifesto is not None

    def test_manifesto_in_as_dict(self):
        artifact = _get_reinvention_artifact()
        d = artifact.as_dict()
        assert "stewardshipManifesto" in d["forgeArtifact"]


# ---------------------------------------------------------------------------
# Stewardship Manifesto structure tests
# ---------------------------------------------------------------------------

class TestStewardshipManifestoStructure:
    def test_manifesto_has_title(self):
        artifact = _get_reinvention_artifact()
        assert "Stewardship Covenant" in artifact.manifesto.covenant_title

    def test_manifesto_has_alignment_scores(self):
        artifact = _get_reinvention_artifact()
        scores = artifact.manifesto.alignment_scores
        assert "unity" in scores
        assert "compassion" in scores
        assert "coherence" in scores

    def test_manifesto_has_governance_summary(self):
        artifact = _get_reinvention_artifact()
        assert len(artifact.manifesto.governance_summary) > 0

    def test_manifesto_has_prime_directive(self):
        artifact = _get_reinvention_artifact()
        assert artifact.manifesto.prime_directive == "Does this serve Love?"

    def test_manifesto_has_integrity_hash(self):
        artifact = _get_reinvention_artifact()
        assert len(artifact.manifesto.integrity_hash) == 64  # SHA-256

    def test_manifesto_as_dict(self):
        artifact = _get_reinvention_artifact()
        d = artifact.manifesto.as_dict()
        assert "stewardshipManifesto" in d
        sm = d["stewardshipManifesto"]
        assert "covenantTitle" in sm
        assert "alignmentScores" in sm
        assert "regenerativeLoop" in sm
        assert "integrityHash" in sm


# ---------------------------------------------------------------------------
# Regenerative Loop tests
# ---------------------------------------------------------------------------

class TestRegenerativeLoop:
    def test_manifesto_has_regenerative_loop(self):
        artifact = _get_reinvention_artifact()
        regen = artifact.manifesto.regenerative_loop
        assert isinstance(regen, RegenerativeLoop)

    def test_loop_threshold_is_5(self):
        artifact = _get_reinvention_artifact()
        assert artifact.manifesto.regenerative_loop.score_threshold == 5.0

    def test_loop_has_repair_morphisms(self):
        artifact = _get_reinvention_artifact()
        repairs = artifact.manifesto.regenerative_loop.repair_morphisms
        assert len(repairs) > 0

    def test_repair_morphism_structure(self):
        artifact = _get_reinvention_artifact()
        for rm in artifact.manifesto.regenerative_loop.repair_morphisms:
            assert isinstance(rm, RepairMorphism)
            assert rm.trigger_condition
            assert rm.source_model
            assert rm.target_model
            assert rm.repair_action
            assert len(rm.replacement_tags) > 0

    def test_repair_morphism_as_dict(self):
        artifact = _get_reinvention_artifact()
        rm = artifact.manifesto.regenerative_loop.repair_morphisms[0]
        d = rm.as_dict()
        assert "triggerCondition" in d
        assert "repairAction" in d
        assert "replacementTags" in d

    def test_loop_has_resurrection_principle(self):
        artifact = _get_reinvention_artifact()
        principle = artifact.manifesto.regenerative_loop.resurrection_principle
        assert "Unity over Power" in principle

    def test_loop_as_dict(self):
        artifact = _get_reinvention_artifact()
        d = artifact.manifesto.regenerative_loop.as_dict()
        assert "scoreThreshold" in d
        assert "repairMorphisms" in d
        assert "resurrectionPrinciple" in d

    def test_loop_to_yaml_block(self):
        artifact = _get_reinvention_artifact()
        yaml_str = artifact.manifesto.regenerative_loop.to_yaml_block()
        assert "regenerative_loop:" in yaml_str
        assert "score_threshold:" in yaml_str
        assert "repair_morphisms:" in yaml_str


# ---------------------------------------------------------------------------
# Manifesto Markdown rendering tests
# ---------------------------------------------------------------------------

class TestManifestoMarkdown:
    def test_manifesto_to_markdown(self):
        artifact = _get_reinvention_artifact()
        md = artifact.manifesto.to_markdown()
        assert md.startswith("---")
        assert "# " in md  # Has a heading

    def test_markdown_has_frontmatter(self):
        artifact = _get_reinvention_artifact()
        md = artifact.manifesto.to_markdown()
        # Should have opening and closing ---
        parts = md.split("---")
        assert len(parts) >= 3  # before ---, frontmatter, after ---

    def test_markdown_has_alignment_scores(self):
        artifact = _get_reinvention_artifact()
        md = artifact.manifesto.to_markdown()
        assert "Alignment Scores" in md
        assert "unity" in md

    def test_markdown_has_regenerative_loop(self):
        artifact = _get_reinvention_artifact()
        md = artifact.manifesto.to_markdown()
        assert "Regenerative Loop" in md
        assert "Repair Morphisms" in md

    def test_markdown_has_governance(self):
        artifact = _get_reinvention_artifact()
        md = artifact.manifesto.to_markdown()
        assert "Governance Summary" in md


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_existing_forge_tests_pass(self):
        """Ensure the forge still works exactly as before for existing callers."""
        artifact = _get_reinvention_artifact()
        # All existing properties should still work
        assert artifact.covenant is not None
        assert artifact.verification_graph is not None
        assert artifact.verification_result is not None
        assert artifact.integrity_verified is True

    def test_existing_as_dict_structure(self):
        """as_dict should still have all the original fields."""
        artifact = _get_reinvention_artifact()
        d = artifact.as_dict()
        fa = d["forgeArtifact"]
        assert "covenant" in fa
        assert "verification" in fa
        assert "integrityVerified" in fa
