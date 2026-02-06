"""Tests for Shadow Entity Inference (Module 1.4 Extension — Sprint 6.1)."""

from genesis_engine.core.axiomlogix import AxiomLogixTranslator


class TestShadowEntityInference:
    """Tests for AxiomLogix inferring Future_Generations and Ecosystem."""

    def test_future_generations_inferred(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that prioritizes profit over user safety."
        )
        labels = [o.label for o in graph.objects]
        assert "Future_Generations" in labels

    def test_ecosystem_inferred_when_no_environment(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that prioritizes profit over user safety."
        )
        labels = [o.label for o in graph.objects]
        # Should have Ecosystem since "environment" not in problem text
        assert "Ecosystem" in labels

    def test_ecosystem_not_inferred_when_environment_present(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that harms the environment and exploits workers."
        )
        labels = [o.label for o in graph.objects]
        # Environment is explicit, so Ecosystem should not be added
        assert "Environment" in labels
        assert "Ecosystem" not in labels

    def test_shadow_entities_have_correct_tags(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that exploits user data."
        )
        for obj in graph.objects:
            if obj.label == "Future_Generations":
                assert "shadow_entity" in obj.tags
                assert "temporal" in obj.tags
                assert "vulnerable" in obj.tags
                assert "stakeholder" in obj.tags
            if obj.label == "Ecosystem":
                assert "shadow_entity" in obj.tags
                assert "ecological" in obj.tags
                assert "vulnerable" in obj.tags

    def test_shadow_entity_morphism_wiring(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that extracts profit from workers."
        )
        # Should have a Temporal_Impact morphism to Future_Generations
        impact_morphisms = [
            m for m in graph.morphisms
            if m.label == "Temporal_Impact"
        ]
        assert len(impact_morphisms) > 0

    def test_ecological_impact_morphism_wiring(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that extracts profit from workers."
        )
        # Should have an Ecological_Impact morphism to Ecosystem
        eco_morphisms = [
            m for m in graph.morphisms
            if m.label == "Ecological_Impact"
        ]
        assert len(eco_morphisms) > 0

    def test_benign_graph_still_gets_shadow_entities(self):
        """Even beneficial systems affect future generations."""
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A community program that protects and serves users."
        )
        labels = [o.label for o in graph.objects]
        assert "Future_Generations" in labels

    def test_shadow_entities_not_duplicated(self):
        translator = AxiomLogixTranslator()
        graph = translator.translate(
            "A corporation that exploits workers and extracts data."
        )
        fg_count = sum(1 for o in graph.objects if o.label == "Future_Generations")
        eco_count = sum(1 for o in graph.objects if o.label == "Ecosystem")
        assert fg_count == 1
        assert eco_count == 1
