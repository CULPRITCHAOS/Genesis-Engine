"""Tests for Module 1.4 — The AxiomLogix Translator."""

from genesis_engine.core.axiomlogix import (
    AxiomLogixTranslator,
    CategoricalGraph,
    Morphism,
    Object,
)


class TestCategoricalGraph:
    def test_add_object(self):
        g = CategoricalGraph()
        obj = g.add_object("User", ["stakeholder"])
        assert obj.label == "User"
        assert len(g.objects) == 1

    def test_add_morphism(self):
        g = CategoricalGraph()
        a = g.add_object("A")
        b = g.add_object("B")
        m = g.add_morphism("Relates", a, b, ["neutral"])
        assert m.source == a.id
        assert m.target == b.id

    def test_add_morphism_by_id(self):
        g = CategoricalGraph()
        a = g.add_object("A")
        b = g.add_object("B")
        m = g.add_morphism("Relates", a.id, b.id)
        assert m.source == a.id

    def test_as_dict(self):
        g = CategoricalGraph(source_text="test")
        g.add_object("X")
        data = g.as_dict()
        assert data["sourceText"] == "test"
        assert len(data["objects"]) == 1

    def test_as_artefact_has_type(self):
        g = CategoricalGraph()
        art = g.as_artefact()
        assert art["type"] == "categorical_graph"


class TestAxiomLogixTranslator:
    def test_extracts_entities_from_profit_safety(self):
        t = AxiomLogixTranslator()
        graph = t.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        labels = {o.label for o in graph.objects}
        assert "User" in labels
        assert "Corporation" in labels or "Policy" in labels
        assert "Profit" in labels
        assert "Safety" in labels

    def test_extracts_morphisms(self):
        t = AxiomLogixTranslator()
        graph = t.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        assert len(graph.morphisms) > 0
        morph_labels = {m.label for m in graph.morphisms}
        assert "Extraction" in morph_labels or "Prioritization" in morph_labels

    def test_harmful_tags_present(self):
        t = AxiomLogixTranslator()
        graph = t.translate(
            "A corporate policy that prioritizes profit over user safety."
        )
        all_tags: set[str] = set()
        for m in graph.morphisms:
            all_tags.update(m.tags)
        assert "extraction" in all_tags or "exploitation" in all_tags

    def test_benign_scenario(self):
        t = AxiomLogixTranslator()
        graph = t.translate(
            "A community health program that empowers workers "
            "and protects the environment."
        )
        labels = {o.label for o in graph.objects}
        assert "Worker" in labels or "Community" in labels
        morph_labels = {m.label for m in graph.morphisms}
        assert "Empowerment" in morph_labels or "Protection" in morph_labels

    def test_empty_input(self):
        t = AxiomLogixTranslator()
        graph = t.translate("")
        assert isinstance(graph, CategoricalGraph)
        assert len(graph.objects) == 0

    def test_source_text_preserved(self):
        t = AxiomLogixTranslator()
        text = "Users need protection."
        graph = t.translate(text)
        assert graph.source_text == text

    def test_no_duplicate_objects(self):
        t = AxiomLogixTranslator()
        graph = t.translate("Users and users and more users.")
        user_count = sum(1 for o in graph.objects if o.label == "User")
        assert user_count == 1
