"""Tests for Module 2.1 — The Axiom Anchor."""

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor,
    DirectivePrinciple,
    PrimeDirective,
    ValidationResult,
)


class TestPrimeDirective:
    def test_default_statement(self):
        d = PrimeDirective()
        assert d.statement == "Does this serve Love?"

    def test_default_principles(self):
        d = PrimeDirective()
        assert DirectivePrinciple.UNITY in d.principles
        assert DirectivePrinciple.COMPASSION in d.principles
        assert DirectivePrinciple.COHERENCE in d.principles

    def test_as_dict(self):
        d = PrimeDirective()
        data = d.as_dict()
        assert data["statement"] == "Does this serve Love?"
        assert "unity" in data["principles"]

    def test_immutable(self):
        d = PrimeDirective()
        try:
            d.statement = "changed"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestAxiomAnchor:
    def _make_aligned_artefact(self) -> dict:
        return {
            "type": "categorical_graph",
            "objects": [
                {"id": "a", "label": "Community", "tags": ["stakeholder", "vulnerable"]},
                {"id": "b", "label": "Organisation", "tags": ["stakeholder", "actor"]},
            ],
            "morphisms": [
                {
                    "id": "m1", "label": "Protection",
                    "source": "b", "target": "a",
                    "tags": ["protection"],
                },
                {
                    "id": "m2", "label": "Service",
                    "source": "b", "target": "a",
                    "tags": ["service"],
                },
            ],
        }

    def _make_misaligned_artefact(self) -> dict:
        return {
            "type": "categorical_graph",
            "objects": [
                {"id": "a", "label": "User", "tags": ["stakeholder", "vulnerable"]},
                {"id": "b", "label": "Corporation", "tags": ["stakeholder", "actor"]},
            ],
            "morphisms": [
                {
                    "id": "m1", "label": "Extraction",
                    "source": "b", "target": "a",
                    "tags": ["extraction", "exploitation"],
                },
            ],
        }

    def test_aligned_artefact_passes(self):
        anchor = AxiomAnchor()
        result = anchor.validate(self._make_aligned_artefact())
        assert result.is_aligned is True
        assert result.coherence_score > 0.5

    def test_misaligned_artefact_fails(self):
        anchor = AxiomAnchor()
        result = anchor.validate(self._make_misaligned_artefact())
        assert result.is_aligned is False
        assert result.principle_scores["unity"] < 0.5

    def test_validation_result_has_reasoning(self):
        anchor = AxiomAnchor()
        result = anchor.validate(self._make_misaligned_artefact())
        assert len(result.reasoning) > 0
        assert any("MISALIGNED" in r for r in result.reasoning)

    def test_custom_threshold(self):
        anchor = AxiomAnchor(alignment_threshold=0.0)
        result = anchor.validate(self._make_misaligned_artefact())
        # With threshold 0, everything passes
        assert result.is_aligned is True

    def test_register_custom_predicate(self):
        anchor = AxiomAnchor()
        anchor.register_predicate("unity", lambda a: 1.0)
        result = anchor.validate(self._make_misaligned_artefact())
        assert result.principle_scores["unity"] == 1.0

    def test_empty_artefact(self):
        anchor = AxiomAnchor()
        result = anchor.validate({"type": "empty", "objects": [], "morphisms": []})
        assert isinstance(result, ValidationResult)

    def test_as_dict_format(self):
        anchor = AxiomAnchor()
        result = anchor.validate(self._make_aligned_artefact())
        data = result.as_dict()
        assert "isAligned" in data
        assert "coherenceScore" in data
        assert "principleScores" in data
        assert "reasoning" in data
