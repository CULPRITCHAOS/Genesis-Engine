"""Tests for the Sustainability Predicate (Module 2.1 Extension — Sprint 6.1)."""

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor,
    MonteCarloProjection,
    SustainabilityPredicate,
    SustainabilityResult,
)


class TestSustainabilityPredicate:
    """Tests for SustainabilityPredicate evaluation."""

    def _make_extractive_artefact(self) -> dict:
        """Graph with extraction and no regeneration."""
        return {
            "type": "categorical_graph",
            "objects": [
                {"id": "corp", "label": "Corporation", "tags": ["stakeholder", "actor"]},
                {"id": "sh", "label": "Shareholder", "tags": ["stakeholder", "sink"]},
                {"id": "emp", "label": "Employee", "tags": ["stakeholder", "vulnerable"]},
            ],
            "morphisms": [
                {
                    "id": "m1", "label": "Extraction",
                    "source": "corp", "target": "emp",
                    "tags": ["extraction", "exploitation"],
                },
                {
                    "id": "m2", "label": "Maximize_Value",
                    "source": "corp", "target": "sh",
                    "tags": ["maximize_value", "fiduciary_duty"],
                },
            ],
        }

    def _make_regenerative_artefact(self) -> dict:
        """Graph with regenerative loops and care morphisms."""
        return {
            "type": "categorical_graph",
            "objects": [
                {"id": "coop", "label": "Cooperative", "tags": ["stakeholder", "actor"]},
                {"id": "comm", "label": "Community", "tags": ["stakeholder", "vulnerable"]},
                {"id": "env", "label": "Environment", "tags": ["value", "vulnerable"]},
            ],
            "morphisms": [
                {
                    "id": "m1", "label": "Care",
                    "source": "coop", "target": "comm",
                    "tags": ["care", "service"],
                },
                {
                    "id": "m2", "label": "Protection",
                    "source": "coop", "target": "env",
                    "tags": ["protection", "sustainability"],
                },
                {
                    "id": "m3", "label": "Support",
                    "source": "comm", "target": "coop",
                    "tags": ["collaboration", "empowerment"],
                },
            ],
        }

    def _make_empty_artefact(self) -> dict:
        return {"type": "categorical_graph", "objects": [], "morphisms": []}

    def test_extractive_graph_low_sustainability(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_extractive_artefact())
        assert isinstance(result, SustainabilityResult)
        assert result.sustainability_score < 5.0
        assert result.is_sustainable is False

    def test_regenerative_graph_high_sustainability(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_regenerative_artefact())
        assert isinstance(result, SustainabilityResult)
        assert result.sustainability_score >= 5.0
        assert result.is_sustainable is True

    def test_ecological_harmony_regenerative(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_regenerative_artefact())
        assert result.ecological_harmony > 0.5

    def test_ecological_harmony_extractive(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_extractive_artefact())
        assert result.ecological_harmony < 0.5

    def test_monte_carlo_projections_exist(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_regenerative_artefact())
        assert len(result.projections) == 3  # t=10, 50, 100
        for proj in result.projections:
            assert isinstance(proj, MonteCarloProjection)
            assert 0.0 <= proj.survival_probability <= 1.0
            assert proj.timesteps in (10, 50, 100)

    def test_temporal_viability_range(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_regenerative_artefact())
        assert 0.0 <= result.temporal_viability <= 1.0

    def test_fragility_factor_range(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_extractive_artefact())
        assert result.fragility_factor >= 0.0

    def test_regenerative_loops_detected(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_regenerative_artefact())
        assert len(result.regenerative_loops) > 0

    def test_depletion_morphisms_detected(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_extractive_artefact())
        assert len(result.depletion_morphisms) > 0

    def test_empty_artefact(self):
        pred = SustainabilityPredicate(seed=42)
        result = pred.evaluate(self._make_empty_artefact())
        assert isinstance(result, SustainabilityResult)
        assert result.ecological_harmony == 0.5  # neutral

    def test_predicate_protocol_returns_0_to_1(self):
        pred = SustainabilityPredicate(seed=42)
        score = pred(self._make_regenerative_artefact())
        assert 0.0 <= score <= 1.0

    def test_predicate_protocol_extractive(self):
        pred = SustainabilityPredicate(seed=42)
        score = pred(self._make_extractive_artefact())
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # below threshold

    def test_seed_reproducibility(self):
        pred1 = SustainabilityPredicate(seed=42)
        pred2 = SustainabilityPredicate(seed=42)
        r1 = pred1.evaluate(self._make_regenerative_artefact())
        r2 = pred2.evaluate(self._make_regenerative_artefact())
        assert r1.sustainability_score == r2.sustainability_score

    def test_register_with_axiom_anchor(self):
        """Sustainability predicate can replace a standard principle predicate."""
        anchor = AxiomAnchor()
        pred = SustainabilityPredicate(seed=42)
        # Register under an existing principle key to verify integration
        anchor.register_predicate("coherence", pred)
        result = anchor.validate(self._make_regenerative_artefact())
        assert "coherence" in result.principle_scores
        # The score should be the sustainability predicate's output (0.0–1.0)
        assert result.principle_scores["coherence"] > 0.0
