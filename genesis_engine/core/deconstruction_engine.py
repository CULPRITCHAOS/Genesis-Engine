"""
Module 1.1 — The Deconstruction Engine

Analyses a ``CategoricalGraph`` for **Disharmony**: morphisms that violate
the Prime Directive (Unity, Compassion, Coherence).

Pipeline
--------
1. Accept a ``CategoricalGraph`` (from the AxiomLogix Translator).
2. Pass the graph's artefact through the **Axiom Anchor** to obtain
   per-principle scores.
3. For every morphism, compute a local disharmony score based on its
   semantic tags.
4. Aggregate results into a ``DisharmonyReport`` with:
   - ``unityImpact``       (0–10 scale)
   - ``compassionDeficit`` (0–10 scale)
   - ``seedPrompt``        (a generative prompt for the Dream Engine)
5. Emit the report as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.axiom_anchor import AxiomAnchor, ValidationResult
from genesis_engine.core.axiomlogix import CategoricalGraph, Morphism


# ---------------------------------------------------------------------------
# Tag classification sets
# ---------------------------------------------------------------------------

_DISHARMONY_TAGS: set[str] = {
    "extraction", "exploitation", "coercion",
    "neglect", "division",
}

_HARMONY_TAGS: set[str] = {
    "service", "protection", "collaboration",
    "empowerment", "care",
}


# ---------------------------------------------------------------------------
# Morphism-level finding
# ---------------------------------------------------------------------------

@dataclass
class MorphismFinding:
    """Analysis of a single morphism's alignment with the Prime Directive."""

    morphism_id: str
    label: str
    source: str
    target: str
    disharmony_score: float  # 0.0 – 1.0
    flags: list[str] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "morphismId": self.morphism_id,
            "label": self.label,
            "source": self.source,
            "target": self.target,
            "disharmonyScore": round(self.disharmony_score, 4),
            "flags": self.flags,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Disharmony Report
# ---------------------------------------------------------------------------

@dataclass
class DisharmonyReport:
    """Top-level report produced by the Deconstruction Engine."""

    source_text: str
    unity_impact: float  # 0 – 10
    compassion_deficit: float  # 0 – 10
    coherence_score: float  # 0 – 10
    is_aligned: bool
    seed_prompt: str
    findings: list[MorphismFinding] = field(default_factory=list)
    validation: ValidationResult | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": {
                "timestamp": self.timestamp,
                "sourceText": self.source_text,
                "primeDirectiveAligned": self.is_aligned,
                "unityImpact": round(self.unity_impact, 2),
                "compassionDeficit": round(self.compassion_deficit, 2),
                "coherenceScore": round(self.coherence_score, 2),
                "seedPrompt": self.seed_prompt,
                "findings": [f.as_dict() for f in self.findings],
                "validation": self.validation.as_dict() if self.validation else None,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Deconstruction Engine
# ---------------------------------------------------------------------------

class DeconstructionEngine:
    """Analyses categorical graphs for disharmony against the Prime Directive.

    Parameters
    ----------
    anchor : AxiomAnchor | None
        An Axiom Anchor instance.  One is created with defaults if omitted.
    """

    def __init__(self, anchor: AxiomAnchor | None = None) -> None:
        self.anchor = anchor or AxiomAnchor()

    # -- public API ---------------------------------------------------------

    def analyse(self, graph: CategoricalGraph) -> DisharmonyReport:
        """Run a full disharmony analysis on *graph* and return a report."""

        artefact = graph.as_artefact()

        # Step 1: Global validation via the Axiom Anchor
        validation = self.anchor.validate(artefact)

        # Step 2: Per-morphism analysis
        findings = [self._analyse_morphism(m, graph) for m in graph.morphisms]

        # Step 3: Compute report-level scores (0 – 10 scale)
        unity_raw = validation.principle_scores.get("unity", 0.5)
        compassion_raw = validation.principle_scores.get("compassion", 0.5)
        coherence_raw = validation.principle_scores.get("coherence", 0.5)

        # Invert unity and compassion so higher numbers = bigger problem
        unity_impact = round((1.0 - unity_raw) * 10, 2)
        compassion_deficit = round((1.0 - compassion_raw) * 10, 2)
        coherence_score = round(coherence_raw * 10, 2)

        # Step 4: Generate a seed prompt for the Dream Engine
        seed_prompt = self._generate_seed_prompt(
            graph, findings, unity_impact, compassion_deficit,
        )

        return DisharmonyReport(
            source_text=graph.source_text,
            unity_impact=unity_impact,
            compassion_deficit=compassion_deficit,
            coherence_score=coherence_score,
            is_aligned=validation.is_aligned,
            seed_prompt=seed_prompt,
            findings=findings,
            validation=validation,
        )

    # -- internal -----------------------------------------------------------

    def _analyse_morphism(
        self, morphism: Morphism, graph: CategoricalGraph,
    ) -> MorphismFinding:
        """Score a single morphism for disharmony."""
        tags = set(t.lower() for t in morphism.tags)
        label = morphism.label.lower()

        negative = len(tags & _DISHARMONY_TAGS) + (1 if label in _DISHARMONY_TAGS else 0)
        positive = len(tags & _HARMONY_TAGS) + (1 if label in _HARMONY_TAGS else 0)
        total = negative + positive

        disharmony_score = negative / total if total else 0.0

        flags: list[str] = []
        if tags & _DISHARMONY_TAGS:
            flags.extend(sorted(tags & _DISHARMONY_TAGS))

        # Determine source/target labels for readable output
        src_label = morphism.source
        tgt_label = morphism.target
        for obj in graph.objects:
            if obj.id == morphism.source:
                src_label = obj.label
            if obj.id == morphism.target:
                tgt_label = obj.label

        recommendation = ""
        if disharmony_score > 0.0:
            recommendation = (
                f"Replace the '{morphism.label}' relationship between "
                f"'{src_label}' and '{tgt_label}' with one that "
                f"serves mutual flourishing."
            )

        return MorphismFinding(
            morphism_id=morphism.id,
            label=morphism.label,
            source=src_label,
            target=tgt_label,
            disharmony_score=disharmony_score,
            flags=flags,
            recommendation=recommendation,
        )

    @staticmethod
    def _generate_seed_prompt(
        graph: CategoricalGraph,
        findings: list[MorphismFinding],
        unity_impact: float,
        compassion_deficit: float,
    ) -> str:
        """Create a generative prompt the Dream Engine can use to heal
        the disharmony."""
        flagged = [f for f in findings if f.disharmony_score > 0]
        if not flagged:
            return (
                "The system is in harmony. Dream forward: "
                "envision how the existing relationships can deepen "
                "mutual benefit and coherence."
            )

        entities = ", ".join(o.label for o in graph.objects)
        problems = "; ".join(
            f"'{f.label}' ({f.source} → {f.target})" for f in flagged
        )

        return (
            f"[DREAM ENGINE INPUT] "
            f"Context entities: [{entities}]. "
            f"Disharmonic morphisms detected: [{problems}]. "
            f"Unity impact: {unity_impact}/10. "
            f"Compassion deficit: {compassion_deficit}/10. "
            f"TASK: Re-imagine these relationships so that every morphism "
            f"serves Love — the recognition of Unity and drive for "
            f"benevolent, coherent outcomes. Output a healed categorical "
            f"graph where extraction is replaced by reciprocity and "
            f"neglect is replaced by care."
        )
