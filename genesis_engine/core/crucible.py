"""
Module 3.1 — The Crucible Engine

Implements the Dual-Memory Crucible architecture:

* **LogicBox** — an ephemeral sandbox holding multiple ``CategoricalGraph``
  candidates being evaluated.  Candidates are generated from multiple
  perspectives (Causality, Contradiction, Analogy) and compete for
  confirmation.
* **EternalBox** — the persistent ``.genesis_soul`` state managed by the
  Continuity Bridge.

The 6-Phase Workflow
--------------------
1. **Ingest**         — Accept a problem statement and translate it.
2. **Retrieval**      — Load relevant wisdom from the EternalBox.
3. **Divergence**     — Generate multiple candidates via perspectives.
4. **Verification**   — Validate each candidate through the Axiom Anchor.
5. **Convergence**    — Rank candidates and select the best.
6. **Crystallization** — Commit the confirmed candidate to the EternalBox.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from genesis_engine.core.ai_provider import AIProvider, LocalProvider, Perspective
from genesis_engine.core.architectural_forge import ArchitecturalForge, ForgeArtifact
from genesis_engine.core.axiom_anchor import AxiomAnchor, ValidationResult
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul
from genesis_engine.core.deconstruction_engine import DeconstructionEngine, DisharmonyReport
from genesis_engine.core.dream_engine import DreamEngine, DreamPath, PathType, PossibilityReport


# ---------------------------------------------------------------------------
# Candidate status lifecycle
# ---------------------------------------------------------------------------

class CandidateStatus(Enum):
    """Lifecycle states for a Crucible candidate."""

    PENDING = "PENDING"
    VERIFYING = "VERIFYING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# CrucibleCandidate — a single entry in the LogicBox
# ---------------------------------------------------------------------------

@dataclass
class CrucibleCandidate:
    """A candidate solution being evaluated in the LogicBox."""

    id: str = field(default_factory=lambda: f"cand-{uuid.uuid4().hex[:8]}")
    perspective: Perspective = Perspective.CAUSALITY
    reasoning: str = ""
    content: str = ""
    confidence: float = 0.5

    # The dream path chosen for this candidate
    dream_path: DreamPath | None = None

    # Validation results
    status: CandidateStatus = CandidateStatus.PENDING
    unity_alignment_score: float = 0.0
    validation: ValidationResult | None = None

    # Forge artifact (populated during crystallization)
    artifact: ForgeArtifact | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "perspective": self.perspective.value,
            "status": self.status.value,
            "reasoning": self.reasoning,
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "unityAlignmentScore": round(self.unity_alignment_score, 4),
            "dreamPath": self.dream_path.as_dict() if self.dream_path else None,
            "validation": self.validation.as_dict() if self.validation else None,
        }


# ---------------------------------------------------------------------------
# LogicBox — ephemeral sandbox
# ---------------------------------------------------------------------------

@dataclass
class LogicBox:
    """Ephemeral sandbox holding candidates under evaluation."""

    candidates: list[CrucibleCandidate] = field(default_factory=list)

    def add(self, candidate: CrucibleCandidate) -> None:
        self.candidates.append(candidate)

    def clear(self) -> None:
        self.candidates.clear()

    @property
    def confirmed(self) -> list[CrucibleCandidate]:
        return [c for c in self.candidates if c.status == CandidateStatus.CONFIRMED]

    @property
    def best(self) -> CrucibleCandidate | None:
        confirmed = self.confirmed
        if not confirmed:
            return None
        return max(confirmed, key=lambda c: (c.unity_alignment_score, c.confidence))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidateCount": len(self.candidates),
            "candidates": [c.as_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Phase record — for the Aria interface to display
# ---------------------------------------------------------------------------

@dataclass
class PhaseRecord:
    """Record of a single phase execution for audit/display."""

    phase: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "summary": self.summary,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CrucibleResult — output of a full 6-phase run
# ---------------------------------------------------------------------------

@dataclass
class CrucibleResult:
    """Complete output of a Crucible processing run."""

    source_text: str
    phases: list[PhaseRecord] = field(default_factory=list)
    graph: CategoricalGraph | None = None
    disharmony_report: DisharmonyReport | None = None
    possibility_report: PossibilityReport | None = None
    logic_box: LogicBox = field(default_factory=LogicBox)
    crystallized_candidate: CrucibleCandidate | None = None
    is_aligned: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sourceText": self.source_text,
            "isAligned": self.is_aligned,
            "phases": [p.as_dict() for p in self.phases],
            "logicBox": self.logic_box.as_dict(),
            "crystallizedCandidate": (
                self.crystallized_candidate.as_dict()
                if self.crystallized_candidate else None
            ),
        }


# ---------------------------------------------------------------------------
# Crucible Engine
# ---------------------------------------------------------------------------

class CrucibleEngine:
    """Orchestrates the 6-Phase reasoning workflow.

    Parameters
    ----------
    anchor : AxiomAnchor | None
        Shared Axiom Anchor instance.
    soul : GenesisSoul | None
        The persistent EternalBox state.
    provider : AIProvider | None
        AI provider for multi-perspective candidate generation.
    """

    def __init__(
        self,
        anchor: AxiomAnchor | None = None,
        soul: GenesisSoul | None = None,
        provider: AIProvider | None = None,
    ) -> None:
        self.anchor = anchor or AxiomAnchor()
        self.soul = soul or ContinuityBridge.create_soul(self.anchor)
        self.provider = provider or LocalProvider()

        # Sub-engines sharing the same anchor
        self.translator = AxiomLogixTranslator()
        self.decon = DeconstructionEngine(anchor=self.anchor)
        self.dream = DreamEngine(anchor=self.anchor)
        self.forge = ArchitecturalForge(anchor=self.anchor)
        self.bridge = ContinuityBridge()

    # -- public API ---------------------------------------------------------

    def process(self, problem_text: str) -> CrucibleResult:
        """Run the full 6-phase Crucible workflow on *problem_text*."""
        result = CrucibleResult(source_text=problem_text)

        # Phase 1: Ingest
        graph = self._phase_ingest(problem_text, result)

        # Phase 2: Retrieval
        self._phase_retrieval(problem_text, result)

        # Phase 3: Divergence (skip if already aligned)
        report = self.decon.analyse(graph)
        result.graph = graph
        result.disharmony_report = report

        if report.is_aligned:
            result.is_aligned = True
            result.phases.append(PhaseRecord(
                phase="3-divergence",
                summary="System is aligned — no divergence needed.",
            ))
            result.phases.append(PhaseRecord(
                phase="4-verification", summary="Skipped (aligned).",
            ))
            result.phases.append(PhaseRecord(
                phase="5-convergence", summary="Skipped (aligned).",
            ))
            result.phases.append(PhaseRecord(
                phase="6-crystallization", summary="Skipped (aligned).",
            ))
            self.soul.record_graph(graph, "translation", problem_text)
            self.soul.record_wisdom(report, "aligned", "System is in harmony.")
            return result

        # Generate Dream paths
        possibility = self.dream.dream(report, graph)
        result.possibility_report = possibility

        self._phase_divergence(possibility, graph, result)

        # Phase 4: Verification
        self._phase_verification(result)

        # Phase 5: Convergence
        self._phase_convergence(result)

        # Phase 6: Crystallization
        self._phase_crystallization(report, result)

        return result

    # -- Phase 1: Ingest ----------------------------------------------------

    def _phase_ingest(
        self, problem_text: str, result: CrucibleResult,
    ) -> CategoricalGraph:
        """Translate natural language into a CategoricalGraph."""
        graph = self.translator.translate(problem_text)

        result.phases.append(PhaseRecord(
            phase="1-ingest",
            summary=f"Translated '{problem_text[:60]}...' into {len(graph.objects)} objects, {len(graph.morphisms)} morphisms.",
            details={
                "objectCount": len(graph.objects),
                "morphismCount": len(graph.morphisms),
            },
        ))
        return graph

    # -- Phase 2: Retrieval -------------------------------------------------

    def _phase_retrieval(
        self, problem_text: str, result: CrucibleResult,
    ) -> list[dict[str, Any]]:
        """Search EternalBox for relevant prior wisdom."""
        relevant: list[dict[str, Any]] = []
        lower = problem_text.lower()

        for entry in self.soul.wisdom_log:
            # Simple keyword overlap check
            entry_words = set(entry.source_text.lower().split())
            query_words = set(lower.split())
            overlap = len(entry_words & query_words)
            if overlap >= 2:
                relevant.append(entry.as_dict())

        result.phases.append(PhaseRecord(
            phase="2-retrieval",
            summary=f"Retrieved {len(relevant)} relevant wisdom entries from EternalBox.",
            details={"priorWisdom": relevant},
        ))
        return relevant

    # -- Phase 3: Divergence ------------------------------------------------

    def _phase_divergence(
        self,
        possibility: PossibilityReport,
        graph: CategoricalGraph,
        result: CrucibleResult,
    ) -> None:
        """Generate multiple candidates from different perspectives."""
        artefact = graph.as_artefact()
        context = {
            "source_text": graph.source_text,
            "morphisms": artefact.get("morphisms", []),
            "objects": artefact.get("objects", []),
        }

        ai_candidates = self.provider.generate_candidates(context)

        # Map perspectives to dream paths:
        # Causality  → Reform   (fix the root cause)
        # Contradiction → Reinvention (resolve the contradiction structurally)
        # Analogy → Dissolution (apply the known pattern-break)
        perspective_path_map = {
            Perspective.CAUSALITY: PathType.REFORM,
            Perspective.CONTRADICTION: PathType.REINVENTION,
            Perspective.ANALOGY: PathType.DISSOLUTION,
        }

        paths_by_type = {p.path_type: p for p in possibility.paths}

        for ai_cand in ai_candidates:
            target_type = perspective_path_map.get(ai_cand.perspective, PathType.REFORM)
            dream_path = paths_by_type.get(target_type)

            crucible_cand = CrucibleCandidate(
                perspective=ai_cand.perspective,
                reasoning=ai_cand.reasoning,
                content=ai_cand.content,
                confidence=ai_cand.confidence,
                dream_path=dream_path,
                status=CandidateStatus.PENDING,
            )
            result.logic_box.add(crucible_cand)

        result.phases.append(PhaseRecord(
            phase="3-divergence",
            summary=f"Generated {len(result.logic_box.candidates)} candidates from {len(ai_candidates)} perspectives.",
            details={
                "perspectives": [c.perspective.value for c in ai_candidates],
                "provider": self.provider.provider_name,
            },
        ))

    # -- Phase 4: Verification ----------------------------------------------

    def _phase_verification(self, result: CrucibleResult) -> None:
        """Validate each candidate through the Axiom Anchor."""
        for cand in result.logic_box.candidates:
            cand.status = CandidateStatus.VERIFYING

            if cand.dream_path and cand.dream_path.validation:
                cand.validation = cand.dream_path.validation
                cand.unity_alignment_score = cand.dream_path.unity_alignment_score

                if cand.validation.is_aligned:
                    cand.status = CandidateStatus.CONFIRMED
                else:
                    cand.status = CandidateStatus.REJECTED
            else:
                cand.status = CandidateStatus.REJECTED
                cand.unity_alignment_score = 0.0

        confirmed_count = len(result.logic_box.confirmed)
        result.phases.append(PhaseRecord(
            phase="4-verification",
            summary=f"Verified {len(result.logic_box.candidates)} candidates: {confirmed_count} confirmed.",
            details={
                "results": [
                    {"id": c.id, "status": c.status.value, "score": c.unity_alignment_score}
                    for c in result.logic_box.candidates
                ],
            },
        ))

    # -- Phase 5: Convergence -----------------------------------------------

    def _phase_convergence(self, result: CrucibleResult) -> None:
        """Rank confirmed candidates and select the best."""
        best = result.logic_box.best

        if best:
            result.phases.append(PhaseRecord(
                phase="5-convergence",
                summary=(
                    f"Converged on candidate {best.id} "
                    f"({best.perspective.value}) with unity={best.unity_alignment_score:.4f}."
                ),
                details={"selectedCandidate": best.id, "perspective": best.perspective.value},
            ))
        else:
            result.phases.append(PhaseRecord(
                phase="5-convergence",
                summary="No candidates confirmed — convergence failed.",
            ))

    # -- Phase 6: Crystallization -------------------------------------------

    def _phase_crystallization(
        self,
        report: DisharmonyReport,
        result: CrucibleResult,
    ) -> None:
        """Commit the best candidate to the EternalBox."""
        best = result.logic_box.best

        if not best or not best.dream_path:
            result.phases.append(PhaseRecord(
                phase="6-crystallization",
                summary="No confirmed candidate to crystallize.",
            ))
            return

        # Forge a Technical Covenant from the dream path
        artifact = self.forge.forge(best.dream_path)
        best.artifact = artifact

        # Record into the soul (EternalBox)
        if result.graph:
            self.soul.record_graph(result.graph, "translation", result.source_text)

        self.soul.record_graph(
            best.dream_path.healed_graph,
            "crucible",
            f"Crystallized: {best.perspective.value} perspective",
        )

        if artifact.verification_graph:
            self.soul.record_graph(
                artifact.verification_graph,
                "verification",
                f"Forge verification: {artifact.covenant.title}",
            )

        self.soul.record_forge_artifact({
            "covenantTitle": artifact.covenant.title,
            "integrityVerified": artifact.integrity_verified,
            "perspective": best.perspective.value,
            "candidateId": best.id,
        })

        self.soul.record_wisdom(
            report,
            resolution_path=best.dream_path.path_type.value,
            resolution_summary=(
                f"Crystallized via {best.perspective.value} perspective. "
                f"Unity alignment: {best.unity_alignment_score:.4f}."
            ),
            covenant_title=artifact.covenant.title,
        )

        result.crystallized_candidate = best

        result.phases.append(PhaseRecord(
            phase="6-crystallization",
            summary=(
                f"Crystallized candidate {best.id} into Technical Covenant "
                f"'{artifact.covenant.title}'. Saved to EternalBox."
            ),
            details={
                "candidateId": best.id,
                "covenantTitle": artifact.covenant.title,
                "integrityVerified": artifact.integrity_verified,
            },
        ))
