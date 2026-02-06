"""Core modules for the Genesis Engine."""

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor, AxiomAnchorFrozenError, IncentiveStabilityPredicate,
    PrimeDirective, SustainabilityPredicate, SustainabilityResult,
    MonteCarloProjection,
)
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph, Object, Morphism
from genesis_engine.core.deconstruction_engine import DeconstructionEngine, DisharmonyReport
from genesis_engine.core.dream_engine import DreamEngine, DreamPath, PossibilityReport, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge, ForgeArtifact, TechnicalCovenant
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge, ForesightProjection, GenesisSoul, HumanOverrideEntry,
    OVERRIDE_REASON_CATEGORIES, redact_sensitive,
)
from genesis_engine.core.ai_provider import AIProvider, LocalProvider, Candidate, Perspective
from genesis_engine.core.crucible import CrucibleEngine, CrucibleResult, CrucibleCandidate, LogicBox, CandidateStatus
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer
from genesis_engine.core.game_theory_console import (
    GameTheoryConsole, WarGameOutcome, OutcomeFlag,
    AgentType, AgentState, RoundResult,
)

__all__ = [
    # Axiom Anchor (Module 2.1)
    "AxiomAnchor",
    "AxiomAnchorFrozenError",
    "IncentiveStabilityPredicate",
    "PrimeDirective",
    # Sustainability Predicate (Module 2.1 Extension — Sprint 6.1)
    "SustainabilityPredicate",
    "SustainabilityResult",
    "MonteCarloProjection",
    # AxiomLogix (Module 1.4)
    "AxiomLogixTranslator",
    "CategoricalGraph",
    "Object",
    "Morphism",
    # Deconstruction Engine (Module 1.1)
    "DeconstructionEngine",
    "DisharmonyReport",
    # Dream Engine (Module 1.2)
    "DreamEngine",
    "DreamPath",
    "PossibilityReport",
    "PathType",
    # Architectural Forge (Module 1.3)
    "ArchitecturalForge",
    "ForgeArtifact",
    "TechnicalCovenant",
    # Continuity Bridge (Module 2.3)
    "ContinuityBridge",
    "ForesightProjection",
    "GenesisSoul",
    "HumanOverrideEntry",
    "OVERRIDE_REASON_CATEGORIES",
    "redact_sensitive",
    # AI Provider (Ollama-ready)
    "AIProvider",
    "LocalProvider",
    "Candidate",
    "Perspective",
    # Crucible Engine (Module 3.1)
    "CrucibleEngine",
    "CrucibleResult",
    "CrucibleCandidate",
    "LogicBox",
    "CandidateStatus",
    # Aria Interface (Module 3.2)
    "AriaInterface",
    "AriaRenderer",
    # Game Theory Console (Module 3.5 — Sprint 6.1)
    "GameTheoryConsole",
    "WarGameOutcome",
    "OutcomeFlag",
    "AgentType",
    "AgentState",
    "RoundResult",
]
