"""Core modules for the Genesis Engine."""

from genesis_engine.core.axiom_anchor import (
    AxiomAnchor, AxiomAnchorFrozenError, IncentiveStabilityPredicate,
    PrimeDirective, SustainabilityPredicate, SustainabilityResult,
    MonteCarloProjection,
)
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph, Object, Morphism
from genesis_engine.core.deconstruction_engine import DeconstructionEngine, DisharmonyReport
from genesis_engine.core.dream_engine import DreamEngine, DreamPath, PossibilityReport, PathType
from genesis_engine.core.architectural_forge import (
    ArchitecturalForge, ForgeArtifact, TechnicalCovenant,
    StewardshipManifesto, RegenerativeLoop, RepairMorphism,
)
from genesis_engine.core.continuity_bridge import (
    ContinuityBridge, ForesightProjection, GenesisSoul, HumanOverrideEntry,
    OVERRIDE_REASON_CATEGORIES, redact_sensitive,
)
from genesis_engine.core.ai_provider import (
    AIProvider, LocalProvider, Candidate, Perspective,
    OllamaProvider, OffloadSkeleton, OffloadPacket, get_default_provider,
)
from genesis_engine.core.crucible import CrucibleEngine, CrucibleResult, CrucibleCandidate, LogicBox, CandidateStatus
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer
from genesis_engine.core.game_theory_console import (
    GameTheoryConsole, WarGameOutcome, OutcomeFlag,
    AgentType, AgentState, RoundResult,
    FinalExam, FinalExamResult,
    BayesianFinalExam, BlackoutShockResult,
    CovenantFinalExam, CovenantExamResult,
)
from genesis_engine.core.mirror_of_truth import (
    MirrorOfTruth, RefinementTrace, CritiqueFinding,
)
from genesis_engine.core.wisdom_mirror import (
    WisdomMirror, MirrorReport, CovenantPatch, DivergencePattern,
)
from genesis_engine.core.obsidian_exporter import (
    ObsidianExporter, ObsidianVault,
    CrystallizationResult, stewardship_frontmatter,
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
    # Architectural Forge (Module 1.3) + Regenerative Blueprinting (Sprint 7)
    "ArchitecturalForge",
    "ForgeArtifact",
    "TechnicalCovenant",
    "StewardshipManifesto",
    "RegenerativeLoop",
    "RepairMorphism",
    # Continuity Bridge (Module 2.3)
    "ContinuityBridge",
    "ForesightProjection",
    "GenesisSoul",
    "HumanOverrideEntry",
    "OVERRIDE_REASON_CATEGORIES",
    "redact_sensitive",
    # AI Provider (Ollama-first, Sprint 9)
    "AIProvider",
    "LocalProvider",
    "OllamaProvider",
    "OffloadSkeleton",
    "OffloadPacket",
    "get_default_provider",
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
    # Game Theory Console (Module 3.5) + Final Exam (Sprint 7)
    "GameTheoryConsole",
    "WarGameOutcome",
    "OutcomeFlag",
    "AgentType",
    "AgentState",
    "RoundResult",
    "FinalExam",
    "FinalExamResult",
    # Bayesian Final Exam (Sprint 8 — Uncertainty Hardening)
    "BayesianFinalExam",
    "BlackoutShockResult",
    "CovenantFinalExam",
    "CovenantExamResult",
    # Mirror of Truth (Module 1.7 — Sprint 8/9)
    "MirrorOfTruth",
    "RefinementTrace",
    "CritiqueFinding",
    # Wisdom Mirror (Module 3.6 — Sprint 7)
    "WisdomMirror",
    "MirrorReport",
    "CovenantPatch",
    "DivergencePattern",
    # Obsidian Exporter (Module 2.3 Extension — Sprint 7/9)
    "ObsidianExporter",
    "ObsidianVault",
    "CrystallizationResult",
    "stewardship_frontmatter",
]
