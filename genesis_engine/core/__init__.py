"""Core modules for the Genesis Engine."""

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective
from genesis_engine.core.axiomlogix import AxiomLogixTranslator, CategoricalGraph, Object, Morphism
from genesis_engine.core.deconstruction_engine import DeconstructionEngine, DisharmonyReport
from genesis_engine.core.dream_engine import DreamEngine, DreamPath, PossibilityReport, PathType
from genesis_engine.core.architectural_forge import ArchitecturalForge, ForgeArtifact, TechnicalCovenant
from genesis_engine.core.continuity_bridge import ContinuityBridge, GenesisSoul, redact_sensitive
from genesis_engine.core.ai_provider import AIProvider, LocalProvider, Candidate, Perspective
from genesis_engine.core.crucible import CrucibleEngine, CrucibleResult, CrucibleCandidate, LogicBox, CandidateStatus
from genesis_engine.core.aria_interface import AriaInterface, AriaRenderer

__all__ = [
    # Axiom Anchor (Module 2.1)
    "AxiomAnchor",
    "PrimeDirective",
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
    "GenesisSoul",
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
]
