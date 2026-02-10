# Genesis Engine -- Startup Guide

A comprehensive guide to setting up, running, and using the Genesis Engine: a high-coherence multi-agent system for AI-driven governance and policy analysis.

**Version:** 0.1.0 | **Python:** 3.10+ | **Sprints Completed:** 1--11

---

## Table of Contents

1. [What Is the Genesis Engine?](#what-is-the-genesis-engine)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Architecture Overview](#architecture-overview)
5. [Core Modules Reference](#core-modules-reference)
6. [The 6-Phase Crucible Workflow](#the-6-phase-crucible-workflow)
7. [Running the Demos](#running-the-demos)
8. [Running the Tests](#running-the-tests)
9. [Sprint History](#sprint-history)
10. [Key Concepts & Glossary](#key-concepts--glossary)
11. [Scenario Files](#scenario-files)
12. [Output Artifacts](#output-artifacts)
13. [API Quick Reference](#api-quick-reference)

---

## What Is the Genesis Engine?

The Genesis Engine is a Python framework that analyses policies, governance structures, and institutional designs through the lens of **categorical reasoning** and **multi-perspective analysis**. It operates under a single Prime Directive -- *"Does this serve Love?"* -- and uses game theory, adversarial debate, Bayesian robustness testing, and regenerative repair to evaluate whether a given policy or system promotes genuine well-being or conceals extractive harm.

The engine can:

- Translate natural-language problem statements into **categorical graphs** (objects + morphisms)
- Deconstruct policies to detect **disharmony**, **cost-shifting**, and **incentive instability**
- Generate three solution paths (Reform, Reinvention, Dissolution) and critique them adversarially
- Run **100-round economic war games** to simulate long-term sustainability
- Forge **Technical Covenants** with governance rules, data models, and regenerative repair loops
- Export results to **Obsidian-compatible Markdown vaults** with bidirectional linking
- Maintain a **hash-chained persistent memory** (`.genesis_soul`) across sessions

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) Ollama for local LLM integration

### Installation

```bash
# Clone the repository
git clone https://github.com/CULPRITCHAOS/Genesis-Engine.git
cd Genesis-Engine

# Install the project (editable mode recommended for development)
pip install -e .

# Install test dependencies
pip install pytest
```

### Verify Installation

```bash
# Run the full test suite (596 tests)
python -m pytest tests/

# Run the main demo
python main.py
```

### First Run

The fastest way to see the engine in action:

```bash
# Sprint 4 demo -- processes 3 sample problems through the full Crucible pipeline
python main.py
```

This will:
1. Translate each problem statement into a categorical graph
2. Run the 6-phase Crucible workflow
3. Display the Aria CLI visualization
4. Export reports to `genesis_engine/reports/`
5. Generate a `.genesis_soul` file with hash-chained wisdom

---

## Project Structure

```
Genesis-Engine/
|
|-- genesis_engine/                 # Main package
|   |-- __init__.py                 # Package metadata (v0.1.0)
|   |-- core/                       # All core modules (18 files, ~14,000 LOC)
|   |   |-- __init__.py             # Public API exports
|   |   |-- axiom_anchor.py         # Module 2.1 -- Prime Directive Validator
|   |   |-- axiomlogix.py           # Module 1.4 -- Categorical Graph Translator
|   |   |-- deconstruction_engine.py# Module 1.1 -- Disharmony Analysis
|   |   |-- dream_engine.py         # Module 1.2 -- Multi-Perspective Path Generation
|   |   |-- architectural_forge.py  # Module 1.3 -- Covenant Synthesis
|   |   |-- continuity_bridge.py    # Module 2.3 -- Persistent Memory (.genesis_soul)
|   |   |-- ai_provider.py          # AI Backend Abstraction (Ollama + Local)
|   |   |-- crucible.py             # Module 3.1 -- 6-Phase Reasoning Engine
|   |   |-- aria_interface.py       # Module 3.2 -- CLI Visualization
|   |   |-- game_theory_console.py  # Module 3.5 -- War Games & Final Exams
|   |   |-- mirror_of_truth.py      # Module 1.7 -- Adversarial Critique
|   |   |-- wisdom_mirror.py        # Module 3.6 -- Override Pattern Learning
|   |   |-- obsidian_exporter.py    # Module 2.3 ext -- Markdown Vault Export
|   |   |-- robustness_harness.py   # Module 3.5 ext -- Bayesian Robustness & Repair
|   |   |-- governance_report.py    # Module 3.6 ext -- Sovereignty Reports
|   |   |-- policy_kernel.py        # Module 2.1 ext -- Constitutional Principles
|   |   |-- adversarial_evaluator.py# Module 3.5 ext -- FAIRGAME Debate Arena
|   |   |
|   |-- ai/                         # AI subsystem
|   |   |-- providers/
|   |       |-- ollama_provider.py  # Ollama integration (local LLM)
|   |
|   |-- reports/                    # Generated output artifacts
|       |-- *.json                  # Crucible results, disharmony reports
|       |-- *.genesis_soul          # Persistent hash-chained wisdom logs
|       |-- obsidian_vault/         # Markdown knowledge graph
|
|-- tests/                          # Test suite (23 files, 596 tests, ~7,300 LOC)
|-- scenarios/                      # Pre-built conflict scenarios
|   |-- grid_war_2026.json          # The Oklahoma Grid War scenario (v4.0.0)
|
|-- main.py                         # Sprint 4 demo: Crucible + Aria Interface
|-- demo_sprint9.py                 # Sovereign Synthesis & Crystallization
|-- demo_sprint10.py                # Sovereign Governance & Oklahoma Grid War
|-- demo_grid_war.py                # Sprint 8: Full Grid War pipeline
|-- demo_wisdom_mirror.py           # Sprint 7: Wisdom Mirror & Regenerative Blueprinting
|-- demo_time_guardian.py           # Sprint 6.1: Sustainability Axiom & Game Theory
|-- demo_shareholder_primacy.py     # Incentive Stability Detection
|-- demo_human_override.py          # Sprint 5: Human Override Protocol
|-- pyproject.toml                  # Build configuration
```

---

## Architecture Overview

### Dual-Memory Architecture

The engine operates with two memory spaces:

- **LogicBox** (ephemeral): A sandbox where candidate solutions are evaluated, scored, and discarded. Nothing persists here between runs.
- **EternalBox** (persistent): The `.genesis_soul` file -- a hash-chained, tamper-evident wisdom log that accumulates insights across sessions.

### The Prime Directive

Every evaluation flows through the **Axiom Anchor** (Module 2.1), which enforces the Prime Directive: *"Does this serve Love?"*. Policies are scored on:

| Principle | What It Measures |
|-----------|-----------------|
| Unity | Does the policy unite or divide stakeholders? |
| Compassion | Does it protect the vulnerable? |
| Coherence | Is the internal logic consistent? |
| Sustainability | Will it survive 100+ years of game-theoretic pressure? |
| Incentive Stability | Are the economic incentives aligned or extractive? |

### Module Map

```
                    Natural Language Problem
                            |
                    [1.4 AxiomLogix] -----------> CategoricalGraph
                            |
                    [1.1 Deconstruction] -------> DisharmonyReport
                            |
                    [1.2 Dream Engine] ---------> 3 Paths (Reform/Reinvention/Dissolution)
                            |
                    [1.7 Mirror of Truth] ------> Adversarial Critique
                            |
                    [1.3 Architectural Forge] --> TechnicalCovenant + Manifesto
                            |
                    [3.5 Game Theory] ----------> Final Exam (100-round war game)
                            |
                    [2.1 Axiom Anchor] ---------> Pass/Fail Validation
                            |
                    [2.3 Continuity Bridge] ----> .genesis_soul (persistent)
                            |
                    [3.2 Aria Interface] -------> CLI Visualization
```

---

## Core Modules Reference

### Module 1.1 -- Deconstruction Engine
**File:** `genesis_engine/core/deconstruction_engine.py`
**Sprint:** 1

Analyses a categorical graph and produces a `DisharmonyReport` with scores for unity impact, compassion deficit, coherence, and incentive stability. Identifies structural disharmony, cost-shifting patterns, and extractive relationships.

```python
from genesis_engine.core import DeconstructionEngine, AxiomAnchor

anchor = AxiomAnchor()
decon = DeconstructionEngine(anchor=anchor)
report = decon.analyse(graph)

print(report.unity_impact)           # 0-10 scale
print(report.compassion_deficit)     # 0-10 scale
print(report.incentive_instability)  # True/False
```

### Module 1.2 -- Dream Engine
**File:** `genesis_engine/core/dream_engine.py`
**Sprint:** 2

Generates three solution paths from a disharmony report:

- **Reform**: Incremental improvement within existing structures
- **Reinvention**: Structural redesign that preserves some elements
- **Dissolution**: Complete replacement with a new model

```python
from genesis_engine.core import DreamEngine

dream = DreamEngine(anchor=anchor)
possibility = dream.dream(report, graph)

for path in possibility.paths:
    print(f"{path.path_type.value}: unity={path.unity_alignment_score:.2f}")
print(f"Recommended: {possibility.recommended_path}")
```

### Module 1.3 -- Architectural Forge
**File:** `genesis_engine/core/architectural_forge.py`
**Sprints:** 3, 7

Synthesizes a selected dream path into a `TechnicalCovenant` with data models, API endpoints, governance rules, and (from Sprint 7) a `StewardshipManifesto` with `RegenerativeLoop` containing `RepairMorphisms`.

```python
from genesis_engine.core import ArchitecturalForge

forge = ArchitecturalForge(anchor=anchor, translator=translator)
artifact = forge.forge(selected_path)

print(artifact.covenant.title)
print(artifact.integrity_verified)
print(artifact.manifesto.alignment_scores)

# Regenerative repair morphisms
for rm in artifact.manifesto.regenerative_loop.repair_morphisms:
    print(f"{rm.source_model} -> {rm.target_model}: {rm.trigger_condition}")
```

### Module 1.4 -- AxiomLogix Translator
**File:** `genesis_engine/core/axiomlogix.py`
**Sprints:** 1, 6.1

Converts natural-language problem statements into `CategoricalGraph` structures (objects + morphisms). From Sprint 6.1, also infers **shadow entities** (Future_Generations, Ecosystem) as implicit stakeholder nodes.

```python
from genesis_engine.core import AxiomLogixTranslator

translator = AxiomLogixTranslator()
graph = translator.translate("A corporation that prioritizes profit over safety.")

for obj in graph.objects:
    print(f"{obj.label}: {obj.tags}")
for m in graph.morphisms:
    print(f"{m.label}: {m.source} -> {m.target}")
```

### Module 1.7 -- Mirror of Truth
**File:** `genesis_engine/core/mirror_of_truth.py`
**Sprint:** 8

Adversarial self-critique system. Probes a solution for surface alignment (looks good but hides harm), deep disharmony, vulnerable node protection failures, incentive instability, cost-shifting, and shadow entity neglect. Can trigger **reinvention** if the critique is severe enough.

```python
from genesis_engine.core import MirrorOfTruth

mirror = MirrorOfTruth(anchor=anchor, vulnerability_priority="Residential_Ratepayer")
selected_path, trace = mirror.critique_and_refine(possibility, report, graph)

print(trace.mirror_score)                    # 0-10
print(trace.surface_alignment_detected)      # True/False
print(trace.deep_disharmony_categories)      # List of categories
print(trace.reinvention_triggered)           # True/False
print(trace.mandatory_repair)                # Repair instructions or None
```

### Module 2.1 -- Axiom Anchor
**File:** `genesis_engine/core/axiom_anchor.py`
**Sprints:** 1, 5, 6.1

The Prime Directive validator. Registers scoring predicates (unity, compassion, coherence, incentive stability, sustainability) and validates artifacts against them. Can be **sealed** to prevent mutation of predicates.

```python
from genesis_engine.core import AxiomAnchor, SustainabilityPredicate

anchor = AxiomAnchor()
anchor.register_predicate("sustainability", SustainabilityPredicate(seed=42))
anchor.seal()  # Immutable from this point

result = anchor.validate(graph.as_artefact())
print(result.is_aligned)
print(result.coherence_score)
print(result.principle_scores)
```

**Key predicates:**
- `IncentiveStabilityPredicate` -- Detects shareholder primacy gravity wells
- `SustainabilityPredicate` -- Monte Carlo temporal viability + ecological harmony

### Module 2.3 -- Continuity Bridge
**File:** `genesis_engine/core/continuity_bridge.py`
**Sprints:** 3, 5, 6.1

Manages the persistent `.genesis_soul` file with hash-chained wisdom entries, human override logging, foresight projections, and integrity verification.

```python
from genesis_engine.core import ContinuityBridge

bridge = ContinuityBridge()
soul = bridge.create_soul(anchor)

# Record wisdom
soul.record_wisdom(report, resolution_path="reinvention",
                   resolution_summary="Replaced extractive model",
                   covenant_title="Stewardship Covenant")

# Record human override
soul.record_human_override(
    system_recommended_id="cand-001",
    system_recommended_score=0.85,
    human_selected_id="cand-002",
    human_selected_score=0.68,
    divergence_reason="Reform preserves extractive structures",
    reason_category="real_world_evidence",
    confidence=8,
    problem_text="...",
    system_recommended_path="reform",
    human_selected_path="dissolution",
)

# Export and verify
envelope = bridge.export_soul(soul)
verified = bridge.verify_integrity(envelope)
chain_valid, errors = bridge.verify_wisdom_chain(soul)
```

**Override reason categories:** `axiomatic_blind_spot`, `cultural_context`, `ethical_nuance`, `real_world_evidence`, `temporal_dynamics`, `other`

### Module 3.1 -- Crucible Engine
**File:** `genesis_engine/core/crucible.py`
**Sprint:** 4

The 6-phase Socratic reasoning engine that orchestrates the entire analysis pipeline. See [The 6-Phase Crucible Workflow](#the-6-phase-crucible-workflow) below.

### Module 3.2 -- Aria Interface
**File:** `genesis_engine/core/aria_interface.py`
**Sprints:** 4, 8, 9, 10

The primary CLI interface and orchestration layer. Provides high-level methods for running the full pipeline:

```python
from genesis_engine.core import AriaInterface

aria = AriaInterface(use_colors=True)

# Process a problem through the full Crucible pipeline
result = aria.process("A policy that prioritizes profit over safety.", verbose=True)

# Load and analyse a conflict scenario
scenario, graph = aria.load_conflict("scenarios/grid_war_2026.json", verbose=True)

# Inject nodes into a conflict graph
graph = aria.inject_node(
    graph=graph,
    label="Local_Innovation_Hub",
    tags=["stakeholder", "actor", "community"],
    connect_to="Residential_Ratepayer",
    morphism_label="Community_Empowerment",
    morphism_tags=["empowerment", "collaboration"],
    verbose=True,
)

# Run a 100-round economic war game
outcome = aria.war_game(rounds=100, seed=42, verbose=True)

# Human override
override = aria.human_override(
    result=result,
    selected_candidate=candidate,
    divergence_reason="...",
    reason_category="real_world_evidence",
    confidence=8,
    verbose=True,
)

# Inspect the soul state
aria.inspect_soul(verbose=True)

# Verify hash chain integrity
is_valid, errors = aria.verify_chain()

# Compare two manifestos/policies
delta = aria.compare_manifestos(policy_a, policy_b, verbose=True)

# Run invariant tracking
violations = aria.invariant_tracker(verbose=True)

# Run Bayesian robustness exam
robustness_result = aria.robustness_exam(seed=42, verbose=True)

# Display the Mirror of Truth refinement panel
aria.refinement_panel(trace, verbose=True)

# Generate governance report (Production Lexicon)
report = aria.generate_governance_report(trace=trace, robustness_result=result, verbose=True)

# Generate the State of the Sovereignty index
sovereign_md = aria.generate_sovereign_index(report, verbose=True)
```

### Module 3.5 -- Game Theory Console
**File:** `genesis_engine/core/game_theory_console.py`
**Sprints:** 6.1, 7, 8

Runs iterated Prisoner's Dilemma simulations between "Aligned" (stewardship) and "Extractive" agents to test long-term sustainability.

**Three exam types:**

| Exam | Threshold | Purpose |
|------|-----------|---------|
| `FinalExam` | 7.0 | Standard 100-round economic war |
| `BayesianFinalExam` | 7.0 | Adds Bayesian uncertainty and blackout shocks |
| `CovenantFinalExam` | 7.5 | Governance-aware exam with covenant strength bonus |

```python
from genesis_engine.core import FinalExam, BayesianFinalExam, CovenantFinalExam

# Standard exam
exam = FinalExam(pass_threshold=7.0, rounds=100)
result = exam.administer(seed=42)
print(f"Score: {result.sustainability_score:.4f}, Passed: {result.passed}")

# Bayesian exam with blackout shock
bayesian = BayesianFinalExam(pass_threshold=7.0, fragility_amplifier=1.5)
shock = bayesian.administer(seed=42)

# Covenant-aware exam
gov_strength = CovenantFinalExam.compute_governance_strength(
    alignment_scores=artifact.manifesto.alignment_scores,
    governance_rule_count=len(artifact.covenant.governance_rules),
    repair_morphism_count=len(artifact.manifesto.regenerative_loop.repair_morphisms),
)
covenant_exam = CovenantFinalExam(pass_threshold=7.5)
result = covenant_exam.administer(governance_strength=gov_strength, seed=42)
```

### Module 3.6 -- Wisdom Mirror
**File:** `genesis_engine/core/wisdom_mirror.py`
**Sprint:** 7

Scans the human override log for recurring divergence patterns and proposes **Covenant Patches** to heal the engine's blind spots.

```python
from genesis_engine.core import WisdomMirror

mirror = WisdomMirror(patch_threshold=3)
report = mirror.scan(soul)

print(f"Patterns found: {len(report.patterns)}")
print(f"Patches proposed: {len(report.patches)}")

for patch in report.patches:
    print(f"[{patch.patch_id}] {patch.title} (priority: {patch.priority:.1f})")
```

### Module 3.5 ext -- Robustness Harness
**File:** `genesis_engine/core/robustness_harness.py`
**Sprints:** 10, 11

Bayesian robustness testing with Monte Carlo simulations, hard invariant tracking, decentralized fork detection, and (Sprint 11) **categorical repair operators** (RepairFunctor, ColimitRepairOperator).

```python
from genesis_engine.core import RobustnessHarness, HardInvariant

harness = RobustnessHarness()
result = harness.run(seed=42)

# Categorical repair
from genesis_engine.core import CategoricalRepairEngine, RepairAction

engine = CategoricalRepairEngine()
# Applies repair functors and colimit operators to damaged graphs
```

### Module 3.6 ext -- Governance Report
**File:** `genesis_engine/core/governance_report.py`
**Sprint:** 10

Generates production-ready governance reports with a **Production Lexicon** that translates internal sacred terminology to neutral policy language. Outputs `State_of_the_Sovereignty.md` for the Obsidian vault.

```python
from genesis_engine.core import GovernanceReportBuilder, SovereignIndexGenerator

builder = GovernanceReportBuilder()
report = builder.build(trace=trace, robustness_result=result)

# Generate sovereign index markdown
gen = SovereignIndexGenerator()
md = gen.generate(report)
```

### Module 2.1 ext -- Policy Kernel (Sprint 11)
**File:** `genesis_engine/core/policy_kernel.py`
**Sprint:** 11

Constitutional principle checking with reason chains, bias detection, and self-critique. Evaluates policies against 8 constitutional principles.

```python
from genesis_engine.core import PolicyKernel, CONSTITUTIONAL_PRINCIPLES

kernel = PolicyKernel()

# Evaluate an artifact against constitutional principles
result = kernel.evaluate(artifact)
print(result.reason_chain)           # Full reasoning trace
print(result.bias_detections)        # Any detected biases
print(result.self_critique)          # Self-critique assessment
```

### Module 3.5 ext -- Adversarial Evaluator / FAIRGAME (Sprint 11)
**File:** `genesis_engine/core/adversarial_evaluator.py`
**Sprint:** 11

FAIRGAME (Formal Adversarial Inquiry for Responsible Governance and Meaningful Evaluation) debate arena. Pits a `ProSocialAgent` against a `HostileLobbyist` in structured debate rounds.

```python
from genesis_engine.core import AdversarialEvaluator, FAIRGAMEAnalyzer

evaluator = AdversarialEvaluator()
debate_result = evaluator.debate(artifact, rounds=5)

# FAIRGAME bias analysis
analyzer = FAIRGAMEAnalyzer()
bias_trace = analyzer.analyze(artifact)
```

### Obsidian Exporter
**File:** `genesis_engine/core/obsidian_exporter.py`
**Sprints:** 7, 9

Exports the `.genesis_soul` state to a linked Markdown vault compatible with Obsidian. Includes Mermaid diagrams, bidirectional links, and Stewardship Manifesto frontmatter.

```python
from genesis_engine.core import ObsidianExporter

exporter = ObsidianExporter(include_mermaid=True)

# Standard export
vault = exporter.export(soul, manifesto_dict=manifesto.as_dict())
vault.write_to_disk(Path("./my_vault"))

# Crystallization (Sprint 9) -- full pipeline snapshot
crystal = exporter.crystallize(
    soul=soul,
    sustainability_score=8.5,
    fragility_index=3.2,
    passed_exam=True,
    trace=trace,
    manifesto_dict=manifesto_dict,
    repair_morphisms=repair_morphisms,
)
crystal.vault.write_to_disk(Path("./crystal_vault"))
```

### AI Provider
**File:** `genesis_engine/core/ai_provider.py`
**Sprints:** 4, 5, 9

Abstraction layer for AI backends. Supports Ollama (local LLM) with automatic fallback to `LocalProvider` (rule-based, no external dependencies).

```python
from genesis_engine.core import get_default_provider, OffloadSkeleton

provider = get_default_provider()  # Ollama -> LocalProvider fallback
print(provider.provider_name)

# Anonymized offload skeleton (for distributed processing)
packet = OffloadSkeleton.prepare(graph.as_dict(), simulation_rounds=100)
print(packet.packet_hash)
print(packet.extraction_ratio)
```

---

## The 6-Phase Crucible Workflow

The Crucible Engine (Module 3.1) processes every problem through six phases:

| Phase | Name | Action |
|-------|------|--------|
| 1 | **Ingest** | Translate natural language to `CategoricalGraph` via AxiomLogix |
| 2 | **Retrieval** | Search EternalBox for prior wisdom on similar problems |
| 3 | **Divergence** | Generate multi-perspective candidates via Dream Engine |
| 4 | **Verification** | Validate each candidate through the Axiom Anchor |
| 5 | **Convergence** | Rank and select the best candidate |
| 6 | **Crystallization** | Commit the winner to EternalBox as a Technical Covenant |

### Running the Crucible

```python
from genesis_engine.core import AriaInterface

aria = AriaInterface(use_colors=True)
result = aria.process("Your policy problem statement here.", verbose=True)

# Access results
print(result.disharmony_report.unity_impact)
print(result.possibility_report.recommended_path)
print(result.crystallized_candidate.artifact.covenant.title)
print(result.logic_box.best.unity_alignment_score)
```

---

## Running the Demos

Each demo showcases features from a specific sprint. Run them from the project root:

| Command | Sprint | What It Demonstrates |
|---------|--------|---------------------|
| `python main.py` | 4 | Core Crucible pipeline + Aria CLI visualization |
| `python demo_human_override.py` | 5 | Human override protocol, Ollama integration, sealed Axiom Anchor |
| `python demo_shareholder_primacy.py` | 5 | Incentive stability predicate, legal gravity well detection |
| `python demo_time_guardian.py` | 6.1 | Sustainability axiom, shadow entities, 100-round war game |
| `python demo_wisdom_mirror.py` | 7 | Wisdom Mirror, regenerative blueprinting, Obsidian export, Final Exam |
| `python demo_grid_war.py` | 8 | Full Grid War pipeline with Mirror of Truth + Bayesian shock |
| `python demo_sprint9.py` | 9 | Sovereign synthesis, crystallization event, offload skeleton |
| `python demo_sprint10.py` | 10 | Sovereign governance, invariant tracking, governance report |

### Demo Walkthrough: The Oklahoma Grid War

The most comprehensive demo is the Grid War pipeline. Run it step by step:

```bash
# Start with the Sprint 8 Grid War (full pipeline)
python demo_grid_war.py

# Then see Sovereign Synthesis (Sprint 9 adds crystallization)
python demo_sprint9.py

# Then see Sovereign Governance (Sprint 10 adds invariant tracking + governance reports)
python demo_sprint10.py
```

The Grid War scenario (`scenarios/grid_war_2026.json`) models the 2026 Oklahoma electricity grid crisis where hyperscale data centre demand, utility shareholder primacy, and cooling-water consumption collide with residential ratepayer protection.

---

## Running the Tests

```bash
# Run all 596 tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific sprint's tests
python -m pytest tests/test_sprint9.py
python -m pytest tests/test_sprint10.py
python -m pytest tests/test_sprint11.py

# Run tests for a specific module
python -m pytest tests/test_axiom_anchor.py
python -m pytest tests/test_crucible.py
python -m pytest tests/test_game_theory_console.py
python -m pytest tests/test_mirror_of_truth.py

# Run tests matching a keyword
python -m pytest tests/ -k "sustainability"
python -m pytest tests/ -k "bayesian"
python -m pytest tests/ -k "fairgame"
```

### Test File Map

| Test File | Module(s) Covered |
|-----------|------------------|
| `test_axiom_anchor.py` | Axiom Anchor, Prime Directive |
| `test_axiomlogix.py` | AxiomLogix Translator, CategoricalGraph |
| `test_deconstruction_engine.py` | Deconstruction Engine, DisharmonyReport |
| `test_dream_engine.py` | Dream Engine, Threefold Path |
| `test_architectural_forge.py` | Architectural Forge, TechnicalCovenant |
| `test_continuity_bridge.py` | Continuity Bridge, .genesis_soul |
| `test_hash_chain.py` | Hash chain integrity verification |
| `test_crucible.py` | Crucible Engine, 6-phase workflow |
| `test_aria_interface.py` | Aria Interface, CLI rendering |
| `test_game_theory_console.py` | Game Theory Console, war games |
| `test_bayesian_final_exam.py` | Bayesian Final Exam, blackout shock |
| `test_final_exam.py` | Standard + Covenant Final Exams |
| `test_mirror_of_truth.py` | Mirror of Truth, adversarial critique |
| `test_wisdom_mirror.py` | Wisdom Mirror, Covenant Patches |
| `test_obsidian_exporter.py` | Obsidian vault export |
| `test_regenerative_forge.py` | Regenerative Blueprinting, manifesto |
| `test_foresight_projections.py` | Foresight Projections in wisdom log |
| `test_shadow_entities.py` | Shadow entity inference |
| `test_sustainability_predicate.py` | Sustainability Predicate, Monte Carlo |
| `test_sprint9.py` | Sprint 9 integration (crystallization) |
| `test_sprint10.py` | Sprint 10 integration (governance) |
| `test_sprint11.py` | Sprint 11 integration (policy auditor, FAIRGAME) |
| `test_adversarial_evaluator.py` | Adversarial Evaluator, debate arena |

---

## Sprint History

### Sprint 1 -- The Socratic Core
**Modules:** Axiom Anchor (2.1), AxiomLogix (1.4), Deconstruction Engine (1.1)

Established the foundation: the Prime Directive validator, natural-language-to-categorical-graph translator, and disharmony analysis engine.

### Sprint 2 -- The Dream Engine
**Module:** Dream Engine (1.2)

Added the Threefold Path: Reform, Reinvention, and Dissolution. The engine now generates three distinct solution paths for every problem, each scored for unity alignment and feasibility.

### Sprint 3 -- Architectural Forge & Continuity Bridge
**Modules:** Architectural Forge (1.3), Continuity Bridge (2.3)

Added covenant synthesis (turning dream paths into Technical Covenants with data models, endpoints, and governance rules) and persistent memory via `.genesis_soul` files with hash-chaining.

### Sprint 4 -- The Socratic Crucible & Aria Interface
**Modules:** Crucible Engine (3.1), Aria Interface (3.2), AI Provider

Introduced the 6-phase Crucible workflow that orchestrates the entire pipeline, the Aria CLI visualization, and the AI provider abstraction layer.

### Sprint 5 -- Honesty Hardening & Ollama Integration
**Modules:** Human Override Protocol, Ollama Provider, Axiom Anchor Sealing

Added the human override system (allowing users to reject machine recommendations with audited reasons), Ollama integration for local LLM processing, and Axiom Anchor sealing to prevent predicate mutation.

### Sprint 6.1 -- The Time Guardian
**Modules:** Sustainability Predicate (2.1 ext), Shadow Entity Inference (1.4 ext), Game Theory Console (3.5)

Added Monte Carlo sustainability analysis, ecological harmony detection, shadow entity inference (Future_Generations, Ecosystem), and 100-round economic war games with SYSTEMIC_COLLAPSE detection.

### Sprint 7 -- Sovereign Actuation & The Wisdom Mirror
**Modules:** Wisdom Mirror (3.6), Regenerative Blueprinting (1.3 ext), Obsidian Exporter (2.3 ext), Final Exam (3.5 ext)

Added the feedback loop between human overrides and machine learning (Wisdom Mirror + Covenant Patches), Stewardship Manifesto with RegenerativeLoop and RepairMorphisms, Obsidian vault export with Mermaid diagrams, and the Final Exam gate (sustainability < 7.0 blocks production).

### Sprint 8 -- The Mirror of Truth & The Grid War Case Study
**Modules:** Mirror of Truth (1.7), Bayesian Final Exam (3.5 ext), Grid War Scenario

Added adversarial self-critique (surface alignment detection, deep disharmony probing, mandatory repair), Bayesian uncertainty hardening with blackout shock simulation, and the Oklahoma Grid War 2026 scenario.

### Sprint 9 -- Sovereign Synthesis & Vertical Actuation
**Features:** Crystallization Event, Offload Skeleton, Conflict Loading, Node Injection

Added the crystallization pipeline (complete policy analysis snapshot to Obsidian vault), anonymized offload skeleton for distributed processing, conflict scenario loading from JSON, and dynamic node injection into conflict graphs.

### Sprint 10 -- Sovereign Governance & The Oklahoma Water/Grid War
**Modules:** Robustness Harness (3.5 ext), Governance Report (3.6 ext), Production Lexicon

Added Bayesian robustness harness with hard invariant tracking, decentralized fork operator for hostile agent detection, governance report generation with Production Lexicon translation, and the State_of_the_Sovereignty.md index.

### Sprint 11 -- Policy Auditor & Regenerative Blueprint Suite
**Modules:** Policy Kernel (2.1 ext), Adversarial Evaluator (3.5 ext), Categorical Repair Operators (1.3 ext)

Added constitutional principle checking with reason chains and bias detection, FAIRGAME debate arena (ProSocialAgent vs HostileLobbyist), self-critique capabilities, and categorical repair operators (RepairFunctor, ColimitRepairOperator) for automated graph healing.

---

## Key Concepts & Glossary

| Term | Definition |
|------|-----------|
| **Axiom Anchor** | The sealed, immutable set of predicates that enforce the Prime Directive |
| **CategoricalGraph** | A graph of Objects (stakeholders/entities) connected by Morphisms (relationships) |
| **Covenant** | A Technical Covenant: the output of the Architectural Forge (data models, endpoints, rules) |
| **Crystallization** | The final act of committing a validated solution to persistent memory + vault |
| **Disharmony** | A detected misalignment between a policy and the Prime Directive |
| **Dream Path** | One of three solution types: Reform, Reinvention, or Dissolution |
| **EternalBox** | Persistent memory (`.genesis_soul` file) with hash-chained integrity |
| **FAIRGAME** | Formal Adversarial Inquiry for Responsible Governance and Meaningful Evaluation |
| **Final Exam** | A 100-round iterated Prisoner's Dilemma that tests long-term sustainability |
| **Foresight Projection** | War-game outcome stored in the wisdom log for future reference |
| **Genesis Soul** | The `.genesis_soul` file: hash-chained, tamper-evident persistent memory |
| **HILL** | High-Impact Large Load (100+ MW data centre connections) |
| **Legal Gravity Well** | A structural incentive trap (e.g., shareholder primacy) that distorts outcomes |
| **LogicBox** | Ephemeral evaluation sandbox (cleared each run) |
| **Manifesto** | Stewardship Manifesto: the index for a regenerative blueprint |
| **Mirror of Truth** | Adversarial self-critique module that probes for hidden extraction |
| **Morphism** | A directed relationship between two objects in a categorical graph |
| **Object** | A stakeholder or entity node in a categorical graph |
| **Obsidian Vault** | Markdown knowledge graph exported for use with Obsidian.md |
| **Override** | A human decision to reject the machine's recommendation (with audited reason) |
| **Production Lexicon** | Translation table from internal sacred terms to neutral policy language |
| **Regenerative Loop** | Self-healing mechanism with repair morphisms that activate on score decay |
| **Repair Morphism** | A conditional repair action triggered when a principle score drops below threshold |
| **Shadow Entity** | An implied stakeholder (Future_Generations, Ecosystem) inferred by AxiomLogix |
| **Sustainability Score** | 0--10 score from game-theoretic simulation; < 7.0 blocks production |
| **Wisdom Mirror** | Module that learns from human overrides and proposes Covenant Patches |

---

## Scenario Files

### `scenarios/grid_war_2026.json` (v4.0.0)

The Oklahoma Grid War scenario. Contains:

- **Legislative references:** HB 2992 (HILL Act), SB 1488 (Water Usage Act), PSO Rate Case
- **Conflict graph:** 10+ stakeholders (Residential Ratepayer, Hyperscale Data Center, Utility Provider, Water Authority, etc.) with relationships
- **PSO Rate Case data:** $380M revenue increase request, cost allocation percentages, invariant violations
- **Water basin data:** Tulsa and Moore basin sustainable withdrawal limits

Use with:
```python
scenario, graph = aria.load_conflict("scenarios/grid_war_2026.json", verbose=True)
```

---

## Output Artifacts

All outputs are written to `genesis_engine/reports/`:

| File Pattern | Format | Contents |
|-------------|--------|----------|
| `crucible_result_*.json` | JSON | Full Crucible pipeline output |
| `disharmony_report_*.json` | JSON | Deconstruction Engine findings |
| `possibility_report_*.json` | JSON | Dream Engine paths |
| `forge_artifact_*.json` | JSON | Architectural Forge covenant |
| `governance_report_*.json` | JSON | Production Lexicon governance report |
| `*.genesis_soul` | JSON | Hash-chained persistent memory |
| `obsidian_vault/` | Markdown | Linked knowledge graph for Obsidian |
| `obsidian_vault/Manifesto.md` | Markdown | Stewardship Manifesto with frontmatter |
| `obsidian_vault/Insights/` | Markdown | Wisdom entries |
| `obsidian_vault/Overrides/` | Markdown | Human override decisions |
| `obsidian_vault/Projections/` | Markdown | Foresight analysis |
| `obsidian_vault/Graphs/` | Markdown | Categorical graph representations |
| `obsidian_vault/State_of_the_Sovereignty.md` | Markdown | Governance status index |

---

## API Quick Reference

### AriaInterface -- Primary Entry Point

```python
from genesis_engine.core import AriaInterface

aria = AriaInterface(use_colors=True)
```

| Method | Purpose |
|--------|---------|
| `aria.process(problem, verbose)` | Run full Crucible pipeline on a problem statement |
| `aria.load_conflict(path, verbose)` | Load a JSON conflict scenario |
| `aria.inject_node(graph, ...)` | Add a stakeholder node to a conflict graph |
| `aria.war_game(rounds, seed, verbose)` | Run an economic war game simulation |
| `aria.human_override(result, ...)` | Record a human override decision |
| `aria.inspect_soul(verbose)` | Display current soul state |
| `aria.verify_chain()` | Verify hash chain integrity |
| `aria.compare_manifestos(a, b, verbose)` | Compare two policies/manifestos |
| `aria.invariant_tracker(verbose)` | Check hard invariant violations |
| `aria.robustness_exam(seed, verbose)` | Run Bayesian robustness harness |
| `aria.refinement_panel(trace, verbose)` | Display Mirror of Truth analysis |
| `aria.generate_governance_report(...)` | Generate Production Lexicon report |
| `aria.generate_sovereign_index(report, verbose)` | Generate sovereignty markdown index |

### Direct Module Usage

```python
from genesis_engine.core import (
    AxiomAnchor,
    AxiomLogixTranslator,
    DeconstructionEngine,
    DreamEngine,
    ArchitecturalForge,
    MirrorOfTruth,
    ContinuityBridge,
    GameTheoryConsole,
    FinalExam,
    WisdomMirror,
    ObsidianExporter,
    PolicyKernel,
    AdversarialEvaluator,
    RobustnessHarness,
    GovernanceReportBuilder,
)
```
