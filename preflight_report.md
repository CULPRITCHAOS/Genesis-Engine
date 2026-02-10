# Preflight Report — Sprint 11.5 Sovereign Actuation

**Date:** 2026-02-10
**Python:** 3.11.14
**Branch:** `claude/preflight-model-calibration-7Zqme`
**Test Suite:** 596/596 PASSED (1.24s)

---

## 1. Code-to-Pipeline Integrity Audit

### 1.1 InputValidator → PolicyKernel Flow

**Status: PASS**

The `AriaInterface.policy_audit()` method (`aria_interface.py:1863-1943`) executes a verified 5-step pipeline:

1. `AdversarialEvaluator.debate()` — FAIRGAME Debate Arena (Pro_Social vs Hostile_Lobbyist)
2. `AriaInterface.generate_governance_report()` — Production Lexicon report
3. `PolicyKernel.evaluate()` — C3AI Self-Critique Loop with constitutional compliance gate
4. `AriaInterface.sovereign_audit_hook()` — Data residency verification
5. Rendered output via `AriaRenderer.render_policy_audit_panel()`

**No hallucinated bypasses detected.** All governance data flows through the PolicyKernel evaluation before export. The constitutional compliance gate (`gate_threshold: 6.0`) blocks non-compliant reports from final export.

### 1.2 PlanCompiler (Forge) — Categorical Repair Operators

**Status: PASS**

- `ArchitecturalForge.forge()` (`architectural_forge.py:440-486`) chains through all 6 steps:
  Data Models → Endpoints → Governance Rules → Technical Covenant → AxiomLogix Verification → Stewardship Manifesto
- `RepairMorphism`, `RegenerativeLoop`, and all Forge dataclasses instantiate cleanly under Python 3.11.14
- No library clashes detected — all imports are standard library (`hashlib`, `json`, `uuid`, `dataclasses`, `datetime`)
- `_generate_repair_morphisms()` correctly maps extractive patterns to healing replacements via `_HEAL_MAP`

### 1.3 Linting — PEP8/Pyflakes Scan

**Status: PASS (28 issues fixed)**

Dead code removed across 15 files:

| File | Issues Fixed |
|---|---|
| `ai_provider.py` | Unused `Protocol` import |
| `robustness_harness.py` | Unused `math`, `Object`, `Morphism` imports; unused `required_ratio` variable |
| `crucible.py` | Unused `Candidate` import; unused `prior_wisdom` variable |
| `wisdom_mirror.py` | Unused `OVERRIDE_REASON_CATEGORIES` import |
| `mirror_of_truth.py` | Unused `Path`, `SustainabilityPredicate`, `ValidationResult` imports |
| `continuity_bridge.py` | Unused `ValidationResult`, `DreamPath`, `PossibilityReport` imports |
| `dream_engine.py` | Unused `MorphismFinding` import |
| `governance_report.py` | Unused `hashlib`, `stewardship_frontmatter` imports |
| `aria_interface.py` | Unused `json`, `uuid`, `dataclass`, `Object`, `Morphism`, `OVERRIDE_REASON_CATEGORIES`, `CritiqueFinding`, `translate_to_production` imports |
| `ollama_provider.py` | Unused `field` import |
| `obsidian_exporter.py` | Unused `json` import |
| `game_theory_console.py` | Unused `score_gap` variable |
| `axiom_anchor.py` | Unused `m_label` variable |
| `policy_kernel.py` | Unused `conflicts`, `robustness`, `v_type`, `violation_categories` variables |

**Post-fix pyflakes:** Zero import/variable errors remaining in core pipeline modules.

---

## 2. Model Calibration (Ollama Optimization)

### 2.1 Socratic Role Mapping

New `SocraticRole` enum and `SOCRATIC_MODEL_MAP` added to `ai_provider.py`:

| Socratic Role | Pipeline Target | Primary Model | Fallback Model |
|---|---|---|---|
| **Thinker** (Policy Analysis) | AdversarialEvaluator, PolicyKernel, MirrorOfTruth | `deepseek-r1:7b` | `huihui_ai/am-thinking-abliterated` |
| **Builder** (PlanCompiler) | ArchitecturalForge, DreamEngine, GovernanceReport | `qwen2.5-coder` | `gemma3:12b` |
| **Sentry** (InputValidator) | InputValidator, CrucibleEngine, AxiomLogixTranslator | `gemma3:4b` | `dolphin-llama3` |

### 2.2 Provider Resolution

`OllamaProvider` now accepts an optional `role: SocraticRole` parameter:
- When set, the model is resolved from `SOCRATIC_MODEL_MAP[role]` (first available)
- Timeout is automatically raised to `OLLAMA_TIMEOUT` (600s) for role-based providers
- `get_default_provider(role=SocraticRole.THINKER)` returns a calibrated provider

### 2.3 Configuration File

`config.yaml` created at project root with:
- Ollama server settings (base_url, timeout, num_ctx)
- Full Socratic Role → Model mapping with pipeline targets and descriptions
- Hardware-aware performance parameters

---

## 3. Hardware-Aware Performance

### 3.1 Timeout Buffers

| Parameter | Previous | Updated | Rationale |
|---|---|---|---|
| `OllamaConfig.timeout` | 60s | **600s** | 100-round sims on Gemma 3 12B / DeepSeek R1 |
| `OLLAMA_TIMEOUT` (core) | N/A | **600s** | Global timeout for role-based providers |
| `OllamaProvider._timeout` (core) | 10s | **600s** (when role set) | Auto-scaled for calibrated models |

### 3.2 Context Management

| Parameter | Value | Rationale |
|---|---|---|
| `OLLAMA_NUM_CTX` | **128,000** | Hard-coded for 4B/12B models handling large legislative PDFs |
| `OllamaConfig.num_ctx` | **128,000** | Passed to Ollama `options.num_ctx` in API requests |
| Core `_query_ollama()` | Now passes `num_ctx` | Ensures context window is set per-request |

### 3.3 Ollama API Payload

Both providers (`core/ai_provider.py` and `ai/providers/ollama_provider.py`) now send `num_ctx` in the Ollama generate request options, ensuring the context window is enforced at the API level.

---

## 4. Test Results

```
596 passed in 1.24s
```

**All 596 tests pass locally** on Python 3.11.14. No regressions introduced by:
- Lint cleanup (28 unused import/variable removals)
- SocraticRole enum addition
- OllamaProvider role-based model resolution
- Timeout and num_ctx configuration changes

---

## 5. Deliverables

| Deliverable | Status | Path |
|---|---|---|
| Preflight Report | **DELIVERED** | `preflight_report.md` |
| Config YAML | **DELIVERED** | `config.yaml` |
| Lint Cleanup | **COMPLETE** | 28 issues fixed across 15 files |
| Model Calibration | **COMPLETE** | `SocraticRole` enum + `SOCRATIC_MODEL_MAP` |
| Hardware Tuning | **COMPLETE** | 600s timeout, 128K num_ctx |
| Test Suite | **596/596 PASS** | `python -m pytest tests/` |
