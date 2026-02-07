"""Tests for Module 3.6 — The Wisdom Mirror (Sprint 7)."""

from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
    HumanOverrideEntry,
)
from genesis_engine.core.wisdom_mirror import (
    CovenantPatch,
    DivergencePattern,
    MirrorReport,
    WisdomMirror,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_override(
    category: str,
    confidence: int = 7,
    sys_score: float = 0.85,
    human_score: float = 0.60,
    reason_suffix: str = "",
) -> HumanOverrideEntry:
    """Create a valid HumanOverrideEntry for testing."""
    base_reason = (
        f"The system recommendation misses critical context in the "
        f"{category} domain that requires human judgement to resolve properly"
    )
    # Pad to meet 100-char minimum
    reason = base_reason + (f" — {reason_suffix}" if reason_suffix else " — override recorded")
    if len(reason) < 100:
        reason += " " * (100 - len(reason))
    return HumanOverrideEntry(
        system_recommended_id="cand-sys-001",
        system_recommended_score=sys_score,
        human_selected_id="cand-human-001",
        human_selected_score=human_score,
        divergence_reason=reason,
        reason_category=category,
        confidence=confidence,
        problem_text="Test problem text for override analysis",
        system_recommended_path="reform",
        human_selected_path="reinvention",
    )


def _make_soul_with_overrides(overrides: list[HumanOverrideEntry]) -> GenesisSoul:
    """Create a GenesisSoul with pre-populated overrides."""
    soul = ContinuityBridge.create_soul()
    for o in overrides:
        soul.human_overrides.append(o)
    return soul


# ---------------------------------------------------------------------------
# WisdomMirror core tests
# ---------------------------------------------------------------------------

class TestWisdomMirror:
    def test_scan_empty_soul(self):
        mirror = WisdomMirror()
        soul = ContinuityBridge.create_soul()
        report = mirror.scan(soul)
        assert isinstance(report, MirrorReport)
        assert report.total_overrides == 0
        assert len(report.patterns) == 0
        assert len(report.patches) == 0

    def test_scan_with_single_override(self):
        mirror = WisdomMirror()
        overrides = [_make_override("axiomatic_blind_spot")]
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert report.total_overrides == 1
        assert len(report.patterns) == 1
        assert report.patterns[0].category == "axiomatic_blind_spot"
        assert report.patterns[0].occurrences == 1
        # Single override should NOT produce a patch (threshold is 3)
        assert len(report.patches) == 0

    def test_scan_below_threshold_no_patch(self):
        mirror = WisdomMirror()
        overrides = [
            _make_override("real_world_evidence"),
            _make_override("real_world_evidence"),
        ]
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert report.total_overrides == 2
        assert len(report.patterns) == 1
        assert report.patterns[0].occurrences == 2
        assert not report.patterns[0].is_actionable
        assert len(report.patches) == 0

    def test_scan_at_threshold_produces_patch(self):
        mirror = WisdomMirror()
        overrides = [
            _make_override("axiomatic_blind_spot", confidence=8),
            _make_override("axiomatic_blind_spot", confidence=7),
            _make_override("axiomatic_blind_spot", confidence=9),
        ]
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert report.total_overrides == 3
        assert len(report.patches) == 1
        assert report.patches[0].category == "axiomatic_blind_spot"
        assert report.patches[0].title == "Axiom Blind Spot Correction"
        assert report.actionable_count == 1

    def test_scan_multiple_categories(self):
        mirror = WisdomMirror()
        overrides = [
            _make_override("axiomatic_blind_spot"),
            _make_override("axiomatic_blind_spot"),
            _make_override("axiomatic_blind_spot"),
            _make_override("cultural_context"),
            _make_override("cultural_context"),
            _make_override("cultural_context"),
            _make_override("cultural_context"),
            _make_override("real_world_evidence"),  # only 1 — below threshold
        ]
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert report.total_overrides == 8
        assert len(report.patterns) == 3
        assert report.actionable_count == 2
        assert len(report.patches) == 2


# ---------------------------------------------------------------------------
# DivergencePattern tests
# ---------------------------------------------------------------------------

class TestDivergencePattern:
    def test_is_actionable_at_3(self):
        pattern = DivergencePattern(
            category="test", occurrences=3,
            average_confidence=7.0, average_score_delta=0.25,
        )
        assert pattern.is_actionable is True

    def test_not_actionable_below_3(self):
        pattern = DivergencePattern(
            category="test", occurrences=2,
            average_confidence=7.0, average_score_delta=0.25,
        )
        assert pattern.is_actionable is False

    def test_as_dict(self):
        pattern = DivergencePattern(
            category="ethical_nuance", occurrences=5,
            average_confidence=8.5, average_score_delta=0.3,
            common_keywords=["trade-off", "complexity"],
        )
        d = pattern.as_dict()
        assert d["category"] == "ethical_nuance"
        assert d["occurrences"] == 5
        assert d["isActionable"] is True
        assert "trade-off" in d["commonKeywords"]


# ---------------------------------------------------------------------------
# CovenantPatch tests
# ---------------------------------------------------------------------------

class TestCovenantPatch:
    def test_patch_structure(self):
        mirror = WisdomMirror()
        overrides = [_make_override("ethical_nuance", confidence=9)] * 4
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert len(report.patches) == 1

        patch = report.patches[0]
        assert patch.category == "ethical_nuance"
        assert patch.title == "Ethical Nuance Recognition"
        assert patch.status == "proposed"
        assert patch.priority > 0.0
        assert "patch-" in patch.patch_id

    def test_patch_as_dict(self):
        mirror = WisdomMirror()
        overrides = [_make_override("stakeholder_knowledge")] * 3
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        d = report.patches[0].as_dict()
        assert "patchId" in d
        assert "suggestedPredicateAdjustment" in d
        assert d["status"] == "proposed"

    def test_patch_priority_increases_with_frequency(self):
        mirror = WisdomMirror()

        # 3 overrides
        overrides_3 = [_make_override("temporal_relevance", confidence=5)] * 3
        soul_3 = _make_soul_with_overrides(overrides_3)
        report_3 = mirror.scan(soul_3)

        # 6 overrides
        overrides_6 = [_make_override("temporal_relevance", confidence=5)] * 6
        soul_6 = _make_soul_with_overrides(overrides_6)
        report_6 = mirror.scan(soul_6)

        assert report_6.patches[0].priority > report_3.patches[0].priority

    def test_patch_priority_increases_with_confidence(self):
        mirror = WisdomMirror()

        # Low confidence
        overrides_low = [_make_override("ethical_nuance", confidence=3)] * 3
        soul_low = _make_soul_with_overrides(overrides_low)
        report_low = mirror.scan(soul_low)

        # High confidence
        overrides_high = [_make_override("ethical_nuance", confidence=9)] * 3
        soul_high = _make_soul_with_overrides(overrides_high)
        report_high = mirror.scan(soul_high)

        assert report_high.patches[0].priority > report_low.patches[0].priority


# ---------------------------------------------------------------------------
# MirrorReport tests
# ---------------------------------------------------------------------------

class TestMirrorReport:
    def test_report_serialisation(self):
        mirror = WisdomMirror()
        overrides = [_make_override("axiomatic_blind_spot")] * 3
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        d = report.as_dict()
        assert "mirrorReport" in d
        mr = d["mirrorReport"]
        assert mr["totalOverrides"] == 3
        assert mr["actionablePatterns"] == 1
        assert len(mr["covenantPatches"]) == 1

    def test_report_to_json(self):
        mirror = WisdomMirror()
        soul = ContinuityBridge.create_soul()
        report = mirror.scan(soul)
        import json
        data = json.loads(report.to_json())
        assert "mirrorReport" in data

    def test_report_integrity_hash(self):
        mirror = WisdomMirror()
        overrides = [_make_override("real_world_evidence")] * 4
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert len(report.integrity_hash) == 64  # SHA-256 hex

    def test_reflect_convenience_method(self):
        mirror = WisdomMirror()
        overrides = [_make_override("implementation_pragmatism")] * 5
        soul = _make_soul_with_overrides(overrides)
        patches = mirror.reflect(soul)
        assert len(patches) == 1
        assert patches[0].category == "implementation_pragmatism"


# ---------------------------------------------------------------------------
# Custom threshold tests
# ---------------------------------------------------------------------------

class TestCustomThreshold:
    def test_threshold_of_5(self):
        mirror = WisdomMirror(patch_threshold=5)
        overrides = [_make_override("cultural_context")] * 4
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        # 4 overrides with threshold=5 → no patches
        assert len(report.patches) == 0

    def test_threshold_of_1(self):
        mirror = WisdomMirror(patch_threshold=1)
        overrides = [_make_override("ethical_nuance")]
        soul = _make_soul_with_overrides(overrides)
        report = mirror.scan(soul)
        assert len(report.patches) == 1


# ---------------------------------------------------------------------------
# All 7 categories test
# ---------------------------------------------------------------------------

class TestAllCategories:
    def test_patches_for_all_categories(self):
        """Ensure every override category produces a correctly-named patch."""
        mirror = WisdomMirror()
        from genesis_engine.core.continuity_bridge import OVERRIDE_REASON_CATEGORIES

        for category in OVERRIDE_REASON_CATEGORIES:
            overrides = [_make_override(category)] * 3
            soul = _make_soul_with_overrides(overrides)
            report = mirror.scan(soul)
            assert len(report.patches) == 1, f"No patch for {category}"
            assert report.patches[0].category == category
