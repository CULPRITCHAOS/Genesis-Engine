"""
Module 3.6 — The Wisdom Mirror

Closes the feedback loop between human wisdom and machine logic by scanning
the Human Override Log for recurring patterns of divergence and suggesting
"Covenant Patches" to the Axiom Anchor.

The Wisdom Mirror implements:

1. **Divergence Pattern Analysis** — Groups human overrides by
   ``reason_category`` and identifies categories with 3+ occurrences.

2. **Covenant Patch Proposals** — For each recurring divergence category,
   generates a structured patch suggestion describing how the Axiom Anchor's
   predicates might be amended to incorporate the human's repeated insight.

3. **Compassion-Driven Prioritisation** — Patches are ranked by frequency
   and average confidence, ensuring the most strongly-felt human corrections
   surface first. Unity over Power: the mirror reflects what the machine
   cannot yet see, not what it wants to enforce.

INVARIANT: The Wisdom Mirror NEVER directly modifies the Axiom Anchor.
It produces *proposals* that require human review and explicit acceptance.
The Anchor remains the immutable Ground Truth until a human deliberately
applies a patch — preserving the sovereignty of both machine logic and
human wisdom.

Integration:
- Reads from ``GenesisSoul.human_overrides``
- Produces ``CovenantPatch`` proposals
- Patches are stored in the soul's ``forge_artifacts`` when accepted
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.continuity_bridge import (
    GenesisSoul,
    HumanOverrideEntry,
)


# ---------------------------------------------------------------------------
# Divergence Pattern
# ---------------------------------------------------------------------------

@dataclass
class DivergencePattern:
    """A recurring pattern of human overrides in a single category.

    Aggregates all overrides sharing the same ``reason_category`` and
    computes summary statistics.
    """

    category: str
    occurrences: int
    average_confidence: float
    average_score_delta: float  # mean(system_score - human_score)
    override_entries: list[HumanOverrideEntry] = field(default_factory=list)
    common_keywords: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """A pattern is actionable when it has 3+ occurrences."""
        return self.occurrences >= 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "occurrences": self.occurrences,
            "averageConfidence": round(self.average_confidence, 2),
            "averageScoreDelta": round(self.average_score_delta, 4),
            "isActionable": self.is_actionable,
            "commonKeywords": self.common_keywords,
            "overrideTimestamps": [e.timestamp for e in self.override_entries],
        }


# ---------------------------------------------------------------------------
# Covenant Patch
# ---------------------------------------------------------------------------

@dataclass
class CovenantPatch:
    """A proposed amendment to the Axiom Anchor based on recurring divergence.

    Covenant Patches do NOT modify the Anchor directly. They are proposals
    that require human review. The machine "dies" to its own certainty and
    "resurrects" with the human's wisdom — Unity over Power.
    """

    patch_id: str
    category: str
    title: str
    description: str
    rationale: str
    suggested_predicate_adjustment: str
    priority: float  # 0.0–10.0, higher = more urgent
    source_pattern: DivergencePattern | None = None
    status: str = "proposed"  # proposed | accepted | rejected
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "patchId": self.patch_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "suggestedPredicateAdjustment": self.suggested_predicate_adjustment,
            "priority": round(self.priority, 2),
            "status": self.status,
            "sourcePattern": self.source_pattern.as_dict() if self.source_pattern else None,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Mirror Report
# ---------------------------------------------------------------------------

@dataclass
class MirrorReport:
    """Complete output of a Wisdom Mirror scan."""

    soul_id: str
    total_overrides: int
    patterns: list[DivergencePattern] = field(default_factory=list)
    patches: list[CovenantPatch] = field(default_factory=list)
    actionable_count: int = 0
    integrity_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mirrorReport": {
                "soulId": self.soul_id,
                "totalOverrides": self.total_overrides,
                "actionablePatterns": self.actionable_count,
                "patterns": [p.as_dict() for p in self.patterns],
                "covenantPatches": [p.as_dict() for p in self.patches],
                "integrityHash": self.integrity_hash,
                "timestamp": self.timestamp,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Patch title/description templates per category
# ---------------------------------------------------------------------------

_PATCH_TEMPLATES: dict[str, dict[str, str]] = {
    "axiomatic_blind_spot": {
        "title": "Axiom Blind Spot Correction",
        "description": (
            "The human has repeatedly identified cases where the axiom "
            "predicates fail to capture a dimension of alignment that is "
            "visible to human moral intuition but invisible to the formal "
            "system."
        ),
        "adjustment": (
            "Extend the Unity predicate to include additional harmony tags "
            "derived from the override reasons. Consider adding a "
            "'moral_intuition' weight factor that softens binary "
            "alignment thresholds."
        ),
    },
    "real_world_evidence": {
        "title": "Empirical Evidence Integration",
        "description": (
            "The human possesses real-world evidence that contradicts "
            "the system's theoretical scoring. The machine's model is "
            "incomplete without empirical grounding."
        ),
        "adjustment": (
            "Add an 'empirical_evidence' bonus to the Coherence predicate "
            "that boosts scores for candidates backed by cited real-world "
            "outcomes. Consider a Bayesian update mechanism."
        ),
    },
    "cultural_context": {
        "title": "Cultural Sensitivity Expansion",
        "description": (
            "The system's axioms operate in a cultural vacuum. The human "
            "repeatedly corrects for cultural contexts that the universal "
            "predicates cannot capture."
        ),
        "adjustment": (
            "Introduce a 'cultural_context' modifier to the Compassion "
            "predicate that adjusts vulnerability assessments based on "
            "culturally-specific power dynamics."
        ),
    },
    "temporal_relevance": {
        "title": "Temporal Awareness Enhancement",
        "description": (
            "The human identifies time-sensitive factors that the system's "
            "static predicates miss. What was aligned yesterday may be "
            "harmful tomorrow."
        ),
        "adjustment": (
            "Add temporal decay/growth factors to the Sustainability "
            "predicate. Weight recent evidence more heavily and flag "
            "solutions whose alignment scores are time-dependent."
        ),
    },
    "stakeholder_knowledge": {
        "title": "Stakeholder Voice Amplification",
        "description": (
            "The human has direct knowledge of stakeholder needs that the "
            "system's graph-based analysis cannot detect. Lived experience "
            "outweighs categorical inference."
        ),
        "adjustment": (
            "Add a 'stakeholder_testimony' input channel to the Compassion "
            "predicate. Allow human-sourced vulnerability assessments to "
            "override tag-based inference when confidence is high."
        ),
    },
    "ethical_nuance": {
        "title": "Ethical Nuance Recognition",
        "description": (
            "The human recognises ethical subtleties — competing goods, "
            "tragic trade-offs, moral complexity — that the binary "
            "aligned/misaligned framework cannot express."
        ),
        "adjustment": (
            "Introduce a 'moral_complexity' dimension to validation that "
            "allows partial alignment with explicit trade-off documentation. "
            "Replace binary gates with graduated confidence bands."
        ),
    },
    "implementation_pragmatism": {
        "title": "Pragmatic Implementation Bridge",
        "description": (
            "The human selects a theoretically less-aligned path because "
            "it is practically achievable. Perfect is the enemy of good; "
            "the system must learn to value incremental progress."
        ),
        "adjustment": (
            "Add a 'feasibility_weight' to the final candidate ranking "
            "that increases the importance of the feasibility score when "
            "the system detects high-confidence pragmatic overrides."
        ),
    },
}


# ---------------------------------------------------------------------------
# Wisdom Mirror
# ---------------------------------------------------------------------------

class WisdomMirror:
    """Scans the Human Override Log and proposes Covenant Patches.

    The mirror reflects back to the machine what it cannot see —
    the recurring patterns where human wisdom consistently diverges
    from machine logic.  It does not enforce; it illuminates.

    Parameters
    ----------
    patch_threshold : int
        Minimum occurrences of a category before a patch is proposed.
        Default is 3 (as specified in the Sprint 7 requirements).
    """

    PATCH_THRESHOLD: int = 3

    def __init__(self, patch_threshold: int = 3) -> None:
        self.PATCH_THRESHOLD = patch_threshold

    # -- public API ---------------------------------------------------------

    def scan(self, soul: GenesisSoul) -> MirrorReport:
        """Analyse the soul's override log and produce a MirrorReport.

        Returns a report containing:
        - All divergence patterns (one per category with overrides)
        - Covenant Patches for categories exceeding the threshold
        """
        overrides = soul.human_overrides

        # 1. Group overrides by category
        by_category: defaultdict[str, list[HumanOverrideEntry]] = defaultdict(list)
        for entry in overrides:
            by_category[entry.reason_category].append(entry)

        # 2. Build divergence patterns
        patterns: list[DivergencePattern] = []
        for category, entries in sorted(by_category.items()):
            avg_conf = sum(e.confidence for e in entries) / len(entries)
            avg_delta = sum(
                e.system_recommended_score - e.human_selected_score
                for e in entries
            ) / len(entries)

            # Extract common keywords from divergence reasons
            keywords = self._extract_keywords(entries)

            pattern = DivergencePattern(
                category=category,
                occurrences=len(entries),
                average_confidence=avg_conf,
                average_score_delta=avg_delta,
                override_entries=entries,
                common_keywords=keywords,
            )
            patterns.append(pattern)

        # 3. Generate Covenant Patches for patterns exceeding threshold
        patches: list[CovenantPatch] = []
        for pattern in patterns:
            if pattern.occurrences >= self.PATCH_THRESHOLD:
                patch = self._generate_patch(pattern)
                patches.append(patch)

        # Sort patches by priority (highest first) — Unity over Power:
        # the most frequent and confident human corrections surface first
        patches.sort(key=lambda p: p.priority, reverse=True)

        actionable_count = sum(
            1 for p in patterns if p.occurrences >= self.PATCH_THRESHOLD
        )

        # Compute integrity hash for the report
        report = MirrorReport(
            soul_id=soul.soul_id,
            total_overrides=len(overrides),
            patterns=patterns,
            patches=patches,
            actionable_count=actionable_count,
        )
        report.integrity_hash = self._compute_hash(report)

        return report

    def reflect(self, soul: GenesisSoul) -> list[CovenantPatch]:
        """Convenience method: scan and return only the actionable patches.

        Named 'reflect' because the mirror reflects wisdom back to the system.
        """
        report = self.scan(soul)
        return report.patches

    # -- internal -----------------------------------------------------------

    def _generate_patch(self, pattern: DivergencePattern) -> CovenantPatch:
        """Generate a CovenantPatch from an actionable DivergencePattern."""
        template = _PATCH_TEMPLATES.get(pattern.category, {
            "title": f"Override Pattern: {pattern.category}",
            "description": (
                f"The human has overridden the system {pattern.occurrences} times "
                f"in the '{pattern.category}' category."
            ),
            "adjustment": (
                "Review the axiom predicates for gaps in this category."
            ),
        })

        # Priority: weighted combination of frequency and confidence
        # Normalise occurrences to 0-10 scale (cap at 10 overrides)
        freq_score = min(10.0, pattern.occurrences * 2.0)
        conf_score = pattern.average_confidence  # already 1-10
        priority = freq_score * 0.6 + conf_score * 0.4

        patch_id = hashlib.sha256(
            f"{pattern.category}|{pattern.occurrences}|{pattern.average_confidence}".encode()
        ).hexdigest()[:16]

        rationale = (
            f"Detected {pattern.occurrences} human overrides in category "
            f"'{pattern.category}' with average confidence {pattern.average_confidence:.1f}/10. "
            f"Average score delta: {pattern.average_score_delta:.4f}. "
        )
        if pattern.common_keywords:
            rationale += f"Common themes: {', '.join(pattern.common_keywords[:5])}. "
        rationale += (
            "The machine acknowledges this blind spot and proposes the "
            "following adjustment — pending human review."
        )

        return CovenantPatch(
            patch_id=f"patch-{patch_id}",
            category=pattern.category,
            title=template["title"],
            description=template["description"],
            rationale=rationale,
            suggested_predicate_adjustment=template["adjustment"],
            priority=priority,
            source_pattern=pattern,
        )

    @staticmethod
    def _extract_keywords(entries: list[HumanOverrideEntry]) -> list[str]:
        """Extract the most common meaningful words from override reasons."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "nor",
            "so", "yet", "both", "either", "neither", "each", "every",
            "all", "any", "few", "more", "most", "other", "some", "such",
            "than", "too", "very", "just", "also", "only", "own", "same",
            "that", "this", "these", "those", "it", "its", "they", "them",
            "their", "we", "our", "you", "your", "he", "she", "his", "her",
            "which", "who", "whom", "what", "when", "where", "why", "how",
            "i", "me", "my", "myself", "about", "up", "out", "if", "then",
        }
        word_counts: Counter[str] = Counter()
        for entry in entries:
            words = entry.divergence_reason.lower().split()
            for w in words:
                cleaned = w.strip(".,;:!?()\"'")
                if len(cleaned) > 3 and cleaned not in stopwords:
                    word_counts[cleaned] += 1

        return [word for word, _ in word_counts.most_common(10)]

    @staticmethod
    def _compute_hash(report: MirrorReport) -> str:
        """Compute SHA-256 integrity hash for the report."""
        content = json.dumps(
            {
                "soulId": report.soul_id,
                "totalOverrides": report.total_overrides,
                "actionableCount": report.actionable_count,
                "patterns": [p.as_dict() for p in report.patterns],
                "patches": [p.as_dict() for p in report.patches],
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
