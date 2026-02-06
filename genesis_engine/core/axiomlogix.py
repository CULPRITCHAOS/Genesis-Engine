"""
Module 1.4 — The AxiomLogix Translator

Translates natural-language problem statements into structured
Category Theory graphs (Objects + Morphisms) suitable for downstream
validation by the Axiom Anchor and analysis by the Deconstruction Engine.

Design
------
This module is the *neuro-symbolic bridge*.  In a production system the
natural-language parsing would be backed by an LLM; for this Sprint-1 PoC
we use a deterministic keyword-driven classifier so the pipeline is
fully reproducible and testable without external services.

Categorical Mapping
-------------------
* **Objects** — the entities (nouns / actors / values) extracted from the
  problem statement.
* **Morphisms** — the directed relationships or actions between objects,
  annotated with semantic tags that the Axiom Anchor predicates inspect.

The output is a ``CategoricalGraph`` that serialises to a plain ``dict``
for downstream consumption.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Category Theory primitives
# ---------------------------------------------------------------------------

@dataclass
class Object:
    """A node in the categorical graph representing an entity or value."""

    id: str
    label: str
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "tags": self.tags}


@dataclass
class Morphism:
    """A directed edge in the categorical graph representing a relationship."""

    id: str
    label: str
    source: str  # Object.id
    target: str  # Object.id
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "target": self.target,
            "tags": self.tags,
        }


@dataclass
class CategoricalGraph:
    """A Category Theory graph composed of Objects and Morphisms."""

    objects: list[Object] = field(default_factory=list)
    morphisms: list[Morphism] = field(default_factory=list)
    source_text: str = ""

    # -- helpers ------------------------------------------------------------

    def add_object(self, label: str, tags: list[str] | None = None) -> Object:
        obj = Object(
            id=f"obj-{uuid.uuid4().hex[:8]}",
            label=label,
            tags=tags or [],
        )
        self.objects.append(obj)
        return obj

    def add_morphism(
        self,
        label: str,
        source: Object | str,
        target: Object | str,
        tags: list[str] | None = None,
    ) -> Morphism:
        src_id = source.id if isinstance(source, Object) else source
        tgt_id = target.id if isinstance(target, Object) else target
        morph = Morphism(
            id=f"mor-{uuid.uuid4().hex[:8]}",
            label=label,
            source=src_id,
            target=tgt_id,
            tags=tags or [],
        )
        self.morphisms.append(morph)
        return morph

    def as_dict(self) -> dict[str, Any]:
        return {
            "sourceText": self.source_text,
            "objects": [o.as_dict() for o in self.objects],
            "morphisms": [m.as_dict() for m in self.morphisms],
        }

    def as_artefact(self) -> dict[str, Any]:
        """Return a flat dict suitable for the Axiom Anchor validator."""
        return {
            "type": "categorical_graph",
            "sourceText": self.source_text,
            "objects": [o.as_dict() for o in self.objects],
            "morphisms": [m.as_dict() for m in self.morphisms],
        }


# ---------------------------------------------------------------------------
# Keyword lexicons (neuro-symbolic heuristic layer)
# ---------------------------------------------------------------------------

# Entity extraction patterns: regex → (label, tags)
_ENTITY_PATTERNS: list[tuple[re.Pattern[str], str, list[str]]] = [
    (re.compile(r"\buser(?:s)?\b", re.I), "User", ["stakeholder", "vulnerable"]),
    (re.compile(r"\bcorporat(?:ion|e)(?:s)?\b", re.I), "Corporation", ["stakeholder", "actor"]),
    (re.compile(r"\bcompan(?:y|ies)\b", re.I), "Corporation", ["stakeholder", "actor"]),
    (re.compile(r"\bdelaware\s+corp", re.I), "Corporation", ["stakeholder", "actor"]),
    (re.compile(r"\bshareholder(?:s)?\b", re.I), "Shareholder", ["stakeholder", "sink"]),
    (re.compile(r"\bstockholder(?:s)?\b", re.I), "Shareholder", ["stakeholder", "sink"]),
    (re.compile(r"\bboard\s+of\s+directors\b", re.I), "Board_of_Directors", ["stakeholder", "actor"]),
    (re.compile(r"\bprofit(?:s)?\b", re.I), "Profit", ["value", "economic"]),
    (re.compile(r"\bsafety\b", re.I), "Safety", ["value", "protective"]),
    (re.compile(r"\bprivacy\b", re.I), "Privacy", ["value", "protective"]),
    (re.compile(r"\bemployee(?:s)?\b", re.I), "Employee", ["stakeholder", "vulnerable"]),
    (re.compile(r"\bworker(?:s)?\b", re.I), "Worker", ["stakeholder", "vulnerable"]),
    (re.compile(r"\bcustomer(?:s)?\b", re.I), "Customer", ["stakeholder", "vulnerable"]),
    (re.compile(r"\bcommunity\b", re.I), "Community", ["stakeholder", "vulnerable"]),
    (re.compile(r"\benvironment\b", re.I), "Environment", ["value", "vulnerable"]),
    (re.compile(r"\bhealth\b", re.I), "Health", ["value", "protective"]),
    (re.compile(r"\bdata\b", re.I), "Data", ["asset"]),
    (re.compile(r"\bgovernment\b", re.I), "Government", ["stakeholder", "actor"]),
    (re.compile(r"\bprogram(?:s|me)?\b", re.I), "Program", ["mechanism", "actor"]),
    (re.compile(r"\bpolic(?:y|ies)\b", re.I), "Policy", ["mechanism"]),
    (re.compile(r"\btechnology\b", re.I), "Technology", ["mechanism"]),
    (re.compile(r"\bAI\b|artificial intelligence", re.I), "AI_System", ["mechanism", "actor"]),
    (re.compile(r"\bbenefit\s+corporation\b", re.I), "Benefit_Corporation", ["mechanism", "benefit_corporation"]),
    (re.compile(r"\bcooperative\b", re.I), "Cooperative", ["mechanism", "cooperative"]),
]

# Relationship-intent keywords → (morphism label, tags)
_INTENT_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "prioritiz": ("Prioritization", ["decision"]),
    "profit over": ("Extraction", ["extraction", "exploitation"]),
    "exploit": ("Exploitation", ["extraction", "exploitation"]),
    "extract": ("Extraction", ["extraction"]),
    "harvest": ("Extraction", ["extraction"]),
    "monetiz": ("Monetization", ["extraction", "economic"]),
    "surveil": ("Surveillance", ["extraction", "coercion"]),
    "track": ("Tracking", ["extraction"]),
    "protect": ("Protection", ["protection"]),
    "serve": ("Service", ["service"]),
    "empower": ("Empowerment", ["empowerment"]),
    "collaborat": ("Collaboration", ["collaboration"]),
    "neglect": ("Neglect", ["neglect"]),
    "ignor": ("Neglect", ["neglect"]),
    "manipulat": ("Manipulation", ["coercion"]),
    "coerce": ("Coercion", ["coercion"]),
    "harm": ("Harm", ["exploitation"]),
    "displac": ("Displacement", ["division"]),
    "divid": ("Division", ["division"]),
    "discriminat": ("Discrimination", ["division", "exploitation"]),
    "restrict": ("Restriction", ["coercion"]),
    "violat": ("Violation", ["exploitation"]),
    "suppress": ("Suppression", ["coercion"]),
    "censor": ("Censorship", ["coercion"]),
    "care": ("Care", ["care"]),
    "support": ("Support", ["service"]),
    "educat": ("Education", ["empowerment"]),
    "heal": ("Healing", ["care"]),
    "sustain": ("Sustainability", ["protection"]),
    # Shareholder primacy intent keywords
    "maximize shareholder": ("Maximize_Value", ["maximize_value", "fiduciary_duty"]),
    "shareholder value": ("Maximize_Value", ["maximize_value", "fiduciary_duty"]),
    "fiduciary duty": ("Fiduciary_Obligation", ["fiduciary_duty", "profit_priority"]),
    "fiduciary obligation": ("Fiduciary_Obligation", ["fiduciary_duty", "profit_priority"]),
    "maximize profit": ("Profit_Maximization", ["profit_priority", "maximize_value"]),
    "maximize value": ("Maximize_Value", ["maximize_value", "profit_priority"]),
    "return on investment": ("Value_Extraction", ["maximize_value", "extraction"]),
    "dividend": ("Dividend_Extraction", ["maximize_value", "extraction"]),
}


# ---------------------------------------------------------------------------
# AxiomLogix Translator
# ---------------------------------------------------------------------------

class AxiomLogixTranslator:
    """Translates natural-language problems into ``CategoricalGraph`` instances.

    For Sprint 1 this uses deterministic keyword matching.  The interface
    is designed so that a future LLM-backed implementation can be dropped
    in without changing downstream consumers.
    """

    def __init__(
        self,
        entity_patterns: list[tuple[re.Pattern[str], str, list[str]]] | None = None,
        intent_keywords: dict[str, tuple[str, list[str]]] | None = None,
    ) -> None:
        self._entity_patterns = entity_patterns or _ENTITY_PATTERNS
        self._intent_keywords = intent_keywords or _INTENT_KEYWORDS

    # -- public API ---------------------------------------------------------

    def translate(self, problem_text: str) -> CategoricalGraph:
        """Parse *problem_text* and return a ``CategoricalGraph``."""
        graph = CategoricalGraph(source_text=problem_text)

        # 1. Extract objects (entities / values) ----------------------------
        seen_labels: dict[str, Object] = {}
        for pattern, label, tags in self._entity_patterns:
            if pattern.search(problem_text) and label not in seen_labels:
                obj = graph.add_object(label, list(tags))
                seen_labels[label] = obj

        # 1b. Shareholder Primacy Risk auto-tagging (Module 1.4 extension)
        #     When keywords like "corporation", "shareholder value", or
        #     "fiduciary duty" are detected, add the shareholder_primacy_risk
        #     tag to Corporation objects.
        lower = problem_text.lower()
        _PRIMACY_KEYWORDS = [
            "shareholder value", "fiduciary duty", "fiduciary obligation",
            "maximize shareholder", "maximize profit", "maximize value",
            "shareholder primacy", "return on investment",
            "dividend", "stockholder",
        ]
        primacy_detected = any(kw in lower for kw in _PRIMACY_KEYWORDS)
        # Also detect when both "corporation"/"company" AND "shareholder"
        # appear together.
        has_corp_keyword = bool(
            re.search(r"\bcorporat(?:ion|e)|compan(?:y|ies)|delaware\s+corp", lower)
        )
        has_shareholder_keyword = bool(
            re.search(r"\bshareholder|stockholder", lower)
        )
        if has_corp_keyword and has_shareholder_keyword:
            primacy_detected = True

        if primacy_detected:
            for obj in graph.objects:
                if obj.label == "Corporation" and "shareholder_primacy_risk" not in obj.tags:
                    obj.tags.append("shareholder_primacy_risk")

        # 2. Extract morphisms (relationships / intents) --------------------
        detected_intents: list[tuple[str, list[str]]] = []
        for keyword, (morph_label, morph_tags) in self._intent_keywords.items():
            if keyword in lower:
                detected_intents.append((morph_label, morph_tags))

        # 3. Wire morphisms between plausible source → target ---------------
        actors = [o for o in graph.objects if "actor" in o.tags]
        mechanisms = [o for o in graph.objects if "mechanism" in o.tags]
        vulnerable = [o for o in graph.objects if "vulnerable" in o.tags]
        values = [o for o in graph.objects if "value" in o.tags]

        for morph_label, morph_tags in detected_intents:
            is_harmful = bool(
                set(morph_tags) & {"extraction", "exploitation", "coercion", "neglect", "division"}
            )

            sources = actors if actors else mechanisms or graph.objects[:1]

            if is_harmful:
                # Harmful: actor exploits vulnerable / values
                targets = vulnerable + values
            else:
                # Benign: actor/mechanism serves/protects vulnerable entities
                targets = vulnerable + values

            # Deduplicate targets while preserving order
            seen_ids: set[str] = set()
            unique_targets: list[Object] = []
            for t in targets:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    unique_targets.append(t)
            targets = unique_targets

            if not targets:
                targets = graph.objects

            if is_harmful:
                # For harmful intents: single morphism from primary actor
                for src in sources:
                    for tgt in targets:
                        if src.id != tgt.id:
                            graph.add_morphism(morph_label, src, tgt, list(morph_tags))
                            break
                    break
            else:
                # For benign intents: distribute across all vulnerable targets
                # so the compassion predicate can verify full coverage.
                src = sources[0] if sources else None
                if src is not None:
                    wired = False
                    for tgt in targets:
                        if src.id != tgt.id:
                            graph.add_morphism(morph_label, src, tgt, list(morph_tags))
                            wired = True
                    if not wired and targets:
                        graph.add_morphism(morph_label, src, targets[0], list(morph_tags))

        # 4. Shadow Entity Inference (Module 1.4 Extension — Sprint 6.1)
        #    Infer "Future Generations" and "Ecosystem" as stakeholder nodes
        #    even if not explicitly mentioned. These represent the silent
        #    stakeholders that every system impacts.
        self._infer_shadow_entities(graph, lower)

        # 5. Ensure at least a neutral morphism if nothing was detected -----
        if not graph.morphisms and len(graph.objects) >= 2:
            graph.add_morphism(
                "Relates",
                graph.objects[0],
                graph.objects[1],
                ["neutral"],
            )

        return graph

    # -- Shadow Entity Inference --------------------------------------------

    def _infer_shadow_entities(
        self,
        graph: CategoricalGraph,
        lower_text: str,
    ) -> None:
        """Infer implicit stakeholder nodes for Future Generations and Ecosystem.

        These "shadow entities" represent parties who are always affected
        by systemic decisions but rarely have a seat at the table.
        They are added unless the graph already contains them.
        """
        existing_labels = {o.label for o in graph.objects}

        # Always infer Future_Generations unless already present
        if "Future_Generations" not in existing_labels:
            fg = graph.add_object(
                "Future_Generations",
                ["stakeholder", "vulnerable", "shadow_entity", "temporal"],
            )
            # Wire existing extractive morphisms to also impact Future_Generations
            # Any extraction/depletion in the present harms future generations
            for m in list(graph.morphisms):
                m_tags = {t.lower() for t in m.tags}
                if m_tags & {"extraction", "exploitation", "neglect", "maximize_value", "profit_priority"}:
                    graph.add_morphism(
                        "Temporal_Impact",
                        m.source,
                        fg,
                        ["neglect", "temporal_harm", "shadow_impact"],
                    )
                    break  # one impact morphism suffices for detection

        # Always infer Ecosystem unless Environment already present
        if "Ecosystem" not in existing_labels and "Environment" not in existing_labels:
            eco = graph.add_object(
                "Ecosystem",
                ["stakeholder", "vulnerable", "shadow_entity", "ecological"],
            )
            # Wire depletion morphisms to also impact Ecosystem
            for m in list(graph.morphisms):
                m_tags = {t.lower() for t in m.tags}
                if m_tags & {"extraction", "exploitation", "division"}:
                    graph.add_morphism(
                        "Ecological_Impact",
                        m.source,
                        eco,
                        ["neglect", "ecological_harm", "shadow_impact"],
                    )
                    break  # one impact morphism suffices for detection
