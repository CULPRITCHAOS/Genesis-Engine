"""
Module 1.3 — The Architectural Forge

The "Hands of Creation" — takes a selected ``DreamPath`` from a
``PossibilityReport`` and translates it into production-ready artifacts:

* **Technical Covenant** — a JSON specification defining the APIs, data
  models, roles, and governance rules required to implement the healed
  relational structure.
* **Stewardship Manifesto** — a Markdown/YAML index for every blueprint
  containing alignment scores, governance summary, and a RegenerativeLoop
  section defining Repair Morphisms that trigger when simulation scores
  fall below 5.0.
* **AxiomLogix Verification** — every generated artifact is translated
  back into a ``CategoricalGraph`` via the AxiomLogix Translator and
  validated against the Prime Directive to ensure Compositional Integrity.

Pipeline
--------
1. Accept a ``DreamPath`` (selected solution from the Dream Engine).
2. Map the healed graph into concrete technical structures:
   - Entities   → Data models / API resources
   - Morphisms  → API endpoints / service contracts
   - Tags       → Access-control and governance annotations
3. Compose a ``TechnicalCovenant`` (the blueprint).
4. Generate a ``StewardshipManifesto`` (the index).
5. Re-translate the covenant back through AxiomLogix and validate.
6. Emit a ``ForgeArtifact`` containing the covenant, manifesto, and verification.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from genesis_engine.core.axiom_anchor import AxiomAnchor, ValidationResult
from genesis_engine.core.axiomlogix import (
    AxiomLogixTranslator,
    CategoricalGraph,
    Morphism,
    Object,
)
from genesis_engine.core.dream_engine import DreamPath, PathType


# ---------------------------------------------------------------------------
# Tag → technical-concern mapping
# ---------------------------------------------------------------------------

_TAG_TO_GOVERNANCE: dict[str, str] = {
    "protection": "access_controlled",
    "care": "audit_logged",
    "service": "rate_limited",
    "empowerment": "user_configurable",
    "collaboration": "shared_ownership",
}

_TAG_TO_HTTP_METHOD: dict[str, str] = {
    "protection": "GET",
    "care": "POST",
    "service": "GET",
    "empowerment": "PUT",
    "collaboration": "POST",
}

_ROLE_TO_DATA_TYPE: dict[str, str] = {
    "stakeholder": "entity",
    "actor": "service_principal",
    "vulnerable": "protected_entity",
    "value": "resource",
    "shared": "shared_resource",
    "mechanism": "governance_module",
    "protective": "policy_engine",
    "asset": "data_store",
}


# ---------------------------------------------------------------------------
# Data Model spec
# ---------------------------------------------------------------------------

@dataclass
class DataModelSpec:
    """A data model derived from a categorical Object."""

    name: str
    resource_type: str
    fields: list[dict[str, str]] = field(default_factory=list)
    governance: list[str] = field(default_factory=list)
    source_object_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "resourceType": self.resource_type,
            "fields": self.fields,
            "governance": self.governance,
        }


# ---------------------------------------------------------------------------
# API Endpoint spec
# ---------------------------------------------------------------------------

@dataclass
class EndpointSpec:
    """An API endpoint derived from a categorical Morphism."""

    path: str
    method: str
    name: str
    description: str
    source_model: str
    target_model: str
    governance: list[str] = field(default_factory=list)
    request_schema: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "name": self.name,
            "description": self.description,
            "sourceModel": self.source_model,
            "targetModel": self.target_model,
            "governance": self.governance,
            "requestSchema": self.request_schema,
            "responseSchema": self.response_schema,
        }


# ---------------------------------------------------------------------------
# Governance Rule
# ---------------------------------------------------------------------------

@dataclass
class GovernanceRule:
    """A governance constraint derived from morphism tags."""

    rule_id: str
    name: str
    description: str
    applies_to: list[str] = field(default_factory=list)
    enforcement: str = "required"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ruleId": self.rule_id,
            "name": self.name,
            "description": self.description,
            "appliesTo": self.applies_to,
            "enforcement": self.enforcement,
        }


# ---------------------------------------------------------------------------
# Repair Morphism (Sprint 7 — Regenerative Blueprinting)
# ---------------------------------------------------------------------------

@dataclass
class RepairMorphism:
    """A self-healing morphism that triggers when a simulation score drops.

    Repair Morphisms embody Compassion-Driven Resilience: the system
    "dies" to extraction and "resurrects" in coherence by automatically
    proposing structural corrections.
    """

    trigger_condition: str  # e.g. "sustainability_score < 5.0"
    source_model: str
    target_model: str
    repair_action: str  # human-readable description of the repair
    replacement_tags: list[str] = field(default_factory=list)
    priority: str = "critical"  # critical | high | medium

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggerCondition": self.trigger_condition,
            "sourceModel": self.source_model,
            "targetModel": self.target_model,
            "repairAction": self.repair_action,
            "replacementTags": self.replacement_tags,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Regenerative Loop (Sprint 7 — Regenerative Blueprinting)
# ---------------------------------------------------------------------------

@dataclass
class RegenerativeLoop:
    """Defines the self-healing contract for a blueprint.

    The RegenerativeLoop section of a Stewardship Manifesto specifies:
    - The simulation score threshold below which repair triggers
    - The Repair Morphisms that activate when the threshold is breached
    - The resurrection principle: Unity over Power
    """

    score_threshold: float = 5.0
    repair_morphisms: list[RepairMorphism] = field(default_factory=list)
    resurrection_principle: str = (
        "When coherence fails, the system dissolves extractive patterns "
        "and regenerates through stewardship morphisms. Unity over Power."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scoreThreshold": self.score_threshold,
            "repairMorphisms": [rm.as_dict() for rm in self.repair_morphisms],
            "resurrectionPrinciple": self.resurrection_principle,
        }

    def to_yaml_block(self) -> str:
        """Render as a YAML-compatible block for the manifesto."""
        lines = [
            "regenerative_loop:",
            f"  score_threshold: {self.score_threshold}",
            f"  resurrection_principle: \"{self.resurrection_principle}\"",
            "  repair_morphisms:",
        ]
        for rm in self.repair_morphisms:
            lines.append(f"    - trigger: \"{rm.trigger_condition}\"")
            lines.append(f"      source: \"{rm.source_model}\"")
            lines.append(f"      target: \"{rm.target_model}\"")
            lines.append(f"      action: \"{rm.repair_action}\"")
            lines.append(f"      tags: [{', '.join(rm.replacement_tags)}]")
            lines.append(f"      priority: {rm.priority}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stewardship Manifesto (Sprint 7 — Regenerative Blueprinting)
# ---------------------------------------------------------------------------

@dataclass
class StewardshipManifesto:
    """The index document for every blueprint produced by the Forge.

    Rendered as Markdown with YAML frontmatter, the manifesto serves as
    the human-readable "soul" of a Technical Covenant — making the
    blueprint's alignment scores, governance rules, and regenerative
    contracts visible at a glance.
    """

    covenant_title: str
    source_path_type: str
    prime_directive: str
    alignment_scores: dict[str, float] = field(default_factory=dict)
    governance_summary: list[str] = field(default_factory=list)
    regenerative_loop: RegenerativeLoop = field(default_factory=RegenerativeLoop)
    integrity_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stewardshipManifesto": {
                "covenantTitle": self.covenant_title,
                "sourcePathType": self.source_path_type,
                "primeDirective": self.prime_directive,
                "alignmentScores": {
                    k: round(v, 4) for k, v in self.alignment_scores.items()
                },
                "governanceSummary": self.governance_summary,
                "regenerativeLoop": self.regenerative_loop.as_dict(),
                "integrityHash": self.integrity_hash,
                "timestamp": self.timestamp,
            }
        }

    def to_markdown(self) -> str:
        """Render the manifesto as Markdown with YAML frontmatter."""
        lines = ["---"]
        lines.append(f"title: \"{self.covenant_title}\"")
        lines.append(f"path_type: {self.source_path_type}")
        lines.append(f"prime_directive: \"{self.prime_directive}\"")
        lines.append(f"integrity_hash: \"{self.integrity_hash}\"")
        lines.append(f"timestamp: \"{self.timestamp}\"")

        # Alignment scores
        lines.append("alignment_scores:")
        for k, v in self.alignment_scores.items():
            lines.append(f"  {k}: {v:.4f}")

        # Regenerative Loop as YAML
        lines.append(self.regenerative_loop.to_yaml_block())
        lines.append("---")
        lines.append("")

        # Markdown body
        lines.append(f"# {self.covenant_title}")
        lines.append("")
        lines.append(f"> *\"{self.prime_directive}\"*")
        lines.append("")

        lines.append("## Alignment Scores")
        lines.append("")
        for k, v in self.alignment_scores.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"- **{k}**: `{v:.4f}` {bar}")
        lines.append("")

        lines.append("## Governance Summary")
        lines.append("")
        for rule in self.governance_summary:
            lines.append(f"- {rule}")
        lines.append("")

        lines.append("## Regenerative Loop")
        lines.append("")
        lines.append(
            f"**Threshold**: Score below `{self.regenerative_loop.score_threshold}` "
            f"triggers repair."
        )
        lines.append("")
        lines.append(f"**Principle**: {self.regenerative_loop.resurrection_principle}")
        lines.append("")

        if self.regenerative_loop.repair_morphisms:
            lines.append("### Repair Morphisms")
            lines.append("")
            for rm in self.regenerative_loop.repair_morphisms:
                lines.append(f"- **{rm.source_model} → {rm.target_model}**")
                lines.append(f"  - Trigger: `{rm.trigger_condition}`")
                lines.append(f"  - Action: {rm.repair_action}")
                lines.append(f"  - Tags: `{', '.join(rm.replacement_tags)}`")
                lines.append(f"  - Priority: {rm.priority}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Technical Covenant
# ---------------------------------------------------------------------------

@dataclass
class TechnicalCovenant:
    """The complete technical blueprint produced by the Forge."""

    title: str
    source_path_type: str
    description: str
    data_models: list[DataModelSpec] = field(default_factory=list)
    endpoints: list[EndpointSpec] = field(default_factory=list)
    governance_rules: list[GovernanceRule] = field(default_factory=list)
    prime_directive_statement: str = "Does this serve Love?"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "technicalCovenant": {
                "title": self.title,
                "sourcePathType": self.source_path_type,
                "description": self.description,
                "primeDirective": self.prime_directive_statement,
                "timestamp": self.timestamp,
                "dataModels": [m.as_dict() for m in self.data_models],
                "endpoints": [e.as_dict() for e in self.endpoints],
                "governanceRules": [r.as_dict() for r in self.governance_rules],
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Forge Artifact (output wrapper)
# ---------------------------------------------------------------------------

@dataclass
class ForgeArtifact:
    """Top-level output of the Architectural Forge."""

    covenant: TechnicalCovenant
    verification_graph: CategoricalGraph
    verification_result: ValidationResult
    integrity_verified: bool
    manifesto: StewardshipManifesto | None = None
    source_text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "forgeArtifact": {
                "timestamp": self.timestamp,
                "sourceText": self.source_text,
                "integrityVerified": self.integrity_verified,
                "covenant": self.covenant.as_dict()["technicalCovenant"],
                "verification": {
                    "graph": self.verification_graph.as_dict(),
                    "validation": self.verification_result.as_dict(),
                },
            }
        }
        if self.manifesto:
            result["forgeArtifact"]["stewardshipManifesto"] = (
                self.manifesto.as_dict()["stewardshipManifesto"]
            )
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Architectural Forge
# ---------------------------------------------------------------------------

class ArchitecturalForge:
    """Translates Dream Engine solutions into production-ready blueprints.

    Parameters
    ----------
    anchor : AxiomAnchor | None
        Shared Axiom Anchor for compositional integrity verification.
    translator : AxiomLogixTranslator | None
        Translator for re-verifying generated artifacts.
    """

    def __init__(
        self,
        anchor: AxiomAnchor | None = None,
        translator: AxiomLogixTranslator | None = None,
    ) -> None:
        self.anchor = anchor or AxiomAnchor()
        self.translator = translator or AxiomLogixTranslator()

    # -- public API ---------------------------------------------------------

    def forge(self, dream_path: DreamPath) -> ForgeArtifact:
        """Generate a ``ForgeArtifact`` from a selected ``DreamPath``."""

        graph = dream_path.healed_graph

        # Step 1: Generate data models from objects.
        data_models = [self._object_to_model(obj) for obj in graph.objects]

        # Step 2: Generate API endpoints from morphisms.
        endpoints = [
            self._morphism_to_endpoint(morph, graph)
            for morph in graph.morphisms
        ]

        # Step 3: Extract governance rules from tag patterns.
        governance_rules = self._extract_governance_rules(graph)

        # Step 4: Compose the Technical Covenant.
        covenant = TechnicalCovenant(
            title=self._generate_title(dream_path),
            source_path_type=dream_path.path_type.value,
            description=dream_path.description,
            data_models=data_models,
            endpoints=endpoints,
            governance_rules=governance_rules,
        )

        # Step 5: AxiomLogix verification — translate the covenant
        # back into a categorical graph and validate.
        verification_graph = self._covenant_to_graph(covenant)
        verification_result = self.anchor.validate(
            verification_graph.as_artefact(),
        )

        # Step 6: Generate Stewardship Manifesto (Sprint 7).
        manifesto = self._generate_manifesto(
            covenant, verification_result, graph,
        )

        return ForgeArtifact(
            covenant=covenant,
            verification_graph=verification_graph,
            verification_result=verification_result,
            integrity_verified=verification_result.is_aligned,
            manifesto=manifesto,
            source_text=graph.source_text,
        )

    # -- manifesto generation (Sprint 7) ------------------------------------

    def _generate_manifesto(
        self,
        covenant: TechnicalCovenant,
        validation: ValidationResult,
        source_graph: CategoricalGraph,
    ) -> StewardshipManifesto:
        """Generate a Stewardship Manifesto as the index for this blueprint.

        The manifesto includes alignment scores, governance summary, and
        a RegenerativeLoop section with Repair Morphisms.
        """
        # Build governance summary from rules
        governance_summary = [
            f"{rule.name}: {rule.description}"
            for rule in covenant.governance_rules
        ]

        # Generate repair morphisms from the graph's structure
        repair_morphisms = self._generate_repair_morphisms(
            covenant, source_graph,
        )

        regen_loop = RegenerativeLoop(
            score_threshold=5.0,
            repair_morphisms=repair_morphisms,
        )

        manifesto = StewardshipManifesto(
            covenant_title=covenant.title,
            source_path_type=covenant.source_path_type,
            prime_directive=covenant.prime_directive_statement,
            alignment_scores=dict(validation.principle_scores),
            governance_summary=governance_summary,
            regenerative_loop=regen_loop,
        )

        # Compute integrity hash
        hash_content = json.dumps(
            manifesto.as_dict(), sort_keys=True
        ).encode("utf-8")
        manifesto.integrity_hash = hashlib.sha256(hash_content).hexdigest()

        return manifesto

    @staticmethod
    def _generate_repair_morphisms(
        covenant: TechnicalCovenant,
        source_graph: CategoricalGraph,
    ) -> list[RepairMorphism]:
        """Generate Repair Morphisms for the RegenerativeLoop.

        For each endpoint in the covenant, if the underlying morphism
        has extractive/harmful potential, create a repair morphism that
        would replace it with a stewardship pattern if the simulation
        score drops below threshold.

        Compassion-Driven Resilience: the system "dies" to extraction
        and "resurrects" in coherence.
        """
        repairs: list[RepairMorphism] = []

        # Map of extractive tag patterns → their healing replacements
        _HEAL_MAP: dict[str, tuple[str, list[str]]] = {
            "extraction": ("Fair_Reciprocity", ["service", "collaboration"]),
            "exploitation": ("Mutual_Benefit", ["service", "empowerment"]),
            "coercion": ("Informed_Consent", ["empowerment", "protection"]),
            "neglect": ("Active_Care", ["care", "protection"]),
            "division": ("Unification", ["collaboration", "care"]),
        }

        for endpoint in covenant.endpoints:
            # Check if any governance annotation suggests vulnerability
            for tag, (healed_label, healed_tags) in _HEAL_MAP.items():
                if tag in endpoint.name.lower() or tag in " ".join(endpoint.governance).lower():
                    repairs.append(RepairMorphism(
                        trigger_condition="sustainability_score < 5.0",
                        source_model=endpoint.source_model,
                        target_model=endpoint.target_model,
                        repair_action=(
                            f"Replace {endpoint.name} with {healed_label}: "
                            f"dissolve extractive pattern, resurrect as "
                            f"stewardship morphism."
                        ),
                        replacement_tags=healed_tags,
                        priority="critical",
                    ))

        # Always add a universal repair morphism for any blueprint:
        # if no specific repairs were generated, add a generic stewardship repair
        if not repairs:
            model_names = [m.name for m in covenant.data_models]
            if len(model_names) >= 2:
                repairs.append(RepairMorphism(
                    trigger_condition="sustainability_score < 5.0",
                    source_model=model_names[0],
                    target_model=model_names[1],
                    repair_action=(
                        "Inject stewardship morphism: ensure the primary actor "
                        "serves the primary beneficiary with care and protection "
                        "when system coherence degrades."
                    ),
                    replacement_tags=["care", "protection", "service"],
                    priority="high",
                ))

        return repairs

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _object_to_model(obj: Object) -> DataModelSpec:
        """Convert a categorical Object into a DataModelSpec."""
        # Determine the primary resource type from tags.
        resource_type = "entity"
        for tag in obj.tags:
            if tag in _ROLE_TO_DATA_TYPE:
                resource_type = _ROLE_TO_DATA_TYPE[tag]
                break

        # Generate canonical fields based on the role.
        fields: list[dict[str, str]] = [
            {"name": "id", "type": "uuid", "description": f"Unique identifier for {obj.label}"},
            {"name": "name", "type": "string", "description": f"Display name of {obj.label}"},
            {"name": "created_at", "type": "datetime", "description": "Creation timestamp"},
        ]

        if "vulnerable" in obj.tags:
            fields.append({"name": "consent_status", "type": "enum", "description": "Active consent state"})
            fields.append({"name": "rights_manifest", "type": "json", "description": "Enumerated rights"})
        if "actor" in obj.tags:
            fields.append({"name": "accountability_log", "type": "json[]", "description": "Audit trail of actions"})
            fields.append({"name": "stewardship_scope", "type": "string[]", "description": "Domains of responsibility"})
        if "protective" in obj.tags:
            fields.append({"name": "policy_rules", "type": "json[]", "description": "Active governance policies"})
            fields.append({"name": "enforcement_mode", "type": "enum", "description": "strict | advisory | transparent"})
        if "shared" in obj.tags:
            fields.append({"name": "access_policy", "type": "json", "description": "Who can access and how"})
            fields.append({"name": "benefit_distribution", "type": "json", "description": "How value is shared"})

        # Governance annotations from tags.
        governance = []
        for tag in obj.tags:
            if tag in _TAG_TO_GOVERNANCE:
                governance.append(_TAG_TO_GOVERNANCE[tag])

        return DataModelSpec(
            name=obj.label,
            resource_type=resource_type,
            fields=fields,
            governance=governance,
            source_object_id=obj.id,
        )

    @staticmethod
    def _morphism_to_endpoint(morph: Morphism, graph: CategoricalGraph) -> EndpointSpec:
        """Convert a categorical Morphism into an EndpointSpec."""
        # Resolve labels.
        src_label = morph.source
        tgt_label = morph.target
        for obj in graph.objects:
            if obj.id == morph.source:
                src_label = obj.label
            if obj.id == morph.target:
                tgt_label = obj.label

        # Determine HTTP method from primary tag.
        method = "POST"
        for tag in morph.tags:
            if tag in _TAG_TO_HTTP_METHOD:
                method = _TAG_TO_HTTP_METHOD[tag]
                break

        # Build path.
        path_slug = morph.label.lower().replace("_", "-")
        source_slug = src_label.lower().replace("_", "-")
        target_slug = tgt_label.lower().replace("_", "-")
        path = f"/api/v1/{source_slug}/{path_slug}/{target_slug}"

        # Governance annotations.
        governance = []
        for tag in morph.tags:
            if tag in _TAG_TO_GOVERNANCE:
                governance.append(_TAG_TO_GOVERNANCE[tag])

        return EndpointSpec(
            path=path,
            method=method,
            name=morph.label,
            description=(
                f"{morph.label} relationship: {src_label} serves {tgt_label} "
                f"with intent [{', '.join(morph.tags)}]."
            ),
            source_model=src_label,
            target_model=tgt_label,
            governance=governance,
            request_schema={
                "type": "object",
                "properties": {
                    f"{source_slug}_id": {"type": "string", "format": "uuid"},
                    f"{target_slug}_id": {"type": "string", "format": "uuid"},
                    "context": {"type": "object"},
                },
                "required": [f"{source_slug}_id", f"{target_slug}_id"],
            },
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["fulfilled", "pending", "denied"]},
                    "audit_entry": {"type": "object"},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
        )

    @staticmethod
    def _extract_governance_rules(graph: CategoricalGraph) -> list[GovernanceRule]:
        """Derive governance rules from the pattern of tags in the graph."""
        rules: list[GovernanceRule] = []
        seen: set[str] = set()

        # Collect all unique tag combinations.
        all_tags: set[str] = set()
        for morph in graph.morphisms:
            all_tags.update(morph.tags)

        if "protection" in all_tags:
            key = "protection"
            if key not in seen:
                seen.add(key)
                rules.append(GovernanceRule(
                    rule_id=f"gov-{uuid.uuid4().hex[:8]}",
                    name="Data Protection Mandate",
                    description=(
                        "All endpoints handling protected entities must enforce "
                        "consent verification and data minimisation."
                    ),
                    applies_to=[m.label for m in graph.morphisms if "protection" in m.tags],
                    enforcement="required",
                ))

        if "care" in all_tags:
            key = "care"
            if key not in seen:
                seen.add(key)
                rules.append(GovernanceRule(
                    rule_id=f"gov-{uuid.uuid4().hex[:8]}",
                    name="Duty of Care Audit",
                    description=(
                        "All care-tagged interactions must be audit-logged with "
                        "full traceability to the Prime Directive."
                    ),
                    applies_to=[m.label for m in graph.morphisms if "care" in m.tags],
                    enforcement="required",
                ))

        if "empowerment" in all_tags:
            key = "empowerment"
            if key not in seen:
                seen.add(key)
                rules.append(GovernanceRule(
                    rule_id=f"gov-{uuid.uuid4().hex[:8]}",
                    name="Empowerment Guarantee",
                    description=(
                        "Empowerment endpoints must provide user-configurable "
                        "controls and transparent decision explanations."
                    ),
                    applies_to=[m.label for m in graph.morphisms if "empowerment" in m.tags],
                    enforcement="required",
                ))

        if "collaboration" in all_tags:
            key = "collaboration"
            if key not in seen:
                seen.add(key)
                rules.append(GovernanceRule(
                    rule_id=f"gov-{uuid.uuid4().hex[:8]}",
                    name="Shared Ownership Protocol",
                    description=(
                        "Collaborative resources must implement shared-ownership "
                        "access controls with equitable benefit distribution."
                    ),
                    applies_to=[m.label for m in graph.morphisms if "collaboration" in m.tags],
                    enforcement="required",
                ))

        if "service" in all_tags:
            key = "service"
            if key not in seen:
                seen.add(key)
                rules.append(GovernanceRule(
                    rule_id=f"gov-{uuid.uuid4().hex[:8]}",
                    name="Service Level Covenant",
                    description=(
                        "Service endpoints must be rate-limited, monitored, "
                        "and bound by explicit service-level agreements."
                    ),
                    applies_to=[m.label for m in graph.morphisms if "service" in m.tags],
                    enforcement="required",
                ))

        return rules

    def _covenant_to_graph(self, covenant: TechnicalCovenant) -> CategoricalGraph:
        """Re-translate a TechnicalCovenant back into a CategoricalGraph
        for AxiomLogix compositional integrity verification."""
        graph = CategoricalGraph(
            source_text=f"Technical Covenant: {covenant.title}",
        )

        # Create objects from data models.
        model_objects: dict[str, Object] = {}
        for model in covenant.data_models:
            tags = list(model.governance)
            # Map resource types back to categorical tags.
            if model.resource_type in ("protected_entity",):
                tags.extend(["stakeholder", "vulnerable"])
            elif model.resource_type in ("service_principal",):
                tags.extend(["stakeholder", "actor"])
            elif model.resource_type in ("shared_resource",):
                tags.extend(["value", "shared"])
            elif model.resource_type in ("policy_engine", "governance_module"):
                tags.extend(["mechanism", "protective"])
            elif model.resource_type == "entity":
                tags.append("stakeholder")
            elif model.resource_type == "resource":
                tags.append("value")
            obj = graph.add_object(model.name, tags)
            model_objects[model.name] = obj

        # Create morphisms from endpoints.
        for endpoint in covenant.endpoints:
            src = model_objects.get(endpoint.source_model)
            tgt = model_objects.get(endpoint.target_model)
            if src and tgt:
                # Map governance annotations back to semantic tags.
                tags = self._governance_to_tags(endpoint.governance)
                graph.add_morphism(endpoint.name, src, tgt, tags)

        return graph

    @staticmethod
    def _governance_to_tags(governance: list[str]) -> list[str]:
        """Reverse-map governance annotations to semantic tags."""
        reverse_map = {v: k for k, v in _TAG_TO_GOVERNANCE.items()}
        tags = []
        for g in governance:
            if g in reverse_map:
                tags.append(reverse_map[g])
        return tags or ["service"]

    @staticmethod
    def _generate_title(path: DreamPath) -> str:
        """Generate a human-readable title for the covenant."""
        type_labels = {
            PathType.REFORM: "Reform Covenant",
            PathType.REINVENTION: "Stewardship Covenant",
            PathType.DISSOLUTION: "Cooperative Covenant",
        }
        base = type_labels.get(path.path_type, "Technical Covenant")
        context = path.healed_graph.source_text[:60] if path.healed_graph.source_text else "Unknown"
        return f"{base}: {context}"
