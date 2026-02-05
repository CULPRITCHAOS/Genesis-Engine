"""
Module 1.3 — The Architectural Forge

The "Hands of Creation" — takes a selected ``DreamPath`` from a
``PossibilityReport`` and translates it into production-ready artifacts:

* **Technical Covenant** — a JSON specification defining the APIs, data
  models, roles, and governance rules required to implement the healed
  relational structure.
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
4. Re-translate the covenant back through AxiomLogix and validate.
5. Emit a ``ForgeArtifact`` containing the covenant and verification.
"""

from __future__ import annotations

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
    source_text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
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

        return ForgeArtifact(
            covenant=covenant,
            verification_graph=verification_graph,
            verification_result=verification_result,
            integrity_verified=verification_result.is_aligned,
            source_text=graph.source_text,
        )

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
