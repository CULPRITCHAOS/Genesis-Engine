"""
Module 2.3 — The Continuity Bridge

Implements the ``.genesis_soul`` file format — a secure, portable package
that captures an individual's evolving relationship with the Genesis Engine.

A ``.genesis_soul`` file contains:

1. **Axiom Anchor State** — the user's Prime Directive configuration and
   any custom axiom predicates registered during the session.
2. **Graph History** — a chronological record of every ``CategoricalGraph``
   generated during interaction (translations, healed graphs, verification
   graphs).
3. **Wisdom Log** — a ledger of identified disharmonies and their
   resolutions, forming an evolving record of ethical growth.
4. **Forge Artifacts** — references to Technical Covenants produced by the
   Architectural Forge.
5. **Hash Chain** — SHA-256 hash-chaining for tamper-proof integrity.

Security Features:
- **Hash Chaining**: Each wisdom entry's hash incorporates the previous
  entry's hash, creating an immutable audit trail.
- **Redaction Discipline**: Sensitive patterns (API keys, credentials,
  prompts with PII) are automatically scrubbed before storage.

The file is serialised as a signed JSON envelope with a SHA-256 integrity
hash, ready for future encryption and portability.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import re

from genesis_engine.core.axiom_anchor import AxiomAnchor, PrimeDirective, ValidationResult
from genesis_engine.core.axiomlogix import CategoricalGraph
from genesis_engine.core.deconstruction_engine import DisharmonyReport
from genesis_engine.core.dream_engine import DreamPath, PossibilityReport


# ---------------------------------------------------------------------------
# Redaction patterns — sensitive data to scrub before storage
# ---------------------------------------------------------------------------

_REDACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API keys and tokens
    (re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(api[_-]?key\s*[:=]\s*['\"]?)[a-zA-Z0-9_-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\b(token\s*[:=]\s*['\"]?)[a-zA-Z0-9_-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\b(secret\s*[:=]\s*['\"]?)[a-zA-Z0-9_-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\b(password\s*[:=]\s*['\"]?)[^\s'\"]+", re.I), r"\1[REDACTED]"),
    # Email addresses
    (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), "[REDACTED_EMAIL]"),
    # Phone numbers (basic patterns)
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    # Credit card numbers (basic patterns)
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[REDACTED_CC]"),
    # Social Security Numbers
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[REDACTED_SSN]"),
]


def redact_sensitive(text: str) -> str:
    """Apply redaction patterns to remove sensitive data from *text*."""
    result = text
    for pattern, replacement in _REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# Graph History Entry
# ---------------------------------------------------------------------------

@dataclass
class GraphHistoryEntry:
    """A single entry in the soul's graph history."""

    graph: CategoricalGraph
    phase: str  # "translation" | "deconstruction" | "dream" | "verification"
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "label": self.label,
            "timestamp": self.timestamp,
            "graph": self.graph.as_dict(),
        }


# ---------------------------------------------------------------------------
# Wisdom Log Entry
# ---------------------------------------------------------------------------

@dataclass
class WisdomEntry:
    """A record of a disharmony identified and (optionally) resolved.

    Includes a hash for chain integrity verification.
    """

    source_text: str
    disharmony_summary: str
    unity_impact: float
    compassion_deficit: float
    resolution_path: str  # "reform" | "reinvention" | "dissolution" | "unresolved"
    resolution_summary: str = ""
    covenant_title: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prev_hash: str = ""  # Hash of previous entry (for chain integrity)
    entry_hash: str = ""  # This entry's hash

    def compute_hash(self, prev_hash: str = "") -> str:
        """Compute SHA-256 hash incorporating the previous entry's hash."""
        content = (
            f"{prev_hash}|{self.source_text}|{self.disharmony_summary}|"
            f"{self.unity_impact}|{self.compassion_deficit}|{self.resolution_path}|"
            f"{self.resolution_summary}|{self.covenant_title}|{self.timestamp}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sourceText": self.source_text,
            "disharmonySummary": self.disharmony_summary,
            "unityImpact": round(self.unity_impact, 2),
            "compassionDeficit": round(self.compassion_deficit, 2),
            "resolutionPath": self.resolution_path,
            "resolutionSummary": self.resolution_summary,
            "covenantTitle": self.covenant_title,
            "timestamp": self.timestamp,
            "prevHash": self.prev_hash,
            "entryHash": self.entry_hash,
        }


# ---------------------------------------------------------------------------
# Genesis Soul
# ---------------------------------------------------------------------------

@dataclass
class GenesisSoul:
    """The portable identity file for a Genesis Engine session.

    This is the in-memory representation of a ``.genesis_soul`` file.
    """

    soul_id: str = field(default_factory=lambda: f"soul-{uuid.uuid4().hex[:12]}")
    version: str = "0.1.0"

    # Core state
    directive: PrimeDirective = field(default_factory=PrimeDirective)
    alignment_threshold: float = 0.5

    # History
    graph_history: list[GraphHistoryEntry] = field(default_factory=list)
    wisdom_log: list[WisdomEntry] = field(default_factory=list)
    forge_artifacts: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- mutation helpers ---------------------------------------------------

    def record_graph(
        self,
        graph: CategoricalGraph,
        phase: str,
        label: str = "",
    ) -> None:
        """Append a graph to the history."""
        self.graph_history.append(GraphHistoryEntry(
            graph=graph, phase=phase, label=label,
        ))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_wisdom(
        self,
        report: DisharmonyReport,
        resolution_path: str = "unresolved",
        resolution_summary: str = "",
        covenant_title: str = "",
    ) -> None:
        """Record a disharmony and its resolution with redaction and hash-chaining."""
        flagged = [f for f in report.findings if f.disharmony_score > 0]
        summary = "; ".join(
            f"{f.label} ({f.source} -> {f.target})" for f in flagged
        ) if flagged else "No disharmony detected"

        # Apply redaction discipline to sensitive fields
        redacted_source = redact_sensitive(report.source_text)
        redacted_summary = redact_sensitive(resolution_summary)

        # Get the previous hash for chaining
        prev_hash = self.wisdom_log[-1].entry_hash if self.wisdom_log else ""

        entry = WisdomEntry(
            source_text=redacted_source,
            disharmony_summary=summary,
            unity_impact=report.unity_impact,
            compassion_deficit=report.compassion_deficit,
            resolution_path=resolution_path,
            resolution_summary=redacted_summary,
            covenant_title=covenant_title,
            prev_hash=prev_hash,
        )
        # Compute and set the entry's hash
        entry.entry_hash = entry.compute_hash(prev_hash)

        self.wisdom_log.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_forge_artifact(self, artifact_dict: dict[str, Any]) -> None:
        """Record a forge artifact reference."""
        self.forge_artifacts.append(artifact_dict)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # -- serialisation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "soulId": self.soul_id,
            "version": self.version,
            "axiomAnchorState": {
                "directive": self.directive.as_dict(),
                "alignmentThreshold": self.alignment_threshold,
            },
            "graphHistory": [e.as_dict() for e in self.graph_history],
            "wisdomLog": [w.as_dict() for w in self.wisdom_log],
            "forgeArtifacts": self.forge_artifacts,
            "metadata": {
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "graphCount": len(self.graph_history),
                "wisdomCount": len(self.wisdom_log),
                "forgeCount": len(self.forge_artifacts),
            },
        }
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Continuity Bridge
# ---------------------------------------------------------------------------

class ContinuityBridge:
    """Manages the lifecycle of ``.genesis_soul`` files.

    Responsibilities:
    - Create new souls
    - Record interactions (graphs, wisdom, artifacts)
    - Export souls with integrity hashes
    - Import and verify souls
    """

    @staticmethod
    def create_soul(
        anchor: AxiomAnchor | None = None,
    ) -> GenesisSoul:
        """Create a new ``GenesisSoul`` from the current Anchor state."""
        anchor = anchor or AxiomAnchor()
        return GenesisSoul(
            directive=anchor.directive,
            alignment_threshold=anchor.alignment_threshold,
        )

    @staticmethod
    def export_soul(soul: GenesisSoul) -> dict[str, Any]:
        """Export a soul as a signed ``.genesis_soul`` envelope.

        The envelope contains the full soul payload plus a SHA-256
        integrity hash for tamper detection.
        """
        payload = soul.as_dict()
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        integrity_hash = hashlib.sha256(payload_bytes).hexdigest()

        return {
            "genesis_soul": {
                "format": "genesis_soul_v1",
                "integrityHash": integrity_hash,
                "payload": payload,
            }
        }

    @staticmethod
    def export_soul_json(soul: GenesisSoul, indent: int = 2) -> str:
        """Export as a JSON string ready for file writing."""
        return json.dumps(ContinuityBridge.export_soul(soul), indent=indent)

    @staticmethod
    def verify_integrity(envelope: dict[str, Any]) -> bool:
        """Verify the integrity hash of an imported soul envelope."""
        inner = envelope.get("genesis_soul", {})
        stored_hash = inner.get("integrityHash", "")
        payload = inner.get("payload", {})

        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(payload_bytes).hexdigest()

        return computed_hash == stored_hash

    @staticmethod
    def import_soul(envelope: dict[str, Any]) -> GenesisSoul | None:
        """Import a soul from a ``.genesis_soul`` envelope.

        Returns ``None`` if integrity verification fails.
        """
        if not ContinuityBridge.verify_integrity(envelope):
            return None

        payload = envelope["genesis_soul"]["payload"]

        soul = GenesisSoul(
            soul_id=payload.get("soulId", f"soul-{uuid.uuid4().hex[:12]}"),
            version=payload.get("version", "0.1.0"),
            directive=PrimeDirective(
                statement=payload.get("axiomAnchorState", {}).get("directive", {}).get("statement", "Does this serve Love?"),
            ),
            alignment_threshold=payload.get("axiomAnchorState", {}).get("alignmentThreshold", 0.5),
            created_at=payload.get("metadata", {}).get("createdAt", ""),
            updated_at=payload.get("metadata", {}).get("updatedAt", ""),
        )

        # Restore forge artifacts (simple dicts).
        soul.forge_artifacts = payload.get("forgeArtifacts", [])

        # Restore wisdom log with hash chain.
        for w in payload.get("wisdomLog", []):
            soul.wisdom_log.append(WisdomEntry(
                source_text=w.get("sourceText", ""),
                disharmony_summary=w.get("disharmonySummary", ""),
                unity_impact=w.get("unityImpact", 0.0),
                compassion_deficit=w.get("compassionDeficit", 0.0),
                resolution_path=w.get("resolutionPath", "unresolved"),
                resolution_summary=w.get("resolutionSummary", ""),
                covenant_title=w.get("covenantTitle", ""),
                timestamp=w.get("timestamp", ""),
                prev_hash=w.get("prevHash", ""),
                entry_hash=w.get("entryHash", ""),
            ))

        return soul

    @staticmethod
    def verify_wisdom_chain(soul: GenesisSoul) -> tuple[bool, list[str]]:
        """Verify the integrity of the wisdom log hash chain.

        Returns (is_valid, list_of_errors).
        """
        errors: list[str] = []
        prev_hash = ""

        for i, entry in enumerate(soul.wisdom_log):
            # Verify the prev_hash links correctly
            if entry.prev_hash != prev_hash:
                errors.append(
                    f"Entry {i}: prev_hash mismatch. "
                    f"Expected '{prev_hash[:16]}...', got '{entry.prev_hash[:16]}...'"
                )

            # Verify the entry's own hash
            computed = entry.compute_hash(entry.prev_hash)
            if entry.entry_hash and entry.entry_hash != computed:
                errors.append(
                    f"Entry {i}: hash mismatch. "
                    f"Expected '{computed[:16]}...', got '{entry.entry_hash[:16]}...'"
                )

            prev_hash = entry.entry_hash

        return (len(errors) == 0, errors)
