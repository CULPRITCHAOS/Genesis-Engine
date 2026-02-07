"""
Module 2.3 Extension — The Obsidian Exporter

Exports a ``.genesis_soul`` into a linked Markdown vault compatible with
`Obsidian <https://obsidian.md>`_.

Vault Structure
---------------
::

    vault_root/
    ├── Manifesto.md            # Frontmatter with alignment scores + hash
    ├── Insights/
    │   ├── wisdom_001.md       # One file per wisdom log entry
    │   └── ...
    ├── Projections/
    │   ├── foresight_001.md    # One file per foresight projection
    │   └── ...
    ├── Overrides/
    │   ├── override_001.md     # One file per human override
    │   └── ...
    └── Graphs/
        ├── graph_001.md        # One file per graph history entry
        └── ...

Features:
- **Bidirectional Linking** — every note links to related notes using
  ``[[wikilink]]`` syntax.
- **Manifesto Frontmatter** — YAML frontmatter on the index note with
  alignment scores, soul ID, and SHA-256 integrity hash.
- **Graph Visualisation** — Mermaid diagrams embedded in Graph notes.
- **Compassion-Driven Narrative** — each note carries a human-readable
  interpretation that foregrounds Unity and stewardship.

Integration:
- Reads from ``GenesisSoul`` (via Continuity Bridge)
- Optionally includes ``StewardshipManifesto`` from the Forge
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis_engine.core.continuity_bridge import (
    ContinuityBridge,
    GenesisSoul,
    WisdomEntry,
    ForesightProjection,
    HumanOverrideEntry,
    GraphHistoryEntry,
)


# ---------------------------------------------------------------------------
# Export Result
# ---------------------------------------------------------------------------

@dataclass
class ObsidianVault:
    """In-memory representation of an exported Obsidian vault.

    Files are stored as (relative_path, content) pairs so the vault
    can be written to disk or inspected programmatically.
    """

    soul_id: str
    files: dict[str, str] = field(default_factory=dict)  # path → content
    integrity_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def file_count(self) -> int:
        return len(self.files)

    def write_to_disk(self, root: Path) -> list[Path]:
        """Write all vault files to *root*, creating directories as needed.

        Returns a list of all written file paths.
        """
        written: list[Path] = []
        for rel_path, content in sorted(self.files.items()):
            full_path = root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            written.append(full_path)
        return written

    def as_dict(self) -> dict[str, Any]:
        return {
            "obsidianVault": {
                "soulId": self.soul_id,
                "fileCount": self.file_count,
                "files": list(self.files.keys()),
                "integrityHash": self.integrity_hash,
                "timestamp": self.timestamp,
            }
        }


# ---------------------------------------------------------------------------
# Obsidian Exporter
# ---------------------------------------------------------------------------

class ObsidianExporter:
    """Exports a ``GenesisSoul`` into an Obsidian-compatible Markdown vault.

    The vault is structured as four directories (Insights, Projections,
    Overrides, Graphs) with a root Manifesto index file.  All notes use
    bidirectional ``[[wikilink]]`` syntax for cross-referencing.

    Parameters
    ----------
    include_mermaid : bool
        Whether to include Mermaid graph diagrams in Graph notes.
    """

    def __init__(self, include_mermaid: bool = True) -> None:
        self._include_mermaid = include_mermaid

    # -- public API ---------------------------------------------------------

    def export(
        self,
        soul: GenesisSoul,
        manifesto_dict: dict[str, Any] | None = None,
    ) -> ObsidianVault:
        """Export a soul into an ``ObsidianVault``.

        Parameters
        ----------
        soul : GenesisSoul
            The soul to export.
        manifesto_dict : dict | None
            Optional Stewardship Manifesto data to include in frontmatter.
        """
        vault = ObsidianVault(soul_id=soul.soul_id)

        # 1. Generate Manifesto (index file)
        vault.files["Manifesto.md"] = self._render_manifesto(
            soul, manifesto_dict,
        )

        # 2. Generate Insight notes (from wisdom log)
        for i, entry in enumerate(soul.wisdom_log, 1):
            filename = f"Insights/wisdom_{i:03d}.md"
            vault.files[filename] = self._render_wisdom(entry, i, soul)

        # 3. Generate Projection notes (from foresight projections)
        proj_idx = 0
        for w_idx, entry in enumerate(soul.wisdom_log, 1):
            for fp in entry.foresight_projections:
                proj_idx += 1
                filename = f"Projections/foresight_{proj_idx:03d}.md"
                vault.files[filename] = self._render_foresight(
                    fp, proj_idx, w_idx,
                )

        # 4. Generate Override notes
        for i, override in enumerate(soul.human_overrides, 1):
            filename = f"Overrides/override_{i:03d}.md"
            vault.files[filename] = self._render_override(override, i, soul)

        # 5. Generate Graph notes
        for i, graph_entry in enumerate(soul.graph_history, 1):
            filename = f"Graphs/graph_{i:03d}.md"
            vault.files[filename] = self._render_graph(graph_entry, i)

        # 6. Compute vault integrity hash
        vault.integrity_hash = self._compute_vault_hash(vault)

        return vault

    # -- rendering ----------------------------------------------------------

    def _render_manifesto(
        self,
        soul: GenesisSoul,
        manifesto_dict: dict[str, Any] | None,
    ) -> str:
        """Render the root Manifesto.md with YAML frontmatter."""
        # Compute alignment scores from the manifesto or derive from soul
        alignment_scores: dict[str, float] = {}
        if manifesto_dict:
            sm = manifesto_dict.get("stewardshipManifesto", manifesto_dict)
            alignment_scores = sm.get("alignmentScores", {})

        # Compute integrity hash for frontmatter
        soul_envelope = ContinuityBridge.export_soul(soul)
        soul_hash = soul_envelope["genesis_soul"]["integrityHash"]

        lines = ["---"]
        lines.append(f"soul_id: \"{soul.soul_id}\"")
        lines.append(f"version: \"{soul.version}\"")
        lines.append(f"prime_directive: \"{soul.directive.statement}\"")
        lines.append(f"integrity_hash: \"{soul_hash}\"")

        if alignment_scores:
            lines.append("alignment_scores:")
            for k, v in alignment_scores.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        lines.append(f"wisdom_count: {len(soul.wisdom_log)}")
        lines.append(f"override_count: {len(soul.human_overrides)}")
        lines.append(f"graph_count: {len(soul.graph_history)}")

        foresight_count = sum(
            len(w.foresight_projections) for w in soul.wisdom_log
        )
        lines.append(f"foresight_count: {foresight_count}")
        lines.append(f"created_at: \"{soul.created_at}\"")
        lines.append(f"updated_at: \"{soul.updated_at}\"")
        lines.append("---")
        lines.append("")

        # Markdown body
        lines.append("# Genesis Soul Manifesto")
        lines.append("")
        lines.append(f"> *\"{soul.directive.statement}\"*")
        lines.append("")
        lines.append(f"**Soul ID**: `{soul.soul_id}`")
        lines.append(f"**Integrity Hash**: `{soul_hash[:32]}...`")
        lines.append("")

        # Sections with wikilinks
        lines.append("## Insights")
        lines.append("")
        for i, entry in enumerate(soul.wisdom_log, 1):
            title = entry.covenant_title or entry.disharmony_summary[:60]
            lines.append(f"- [[wisdom_{i:03d}|{title}]]")
        lines.append("")

        lines.append("## Projections")
        lines.append("")
        proj_idx = 0
        for entry in soul.wisdom_log:
            for fp in entry.foresight_projections:
                proj_idx += 1
                lines.append(
                    f"- [[foresight_{proj_idx:03d}|"
                    f"{fp.outcome_flag} ({fp.war_game_rounds} rounds)]]"
                )
        lines.append("")

        lines.append("## Overrides")
        lines.append("")
        for i, override in enumerate(soul.human_overrides, 1):
            lines.append(
                f"- [[override_{i:03d}|"
                f"{override.reason_category} (confidence: {override.confidence})]]"
            )
        lines.append("")

        lines.append("## Graphs")
        lines.append("")
        for i, graph_entry in enumerate(soul.graph_history, 1):
            lines.append(
                f"- [[graph_{i:03d}|{graph_entry.phase}: {graph_entry.label or 'Graph'}]]"
            )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_wisdom(
        entry: WisdomEntry,
        index: int,
        soul: GenesisSoul,
    ) -> str:
        """Render a single wisdom log entry as a Markdown note."""
        title = entry.covenant_title or f"Insight #{index}"
        lines = ["---"]
        lines.append(f"title: \"{title}\"")
        lines.append(f"resolution_path: \"{entry.resolution_path}\"")
        lines.append(f"unity_impact: {entry.unity_impact}")
        lines.append(f"compassion_deficit: {entry.compassion_deficit}")
        lines.append(f"entry_hash: \"{entry.entry_hash[:32]}...\"")
        lines.append(f"timestamp: \"{entry.timestamp}\"")
        lines.append("---")
        lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Resolution Path**: {entry.resolution_path}")
        lines.append(f"**Unity Impact**: {entry.unity_impact:.2f}")
        lines.append(f"**Compassion Deficit**: {entry.compassion_deficit:.2f}")
        lines.append("")

        lines.append("## Disharmony")
        lines.append("")
        lines.append(entry.disharmony_summary)
        lines.append("")

        if entry.resolution_summary:
            lines.append("## Resolution")
            lines.append("")
            lines.append(entry.resolution_summary)
            lines.append("")

        lines.append("## Source")
        lines.append("")
        lines.append(f"> {entry.source_text}")
        lines.append("")

        # Bidirectional links
        lines.append("## Links")
        lines.append("")
        lines.append("- [[Manifesto]]")

        # Link to foresight projections
        proj_offset = 0
        for w_idx, w in enumerate(soul.wisdom_log, 1):
            if w_idx < index:
                proj_offset += len(w.foresight_projections)
            elif w_idx == index:
                for fp_idx in range(len(w.foresight_projections)):
                    lines.append(f"- [[foresight_{proj_offset + fp_idx + 1:03d}]]")
                break

        if index > 1:
            lines.append(f"- [[wisdom_{index - 1:03d}|Previous Insight]]")
        if index < len(soul.wisdom_log):
            lines.append(f"- [[wisdom_{index + 1:03d}|Next Insight]]")
        lines.append("")

        # Hash chain integrity marker
        lines.append("## Integrity")
        lines.append("")
        lines.append(f"- **Entry Hash**: `{entry.entry_hash}`")
        lines.append(f"- **Previous Hash**: `{entry.prev_hash or 'GENESIS'}`")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_foresight(
        fp: ForesightProjection,
        index: int,
        wisdom_index: int,
    ) -> str:
        """Render a foresight projection as a Markdown note."""
        title = f"Foresight: {fp.outcome_flag} ({fp.war_game_rounds} rounds)"
        lines = ["---"]
        lines.append(f"title: \"{title}\"")
        lines.append(f"outcome: \"{fp.outcome_flag}\"")
        lines.append(f"rounds: {fp.war_game_rounds}")
        lines.append(f"sustainability_score: {fp.sustainability_score}")
        lines.append(f"timestamp: \"{fp.timestamp}\"")
        lines.append("---")
        lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Aligned Score | {fp.aligned_score:.2f} |")
        lines.append(f"| Extractive Score | {fp.extractive_score:.2f} |")
        lines.append(f"| Sustainability | {fp.sustainability_score:.4f} |")
        lines.append(f"| Aligned Cooperation | {fp.aligned_cooperation_rate:.2%} |")
        lines.append(f"| Extractive Cooperation | {fp.extractive_cooperation_rate:.2%} |")
        lines.append(f"| Outcome | **{fp.outcome_flag}** |")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(fp.foresight_summary)
        lines.append("")

        # Bidirectional links
        lines.append("## Links")
        lines.append("")
        lines.append("- [[Manifesto]]")
        lines.append(f"- [[wisdom_{wisdom_index:03d}|Source Insight]]")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_override(
        override: HumanOverrideEntry,
        index: int,
        soul: GenesisSoul,
    ) -> str:
        """Render a human override entry as a Markdown note."""
        title = f"Override #{index}: {override.reason_category}"
        lines = ["---"]
        lines.append(f"title: \"{title}\"")
        lines.append(f"category: \"{override.reason_category}\"")
        lines.append(f"confidence: {override.confidence}")
        lines.append(f"system_score: {override.system_recommended_score}")
        lines.append(f"human_score: {override.human_selected_score}")
        lines.append(f"timestamp: \"{override.timestamp}\"")
        lines.append("---")
        lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Category**: `{override.reason_category}`")
        lines.append(f"**Confidence**: {override.confidence}/10")
        lines.append("")

        lines.append("## Divergence")
        lines.append("")
        lines.append(
            f"| | ID | Score | Path |"
        )
        lines.append(f"|---|---|---|---|")
        lines.append(
            f"| System Recommended | `{override.system_recommended_id}` | "
            f"{override.system_recommended_score:.4f} | "
            f"{override.system_recommended_path} |"
        )
        lines.append(
            f"| Human Selected | `{override.human_selected_id}` | "
            f"{override.human_selected_score:.4f} | "
            f"{override.human_selected_path} |"
        )
        lines.append("")

        lines.append("## Reason")
        lines.append("")
        lines.append(override.divergence_reason)
        lines.append("")

        if override.problem_text:
            lines.append("## Context")
            lines.append("")
            lines.append(f"> {override.problem_text}")
            lines.append("")

        # Bidirectional links
        lines.append("## Links")
        lines.append("")
        lines.append("- [[Manifesto]]")
        if index > 1:
            lines.append(f"- [[override_{index - 1:03d}|Previous Override]]")
        if index < len(soul.human_overrides):
            lines.append(f"- [[override_{index + 1:03d}|Next Override]]")
        lines.append("")

        return "\n".join(lines)

    def _render_graph(
        self,
        entry: GraphHistoryEntry,
        index: int,
    ) -> str:
        """Render a graph history entry as a Markdown note with optional Mermaid."""
        title = f"Graph #{index}: {entry.phase}"
        if entry.label:
            title += f" — {entry.label}"

        lines = ["---"]
        lines.append(f"title: \"{title}\"")
        lines.append(f"phase: \"{entry.phase}\"")
        lines.append(f"objects: {len(entry.graph.objects)}")
        lines.append(f"morphisms: {len(entry.graph.morphisms)}")
        lines.append(f"timestamp: \"{entry.timestamp}\"")
        lines.append("---")
        lines.append("")
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Phase**: {entry.phase}")
        lines.append(f"**Objects**: {len(entry.graph.objects)}")
        lines.append(f"**Morphisms**: {len(entry.graph.morphisms)}")
        lines.append("")

        if entry.graph.source_text:
            lines.append("## Source")
            lines.append("")
            lines.append(f"> {entry.graph.source_text}")
            lines.append("")

        # Objects table
        lines.append("## Objects")
        lines.append("")
        lines.append("| Label | Tags |")
        lines.append("|-------|------|")
        for obj in entry.graph.objects:
            lines.append(f"| {obj.label} | `{', '.join(obj.tags)}` |")
        lines.append("")

        # Morphisms table
        lines.append("## Morphisms")
        lines.append("")
        lines.append("| Label | Source → Target | Tags |")
        lines.append("|-------|-----------------|------|")
        for morph in entry.graph.morphisms:
            # Resolve labels
            src_label = morph.source
            tgt_label = morph.target
            for obj in entry.graph.objects:
                if obj.id == morph.source:
                    src_label = obj.label
                if obj.id == morph.target:
                    tgt_label = obj.label
            lines.append(
                f"| {morph.label} | {src_label} → {tgt_label} | "
                f"`{', '.join(morph.tags)}` |"
            )
        lines.append("")

        # Mermaid diagram
        if self._include_mermaid and entry.graph.objects:
            lines.append("## Diagram")
            lines.append("")
            lines.append("```mermaid")
            lines.append("graph LR")
            # Map IDs to short labels for Mermaid
            id_map: dict[str, str] = {}
            for obj in entry.graph.objects:
                safe_label = obj.label.replace(" ", "_")
                id_map[obj.id] = safe_label
                lines.append(f"    {safe_label}[\"{obj.label}\"]")
            for morph in entry.graph.morphisms:
                src = id_map.get(morph.source, morph.source)
                tgt = id_map.get(morph.target, morph.target)
                lines.append(f"    {src} -->|{morph.label}| {tgt}")
            lines.append("```")
            lines.append("")

        # Bidirectional links
        lines.append("## Links")
        lines.append("")
        lines.append("- [[Manifesto]]")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _compute_vault_hash(vault: ObsidianVault) -> str:
        """Compute a SHA-256 hash over all vault file contents."""
        hasher = hashlib.sha256()
        for path in sorted(vault.files.keys()):
            hasher.update(path.encode("utf-8"))
            hasher.update(vault.files[path].encode("utf-8"))
        return hasher.hexdigest()
