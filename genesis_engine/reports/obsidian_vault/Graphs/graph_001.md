---
title: "Graph #1: translation — Extractive corporate graph"
phase: "translation"
objects: 5
morphisms: 4
timestamp: "2026-02-07T14:21:35.809532+00:00"
---

# Graph #1: translation — Extractive corporate graph

**Phase**: translation
**Objects**: 5
**Morphisms**: 4

## Source

> A corporation that prioritizes shareholder value over employee welfare and environmental sustainability.

## Objects

| Label | Tags |
|-------|------|
| Corporation | `stakeholder, actor, shareholder_primacy_risk` |
| Shareholder | `stakeholder, sink` |
| Employee | `stakeholder, vulnerable` |
| Future_Generations | `stakeholder, vulnerable, shadow_entity, temporal` |
| Ecosystem | `stakeholder, vulnerable, shadow_entity, ecological` |

## Morphisms

| Label | Source → Target | Tags |
|-------|-----------------|------|
| Prioritization | Corporation → Employee | `decision` |
| Sustainability | Corporation → Employee | `protection` |
| Maximize_Value | Corporation → Employee | `maximize_value, fiduciary_duty` |
| Temporal_Impact | Corporation → Future_Generations | `neglect, temporal_harm, shadow_impact` |

## Diagram

```mermaid
graph LR
    Corporation["Corporation"]
    Shareholder["Shareholder"]
    Employee["Employee"]
    Future_Generations["Future_Generations"]
    Ecosystem["Ecosystem"]
    Corporation -->|Prioritization| Employee
    Corporation -->|Sustainability| Employee
    Corporation -->|Maximize_Value| Employee
    Corporation -->|Temporal_Impact| Future_Generations
```

## Links

- [[Manifesto]]
