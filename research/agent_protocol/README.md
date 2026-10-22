# AutoResearcher v2.3 in FFNDS

Pinned source: [Qesire/AutoResearcher@d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818](https://github.com/Qesire/AutoResearcher/tree/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818).

Read the [local complete AutoResearcher AGENTS](AUTORESEARCHER_AGENTS_v2_3.md), repository-root `core/` protocols, and `.agents/skills/mechanism-research/SKILL.md` plus `.agents/skills/research-expansion/SKILL.md`. The entry point for all FFNDS agents is [root AGENTS.md](../../AGENTS.md). The local [ffnds-paper-discovery skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md) implements this approach for the post-training dynamic FFN paper.

Agent behavior: lead with a scientific insight, generate distinguishable mechanisms, reconstruct only decision-critical real source, derive equations, actively execute cheap checks, then register strictly controlled model trials where appropriate. Evidence and experimental gates are strict when claiming results; ideas are not refused for lack of a completed implementation. See [current R12 handoff](../20261022/AGENT_EXECUTION_HANDOFF.md).
