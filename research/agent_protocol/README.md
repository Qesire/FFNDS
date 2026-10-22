# AutoResearcher v2.3 in FFNDS

Pinned source: [Qesire/AutoResearcher@d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818](https://github.com/Qesire/AutoResearcher/tree/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818).

Read the [local complete AutoResearcher AGENTS](AUTORESEARCHER_AGENTS_v2_3.md), repository-root `core/` protocols, and `.agents/skills/mechanism-research/SKILL.md` plus `.agents/skills/research-expansion/SKILL.md`. The entry point for all FFNDS agents is [root AGENTS.md](../../AGENTS.md). The local [ffnds-paper-discovery skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md) implements this approach for the post-training dynamic FFN paper.

Agent behavior: lead with a scientific insight, generate distinguishable mechanisms, reconstruct only decision-critical real source, derive equations, actively execute cheap checks, then register strictly controlled model trials where appropriate. Evidence and experimental gates are strict when claiming results; ideas are not refused for lack of a completed implementation. See [current R12 handoff](../20261022/AGENT_EXECUTION_HANDOFF.md).

## Exploration-library integration

Agents should open [research exploration index](../exploration/20261022/EXPLORATION_INDEX.md) and [new synthesis](../../docs/67_后训练动态FFN开放探索与质量速度目标_20261022.md) as a **menu of evidence and falsifiable seeds**, not an experiment whitelist. This expands on the historical R12 single-pair candidate. See the current root AGENTS.md before selecting an experiment; avoid blocking genuinely new methods because a different old registered recipe failed.

## Updated candidate evaluation (2026-10-22)

The user clarified that earlier 50% MAC, 1.5× full-FFN speed and 2% per-axis quality values were arbitrary illustrative targets. Read [matched-competitor quality-speed guidance](../../docs/72_与同前提方法竞争的质量速度评价准则_20261022.md) before choosing new work. Do not make global stopping rules out of the former values; do preserve every historical frozen registered protocol/outcome. Scientific performance claims require actually comparable methods/conditions and full native latency + free-video quality, not MAC arithmetic or local flow alone.
