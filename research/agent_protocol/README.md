# FFNDS · AutoResearcher source and task routing

The default agent entry is [root AGENTS.md](../../AGENTS.md); the **only live research state** is [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json). For a small decision-relevant evidence bundle, read [CURRENT_CONTEXT_PACKET.md](CURRENT_CONTEXT_PACKET.md). The research directory [navigation](../RESEARCH_NAVIGATION.md) separates latest science from old studies.

The project uses pinned [AutoResearcher current source](ACTIVE_SOURCE_MANIFEST.json) (`5bd7ec53`) with upstream core files and three specialist skills kept **verbatim**. FFNDS-specific application lives in root `AGENTS.md` and `.agents/skills/ffnds-paper-discovery/SKILL.md`; generic AutoResearcher source files should not contain appended FFNDS policies. Apply the mechanism/reformulation/expansion methods that materially advance the question; combine them when a mathematical reformulation and source reconstruction need each other.

Earlier `SOURCE_MANIFEST.json`, `PRD_SOURCE_MANIFEST.json` and `AUTORESEARCHER_AGENTS_v2_3.md` are **archived provenance only**, pinned to their historical revisions. Similarly, old 2026-10-23 `COUPLING_PIVOT_TASK.json` may still say `ACTIVE_DISCOVERY_NOT_REGISTERED` in its original source; it has **zero current task authority**. Do not rewrite its original evidence to hide what occurred.

The content-addressed check `tools/check_autoresearcher_binding.py` confirms local version integrity and active-route agreement. It does not measure scientific creativity; [behavior cases](DEFAULT_BEHAVIOR_CASES.json) supply the separate evaluation corpus for authorized future agent comparisons.

All CPU/GPU/model experiments and automatic followups remain paused per [explicit receipt](../20261024/EXPERIMENT_PAUSE.json).

Superseded operational summaries have moved to the [dated handoff archive](../../archive/20261024/historical_handoffs/README.md). Reproduce the original wording with the archived Git blobs or original source commits; dated execution instructions are historical, not a reason to launch an experiment.
