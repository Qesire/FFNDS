# AutoResearcher → FFNDS source synchronization (v2.4)

**Current verified binding:** [Qesire/AutoResearcher@8b88a52d894568dc303db727decfbf13b5fabf3b](https://github.com/Qesire/AutoResearcher/commit/8b88a52d894568dc303db727decfbf13b5fabf3b) · Search-Augmented Research Discovery v2.4.

FFNDS **pins an immutable upstream commit**, not a floating branch. The single authoritative lockfile is [ACTIVE_SOURCE_MANIFEST.json](ACTIVE_SOURCE_MANIFEST.json), which contains the commit SHA and exact Git blob SHA1 for all copied upstream files. [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) and [DEFAULT_BEHAVIOR_CASES.json](DEFAULT_BEHAVIOR_CASES.json) bind the same source commit. Each local code/Skill/protocol file is a **verbatim** copy at that revision.

## What is shared and what is project-specific

- **Upstream pinned:** three general scientific Skills, the new [web-research-discovery Skill](../../.agents/skills/web-research-discovery/SKILL.md), [WARD protocol](../../core/WEB_RESEARCH_DISCOVERY.md), research harness, expansion, novelty, prompt evaluation/playbook, mathematical protocols and the example evidence context packet.
- **FFNDS-local specialization:** root and scoped AGENTS, Copilot, [FFNDS domain skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md), [deep web search](DEEP_WEB_RESEARCH.md), E0 stage/evidence and experiment-permission records. These remain project-specific and **refer to** the pinned upstream Skills.
- **Historical manifests/old proposals:** `SOURCE_MANIFEST.json`, `PRD_SOURCE_MANIFEST.json`, old v2.3 notes and completed experiments remain historical evidence, not the active lock.

## Reproducible future version update

1. Confirm the AutoResearcher `main` commit and ensure its validation/behavior-case checks passed. Read the current changelog/diff, including affected Skills, prompts and schemas.
2. Open a serial FFNDS sync branch. Update copied upstream files **verbatim**, using source commit content; add any newly shared Skills/core protocols.
3. Replace `ACTIVE_SOURCE_MANIFEST.json.source_commit` and every `git_blob_sha1` entry with the verified **upstream tree blob hash** at that exact commit. Update `ACTIVE_RESEARCH.autoResearcher.source_revision`, `harness_url`, skills, default methodology source revision and `DEFAULT_BEHAVIOR_CASES.source_revision`.
4. For a search method update, retain FFNDS-specific `DEEP_WEB_RESEARCH.md` as specialization; align its references and prompts. Preserve stage E0 and latest experimental authorization without historical rewriting.
5. Run the pure-static `python tools/check_autoresearcher_binding.py` through the manual research validation workflow, or reproduce Git blob audits; inspect actual research trajectories separately. Compare the entire pinned source list against upstream, review diff and merge only after synchronization is complete.

**Meaning of “stay synchronized”:** FFNDS is currently synchronized **exactly to the recorded pinned AutoResearcher revision**; automatic adoption of future upstream pushes is intentionally avoided for experimental provenance. A later upstream change should trigger a reviewed repeat of this procedure. The static checker guards byte-identical pinned files and source revision consistency; it cannot know a newer upstream commit exists without an explicit remote freshness query.

**Current experiment authority:** The registered E0 finite-action observations have completed. Syncing the research methodology does not authorize a new CPU/GPU/model run or alter the FFNDS science stage.
