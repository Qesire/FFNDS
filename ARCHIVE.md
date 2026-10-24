# FFNDS archive and evidence preservation

The **live research** entry is [research/ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json), and [AGENTS.md](AGENTS.md) governs research agents. This file indexes older materials, which provide evidence/provenance rather than next actions.

- Full server handoff: [2026-10-20 release](https://github.com/Qesire/FFNDS/releases/tag/handoff-20261020). Original release SHA/protocols are not rewritten. The original A100 configuration has not been verified for H20.
- [Upstream restoration](upstream_audit/README.md) corrects the first ZIP's omission of 15 W4A4 files. Apply the audited 92 changed files/5 patches to the pinned upstream revision rather than stacking the first ZIP patches.
- Historical experiment decisions, registrations and their exact outcome boundaries: [research/CURRENT_DECISION.json](research/CURRENT_DECISION.json), [method triage](research/20261021/method_triage.json), [research/20261023](research/20261023), and [raw reports](reports/).
- Scientific problem revision: [information-first three questions](docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md), [dense-completion goal drift](docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md). The shallow algorithm-first A/B/C status document and machine seed (former `docs/94_...` and `research/20261024/operator_first_autoresearcher_reexploration.json`) were deleted from the working tree because they misdirected current work; their exact version remains retrievable in Git at commit [f6904e8](https://github.com/Qesire/FFNDS/tree/f6904e8190b34b7e38706bc829ed6a59e564e1e4).
- [Historical handoffs archive](archive/20261024/historical_handoffs/README.md) holds 10 dated operational documents (AR2, R15, R17 and C2 handoffs). [Content-addressed manifest](archive/20261024/historical_handoffs/MANIFEST.json) preserves original Git blob SHAs; original docs paths are now short forwarding pages.
- Dated documents with `active`, `mandatory`, `retry`, `run`, `next` or `authorized` language refer to their **historical original context** and do not trigger live experiments. Preserve native results, immutable logs, source bindings, frozen protocols and matched evaluation receipts at their original paths.

The user-requested experiment pause is explicit and remains recorded in [EXPERIMENT_PAUSE.json](research/20261024/EXPERIMENT_PAUSE.json).

## Content lifecycle

| Tier | Meaning | Location and management |
| --- | --- | --- |
| **LIVE** | Default scientific behavior, Q1/Q2/Q3 research lenses, current evidence index | `AGENTS.md`, `research/ACTIVE_RESEARCH.json`, `research/agent_protocol/CURRENT_CONTEXT_PACKET.md`, `research/RESEARCH_NAVIGATION.md` |
| **EVIDENCE** | Actual model/code/protocol measurements and method-specific claims | `research/2026102*/`, `reports/`, `tools/`, `upstream_audit/`; preserve source locators, bytes and frozen outcomes |
| **ARCHIVED** | Dated operational handoffs, obsolete agent to-dos, duplicate interim summaries | `archive/20261024/historical_handoffs/`; redirect former Markdown URLs with small stubs and preserve original blob identities |

An archived task is a **source about what was believed/done**, not an outstanding action. A negative experiment remains citable evidence at its exact scope, and earlier inspiring mathematical constructions can still seed new work. Before relocating more material, check for content-hash bindings, in-repository links and reproduction dependencies. Keep complete historical source available through Git even if a shorter active guide supersedes its advice.
