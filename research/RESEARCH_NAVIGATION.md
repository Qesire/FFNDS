# FFNDS research navigation

This page is **navigation**, not another task schedule. Cleanup provenance and verified static findings are recorded in [2026-10-24 cleanup audit](20261024/RESEARCH_CLEANUP_AUDIT.json). A new research turn begins with [AGENTS](../AGENTS.md), [one active research file](ACTIVE_RESEARCH.json) and [current short evidence packet](agent_protocol/CURRENT_CONTEXT_PACKET.md). Use the local AutoResearcher [harness](../core/AGENT_HARNESS.md) to choose the relevant mechanism, expansion and/or problem-reformulation methods for the current scientific fork. Retrieve additional historical evidence when it changes a decision.

| Area | Read when | Canonical reference |
| --- | --- | --- |
| Current science | Defining the open question, hypotheses or next decision | [Three questions and mechanism analysis](../docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md) |
| Wan source | Asserting a real operator/tensor/quantizer/condition | [C0 native source bindings](20261023/C0_coupling_offline_results.json), [reconstructed runtime](20261022/R12_E0_runtime_reconstruction.json), `tools/packed_ffn_reference.py` |
| Conditional information | Investigating pre-LN/attention state and downstream values | [C0 offline result](20261023/C0_coupling_offline_results.json), [C0 next discriminator (historical)](20261023/C0_next_discriminator.json) |
| Existing method outcomes | Seeing exactly what failed or passed | [Method triage](20261021/method_triage.json) and [frozen decision](CURRENT_DECISION.json) |
| Full-video/scientific metrics | Interpreting terminal quality and true execution time | [Metrics](../docs/77_FFNDS学术指标族与评测协议_20261023.md), [candidate video smoke](candidate_rollout/README.md) |
| AutoResearcher implementation | Checking default scientific behavior and source provenance | [Active upstream SHA manifest](agent_protocol/ACTIVE_SOURCE_MANIFEST.json), [behavior cases](agent_protocol/DEFAULT_BEHAVIOR_CASES.json) |
| Past tasks/proposals | Looking up historical hypotheses, old authorizations or failures | `research/20261021/`, `research/20261022/`, `research/20261023/`, `research/20261024/`, [archive guide](../ARCHIVE.md) |

**Task status precedence:** `research/ACTIVE_RESEARCH.json` → root `AGENTS.md` → project `research/OBJECTIVE_CONTRACT.json` for scientific goals; `research/20261024/EXPERIMENT_PAUSE.json` and the existing eligibility ledger govern whether experiments may run. Every dated `ACTIVE`, `RUN`, `RETRY`, `NEXT` in archived files applies only to its original timestamp and scope. Preserve original result and protocol bytes.

No experiments or model execution were started by this navigation update.
