# FFNDS research navigation

**2026-10-27 actual completion / pause:** [four new sources, six seed paths, twelve videos](20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md), [six exposed-source native comparison](20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md), [conditional mean information](20261026/E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.md). Ten source identities/twelve nested source×seed trajectories/1,920sites now exist. Formal quality NOT_RUN/false. User requested completion, GitHub upload, then pause; model/GPU jobs are complete and released, new execution is paused in ACTIVE.

**研读与写作原则：** 优先读[当前研究任务](20261025/E0_FFN_FIRST_TASK_20261025.md)，独立构造新的机制与可区分预测；仅在论证涉及既有实验时打开精确的历史证据。研究结论保留负结果的准确适用范围，活跃Agent指令采用[生成性表达准则](../AGENTS.md)。静态检查和手动CI用于技术一致性，科学判断来自数学、源码和原生实测。

**最新原生与函数研究（2026-10-26）**：[v3两来源320-site配对](20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md) → [完整dense有限反事实（已完成）](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) → [末步Dense视频桥（已完成）](20261026/E0_SAVED_FLOW_DECODE_RESULTS.md)。新的[四坐标固定函数oracle](20261026/E0_FIXED_COORDINATE_BASIS_ORACLE.md)检验128个已暴露行的条件非线性能力，下一应区分anchor自身与neighbor12的可迁移函数信息；[SM120 packed FP4原生基元](20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)属于独立硬件能力（没有真实Wan FFN替换）。[最新机制审查与引导](20261026/E0_NEW_WORK_MECHANISM_AUDIT_20261026.md)按功能发现→不同预测→最小判别→实际工作→整段视频划分科研优先级。历史[v2 DD故障](20261026/E0_PAIRED_V2_FAILURE_AND_RECOVERY.md)只支持其实际故障范围；正式质量仍未验收。

**机制优先的最新读法：** [2025–2026文献与机制纲领](20261026/E0_FRONTIER_MECHANISM_FIRST_20261026.md)与[新工作纠偏](20261026/E0_NEW_WORK_MECHANISM_AUDIT_20261026.md)为讨论入口；source数量、SSE、函数特征基和NVFP4只是识别条件结构后的不同证据/实现对象。

**当前统一科学解释**：[完整视频、Affine54家族与开放机制](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)；**唯一运行状态与权限**：[ACTIVE_RESEARCH.json](ACTIVE_RESEARCH.json)；**一页研究任务**：[E0第一任务](20261025/E0_FFN_FIRST_TASK_20261025.md)。旧科学审查已合并，2026-10-25的方向文件仅提供当时的原始观察和机制历史；不应被当成当前事实。

**Current canonical research program (2026-10-25):** [Intrinsic FFN science reset](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md), [first question](20261025/E0_FFN_FIRST_TASK_20261025.md), [FFN-focused prior art](20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md), [knowledge gaps and discriminators](20261025/E0_FFN_KNOWLEDGE_GAPS_AND_DISCRIMINATORS_20261025.md). [Newest divergence audit](20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md) translates completed full-hidden/contiguous token and original HiF4 group-closure findings into competing mechanism studies. The old quantization-first E0 plans are history; source-specific numeric results and peer reviews remain evidence.

This page is **navigation**, not another task schedule. Cleanup provenance and verified static findings are recorded in [2026-10-24 cleanup audit](20261024/RESEARCH_CLEANUP_AUDIT.json). A new research turn begins with [AGENTS](../AGENTS.md), [one active research file](ACTIVE_RESEARCH.json) and [current short evidence packet](agent_protocol/CURRENT_CONTEXT_PACKET.md). Use the local AutoResearcher [harness](../core/AGENT_HARNESS.md) to choose the relevant mechanism, expansion and/or problem-reformulation methods for the current scientific fork. Retrieve additional historical evidence when it changes a decision.

| Area | Read when | Canonical reference |
| --- | --- | --- |
| Current science | Interpreting verified full-depth observations and open causal questions | [E0 FFN reset](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md), [first task](20261025/E0_FFN_FIRST_TASK_20261025.md); [historic Q1/Q2/Q3 framework](../docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md) |
| Agent–computation-graph collaboration | Node-based mechanism research, independent evidence annotation and return of verified explanations to source graph | [Graph↔Agent protocol](agent_protocol/GRAPH_AGENT_COLLABORATION.md), [node/edge bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json), [graph-anchored E0 discoveries](method_discussion/graph_annotations/E0/README.md) |
| Computation graph & finding taxonomy | Reconstructing real FFN execution from source or tracing evidence to a tensor/edge | [Detailed G0–G5 graph](../docs/98_端到端计算图与证据注解_20261025.md), [mechanism-classified findings atlas](EXPERIMENT_FINDINGS_ATLAS.md); consult original RUN JSON before asserting numeric claims |
| Wan source | Asserting a real operator/tensor/quantizer/condition | [C0 native source bindings](20261023/C0_coupling_offline_results.json), [reconstructed runtime](20261022/R12_E0_runtime_reconstruction.json), `tools/packed_ffn_reference.py` |
| Conditional information | Investigating pre-LN/attention state and downstream values | [C0 offline result](20261023/C0_coupling_offline_results.json), [C0 next discriminator (historical)](20261023/C0_next_discriminator.json) |
| Ranked experimental evidence | Selecting evidence by current scientific importance and maturity | [Readable priority guide](EXPERIMENT_EVIDENCE_INDEX.md), [ranked evidence with source SHA](experiment_evidence/PRIORITY_RANKING.json), [complete experiment file inventory](experiment_evidence/FILE_INVENTORY.json) |
| Existing method outcomes | Seeing exactly what failed or passed | [Method triage](20261021/method_triage.json) and [frozen decision](CURRENT_DECISION.json) |
| Full-video/scientific metrics | Interpreting terminal quality and true execution time | [Metrics](../docs/77_FFNDS学术指标族与评测协议_20261023.md), [candidate video smoke](candidate_rollout/README.md) |
| AutoResearcher implementation | Checking default scientific behavior and source provenance | [Active upstream SHA manifest](agent_protocol/ACTIVE_SOURCE_MANIFEST.json), [behavior cases](agent_protocol/DEFAULT_BEHAVIOR_CASES.json) |
| Past tasks/proposals | Looking up historical hypotheses, old authorizations or failures | [Dated handoff archive](../archive/20261024/historical_handoffs/README.md), `research/20261021/`–`20261024/`, [archive policy](../ARCHIVE.md) |

**Task status precedence:** `research/ACTIVE_RESEARCH.json` → root `AGENTS.md` → project `research/OBJECTIVE_CONTRACT.json` for scientific goals; `research/20261024/EXPERIMENT_PAUSE.json` and the existing eligibility ledger govern whether experiments may run. Every dated `ACTIVE`, `RUN`, `RETRY`, `NEXT` in archived files applies only to its original timestamp and scope. Preserve original result and protocol bytes.

No experiments or model execution were started by this navigation update.

## Discussion-only collaboration

**All discussion agents automatically follow pinned AutoResearcher principles** through [discussion AGENTS.md](method_discussion/AGENTS.md), including proposers, external reviewers and consensus integrators. Use original Wan/source/math/competing-mechanism reasoning to substantiate support and criticism. It is a method of inquiry, not a requirement for every new seed to have an implementation or test result.

For multiple scientific agents, use [the independent method discussion queue](method_discussion/QUEUE.md), [governance](method_discussion/README.md) and [templates](method_discussion/TEMPLATES.md). It is a parallel discussion ledger, **not another active research source, experiment scheduler or approval gate for creativity**. Peer-supported entries reference real external review files; previously authored theory seeded into the queue is explicitly unreviewed until a different agent reviews it. Root AGENTS.md governs behavior; permitted experimental resource use is determined by the live ACTIVE_RESEARCH.json authorization and its bounded run records.
