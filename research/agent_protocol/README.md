# FFNDS · AutoResearcher source and task routing

**Latest upstream synchronization:** AutoResearcher v2.4 WARD is pinned to [ACTIVE_SOURCE_MANIFEST.json](ACTIVE_SOURCE_MANIFEST.json). Read [generic research protocol](../../core/WEB_RESEARCH_DISCOVERY.md), [dedicated Skill](../../.agents/skills/web-research-discovery/SKILL.md) and [FFNDS-specific deep Web Search](DEEP_WEB_RESEARCH.md); see [sync procedure](UPSTREAM_SYNC.md) for reproducible future upgrades.


**表达与验证：** 科研结果首先呈现新结构、数学机制与竞争预测。历史失败使用原作用域陈述；授权和质量按实际行动说明。按照[根指令](../../AGENTS.md)的生成性表达准则审阅当前引导，按风险选择本地验证，CI保持手动按需。

## Graph-Agent协作入口

[双向协议](GRAPH_AGENT_COLLABORATION.md) · [G0–G6节点索引](GRAPH_AGENT_BINDINGS.json) · [计算图原文](../../docs/98_端到端计算图与证据注解_20261025.md) · [E0独立图注解](../method_discussion/graph_annotations/E0/README.md)。相关科研 Agent 从真实计算节点检索来源/证据，独立提出与审查机制，再将确定范围的发现反向更新图与实验谱系。

## 当前科研入口：FFN 本征结构

请从 [E0 FFN 本征研究重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) 和 [唯一活跃第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) 开始；真实状态与实验授权见 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)。探索 FFN hidden/GELU/W2 有符号贡献、时空条件创新、下游视频敏感性及计算重构。此前的 [E0 gap 总表](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md) 和相关 BF16/HiF4 径向观察已经完成其特定研究阶段，继续用作证据，不再决定下一次 Agent 的首要任务。

跨领域原始文献检索始终依照 [DEEP_WEB_RESEARCH.md](DEEP_WEB_RESEARCH.md)，量化元信息一般作为低成本幅值线索；后续如发现真正新的量化结构可建立独立问题。历史E0运行范围已完成；另外登记的paired-native v2权限、当前GPU作业及其执行结果必须实时读取[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)和原生RUN。讨论规则不会自动授予任何额外任务。

## 深度 Web Search 是默认研究能力

科研发现、跨 Agent 方法评议与查重应当主动进行**广泛、深入、迭代式联网检索**，而非依赖固定知识或一轮关键词搜索；统一遵循 [深度文献与源码搜索协议](DEEP_WEB_RESEARCH.md)。从经典和最新论文、相邻机制、前后向引用链、官方模型/硬件源码、失败与反例多方向追踪，再把可核验结果回到真实 FFN 计算流。E0 的结论仍是科学现象与竞争解释，不自动变成新算法。对无法访问互联网的 Agent，应在结果中明确检索缺口并避免编造已审阅来源。


The default agent entry is [root AGENTS.md](../../AGENTS.md); the **only live research state** is [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json). For a small decision-relevant evidence bundle, read [CURRENT_CONTEXT_PACKET.md](CURRENT_CONTEXT_PACKET.md). The research directory [navigation](../RESEARCH_NAVIGATION.md) separates latest science from old studies.

The project uses pinned [AutoResearcher current source](ACTIVE_SOURCE_MANIFEST.json) (`5bd7ec53`) with upstream core files and three specialist skills kept **verbatim**. FFNDS-specific application lives in root `AGENTS.md` and `.agents/skills/ffnds-paper-discovery/SKILL.md`; generic AutoResearcher source files should not contain appended FFNDS policies. Apply the mechanism/reformulation/expansion methods that materially advance the question; combine them when a mathematical reformulation and source reconstruction need each other.

Earlier `SOURCE_MANIFEST.json`, `PRD_SOURCE_MANIFEST.json` and `AUTORESEARCHER_AGENTS_v2_3.md` are **archived provenance only**, pinned to their historical revisions. Similarly, old 2026-10-23 `COUPLING_PIVOT_TASK.json` may still say `ACTIVE_DISCOVERY_NOT_REGISTERED` in its original source; it has **zero current task authority**. Do not rewrite its original evidence to hide what occurred.

The content-addressed check `tools/check_autoresearcher_binding.py` confirms local version integrity and active-route agreement. It does not measure scientific creativity; [behavior cases](DEFAULT_BEHAVIOR_CASES.json) supply the separate evaluation corpus for authorized future agent comparisons.

已保存的历史暂停回执记录其发生时的执行状态。当前 paired-native v2 与后续授权及运行进度以 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 和原生 RUN 为准。

Superseded operational summaries have moved to the [dated handoff archive](../../archive/20261024/historical_handoffs/README.md). Reproduce the original wording with the archived Git blobs or original source commits; dated execution instructions are historical, not a reason to launch an experiment.
