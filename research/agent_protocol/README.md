# FFNDS · AutoResearcher source and task routing

## 当前科研问题审查入口

现在的活跃 E0 任务应结合 [2026-10-25 未闭合问题与必需改进](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md) 阅读：Block0 残差径向结构已被真实保存操作数观察到，但还缺独立评议及完整下游因果支持；Q2 时空、来源覆盖、HiF4/native FP4 的研究边界也必须单独注明。所有 agent 用**来源可追溯的竞争性解释与区分观察**弥补问题；不能把有界且已完成的旧 A100 采集权当成可继续自动运行模型的授权。


## 深度 Web Search 是默认研究能力

科研发现、跨 Agent 方法评议与查重应当主动进行**广泛、深入、迭代式联网检索**，而非依赖固定知识或一轮关键词搜索；统一遵循 [深度文献与源码搜索协议](DEEP_WEB_RESEARCH.md)。从经典和最新论文、相邻机制、前后向引用链、官方模型/硬件源码、失败与反例多方向追踪，再把可核验结果回到真实 FFN 计算流。E0 的结论仍是科学现象与竞争解释，不自动变成新算法。对无法访问互联网的 Agent，应在结果中明确检索缺口并避免编造已审阅来源。


The default agent entry is [root AGENTS.md](../../AGENTS.md); the **only live research state** is [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json). For a small decision-relevant evidence bundle, read [CURRENT_CONTEXT_PACKET.md](CURRENT_CONTEXT_PACKET.md). The research directory [navigation](../RESEARCH_NAVIGATION.md) separates latest science from old studies.

The project uses pinned [AutoResearcher current source](ACTIVE_SOURCE_MANIFEST.json) (`5bd7ec53`) with upstream core files and three specialist skills kept **verbatim**. FFNDS-specific application lives in root `AGENTS.md` and `.agents/skills/ffnds-paper-discovery/SKILL.md`; generic AutoResearcher source files should not contain appended FFNDS policies. Apply the mechanism/reformulation/expansion methods that materially advance the question; combine them when a mathematical reformulation and source reconstruction need each other.

Earlier `SOURCE_MANIFEST.json`, `PRD_SOURCE_MANIFEST.json` and `AUTORESEARCHER_AGENTS_v2_3.md` are **archived provenance only**, pinned to their historical revisions. Similarly, old 2026-10-23 `COUPLING_PIVOT_TASK.json` may still say `ACTIVE_DISCOVERY_NOT_REGISTERED` in its original source; it has **zero current task authority**. Do not rewrite its original evidence to hide what occurred.

The content-addressed check `tools/check_autoresearcher_binding.py` confirms local version integrity and active-route agreement. It does not measure scientific creativity; [behavior cases](DEFAULT_BEHAVIOR_CASES.json) supply the separate evaluation corpus for authorized future agent comparisons.

All CPU/GPU/model experiments and automatic followups remain paused per [explicit receipt](../20261024/EXPERIMENT_PAUSE.json).

Superseded operational summaries have moved to the [dated handoff archive](../../archive/20261024/historical_handoffs/README.md). Reproduce the original wording with the archived Git blobs or original source commits; dated execution instructions are historical, not a reason to launch an experiment.
