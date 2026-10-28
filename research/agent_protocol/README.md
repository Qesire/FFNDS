# FFNDS · Agent协议与实际操作索引（2026-10-28）

> **优先级和研究进度的权威入口**：[P0–P4阅读导航](../RESEARCH_NAVIGATION.md)。本目录负责协议、程序约束、计算图的机器索引，不维护第二份完整“当前研究状态”。[任务状态与依赖台账](CURRENT_TASK_LEDGER_20261028.json)仅记录READY/WAITING/DEFERRED及当前证据身份，不是新的科研路线或实验授权。[根AGENTS](../../AGENTS.md)和[领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)决定科研行为；[ACTIVE](../ACTIVE_RESEARCH.json)决定原生模型作业权限。

## P0 · 下一位Agent直接使用

| 文件 | 应当读到/执行的合同 |
| --- | --- |
| [任务状态台账](CURRENT_TASK_LEDGER_20261028.json) | 当前M-B B0已完成、B1有限head已完成一部分、QCFB/UB1并行数据分析、M-A0部分完成及模型作业依赖 |
| [最小上下文包](CURRENT_CONTEXT_PACKET.md) | 原生Wan/QDQ/step、M-A0已完成局部实验、M-B B0原生已完成且B1首轮finite已有结果、M-C开放 |
| [三方法权威方案v2](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) | G14科学先验转换成M-A/M-B/M-C真正的函数与部署约束 |
| [B0/B1当前任务](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md) | post-write u/head/flow的producer/neutral准备→有限响应、B-H/B-F学生 |
| [现行动态GPU策略](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md) | 无预设全局卡数限制；每轮按实际可调用量及READY任务最大化并发，旧冻结scope不改SHA |
| [现行范围/预算](../ACTIVE_RESEARCH.json) | 具体模型、GPU、来源、预算、独立来源与RUN身份；旧scope不自动续期 |

## P1 · 按研究主张再打开相应协议

**当前量化格式的源事实**：HiF4原码为带符号S1P2，尺度元信息无符号；现行Q1/Q2是BF16 fake-QDQ输出，具体见[HiF4符号存储审计](../20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)与[signed格式机器合同](HIF4_SIGNED_FORMAT_CONTRACT_20261028.json)。

| 场景 | 主协议与工具 |
| --- | --- |
| 原论文映射到Wan计算边 | [94＋10篇G14机器焦点图](PAPER_METHOD_FOCUS_GRAPH.json)、[G0–G14可见源图](../../docs/98_端到端计算图与证据注解_20261025.md)、[原生Graph Bindings](GRAPH_AGENT_BINDINGS.json) |
| 五操作与方法先验查重 | [O-A～E×I/F/C](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)、[12条typed prior机器索引](METHOD_OPERATION_CROSSWALK.json)、[G14反向审计](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md) |
| 新数据、校准、配对视频、来源分割 | [数据采样/校准/评价合同](DATA_SAMPLING_CALIBRATION_EVALUATION.md)、[Calibration V2](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)、[实验专用Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md)、[可执行实验手册](EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md) |
| 新功能/因果定位或Graph Agent | [Function–Structure Objects](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G13函数/结构方法](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[Graph Agent协作](GRAPH_AGENT_COLLABORATION.md) |
| 独立新方法/同行复核 | [method_discussion讨论规则](../method_discussion/AGENTS.md)、[原始提案及审阅队列](../method_discussion/QUEUE.md)、[独立discovery](../method_discussion/discoveries/E0/README.md) |
| 研究来源与固定上游 | [ACTIVE_SOURCE_MANIFEST](ACTIVE_SOURCE_MANIFEST.json)、[UPSTREAM_SYNC](UPSTREAM_SYNC.md)、[AutoResearcher绑定静态检查](../../tools/check_autoresearcher_binding.py) |

## 证据边界、保存与回归

原始RUN与哈希、teacher/student状态、source image-cluster、step/block嵌套统计、连续视频和VBench原函数/官方采样分别保持原记录和实际身份证据。ALG/SRC/LIT对科学解释有价值；NATIVE/VIDEO/SYS要有真实本地执行收据。静态图有效性不构成实际Wan推理、同行支持或模型质量通过。

只读图谱及回归：
```bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py paper L34
python tools/graph_agent_bridge.py focus T06
python tests/test_graph_agent_bridge.py
python tools/check_autoresearcher_binding.py
```

**资料生命周期**：本目录只维护协议和机器权威表；各研究的公式、竞争机制与已完成实验归对应方案及[发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)；旧研究入口、暂停和历史结论由[ARCHIVE](../../ARCHIVE.md)供溯源，绝不以导航清理修改实际实验资产。
