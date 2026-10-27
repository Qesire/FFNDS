# FFNDS · 当前研究导航（2026-10-27）

**一页进入研究：** [FFN与实时视频/量化信息的统一综合报告](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) → [G0–G12权威源码计算图与外部机制分支](../docs/98_端到端计算图与证据注解_20261025.md) → [12机制分组/节点关系/成本/证据矩阵](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) → [79篇研究与真实原文审计状态](literature/20261026/PRIMARY_LITERATURE_INDEX.json)。

**研究对象：** LightX2V Wan2.2 I2V A14B 四步D-Q，high/high/low/low，原HiF4 fake-QDQ→BF16 FFN；existing native 10个独立source身份、12个source×seed完整生成轨迹、1920个全深度FFN site，已有方法自由视频质量尚未正式验收。**当前允许论文、代码阅读、数学推导、文档与图；全部新模型CPU/GPU/视频实验暂停**。当前唯一状态：[ACTIVE_RESEARCH.json](ACTIVE_RESEARCH.json)。

## 按问题检索，而不是按累计更新记录检索

| 想回答的问题 | 当前入口 | 原始证据分支 |
|---|---|---|
| 原 Wan 在什么时候、以什么dtype和shape计算W1/GELU/HiF4 Q2/W2？ | [G0–G5原生计算图](../docs/98_端到端计算图与证据注解_20261025.md)；[运行时重建](20261022/R12_E0_runtime_reconstruction.json) | [HiF4跨64组尺度反例](20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)；[共享tile](20261025/E0_SHARED_TILE_PRE_W1.md) |
| 原生实验已经确知什么、对视频功能有何限制？ | [实验发现谱系](EXPERIMENT_FINDINGS_ATLAS.md)；[原生证据优先级](EXPERIMENT_EVIDENCE_INDEX.md) | [六来源](20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)、[四新source×seed配对](20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)、[有限整步反事实](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[正式视频/指标协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md) |
| 核心论文究竟做了什么训练、在线选择、实际计算？ | [统一机制综合](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[现行机制竞争矩阵](literature/20261026/MECHANISM_COMPETITION_MATRIX.md) | [79论文索引](literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[G7一手DIFS/FIS](literature/20261026/PRIMARY_METHOD_EQUATION_AUDIT_20261027.md)、[G8四方法](literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md)、[G9](literature/20261027/G9_CONDITIONAL_EXECUTION_TRAJECTORY_AND_FUNCTION_REALLOCATION.md) |
| 实时视频为什么不同、FFN短chunk何时真受益？ | [实时因果/流式14项研究](literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)、[外部模型源码等级](literature/20261027/REALTIME_VIDEO_SOURCE_LEDGER.json) | [完整G10图与SLO](../docs/98_端到端计算图与证据注解_20261025.md)、[双线研究纲领](20261027/PARALLEL_FFN_MECHANISM_AND_STREAMING_RESEARCH_BRIEF.md) |
| FP4有符号码/尺度究竟何时可得、能否预判W1与gate×W2？ | [G11量化信息的可用性和代价](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12直方图严格区间和bitplane](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [Prox/Spark/SpQt/SQ-DM/CodeGEMM深读](literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ双GEMM及Wan T2V](literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md)、[G12源码与阅读等级](literature/20261027/G12_SOURCE_LEDGER.json) |
| 哪些算法仅对单向量GEMV有效？哪种确实省GEMM？ | [G11–G12源码核查与统一成本](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) | CodeGEMM与LUT-GEMM多token均解量化转稠密；SpQt的prefill dense；SharQ需sparse+**dense residual**，见上述一手文献文档 |
| 如何开展可审计多Agent讨论、归档和图同步？ | [Graph↔Agent协议](agent_protocol/GRAPH_AGENT_COLLABORATION.md)、[稳定节点索引](agent_protocol/GRAPH_AGENT_BINDINGS.json) | [文献深读协议](literature/20261026/READING_GUIDE.md)、[独立讨论队列](method_discussion/QUEUE.md)、[证据图注目录](method_discussion/graph_annotations/E0/README.md) |

## 当前研究边界和维护顺序

每一机制从真实数据/源码与原始论文出发，明确**数学对象 → 条件和信息时点 → 获取成本 → 实际可省的W1/Q2/W2、step或block → gate/flow/视频质量与SLO → 竞争解释和最低成本判别**。经典G12重排界是单W1行直方图层级的数学紧界，联合W2功能仍OPEN。不要将signed code的存在说成免费已判定点积，也不要将外部作者硬件/视频结果改标本项目native证据。

更新时只把新原文审计放到单篇来源文件；将改变判断的合成结论更新[统一综合报告](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)，同步[Graph Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json)和[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)。静态检查只证明图与链接/JSON一致，数值及视频结论由原native RUN与完整质量协议决定。

## 历史材料和真实状态

[2026-10-25原本征科学纲领](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[旧FFN第一任务](20261025/E0_FFN_FIRST_TASK_20261025.md)、[2026-10-24科学分歧与机制审计](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[历史交接存档](../archive/20261024/historical_handoffs/README.md)是有证据价值的历史入口，其当时的论文数量与任务指令不再是现行状态。

**状态/权限优先级：** [ACTIVE_RESEARCH.json](ACTIVE_RESEARCH.json) → [AGENTS.md](../AGENTS.md) → [OBJECTIVE_CONTRACT.json](OBJECTIVE_CONTRACT.json)。此导航不产生执行许可、队列任务或新的模型运行。
