# E0 · 独立行为函数与计算结构发现

**当前科研目标**：[G13行为意义与真实承担结构](../../../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)；来自自由生成视频、signed FFN写回与无标签函数响应的现象，沿原生W1/GELU/HiF4-Q2/W2/attention/step关系提出联合结构，再以独立机制预测判别。[Function–Structure登记](../../../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)维护BEHAVIOR、FUNCTION与STRUCTURE；[主图G0–G13](../../../../docs/98_端到端计算图与证据注解_20261025.md)和[Graph-Agent协作](../../../agent_protocol/GRAPH_AGENT_COLLABORATION.md)保留源及下游边。

**研究方法**：按[FFNDS领域Skill](../../../../.agents/skills/ffnds-paper-discovery/SKILL.md)主动从反直觉数据、新数学、源码与原论文发现不同的函数解释；经[WARD深度搜索](../../../agent_protocol/DEEP_WEB_RESEARCH.md)核对基础、前沿、作者实现和相反先例。在科研记录中清晰写出新结构、真正依据、竞争机制与预期不同结果。此处为**E0_OPEN_DISCOVERY**独立讨论空间，历史提案与失败均保留具体版本和可追溯解释。

**当前执行状态**：[ACTIVE_RESEARCH.json](../../../ACTIVE_RESEARCH.json)保留已完成模型范围及质量记录；新CPU/GPU/模型/视频执行暂停。发现目录用于数学、源码、论文和原证据上的科研讨论。

**当前原生证据**：10个独立source身份、12条source×seed完整轨迹、1920个嵌套FFN位置。[cross-seed有符号作用](../../../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)、[整步有限状态响应](../../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[既有视频失配](../../../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)和[HiF4/物理tile](../../../20261025/E0_SHARED_TILE_PRE_W1.md)是研究功能和结构的互补入口。[初始FFN本征研究](../../../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[第一任务](../../../20261025/E0_FFN_FIRST_TASK_20261025.md)与旧A4/N12、SM120研究继续作为精确历史证据。

## 探索的三种互补切面

- **FFN 输出机制**：从自然输入的gate×W2输出、共享q多W1行、GELU/HiF4分组与不同符号叠加出发，探索实际承担行为作用的联合函数或其它新坐标。
- **时空动态机制**：哪些时空/step/expert条件下的函数响应构成共同或独有职责？它们如何通过真正的残差消费者传递到下一状态？
- **功能/计算机制**：哪种跨W1/W2及后缀路径对背景、运动、光照与时序功能作出贡献？新的代数基或计算单位如何产生不同的物理动作？

独立审查需要亲自看源码、数学或反例，形成真正不同的预测；按 [DEEP_WEB_RESEARCH.md](../../../agent_protocol/DEEP_WEB_RESEARCH.md) 搜索首发论文、官方代码和最强竞争机制。本入口指向现行功能研究，历史论文、原生RUN和旧构造仍按版本保留其证据价值。

## 简短科学记录模板

```markdown
# 新发现/待解释结构：一句准确、可检验的话

**科学张力：** 什么现象值得解释？它对 FFN 的理解意味着什么？

**原始依据：** 精确源码/计算流、已保存数据、条件数学或原论文；注明证据来源和版本。

**机制 A 与机制 B：** 它们各自如何解释这一现象？在哪个条件下会产生不同预测？

**最有价值的检验：** 一个证明、现有数据对照、独立源码检查或未来的最小观测需求。

**可能开启的新自由度（可选）：** 信息源、表示、计算动作或新问题；算子构想先保留为 idea seed。

**实际 Web 检索与学术关系：** 日期、关键词簇、原论文/代码、最接近机制和区别。
```

评价发现的重点是**新增的解释力与可证伪性**。挑战者通过独立论证使科学问题更锐利，整理者根据真实不同作者的记录归纳结果。当前目录用于方法和机理讨论；模型/CPU/GPU 的新增执行由用户另行授权。

## 已归档发现

新增[Affine、GELU奇偶部与dense residency独立审查](independent-affine-residency-audit-20261025.md)：独立核对跨来源校准、18视频轨迹身份与计时边界，并给出半斜率条件证明及signed-group反例。[主研究数据](../../../20261025/E0_AFFINE_FFN_DISCOVERY.json)与[runtime数据](../../../20261025/E0_DENSE_WEIGHT_RESIDENCY.json)保留完整范围；此记录不构成正式方法认可或新增执行授权。

| 作者 / 日期 | 发现记录 | 配套核算 | 证据范围 |
| --- | --- | --- | --- |
| Codex `01a171da` / 2026-10-25 | [最新有限干预的竞争机制与应用种子](codex-01a171da-competing-mechanisms-20261025.md) | [标量重表达 JSON](codex-01a171da-competing-mechanisms-20261025.json) | 基于 `375dc760` 六来源固定四臂结果；新增共同参考/误差去相关零模型及矩约束检查，独立同行评议待进行。 |

2026-10-26新增[函数基/可预测性独立推导](independent-function-basis-prediction-20261026.md)与[四项结果独立审查](independent-function-basis-results-audit-20261026.md)：RRR惩罚目标、表示/系数误差分解、统计增益和保留真实非线性重构；保留跨设置校准混杂与native局部/视频边界的挑战。

2026-10-26补充[匹配控制与全模型源码/计时审查](independent-matched-calibration-and-runtime-audit-20261026.md)，以及[功能后缀/受扰轨迹问题改写](transported-function-and-trajectory-target-20261026.md)。真实窄FFN提速已测，固定自由视频质量失败；局部风险、输入偏移和功能传递须区分。

[独立受扰轨迹/CTEC/TCEC审查](independent-trajectory-error-and-correction-20261026.md)重建实际FP32调度与层内/跨步偏移，给出修复方向的有符号条件，并用原PDF核实一个近邻递推展开的边界反例；不授予新训练/生成。
