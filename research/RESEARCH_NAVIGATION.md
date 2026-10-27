# FFNDS · 研究导航（2026-10-27）

**现行科研目标**：理解并改进视频生成FFN中信息、函数与实际计算的组织方式，获得独立的新机制科学结论或质量–成本Pareto方法。**I信息与表示、F函数与动力学、C计算与执行三条研究入口并行且相互反馈**。[统一目标](RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) · [G13 F线专题](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [O/D/Q/P定位方法](20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md) · [G0–G13源图](../docs/98_端到端计算图与证据注解_20261025.md)。

**研究规模**：89项相关论文、177个计算图节点、371条有类型关系、14机制家族。原生视频/状态实验覆盖10个source、12个source×seed完整轨迹及1920处嵌套FFN观测。新模型CPU/GPU/视频运行处于暂停状态；当前[ACTIVE](ACTIVE_RESEARCH.json)提供任务和授权身份。

## 按科研问题选择证据

| 要回答的问题 | 主入口 | 精确证据与下一层 |
|---|---|---|
| **哪些信息/表示能改变真实计算？（I）** | [G11编码信息](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12编码区间数学](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [原生信息与成本比较](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[信息→决策→省算](../docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md) |
| **哪些输出函数具有视频行为意义？（F）** | [G13函数发现](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) | [历史视频/函数失配](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[多来源signed响应](20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)、[交叉seed](20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md) |
| **哪些真实计算子结构承担这些函数？** | [功能/结构索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G0–G13主图](../docs/98_端到端计算图与证据注解_20261025.md) | [Wan真实配置/shape](20261022/R12_E0_runtime_reconstruction.json)、[HiF4组尺度反例](20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)、[全网络finite response](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) |
| **哪个先例有不同的科学机制？** | [综合文献研究](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) | [89项论文索引及阅读等级](literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[竞争矩阵](literature/20261026/MECHANISM_COMPETITION_MATRIX.md)、[一手阅读指南](literature/20261026/READING_GUIDE.md) |
| **实时模型的FFN职责和物理形状如何改变？** | [G10实时视频生成/架构/FFN](literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md) | [真实外部模型/源码账本](literature/20261027/REALTIME_VIDEO_SOURCE_LEDGER.json)、[两线机制衔接](20261027/PARALLEL_FFN_MECHANISM_AND_STREAMING_RESEARCH_BRIEF.md) |
| **哪些计算或执行结构可独立重组？（C）** | [G11 FP4信息/具体方法](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12有符号编码/计算](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [Prox/Spark/SpQt/SQ-DM/CodeGEMM深读](literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ](literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) |
| **已有原生结果支持到哪一层？** | [实验发现谱系](EXPERIMENT_FINDINGS_ATLAS.md)、[证据索引](EXPERIMENT_EVIDENCE_INDEX.md) | [原生全深度与视频验收协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)、[保存状态频谱](20261026/E0_SAVED_FLOW_SPECTRUM.md) |
| **如何进行多Agent独立讨论与图回写？** | [领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)、[Graph↔Agent协议](agent_protocol/GRAPH_AGENT_COLLABORATION.md) | [讨论队列](method_discussion/QUEUE.md)、[图注原始发现](method_discussion/graph_annotations/E0/README.md)、[Graph Agent Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json) |

## 研究记录按主张类型维护

1. **机制种子（I/F/C任选）**：记录问题、观察/公式/源码、竞争机制及其不同预测；新信息界、新算子或新表示无需先登记成FUNCTION。
2. **关键图与信息身份**：对主张涉及的实际算子/状态重建shape、量化路径、step/expert、在线信息可得时间及主要下游；F线的行为–函数–结构定位使用[Function–Structure索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)。
3. **可区分的原始证据**：文献原方程/作者代码留在source ledger，既有RUN/哈希保留原位；机制比较和条件性新关系进入[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)及[Graph Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json)。
4. **按主张升级证据**：机制性数学/因果成果可独立推进；拟发表的加速/视频方法核算真实W1/Q2/W2或整体执行和独立视频质量，按[指标协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)检验。
5. **当前合成**：只有改变认识的跨机制结论进入[综合报告](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。

**历史与权限**：[早期科学任务](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[2026-10-25第一任务原始记录](20261025/E0_FFN_FIRST_TASK_20261025.md)、[原方法反例](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)继续提供真实的时间点证据；现行ACTIVE统一当前优先级与运行授权。相关模型新执行暂停，科研文献、源码、数学、图谱及已保存证据的解释整理继续推进。
