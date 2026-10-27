# FFNDS · 研究导航（2026-10-27）

**现行科研目标**：定位视频FFN中有意义的输出函数、承担这些函数的真实计算结构，然后研究其可共享、可预测或可更经济执行的表示。[G13函数与结构](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) → [行为/函数/结构登记](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) → [G0–G13真实计算及研究图](../docs/98_端到端计算图与证据注解_20261025.md)。

**研究规模**：85项相关论文、171个计算图节点、354条有类型关系、14机制家族。原生视频/状态实验覆盖10个source、12个source×seed完整轨迹及1920处嵌套FFN观测。新模型CPU/GPU/视频运行处于暂停状态；当前[ACTIVE](ACTIVE_RESEARCH.json)提供任务和授权身份。

## 按科研问题选择证据

| 要回答的问题 | 主入口 | 精确证据与下一层 |
|---|---|---|
| **哪些输出函数具有视频行为意义？** | [G13函数发现](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) | [历史视频/函数失配](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[多来源signed响应](20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)、[交叉seed](20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md) |
| **哪些真实计算子结构承担这些函数？** | [功能/结构索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G0–G13主图](../docs/98_端到端计算图与证据注解_20261025.md) | [Wan真实配置/shape](20261022/R12_E0_runtime_reconstruction.json)、[HiF4组尺度反例](20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)、[全网络finite response](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) |
| **哪个先例有不同的科学机制？** | [综合文献研究](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) | [85项论文索引及阅读等级](literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[竞争矩阵](literature/20261026/MECHANISM_COMPETITION_MATRIX.md)、[一手阅读指南](literature/20261026/READING_GUIDE.md) |
| **实时模型的FFN职责和物理形状如何改变？** | [G10实时视频生成/架构/FFN](literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md) | [真实外部模型/源码账本](literature/20261027/REALTIME_VIDEO_SOURCE_LEDGER.json)、[两线机制衔接](20261027/PARALLEL_FFN_MECHANISM_AND_STREAMING_RESEARCH_BRIEF.md) |
| **定位后有什么计算实现？** | [G11 FP4信息/具体方法](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12有符号编码/计算](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [Prox/Spark/SpQt/SQ-DM/CodeGEMM深读](literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ](literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) |
| **已有原生结果支持到哪一层？** | [实验发现谱系](EXPERIMENT_FINDINGS_ATLAS.md)、[证据索引](EXPERIMENT_EVIDENCE_INDEX.md) | [原生全深度与视频验收协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)、[保存状态频谱](20261026/E0_SAVED_FLOW_SPECTRUM.md) |
| **如何进行多Agent独立讨论与图回写？** | [领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)、[Graph↔Agent协议](agent_protocol/GRAPH_AGENT_COLLABORATION.md) | [讨论队列](method_discussion/QUEUE.md)、[图注原始发现](method_discussion/graph_annotations/E0/README.md)、[Graph Agent Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json) |

## 研究记录按“发现—结构—判别—实现”维护

1. 在[Function–Structure索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)分别记录行为现象、候选函数、正常计算结构。条件信息包括source/seed/step/expert/block、原生证据和真实下游消费者。
2. 将候选函数与原生G0–G5源码路径关联，提出若干不同解释，给出最有信息价值的数学推导、已保存数据辨识或未来特定图边干预。
3. 深读相关论文原文/实现及失败消融，将可转移的机制及不同预测对应到LID、阅读层级和[Graph Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json)。研究原创思想可以引入新的数学对象、条件边或图节点。
4. 当一种计算结构的功能作用逐渐明确，再选择合适的量化code/scale、proxy、编码域LUT、稀疏格式、cache或训练后构造，以匹配实际W1/Q2/W2工作与视频保真。
5. 现行机制结论进入[综合报告](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)，实验保留原source/RUN/哈希，图谱只标与证据身份相符的SRC、OBS/NATIVE、ALG、LIT或OPEN。

**历史与权限**：[早期科学任务](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[2026-10-25第一任务原始记录](20261025/E0_FFN_FIRST_TASK_20261025.md)、[原方法反例](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)继续提供真实的时间点证据；现行ACTIVE统一当前优先级与运行授权。相关模型新执行暂停，科研文献、源码、数学、图谱及已保存证据的解释整理继续推进。
