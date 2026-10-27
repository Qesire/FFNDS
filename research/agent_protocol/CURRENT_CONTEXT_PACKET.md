# FFNDS · 下一Agent最小上下文（2026-10-28）

> **职责**：使接手者正确恢复现行状态，**不重复解释**所有论文与历史实验。全文阅读顺序由[P0–P4优先级导航](../RESEARCH_NAVIGATION.md)管理；研究数学/方法的权威正文是[M-A/M-B/M-C方案v2](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，具体B0/B1交接在[下一Agent任务](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。此上下文不触发新模型实验。

## 一、科学对象与真实计算边

科学入口：I信息/表示、F函数/动力学、C实际计算；方法操作O-A计算选择、O-B函数表示、O-C历史复用、O-D计算代数、O-E学习与验证可组合。用户指定从**M-B候选**继续研究（O-B＋O-E，可选O-D）；它**不是**要研究所有操作分类O-B。由[FFNDS领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[五操作分析](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)统一研究方法。

固定LightX2V Wan2.2 I2V A14B四步D-Q：high/high/low/low，每个expert 40 blocks。当前FFN `norm2/AdaLN→Smooth1/Q1→W1(5120→13824)→GELU(tanh)→Smooth2/HiF4 Q2(64→8→4)→W2(13824→5120)→e5 gate+BF16残差→downstream flow`。典型M=31668；**4去噪step、21 latent时间槽、81 RGB帧不同**。本地fake-QDQ与BF16 GEMM是原生数值边界，外部packed NVFP4/MXFP4另设对照身份。G14有92主＋10补充论文、16焦点；205节点/424原始图边；[方法typed先验](METHOD_OPERATION_CROSSWALK.json)另有12条，不算NATIVE。

## 二、已完成与尚未完成的最短事实表

| 方法/证据 | 已完成 | 后续开放 |
| --- | --- | --- |
| **M-B数学v2** | [I19同来源联合敏感度](../method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)、[U28/block39固定调制的LN/head 65连续中间充分统计](../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)、实际head源码/权重形状SRC | **B0真实full-M post-write u/head/flow原生捕获、中性重放**未完成；B1有限响应、B-H/B-F学生及完整视频OPEN |
| **M-A0/P30** | [6016 hidden、真实窄Q2、四site同预算Energy94/Random94训练/回读](../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)；已暴露train16/heldout8局部恢复 | 全80 FFN、真实student自由轨迹、独立新source视频OPEN |
| **M-C** | 输入信息充分性、输出秩、量化后非线性、码表预算ALG/LIT：[独立A/B/C评议](../20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md) | 原生训练、Wan真实系统加速与视频未证实 |
| **旧UB1/RR consumer** | [20来源40条原生视频及质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：开发来源画质≈Dense，RAFT运动幅值明显衰减；保留其source/version | 结果是视频运动与代理机制的动机，不是M-B失败或已通过 |
| **O-D物理竞争** | [合成BF16 activation 2:4 C5](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)；online相对dense收益有限 | 真实Wan整端QDQ/视频/系统Pareto独立测量 |

**来源口径**：旧10个source/12个source×seed/1920 site是早期机制研究；consumer20与P30另有独立的历史曝光及统计单位。按image-cluster计真正独立来源，不以tokens/rows/steps/同图多prompt增加样本数。校准先读[实验Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md)和[Calibration V2](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)。

## 三、下一项工作：从B0推进，不重开初版文献评议

**B0-PREP**：按[当前任务](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)定位真实producer、源状态/seed/step/expert/模型SHA与真实$u$、residual、gate、head modulation、flow；依据原始源码/已保存资产建立full-M中性同状态重放的精确代码与协议。**实际捕获需运行授权**，不要将仅存在的Q1/Q2/teacher pre-gate数据当作post-write状态。进入B1后以预冻结方向和幅度比较真实native finite head/flow；再构造合法pre-W1输入的B-H direct-65与B-F通用factored学生，对照A0、PCA、marginal/joint、source shuffle。数学反例已证明联合假说可能不同；Wan真正差异仍需native确认。

**研究先验顺序**：[G14反向审计](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md) → [原生论文/焦点图](PAPER_METHOD_FOCUS_GRAPH.json) → [typed先验/强对照](METHOD_OPERATION_CROSSWALK.json) → 需要时再开[完整计算图](../../docs/98_端到端计算图与证据注解_20261025.md)。核查非线性/Q2合法性、正常path与patch幻象、跨step可预测性和真实big-M/kernel成本。

## 四、权限、独立评议与交付

[ACTIVE](../ACTIVE_RESEARCH.json)记录所有真实CPU/GPU/model/video执行权限及历史范围；新范围应由协调者按实际研究问题、资产、来源、对照、预算冻结。数学/文献/源码/保存数据可优先探索。独立方法提案和真实peer审阅按[method_discussion/QUEUE](../method_discussion/QUEUE.md)；QUEUE不是GPU任务队列，已完成M-B数学快照不自动成为认可的部署方法。

**下一位Agent有价值的交付**：可重放的B0 producer/asset/neutral实现与测试、对B1对向预测及finite数值界的冻结合同、B-H/B-F真实函数图和线上成本，逐项分开ALG/SRC/NATIVE/VIDEO/SYS/OPEN，变化同步主图和方法快照。已有历史失配/论文记录照原证据身份保留，不再重复撰写“B首轮设计任务”。
