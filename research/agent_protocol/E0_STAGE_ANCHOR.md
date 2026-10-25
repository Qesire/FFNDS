# FFNDS E0 · 当前科研 Agent 锚点（2026-10-25）

**最新机制审查**：[2026-10-25 开放分歧](../20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md)；[已完成的K1–K3与剩余K4–K5](../20261025/E0_FFN_KNOWLEDGE_GAPS_AND_DISCRIMINATORS_20261025.md)。原生全hidden、连续M128和原组支持研究已完成；优先发现新的函数、时空、质量和系统机制。

**阶段：`E0_OPEN_DISCOVERY` · 唯一科学状态：[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) · 当前任务：[E0_FFN_FIRST_TASK_20261025.md](../20261025/E0_FFN_FIRST_TASK_20261025.md) · 科研纲领：[E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)。**

## 问题的核心

研究 Wan 四步蒸馏视频模型 FFN 中的条件计算需求。**从 `W1→GELU→W2` 的激活、W2 有符号贡献、token/step/expert 条件变化、下游视频敏感性和可替代真实计算出发**，主动寻找数学结构、反例和新表示。HiF4 QDQ 是已有实验的数值实现，不是所有方法的默认起点；量化元信息优先视为粗略幅值/范围线索。

**进一步打开的数学表示**：将 FFN 看作按输入与 gate 变化的 weighted-value composition；分析神经元重要性是否受基表示和 superposition 影响，区分 `Σ_j h_j(g⊙W2[:,j])` 的净向量与 `W2 diag(GELU'(a)) W1` 的局部敏感方向。详见 [增补文献地图](../20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md)。

特别区分：

- `|h_j|` 小：某激活大小的描述。
- `||h_j W2[:,j]||` 或有符号组和小：在 W2 下的 FFN 局部输出作用。
- finite downstream response 小：residual/attention/多步视频的实际函数影响。
- 可省真实计算：W1/GELU/W2、访存、布局与决策代价合计使物理代价下降。

这些可能相关，也可能彼此分离。**当前目的不是先构造新的 Proxy、固定64组或 Dense completion，而是发现决定上述关系的机制。**

## 计算图参与研究的默认方式

从 [G0–G4计算图](../../docs/98_端到端计算图与证据注解_20261025.md) 和 [Graph-Agent协作协议](GRAPH_AGENT_COLLABORATION.md) 获取真实算子路径、Q1/W1/GELU/Q2/W2数值边界、历史可用性与下游消费者；借 [JSON节点索引](GRAPH_AGENT_BINDINGS.json) 查证据和未解问题。每项新科学认识形成一份 [图锚点讨论记录](../method_discussion/graph_annotations/E0/README.md) 或在原发现中声明相关节点；独立reviewer沿路径反推源码/数学反例，整理者把受支持的注释回写计算图和实验谱系。新表示可以提出新的图节点，勿将现有G4 shared-column视为所有实验的默认路线。

## 独立科学发现入口

可从下列三组互补问题进入，也允许新的原创问题：

1. **hidden 与净贡献**：激活有效支持、W2 方向与符号交互、非线性、Gram 与可执行块是否呈真实条件结构？
2. **条件时空创新**：四步、双专家、跨位置/跨帧的共同/变化模式，哪些可被便宜地预测，哪些需重新计算？
3. **下游功能与计算重构**：为什么某些低局部误差引起背景光照/小物体波动？有无与原始 mask/缓存不同的可替代计算表示？

**待区分的强张力**：后验25%逐token单元可重建同专家创新（2.44%–7.07% SSE），中间层共享后验25%列则约39%–42%，原HiF4组尺度几乎全宽。继续比较函数坐标、token空间组织、下游视频功能和真实执行时点，而不是将共享列、micro16或量化树升级为默认优化任务。旧LayerNorm径向、四臂、跨来源方向与scale结果仍可按机制调用。详见 [既有观测导航](../EXPERIMENT_EVIDENCE_INDEX.md)、[历史 E0 gap 文档](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md) 和 [研究重置文献地图](../20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md)。

## AutoResearcher 默认操作

- **发现与扩展**：发现最强异象/机制张力，允许新算子作为 idea seed。
- **源码重建**：LightX2V 冻结配置下的输入、shape、gate、W1/GELU/W2、residual、step/expert；优先审计决策关键子图。
- **数学与问题重构**：精确公式、相反预测、最强反例；可以跳出熟悉的通道空间和单一稀疏单位。
- **WARD 搜索**：主动执行 [DEEP_WEB_RESEARCH.md](DEEP_WEB_RESEARCH.md)；深入比较经典、近期 FFN/DiT/视频研究及其原始公式和代码，从引用链提炼新机制。
- **区别与交付**：提出最廉价可判别的证明、源码或已存证据；将未完成的实验写成以后单独授权的提议。

所有 Agent 以 [E0 发现目录](../method_discussion/discoveries/E0/README.md) 提交独立署名科学发现或挑战，其他 Agent 不靠措辞上的支持替代自己的数学/源码审查。方法讨论仅讨论；整理者串行维护共享 queue 和认可快照。

## 证据与执行

新模型/GPU/视频/训练/原生干预在旧协议全部完成后**没有自动续授权**；目前没有待运行的新模型任务，最多四卡属于历史完成授权上限，不代表可以自行重启。可开展读取、数学、源码、文献和获准的已存资产离线分析。未来可执行方法仍须检查 Dense 及最强等成本静态/动态先例、整 FFN 和端到端延迟、四步自由视频的运动、画面、背景光照与细节稳定性。E0 的新结构/条件定理本身已有科研价值，不需要在发现时预先证明完整部署获益。

**档案关系**：原 [量化优先第一任务](../20261024/FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md) 与 [径向配对后续引导](../20261025/E0_NEXT_MECHANISM_GUIDANCE_20261025.md) 属于历史阶段文档，不改变其保存实验结论；当前后续始终从此锚点与新第一任务开始。
