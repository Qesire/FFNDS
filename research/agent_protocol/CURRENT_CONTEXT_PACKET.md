# FFNDS · 下一Agent最小上下文（2026-10-28）

> **职责**：使接手者正确恢复现行状态，**不重复解释**所有论文与历史实验。全文阅读顺序由[P0–P4优先级导航](../RESEARCH_NAVIGATION.md)管理；研究数学/方法的权威正文是[M-A/M-B/M-C方案v2](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，具体B0/B1交接在[下一Agent任务](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。[当日任务状态/依赖台账](CURRENT_TASK_LEDGER_20261028.json)负责READY/BLOCKED/DEFERRED状态；此上下文不触发新模型实验。

## 一、科学对象与真实计算边

科学入口：I信息/表示、F函数/动力学、C实际计算；方法操作O-A计算选择、O-B函数表示、O-C历史复用、O-D计算代数、O-E学习与验证可组合。用户指定从**M-B候选**继续研究（O-B＋O-E，可选O-D）；它**不是**要研究所有操作分类O-B。由[FFNDS领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[五操作分析](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)统一研究方法。

固定LightX2V Wan2.2 I2V A14B四步D-Q：high/high/low/low，每个expert 40 blocks。当前FFN `norm2/AdaLN→Smooth1/Q1→W1(5120→13824)→GELU(tanh)→Smooth2/HiF4 Q2(64→8→4)→W2(13824→5120)→e5 gate+BF16残差→downstream flow`。典型M=31668；**4去噪step、21 latent时间槽、81 RGB帧不同**。本地fake-QDQ与BF16 GEMM是原生数值边界，外部packed NVFP4/MXFP4另设对照身份。G14有98主＋10补充论文、16焦点；209节点/431原始图边；[方法typed先验](METHOD_OPERATION_CROSSWALK.json)另有12条，不算NATIVE。

## 二、已完成与尚未完成的最短事实表

| 方法/证据 | 已完成 | 后续开放 |
| --- | --- | --- |
| **M-B数学v2** | [I19同来源联合敏感度](../method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)、[U28/block39固定调制的LN/head 65连续中间充分统计](../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)、实际head源码/权重形状SRC | [B0实际捕获已完成，B1 head有限子范围已完成](../20261028/B_TERMINAL_NATIVE_CAPTURE_AND_FINITE_RESPONSE_RESULTS_20261028.md)；paired判别、B-H/B-F学生/视频OPEN |
| **M-A0/P30** | [6016 hidden、真实窄Q2、四site同预算Energy94/Random94训练/回读](../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)；已暴露train16/heldout8局部恢复 | 全80 FFN、真实student自由轨迹、独立新source视频OPEN |
| **M-C** | 输入信息充分性、输出秩、量化后非线性、码表预算ALG/LIT：[独立A/B/C评议](../20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md) | 原生训练、Wan真实系统加速与视频未证实 |
| **旧UB1/RR consumer** | [20来源40条原生视频及质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：开发来源画质≈Dense，RAFT运动幅值明显衰减；保留其source/version | 结果是视频运动与代理机制的动机，不是M-B失败或已通过 |
| **O-D物理竞争** | [合成BF16 activation 2:4 C5](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)；online相对dense收益有限 | 真实Wan整端QDQ/视频/系统Pareto独立测量 |

**来源口径**：旧10个source/12个source×seed/1920 site是早期机制研究；consumer20与P30另有独立的历史曝光及统计单位。按image-cluster计真正独立来源，不以tokens/rows/steps/同图多prompt增加样本数。校准先读[实验Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md)和[Calibration V2](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)。

## 三、下一项工作：B1源配对判别和对齐标签

已完成对象、真实分歧和下一依赖只按[任务台账](CURRENT_TASK_LEDGER_20261028.json)核对；不重复B0或把B1 head有限子实验当成paired学生通过。当前数据缺口是原CAL24的residual/u/head，原V1/Y/gate和真实256行已存在；按[producer审计](../method_discussion/discoveries/E0/bh-bf-cal24-aligned-label-source-audit-20261028.md)重建原四步链，逐步核原poststate SHA及同坐标V1/Y/gate。现有A0资产与任务状态见台账；后续研究优先M-B；GPU并发按[动态资源政策](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)使用当前实际可调度的最大数量，不预设全局上限。新native范围必须先在ACTIVE冻结登记，当前真实获批GPU数需运行时记录。

**研究先验顺序**：[G14反向审计](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md) → [原生论文/焦点图](PAPER_METHOD_FOCUS_GRAPH.json) → [typed先验/强对照](METHOD_OPERATION_CROSSWALK.json) → 需要时再开[完整计算图](../../docs/98_端到端计算图与证据注解_20261025.md)。核查非线性/Q2合法性、正常path与patch幻象、跨step可预测性和真实big-M/kernel成本。

## 四、权限、独立评议与交付

[ACTIVE](../ACTIVE_RESEARCH.json)记录所有真实CPU/GPU/model/video执行权限及历史范围；新范围应由协调者按实际研究问题、资产、来源、对照、预算冻结。数学/文献/源码/保存数据可优先探索。独立方法提案和真实peer审阅按[method_discussion/QUEUE](../method_discussion/QUEUE.md)；QUEUE不是GPU任务队列，已完成M-B数学快照不自动成为认可的部署方法。

**下一位Agent有价值的交付**：可重放的B0 producer/asset/neutral实现与测试、对B1对向预测及finite数值界的冻结合同、B-H/B-F真实函数图和线上成本，逐项分开ALG/SRC/NATIVE/VIDEO/SYS/OPEN，变化同步主图和方法快照。已有历史失配/论文记录照原证据身份保留，不再重复撰写“B首轮设计任务”。

**跨线同步补充**：QCFB/Q-STAGE的R1–R3讨论已整理在[独立评议目录](../independent_reviews/QCFB_20261028/README.md)，R3提供字典覆盖/信息/学习误差三项分解及跨step合法状态空间注入纠错；这仍是ALG/SRC/LIT+OPEN，优先开展旧数据资产可用性检查，而不是将其视为现行GPU任务。当前`ACTIVE`中的历史暂停、已完成scopes和后来的`user_continuation_authorization_20261027`须按**时间与精确scope**解读；B0已完成且有原始RUN；未来B1配对标签采集尚未登记。

F线因果结构按[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)归档，I/C候选保持并行研究入口。
