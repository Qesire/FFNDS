# FFNDS · 下一Agent最小上下文（2026-10-28）

> **职责**：使接手者正确恢复现行状态，**不重复解释**所有论文与历史实验。全文阅读顺序由[P0–P4优先级导航](../RESEARCH_NAVIGATION.md)管理；研究数学/方法的权威正文是[M-A/M-B/M-C方案v2](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，具体B0/B1交接在[下一Agent任务](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。[当日任务状态/依赖台账](CURRENT_TASK_LEDGER_20261028.json)负责READY/BLOCKED/DEFERRED状态；此上下文不触发新模型实验。

## 一、科学对象与真实计算边

科学入口：I信息/表示、F函数/动力学、C实际计算；方法操作O-A计算选择、O-B函数表示、O-C历史复用、O-D计算代数、O-E学习与验证可组合。用户指定从**M-B候选**继续研究（O-B＋O-E，可选O-D）；它**不是**要研究所有操作分类O-B。由[FFNDS领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[五操作分析](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)统一研究方法。

**权重与激活动态量化（现行SRC/LIT事实）**：[W静态×A动态×原生FP4](../20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)：W1/W2 weight在离线PTQ后固定，Q1/A1与Q2/A2 activation每次FFN动态产生；原生NVFP4/MXFP4 W4A4都有W/A各自码与scale，能否给router免费/及时消费取决于producer和layout。W4A16没有当前FP4 activation-scale。任务合同：[机器表](WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)。

**HiF4格式确定性**：[S1P2 signed存储审计](../20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)：64×4bit signed elements（每元素1bit符号）＋32bit unsigned base/micro-scale；原Wan Q1/Q2 fake-QDQ只输出带符号BF16，packed code/scale未作为常驻tensor导出。scale-only不含符号，signed code本身包含符号；所有I/F/C路线按[机器合同](HIF4_SIGNED_FORMAT_CONTRACT_20261028.json)解释。

固定LightX2V Wan2.2 I2V A14B四步D-Q：high/high/low/low，每个expert 40 blocks。当前FFN `norm2/AdaLN→Smooth1/Q1→W1(5120→13824)→GELU(tanh)→Smooth2/HiF4 Q2(64→8→4)→W2(13824→5120)→e5 gate+BF16残差→downstream flow`。典型M=31668；**4去噪step、21 latent时间槽、81 RGB帧不同**。本地fake-QDQ与BF16 GEMM是原生数值边界，外部packed NVFP4/MXFP4另设对照身份。G14有98主＋10补充论文、16焦点；216节点/448原始图边；[方法typed先验](METHOD_OPERATION_CROSSWALK.json)另有12条，不算NATIVE。

## 二、已完成与尚未完成的最短事实表

| 方法/证据 | 已完成 | 后续开放 |
| --- | --- | --- |
| **M-B数学v2** | [I19同来源联合敏感度](../method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)、[U28/block39固定调制的LN/head 65连续中间充分统计](../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)、实际head源码/权重形状SRC | [CAL24 96flow/byte audit、B1固定字典及B0 selected-row neutral](../20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)已完成；一般paired机制、B-F及视频OPEN |
| **M-A0/P30** | [全80函数160学生真实窄Q2局部恢复与metadata组装](../20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md)；已暴露train16/heldout8 | 160 export/1920 tensor独立字节审计exit0；[两来源六视频v2实际完成](../20261028/COMPACT6016_SIX_VIDEO_NATIVE_RESULTS_20261028.md)；正式质量/人审、独立新来源及公平测速OPEN |
| **M-C** | 输入信息充分性、输出秩、量化后非线性、码表预算ALG/LIT：[独立A/B/C评议](../20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md) | 原生训练、Wan真实系统加速与视频未证实 |
| **旧UB1/RR consumer** | [20来源40条原生视频及质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：开发来源画质≈Dense，RAFT运动幅值明显衰减；保留其source/version | 结果是视频运动与代理机制的动机，不是M-B失败或已通过 |
| **O-D物理竞争** | [合成BF16 activation 2:4 C5](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)；online相对dense收益有限 | 真实Wan整端QDQ/视频/系统Pareto独立测量 |

**来源口径**：旧10个source/12个source×seed/1920 site是早期机制研究；consumer20与P30另有独立的历史曝光及统计单位。按image-cluster计真正独立来源，不以tokens/rows/steps/同图多prompt增加样本数。校准先读[实验Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md)和[Calibration V2](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)。

## 三、当前工作：B-H真实学生与一般配对机制

[CAL24实测汇总](../20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)记录96个原生flow及byte audit；固定16维span/rank4/18候选中paired、shuffle、marginal、PCA、head-only同选PCA，CPU M256不代表native floor；B0四case的16次full-M选中行neutral已通过。B-H v1（协议SHA `2832d5be8c5d248c923c3a3b244c5d3c4f84e957efbf95bf36d9605af2a88ce1`）已实际完成GPU训练和worker exit0；[结果](../20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)：两expert各256次Adam，held8 head SSE高/低0.003958/0.003615；[同head A0比较](../20261028/BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.md)已完成：高/低held8 A0 energy 0.000609/0.000661，B-H 0.003958/0.003615，B-H对两A0均0/8胜；96原M neutral和192原A0输出SHA匹配。A0六视频v1首block前失败并保留原身份，[v2六视频actual exit0](../20261028/COMPACT6016_SIX_VIDEO_NATIVE_RESULTS_20261028.md)；正式质量/人审仍待验；B-F及一般paired机制保持OPEN。当前会话最多2张GPU覆盖[通用动态策略](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)的无预设全局上限默认；新范围仍须ACTIVE登记。

## 四、权限、独立评议与交付

[ACTIVE](../ACTIVE_RESEARCH.json)记录所有真实CPU/GPU/model/video执行权限及历史范围；新范围应由协调者按实际研究问题、资产、来源、对照、预算冻结。数学/文献/源码/保存数据可优先探索。独立方法提案和真实peer审阅按[method_discussion/QUEUE](../method_discussion/QUEUE.md)；QUEUE不是GPU任务队列，已完成M-B数学快照不自动成为认可的部署方法。

**下一位Agent有价值的交付**：利用已完成B-H与同head A0匹配对照发展下一机制，继续B-F真实函数图和线上成本，逐项分开ALG/SRC/NATIVE/VIDEO/SYS/OPEN，变化同步主图和方法快照。已有历史失配/论文记录照原证据身份保留，不再重复撰写“B首轮设计任务”。

**跨线同步补充**：QCFB/Q-STAGE的R1–R3讨论已整理在[独立评议目录](../independent_reviews/QCFB_20261028/README.md)，R3提供字典覆盖/信息/学习误差三项分解及跨step合法状态空间注入纠错；这仍是ALG/SRC/LIT+OPEN，优先开展旧数据资产可用性检查，而不是将其视为现行GPU任务。当前`ACTIVE`中的历史暂停、已完成scopes和后来的`user_continuation_authorization_20261027`须按**时间与精确scope**解读；B0、CAL24配对标签及固定字典B1均已完成；后续运行以ACTIVE精确scope为准。

F线因果结构按[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)归档，I/C候选保持并行研究入口。
