# FFNDS · 阅读优先级与当前研究导航（2026-10-28）

> **本文件只负责“先读什么、找到什么、哪些是历史”。** 科学目标及研究边界见[统一研究目标](RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)；**当前方法的权威正文**是[三路线整合研究方案 v2](20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)，下一Agent的**具体执行交接**在[B0/B1任务](agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。**当前任务状态与依赖**见[唯一状态台账](agent_protocol/CURRENT_TASK_LEDGER_20261028.json)；[动态资源规则](agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)通用默认不预设全局上限；具体实验的冻结资源预算与ACTIVE中的当前授权单独核对，GPU/模型/视频作业范围仍以[ACTIVE](ACTIVE_RESEARCH.json)为准。本导航与台账均不改实验授权或独立评议。

**数值执行路线已调整（2026-10-28）**：[BF16与RTX5090 NVFP4双轨Agent执行合同](agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md)。主机制基线为原始BF16 Dense/候选；低比特系统目标为已具备的RTX5090 SM120上成熟NVFP4 W4A4（需先核对CUDA/编译/32GB显存、完整Wan模型身份）；HiF4 fake-QDQ退为数值诊断。旧HiF4 weight_qdq非原始BF16。官方NVFP4预设的时刻/attention与旧HiF4不同，先在同一recipe内配对Dense/候选。现有 M-B/B-H、A0未完成的公平对照继续，并行F0无需新GPU。

## P0 · 立即阅读：恢复上下文与当前首项研究

| 顺序 | 阅读对象 | 得到的具体信息 | 进入下一步的条件 |
| --- | --- | --- | --- |
| **1** | [当前最小上下文包](agent_protocol/CURRENT_CONTEXT_PACKET.md) | Wan原生FFN/QDQ/step、已完成与OPEN边界、当前用户指定的M-B | 确认本轮研究对象及source/模型版本 |
| **2** | [M-A/M-B/M-C统一方案v2](20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) | 三种新函数、G14约束、强比较器与已登记研究顺序 | 不把旧初版A/B/C意见作为新任务 |
| **3** | [M-B数学快照v2](20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) | 配对敏感度、末层65个连续充分统计、B-H/B-F学生及线上代价 | 继承ALG/SRC及实际B0/B1结果；区分已测固定字典与一般配对机制 |
| **4** | [下一Agent B0/B1交接](agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md) | 原生producer/资产/neutral→有限响应→学生函数判别与输出合同 | 复用已完成B0及CAL24配对判别；B-H训练状态按ACTIVE核验 |
| **5** | [ACTIVE真实研究状态](ACTIVE_RESEARCH.json) | 当轮模型/资源权限、已完成的原始RUN、模型和实验范围 | 不将任务文档视为自动运行授权 |

**M-B仍为首要研究：** [CAL24 96flow对齐及byte audit、B1配对判别和B0选中行neutral](20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)已完成。固定16维span、rank4、18候选中paired/shuffle/marginal/PCA/head-only同选PCA；CPU M256结果不是native数值floor。[B-H真实学生](20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)已完成两expert各256次Adam，held8 head SSE高/低0.003958/0.003615；[原M几何同head A0对照](20261028/BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.md)已完成：held8高/低A0 energy 0.000609/0.000661，B-H 0.003958/0.003615，B-H对两A0均0/8胜；B-F与一般paired机制仍OPEN。[A0全80函数/160学生](20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md)已完成局部训练和metadata组装，160 export/1920 tensor独立字节审计exit0；[六视频v2实际exit0](20261028/COMPACT6016_SIX_VIDEO_NATIVE_RESULTS_20261028.md)：两暴露来源、shared seed42、960 FFN、两Dense历史字节一致；v1首block前失败保留原身份。正式质量/人审和公平系统速度仍OPEN。具体运行以[ACTIVE](ACTIVE_RESEARCH.json)为准。

## P1 · 研究方法必备：按问题选择阅读，不必全读

| 想回答的问题 | 首先阅读 | 研究产物应形成什么 |
| --- | --- | --- |
| **为何研究，允许什么创新？** | [I/F/C科学北极星](RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)；[五操作O-A～E](20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md) | 科学问题和实际组合动作，不以“稀疏”代替机制 |
| **先验是否制约了新函数？** | [G14反向先验审计](20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md)；[REQUIRES/CONSTRAINS/COMPETES_WITH/PREDICTS_DIFFERENTLY机器索引](agent_protocol/METHOD_OPERATION_CROSSWALK.json) | 数学条件、竞争解释与对向预测 |
| **这篇论文在真正计算边上做了什么？** | [G14论文到16算子焦点映射](agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)；[G0–G14有注解的源码计算图](../docs/98_端到端计算图与证据注解_20261025.md)；[论文索引](literature/20261026/PRIMARY_LITERATURE_INDEX.json) | 正文/公式/作者代码/decision_time/原生边界的分级出处 |
| **哪些本地实验已真正做过？** | [跨线证据组合](20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)；[实验发现谱系](EXPERIMENT_FINDINGS_ATLAS.md)；[证据索引](EXPERIMENT_EVIDENCE_INDEX.md) | 明确LIT/ALG/SRC/NATIVE/VIDEO/SYS/OPEN与source计数 |
| **需要采样、校准与官方指标？** | [实验Skill](../.agents/skills/ffnds-experiment-evidence/SKILL.md)；[统一数据协议](agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)；[Calibration V2](20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)；[可执行手册](agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md) | source image-cluster隔离、teacher/student配对、真实评测合同 |
| **需要提出或审阅方法？** | [机制Skill](../.agents/skills/mechanism-research/SKILL.md)；[FFNDS领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)；[独立方法讨论队列](method_discussion/QUEUE.md) | 方法/反例、独立Agent来源、独立评议及版本身份 |


**量化operand来源与原生FP4动态元信息（SRC/LIT）**：[权重静态 × 激活动态时序审计](20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md) · [机器时序合同](agent_protocol/WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)。W1/W2权重在PTQ离线固定、Q1/A1和Q2/A2激活在当前FFN动态编码；原生NVFP4/MXFP4 W4A4含两侧scale载荷，weight-only W4A16没有activation FP4元信息。Native FP4要按实际量化核/scale布局和合法路由截止测量。

**HiF4 存储带符号的权威澄清**：[HiF4 S1P2 码字与现行 fake-QDQ 对照](20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md) · [signed format 机器合同](agent_protocol/HIF4_SIGNED_FORMAT_CONTRACT_20261028.json)。元素码字自带1bit符号，纯scale/micro-exponent无符号；现行Wan当前写出的是带符号BF16，不是持久 packed S1P2。该区别直接约束QCFB、G12及Q-Meta的观察成本。

**QCFB/Q-STAGE 研究整合与三轮交锋（并行分析，非新的模型作业）**：[统一前沿v2](20261028/INTEGRATED_QMETA_FUNCTION_EXECUTION_FRONTIER_20261028.md) · [原数学命题审计](20261028/QCFB_QSTAGE_PROPOSITION_AUDIT_AND_DERIVATION_20261028.md) · [R1–R3评议/回应顺序](independent_reviews/QCFB_20261028/README.md) · [QCFB机器判别关系](agent_protocol/QUANT_FUNCTION_EXECUTION_FRONTIER_CROSSWALK_20261028.json)。R3已澄清：分箱碰撞不能充当原连续Q1信号的信息下界；native数值余项需要和行动差距比较；类地址不自动修复tile不可执行性；跨步Gram须使用状态空间注入、合法时序可见信息。**真实Q1信息净价值、native、省算与完整运动仍OPEN；M-B B0/B1优先不变。**

**2026-10-28 M-B 四轮独立评议综合（非方法权威、非 ACTIVE 任务更改）**：[研究结论、第四轮行/通道误差纠错与 W0–W3 判别](independent_reviews/M_B_20261028/SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md)；[原始四轮评议目录](independent_reviews/M_B_20261028/README.md)。需要争议的版本演进时再进入该目录；**当前模型方法任务仍按 P0 的M-B交接执行**。

**使用论文图的最短工具路径**（只读/静态）：\`python tools/graph_agent_bridge.py paper L34\` 查论文，\`focus T06\` 查真实Wan算子焦点，\`validate\` 查关系完整性。99/102项旧口径是时间点差异：当前索引为**98主＋10补充，共108项**；主计算图为216节点/448原始关系，新的方法约束是单独的12条typed先验关系，不混作新增原生因果边。

## P2 · 与当前方法直接有关的真实证据

| 原生证据及身份 | 结论的精确适用范围 | 用于哪条研究问题 |
| --- | --- | --- |
| [M-A0全80函数6016恢复](20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md)（NATIVE局部） | 160学生训练、metadata组装及完整原始字节审计exit0；train16/旧暴露heldout8，v2两来源自由视频已实际完成；正式质量/人审与系统测速待验 | M-B/B-F与M-C的共同强比较器 |
| [M-B配对敏感度I19](method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)（ALG）和[原始head SRC](20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) | 平均代理与联合代理可反向排序；末层固定调制下65个连续充分统计；[CAL24配对字典与B0 neutral](20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)已实测；一般paired机制与学生结果仍OPEN | M-B B0/B1的数学及数值任务 |
| [20来源40条原生配对视频](20261027/UB1_CONSUMER_DISJOINT20_NATIVE_RESULTS_20261027.md)和[质量/连续运动](20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)（VIDEO） | 当前fixed-RR consumer成像平均−1.38%，RAFT幅值−83.28%；20个开发暴露来源，不能归因为M-B已失效或已恢复 | 视频动作与静态画面差异的科学动机 |
| [CAL24来源组稳定性](20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md)（NATIVE/统计） | 4个teacher位置支持选择较稳定；不是配对$s$的真实后缀$J_s$ | 区分固定gate和联合后缀敏感假说 |
| [激活2:4 C5 A100](20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)（合成SYS） | 在线候选与Dense比较仅约3.5%/0.7%余量；未证明原Wan端到端质量与真实加速 | O-D选择真实算子与物理成本 |
| [两开发来源fixed-RR完整视频/官方轴](20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)（VIDEO） | 与其后的20来源consumer和A0 pilot属于不同来源、模型版本和验收范围 | 功能保真和质量证据分层 |

**来源单位重要**：早期“10个source/12条source×seed/1920 FFN sites”是指定历史原生机制研究；新的20来源配对视频、CAL24、P30 train16/旧heldout8均有自己的统计身份。token、step、site及一张图的多个prompt不自动增加独立image-cluster数。

## P3 · 专题深入（有相应研究问题再阅读）

- **量化元信息决策价值（I+F+C，2026-10-28）**：[Q-Meta / Q-Joint 研究恢复与融合快照](20261028/QMETA_DECISION_VALUE_AND_FUNCTION_FUSION_METHOD_SNAPSHOT_20261028.md) → [量化时点、获取与消费成本](20261028/QMETA_OBSERVATION_TIMING_AND_EFFICIENT_USE_20261028.md) → [已核查的Q0签名资产清单](20261028/P1_Q0_SAVED_SIGNED_ASSET_AUDIT_20261028.json) → [新图谱 claim](method_discussion/graph_annotations/E0/E0_GRAPH_QMETA_NETVOI_JOINT_20261028.json) → 旧G6D/G6E/G6F及G11/G12；**ALG/SRC/LIT+OPEN，不是原生速度或视频验证**，与M-A/B/C并行，**不变更P0指定的M-B优先级**。
- **I信息线**：[G11低位码](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md) → [G12有符号量化证书](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) → [精确投影和跨线综合](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。
- **F函数线**：[G13函数研究](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) → [O/D/Q/P定位](20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md) → [whole-flow反事实](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)。
- **C执行线**：[RT-Lynx与Wan2:4](20261027/RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md) → [G11五篇原文/代码](literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md) → [物理成本证据](20261027/E0_ACTIVATION24_ONLINE_COMPONENTS_RESULTS_20261027.md)。
- **视频动作功能**：[A5+HWG现象](20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md) → [UB1运动衰减独立审计](20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md) → [指标合同](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)。
- **时序缓存/实时模型**：[跨模型文献及证据](literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)仅作为OPEN迁移/先例研究。实时/流式**实验**当前未恢复，不把论文可读性与模型作业权限混同。

## P4 · 历史存档与原始证据（追溯时读）

[ARCHIVE总索引](../ARCHIVE.md) · [早期机制重置](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) · [旧first-task时间点](20261025/E0_FFN_FIRST_TASK_20261025.md) · [旧Affine54失配](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md) · [原A/B/C独立评议](20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md) · [最初提案Git版本](https://github.com/Qesire/FFNDS/blob/dd21c539ca6695fb4b470fb21ba459234ff7a588/research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)。

历史报告的“现在运行”“下一步”“暂停/恢复”只对当时冻结scope成立；目前方法进度依P0及[ACTIVE](ACTIVE_RESEARCH.json)核验。历史RUN、HASH、模型/来源、独立同行评议和失败边界不通过此索引重写。只在实际新的SRC/ALG/NATIVE/VIDEO或论文结论改变研究理解时，升级P0–P2；否则将新材料归入对应专题/历史证据而非新增并列权威方案。

