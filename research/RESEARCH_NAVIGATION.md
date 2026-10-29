# FFNDS · 当前研究导航（2026-10-29）

**本文件是唯一的研究阅读导航；最新科研路线以[现有与新增证据驱动的方法推导](20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md)为准，实际任务与交接以[资产先行Agent交接](agent_protocol/ASSET_FIRST_AGENT_HANDOFF_20261029.md)为准，真实作业权限以[ACTIVE](ACTIVE_RESEARCH.json)为准。** 旧M-A/B/C文档继续保存有效理论与实验，不作为所有新研究的前置检查。

**最重要的已实测观察**：原BF16 8不同历史source×全部160执行位置的原生操作数、shared80与step160 rank960同条件teacher-state来源留出；旧HiF4 2source×5Block显示时空冗余的层异质性与whole64物理闭包；旧UB1 20source×真实free video显示运动显著退化及有符号误差累计。新方法从结构、可得信息、物理动作出发并行研究；局部SSE/负交叉/末层head用于推导与诊断。原数据和[复算JSON](20261029/ASSET_FIRST_METHOD_DERIVATION_RECALC_20261029.json)分别保留资产身份。

**实验GPU执行优先级**：实际ACTIVE/原生RUN已记录VIDEO20等两卡会话上限，执行该会话必须遵守 `session_max_GPUs=2`，通用动态GPU资源政策不得覆盖它；未来明确授权的新执行范围另行注册。

**实际数值后端**：原始BF16与HiF4 fake-QDQ是不同模型执行配方；已完成[PRO6000 native NVFP4真实FFN的有限原生对照](20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.md)，可用作后端/执行研究锚点，但没有因此获得NVFP4视频候选质量。权重离线静态、激活在线动态量化的producer/时点分别由[数值合同](agent_protocol/WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)核验；HiF4原4bit S1P2含符号。新源簇、新作业和实际GPU资源必须按ACTIVE每个范围冻结；资源遵循[动态政策](agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)，不提前设卡数上限。

**P0开放式数据获取**：[问题驱动采集与判别协议](agent_protocol/EVIDENCE_DRIVEN_ACQUISITION_20261029.md)＋[真实采集需求队列](agent_protocol/ACQUISITION_QUEUE_20261029.json)。对每个机制先判断现有观测是否足够；缺失真实source分层、video latent时间配对、完整160-site、student state、动态FP4代码/尺度或真实Kernel成本时，主动设计并执行注册的新采集。现有资产是第一批证据，不是科研边界；允许增加新机制/新数据域/新变量，不预设固定样本上限。

**S2的时间统计证据维度（并非新增并行研究路线）**：[完整数学、论文与判别顺序](20261029/TEMPORAL_DISTANCE_MOTION_COMPENSATION_FFN_HYPOTHESES_20261029.md) · [OPEN计算图claim](method_discussion/graph_annotations/E0/E0_GRAPH_MTC_TDIC_TEMPORAL_INNOVATION_20261029.json) · [新增采集任务AQ-S2-MOTION-INNOVATION](agent_protocol/ACQUISITION_QUEUE_20261029.json)。在20来源60条原BF16完整视频运动降至Dense约0.49/0.48的真实负结果基础上，优先以同Dense-h的BF16 factor重放区分时间DC/AC丢失和学生前缀漂移；再用真实空间邻域采集判别运动对齐的额外收益。三种OPEN行动：同预算AC加权W2函数表示、计费锚点的误差运输、按廉价运动创新信息决定W2成本。时间相似不等于运动向量；已收集的8xy固定轨迹不构成完整运动对应。该新机制与S2、S1、S4并行，无新模型/测试运行或同行批准。

**S2时间统计的当前必读入口：**[完整21τ时间统计结构与新增采集计划](20261029/S2_TEMPORAL_STATISTICAL_STRUCTURE_AND_DATA_EXPANSION_20261029.md)。十张新增开发图的完整21τ/全160采集、原BF16同Dense-h-r960重放已实际exit0；AC相对误差显著大于DC。但已有8xy均值/能量与横邻对照仍欠缺**多τ滞后相关、频谱/相位、时间Gram有效秩、非平稳事件与真实运动空间邻域**，因此时间应作为S2与其余FFN研究方法共享的统计维度，而非第五条部署方向。新纯统计模块和限定CPU producer仅提供后续真实采集能力，尚无新增T1–T6数值或实验RUN。

## P0 · 当前最短阅读和行动路径

| 顺序 | 读取对象 | 立即产生的工作成果 |
| --- | --- | --- |
| **1** | [资产先行方法推导](20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md) + [来源级复算](20261029/ASSET_FIRST_METHOD_DERIVATION_RECALC_20261029.json) | 从真实Wan功能、跨step共享、时空分组、可见量化信息提出竞争性机制 |
| **2** | [统一Agent交接/任务队列](agent_protocol/ASSET_FIRST_AGENT_HANDOFF_20261029.md) + [当前上下文](agent_protocol/CURRENT_CONTEXT_PACKET.md) | 选择A视频基线或S1/S4/S2/S3资产判别，冻结第一项最小对照和产生的证据 |
| **3** | [原BF16全深度真实资产](20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.md) + [shared/step r960](20261029/RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.md) + [旧视频×四步状态](20261029/UB1_20SOURCE_FULL_DEPTH_VIDEO_STATE_EVIDENCE_RECOMPUTATION_20261029.md) | 精确重用producer、tensor row、source exposure、结果与未完成边界 |
| **4** | [视频质量/真实成本终端协议](20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md) + [source×method成本证据](20261029/VIDEO_FIRST_SOURCE_METHOD_COST_MATRIX_20261029.md) | 定义全4step×40layer、同后端多source video与物理FFN/System延迟比较 |
| **5** | [ACTIVE研究范围](ACTIVE_RESEARCH.json) + [历史任务状态台账](agent_protocol/CURRENT_TASK_LEDGER_20261028.json) | 检查当前已有真实运行，实际GPU权限、资产身份、来源与登记范围；完成的RUN不重跑 |

**当前最省成本的完整模型验证对象**：仓库已有BF16 shared80-r960和step160-r960因子，计划和同checkpoint BF16 Dense做全160位多来源自由视频对照；已完成[20来源60条全160位置视频与实际成本](20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)及[官方八轴/连续RAFT](20261029/RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.md)：shared/step运动幅度来源均值比约0.490/0.477，当前质量未通过，单次成本不作公平预热SYS结论。已补齐[S2全160稀疏时间配对OBS](20261029/RAW_BF16_S2_TEMPORAL_PAIR_OBS_RESULTS_20261029.md)，[十张新开发图完整21τ×8xy采集及CPU几何](20261029/RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)和[固定因子同输入时间分量复算](20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)已实际完成：shared门后DC/AC误差0.0293/0.1842。当前新构造为同数据/同秩普通目标与时间平衡目标，尚未拟合。S1（共享/step特化）、S4（行动收益动态秩）、S2（时空函数单元）、S3（前置量化/整组）可并行做已有资产判别，并由实际证据调整优先级。相应论文与旧M-B/G14专题按P1–P3按需深读。

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

**QCFB/Q-STAGE 研究整合与三轮交锋（并行分析，非新的模型作业）**：[统一前沿v2](20261028/INTEGRATED_QMETA_FUNCTION_EXECUTION_FRONTIER_20261028.md) · [原数学命题审计](20261028/QCFB_QSTAGE_PROPOSITION_AUDIT_AND_DERIVATION_20261028.md) · [R1–R3评议/回应顺序](independent_reviews/QCFB_20261028/README.md) · [QCFB机器判别关系](agent_protocol/QUANT_FUNCTION_EXECUTION_FRONTIER_CROSSWALK_20261028.json)。R3已澄清：分箱碰撞不能充当原连续Q1信号的信息下界；native数值余项需要和行动差距比较；类地址不自动修复tile不可执行性；跨步Gram须使用状态空间注入、合法时序可见信息。**真实Q1净价值与省算仍OPEN；B0/B1已完成，当前M-B后续优先级见v3方案。**

**2026-10-28 M-B 四轮独立评议综合（非方法权威、非 ACTIVE 任务更改）**：[研究结论、第四轮行/通道误差纠错与 W0–W3 判别](independent_reviews/M_B_20261028/SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md)；[原始四轮评议目录](independent_reviews/M_B_20261028/README.md)。需要争议的版本演进时再进入该目录；**当前模型方法任务仍按 P0 的M-B交接执行**。

**使用论文图的最短工具路径**（只读/静态）：\`python tools/graph_agent_bridge.py paper L34\` 查论文，\`focus T06\` 查真实Wan算子焦点，\`validate\` 查关系完整性。99/102项旧口径是时间点差异：当前索引为**98主＋10补充，共108项**；主计算图为225节点/468原始关系，新的方法约束是单独的12条typed先验关系，不混作新增原生因果边。

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
