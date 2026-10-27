# FFNDS · 研究导航（2026-10-27）

**现行科研目标**：理解并改进视频生成FFN中信息、函数与实际计算的组织方式，获得独立的新机制科学结论或质量–成本Pareto方法。**I信息与表示、F函数与动力学、C计算与执行三条研究入口并行且相互反馈**。[统一目标](RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) · [G13 F线专题](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [O/D/Q/P定位方法](20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md) · [G0–G13源图](../docs/98_端到端计算图与证据注解_20261025.md)。

[I/F/C并行研究路线、方法与证据组合](20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)是当前跨路线的统一对照表：G11/G12编码证书、G13函数/流传播、UB1/A5视频运动、UB1与Affine54函数表示、SpTC 2:4与packed低位、轻量后训练，以及Calibration V2的共用数据接口。该表管理**科学对象及下一判别**，不替代方法QUEUE的同行评议或ACTIVE的执行登记。

**研究规模**：89项相关论文、186个计算图节点、397条有类型关系、14机制家族。原生视频/状态实验覆盖10个source、12个source×seed完整轨迹及1920处嵌套FFN观测。2026-10-27用户已明确恢复推进，同时最多使用四张空闲GPU；当前[ACTIVE](ACTIVE_RESEARCH.json)登记每项冻结范围及真实执行状态。

**UB1历史来源对照与开放判别**：[来源级泛化审计](20261027/UB1_SOURCE_GENERALIZATION_AUDIT_20261027.md)保留两批已暴露来源的成像尾部与同源对照；[原M一致性及独立审阅](20261027/UB1_NATIVE_CONSISTENCY_RESULTS_20261027.md)对应图P26；[匹配恢复及独立审阅](20261027/UB1_MATCHED_RECOVERY_RESULTS_20261027.md)对应P27。后者在12/12探索折上比较支持和消费者；它自身尚无独立确认来源质量证据。后续fixed-RR已完成另两个开发来源的实际完整视频与分维评分（见下文），但不把后续结果回填为这项旧测试当时已经运行；执行状态以ACTIVE为准。

**近期一手论文与原生算子重要更新**：[RT-Lynx激活/权重稀疏、superposition解释和Wan 2:4实际执行审查](20261027/RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)提出三个各自有证据身份的问题：稠密权重的跨输入长期能力、当前输入的2:4使用、SpTC真实硬件操作。原whole64满tile负例**只适用于相应原布局group剪枝**，不是稀疏Tensor Core 2:4的反例。该审查直接关联[修订后的UB1](20261027/UB1_STEP_BUDGET_CALIBRATION_AND_VIDEO_GENERALIZATION_RESEARCH_PROPOSAL_20261027.md)与[Affine54](20261027/QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)方案；这一文献审查本身不包含新Wan视频运行；后续A100合成2:4后端布局探针已运行，仅为预打包算子证据。

**新增视频时空函数入口**：[A5+HWG历史背景闪烁与小物体模糊的机制推导](20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)。研究区分主体结构、细节保持、背景时序协调，拆分空间高低频与视频时间变化，并分别检验mask开关、持续弱功能遗漏、量化尺度和下游传播。它是HISTORY-OBS/ALG/OPEN组合，不是新A5原生实验；[指标协议细化](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)保留已有工程gate，不要求每个数学种子立即跑完整视频。

**UB1运动保真已提升为正式研究假说**：[原始RAFT 40-pair×4 clips重检与运动衰减因果审查](20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)。两个已暴露formal20来源的连续motion幅值UB1/Dense约38.13%（犀牛38/40采样对较小）/9.19%（住宅40/40较小），官方二值Dynamic Degree全0；相对减幅不等于动作语义失败或普遍动态能力崩溃，**纹理模糊降低RAFT可追踪性**是优先判别的对手。按motion-rich subject/camera/static分层，在已有四步源图上区分static appearance、motion function、trajectory propagation；[UB1方案](20261027/UB1_STEP_BUDGET_CALIBRATION_AND_VIDEO_GENERALIZATION_RESEARCH_PROPOSAL_20261027.md)和[视频指标合同](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)已联动修正。VideoJAM/Motive/DCM是成熟运动监督/来源选择近邻，供直接迁移，不要求另造loss。

**新训练、视频与执行结果**：[固定R256 consumer bank](20261027/UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)在24校准来源160位置的训练内gate误差下降83.01%；[两开发来源六条完整视频](20261027/UB1_FIXED_RR_CONSUMER_VIDEO_RESULTS_20261027.md)通过历史Dense/neutral四步latent与视频字节复现。[八轴官方函数评分](20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)显示成像更接近Dense、连续运动部分恢复而美学等轴有取舍；[保存latent几何](20261027/UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)揭示负交叉项与末步距离略增可以并存。下一项保持参数不变，扩大到旧disjoint20开发来源。[合成2:4布局对照](20261027/E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)发现部分预打包路径较快，在线选择与压缩仍另计。图P28/P29分别保留模型恢复与合成物理算子的身份。

**Calibration V2 当前边界**：[冻结设计及METHOD_SNAPSHOT](20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)将来源数、每源行数、Student状态与校准目标拆开；`coverage/diagnose`目前是CPU层的配额与来源级诊断合同，新原生操作数、cohort-aware拟合和冻结视频确认各有独立完成条件。抽样先选独立image cluster，再在簇内登记source/prompt；标准VBench Image Suite可以包含同图多prompt条件，须把条件计数与独立image数分别报告。

**新数据流程索引**：[实验Skill](../.agents/skills/ffnds-experiment-evidence/SKILL.md) · [可执行手册](agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md) · [统一数据协议](agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)。实现分别维护来源历史暴露、固定原生抽样行身份、真实Dense/UB1配对视频、官方分维调用和图像簇统计；旧VBench三轴与新正式九维实验有不同证据身份，当前执行范围以ACTIVE为准。

**新研究方案独立评议（2026-10-27）**：[A/B/C数学、最近先例与最小判别](20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)。A0普通6016学生与A1新增方法必须分开；B对比联合gate/后缀敏感、folded/factored真实MAC；C先测输入编码歧义与输出秩上界，讨论状态，不改变实验授权或Agent同行投票。

**新研究构造入口（A/B/C，讨论状态）**：[可恢复6016维紧凑FFN／下游敏感坐标编译／编码直接计算完整FFN](20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)。该报告以Affine54等MAC恒等、固定RR视频运动缺口、G12精确投影和LUT/CoBRAS/TinyFusion/DIFS文献构造可检验的新函数类；它是新候选研究方案而非替代I/F/C科学框架的终止决议，新的模型效果未运行且不会改变当前ACTIVE授权。

## 按科研问题选择证据

| 要回答的问题 | 主入口 | 精确证据与下一层 |
|---|---|---|
| **哪些信息/表示能改变真实计算？（I）** | [G11编码信息](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12编码区间数学](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [原生信息与成本比较](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[信息→决策→省算](../docs/95_AutoResearcher_三个问题_信息存在决策价值与真实省算_20261024.md) |
| **哪些输出函数具有视频行为意义？（F）** | [G13函数发现](20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) | [历史视频/函数失配](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[多来源signed响应](20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)、[交叉seed](20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md) |
| **哪些真实计算子结构承担这些函数？** | [功能/结构索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G0–G13主图](../docs/98_端到端计算图与证据注解_20261025.md) | [Wan真实配置/shape](20261022/R12_E0_runtime_reconstruction.json)、[HiF4组尺度反例](20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)、[全网络finite response](20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) |
| **哪个先例有不同的科学机制？** | [综合文献研究](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) | [89项论文索引及阅读等级](literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[竞争矩阵](literature/20261026/MECHANISM_COMPETITION_MATRIX.md)、[一手阅读指南](literature/20261026/READING_GUIDE.md) |
| **实时模型的FFN职责和物理形状如何改变？** | [G10实时视频生成/架构/FFN](literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md) | [真实外部模型/源码账本](literature/20261027/REALTIME_VIDEO_SOURCE_LEDGER.json)、[两线机制衔接](20261027/PARALLEL_FFN_MECHANISM_AND_STREAMING_RESEARCH_BRIEF.md) |
| **哪些计算或执行结构可独立重组？（C）** | [RT-Lynx真实2:4 activation operand](20261027/RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)、[G11 FP4信息](literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[G12编码](literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | [G11 Prox/SpQt/CodeGEMM原文与代码](literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ](literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) |
| **已有原生结果支持到哪一层？** | [实验发现谱系](EXPERIMENT_FINDINGS_ATLAS.md)、[证据索引](EXPERIMENT_EVIDENCE_INDEX.md) | [原生全深度与视频验收协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)、[保存状态频谱](20261026/E0_SAVED_FLOW_SPECTRUM.md) |
| **如何进行多Agent独立讨论与图回写？** | [领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)、[Graph↔Agent协议](agent_protocol/GRAPH_AGENT_COLLABORATION.md) | [讨论队列](method_discussion/QUEUE.md)、[图注原始发现](method_discussion/graph_annotations/E0/README.md)、[Graph Agent Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json) |

## 研究记录按主张类型维护

1. **机制种子（I/F/C任选）**：记录问题、观察/公式/源码、竞争机制及其不同预测；新信息界、新算子或新表示无需先登记成FUNCTION。
2. **关键图与信息身份**：对主张涉及的实际算子/状态重建shape、量化路径、step/expert、在线信息可得时间及主要下游；F线的行为–函数–结构定位使用[Function–Structure索引](agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)。
3. **可区分的原始证据**：文献原方程/作者代码留在source ledger，既有RUN/哈希保留原位；机制比较和条件性新关系进入[Crosswalk](agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)及[Graph Bindings](agent_protocol/GRAPH_AGENT_BINDINGS.json)。
4. **按主张升级证据**：机制性数学/因果成果可独立推进；拟发表的加速/视频方法核算真实W1/Q2/W2或整体执行和独立视频质量，按[指标协议](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)检验。
5. **当前合成**：只有改变认识的跨机制结论进入[综合报告](20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。

**历史与权限**：[早期科学任务](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[2026-10-25第一任务原始记录](20261025/E0_FFN_FIRST_TASK_20261025.md)、[原方法反例](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)继续提供真实的时间点证据；现行ACTIVE统一当前优先级与运行授权。用户本轮恢复I/F/C研究及登记后的有限实验；旧暂停和历史RUN保持原样，实时/流式实验未恢复。
