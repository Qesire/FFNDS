# FFNDS · 当前Agent最小上下文（2026-10-29）


**最高效力用户最新指令（2026-10-29）：只研究FFN稀疏/省算的Wan2.2-I2V四步基线。** [唯一现行筛选](../20261029/WAN_NATIVE_TRAINING_FREE_BASELINE_REPRIORITIZATION_20261029.md) / [ACTIVE.current_user_ffn_only_priority_20261029](../ACTIVE_RESEARCH.json)。**P0 #59 SharQ原生Wan激活N:M FP4+Dense残差双GEMM，P0 #58 TEAL已完成Wan全160四来源质量后探索真正大M稀疏W2；P1 #57 Chipmunk Wan FFN仅单组paid proxy成功。** 已有TEAL 20视频/四来源RAFT八轴EXIT0，25%/50%运动比1.170/1.376（50%额外物体/来源比0.423–2.231），但完整Dense W2，零物理FFN省算；Chipmunk K5120单组误差0.279554优于等K随机，但未全160视频。**立即停止/不启动原作者LLM Llama、FLUX原系统、MagCache/FIS帧跳、Attention、Token/Block/step缓存等非FFN执行。** 这些历史资产留存。后文此前Chipmunk与TEAL并列P0及LLM优先表述是过去阶段，已被本段覆盖；新实验仍需针对本次实验独立review/recheck和ACTIVE冻结，最新真实RUN见下段。



**最新真实证据与下一任务（2026-10-30）**：[TEAL物理可达性](../20261029/TEAL_WAN_MASK_PHYSICAL_REACHABILITY_RESULTS_20261030.md)320包/640次CPU评估实际EXIT0并closed：25/50阈值整M冷列平均上界仅0.019938/0.054480%，采样行直接完整K4方向2:4合格均0。保持mask需另寻真实局部行组/残差，改变mask则重新评价；非raster样本不决定M16成本。[四来源完整视频质量](../20261029/TEAL_WAN_ALL160_FOUR_SOURCE_QUALITY_RESULTS_20261029.md)原结果及Dense W2身份不变。[SharQ原生V2](../20261029/SHARQ_WAN_FFN_NATIVE_V2_RESULTS_20261030.md)240次FFN已实际EXIT0/闭合：两专家block20，M31668重复真32行，sameW Dense32对照下gate后L2改善约23%、FFN时间增加约39%，没有视频/SYS结论。[真实两源8个full-M H采集](../20261030/TEAL_WAN_FULLM_H_CAPTURE_RESULTS_20261030.md)已EXIT0且全部独立输出SHA/20state/2video中性验证，连续行组CPU范围正在精确复审；完整全160 native视频producer继续实现。继续Wan FFN-only；不再启动LLM/FLUX等离题范围。当前会话最多同时4卡，旧scope/失败身份及收据不改写。



**新增用户优先指令（2026-10-29）**：所有新的真实 CPU/GPU/模型/视频/采集研究实验，在执行前必须重新独立评议 **本次实验本身**，而不只验证方法大方向。先主Agent自审→真实不同Agent反例审查→修订与recheck→ACTIVE冻结有限scope→launch前核SHA；协议见[EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW](EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)。当前复现任务为 [Issue #56](https://github.com/Qesire/FFNDS/issues/56) 与[Chipmunk/TEAL正向FFN基线评议](../20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)：作者条件忠实复现和当前四步Wan2.2迁移是不同scope，Chipmunk MLP-only和含attention/step-cache不是同一FFN基线，TEAL preW1和postGELU节省不同计算。每个尚未完成的新scope须独立审查；已闭合的TEAL有限kernel scope有其自身审查及执行记录，不自动授权其他scope。



**唯一当前路线**：[研究导航](../RESEARCH_NAVIGATION.md) → [资产先行四路线推导](../20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md) → [现行执行队列](ASSET_FIRST_AGENT_HANDOFF_20261029.md)。此文件只是接手所需的事实与状态入口，完整数学在对应方法文档、真实作业状态在[ACTIVE](../ACTIVE_RESEARCH.json)。[M-A/M-B/M-C旧整合方案v4](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)和[B专题旧交接](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)为有用的专题与历史来源，不再规定默认“先末层”顺序。

**新增最高优先级规则：发现缺口就补采**。参见[开放式科学采集协议](EVIDENCE_DRIVEN_ACQUISITION_20261029.md)、[机器采集队列](ACQUISITION_QUEUE_20261029.json)：新source/新的scene stratum、4×40全深度time×space tubelet、native Q1/Q2 activation metadata、student前缀状态、视频动作与真实SYS等可主动补采，旧8来源、5 Block及现有算法仅为已知证据。先采用现有数据满足已有科学问题；一旦现有观测不足，转为新producer/数据集/新采集，而不是将问题降格为旧数组可以算的proxy。按每次ACTIVE冻结有限scope并可持续扩展下一批。

## 科学目标和新方法

研究目标：在保留Wan2.2 I2V生成视频主体动作、语义、背景、相机、时间行为和感知质量的前提下，发现FFN在信息、函数、真实执行上的低成本组织规律。以**实际源已核的结构发现**作为起点，提出相互竞争且可证伪的解释，再设计在线可用、物理可执行的新行动。SSE、有符号抵消、范数补偿、补算是解释和对照工具，不是默认研究目标。

- **S1-CSFB**：共享跨step的函数基＋适量step收缩特化。来自真实BF16 r960：shared在6来源留一的8函数锚点6/6更优、2旧开发来源2/2更优；但训练集上step特化更准。**新收缩拟合与全视频效果待测**。
- **S4-FRANK**：利用原BF16真实h和W2构造能在W2前评估额外低秩块收益的低价草图；该方法在原W1/GELU后决策，最多减少W2计算。**数学投影性质已推导，真实source-held regret与部署成本未测**。
- **S2-FTFD**：从旧HiF4连续tile在5层首中尾的创新差异出发，发现真实视频latent时间×空间的共享函数计算单元。**已有[八来源全160处26389个稀疏时间对观察](../20261029/RAW_BF16_S2_TEMPORAL_PAIR_OBS_RESULTS_20261029.md)；相同xy优势已出现在V1。[十张新开发图完整21τ、8xy、40层×4步采集及CPU观察](../20261029/RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)已实际完成：1600包/全部u字节neutral；[同Dense h固定因子复算](../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)显示shared门后输出DC误差0.0293、AC误差0.1842，小M舍入不足解释。同数据/秩[普通与DC/AC平衡目标的80函数拟合](../20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)现已实际exit0；训练AC误差0.051229→0.047267，旧8图rowLN(u)误差基本持平。[20开发图两候选完整40视频及八轴](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)已实际exit0、复用原Dense；RAFT为Dense的0.451/0.439，未达到要求。下一变量是真实空间邻域与student状态，不继续调本目标权重**。
- **S3-QFAR**：以HiF4 whole64组闭包和W1前可用的信息为真实合法行动集合。旧任意列25%保留触及216/216组，whole64对角25%保留在所测创新上表现差；**新的条件有符号联合组行动及原生NVFP4独立闭包待判别**。

四支路线并行，任意Agent可以根据真实源与论文提出更优竞争机制。第一项**成本最小且可执行的完整视频基线**是已导出的原BF16 shared80-r960、step160-r960和相同原BF16 Dense的真实160位置同source生成；这项基线已完成[20来源60视频及单次成本](../20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)和[八轴官方函数/连续RAFT](../20261029/RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.md)，当前质量未通过：shared/step来源等权RAFT幅值为Dense的0.490/0.477。公平预热重复SYS仍待测；不用重复已完成视频。

**S2时间统计及空间/学生新数据**：时间T1–T4 CPU已exit0，见[实际统计报告](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)。新的[52xy完整21τ×160位置三臂采集](../20261029/S2_SPATIAL52_DENSE_STUDENT_CAPTURE_RESULTS_20261029.md)也已双路exit0：2曝光诊断图、960包、60原生成状态逐字节中性；没有新视频decode。现在具备真实学生h/V1/Y/g/r/u，与旧Dense h重放分开。时间仍从属于S2统计维度，非第五方法。

**当前P0已取得的实际结果**：[spatial52采集](../20261029/S2_SPATIAL52_DENSE_STUDENT_CAPTURE_RESULTS_20261029.md)之后，[同h四角全160 L/P/I重放](../20261029/S2_SPATIAL52_FULL160_LPI_REPLAY_RESULTS_20261029.md)已双路实际exit0，640条唯一记录、额外12条控制均闭合。gY中央AC的P²约为L²的4.7倍，但signed cross显示强抵消；[独立投影评议](../method_discussion/discoveries/E0/s2-projected-prefix-interaction-independent-20261029.md)指出I可反映滤除的状态差，真实u/rowLN及逐site有反例，不作因果占比。[12张新开发图Dense＋2条中性对照](../20261029/S2_NEW12_DENSE_BASELINE_RESULTS_20261029.md)实际exit0、20项历史状态byte-neutral；[七源28页真实连续截图](../20261029/S2_CONSECUTIVE_SCREENSHOT_VISUAL_RESULTS_20261029.md)已实际CPU exit0并有独立AI评议，明确区分拍翼/鳍肢动作、相机平移、背景变化和仍保留运动的反例。下一项为固定物理行动的正常学生forward与真实消费者/完整视频判别，先冻结代码、匹配对照与成本，不继续调整上轮AC目标权重。

## 原始计算对象与已用数据

**新12来源四臂同机对照已闭合**：[56条原生生成](../20261029/S2_NEW12_SM120_FIXED_DEPTH_NATIVE_RESULTS_20261029.md)（含8条中性）及[48条研究视频八轴/连续RAFT/全81帧接触表](../20261029/S2_NEW12_SM120_FIXED_DEPTH_QUALITY_RESULTS_20261029.md)均实际EXIT0并目标端验证；三种低秩行动的来源内运动比值均值为0.258/0.275/0.290，未恢复全局运动。source979后10层Dense为0.929的反例。whole-FFN的[真实预GELU a桥接](../20261029/S2_WHOLE_FFN_NATIVE_BOUNDARY_BRIDGE_RESULTS_20261029.md)四视频/16包/40状态byte-neutral已完成，后续诊断未注册，按最新Chipmunk/TEAL共同P0降P1。

- Wan2.2 I2V A14B四步high/high/low/low，两个expert、各40 Block；每个四步视频实际160个FFN位置。原BF16 W1形状5120×13824、W2形状13824×5120，视频latent网格21×29×52（21≠四个去噪step）。
- [原BF16真实全深度数据](../20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.md)：8个历史暴露image clusters（6校准、2开发）×160位置×256同坐标行=1280包，包含h、W2输出、gate、pre/post residual；同step/block的gate/shift/scale在这些来源中相同。BF16无Q1/A1或Q2/A2 FP4量化。
- [shared/step-r960全80原BF16函数](../20261029/RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.md)：已导出，W2 per-token逻辑MAC占原W2约25.69%，**保留W1后整FFN逻辑MAC为62.85%**；[完整学生视频与实际wall已测](../20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)，官方质量未通过。保留完整export SHA和源身份。
- [HiF4旧连续tile](../20261025/E0_SHARED_TILE_PRE_W1.md)：旧2已暴露source×5位置，首尾token共同变化明显、中层很少；每项为同expert相邻去噪步变化，不能将raster相邻误当video运动配对。
- [HiF4 UB1 20来源视频与四步状态](../20261029/UB1_20SOURCE_FULL_DEPTH_VIDEO_STATE_EVIDENCE_RECOMPUTATION_20261029.md)：真实free video中运动幅值均值比0.1672、18/20下降，成像分数可以变好而动作退化；20/20四步signed误差累计。其numeric family与BF16不同，**不能用作BF16候选的Dense对照**。
- [真实PRO6000 NVFP4 FFN有限实验](../20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.md)已经执行；这建立硬件算子可行性证据，而不是NVFP4整个模型的生成质量或加速。静态权重与在线激活码及scale、producer、决策时点按[合同](WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)区分；旧HiF4 S1P2单码**含符号位**，fake-QDQ输出带符号BF16而非原生packed。

**当前GPU执行约束**：当前用户会话最多同时4卡，新范围在这一上限内采用动态实际空卡分配；每次启动先冻结ACTIVE的有限范围及实际可用资源。旧VIDEO20和近期两路采集原2卡scope保持不变，原RUN/EXIT/hash不改写。

## 实际研究动作、证明与权限

1. 从上面的资产中选一个**真实结构/矛盾**；读[原文98主＋20补充与G14落点](PAPER_METHOD_FOCUS_GRAPH.json)以及[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[web-research-discovery](../../.agents/skills/web-research-discovery/SKILL.md)对应工作方式。提出至少两种可被最小保存数据判别的机制，再决定算法与物理动作。
2. 写清source（独立统计单位）、step/expert/block/token、视频latent时间槽、观察时点和数值recipe。先用原始tensor/证明/已存RUN代替重复模型作业；新数据只采集能改变机制结论的缺口。
3. 若进入部署方法试验，冻结真实4step×40site候选、≥20独立开发image clusters目标、适用的动作/语义/连续运动/背景/时序/感知指标、Dense配对与FFN/整机时间；新盲确认按独立source/seed协议另外冻结。参见[视频合同](../20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。
4. 新CPU/GPU/model/video运行先写入[ACTIVE](../ACTIVE_RESEARCH.json)的source/配置/预算，遵循适用会话/冻结RUN的并发上限（实际VIDEO20为2卡），然后在边界内按[动态GPU政策](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)使用实际有效最大并行。新有限范围依当前持续授权登记和冻结实际分配；无实际调度收据不假定有空闲卡。
5. 使用[独立方法讨论队列](../method_discussion/QUEUE.md)登记**真实**不同Agent提出及评议的方法；历史B-H/B-F/A0与Affine54记录不改原身份。新论文/计算边按[图谱协作](GRAPH_AGENT_COLLABORATION.md)同步。

**下一Agent交付**：按[资产先行统一队列](ASSET_FIRST_AGENT_HANDOFF_20261029.md)完成其中一个已注册的真实机制判别；输出正常计算图、至少两种竞争解释、source-held结果或确切缺失资产、在线合法动作与代价、能覆盖160位置的视频及系统桥梁。新结论以原生证据更新现行文件，不另造重复的“权威进度”。
