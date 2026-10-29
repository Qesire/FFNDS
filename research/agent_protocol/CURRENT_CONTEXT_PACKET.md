# FFNDS · 当前Agent最小上下文（2026-10-29）

**唯一当前路线**：[研究导航](../RESEARCH_NAVIGATION.md) → [资产先行四路线推导](../20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md) → [现行执行队列](ASSET_FIRST_AGENT_HANDOFF_20261029.md)。此文件只是接手所需的事实与状态入口，完整数学在对应方法文档、真实作业状态在[ACTIVE](../ACTIVE_RESEARCH.json)。[M-A/M-B/M-C旧整合方案v4](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)和[B专题旧交接](NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)为有用的专题与历史来源，不再规定默认“先末层”顺序。

**新增最高优先级规则：发现缺口就补采**。参见[开放式科学采集协议](EVIDENCE_DRIVEN_ACQUISITION_20261029.md)、[机器采集队列](ACQUISITION_QUEUE_20261029.json)：新source/新的scene stratum、4×40全深度time×space tubelet、native Q1/Q2 activation metadata、student前缀状态、视频动作与真实SYS等可主动补采，旧8来源、5 Block及现有算法仅为已知证据。先采用现有数据满足已有科学问题；一旦现有观测不足，转为新producer/数据集/新采集，而不是将问题降格为旧数组可以算的proxy。按每次ACTIVE冻结有限scope并可持续扩展下一批。

## 科学目标和新方法

研究目标：在保留Wan2.2 I2V生成视频主体动作、语义、背景、相机、时间行为和感知质量的前提下，发现FFN在信息、函数、真实执行上的低成本组织规律。以**实际源已核的结构发现**作为起点，提出相互竞争且可证伪的解释，再设计在线可用、物理可执行的新行动。SSE、有符号抵消、范数补偿、补算是解释和对照工具，不是默认研究目标。

- **S1-CSFB**：共享跨step的函数基＋适量step收缩特化。来自真实BF16 r960：shared在6来源留一的8函数锚点6/6更优、2旧开发来源2/2更优；但训练集上step特化更准。**新收缩拟合与全视频效果待测**。
- **S4-FRANK**：利用原BF16真实h和W2构造能在W2前评估额外低秩块收益的低价草图；该方法在原W1/GELU后决策，最多减少W2计算。**数学投影性质已推导，真实source-held regret与部署成本未测**。
- **S2-FTFD**：从旧HiF4连续tile在5层首中尾的创新差异出发，发现真实视频latent时间×空间的共享函数计算单元。**已有[八来源全160处26389个稀疏时间对观察](../20261029/RAW_BF16_S2_TEMPORAL_PAIR_OBS_RESULTS_20261029.md)；相同xy优势已出现在V1。[十张新开发图完整21τ、8xy、40层×4步采集及CPU观察](../20261029/RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)已实际完成：1600包/全部u字节neutral；[同Dense h固定因子复算](../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)显示shared门后输出DC误差0.0293、AC误差0.1842，小M舍入不足解释。下一具体构造是同数据/秩普通目标与DC/AC平衡目标，不重复旧视频或直接拷邻居**。
- **S3-QFAR**：以HiF4 whole64组闭包和W1前可用的信息为真实合法行动集合。旧任意列25%保留触及216/216组，whole64对角25%保留在所测创新上表现差；**新的条件有符号联合组行动及原生NVFP4独立闭包待判别**。

四支路线并行，任意Agent可以根据真实源与论文提出更优竞争机制。第一项**成本最小且可执行的完整视频基线**是已导出的原BF16 shared80-r960、step160-r960和相同原BF16 Dense的真实160位置同source生成；这项基线已完成[20来源60视频及单次成本](../20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)和[八轴官方函数/连续RAFT](../20261029/RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.md)，当前质量未通过：shared/step来源等权RAFT幅值为Dense的0.490/0.477。公平预热重复SYS仍待测；不用重复已完成视频。

**S2时间统计子问题 MTC/TDIC（非并行研究方向）**：[时间距离与FFN运动创新补偿](../20261029/TEMPORAL_DISTANCE_MOTION_COMPENSATION_FFN_HYPOTHESES_20261029.md)及[OPEN图claim](../method_discussion/graph_annotations/E0/E0_GRAPH_MTC_TDIC_TEMPORAL_INNOVATION_20261029.json)。从已有20来源视频r960候选运动幅值约0.49/0.48倍Dense、8来源26389稀疏same-xy跨τ配对出发，优先做**同Dense h/同原BF16 W2 bias和gate的DC/AC重放判别**，再扩展真实邻域/光流oracle与学生前缀。核心恒等式 eτp−eτ−1,q = gτp⊙Δh(W−UV)+(gτp−gprev,q)⊙hprev,q(W−UV) 为连续代数，不是BF16逐bit。teacher残差锚点/前τ输出在线并不免费；旧固定8xy不支持任意运动搜索。相关缺口已合并到[机器采集队列](ACQUISITION_QUEUE_20261029.json)的 `AQ-S2-TIME-TUBELET` 子任务；新增滞后/频谱/Gram统计为CODE_PREPARED_NOT_EXECUTED，无新增GPU或视频许可。该方向不改变S1/S4/S2/S3的并行资格，不视作同行批准。

**当前S2研究已扩展为时间统计结构工作：** [T1–T6时序统计、原生来源扩张与S2执行顺序](../20261029/S2_TEMPORAL_STATISTICAL_STRUCTURE_AND_DATA_EXPANSION_20261029.md)。原十图21τ×8xy×160位置与同Dense h低秩因子E1/E2均已实际完成，不能再称它们“运行中”；下一缺口为相关函数/谱/时间Gram/相位/突发运动及真实连续空间邻域和学生状态。MTC只保留可选的代数/条件表示注释，不独占并行方法队列或单独GPU作业身份。任何新统计运行和扩源需新ACTIVE范围；已记录两卡会话限制在对应执行时有效。

## 原始计算对象与已用数据

- Wan2.2 I2V A14B四步high/high/low/low，两个expert、各40 Block；每个四步视频实际160个FFN位置。原BF16 W1形状5120×13824、W2形状13824×5120，视频latent网格21×29×52（21≠四个去噪step）。
- [原BF16真实全深度数据](../20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.md)：8个历史暴露image clusters（6校准、2开发）×160位置×256同坐标行=1280包，包含h、W2输出、gate、pre/post residual；同step/block的gate/shift/scale在这些来源中相同。BF16无Q1/A1或Q2/A2 FP4量化。
- [shared/step-r960全80原BF16函数](../20261029/RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.md)：已导出，W2 per-token逻辑MAC占原W2约25.69%，**保留W1后整FFN逻辑MAC为62.85%**；[完整学生视频与实际wall已测](../20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)，官方质量未通过。保留完整export SHA和源身份。
- [HiF4旧连续tile](../20261025/E0_SHARED_TILE_PRE_W1.md)：旧2已暴露source×5位置，首尾token共同变化明显、中层很少；每项为同expert相邻去噪步变化，不能将raster相邻误当video运动配对。
- [HiF4 UB1 20来源视频与四步状态](../20261029/UB1_20SOURCE_FULL_DEPTH_VIDEO_STATE_EVIDENCE_RECOMPUTATION_20261029.md)：真实free video中运动幅值均值比0.1672、18/20下降，成像分数可以变好而动作退化；20/20四步signed误差累计。其numeric family与BF16不同，**不能用作BF16候选的Dense对照**。
- [真实PRO6000 NVFP4 FFN有限实验](../20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.md)已经执行；这建立硬件算子可行性证据，而不是NVFP4整个模型的生成质量或加速。静态权重与在线激活码及scale、producer、决策时点按[合同](WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)区分；旧HiF4 S1P2单码**含符号位**，fake-QDQ输出带符号BF16而非原生packed。

**当前GPU执行约束**：遵循本会话及实际ACTIVE/RUN中两卡并发上限（已完成VIDEO20等范围写明 `session_max_GPUs=2`）；在有效两卡约束内按调度器实际可用资源优化，单个分片通常1卡。通用动态GPU原则只是规划默认值，不能覆盖冻结执行记录。变更约束须先明确新授权/新协议范围，原RUN/EXIT/hash保持不变。

## 实际研究动作、证明与权限

1. 从上面的资产中选一个**真实结构/矛盾**；读[原文98主＋10补充与G14落点](PAPER_METHOD_FOCUS_GRAPH.json)以及[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[web-research-discovery](../../.agents/skills/web-research-discovery/SKILL.md)对应工作方式。提出至少两种可被最小保存数据判别的机制，再决定算法与物理动作。
2. 写清source（独立统计单位）、step/expert/block/token、视频latent时间槽、观察时点和数值recipe。先用原始tensor/证明/已存RUN代替重复模型作业；新数据只采集能改变机制结论的缺口。
3. 若进入部署方法试验，冻结真实4step×40site候选、≥20独立开发image clusters目标、适用的动作/语义/连续运动/背景/时序/感知指标、Dense配对与FFN/整机时间；新盲确认按独立source/seed协议另外冻结。参见[视频合同](../20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。
4. 新CPU/GPU/model/video运行先写入[ACTIVE](../ACTIVE_RESEARCH.json)的source/配置/预算，遵循适用会话/冻结RUN的并发上限（实际VIDEO20为2卡），然后在边界内按[动态GPU政策](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)使用实际有效最大并行。其他新的执行范围若需不同并发须独立授权和冻结；无实际调度收据不假定有空闲卡。
5. 使用[独立方法讨论队列](../method_discussion/QUEUE.md)登记**真实**不同Agent提出及评议的方法；历史B-H/B-F/A0与Affine54记录不改原身份。新论文/计算边按[图谱协作](GRAPH_AGENT_COLLABORATION.md)同步。

**下一Agent交付**：按[资产先行统一队列](ASSET_FIRST_AGENT_HANDOFF_20261029.md)完成其中一个已注册的真实机制判别；输出正常计算图、至少两种竞争解释、source-held结果或确切缺失资产、在线合法动作与代价、能覆盖160位置的视频及系统桥梁。新结论以原生证据更新现行文件，不另造重复的“权威进度”。
