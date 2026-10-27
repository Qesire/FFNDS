# E0 · 真实FFN输出函数与计算结构的定位任务

> **当前优先级更新（2026-10-27）：** [G13行为→FFN输出函数→真实计算结构研究](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[行为/函数/结构三类对象分开登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[G0–G13源码/文献/功能图](../../docs/98_端到端计算图与证据注解_20261025.md)。首先确定哪些输出函数在自由生成视频/flow中具有条件性行为作用，随后找出正常前向中承担这些函数的真正计算结构；量化code、mask、proxy及核实现是**此后**的研究空间。

**已知证据**：10个source、12条source×seed轨迹、1920个嵌套FFN site已完成。已有局部FFN误差/视频失配、相同prompt跨seed有符号作用改变的现象，但尚未用新的组件级因果干预识别具体W1行/组的函数身份。先复用原生数据的已发表解释、真实源码依赖与方法文献，再设计最小辨识实验；**所有新的CPU/GPU模型与视频实验仍按用户指令暂停**，本文不授权运行。

**研究交付**：FUNCTIONAL-OBS标记有条件的自然输出响应/视频现象；FUNCTIONAL-CAUSAL通过未来单独获准的合法图边干预证明实际行为作用；STRUCTURE记录实现这些函数的W1/GELU/HiF4 Q2/W2联合非线性结构、attention与跨层/step依赖；FUNCTIONAL-IMPLEMENTABLE最终还要证明物理计算可减少且端到端质量可接受。G13 L80–L85提供causal abstraction、activation patching边界、video DiT运动定位和时空SAE时间完整性等方法学近邻。

**历史说明**：2026-10-25/26本征探索、旧33→51文献初查、早先6source/4source扩展和实时场景暂停的计划属于当时版本；其source/算子/实验报告可继续用于比较，不改变当前G13第一目标、已完成12轨迹状态和实时文献研究已恢复的事实。

## 目标与研究对象

在Wan2.2 I2V四步蒸馏模型中减少真实FFN计算、减轻完整生成成本，保持主体、运动、纹理、光照与时序表现。研究对象按真实W1→tanh-GELU→Q2/postQ2→W2→gate/residual→后续层→flow/scheduler/视频计算流定位。[FFN本征研究纲领](E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[源图G0–G6](../../docs/98_端到端计算图与证据注解_20261025.md)与[AutoResearcher Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)共同提供可扩展基础。

历史科学分歧与未闭合机制分类仍可按需追溯[独立分歧审查](E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md)，它是证据和思路来源，不再是当前判断入口。

已有证据的精确数据与边界，统一进入[科学审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[实验发现图谱](../EXPERIMENT_FINDINGS_ATLAS.md)。**研究张力：** Affine54已实现局部SSE改善与原生FFN加速，而固定160位置全层替换产生了明显视频失配。由此打开部署数值、函数容量、受扰前缀、下游有符号传播和调度状态之间的机制识别问题；Affine54是可复用的判别资产，新的函数表示与算子构想同样作为独立科学方向。

**最新目标尺度（2026-10-26）**：[扩大source与signed方向评价](../20261026/E0_MULTISOURCE_DIRECTIONAL_EVALUATION_20261026.md)是现行跨源机制与指标设计：既有0/12之外补4/8/16/20的原生范围（含既有校准样本标注），并追求至少四个真正未暴露来源与交叉seed。以signed P/N/cross、互斥频带、全帧质量向量和实际省算判别新机制；全局SSE维持辅助观测。方案已登记，这些来源在2026-10-27已完成对应固定范围的原生验证并暂停新执行，见[四新来源六条路径](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。

**当前首要科学步骤（机制优先、2026-10-26）**：[前沿方法学再审与FFN三个发现问题](../20261026/E0_FRONTIER_MECHANISM_FIRST_20261026.md)。先研究真实FFN计算为何产生某些功能、什么条件下可重组/共享/预测，构造相互区分的结构假说；再选择相应的干预和多来源验证。Affine54的局部/全局SSE、signed能量及频带只是已有原型的仪表，扩展640 site不是独立科研目标。

**新完成工作的研究判别（2026-10-26）**：[F1固定四坐标＋F1邻居外推／原生packed基元证据分级](../20261026/E0_NEW_WORK_MECHANISM_AUDIT_20261026.md)。现在需要区分anchor自身解析非线性与它对neighbor12的可迁移预测；同时将SM120硬件能力留作可执行格式比较，而非默认NVFP4稀疏算法。上述是平行的研究方向之一；F2的条件不变性与F3的原生视频功能继续开放。单纯增加多项式秩或固定FP4算子并非必要下一任务。

## 并行而互补的研究任务

### A · 解释已测原型为何失配

在**实际源码/参数/条件分支**下，区分部署数值NUM、函数容量CAP、真实Student前缀SHIFT、下游有符号传递TRANS及四步调度状态STATE。**已得到两来源320处同输入四臂/写回及四步状态原生数据**；现在先复用[完整结果](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)中的有符号观察区分竞争预测，再选择价值最高的有限验证。单site四臂包括 `f_D(x_D)`、`f_S(x_D)`、`f_D(x_S)`、`f_S(x_S)`；另测真实gate/residual、block、flow与scheduler `step_pre/post`。现有两个已暴露来源已覆盖每来源40层×四步160个site；下一步应重视函数和状态因果分解、外部来源泛化与真实时空分层。相关受控后缀试验须由实际观测选点，再独立冻结范围。

**当前工具**：[原生paired采集与中性对照](../20261026/E0_PAIRED_NATIVE_IMPLEMENTATION.md)、[新严格视频身份与清单校验](../20261026/E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)、[有符号数据账本](../20261026/E0_FULL_DEPTH_TRACE_TOOLING.md)。**v2的DD中性视频及后处理失败**与**v3的320处完整原生记录**已分别归档；可从后一结果提出两种流差的可区分分解：`v_S(z_S)-v_D(z_D)=[v_S(z_S)-v_D(z_S)]+[v_D(z_S)-v_D(z_D)]`。执行状态由ACTIVE判定；该完整Dense受扰前缀反事实已完成并复核，见[全网络有限响应](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)。

### B · 发现不同的低成本FFN机制

可从信息何时可得、W1/GELU/W2 signed函数值、奇偶/条件非线性、时空共享与局部独有创新、层/专家/step差异、可执行GEMM/tile/布局、量化与轻量后训练出发。**研究目标是必要FFN功能与可节省物理计算之间的新联系。** Affine、group64、前置代理、掩码、补算、cache、其它FP4格式以及尚未命名的数学基均是可展开并相互竞争的候选表达与动作。

每项方法讨论交付：**新结构/问题重构 → 实际数学与源码依据 → 与最强近邻/旧方法不同的预测 → 最低成本有辨识力的证明/复用实验 → 可能的实际省算与终端质量路线**。多Agent只在方法讨论中独立提出、质疑与互相审查，Graph-Agent将已证实/争议/OPEN命题回写到源图，不替代实验授权或强制指定研究方向。

## 每次生成的质量与物理验收

任何新Dense或候选视频必须满足[统一指标与全层协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)：真实视频及source/seed/model/协议/原生回执SHA、预登记解码帧数与完整逐帧核心RGB/亮度/细节/时序/运动代理、静态/背景区分、FFN与full wall/setup/显存/存储费用；候选必须绑定真正Dense对应视频及身份。新paired入口使用严格原生视频inventory检查；历史候选四个dense身份元数据错误使用冻结勘误，不改原报告。

正式VBench-I2V/VBench2和人工全片没做时必须标`NOT_RUN`、`quality_accepted=false`；工程proxy不能直接证明视频保真，速度成功也不代表方法验收。

## 实时场景与职责边界

G6原始实时场景在2026-10-26曾暂停；2026-10-27用户已另行恢复[G10实时论文/计算图研究](../literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)。本轮可继续论文和数学分析，**原生流式视频/CPU/GPU模型实验仍暂停**；G13首先定位函数和结构，真实服务SLO属于后续运行时评价。

## 下一次交接读取顺序

[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)（唯一状态/权限）→ [统一科学结论](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)（证据解释）→ [v3配对原生结果](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)（最新实测）→ [完整指标政策](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md) → [实验发现图谱](../EXPERIMENT_FINDINGS_ATLAS.md)或相关源图节点。只打开与当前科学问题真正相关的旧实验与独立评议。
