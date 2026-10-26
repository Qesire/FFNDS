# E0 · 当前研究任务：真实FFN必要计算与完整视频功能

**当前权威机制与方向解释**：[2026-10-26统一科学审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)。本页只定义**研究目标、任务分工和交付**，不重复过时的实验进度快照；当前阶段、用户授权、模型运行进度只以[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)及实际RUN为准。

## 目标与研究对象

在Wan2.2 I2V四步蒸馏模型中减少真实FFN计算、减轻完整生成成本，保持主体、运动、纹理、光照与时序表现。研究对象按真实W1→tanh-GELU→Q2/postQ2→W2→gate/residual→后续层→flow/scheduler/视频计算流定位。[FFN本征研究纲领](E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[源图G0–G6](../../docs/98_端到端计算图与证据注解_20261025.md)与[AutoResearcher Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)共同提供可扩展基础。

已有证据的精确数据与边界，统一进入[科学审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[实验发现图谱](../EXPERIMENT_FINDINGS_ATLAS.md)。请注意：Affine54局部SSE改善与原生FFN加速均为实测积极结果；当前两个来源固定160位置全层替换的视频质量不符合研究目标。局部teacher输入SSE不是视频保真充分条件，尚不知它是否是主导损失原因；**这既不是对整个Affine族的不可行定理，也不是给固定54组重调参的指令**。

## 并行而互补的研究任务

### A · 解释已测原型为何失配

在**实际源码/参数/条件分支**下，区分部署数值NUM、函数容量CAP、真实Student前缀SHIFT、下游有符号传递TRANS及四步调度状态STATE。先复用保存数据与代数验证，逐步用同输入和student双轨原生观测排除竞争预测。单site四臂包括 `f_D(x_D)`、`f_S(x_D)`、`f_D(x_S)`、`f_S(x_S)`；另测真实gate/residual、block、flow与scheduler `step_pre/post`。对整个全层方法以40层×四步、不同来源与真实时空分层为覆盖单位；不以首末层或历史Block0/30的partial-native代替全160个site。相关受控后缀试验须由实际观测选点，再独立冻结范围。

**当前工具**：[原生paired采集与中性对照](../20261026/E0_PAIRED_NATIVE_IMPLEMENTATION.md)、[新严格视频身份与清单校验](../20261026/E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)、[有符号数据账本](../20261026/E0_FULL_DEPTH_TRACE_TOOLING.md)。固定v2范围若已在运行，仅跟进其注册protocol和真实RUN；**没有新的自动重试、参数扩展或模型训练权限**。源码软件测试≠原生模型验证。

### B · 发现不同的低成本FFN机制

可从信息何时可得、W1/GELU/W2 signed函数值、奇偶/条件非线性、时空共享与局部独有创新、层/专家/step差异、可执行GEMM/tile/布局、量化与轻量后训练出发。Affine、group64、前置代理、掩码、补算、cache和其它FP4格式都仅是候选**表达与动作**，不是项目目标；不要求先解决Affine54才能提出新理论或新算子。

每项方法讨论交付：**新结构/问题重构 → 实际数学与源码依据 → 与最强近邻/旧方法不同的预测 → 最低成本有辨识力的证明/复用实验 → 可能的实际省算与终端质量路线**。多Agent只在方法讨论中独立提出、质疑与互相审查，Graph-Agent将已证实/争议/OPEN命题回写到源图，不替代实验授权或强制指定研究方向。

## 每次生成的质量与物理验收

任何新Dense或候选视频必须满足[统一指标与全层协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)：真实视频及source/seed/model/协议/原生回执SHA、预登记解码帧数与完整逐帧核心RGB/亮度/细节/时序/运动代理、静态/背景区分、FFN与full wall/setup/显存/存储费用；候选必须绑定真正Dense对应视频及身份。新paired入口使用严格原生视频inventory检查；历史候选四个dense身份元数据错误使用冻结勘误，不改原报告。

正式VBench-I2V/VBench2和人工全片没做时必须标`NOT_RUN`、`quality_accepted=false`；工程proxy不能直接证明视频保真，速度成功也不代表方法验收。

## 实时场景与职责边界

G6实时/流式生成已由用户暂停，只保留[历史独立场景文档](../20261026/E0_REALTIME_GENERATION_SCENARIO_20261026.md)作研究资产；**当前不布置实时文献、SLO、架构或GPU任务**。所有新计算必须遵循活动状态中的明确授权；本页没有单独的模型执行权。

## 下一次交接读取顺序

[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)（唯一状态/权限）→ [统一科学结论](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)（证据解释）→ [paired原生结果/回执](../20261026/E0_PAIRED_NATIVE_IMPLEMENTATION.md)（具体注册范围）→ [完整指标政策](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md) → [实验发现图谱](../EXPERIMENT_FINDINGS_ATLAS.md)或相关源图节点。只打开与当前科学问题真正相关的旧实验与独立评议。
