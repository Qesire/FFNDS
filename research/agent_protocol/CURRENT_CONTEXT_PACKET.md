# FFNDS · 当前科研最小上下文（E0_OPEN_DISCOVERY · 2026-10-27）

**共同科学问题**：生成FFN的信息、计算与视频行为如何组织？如何发现更廉价的表示、计算或执行？I（信息/表示）、F（函数/动力学）、C（计算/执行）并列推进。参考[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)；G13的[行为/函数/结构](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)仅用于F线具体定位。

**研究方法**：[领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[AutoResearcher机制研究](../../.agents/skills/mechanism-research/SKILL.md)、[扩展探索](../../.agents/skills/research-expansion/SKILL.md)、[问题重构](../../.agents/skills/problem-reformulation/SKILL.md)及[深度论文/源码检索](DEEP_WEB_RESEARCH.md)。科研成果由新的现象/函数/结构、关键数学或源码、竞争解释和可区分预测组成。

**当前状态**：[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 为科学阶段、模型执行授权、历史RUN和质量身份来源。2026-10-27用户明确恢复持续推进I/F/C研究及冻结登记后的有限实验，同时最多使用四张空闲GPU，简单委派使用GPT‑6.1 Terra。原E0固定批次和历史暂停保持原始身份；此上下文包仅提供导航，具体范围在ACTIVE登记，实时/流式实验未恢复。

## 真实计算对象

**LightX2V Wan2.2 I2V A14B，四步D-Q，high/high/low/low两套独立expert，40 blocks**。单FFN实际顺序：
\`cross-attention residual → norm2/AdaLN → Smooth1/Q1 → W1 → GELU(tanh) → Smooth2/HiF4 Q2(64→8→4) → W2 → e5 gated BF16 residual → 后续block/flow/scheduler\`。

原路径使用HiF4 fake-QDQ + BF16 Linear；[源码/数值图G0–G13](../../docs/98_端到端计算图与证据注解_20261025.md)追踪真实输入、张量和消费者，[机器节点与边](GRAPH_AGENT_BINDINGS.json)负责有类型证据索引。

## 已完成的最有信息价值的原生证据

- **实验覆盖**：10个独立source身份、12条source×seed完整生成轨迹、1920个嵌套FFN观测位置。质量是否验收取决于原始视频/指标RUN及[质量规则](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)。
- **局部函数与终端行为不一致**：[已有原型综合评议](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)包含Affine54局部拟合/加速和背景、光照及细小物体时序失配，提示研究具有行为意义的联合FFN功能。
- **条件性的有符号完整响应**：[六source](../20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)与[四新source×seed配对](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。来源425在相同图/提示、不同seed下，后3步signed P/N cross方向改变，而FFN raw局部误差位置计数相同。
- **有限整步分解**：[完整dense状态反事实](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)已测 \(P=T_D(z_S)-T_D(z_D)\)、\(N=T_S(z_S)-T_D(z_S)\)及其和，区分继承状态传播与候选当前注入。
- **时空/物理结构**：[保存状态频带](../20261026/E0_SAVED_FLOW_SPECTRUM.md)出现不同范围signed cross异号；[全hidden及真实共享tile](../20261025/E0_SHARED_TILE_PRE_W1.md)揭示逐token后验选择与合法GEMM执行单位不同；[HiF4组尺度源码数学](../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)说明部分隐藏坐标对其它坐标存在量化尺度依赖。

这些证据定位了具有科学价值的现象与真实计算依赖；具体的FFN行为函数、联合承担路径和compensation机制保留为 G13 FUNCTIONAL-OBS/OPEN/ALG 的可检验研究对象。现有P23/P24为已完成的 **NATIVE-OBS**，具体组件因果归属仍OPEN。

**本轮的新判别**：[共享输入联合支持/精确投影比较器](../20261027/E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)为ALG＋604有限排列；[原M Q2直接/补集](../20261027/E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)及[W2形状控制](../20261027/E0_W2_SHAPE_CONTROL_RESULTS.md)为保存操作数NATIVE-OBS，未知行填零、无后缀/新视频。用户另要求的[UB1静态全帧复核](../20261027/UB1_INDEPENDENT_VISUAL_REVIEW.md)有原视频SHA及三项历史逐视频评分，属于LLM视觉观察。

**新增论文与算子判断（2026-10-27，LIT/SRC；尚无新Wan 2:4运行）**：[RT-Lynx原文深审](../20261027/RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)确认其输入侧activation 2:4 + norm + rank64 LoRA + SpTC核，**不是weight删列或whole64通道剪枝**。强制预训练weight 2:4可能破坏跨输入长期容量；输入活动接近零不等于neuron语义证明。原E0的64组/tile全满不能排除2:4 SpTC。对Wan分别考察**Q1后W1输入**及**Q2后W2输入**：后一种只可能省W2乘加、仍全付W1/GELU/Q2；真实HiF4/BF16与A100执行、视频质量保持OPEN。由[研究导航](../RESEARCH_NAVIGATION.md)进入[UB1](../20261027/UB1_STEP_BUDGET_CALIBRATION_AND_VIDEO_GENERALIZATION_RESEARCH_PROPOSAL_20261027.md)、[Affine54](../20261027/QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)的研究计划，不自动授权新的GPU/训练。

## 并行可发展的机制问题

**I · 信息与表示**：从Q1输入何时可得、权重/码字、共享比例尺、已有latent/历史状态和有符号约束出发，探索新信息界、等价代数与信息获取净价值；允许先产生直接编码或可证明省算机制。

**F · 函数与动力学**：从真实source/seed/step视频、无标签signed共变和下游状态变化提出行为/函数候选，定位真实共享q的W1、GELU、HiF4 Q2、W2 gate、attention和专家/步间的正常贡献；登记观察与因果证据。

**C · 计算与执行**：从实际M/D/H、物理tile占用、QDQ/packed内核、内存访问、延迟和训练预算出发发明可执行算子或重组函数，再由质量/速度和可能的信息/功能解释反馈研究。

三类研究都可参考[G7–G13综合](../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)和[89项原文](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)。它们共享真实原生证据，主张分别验证，无固定先后。

## 有效科研交接

从[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)与[研究导航](../RESEARCH_NAVIGATION.md)选择最有信息量的I/F/C科学分歧；[2026-10-25第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)保留为历史阶段证据。给出新发现与数学/源码证据、最强竞争解释及不同预测、下一条最经济的可判别来源/证明/未来干预。沿[Graph-Agent协作](GRAPH_AGENT_COLLABORATION.md)登记新函数及OPEN图边，独立同行评议进入[讨论目录](../method_discussion/discoveries/E0/README.md)。

历史原生观察、早期[研究重置纲领](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)和各次源审计仍作为有版本的研究证据。新文献可从经典与最新学科自由扩展，进入新数学表达和新有类型图节点。
