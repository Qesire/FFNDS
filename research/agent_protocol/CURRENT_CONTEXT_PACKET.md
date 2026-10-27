# FFNDS · 当前科研最小上下文（E0_OPEN_DISCOVERY · 2026-10-27）

**首要科学问题**：在四步Wan视频生成中，哪些FFN输出函数真正对生成行为有意义？由哪些正常计算结构承担它们？参考[G13功能定位与结构研究](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)和[BEHAVIOR/FUNCTION/STRUCTURE独立登记](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)。

**研究方法**：[领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[AutoResearcher机制研究](../../.agents/skills/mechanism-research/SKILL.md)、[扩展探索](../../.agents/skills/research-expansion/SKILL.md)、[问题重构](../../.agents/skills/problem-reformulation/SKILL.md)及[深度论文/源码检索](DEEP_WEB_RESEARCH.md)。科研成果由新的现象/函数/结构、关键数学或源码、竞争解释和可区分预测组成。

**当前状态**：[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 为科学阶段、模型执行授权、历史RUN和质量身份来源。目前允许阅读源码/文献、数学推导、已保存证据解释、图与文档维护；**新的CPU/GPU/模型/视频实验暂停**。原E0固定实验批次均已完成；此上下文包仅提供科研导航，不建立执行任务。

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

## 研究可发展的结构问题

**BEHAVIOR → FUNCTION**：根据自然source/seed/step视频行为及无标签signed共变，寻找条件性输出函数（主体、运动、背景/光照、时间稳定或尚未命名的关系）。

**FUNCTION → STRUCTURE**：从真实共享q的W1行、GELU、HiF4 Q2原group64依赖、W2有符号合成、门控、attention/后续层及high-low跨step递推找到实际承担者。研究联合作用、跨路径补偿、正常前向与有限干预的差异；按具体证据逐渐提高从观察到因果的可信度。

**STRUCTURE → IMPLEMENTATION**：以[G7–G12机制比较](../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)及[85项文献原文深度索引](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)探索训练分工、历史复用、时空分块、量化符号/尺度、LUT或物理稀疏；后续信息可用时间、付费代价和终端视频表现由方法决定。

## 有效科研交接

从[当前第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)与[研究导航](../RESEARCH_NAVIGATION.md)选择最有信息量的科学分歧。给出新发现与数学/源码证据、最强竞争解释及不同预测、下一条最经济的可判别来源/证明/未来干预。沿[Graph-Agent协作](GRAPH_AGENT_COLLABORATION.md)登记新函数及OPEN图边，独立同行评议进入[讨论目录](../method_discussion/discoveries/E0/README.md)。

历史原生观察、早期[研究重置纲领](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)和各次源审计仍作为有版本的研究证据。新文献可从经典与最新学科自由扩展，进入新数学表达和新有类型图节点。
