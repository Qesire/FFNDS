# FFNDS · E0_OPEN_DISCOVERY阶段科研锚点

**当前科学问题：** 按[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)并行发展 **I/F/C：信息与表示、函数与动力学、计算与执行**。保留原生四步Wan证据锚点，允许源信息、算子代数、视频功能及系统成本独立形成新机制。

**统一资料**：[研究导航](../RESEARCH_NAVIGATION.md) · [当前ACTIVE状态](../ACTIVE_RESEARCH.json) · [研究上下文](CURRENT_CONTEXT_PACKET.md) · [G0–G13主图](../../docs/98_端到端计算图与证据注解_20261025.md) · [跨机制综合](../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[Function–Structure](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)为F线专门资料。[2026-10-25第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)与[早期本征重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)保留历史意义。

## 科研材料与可比较的结构

原生 LightX2V Wan2.2 I2V A14B、4step D-Q、high/high/low/low两套expert、40个blocks，HiF4 fake-QDQ+BF16 GEMM。重要实际路径包括\`norm2/AdaLN→Q1→W1→GELU→Q2(64→8→4共享尺度)→W2→e5门控写回→后续attention/flow/scheduler\`。

已完成的10个source身份、12条source×seed完整视频及1920个嵌套site显示几种互相连接的科学事实：
- 局部FFN误差与完整视频质量存在不同响应；旧Affine近似在真实运行中有局部速度收益及视频背景/光照时序失配。[原生审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)
- [完整有限状态对照](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)已测\(P=T_D(z_S)-T_D(z_D)\)及\(N=T_S(z_S)-T_D(z_S)\)，用于区分历史状态传递与当前候选注入。P23/P24的整体观测身份是NATIVE-OBS，具体神经元/结构因果角色仍是OPEN。
- [四新来源交叉seed](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)显示source425两seed后三步P/N cross符号相反，而局部raw误差位置计数相同。这一现象推动条件性、联合输出函数研究。
- [全hidden和共享tile](../20261025/E0_SHARED_TILE_PRE_W1.md)说明后验逐token集中度与当前kernel合法执行颗粒不同；[HiF4尺度依赖](../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)揭示有符号功能必须跨Q2组考虑。

## 独立探索与证据升级

**I线，信息与表示**：研究输入/weight/码字/历史状态的首次可得时间、信息价和共享编码，探索可以单独被证明的界与执行策略。

**F线，函数与动力学**：从视频、latent/flow与无标签signed响应探索条件性生成行为，再用G13定位W1/GELU/Q2/W2与下游attention/step的真实承担结构；区分观察、因果及部署效果。

**C线，计算与执行**：从合法tile、GEMM/QDQ、kernel访存和量化/训练后表示研究低成本实现。可以先证明成本或实现优势，再把结果反馈至I/F线，不需要先完成函数定位。

**共同方法**：使用[AutoResearcher领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)及[独立讨论](../method_discussion/AGENTS.md)，取最有辨识力的源码/数学/论文与已有数据，建立最强竞争机制及不同预测；图注解不规定研究范围。

## 状态、质量和交接

**当前科学活动（2026-10-27恢复）**：用户明确要求持续推进I/F/C研究及登记后的有限实验，同时最多使用四张空闲GPU，简单委派使用GPT‑6.1 Terra。[ACTIVE](../ACTIVE_RESEARCH.json)控制冻结范围与真实状态；先前paired-native、历史暂停和RUN保持原样，实时/流式实验未恢复。

新模型或视频方法的质量需覆盖40层×4step、自由视频主体/背景/时序表现、same-source/seed比较及实际cost/SLO（见[统一指标协议](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)）。纯理论和已测观察以各自数学/来源证据交付。

**交接优先顺序**：[统一I/F/C目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) → [已有真实证据包](CURRENT_CONTEXT_PACKET.md) → [主图](../../docs/98_端到端计算图与证据注解_20261025.md) → [ACTIVE](../ACTIVE_RESEARCH.json)。留下有价值的信息、函数或执行机制、竞争预测及其证据身份。
