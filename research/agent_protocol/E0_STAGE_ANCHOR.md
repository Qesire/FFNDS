# FFNDS · E0_OPEN_DISCOVERY阶段科研锚点

**当前科学问题：** [G13视频FFN输出函数与真实计算承担结构](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)。先发现哪类自然行为/无标签有符号输出函数需要解释，再定位由正常执行中的哪些W1/GELU/HiF4-Q2/W2/attention/step子图承担；新函数/结构及竞争机制可随原论文与原生证据继续扩展。

**统一资料**：[当前第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) · [Function–Structure registry](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · [主计算图G0–G13](../../docs/98_端到端计算图与证据注解_20261025.md) · [Graph-Agent双向协议](GRAPH_AGENT_COLLABORATION.md) · [85项论文索引](../literature/20261026/PRIMARY_LITERATURE_INDEX.json) · [当前ACTIVE研究状态](../ACTIVE_RESEARCH.json)。[早期FFN本征重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)与[2026-10-26机制审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)保留其已测研究线索。

## 科研材料与可比较的结构

原生 LightX2V Wan2.2 I2V A14B、4step D-Q、high/high/low/low两套expert、40个blocks，HiF4 fake-QDQ+BF16 GEMM。重要实际路径包括\`norm2/AdaLN→Q1→W1→GELU→Q2(64→8→4共享尺度)→W2→e5门控写回→后续attention/flow/scheduler\`。

已完成的10个source身份、12条source×seed完整视频及1920个嵌套site显示几种互相连接的科学事实：
- 局部FFN误差与完整视频质量存在不同响应；旧Affine近似在真实运行中有局部速度收益及视频背景/光照时序失配。[原生审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)
- [完整有限状态对照](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)已测\(P=T_D(z_S)-T_D(z_D)\)及\(N=T_S(z_S)-T_D(z_S)\)，用于区分历史状态传递与当前候选注入。P23/P24的整体观测身份是NATIVE-OBS，具体神经元/结构因果角色仍是OPEN。
- [四新来源交叉seed](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)显示source425两seed后三步P/N cross符号相反，而局部raw误差位置计数相同。这一现象推动条件性、联合输出函数研究。
- [全hidden和共享tile](../20261025/E0_SHARED_TILE_PRE_W1.md)说明后验逐token集中度与当前kernel合法执行颗粒不同；[HiF4尺度依赖](../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)揭示有符号功能必须跨Q2组考虑。

## 独立探索与证据升级

**从行为观察到函数**：整理完整视频的运动、相机、主体、照明、背景细节及无标签signed/时空响应，在[Function registry](FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)建立带source、seed、expert、step的FUNCTIONAL-OBS候选。

**从函数到结构**：在正常源码路径中研究多个W1行的共享输入、GELU与原HiF4 Q2耦合、W2的signed向量合成及后续层补偿。其计算图重要性随行为目标、上下文及可验证的干预而定义。

**从结构到新实现**：由适用函数选择未来的量化码幅值/符号、静态/动态函数重参数化、数据布局、时空cache或小训练；记录真实可用信息时点、物理W1/Q2/W2费用与完整视频功能。

**研究方法**：[AutoResearcher领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)与[独立方法讨论](../method_discussion/AGENTS.md)、[深度原论文检索](DEEP_WEB_RESEARCH.md)。独立发现者和reviewer可提出新数学表示或不同假说，沿[图注解](../method_discussion/graph_annotations/E0/README.md)提供源码/ALG/OBS/LIT和OPEN对照边。证据状态紧贴命题，用于决定最有价值的下一份来源或最小判别。

## 状态、质量和交接

**当前科学活动**：论文/源码、数学推导、已存证据解释、独立讨论与文档/图谱。**新CPU/GPU/模型/视频原生执行暂停**；先前paired-native固定范围已完成并保存RUN。后续运行需用户明确授权，以[ACTIVE](../ACTIVE_RESEARCH.json)控制。

新模型或视频方法的质量需覆盖40层×4step、自由视频主体/背景/时序表现、same-source/seed比较及实际cost/SLO（见[统一指标协议](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)）。纯理论和已测观察以各自数学/来源证据交付。

**交接优先顺序**：[G13科学问题](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) → [已有真实证据包](CURRENT_CONTEXT_PACKET.md) → [主图](../../docs/98_端到端计算图与证据注解_20261025.md) → [ACTIVE](../ACTIVE_RESEARCH.json)。留下新的函数/结构、竞争解释、可判别证据与图节点，保持积极构造的研究表达。
