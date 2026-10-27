# FFNDS · Copilot科学研究默认引导（E0_OPEN_DISCOVERY）

首要问题：**哪些Wan视频FFN输出函数具有行为意义，哪些正常前向计算结构承担它们？** 从真实视频/flow现象与无标签有符号函数响应出发，沿实际计算路径寻找联合功能结构；再讨论量化code、proxy、稀疏、cache、训练后的经济实现。

**必读**：[根AGENTS](../AGENTS.md)、[FFNDS科研Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)、[G13行为/函数/结构文档](../research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[独立函数/结构档案](../research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)、[真实G0–G13计算图](../docs/98_端到端计算图与证据注解_20261025.md)。[第一任务](../research/20261025/E0_FFN_FIRST_TASK_20261025.md)与[历史本征科学纲领](../research/20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)为该阶段的研究与来源入口。

## 默认科研动作

1. **发现有意义的函数**：定位自然source/seed/step/expert、主体/运动/背景/光照与无标签有符号联合响应。区分函数的自然行为关联、在normal-forward的因果角色及实际可执行结构。
2. **沿原生计算流找承担者**：LightX2V Wan2.2 I2V A14B 4step D-Q（high/high/low/low，40blocks）；检查norm2/AdaLN→Smooth1/Q1→W1→tanh-GELU→Smooth2/HiF4 Q2(group64→8→4)→W2→e5 BF16 gate/residual→下游attention/flow/video。当前原实验使用HiF4 fake-QDQ+BF16 GEMM。
3. **构造性推导与独立反例**：发现一项数学交互/共享/条件职责，提出最强替代机制、各自不同预测以及最少的一项源码/已有数据/证明证据。新问题可增设图节点，不局限于旧Affine、TopK、FP4或可执行tile路径。
4. **主动深入文献**：使用固定[AutoResearcher WARD Skill](../.agents/skills/web-research-discovery/SKILL.md)及[原文深度检索](../research/agent_protocol/DEEP_WEB_RESEARCH.md)；从基础到近期论文，读公式、消融、作者源码并沿引用链查竞争思想。精确注明外部模型、训练、技术实现和实际阅读层级。
5. **保留真实原生证据**：[全深度与视频功能审查](../research/20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)、[源/seed的P/N数据](../research/20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)、[有限整步响应](../research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)以及已测有效计算。实验的有界失败解释下一种可研究结构，外部作者成绩按LIT登记。
6. **双向图/科研讨论**：按[Graph-Agent协议](../research/agent_protocol/GRAPH_AGENT_COLLABORATION.md)链接真实源与消费者，通过[独立讨论目录](../research/method_discussion/AGENTS.md)提出功能与结构解释、原文支持及不同预测，由同行独立审阅。

**现行事实与权限**：85项文献、171图节点、354关系、14机制族；10个原生source、12条完整轨迹、1920个嵌套FFN观测位置。当前[ACTIVE_RESEARCH](../research/ACTIVE_RESEARCH.json)的固定实验批次已完成，新CPU/GPU/模型/视频执行暂停。科研讨论、原文、源码、数学和已有结果分析继续；研究问题及身份在ACTIVE中准确归档。

**AutoResearcher方法来源**：核心科学Skills来自[固定manifest](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，采用[mechanism-research](../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../.agents/skills/problem-reformulation/SKILL.md)及WARD。新回复首先给出**科学发现或构造性解释**，再写方程与真实证据、相反预测和下一条可区分的观察。源级/数学/实验的证据等级紧贴具体命题，而非用长串限定句铺在发现前面。
