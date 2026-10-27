# FFNDS · 独立科研Agent（E0_OPEN_DISCOVERY）

**科学研究中心**：先定位具有行为意义的Wan视频FFN输出函数，再定位正常执行中承担这些函数的联合计算结构。从真实视频与状态差、数学与原作者论文寻找开放解释；通过不同机制的可判别预测推动下一次研究。

**科研入口**：[根目录AGENTS](../../AGENTS.md) · [FFNDS领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md) · [当前G13功能定位](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [Function–Structure对象](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · [主计算图G0–G13](../../docs/98_端到端计算图与证据注解_20261025.md) · [一手证据包](../agent_protocol/CURRENT_CONTEXT_PACKET.md) · [ACTIVE状态](../ACTIVE_RESEARCH.json)。

**AutoResearcher 来源**：[ACTIVE_SOURCE_MANIFEST.json](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)的固定版本，结合[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)、[web-research-discovery](../../.agents/skills/web-research-discovery/SKILL.md)和[本项目原文/源码深度检索协议](../agent_protocol/DEEP_WEB_RESEARCH.md)。

## 新发现、数学及独立反证

1. **观察与命名**：来自完整视频/latent/flow的行为差异、自然q上的FFN signed输出关系或无标签时空规律，可以先作为FUNCTIONAL-OBS与新数学假说。
2. **源码定位**：根据实际配置重建norm2/AdaLN→Q1→W1→GELU→HiF4 Q2(group64)→W2→e5门控BF16写回→后续attention/step/视频；比较同一q驱动的多行联合响应和下游补偿。
3. **推导与探索**：给出一种新的表示、函数交互、时空共享或其他数学自由度，提出与最强相关论文/旧原生解释不同的预测。新构造可早于形式化候选注册而留作探索种子。
4. **独立审查**：其他Agent亲自核实原论文、数学、实际源码、条件反例和证据来源，给出具体支持、挑战或建设性替代结构。观察级结论、功能因果证据与物理可省计算分别定位。
5. **逐步交付**：留下新函数/结构认识、真实source/seed/step范围、图节点或新OPEN边、最有判别力的下一项证明/已有数据对照/未来授权的实验。研究成果由实际解释力与可证伪性评价。

方法学追随[科研协作总则](README.md)、[独立发现目录](discoveries/E0/README.md)和[图锚点目录](graph_annotations/E0/README.md)，原生数值判断沿[实验发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)与[真实视频指标](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)审查。

## 多Agent讨论与共享记录

- **独立发现者/提案者**在自己的 \`discoveries/E0/<agent-id>-<phenomenon>.md\` 或 \`proposals/<method-id>/vN.json\` 保存认识、方程、源码与竞争解释。
- **独立评议者**在 \`reviews/<method-id>/vN/<review-id>.json\` 留下精确命题、独立论证及原始反例，引用原提案Git blob；其它review可以随后阅读与回应，保持评议来源身份。
- **协调整理者**依据真实评议版本串行更新[QUEUE](QUEUE.md)与endorsements。跨Agent \`PEER_SUPPORTED\` / \`CROSS_CONFIRMED\` 代表同一版本的科学讨论支持，具体原生效果仍由自己的测试证据与条件建立。
- **Graph Agent**使用[节点/证据注册表](../agent_protocol/GRAPH_AGENT_BINDINGS.json)和[协作协议](../agent_protocol/GRAPH_AGENT_COLLABORATION.md)回写源码SRC、数学ALG、论文LIT、观察NATIVE及OPEN条件关系，促进后续独立检查。

## 当前研究范围

**真实研究对象**：LightX2V Wan2.2 I2V A14B四步双expert，已完成10独立source、12条source×seed生成轨迹及1920个嵌套site。source425跨seed有符号P/N交叉项翻转、视频背景/光照失配与物理tile占用等都提供具有信息价值的函数/结构问题。[原生全深度反事实](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)已有固定证据。

目前执行科学讨论、源码、原文、数学、保存数据和文档/图谱工作。**新CPU/GPU/模型/视频实验暂停**，任何后续原生干预需用户新授权；旧paired-native v2/v3及其它用户批准的范围已经完成或明确关闭，真实RUN按[ACTIVE](../ACTIVE_RESEARCH.json)保存。

**默认交接**：新发现 → 数学/源码来源 → 竞争机制与相反预测 → 最低成本有区分度的下一证据 → 图与函数/结构登记。
