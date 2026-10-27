# FFNDS · Wan2.2 四步视频生成的FFN函数与计算结构研究

**科学问题**：视频生成模型的FFN输出中，哪些函数对真实生成行为有意义？这些函数由哪些正常前向计算结构承担？怎样保留或重组重要功能，同时减少W1、GELU、量化、W2及完整生成的物理成本？

研究以 LightX2V Wan2.2 I2V A14B 四步D-Q 的源码、已完成的真实自由视频和原生数值证据为起点，通过AutoResearcher、原论文及有类型计算图不断扩展。**先定位行为与函数，再辨认计算承担结构，最后研究信息源及高效实现。**

当前索引：**85项相关论文、171个稳定计算图节点、354条类型化关系、14个机制家族**；G13包含18个功能定位节点和41条研究关系。不同论文的实际阅读层级可在source ledger中逐项追溯。

## 研究入口

| 研究目标 | 当前资料 |
|---|---|
| 定位有意义的FFN输出函数、联合计算结构 | [G13功能发现与结构定位](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[行为/函数/结构登记](research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) |
| 沿真实Wan算子、残差、flow逐坐标追踪 | [G0–G13权威计算图](docs/98_端到端计算图与证据注解_20261025.md)与[Graph Bindings](research/agent_protocol/GRAPH_AGENT_BINDINGS.json) |
| 综合已有实验、竞争机制与原论文 | [FFN科学综合报告](research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md) |
| 检索原论文、机制家族与实际源码证据 | [85项文献索引](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[机制Crosswalk](research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)、[竞争矩阵](research/literature/20261026/MECHANISM_COMPETITION_MATRIX.md) |
| 使用科研Skill与多Agent独立讨论 | [FFNDS领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md)、[AGENTS](AGENTS.md)、[Graph↔Agent协作](research/agent_protocol/GRAPH_AGENT_COLLABORATION.md) |
| 实际研究状态、运行历史和授权 | [ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)、[研究导航](research/RESEARCH_NAVIGATION.md) |

## 真实研究对象与已有结果

原生Wan：**40个Transformer blocks，4个去噪step依次为high/high/low/low，两套独立expert**。FFN执行依次是

\[
\text{norm2/AdaLN}\to \text{Smooth1/Q1}\to W_1
\to \mathrm{GELU}_{\tanh}\to\text{Smooth2/HiF4 Q2}
\to W_2\to e_5\text{ gated BF16 residual}.
\]

原HiF4 Q2带64→8→4共享尺度，原数值实现使用**fake-QDQ + BF16 Linear**；packed NVFP4/MXFP4 GEMM是独立物理比较对象。代表性一次FFN尺寸 \(M=31668,D=5120,H=13824\)，81个最终视频帧对应21个latent时间位置等真实shape须沿源码确认。

已经完成**10个独立source、12条source×seed完整轨迹、1920个嵌套FFN观察位置**。旧候选的局部误差和原生计算速度提供了有用的候选对照，而跨seed signed状态作用、完整dense有限响应与视频背景/光照时序失配指出了**仍需定位的联合输出函数**。正式VBench与人工画质验收使用各自记录的真实状态。

## 当前科研方向

**G13 功能研究**：从真实运动、背景、照明、主体与无标签signed响应发现条件性的函数候选；追踪共享q的多W1行、GELU、HiF4 Q2尺度、W2正负合成及attention/后续step，研究共同驱动、抵消、补偿与时空职责。分别登记FUNCTIONAL-OBS、FUNCTIONAL-CAUSAL、FUNCTIONAL-IMPLEMENTABLE；目前新G13因果定位仍是待独立判别的研究对象。

**G7–G12 提供实现和竞争先例**：step-specific compact FFN、frame×layer分工、缓存/函数轨迹预测、实时causal chunk、FP4代理、SharQ双GEMM、编码域查表和G12 signed-code区间等，都可以帮助进一步解释或重构已定位的实际函数。选择实现时共同评价在线信息时间、真实W1/Q2/W2工作、部署代价和完整视频质量。

后续阶段三个执行问题彼此联系：可用信息是什么、获取/使用信息的净价值如何、真正降低什么计算。科学研究先从**函数/结构意义**获得目标，再运用这些问题构造候选方法。

## 只读计算图与验证

~~~bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py inspect F08R
python tools/graph_agent_bridge.py trace U07 F12
python tools/graph_agent_bridge.py mechanisms
python tools/graph_agent_bridge.py mechanism M13
~~~

该工具验证节点ID、关系、证据文件和机制索引，提供源图检索。模型数值、因果功能和物理加速各有独立的证据层级。[静态测试](tests/test_graph_agent_bridge.py)与[真实视频完整指标规则](research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)分别服务于图谱可靠性和正式生成评价。

**当前工作权限**：论文原文/作者源码、数学、现有数据与研究文档/图谱分析。**新CPU/GPU/模型/视频执行暂停**，未来原生干预或生成由用户另行明确授权；历史RUN、SHA、先前候选的具体质量结论继续保留在原始材料中。
