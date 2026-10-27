# E0 / G13 · 真实视频FFN输出函数与计算结构定位：当前第一研究任务

**科学问题**：FFN在四步生成中产生哪些具有行为意义的输出函数？这些函数实际由哪些共享输入、非线性、W2门控、时空和跨层/step计算结构承担？由此发现可以共享、重组、预测或缩减的真实工作。

现行定位依据：[G13科学文档](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [独立行为/函数/结构登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · [源图G0–G13](../../docs/98_端到端计算图与证据注解_20261025.md) · [实验发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)。

## 实际模型及已知现象

**原生对象**：LightX2V Wan2.2 I2V A14B 4step D-Q，high/high/low/low两套expert，40 blocks。实际FFN在norm2/AdaLN后执行Q1→W1→GELU(tanh)→HiF4 Q2（64→8→4）→W2→e5 gated BF16 residual，并经剩余blocks/flow/scheduler进入视频。原量化路径是fake-QDQ + BF16 Linear。

已完成的**10个source身份、12条source×seed完整轨迹、1920个嵌套FFN site**提供丰富但不同层级的证据：

- [已存完整状态有限响应](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)：按真实dense与候选整步映射拆分 \(P=T_D(z_S)-T_D(z_D)\)、\(N=T_S(z_S)-T_D(z_S)\)、\(\delta_{t+1}=P+N\)；局部FFN误差、继承状态响应和本步有符号作用形成三种不同科学对象。
- [跨来源与交叉seed](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)：同source425、图片/提示固定、seed42/54时后三步P/N交叉作用符号相反，局部误差计数相同。下一题是定位产生这种条件性有符号效应的FFN函数与消费者。
- [频带观察](../20261026/E0_SAVED_FLOW_SPECTRUM.md)：部分空间频带内与全域的signed cross可反号，提示研究可以在时空关系和无标签功能上建立多个独立观察轴。
- [HiF4分组与物理tile](E0_SHARED_TILE_PRE_W1.md)：后验token级集中度与原M16×K64物理tile活跃度形成差异，说明寻找**功能本身的组织单位**与寻找**物理执行单位**是相互连接的两个问题。
- [旧近似方法与质量事实](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)：Affine54等方案的局部误差/速度与真实视频功能表现存在差异，为进一步的函数/因果识别提供了真实反例。

## 三个相互推进的工作对象

**A · 行为与无标签函数候选**：对source、seed、step、expert、layer及token时空位置分别记录自然可见的运动、主体、背景、照明、时序及其变化，连同未命名的有符号协作/抵消、共变、空间–时间稳定性。用[登记表](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)明确自然条件、原生证据、FUNCTIONAL-OBS层级、竞争解释和未来有价值的判别。

**B · 实际计算结构的定位**：沿原代码追踪当前q对所有W1行的共同驱动、GELU非线性、HiF4 Q2共享尺度、W2列组合与门控写回；追踪attention、深层FFN、flow/scheduler和相邻steps的继续/补偿。每个STRUCTURE候选给出真实图边、可能的联合函数、物理计算形状与最早的信息可得点。

**C · 因果与计算价值的连接**：先区分不同机制在保存证据中的预期表现；未来获新实验许可时，选一个同时符合normal-forward忠实性、真实状态控制、必要/充分和完整视频指标的最小干预。找到函数及结构后，才把quant-code、proxy、LUT、压缩算子、跨step cache和少量后训练纳入实际计算经济性比较。

三者可由新视频现象、不同的数学表示或相关原论文交替驱动；**研究者自由提出超出已有图节点的新功能结构**，然后以源事实和比较预测把它接回计算图。

## 当前交付与科研Agent分工

本阶段交付一份**有行为意义的候选函数地图**、一份**已核原生计算承担结构地图**，以及若干**能真正区分机制的下一条证据**。每条结论包含具体source/seed/step、源码切面、最强不同解释、现成已存资料、理论推导或未来最小干预。初步数学结果或可复现的FUNC-OBS现象本身就可以推进科学，不必立即附上某个固定稀疏kernel。

多Agent独立讨论遵循[AutoResearcher领域skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)及[图协作约定](../agent_protocol/GRAPH_AGENT_COLLABORATION.md)；发现者提出机制和数学，独立审阅者核实正常计算路径、来源和反例，整合者更新图、函数记录与现行综合。[85项文献索引](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)与[方法学G13 L80–L85](../literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)提供强比较研究。

未来原生视频候选的完整质量、算子速度/显存、独立source×seed和40×4步证据按[指标与全层协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)记录，VBench/人工评议依实际运行状态标明。

## 状态与交接

**当前工作**：论文与作者源码研究、数学推导、已有证据的解释性整理、函数/结构记录和计算图同步。**所有新CPU/GPU/模型/视频实验暂停**；原生复验与未来干预需用户新的明确授权。[ACTIVE](../ACTIVE_RESEARCH.json)为权限源。2026-10-25/26既有实验全部完成且原始RUN、哈希和严格结论持续保留。

**交接先读** [G13科学问题](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) → [当前功能/结构登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) → [具体原生RUN证据](../EXPERIMENT_FINDINGS_ATLAS.md) → [源码图](../../docs/98_端到端计算图与证据注解_20261025.md) → [ACTIVE](../ACTIVE_RESEARCH.json)；按当前问题取必要资料。每轮留下一个新的机制理解、对应有类型图边和下一条有信息价值的判别。
