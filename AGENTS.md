# FFNDS · AutoResearcher 科研Agent默认约定

**现行科学问题**：在四步Wan视频生成中，**哪些FFN输出函数具有真实行为意义？哪些正常前向计算结构承担这些函数？** 依次追索观测函数、因果贡献、实际计算和未来高效实现，同时欢迎从新论文、数学表达或原生反直觉数据引入开放的解释。

统一入口：[领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md) · [G13研究](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [行为/函数/结构索引](research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · [G0–G13原生计算图和研究分支](docs/98_端到端计算图与证据注解_20261025.md) · [研究综合](research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) · [ACTIVE状态与授权](research/ACTIVE_RESEARCH.json)。

**基准对象与证据**：LightX2V Wan2.2 I2V A14B 四步D-Q，high/high/low/low，40 blocks；\`norm2/AdaLN → Smooth1/Q1 → W1 → GELU(tanh) → Smooth2/HiF4 Q2(64→8→4) → W2 → e5门控BF16残差 → 后续attention/flow/scheduler/video\`。原QDQ和BF16 GEMM与外部native NVFP4格式分别建立数值/物理比较。历史原生10个source、12条source×seed轨迹、1920个嵌套FFN观测位置，已有真实速度收益和自由视频功能失配的研究资产。

**图谱现状**：85项LID、171个稳定节点、354条类型化关系，14个机制族。G0–G5包含真实SRC及既有native证据；G6为历史场景；G7–G12为各方法论文、理论与开放候选；G13专门研究行为–功能–结构定位。研究与实验状态由[ACTIVE](research/ACTIVE_RESEARCH.json)控制。

## 一、科学研究的默认语言与思路

研究回答以**构造性内容**开始：一个真实现象、特别的数学依赖或新的功能假说；随后给出最强竞争机制、彼此不同的预测、代价最低且有辨识力的下一证据、与视频功能和真实执行的联系。

**表达原则**：
- 用**“这个数据说明什么、揭示什么自由度、允许怎样的新机制”**取代连串“不能/不是/没有”开头的免责句。数学证明和实验反例保留准确的对象、前提、位置与指标，负面结果帮助发现另一可探索的表示或行动单位。
- 像“后验token级稀疏在原M16×K64物理tile仍100%占用”这样的具体结论应明确保留，并由此提问联合函数分组、不同执行颗粒、权重/状态共享是否可改善。将失败范围写成**已检验结构的有界结论**。
- 对未完成的科学问题积极提出OPEN机制及可证伪预测。对已测量的数值、作者论文和原生运行分别贴上ALG/SRC/OBS/NATIVE/VIDEO/LIT证据来源；精准的证据身份足以校准结论。
- 函数发现允许先从自然视频、无标签有符号关系、数学结构或文献得到灵感，**无需预先选择TopK、SSE、proxy、FP4格式、补算和某一种部署算子**。这些手段在已识别函数与结构后作为可付费的实现备选进行比较。

## 二、从正常计算定位具有行为意义的函数

**DISCOVER：现象到函数。** 用已有source×seed×step×expert条件、真实视频与保存状态提出候选行为轴（运动、相机、主体、背景、光照、时序、细节）和未命名函数轴（联合signed响应、抵消、稳定性、非线性协作）。在[注册表](research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)区别BEHAVIOR / FUNCTION / STRUCTURE。FUNCTIONAL-OBS可以推动机理研究；FUNCTIONAL-CAUSAL需要合法normal-forward有限干预；FUNCTIONAL-IMPLEMENTABLE再核查真实执行和完整视频。

**MAP_FLOW：函数到承担者。** 对决策关键子图查真实版本/权重/配置、输入输出维度、数据类型、step/expert、Q1、W1行对同一q的依赖、GELU、HiF4 Q2完整组尺度、W2列组有符号合成、e5写回、下游层/flow/视频消费者。数学式须保留真实非线性、数值闭包、残差和状态。跨layer/step补偿、共同输入驱动、条件性职责分配以及备用通路都能形成新的结构解释。

**DERIVE / EXPAND：放宽研究表示。** 用[research-expansion](.agents/skills/research-expansion/SKILL.md)和[problem-reformulation](.agents/skills/problem-reformulation/SKILL.md)寻找不同的函数坐标、时空计算单位、代数恒等式、因果目标与跨模型方法。可以先提出有趣的条件性假说，随后按其真实源路径验证决定性前提。

**DISCRIMINATE：区分科学机制。** 优先比较有不同预测的解释，依序利用证明/反例、原论文+作者源码、已有保存证据、未来获准的原生定点干预和完整视频。区分低局部误差、已继承状态的dense响应、当前候选signed作用及自由视频实际质量；把独立source/seed当成独立性单位，把全层site视为嵌套测量。

**ADVANCE：把发现写到应在的位置。** 行为和候选FUNCTION/STRUCTURE登记其上下文与证据；原文方程和作者代码只写到source ledger；综合报告只维护改变判断的机制解释；Graph Agent Bindings更新合法源码/数学/文献/OPEN节点与有类型关系。共享路径可检索，新的数学发现可以增设节点，图索引用来支持探索而非限定探索。

## 三、原生证据、论文及真实成本

- **原生**：[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md)、[证据优先级](research/EXPERIMENT_EVIDENCE_INDEX.md)、[已有full-flow finite-map](research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)。source425相同图片prompt跨seed的P/N cross符号改变和部分频带与全域交叉项异号，是函数定位的现实入口。过去的Affine54等候选已经测到局部误差/速度及视频失配，作为有界机制证据保留。
- **文献**：按[广域原文检索](.agents/skills/web-research-discovery/SKILL.md)探索基础/前沿及相邻论文，核对公式、默认实际源码路径、消融、训练、硬件与任务身份。[85项索引](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json)提供LID和阅读层级，[竞争矩阵](research/literature/20261026/MECHANISM_COMPETITION_MATRIX.md)和[Crosswalk](research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)提供对照。
- **机制与代价**：量化code/FP4 signed直方图的理论界（G12）、Prox付费代理、SharQ两路GEMM、CodeGEMM/LUT-GEMM单vector快路、训练后稀疏、step/history cache、实时因果系统都可成为候选的真实实现路径。模型对象、首次信息可用时间、online observer/metadata/cache/packing价格和完整FFN/video成本共同确定其物理意义。
- **质量与运行时**：[全深度与完整视频质量规则](research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)维护same-source/seed、40层×4步、gate×W2、质量、latency/setup/显存与SHA。独立的正式VBench/人工验收按实际运行标记。外部作者成绩依其自身模型、硬件、任务登记。
- **静态工具**：\`python tools/graph_agent_bridge.py validate\`检查节点、边、论文、机制家族；\`mechanisms\`、\`mechanism M13\`支持只读索引。静态一致性属于文档/代码协议级证据，数值真实性与视频因果作用仍由相应对象证明。

## 四、多Agent独立方法讨论

所有Agent继承[项目领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md)及[讨论约定](research/method_discussion/AGENTS.md)。独立提出函数或结构、新机制、数学或原始源码对照，接着由其它Agent发表独立支持、具体反例或替代解释；通过[QUEUE](research/method_discussion/QUEUE.md)保留方法版本和独立审阅。\`SUPPORT\`、\`PEER_SUPPORTED\`、\`CROSS_CONFIRMED\`、\`CONTESTED\`等标签针对**确切版本**，表达讨论证据，不是实验通过或部署结论。

图是实际研究对象的导航而非运行授权，既有RUN和source SHA保持原始身份；新图节点、研究假说和读文献可以独立展开，随后将最有信息价值的区别回写到[Graph↔Agent协议](research/agent_protocol/GRAPH_AGENT_COLLABORATION.md)。

## 五、当前研究权限与交接

**现阶段**：功能与计算结构的原始源码/数学、文献、已有证据复核、科研讨论、图与文档优化继续。**所有新的CPU/GPU/模型/视频实验暂停**，另获用户明确指令后按具体授权范围启动。2026-10-25/26的四GPU预算、dense视频及多来源运行属于已经完成的历史范围；[ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)保存真实授权与运行状态。

交接只需**新机制/功能发现、证据出处、真正分歧、对应图节点、最小后续判别和已完成的原生范围**。历史文档当时的“Next/Active/Run”用于溯源，由现行研究导航与ACTIVE区分当前状态。新研究以构造性内容优先、以真实证据调整确定度。
