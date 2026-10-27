# FFNDS · 视频生成FFN的信息、功能与高效计算研究

**共同科学问题**：视频生成模型如何组织信息、计算与真实生成行为？能否据此发现更优的表示、运算结构和实际执行方式，在维持完整视频能力的同时减少物理成本？

以 LightX2V Wan2.2 I2V A14B 四步D-Q为原生证据锚点，沿**信息与表示（I）／函数与动力学（F）／计算与执行（C）**三条互相反馈的研究路径探索。G13是函数因果与承担结构的专题，G7–G12的量化、复用、重构、系统机制也能独立产生新发现。研究路线从现象、数学、源码或性能任一入口开始，见[统一目标](research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

当前索引：**89项相关论文、186个稳定计算图节点、397条类型化关系、14个机制家族**；G13包含27个定位节点；新增保存操作数观察与OPEN后缀分别标注。不同论文的实际阅读层级可在source ledger中逐项追溯。

## 研究入口

| 研究目标 | 当前资料 |
|---|---|
| **统一目标与三条并行入口** | [研究北极星与I/F/C路径](research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)、[研究导航](research/RESEARCH_NAVIGATION.md)、[Objective Contract](research/OBJECTIVE_CONTRACT.json) |
| 定位有意义的FFN输出函数、联合计算结构（F线） | [G13功能发现与结构定位](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P四层定位方法学](research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[三张机器定位卡](research/agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)、[行为/函数/结构登记](research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) |
| 沿真实Wan算子、残差、flow逐坐标追踪 | [G0–G13权威计算图](docs/98_端到端计算图与证据注解_20261025.md)与[Graph Bindings](research/agent_protocol/GRAPH_AGENT_BINDINGS.json) |
| 综合已有实验、竞争机制与原论文 | [FFN科学综合报告](research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md) |
| 检索原论文、机制家族与实际源码证据 | [89项文献索引](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[机制Crosswalk](research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)、[竞争矩阵](research/literature/20261026/MECHANISM_COMPETITION_MATRIX.md) |
| 使用科研Skill与多Agent独立讨论 | [FFNDS领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md)、[AGENTS](AGENTS.md)、[Graph↔Agent协作](research/agent_protocol/GRAPH_AGENT_COLLABORATION.md) |
| 实际研究状态、运行历史和授权 | [ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)、[研究导航](research/RESEARCH_NAVIGATION.md) |
| 数据抽样、校准、生成与评价 | [实验Skill](.agents/skills/ffnds-experiment-evidence/SKILL.md)、[可执行手册](research/agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)、[数据协议](research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md) |

## 真实研究对象与已有结果

原生Wan：**40个Transformer blocks，4个去噪step依次为high/high/low/low，两套独立expert**。FFN执行依次是

\[
\text{norm2/AdaLN}\to \text{Smooth1/Q1}\to W_1
\to \mathrm{GELU}_{\tanh}\to\text{Smooth2/HiF4 Q2}
\to W_2\to e_5\text{ gated BF16 residual}.
\]

原HiF4 Q2带64→8→4共享尺度，原数值实现使用**fake-QDQ + BF16 Linear**；packed NVFP4/MXFP4 GEMM是独立物理比较对象。代表性一次FFN尺寸 \(M=31668,D=5120,H=13824\)，81个最终视频帧对应21个latent时间位置等真实shape须沿源码确认。

已经完成**10个独立source、12条source×seed完整轨迹、1920个嵌套FFN观察位置**。旧候选的局部误差和原生计算速度提供了有用的候选对照，而跨seed signed状态作用、完整dense有限响应与视频背景/光照时序失配指出了**仍需定位的联合输出函数**。正式VBench与人工画质验收使用各自记录的真实状态。

[并行机制证据组合](research/20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)展示四项开放机制、当前已知证据和下一项可区分检验；图谱不预设算法优先级。

**本轮已完成**：[共享输入联合支持与精确投影强比较器](research/20261027/E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)、[原M Q2直接/补集数值判别](research/20261027/E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)、[W2形状控制](research/20261027/E0_W2_SHAPE_CONTROL_RESULTS.md)。用户要求的[UB1独立全帧复核](research/20261027/UB1_INDEPENDENT_VISUAL_REVIEW.md)覆盖两组历史配对；完整视频质量与新方法优势仍按精确范围判断。

## 新数据与评价管线的实际状态

2026-10-27合并的实验管线提供来源登记、全160位置校准行**计划**、原生操作数索引/文件头核验、注册后Dense/UB1原生视频生成入口、原版VBench分维调用入口，以及图像簇配对统计工具。各环节独立读取真实收据：**计划生成 ≠ 原生操作数已采集或拟合；原版评测函数执行 ≠ 九维完整官方榜单；来源级重采样 ≠ 新的独立来源**。

同一图像簇内先对各source/prompt的seed×sample重复观测求均值，再对source/prompt等权，最后对独立图像簇等权。历史单视频复制五个文件名的VBench结果仍按单次生成的旧协议解释。当前工作与模型执行权限统一以 `research/ACTIVE_RESEARCH.json` 为入口；`research/CURRENT_DECISION.json` 中的早期暂停状态及旧RUN保留为对应阶段的历史记录。CPU静态/统计测试与真实GPU/官方质量运行分别验收。

**最新原生判别（不同证据对象）**：[固定R256 consumer](research/20261027/UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)在24来源校准训练内恢复83.01%的局部gate误差；[六条配对视频的八维官方函数结果](research/20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)仅覆盖两条已暴露来源，连续RAFT运动相对neutral部分恢复，仍低于Dense，camera motion未运行；[保存latent有符号几何](research/20261027/UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)显示局部纠偏与末步距离增大并存。[A100合成激活2:4比较](research/20261027/E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)找到部分更快的预打包乘法布局，在线动态选择、压缩与完整视频收益仍分别验证。该组发现已索引于[研究导航](research/RESEARCH_NAVIGATION.md)。

## 当前科研方向

**I · 信息与表示**：研究何种输入、权重编码、历史状态或时空共享信息在何时可得，以及其获取/使用的净计算价值。G11–G12的signed码、重排界和编码域计算可作为独立数学或算子研究，而非某类mask的附属筛选器。

**F · 函数与动力学（G13）**：从真实运动、背景、照明、主体与无标签signed响应提出函数候选，研究共享q的多W1行、GELU、HiF4 Q2尺度、W2正负合成、下游attention/step及正常前向的联合职责。O/D/Q/P服务于这一因果定位问题；其新的具体承担结构仍需证据判别。

**C · 计算与执行**：独立探索step-specific紧凑FFN、跨步复用、稀疏tile、低位dense与编码域算术、训练后函数重分配以及端到端部署。以真实GEMM/QDQ、布局和访存、全视频质量、latency/VRAM及可比先例建立Pareto。

**两个可独立成立的成果**：新的可检验信息/函数/计算规律，及具有竞争力的完整质量–成本方法。Q1可用信息、Q2净决策价值、Q3实际省算彼此关联，但不是必须按顺序完成的前置关卡。

## 只读计算图与验证

~~~bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py inspect F08R
python tools/graph_agent_bridge.py trace G18 P12
python tools/graph_agent_bridge.py mechanisms
python tools/graph_agent_bridge.py mechanism M13
~~~

该工具验证节点ID、关系、证据文件和机制索引，提供源图检索。模型数值、因果功能和物理加速各有独立的证据层级。[静态测试](tests/test_graph_agent_bridge.py)与[真实视频完整指标规则](research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)分别服务于图谱可靠性和正式生成评价。

**当前工作权限（2026-10-27恢复）**：用户明确要求按最新main持续推进，同时最多使用四张空闲GPU，简单委派使用GPT‑6.1 Terra。I/F/C研究及冻结登记后的有限实验由[ACTIVE](research/ACTIVE_RESEARCH.json)统一维护；历史暂停、RUN、SHA和质量结论保持原样，实时/流式实验未恢复。
