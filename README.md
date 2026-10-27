# FFNDS · Wan2.2 四步视频生成中的 FFN 函数与真实计算研究

FFNDS研究**能以什么信息和数学表示减少真正的 FFN / 整段视频计算，同时保持生成视频的运动、主体、背景、光照与时间一致性**。研究从真实 LightX2V Wan2.2 I2V A14B 四步D-Q代码出发，并沿原始科研论文扩展函数、信息、时间和硬件机制。文献与计算图是研究工具，不是某一候选稀疏算子的实现声明。

**当前图谱规模**：85项关联研究、171个稳定节点、354条有类型关系（包含26条新跨机制比较边）。数量代表可检索覆盖，不等于79篇均已完成全文或源码核查。

## 当前第一科学问题：定位输出函数及其计算结构

**先识别视频生成中具有行为作用的FFN输出函数，再研究承担这些函数的正常前向计算结构。** [G13函数定位、因果证据与实际算子路径](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)是现行首要研究入口；[行为–函数–结构独立登记](research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)明确已有视频现象、开放的函数身份与待验证的计算图关系。量化code/sign、proxy、LUT、TopK或各种内核只在函数与结构定位后考虑信息/执行经济性。

当前图谱为**85项相关研究、171稳定节点、354条类型化关系、14个机制家族**，其中G13新增18节点/41条件比较边；没有新模型功能因果实验，原模型CPU/GPU/视频运行保持暂停。G13原文方法学先例详见[来源账本](research/literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)。

## 现行研究入口（2026-10-27）

| 目的 | 权威资料 |
|---|---|
| 当前已经知道的事实、竞争假说及研究综合 | [统一 FFN 机制、文献与计算图研究报告](research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) |
| 一张可逐算子追踪的 Wan + 外部方法计算图 | [G0–G12 完整源码计算图与证据注解](docs/98_端到端计算图与证据注解_20261025.md) |
| 机器可读的稳定节点、边和12类机制互联 | [Graph Bindings](research/agent_protocol/GRAPH_AGENT_BINDINGS.json) / [Mechanism Crosswalk](research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) |
| 85篇论文及各自真实阅读等级 | [Primary Literature Index](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json) / [机制竞争矩阵](research/literature/20261026/MECHANISM_COMPETITION_MATRIX.md) |
| 已完成原生实验的发现、证据身份和验收范围 | [Experiment Findings Atlas](research/EXPERIMENT_FINDINGS_ATLAS.md) / [研究导航](research/RESEARCH_NAVIGATION.md) |
| 默认AutoResearcher原则与科研Agent讨论 | [AGENTS.md](AGENTS.md) / [Graph↔Agent协作](research/agent_protocol/GRAPH_AGENT_COLLABORATION.md) |
| 当前模型执行、实验权限和工作阶段 | [ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json) |

**研究状态**：已覆盖10个独立source身份、12条source×seed生成轨迹、1920处全深度FFN观察位置（不是1920条独立视频）。既有候选完整四步视频呈现可见质量失配，正式外部VBench与人工验收状态不属于已通过。原已授权任务全部结束，**所有新的CPU/GPU/模型/视频实验暂停**；仅文献原文与源码、数学、保存证据分析、研究讨论及计算图维护继续进行。新执行需用户另行授权。

## 原始模型：准确的计算对象

- **Wan2.2 I2V A14B D-Q四步蒸馏模型**，40 blocks，去噪四个step按 **high/high/low/low** 启用两套独立的专家网络。
- 原生运行中代表性一次FFN输入 **M=31668 token行、D=5120、hidden H=13824**。81个最终视频帧和21个latent时间bin是不同对象。
- 原FFN实际顺序：**cross-attention residual → norm2 / AdaLN → Smooth1 / HiF4 Q1 → W1 → tanh-GELU → Smooth2 / HiF4 Q2（64→8→4共享尺度依赖）→ W2 → e5 gated BF16 residual → downstream blocks / flow / scheduler / video**。
- 本仓库原生数值是**HiF4 fake-QDQ + BF16 Linear**，不能与作者论文或外部示例的packed NVFP4/MXFP4 Tensor Core kernel替换理解。真正的W1、W2以及Q2依赖和物理代价都在[计算图G0–G5](docs/98_端到端计算图与证据注解_20261025.md)中标明。

## 三个相互连接的科研问题

1. **信息（Q1）：** 是否存在本来已可廉价读取的信息，例如当前q、离线权重统计、量化code/scale/sign、time/expert或合法历史，足以描述某个FFN函数所需的结构？
2. **决策（Q2）：** 取得这些信息、router/proxy、code读取、group界计算、缓存、packed layout是否比直接完整计算更便宜？其对有符号gate×W2和视频功能的预测力在不同source/seed/step是否保持？
3. **真实省算（Q3）：** 到底跳过或改写了哪些W1/GELU/HiF4 Q2/W2/GEMM tile、step/frame/block，在同等模型质量与部署硬件上是否降低完整生成或实时TTFF/p99成本？

这三个问题可以并行提出不同假说，无需先规定TopK、Dense恢复、SSE、FP4格式、特定路由或单一路线。局部raw SSE改善不等于最终视频稳定，也不反证其它未来函数表示。

## 论文与图谱的七条主线

| 外部图分区 | 比较什么 | 一手材料 |
|---|---|---|
| **G7** | 按step训练的精简模型、frame×layer分配、FFN宽度与谱 | [DIFS/FIS原文审查](research/literature/20261026/PRIMARY_METHOD_EQUATION_AUDIT_20261027.md) |
| **G8** | 当前GEMM的N:M、训练MoE、跨步主/残差或FFN缓存 | [四种方法一手分析](research/literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md) |
| **G9** | 整个函数历史预测、硬树条件执行、全网功能迁移 | [G9一手阅读](research/literature/20261027/G9_CONDITIONAL_EXECUTION_TRAJECTORY_AND_FUNCTION_REALLOCATION.md) |
| **G10** | 普通视频teacher如何变成实时causal/rolling学生及其系统SLO | [实时视频14项论文与源码](research/literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md) |
| **G11** | FP4符号/尺度的信息时点，Prox、CodeGEMM、SharQ等计算代价 | [核心5篇深读](research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ的Wan-T2V强对照](research/literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) |
| **G13** | 先定位有意义的输出函数、真实计算子图和因果证据 | [首要函数定位报告](research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) |
| **G12** | 有符号量化直方图的单行紧内积区间、部分配对精读与bitplane/LUT/GEMV | [严格代数和作者源码](research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) |

G7–G12新外部节点使用LIT/ALG/SPEC/OPEN证据等级。CodeGEMM、LUT-GEMM的公开实现单向量查表与多token解量化dense回退、SharQ的sparse加dense residual GEMM是实质性的**原作者源码实现边界**；相关结果不是当前Wan本地运行。

## 只读计算图检索

以下命令仅检查和查询仓库现有JSON/Markdown，不运行模型、GPU或生成视频：

~~~bash
python tools/graph_agent_bridge.py validate
python tools/graph_agent_bridge.py inspect F08R
python tools/graph_agent_bridge.py trace I04 F12
python tools/graph_agent_bridge.py mechanisms
python tools/graph_agent_bridge.py mechanism M02
~~~

validate检查SPEC/LIT/ALG/OBS/OPEN与原生证据节点的存在性、Mermaid标签、现行79篇论文ID、12机制组及26条crosswalk关系。它只验证**静态文档/图谱一致性**，不证明数学值、模型正确性、内核速度和视频质量。对应轻量测试：[test_graph_agent_bridge.py](tests/test_graph_agent_bridge.py)。

## 复核、科研协作和实验权限

先通过[研究导航](research/RESEARCH_NAVIGATION.md)定位原始配置、论文ID和evidence类型。每项研究形成**可检验的结构或机制 → 最强竞争解释 → 最低成本区分方法 → 实际省算范围与视频功能**，并回写[图谱](research/agent_protocol/GRAPH_AGENT_BINDINGS.json)和[综合报告](research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。静态graph/文档检查说明链接及关系一致；数值证明要源代码/数学，真实速度与质量需授权后原生完整试验。

[AutoResearcher](https://github.com/Qesire/AutoResearcher)原则默认生效；多科研Agent可通过[讨论队列](research/method_discussion/QUEUE.md)独立提出方法与复核。所有历史实验的run协议、source SHA、指标与失败结论保留在原始报告和[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md)，当前任务与权限只在[ACTIVE](research/ACTIVE_RESEARCH.json)中确定。
