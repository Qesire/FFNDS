# FFNDS · Copilot科学研究默认引导（E0_OPEN_DISCOVERY）

统一科学问题：**视频生成模型如何组织信息、计算与生成行为，怎样在保持完整视频能力时降低真实执行成本？** 可从I（信息与表示）、F（函数与动力学）、C（计算与执行）任意入口研究，G13是F线专题。见[统一目标](../research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

**方法统一五操作分类**：[O-A计算选择、O-B表示、O-C历史复用、O-D算子代数、O-E学习/验证](../research/20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)。与I/F/C科学入口正交，按G14一手论文source+decision_time+runtime action建立组合指纹。原三候选明确为M-A/M-B/M-C，下一Agent先研究**M-B敏感坐标**，避免与操作O-B同名混用；现有方法讨论批准及运行范围不变。

**本轮用户指定的首个方法研究任务**：[从B联合gate/后缀敏感方向开始](../research/agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。研究Agent先完成B的数学与保存资产判别、G14论文/源码近邻核验，区分B-FOLDED/B-FACTORED、真实原生QDQ与来源状态，再提出独立METHOD_SNAPSHOT；不能改用A0训练或C码本拟合作为首项任务。A0仍是matched基线，I/F/C整体未改变。

**必读**：[根AGENTS](../AGENTS.md)、[领域Skill](../.agents/skills/ffnds-paper-discovery/SKILL.md)、[研究导航](../research/RESEARCH_NAVIGATION.md)、[跨机制综合](../research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[主计算图](../docs/98_端到端计算图与证据注解_20261025.md)及[ACTIVE执行状态](../research/ACTIVE_RESEARCH.json)。F线深入时使用[G13](../research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)与[O/D/Q/P](../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)；旧[第一任务](../research/20261025/E0_FFN_FIRST_TASK_20261025.md)是历史研究阶段的证据来源。

## 数据、校准与评测的实验思维

遵循[新实验数据协议](../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)与[专用Skill](../.agents/skills/ffnds-experiment-evidence/SKILL.md)。恢复真实来源与source×seed独立性、样本行坐标和状态来源、候选/ Dense完全配对、指标覆盖与视频原始证据；用VBench-I2V官方五次真实生成和CUSTOM_PAIRED单次自定义评价的严格区分建立科学结论。每个问题正向给出来源级对照和最便宜的可判别实验；官方未测维度保留NOT_RUN。历史记录维持原有身份。

## 默认科研动作

1. **任选真实入口**：从有趣的视频行为/有符号差异、已可得的信息与表示、数学算子恒等式、硬件执行异常或原论文机制提出新构造及竞争解释。
2. **追溯关键计算**：按主张定位真实Wan2.2四步D-Q的norm2/AdaLN→Q1→W1→GELU→HiF4 Q2→W2→e5 BF16 gate/residual→flow/scheduler；仅对功能因果主张使用G13 O/D/Q/P完整定位方法。
3. **推导不同预测**：信息时点、计算粒度、功能迁移、因果路径和部署代价都可成为独立新规律；按AutoResearcher MEAR/PRD分析，结合最强相近方法与反例。
4. **主动深读原文与源码**：使用[WARD](../.agents/skills/web-research-discovery/SKILL.md)查原始数学、消融、作者代码、负例及引用链；保留实际模型、训练、数值与运行环境。
5. **评价恰当主张**：机制论文先验证数学/源码/因果范围；执行加速论文另外计量真实FFN+系统成本与独立完整视频质量，严格区分静态图验证、原生运行与外部论文证据。
6. **独立协作与真实图谱**：经[Graph-Agent协议](../research/agent_protocol/GRAPH_AGENT_COLLABORATION.md)与[讨论队列](../research/method_discussion/QUEUE.md)回写I/F/C任一路线的源节点、ALG/LIT/NATIVE/OPEN关系和独立评议。

**现行事实与权限**：89项文献、177图节点、371关系、14机制族；10个原生source、12条完整轨迹、1920个嵌套FFN观测位置。2026-10-27用户明确恢复持续推进，同时最多使用四张空闲GPU，简单委派使用GPT‑6.1 Terra。[ACTIVE_RESEARCH](../research/ACTIVE_RESEARCH.json)逐项登记冻结的有限执行范围和实际状态；旧暂停及已完成批次保持历史身份，实时/流式实验未恢复。

**AutoResearcher方法来源**：核心科学Skills来自[固定manifest](../research/agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，采用[mechanism-research](../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../.agents/skills/problem-reformulation/SKILL.md)及WARD。新回复首先给出**科学发现或构造性解释**，再写方程与真实证据、相反预测和下一条可区分的观察。源级/数学/实验的证据等级紧贴具体命题，而非用长串限定句铺在发现前面。
