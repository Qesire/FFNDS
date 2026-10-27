# FFNDS · AutoResearcher研究协议与任务入口（E0_OPEN_DISCOVERY）

**当前研究问题**：在真实视频生成FFN中发现信息、函数与计算的组织规律，并探索在保持完整视频质量下的真实低成本实现。**I/F/C并行入口**为信息与表示、函数与动力学、计算与执行。见[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

**统一方法操作分析（O-A～O-E）**：[五操作G14交叉方法文档](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)及[结构化source/方案表](METHOD_OPERATION_CROSSWALK.json)。I/F/C为科学问题，O-A选择计算、O-B函数表示、O-C缓存复用、O-D计算代数、O-E学习验证为可组合的算法动作；先通过G14读论文实际改变的边和decision_time，再判断本地Wan的合法数值/物理/时序条件。下一位方法Agent仍从**M-B**而非整个O-B起步。

**并行机制组合**：[多条开放机制、现有方法与最小判别](../20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)供科学推导与独立评议使用。

**科学导航**：[研究导航](../RESEARCH_NAVIGATION.md) · [I/C/F跨机制综合](../20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) · [G13函数定位（F线）](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [O/D/Q/P专题](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md) · [最小上下文](CURRENT_CONTEXT_PACKET.md) · [阶段入口](E0_STAGE_ANCHOR.md) · [ACTIVE](../ACTIVE_RESEARCH.json)。[2026-10-25首项任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)属于历史过程。

**数据与图**：[G0–G13权威计算图](../../docs/98_端到端计算图与证据注解_20261025.md) · [Graph Bindings](GRAPH_AGENT_BINDINGS.json) · [Graph-Agent协作](GRAPH_AGENT_COLLABORATION.md) · [实验发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)。当前89篇相关研究、186个稳定节点、397条有类型关系、14个机制族。原始研究从[本征机制重置](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)到[全模型方法审查](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)保留完整历史证据。

## 数据与实验可信度

[Calibration V2 方法与资产边界](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md) · [采样/校准/评测数据合同](DATA_SAMPLING_CALIBRATION_EVALUATION.md) · [实验专用Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md) · [2026-10-27源码审查](../20261027/FFNDS_DATA_PROTOCOL_AUDIT_20261027.md) · [CPU预检](../../tools/ffnds_cohort_preflight.py)。正式官方多次采样、单视频自定义诊断、Dense/Student校准及独立新确认数据采用各自明确的统计单位和用途；历史RUN保持原样。优先从来源级效果差异提出竞争机制和低成本有区别的对照。

## 科学方法与实际检索

可从已可得信息、非线性代数、自然视频响应、执行内核或相关论文任一入口开始。有效交付是：**新的信息/函数/计算机制 → 与主张有关的源码和数学 → 最强竞争机制和不同预测 → 最小判别证据 → 可能的科研或部署价值**。探索期间可以增加新的图节点、数学基与论文比较。

[FFNDS领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)按当前三条并行科研路径组织；通用源由[固定AutoResearcher manifest](ACTIVE_SOURCE_MANIFEST.json)绑定，执行[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md)和[web-research-discovery](../../.agents/skills/web-research-discovery/SKILL.md)。通过[深度Web原论文/源码协议](DEEP_WEB_RESEARCH.md)持续发现前沿与强先例、读方法、消融及作者实际实现，从机制差异拓展信息、函数与计算的研究空间。

[89篇索引](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)与[Mechanism Crosswalk](MECHANISM_GRAPH_CROSSWALK.json)记录论文阅读等级和方法竞争；G13主要对照JMLR causal abstraction、ICLR patching与CVPR Video-DiT时空定位，G7–G12提供高效计算/实时系统竞争。

## 科研Agent独立讨论

根[AGENTS](../../AGENTS.md)、[Copilot](../../.github/copilot-instructions.md)、[方法讨论AGENTS](../method_discussion/AGENTS.md)、[独立发现目录](../method_discussion/discoveries/E0/README.md)共享这项目标。独立Agent产生函数/结构解释、数学反例与源码证据；审阅者给出新论证，整理者按实际证据与图边串行更新[方法QUEUE](../method_discussion/QUEUE.md)。提出一个函数候选或新数学对象，可以先作为科学发现，成熟后再形成方法提案或实验候选。

## 固定上游来源与历史范围

[ACTIVE_SOURCE_MANIFEST.json](ACTIVE_SOURCE_MANIFEST.json)固定AutoResearcher v2.4-source-commit \`8b88a52d894568dc303db727decfbf13b5fabf3b\`，对应[UPSTREAM_SYNC](UPSTREAM_SYNC.md)与[检验工具](../../tools/check_autoresearcher_binding.py)。通用 upstream Skills/core 文件以commit及Git blob SHA逐项锁定；本项目专用Skill、AGENTS、Copilot和图/证据协议各自维护。原始旧SOURCE_MANIFEST与2026-10-24量化路线/旧E0 gap文件保留时间点证据，现行优先级由[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)及ACTIVE控制。

**研究权限**：[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)记录10个source、12条source×seed完整轨迹及1920个嵌套FFN观测，以及已完成的固定实验范围。源码、文献、数学、已有保存证据和图均可用于当前科研；**具体CPU/GPU、模型与视频实验按 ACTIVE_RESEARCH 当轮范围、数据、资源和授权登记执行**。本导航不单独构成运行授权。既有实验原始RUN、质量与严格候选失败结论按其来源版本保留。

静态检验验证科学入口/证据索引/源锁及图结构一致性；真正的功能因果、数值和视频质量通过各自独立证据判定。[DEFAULT_BEHAVIOR_CASES](DEFAULT_BEHAVIOR_CASES.json)供未来Agent研究能力的独立评估，静态“通过”不代表已经具有相应科研表现。
