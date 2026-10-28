# FFNDS · AutoResearcher 科研Agent默认约定

本文件只规定科研行为及权限。**唯一当前阅读入口**：[RESEARCH_NAVIGATION](research/RESEARCH_NAVIGATION.md) → [CURRENT_CONTEXT_PACKET](research/agent_protocol/CURRENT_CONTEXT_PACKET.md) → [资产先行方法推导](research/20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md)。科学北极星由[统一I/F/C目标](research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)定义；[ACTIVE](research/ACTIVE_RESEARCH.json)是实际CPU/GPU/model/video运行范围的唯一注册依据。原[M-A/M-B/M-C整合研究方案v4](research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)保存其适用数学和旧实验，作为按需专题，不规定当前全局方法顺序。

## 当前方法研究的默认行动

1. **开放式机制发现与证据获取**：从真实源码、论文、数学结构、已测视频或现有资产任一入口提出科学问题。旧资产优先复用以减少重复；**一旦需要未采集的source、时空匹配token、全深度状态、student轨迹、原生FP4动态量化元信息或其他变量，主动设计并扩展新的真实采集**。所需数据和采集范围由问题决定，不以旧8源/20源、5个Block、256行、固定rank或S1–S4为永久边界。
2. **提出并检验竞争机制**：为同一个发现给出至少两种具有不同预测的机制解释；先用能直接判别的代数、反例或已有原生证据开展研究；**若缺数据，则首先补采有区分力的新source/状态/计算边界**，而不是降低问题规模或调整结论。新数据也可以用于探索此前未被观察到的结构。
3. **扩展动作而非锁定算子**：[S1跨步共享函数＋S4行动价值变秩＋S2时空函数单元＋S3量化前置合法组动作](research/20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md)是**并行研究分支**，可增删新机制，互不强制依赖；历史M-B末层、QCFB组间符号抵消、补算、gain均为专题工具或强对照。执行前写清可用信息截止、合法的物理行动、真正改变的W1/W2/量化支路和成本。
4. **结构发现与部署验收分层**：ALG/SRC/已存OBS可以产生独立科研发现；若声称可用的低计算方案，必须在真实4步×40层160位置学生自由生成上，用多来源、同checkpoint/数值recipe的Dense配对完整视频、动作/运动/语义/时间/画质及实测FFN/全系统成本证明；历史暴露来源用于开发，冻结的新来源及独立seed用于确认。参见[视频证据合同](research/20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。
5. **有需求就持续扩展采集**：按[问题驱动新增资产协议](research/agent_protocol/EVIDENCE_DRIVEN_ACQUISITION_20261029.md)与[采集需求队列](research/agent_protocol/ACQUISITION_QUEUE_20261029.json)，把缺失观测、来源分层、全40层×4步与动作/语义影响转为具体producer和下一批新source；一次有限范围结束后允许基于新的科学问题继续注册采集，避免把历史配额当科研天花板。
6. **真实研究资源**：CPU/GPU、来源、原始/量化checkpoint、seed、模型状态和最大预算按[ACTIVE](research/ACTIVE_RESEARCH.json)冻结每个新的有限范围，任务文档不自动授权，也不自动重跑失败或已完成范围。GPU使用实际可调用的最大有效数量，按[动态资源政策](research/agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)分配，不继承旧实验2/4卡历史上限。
7. **写回与协作**：科学主张按ALG/SRC/LIT/NATIVE/VIDEO/SYS分开；来源身份与negative evidence精确保留，复算另行版本化。[独立Agent方法讨论队列](research/method_discussion/QUEUE.md)只登记真实提出者和独立评议，不预填同意票。新研究结果同步计算图与当前任务，避免复制一套新的“权威进度文档”。

**数值机制事实**：原BF16路径没有激活FP4 Q1/Q2；旧HiF4 fake-QDQ有带符号S1P2码字、BF16重建与W1前Q1/W1后Q2；原生NVFP4/MXFP4的静态W scale、动态A scale、packed layout和实际生产时点独立验证。已有[权重/激活时序审计](research/20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)与[HiF4有符号码字审计](research/20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)。A100 BF16与RTX PRO 6000原生NVFP4可并行研究；可用硬件与可执行范围以实际核查为准。

## 一、科学研究的默认语言与思路

研究回答以**构造性内容**开始：一个真实现象、特别的数学依赖或新的功能假说；随后给出最强竞争机制、彼此不同的预测、代价最低且有辨识力的下一证据、与视频功能和真实执行的联系。

**表达原则**：
- 用**“这个数据说明什么、揭示什么自由度、允许怎样的新机制”**取代连串“不能/不是/没有”开头的免责句。数学证明和实验反例保留准确的对象、前提、位置与指标，负面结果帮助发现另一可探索的表示或行动单位。
- 像“后验token级稀疏在原M16×K64物理tile仍100%占用”这样的具体结论应明确保留，并由此提问联合函数分组、不同执行颗粒、权重/状态共享是否可改善。将失败范围写成**已检验结构的有界结论**。
- 对未完成的科学问题积极提出OPEN机制及可证伪预测。对已测量的数值、作者论文和原生运行分别贴上ALG/SRC/OBS/NATIVE/VIDEO/LIT证据来源；精准的证据身份足以校准结论。
- 从自然视频、无标签有符号关系、信息界、数学恒等式、真实算子或文献任一入口提出机制。TopK、SSE、proxy、FP4格式、补算和特定kernel均是可比较的手段；**功能定位可以启发执行方案，但不是I/C线开始研究的前提**。

## 二、并行入口与共同研究循环

**I · 信息与表示**：考察信息第一次可用的时点、在线观测与离线编码的成本、共享变量、条件依赖与可证明界。允许直接研究编码域计算、状态复用和新数学表示，并对历史oracle/已付费信号严格区分。

**F · 函数与动力学（G13专题）**：根据自然source×seed×step×expert的视频/latent/flow响应提出行为轴与无标签signed函数，定位真实W1/GELU/HiF4 Q2/W2/e5及下游attention/flow的承担结构。FUNCTIONAL-OBS、FUNCTIONAL-CAUSAL和FUNCTIONAL-IMPLEMENTABLE是此专题内不同主张的证据层级；[O/D/Q/P](research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)为具体干预工具。

**C · 计算与执行**：从实际shape、非线性代数、kernel、布局、GEMM/QDQ/访存及训练后参数结构提出可运行的新算子，研究真实省算与质量速度Pareto；允许先展示有条件的成本规律，再追究功能解释。

**DISCOVER / DERIVE / EXPAND**：用[research-expansion](.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](.agents/skills/problem-reformulation/SKILL.md)和[mechanism-research](.agents/skills/mechanism-research/SKILL.md)发展新的数学坐标与竞争解释。按照当前具体主张重建决策相关的真实源码计算路径；新函数定位、新算子和新信息理论可各自新增图节点。

**DISCRIMINATE / BUILD / CLAIM**：选择最有区别性的证明、源码、原论文、现存证据或将来获准的实验。科学主张按其证据证明；方法的可部署性另外核实同等前提的真实全FFN/系统成本与完整自由生成视频质量。已关闭实验的版本/结果保持原始身份。

**ADVANCE**：G13因果候选进入Function–Structure档案；I/C的新信息、表示或算子进入独立机制发现与[Graph Agent Bindings](research/agent_protocol/GRAPH_AGENT_BINDINGS.json)，无需先写成现成FUNCTION。原论文读深与作者源码留在source ledger，跨机制共识留在综合报告。

## 采样、校准与质量证据的默认行为

[数据/采样/校准/评测统一协议](research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)与[专用实验Skill](.agents/skills/ffnds-experiment-evidence/SKILL.md)适用于涉及独立source、采样行、Student/Dense状态、校准拟合、视频随机数及结果泛化的新任务。先确认研究对象和独立观察单位，再匹配成熟VBench/Wan/压缩实践、复核历史暴露、冻结对照、指标和真实可重用资产。历史UB1一个视频复制五个VBench文件名属于单seed自定义历史比较；新官方五采样必须是五次真实独立生成。以已有结果揭示的新机制、可区分预测和最小有效实验作为正向引导，证据等级贴着相应主张。

## 新实验的具体执行入口

[可执行实验手册](research/agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)给出真实命令、输入/输出和验收路径。新cohort依次使用 `ffnds_experiment_plans.py`、`ffnds_native_geometry.py`、`ffnds_validate_calibration_operands.py`、`ffnds_run_registered_video.py`、`ffnds_stage_registered_i2v.py` 和 `ffnds_cluster_metric_report.py`（seed×sample重复观测→source/prompt等权→独立图像簇等权）。原生模型运行检查活动研究授权及本实验注册的run IDs；校准记录必须携带真实行坐标与操作数header；多样本视频经过逐run完整验证后再交原版VBench。数据和质量主张按各自真实RUN标记，主动给出能够区分机制的下一步，而不是把静态计划写成已测结果。

## 分目标校准 · 2026-10-27 新实验优先规则

[Calibration V2 数学与实操快照](research/20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)和[冻结设计](config/calibration_design_v2_20261027.json)定义支持选择、gain/bias、rank256、Student状态、完整视频不同 estimand。对新研究的校准判断以source-cluster有效性和heldout/原生功能证据决定采样扩容，旧24×256是冻结基线预算。先执行 `ffnds_calibration_design.py coverage/diagnose`，数据配额、teacher/student真实坐标和历史暴露依各自新版本注册；缺数据状态保持NOT_MEASURED，提出明确producer。结论从观察到的有价值现象与区分性解释出发，支持、误差和时空视频结论分别定位。

## 三、原生证据、论文及真实成本

- **原生**：[实验发现谱系](research/EXPERIMENT_FINDINGS_ATLAS.md)、[证据优先级](research/EXPERIMENT_EVIDENCE_INDEX.md)、[已有full-flow finite-map](research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)。source425相同图片prompt跨seed的P/N cross符号改变和部分频带与全域交叉项异号，是函数定位的现实入口。过去的Affine54等候选已经测到局部误差/速度及视频失配，作为有界机制证据保留。
- **文献**：按[广域原文检索](.agents/skills/web-research-discovery/SKILL.md)探索基础/前沿及相邻论文，核对公式、默认实际源码路径、消融、训练、硬件与任务身份。[98项索引](research/literature/20261026/PRIMARY_LITERATURE_INDEX.json)提供LID和阅读层级，[G14全108论文机制映射](research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)提供算子位置、信息可见时点、真实动作和迁移判别，[竞争矩阵](research/literature/20261026/MECHANISM_COMPETITION_MATRIX.md)和[Crosswalk](research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)提供对照。
- **机制与代价**：量化code/FP4 signed直方图的理论界（G12）、Prox付费代理、SharQ两路GEMM、CodeGEMM/LUT-GEMM单vector快路、训练后稀疏、step/history cache、实时因果系统都可成为候选的真实实现路径。模型对象、首次信息可用时间、online observer/metadata/cache/packing价格和完整FFN/video成本共同确定其物理意义。
- **质量与运行时**：[全深度与完整视频质量规则](research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)维护same-source/seed、40层×4步、gate×W2、质量、latency/setup/显存与SHA。独立的正式VBench/人工验收按实际运行标记。外部作者成绩依其自身模型、硬件、任务登记。
- **静态工具**：\`python tools/graph_agent_bridge.py validate\`检查节点、边、论文、机制家族；\`mechanisms\`、\`mechanism M13\`支持只读索引。静态一致性属于文档/代码协议级证据，数值真实性与视频因果作用仍由相应对象证明。

## 四、多Agent独立方法讨论

所有Agent继承[项目领域Skill](.agents/skills/ffnds-paper-discovery/SKILL.md)及[讨论约定](research/method_discussion/AGENTS.md)。独立提出函数或结构、新机制、数学或原始源码对照，接着由其它Agent发表独立支持、具体反例或替代解释；通过[QUEUE](research/method_discussion/QUEUE.md)保留方法版本和独立审阅。\`SUPPORT\`、\`PEER_SUPPORTED\`、\`CROSS_CONFIRMED\`、\`CONTESTED\`等标签针对**确切版本**，表达讨论证据，不是实验通过或部署结论。

图是实际研究对象的导航而非运行授权，既有RUN和source SHA保持原始身份；新图节点、研究假说和读文献可以独立展开，随后将最有信息价值的区别回写到[Graph↔Agent协议](research/agent_protocol/GRAPH_AGENT_COLLABORATION.md)。

## 五、当前研究权限与交接

**运行授权核验与动态GPU调度（2026-10-28现行）**：I/F/C科研及登记后的有限CPU/GPU/原生实验继续。GPU**不提前设置2/4张等任何固定全局上限**，按[动态资源调度规则](research/agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)在每次启动时尽可能使用实际可申请/调度/获批且能投入独立READY任务的全部GPU。原先2/4张属于历史授权和冻结scope，不再是当前全局限额；简单委派按现行Agent可用规则执行。协调者在运行前冻结具体问题、对照、代码/资产和预算并登记[ACTIVE_RESEARCH.json](research/ACTIVE_RESEARCH.json)，按实际结果继续下一项有辨别力的工作。旧暂停、已完成范围和失败记录保持历史身份；本轮恢复不自动重跑旧失败，也不扩展实时/流式实验。

交接只需**新机制/功能发现、证据出处、真正分歧、对应图节点、最小后续判别和已完成的原生范围**。历史文档当时的“Next/Active/Run”用于溯源，由现行研究导航与ACTIVE区分当前状态。新研究以构造性内容优先、以真实证据调整确定度。

**G14 文献落图强制同步**：新增论文或深化读过的方法，先保留原文、公式及代码深度，再为对应论文登记稳定G14焦点及原生anchor，附真实方法动作和信息时点，通过 `python tools/graph_agent_bridge.py validate` 验证索引、Mermaid和边。对应G0–G13边仅为 LIT 作用映射。
