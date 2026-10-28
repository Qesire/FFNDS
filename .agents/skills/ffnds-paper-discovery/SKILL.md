---
name: ffnds-paper-discovery
description: Apply AutoResearcher science to parallel information, function/dynamics and computation/execution discovery for efficient video FFN research.
---

# FFNDS · AutoResearcher 机制方法 Skill（更新至2026-10-28）

## 开始研究前的阅读顺序与工作对象

**最短阅读链**：[P0–P4统一阅读优先级](../../../research/RESEARCH_NAVIGATION.md) → [现行M-A/M-B/M-C整合方案](../../../research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) → 当前研究对象对应的[真实计算图G14](../../../research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)或[原生数据协议](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)。本Skill约束**科学方法与证据表达**，进度数字、后续任务及历史实验以P0–P4入口和[ACTIVE](../../../research/ACTIVE_RESEARCH.json)为准，不在每次更新时重复抄写。

研究北极星：解释视频FFN的信息、函数及真实计算组织规律，产生可证伪的科学机制或维持完整视频能力的低成本表示与执行。[I/F/C并行目标](../../../research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)允许从有趣的数学恒等式、原生异常、真实行为、论文或kernel任一端开始。**O-A计算选择、O-B函数替代、O-C历史复用、O-D代数/原生执行、O-E学习/验证**是[可组合操作语法](../../../research/20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)，不是五个互斥方法。论文与新方法的typed关系收录在[机器交叉表](../../../research/agent_protocol/METHOD_OPERATION_CROSSWALK.json)。

**2026-10-29 · 视频级科学准入（覆盖下文过时的B末层任务顺序）**：[全深度×多来源×视频优先方法合同](../../../research/20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。从已完成UB1的20个source真实质量与Affine54 320site状态数据出发，所有待排序的部署候选先说明40×4全160site的可执行动作，并给出≥20独立image-cluster的Dense配对完整视频、运动/动作/主体/背景/时序/成像/语义来源级向量和FFN/整机成本；正式确认须未暴露来源并按官方独立seed协议冻结。末层head/范数修正、局部SSE、MAC/token是创造和机制定位工具，不再作为新方法优胜的默认首步。源/step/block/token嵌套不能充作来源规模。只开展ALG/SRC时可提出全深度设计，但不要以局部改善宣称部署视频有效。

**双数值原生实验轨（2026-10-28最新指令）**：[A100原始BF16与RTX PRO 6000 Blackwell NVFP4并行](../../../research/agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md)。没有本地RTX5090，但RTX PRO 6000可作为SM120原生FP4目标；BF16功能/视频与原生NVFP4源码/准备任务同时READY，互不阻塞。PRO6000 R0先核验真实可访问权限/MIG/环境及匹配模型，保留历史E0在Pro6000跑过的证据；后续native作业先绑定ACTIVE。当前上游nvfp4_rtn_backend.py 是 Fake-QDQ后接BF16线性算子，不等于packed W4A4，必须区分。HiF4 Fake-QDQ仅保留为格式/数值机制诊断。

**量化operand强类型规则**：**W1/W2权重是离线静态量化**，本次FFN的**Q1/A1、Q2/A2激活各自在线动态量化**；Q1在W1前，Q2在W1/GELU后。原生NVFP4/MXFP4的W4A4分别有W和A的packed code及block-scale，A scale动态求取（NVFP4的global激活scale可按recipe预校准），原生scale存在但是否可免费/及时读到取决于kernel布局。W4A16没有A FP4 scale。每个研究方案必须指明operand role、producer、scale来源与决策截止；固定weight信息不能被称为输入动态证据。[真实源和成熟后端审计](../../../research/20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)。

**HiF4事实边界**：每个4-bit **S1P2码字自带1bit符号**，64元素共享32bit unsigned尺度元信息；不要把“scales不含符号”误说成“HiF4没存符号”。现行CUDA fake-QDQ将带符号量化重建为BF16而不持久导出packed码字，相关研究必须计入提取signed/position摘要的真实代价。参见[格式审计](../../../research/20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)和[机器合同](../../../research/agent_protocol/HIF4_SIGNED_FORMAT_CONTRACT_20261028.json)。

原生基础：LightX2V Wan2.2 I2V A14B四步high/high/low/low，两套expert、每套40 blocks；`norm2/AdaLN→Q1→W1→GELU(tanh)→HiF4 Q2(64→8→4)→W2→e5 gated BF16 residual→downstream flow`。当前原生实现为fake-QDQ＋BF16 GEMM，新的packed NVFP4/MXFP4属于需要另验证的算子路径。

**资源调度（2026-10-28最新指令）**：按[现行动态GPU政策](../../../research/agent_protocol/DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)使用实际可申请并运行的最多GPU；没有预设全局张数限制。旧2/4卡可属于历史实验的冻结字段；新scope按实际可调度量和ACTIVE冻结模型/来源/预算；不要凭旧上限拒绝并行的独立已注册工作。

**当前指定继续M-B（方法研究）**：复用B0/CAL24/B-H/A0/B-F现存数值与20-source UB1、320-site Affine54以及A0六视频的真实证据；**首个可排序方法实验**从全160-site可执行函数、≥20独立image clusters同源Dense完整视频、运动/画质/主体/背景/时序/语义和真实FFN/全wall共同决策。末层L39/head/gain保留为推导、解释和诊断；它单独的成功不提升全模型部署方法状态。开发历史曝光与未暴露确认严格分离；R0原生NVFP4并行。最新合同见[视频先行](../../../research/20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。

已完成机制证据复用：[B-H Direct65](../../../research/20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)、[同head A0](../../../research/20261028/BH_A0_MATCHED_TERMINAL_HEAD_RESULTS_20261028.md)、[B-F](../../../research/20261028/BF_FROZEN6016_TERMINAL_RESULTS_20261028.md)。这些已完成pilot保留原范围，由现行全深度多来源视频合同决定后续方法排序。

## 实验采样与校准的可解释证据

**跨方法同预算和同统计对象**：UB1/固定RR、Affine54、激活2:4与训练后表示均可利用[Calibration V2](../../../research/20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)的独立image-cluster配额、teacher/student同坐标、真实QDQ、完整自由视频和成本；每条路线自己冻结estimand、source exposure、训练预算、数值插入点，互不继承他人的“已验证质量”。

涉及采样/校准/视频评测或声称泛化时加载[实验数据Skill](../ffnds-experiment-evidence/SKILL.md)和[统一协议](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)。先明确独立source与嵌套seed/step/block/row，接着复核旧cohort的拟合/择优暴露、原生QDQ/student状态、真正独立的video run及各维覆盖。历史VBench五文件名对应每source一个真实生成，和官方五次采样分开报告。把证据不足转为**有辨别力的新增数据设计**，将可靠源码和既有原生现象作为构造性实验起点。

## 默认研究思路：三种入口、共同判别、分别验证

**I · 信息与表示（DISCOVER）**：从已知权重、当前输入、数值编码、跨token/step状态或历史信息发现条件性结构；给出信息首次可用的时间、获取价与信息利用机制。允许直接探索编码域内积、有限码本、低秩统计、信息证书及无需语义标签的算术。

**F · 函数与动力学（DISCOVER / MAP）**：从自然视频/latent/flow现象和无标签signed响应寻找有意义的函数，沿W1→GELU→HiF4 Q2(64→8→4)→W2→e5写回和下游attention/step定位正常计算承担者；这条专门路线使用[G13](../../../research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)与[O/D/Q/P](../../../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)。FUNCTIONAL-OBS / CAUSAL / IMPLEMENTABLE是这类功能主张的证据层级。

**C · 计算与执行（DISCOVER / BUILD）**：从真实M/D/H shape、有效GEMM/QDQ、内核、访存与时延出发研究算子重写、低位dense、稀疏/共享计算、时空复用及轻量后训练后的新函数结构。**I/C线不需要先在原模型中找到语义功能子图**；性能异常本身也可倒逼新信息或动力学解释。

**DERIVE / EXPAND**：调用research-expansion、problem-reformulation及mechanism-research，用新坐标、不同训练状态、替代动作空间或时空尺度构造竞争解释。仅针对当前关键命题重建完整的决策相关源码路径和数值语义，确认必要的输入、输出和依赖。

**DISCRIMINATE / ADVANCE**：用可区分的数学证明/反例、原始论文与作者源码、已有证据及未来被授权的定点实验推进。功能因果论文使用正常路径忠实性；经济计算论文使用前置可得信息、强同条件基线、真实物理开销；宣称视频级改善必须加上完整free-rollout/正式质量与系统速度证据。无需为了探索种子先提供整段视频。

## 科学写法：积极构造、明确对象与证据

默认陈述顺序：**有趣的结构或现象 → 对应数学/真实机制 → 至少一个竞争性解释及不同预测 → 最便宜的辨别证据 → 对信息、功能或执行的启示**。

- 将真实负例写成**作用于指定版本、输入、算子和指标的正面事实**。例：“旧token mask在M16/K64 tile上100%占用，说明现有布局仍执行所有tile；新的候选空间转向联合输出结构或另一合法执行单位。”
- 将来源边界贴近相应结果：LIT=外部作者论文/代码；ALG=数学推导；SPEC=可用的表示定义；SRC=当前Wan实现；NATIVE-OBS/VIDEO/SYS=既有真实实验；OPEN=待判别机制。用证据分级促进新方向，**重复的自我免责声明与限制性前言通常只会遮蔽研究结论**。
- 将单一强先例转为科学资源：Prox有付费INT4代理；CodeGEMM/LUT-GEMM的公开快路集中于单vector；SharQ在线稀疏FP4加稠密残差；G12重排区间对单W1行紧、对共同输入的多行联合函数仍有研究空间。这些事实**促进**新联合函数/新编码表示的构造。
- 每次结论尽可能给出具体源码函数、公式、tensor形状或已经固定的原生RUN引用。源可信度与外部模型身份写清楚一次；保留当前实验未正式通过视频质量验收的事实，并指出具体已测场景。

建议组合：
[mechanism-research](../mechanism-research/SKILL.md)（源码/结构）、
[research-expansion](../research-expansion/SKILL.md)（扩大机制空间）、
[problem-reformulation](../problem-reformulation/SKILL.md)（新数学对象）、 
[web-research-discovery](../web-research-discovery/SKILL.md)（原文、正反先例和源码链）。
从项目[GOAL_ALIGNMENT](../../../core/GOAL_ALIGNMENT.md)、[EXPANSION_PROTOCOL](../../../core/EXPANSION_PROTOCOL.md)、[PROMPT_PLAYBOOK](../../../core/PROMPT_PLAYBOOK.md)按当前研究分歧取用提示操作，作为生成创造性解释的工具，而非合规问卷。

## 历史现象如何转为新的科学问题（例证索引）

- [A5+HWG背景闪烁和小物体模糊](../../../research/20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)：区别空间细节、视频时间相干性、mask时点与HiF4 Q2连带。历史A5的定性现象是OPEN机制种子，不是当前原生视频验收。
- [UB1外观—连续运动衰减及因果审计](../../../research/20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)：用同来源真实视频区分动作幅度变化、RAFT因纹理退化变弱、固定组选择、来源偏置与下游传播，选择有相反预测的最小判别；旧两来源formal20与后续20来源consumer分开计数。

**例证用途**是学习从异常→数学/代码→可证伪竞争机制的方法，而不是要求新研究重跑旧mask、发明运动loss或先定位语义神经元。完整数值和已排除范围读取原报告及[P2证据导航](../../../research/RESEARCH_NAVIGATION.md)。

## 文献、计算图和协作回写

实际发现的节点、类型和原生证据链接回写[Graph Agent绑定表](../../../research/agent_protocol/GRAPH_AGENT_BINDINGS.json)，由静态工具检查引用；真实运行另以ACTIVE及原始receipt核验。

广泛搜索基础/前沿工作和相邻学科，以**原论文方程→训练/执行实际动作→作者代码分支→适用模型/系统形状→竞争性机制**记录一手阅读深度。对G7–G12已验证论文保留其source ledger：[G11核心深读](../../../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[G12原理](../../../research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)、[G13因果与干预方法原文L80–L89](../../../research/literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)。先分析对应的实际机制、信息或功能角色，再补图节点与**有类型**对照边；原始作者结果在LIT层保留、现行原生证据在SRC/NATIVE层保留。

独立研究Agent可以分别提出信息/表示、行为函数、计算执行和检验思路，写入[讨论队列](../../../research/method_discussion/QUEUE.md)；独立审阅以公式、源码、证据和真正不同的预测支持/质疑；串行整合者更新综合报告、图谱及研究登记。旧方法的明确失败边界作为研究资源，已关闭RUN和哈希保持原样。图节点可以由真实新问题生成，无须从已列候选中选一个。

## 实际执行和证据身份

对模型/原生CPU-GPU/视频运行，先检查[ACTIVE_RESEARCH.json](../../../research/ACTIVE_RESEARCH.json)当轮登记的模型、source、代码、GPU资源和预算；历史已完成scope及过去的暂停/恢复状态留在原始RUN。实时/流式模型实验仍须独立授权。视频质量与系统成本按[全深度指标协议](../../../research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)验收；纯数学/源码/文献研究按自身ALG/SRC/LIT证据完成。

## G14 全文献逐计算边聚焦（现行索引）

提出和评议新的 FFN 方法时，先按[机器G14逐论文映射](../../../research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)查论文作用节点、需要的前置信息、真实选择/训练/缓存/计算行动、原文核验深度和与冻结Wan的异同。当前已收录98个主LID及10个补充校准研究，分组Mermaid见[全计算图G14](../../../docs/98_端到端计算图与证据注解_20261025.md)。`python tools/graph_agent_bridge.py paper L74`检索作者方法与原生锚点，`focus T06`检索针对相同Q2阶段的替代论文。先从作者明确实际做了什么总结方法贡献，形成同一算子多方法竞争假说；以原Wan输入state/shape、HiF4 QDQ及消费者数据判断可迁移性。G14的LIT类比不会成为真实本地跑通的NATIVE/VIDEO/SYS。索引增删更新原始来源、机器映射与可见图，并执行`graph_agent_bridge.py validate`。
