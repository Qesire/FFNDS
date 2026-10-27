---
name: ffnds-paper-discovery
description: Apply AutoResearcher science to parallel information, function/dynamics and computation/execution discovery for efficient video FFN research.
---

# FFNDS · Parallel information, function and compute discovery (2026-10-27)

## 研究对象与可扩展目标

本项目的共同科学目标：**研究视频生成FFN中信息、计算与生成行为的组织规律，发现更低物理代价实现高质量生成的表示、函数机制或执行结构**。机制科学与部署方法都是可独立成立的成果。**I信息与表示、F函数与动力学、C计算与执行并列且互相反馈**；从实际现象、数学恒等式、源实现、原论文或性能现象任意起步，形成OPEN假说并逐步判别。详细[统一目标](../../../research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。

固定原生基准：LightX2V Wan2.2 I2V A14B 四步 D-Q，high/high/low/low 两套独立专家、40 Transformer blocks。典型 FFN：
\`norm2/AdaLN → Smooth1/Q1 → W1 → GELU(tanh) → Smooth2/HiF4 Q2(64→8→4) → W2 → e5门控BF16残差 → 后续block/flow/scheduler/video\`。
现有原生实现采用 HiF4 fake-QDQ 与 BF16 Linear；外部原生packed NVFP4/MXFP4算子属于独立比较对象。

当前入口：
- [统一目标与I/F/C研究路径](../../../research/RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)、[问题导航](../../../research/RESEARCH_NAVIGATION.md)
- [G13输出函数与实际计算结构](../../../research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P函数作用与来源定位](../../../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[C01–C03提案](../../../research/agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)
- [行为／函数／结构独立索引](../../../research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)
- [完整G0–G13图](../../../docs/98_端到端计算图与证据注解_20261025.md)与[机器绑定](../../../research/agent_protocol/GRAPH_AGENT_BINDINGS.json)
- [机制交叉索引](../../../research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)
- [当前综合与现行证据](../../../research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[ACTIVE权限](../../../research/ACTIVE_RESEARCH.json)

## 实验采样与校准的可解释证据

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

### 历史A5+HWG提醒：视频时空功能的科学问题

[A5+HWG背景闪烁与小物体模糊的重审](../../../research/20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)提供一个适合NOTICE→DERIVE→DISCRIMINATE的开放样例：主体结构基本保持时，静态/运动背景闪烁与小目标模糊仍可能发生。将**空间细节**（小物体、纹理）、**视频时间相干**（运动补偿后的背景光度/结构）、**真实动作幅度**分开；四个去噪step与81帧输出不能混为同一时轴。对逐token mask，区分阈值切换与长期弱信号固定遗失；若mask在HiF4 Q2前，还需原64组尺度连带与BF16，若postQ2则不能把preQ2效应当作已测因果。现有A5是历史视频定性记录，Affine54/UB1的后续VIDEO/RAFT另有来源身份。视频功能研究可使用G13因果定位，I/C路线也可直接研究时空共享信息与2:4物理算子；**不以找到语义神经元、发明新的loss或独创方法为推进门槛**。

### UB1外观—运动分离的研究范例（原版RAFT证据）

[UB1持续运动幅值衰减一手数据与机制重审](../../../research/20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)说明：在两个已暴露formal20来源，UB1相对Dense的原版top5连续RAFT幅值为38.13%/9.19%，40对中38/40、40/40更低；四原版Dynamic Degree binary均0，原20 Motion Smoothness接近。Skill的NOTICE对象因此是**外观近似而相对运动测度衰减**，不是输入动态方法失败或所有prompt违反动作语义。MAP_FLOW必须恢复Wan四denoise-step与21 latent视频时间/81 RGB帧的不同轴，以及UB1原64组静态预算(54/108/108/162)+R256和full-FFN成本。DERIVE可用冻结线性时间差的signed cross，不得将其套在非线性的RAFT上；REFORMULATE用VideoJAM/Motive等已有appearance-motion原理为成熟比较器。DISCRIMINATE优先区分**真实对象运动变化 vs 纹理模糊令RAFT难追踪**，其次竞争固定组丢失稀有动态能力、校准24source偏置和Attention/去噪状态传播。F线因果定位有其价值，I/C可独立探索动态所需信息/实际省算；不要求寻找“运动专属神经元”、发明新loss或把静态UB1强行改成token动态路由。录像/模型作业依ACTIVE精确范围。

## 文献、计算图和协作回写

广泛搜索基础/前沿工作和相邻学科，以**原论文方程→训练/执行实际动作→作者代码分支→适用模型/系统形状→竞争性机制**记录一手阅读深度。对G7–G12已验证论文保留其source ledger：[G11核心深读](../../../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[G12原理](../../../research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)、[G13因果与干预方法原文L80–L89](../../../research/literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)。先分析对应的实际机制、信息或功能角色，再补图节点与**有类型**对照边；原始作者结果在LIT层保留、现行原生证据在SRC/NATIVE层保留。

独立研究Agent可以分别提出信息/表示、行为函数、计算执行和检验思路，写入[讨论队列](../../../research/method_discussion/QUEUE.md)；独立审阅以公式、源码、证据和真正不同的预测支持/质疑；串行整合者更新综合报告、图谱及研究登记。旧方法的明确失败边界作为研究资源，已关闭RUN和哈希保持原样。图节点可以由真实新问题生成，无须从已列候选中选一个。

## 状态与运行时资源

截至2026-10-27，原生覆盖 **10个source身份、12条source×seed轨迹、1920个嵌套FFN位置**。旧候选的局部误差与实际视频失配已有明确观察；G13功能–结构的针对性因果定位仍在探索阶段。

**当前活动（2026-10-27恢复）**：按用户本轮明确指令持续推进I/F/C研究及冻结登记后的有限实验，同时最多使用四张空闲GPU，简单委派使用GPT‑6.1 Terra；[ACTIVE_RESEARCH.json](../../../research/ACTIVE_RESEARCH.json)保存当前范围与真实状态。旧暂停、已完成范围、原视频质量和每次独立评估保持历史身份；实时/流式实验未恢复。

质量/部署论断最终以[统一全深度视频指标协议](../../../research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)的same-source/seed、全40层×4步、gate×W2、完整视频/系统指标和物理成本支持。纯数学与源码研究按其自身证据等级交付，无须为每个探索种子制造完整生成任务。
