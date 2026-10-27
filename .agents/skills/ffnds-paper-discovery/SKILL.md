---
name: ffnds-paper-discovery
description: Apply AutoResearcher science to locate meaningful video-FFN output functions, identify their actual computation structures, and develop evidence-grounded efficiency ideas.
---

# FFNDS · Function-led scientific discovery (2026-10-27)

## 研究对象与可扩展目标

本项目的科学目标：**发现视频生成FFN中具有行为意义的输出函数及其实际计算承担结构，研究这些结构如何以更低的真实计算成本实现，同时保持完整视频功能**。探索以问题与真实证据驱动，允许从新数学、新论文、视频现象或源码异常提出大胆、可证伪的机制；新机制可以先以 OPEN 登记。

固定原生基准：LightX2V Wan2.2 I2V A14B 四步 D-Q，high/high/low/low 两套独立专家、40 Transformer blocks。典型 FFN：
\`norm2/AdaLN → Smooth1/Q1 → W1 → GELU(tanh) → Smooth2/HiF4 Q2(64→8→4) → W2 → e5门控BF16残差 → 后续block/flow/scheduler/video\`。
现有原生实现采用 HiF4 fake-QDQ 与 BF16 Linear；外部原生packed NVFP4/MXFP4算子属于独立比较对象。

当前入口：
- [G13输出函数与实际计算结构](../../../research/20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P函数作用与来源定位](../../../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[C01–C03提案](../../../research/agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)
- [行为／函数／结构独立索引](../../../research/agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)
- [完整G0–G13图](../../../docs/98_端到端计算图与证据注解_20261025.md)与[机器绑定](../../../research/agent_protocol/GRAPH_AGENT_BINDINGS.json)
- [机制交叉索引](../../../research/agent_protocol/MECHANISM_GRAPH_CROSSWALK.json)
- [当前综合与现行证据](../../../research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)、[ACTIVE权限](../../../research/ACTIVE_RESEARCH.json)

## 默认科学思路：功能 → 承担结构 → 低成本实现

**一、发现真实功能（DISCOVER）。** 从自然视频/latent/flow现象，以及有符号FFN写回的条件响应，提出行为功能候选。观察轴可以是运动、相机、主体、背景、照明、细小物体和长期稳定性，也可以是未命名的联合有符号函数。以 \`source × seed × step × expert × block × token\` 限定条件。区分 FUNCTIONAL-OBS（可追溯的相关现象）与未来经合法干预支持的 FUNCTIONAL-CAUSAL（条件性的真实贡献）；探索阶段的候选可以只有一个引人注目的现象及不同预测。

**二、定位计算承担者（MAP）。** 沿真实源码/配置/张量/数值顺序重建该函数的路径：共享q上的多W1行、GELU、HiF4 Q2整组尺度、W2正负组合、门控写回、后续attention/深层残差、四步状态与两套expert。研究联合协作、互相抵消、条件分工、补偿通路、跨token/跨帧/跨层/跨step的功能结构。**功能身份可以是分布式与条件性的**；观察手段与真实承担计算的子图分别建档。

**三、扩展机制（DERIVE / EXPAND）。** 寻找能改变理解的数学关系、可预测不变量、联合函数、非线性交互、执行单位与共享计算。以最强竞争机制提出能观察到不同结果的预测，研究者可以独立探索低维表示、时间尺度、物理重编码、训练职责迁移或其它新方向。**量化code/sign/scale、Proxy、TopK/SAE、补算、Cache、稀疏tile与LUT是定位后可研究的信息与执行手段**，允许发挥启发作用，并由其实际数据时点与成本决定物理价值。

**四、判别并推动（DISCRIMINATE）。** 选取最低成本、有决策意义的一项或多项证据：代数/反例 → 原始源码与论文 → 已保存数据 → 经单独授权的原生局部干预 → 完整视频质量与真实加速。必要性、充分性、补偿/备用路径和normal-forward忠实性分别检查，避免把由干预触发的新通路与日常计算职责混为一谈。测试次数取决于科学分歧；早期探索保持广阔候选空间。

## 科学写法：积极构造、明确对象与证据

默认陈述顺序：**有趣的结构或现象 → 对应实际函数/数学 → 至少一个竞争性机制及不同预测 → 最便宜的辨别证据 → 对计算图/视频目标的启示**。

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

## 文献、计算图和协作回写

广泛搜索基础/前沿工作和相邻学科，以**原论文方程→训练/执行实际动作→作者代码分支→适用模型/系统形状→竞争性机制**记录一手阅读深度。对G7–G12已验证论文保留其source ledger：[G11核心深读](../../../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[G12原理](../../../research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)、[G13因果与干预方法原文L80–L89](../../../research/literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)。先分析对应的实际机制和功能角色，再补图节点与**有类型**对照边；原始作者结果在LIT层保留、现行原生证据在SRC/NATIVE层保留。

独立研究Agent可以分别提出新函数、计算结构和检验思路，写入[讨论队列](../../../research/method_discussion/QUEUE.md)；独立审阅以公式、源码、证据和真正不同的预测支持/质疑；串行整合者更新综合报告、图谱及研究登记。旧方法的明确失败边界作为研究资源，已关闭RUN和哈希保持原样。图节点可以由真实新问题生成，无须从已列候选中选一个。

## 状态与运行时资源

截至2026-10-27，原生覆盖 **10个source身份、12条source×seed轨迹、1920个嵌套FFN位置**。旧候选的局部误差与实际视频失配已有明确观察；G13功能–结构的针对性因果定位仍在探索阶段。

**当前活动**：论文/源码、数学、保存证据阅读、功能与结构整理、图谱/科研Agent讨论。**模型CPU/GPU/视频新执行暂停**，未来需要用户新的明确授权；[ACTIVE_RESEARCH.json](../../../research/ACTIVE_RESEARCH.json)负责最终执行状态。原视频质量结果及每次独立正式评估继续保持真实记录。历史四GPU并发预算属于已经完成的授权范围。

质量/部署论断最终以[统一全深度视频指标协议](../../../research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)的same-source/seed、全40层×4步、gate×W2、完整视频/系统指标和物理成本支持。纯数学与源码研究按其自身证据等级交付，无须为每个探索种子制造完整生成任务。
