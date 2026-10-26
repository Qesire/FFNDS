# FFNDS · 从 SSE-first 研究路线转向生成机制发现（2026-10-26）

> **当前科研引导：MECHANISM_FIRST_DISCOVERY。** 2025–2026 原论文/会议核对，完成对先前“global SSE → signed band SSE → 增样拟合”的研究顺序纠正。此文件是科学问题与设计更新，不改变冻结观测、运行收据、原始视频/质量身份；本轮没有执行新 FFN 模型。

**已合并的新证据审查（2026-10-26）**：[四anchor固定基、neighbor12独立判别与原生SM120 packed基元的机制归类](E0_NEW_WORK_MECHANISM_AUDIT_20261026.md)。四坐标非线性族在128个已暴露行上获得样本内表示信息，尚未证明它可预测另外12个neuron；Blackwell合成packed FP4仅证实可执行基元。两条线索分别产生功能共享与真实数值执行的可证伪问题，均不自动成为研究唯一方法。

## 0. 研究目标与真正的第一问

在 Wan2.2 I2V 四步蒸馏原生计算中，发现 FFN 哪些计算承担不可替代的生成**功能**，哪些具有可以实际执行的结构性冗余，以及在何种输入、层、expert、去噪step和时空条件下成立。把发现进一步转化为可实现的 W1/Q1/GELU/Q2/W2 低成本条件计算或重新表示，最终比较真实 FFN/整网 wall 与多维完整视频质量。

**先寻找冗余的起源与它的可预测条件，之后构造对应算子与因果判别；SSE属于实验仪表，不决定科研对象。** 一套方法可能减少矩阵乘、一套方法可能通过共享有限函数获得同等生成作用、还有方法需要少量后训练。当前不预选补算、通道稀疏、低秩、MoE或量化格式。

此前两个source的318/320局部误差下降、全dense有限状态响应、末步视频对照、互斥频带signed cross都是真实且有价值的结果；它们解释**已测Affine54为何未保真**，不构成“所需FFN特征都沿某个误差谱/频段存在”的证明。把global SSE替换成频带signed SSE，再调band、source、权重依旧把问题锁定在**误差重建**上。新来源应用于检验先于实验定义的计算/功能机制，而不被当成解决研究问题的手段本身。

## 1. 前沿路线：先观察哪一类可利用结构，再决定如何计算

| 研究及公开来源 | 真实发现切入点 | 方法/评价对象 | 对FFNDS的可迁移方法论 |
| --- | --- | --- | --- |
| [Sparse VideoGen, ICML2025](https://proceedings.mlr.press/v267/xi25c.html) | 视频attention有空间head与时间head的不同支持拓扑 | 在线profiling、布局重排、真实稀疏kernel；整视频质量/加速 | 先寻找可执行结构类别，再选择动态/静态决策与硬件形状（该论文不处理FFN） |
| [SVOO, ICML2026](https://proceedings.mlr.press/v306/luo26a.html)、[DFSAttn, ICML2026](https://proceedings.mlr.press/v306/hu26p.html) | 部分attention head的稀疏结构具有跨输入稳定性，但细粒度动态模式与block划分存在张力 | 离线分层profiling＋在线QK co-clustering；token重排/分层选块与kernel | 需要分开“可迁移不变性”和“输入可变性”；物理执行形状与决策成本同等重要 |
| [AdaCache, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html) | 视频间运动信息和生成难度异质 | 内容/运动感知cache schedule、端到端质量–时延 | 把可观察的条件信息作为分配计算的信号，而非从离线误差拟合重要性 |
| [TeaCache, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) | 随step变化的输出差可被廉价timestep-modulated输入差部分预测 | 输入侧估计控制cache决策，报告生成质量 | 发现**何时**已有信息可预测昂贵计算，严守推理时信息边界 |
| [Token-wise Feature Caching, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbe024e0517fe12ac3a8d388b19ff9fe-Abstract-Conference.html)、[ProfilingDiT, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.html) | token的reuse质量敏感性、block对前景/背景的功能偏好不同 | token/block选择和真实多维视频对照 | 调查功能分工与跨层异质；背景静态不自动等于可省FFN |
| [TaylorSeer, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.html) | 跨step特征形成可预测轨迹，直接复制不足以处理步长间隔 | 用特征导数预测而非简单缓存；真实生成验证 | 关注真正稳定/可预测的函数坐标及变化，而非只看幅值和误差 |
| [MosaicDiff, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_MosaicDiff_Training-free_Structural_Pruning_for_Diffusion_Model_Acceleration_Reflecting_Pretraining_ICCV_2025_paper.html)、[Diff-ES, 2026 preprint](https://arxiv.org/abs/2603.05105) | 去噪阶段所需容量不同、固定剪枝日程可能不合理 | 时序阶段调度/搜索与完整生成品质对照 | 四步high/high/low/low可分别研究条件功能；不能照搬多步扩散的可重用性假设 |
| [OBS-Diff, 2025 preprint](https://arxiv.org/abs/2510.06751) | 迭代去噪改变二阶结构剪枝的敏感性 | timestep-aware Hessian、FFN neuron等粒度与图像质量 | 二阶加权误差可成为**由动力学机理导出的工具**，但不是用SSE直接发现机理 |
| [TinyFusion, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) | 低即时calibration loss不可靠预测剪枝后微调性能 | 学习“post-finetuning recoverability”剪枝结构 | 如果允许小型后训练，应比较恢复能力，而不是直接优化剪枝瞬间的局部吻合 |
| [Dense2MoE, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html) | Dense FFN容量与每次必要激活量可分离 | 将FFN变成MoE并做多阶段蒸馏；验证实际启用参数及生成质量 | 原FFN可通过**新结构**与条件路由重组；该论文需训练，不等于Wan可直接零训练迁移 |
| [SLA, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/df77cca99e2785eafb5b2416d688ae24-Abstract-Conference.html) | attention包含稀疏高秩重要项、低秩边缘项、可忽略项 | 不同部分用不同算子并融合kernel，少量微调 | 将功能异质性变成不同实际计算形态；注意该论文专注attention，而非FFN |
| [QuantSparse, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) | 稀疏与量化交互会改变特征/注意力传递 | 多尺度显著性蒸馏＋二阶剩余结构利用 | 研究W1/GELU/Q2/W2的耦合，量化不只提供大致重要性先验 |

**论文实际采用多种损失**：有些使用特征相似度、局部重构、Hessian或蒸馏。进步不在于“禁止SSE”，而在于先确定**可压缩对象及其生成作用**，让代理量服务于机制检验/算法实现和真实质量，再去优化。

## 2. 对当前Wan FFN最有辨识力的三个无先验发现问题

### F1 · 条件功能单位来自哪里？

不要先问“删掉某通道造成多少SSE”。先在实际\`q1→W1→GELU→Q2→W2\`下寻找：哪些计算可用一组更少的**推理时可获得的输入坐标**表示？原W1方向、GELU函数曲线、原Q2联合码字/尺度、W2贡献分布、门控写回是否存在按层/专家/条件稳定的结构单元？与可判别的独立坐标/低秩/低复杂度函数空间对照比较。真正的结果可以是“共享原权重投影不成立，但条件数据流形仍支持某些共享表示”。

**最低成本**：重用已保存的六来源Q1/W1/GELU/Q2/W2、原block10 16neuron曲线与完整原Q2信息；优先检查条件不变性和在线计算依赖图，再决定是否值得做新的稀疏kernel。局部signed投影SSE只作为这个明确函数空间问题的证据。

### F2 · 结构到底随什么条件改变？

从真实head/step/expert/token的**计算条件性**寻找规律。跨源、不同seed、运动类别、low/high专家，必须观察的是同一候选函数单元的出现/消失、迁移、预测可用性、空间共享与tile闭包。区分：
- 稳定的函数/通道簇可静态离线折叠；
- 有输入依赖但可由W1之前廉价统计预测的群组可做在线稀疏；
- 只在昂贵后验操作数出现的稀疏模式属于分析oracle，需另找前置可获得信号；
- 输出功能可由不同算子代表，可能无需传统二值稀疏掩码。

Wan仅四步且high→low有专家切换；任何跨步reuse要单独证明实际合法时点及条件稳定性。

### F3 · 这些计算承担什么生成作用？

使用已有两来源四步full model的真实异常构造**可证伪的功能假设**：例如一个便宜函数单元控制主体运动相位、背景亮度稳定、纹理细节或条件遵循；竞争解释是这些变化主要由上游attention/调度状态决定。设计保持其余计算等值的有限FFN作用干预、跨层/step/空间追踪，并在真实完整生成中比较对应功能。需控制符号、幅度、数据来源、模型身份和强Dense参照；不同可行单元比较的是**生成作用与物理成本**，而非等范数误差排名。

已有P/N/state谱、末步SSSD视频属于**现象地图**，在F3不能把预先观察到的“错误频带”当作语义功能真值；额外source测试真正预注册的机制预测。

## 3. 新的研究循环和分批样本

\`真实执行图重建 → 新现象/不变性 → 多个竞争性的结构解释 → 数学/源码可计算性与推理时信息边界 → 最小原生/对照干预 → 新source跨场景功能迁移与实际部署成本\`。

每一候选机制输出：
1. **科学命题**：适用于什么层/专家/step/source/空间或函数坐标；
2. **替代解释**：至少一个会产生不同可观测预测的竞争机制；
3. **廉价判别**：首先重用保存张量、精确数学和代码模拟，必要时少量原生controlled动作；
4. **物理计算单位**：预计具体删/改W1、GELU、Q2、W2哪一维tile/GEMM，以及预测/pack/weight流量；
5. **整段视频功能验证**：冻结真实场景划分、seed、motion/lighting/background/detail和官方/人工审核规则。

**source数不是起点**：0/12可提示现象；4/8/16/20来自原校准来源，只能做机制复验；新source+交叉seed用于验证未参与机制拟合的预测。已登记的[多来源方案](E0_MULTISOURCE_DIRECTIONAL_EVALUATION_20261026.md)继续作为**机制明确后的验证资源计划**，而不是无条件先执行640site的任务单。

## 4. 当前研究资产如何重新归类

- [原Affine54/paired-native](E0_PAIRED_FULL_DEPTH_RESULTS.md)：一个具体固定表示的反例与真实机制对照，不是算法通用目标；
- [full-network finite Dense](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)：状态依赖响应边界，支持分析动力学与动作时点，不归因到一个FFN机制；
- [saved latent decode](E0_SAVED_FLOW_DECODE_RESULTS.md)：终端运动、亮度和细节的分歧样本；
- [frequency signed bands](E0_DISJOINT_SIGNED_BANDS.md)：正交**描述性**统计，适于反驳无差别能量假设，不能独立确定计算功能；
- [已有16neuron曲线](E0_VECTOR_CURVE_CPU_FEASIBILITY.md)及[固定四坐标函数oracle](E0_FIXED_COORDINATE_BASIS_ORACLE.md)：同rank非线性在已暴露128行上有样本内改进；核心未知是4个anchor是否能预测其余12个neuron，而非如何继续压低16个的总体SSE；
- [Blackwell合成packed E2M1 GPU基元](E0_SM120_NVFP4_PHYSICAL_PATH.md)：已确认实机原生tensorcore数值执行，实际Wan在线量化、真实FFN省算及视频质量仍未测；
- [历史HiF4 closure和tile](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)：信息与物理可执行约束，而不是先验稀疏支持的定义。

**当前目标**：沿F1/F2/F3建立真正新的FFN函数/时空/生成作用联系；找到被跨source、有限干预与真实部署共同支持的机制，才进入方法实施、完全自由视频质量和系统性能比对。多Agent只参与方法发现与独立评议，原任务门禁继续由ACTIVE/原生协议管理。
