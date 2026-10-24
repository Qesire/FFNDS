# 第一任务 E0 · FFN 量化信息与内部结构：现象发现和竞争性解释

**唯一当前讨论任务，不是算法实现任务。** 2026-10-24；AutoResearcher `OPEN_DISCOVERY → SOURCE_RECONSTRUCTION → MECHANISM_CONTRAST`。所有参与 agent **只进行方法/机理讨论**。

## E0 要回答的一个问题

> 在真实的四步蒸馏 Wan FFN 内，量化表示及其周边已存在的状态究竟揭示什么**不同于普通幅度指标**的结构？这些结构是否更像格式伪影、FFN 固有的计算特征，还是与视频输出的因果敏感性有关？

不设定隐藏通道、K 面板、W2 路由、低秩补偿、dense 写回、code-cache、特定掩码或 Proxy 为默认答案。**不要在 E0 设计新算子，也不要将已有 G6D/E/F 反过来当成现象必须验证的目标。**

## 具体起点：从源码与原始证据制作“信息—计算流地图”

冻结原始 FFN 的最小相关范围：
`norm2/AdaLN→(Smooth1)→Q1→W1(5120→13824)→GELU(tanh)→(Smooth2)→Q2→(post-Q2 alpha/cast)→W2(13824→5120)→gate/residual→下一层`，且四步推理的 step/expert/block/latent tile 被准确标明。

对每个位置只核对**真正改变科学结论的属性**：
- 输入/输出张量语义和 K/N 轴、dtype、有效缩放与舍入时机；
- 激活、权重、量化局部/全局状态的**产生者、消费者、存在时间、跨组依赖**；
- 哪些内容是当前执行真的写入持久存储、哪些只在 kernel 内临时存在、哪些只有额外计算才得到；
- 该内容属于已冻结的 **A100 HiF4 QDQ→BF16** 路径，还是尚未在 FFNDS 真实运行的 **native MXFP4/NVFP4** 假定格式；不从格式规范推导已有运行数据。

**源码入口：** `tools/packed_ffn_reference.py`；`upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/wan_i2v.py`；`upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu`；已保存 `research/20261023/C0_coupling_offline_results.json` 和 `research/20261022/R12_HiF4_group_separability.json`。用原始源码/哈希，不依赖前一位 agent 的概述替代证据。

## 最重要的互相竞争解释（启发式，不是必须接受的清单）

**H-A：量化元数据主要是廉价幅度压缩。**
- 格式尺度/指数的主要作用可能已被 input RMS、amax 或静态权重解释；
- 将尺度信息加入条件上游变量后，结构性排序差异或额外机理消失；
- 若成立，应寻找新的内部几何/因果信息，而非追加指数阈值。

**H-B：量化内部状态表征了局部异质性，而不仅是一个幅度。**
- 相同全局 RMS 的输入可以有不同分块尺度轮廓；相同 block scale 的 FP4 码字又可能有完全不同的符号、占据与投影响应；
- 共享尺度 / 层级 max / 局部 amax 领导者可能使码字响应、方向和分组相关性与 BF16 连续表示产生区别；
- 可从严格源性质/反例构造入手，区分真正新增信息与只换单位或重新计算 RMS。**不假设它已在真实 Wan 中频繁发生。**

**H-C：真正的功能结构主要来自 W1—GELU—W2，而非量化本身。**
- 偏置、负端 GELU、有效特征子空间、W2 有符号抵消、组间相互作用可能主导哪些输出变化；
- 若相同量化尺度/码字统计对应截然不同的输出方向，说明“谁重要”的科学对象必须从幅度转到有效功能表示；
- 不必因此立刻尝试重基、低秩或稀疏。

**H-D：蒸馏与时空旁路决定变化是否具有视频意义。**
- 同一局部 FFN 变化在不同 denoising step/expert/block、latent 位置和 residual/attention 状态下可能不同；
- 旧观察中的背景、光照不稳与主体相对稳定，是值得解释的现象，不是已证的 FFN 机制；
- 不要把源内多行、多 block 当成独立视频，也不把 C0 synthetic residual action flip 当真实 Wan 发生频率。

每个 agent 可以反驳这些假说或提出明显不同的 E0 解释；**有价值的是新现象、数学联系和不同预言，不是完成“至少四个”形式任务。**

## 现有证据：允许与不允许得到的结论

| 已有资产 | 当前明确支持的观察 | 没有提供 |
| --- | --- | --- |
| `R12_HiF4_group_separability.json` | 历史 HiF4 的原顺序 64-group / 4-8-64 层级局部性，在报告的参考范围内逐行/完整组可分离 | Native MX/NV 数据分布、GPU kernel 省算 |
| `R13_spatial_sharing_limits.json` | 已保存成对轨迹上的 whole-output-common 等值约束及条带参考 | 所有组级相似性都缺失、所有复用都无意义 |
| `HP1/HP2/NP1_proxy_RUN.json` | 历史具体 features/labels/留出方案下，线性 head 有局部信息，非线性 head 未明显超越旧匹配对照，原 gate 未通过 | 对新量化元数据的直接失败结论、全视频性能 |
| `C0_coupling_offline_results.json` | 同 q 的合成 residual-scale 反例可翻转两完整动作下游 nextnorm 排序 | Wan 中发生该现象的概率 |
| `C1_boundary_target_audit.json` | 写回误差与下游 nextnorm 目标不完全等价，已有六来源/72位置的标量统计 | 原生 quantizer code/scale 与新机制的因果增益 |
| `OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json` | 归档标量重聚合与 E2M1 码轨道穷举 | 真实 Wan Q1/Q2 scale/code occupancy 及 native FP4 测量 |

尤其是 **OP2 3,840 已验证历史 payload** 的索引不等于本环境拥有其中 FP4 元数据字段。先对照保存字段；不能“根据 V1 cosine”推定量化块熵或重要性。

## 自然的三个数学现象入口

1. **信息没有凭空创造，但成本改变。** 记实际 side-state `S_Q=E_Q(x).state`。确定性编码下 `I(U;S_Q|x,θ)=0`；这不意味着读取 `S_Q` 与读取全部 `x` 的物理费用相同。E0 优先区分**表征可用性**与**观测价格**。
2. **同尺度不代表同方向。** 只靠一个块 `amax` 或编码尺度，无法恢复码字符号/内部方向；同尺度的两个合法码元向量可对 W1 的相同权重产生不同投影。由此提出 *尺度-only / 尺度+少量码字 / 完整数值* 的互斥解释，而非先做三种路由器。
3. **同本地输入不代表同下游价值。** `q` 在固定 θ 下决定当前 FFN 的本地确定性输出，但 pre-norm residual 绕过 Q1，后续 gate/norm/attention 接收的状态可能不同。允许探索与局部 hidden 预测**不同的问题**：哪一种结构沿后续计算被放大？

这些是解释现象的起点，不是三个已经选定的指标或 loss。

## E0 完成时只需交付一份可复核的“发现档案”

格式不强制，但核心必须足以让另一位 agent **独立挑战**：

- **计算流与信息地图**：图/表呈现信息在何处产生、何时存在、是否真实保存，HiF4-QDQ 与原生 MX/NV 独立标记 source-grounded vs format-only；
- **事实—猜想分栏**：从已存源码、已存实验、条件定理和缺失的原始观测中，挑选真正可区分的现象；不要把另一个 agent 的新提案当已测结果；
- **少数竞争性解释**：至少阐明一个支持和一个反解释；每条给出它们会在哪个 observable 上分歧；
- **唯一下一步建议**：下一件最值得验证的*现象*是什么，现有证据是否足够；若不足，指明缺哪个原始 tensor/字段及最低量级采集方案**仅作为后续研究请求**，不在 E0 运行。

### E0 最重要的完成判据

不是模型选择、算法创新数量、平均相关系数，而是**对一个有价值的未知机制给出清楚的问题、两种竞争解释及一项真正能把它们区分开的后续观察**。

如果当前资产已能给出严格数学/源码答案，也可直接输出“已证的现象 → 更深的新问题”，不强求额外实验。若不存在能据此选择算子的结构，E0 仍可成功，并将下一轮转移到 B（FFN 内部结构）或 D（后续传播）。

## 执行权限与协作

本轮 **只读源码、已保存数据、数学推导、讨论**；不调用实验、启动模型、创建 coding subtask、打开未来作业、运行 synthetic CPU demo。实验仍由 `EXPERIMENT_PAUSE.json` 和 `OFFLINE_ONLY_RESUME_SCOPE_R1.json` 约束。

并行 agent 对 **E0 科学解释**独立发言；不要共同争写 `QUEUE.md`、`AGENTS.md`、`ACTIVE_RESEARCH.json`，不要以自己提案的复查制造 peer support。旧 G6A—G6F 均作为 **候选机制的历史种子**，并非本任务默认实施目标。
