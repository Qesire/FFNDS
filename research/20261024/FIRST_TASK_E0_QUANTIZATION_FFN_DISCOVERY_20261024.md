# FFNDS 第一研究任务 E0 · 发现真实量化—FFN 结构

**阶段：`E0_OPEN_DISCOVERY`** · AutoResearcher 路线：OPEN_DISCOVERY / MAP_FLOW / DERIVE / DISCRIMINATE

本阶段从实际 Wan 四步视频 FFN 的结构出发，研究量化表示、FFN 内部计算、蒸馏时空与下游状态之间的联系。目标是获得**可被独立重建的科学认识**，让后续方法来自已经发现的机制。

[当前 E0 进展与关键问题](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md)给出了六来源观测后的主要研究机会。每个 Agent 选择最有区分力的一项展开；欢迎沿旧发现以外的新数学表征或领域文献发现更好的问题。

## 唯一核心问题

> 在真实四步蒸馏 Wan FFN 内，量化表示及其已有状态与有效计算结构、输出方向和视频传播之间，存在什么尚未解释的规律？哪些竞争机制对这个规律有不同预言？

**首轮科学入口为量化信息结构（A），同时用 FFN 内部数学（B）、蒸馏时空（C）与下游功能几何（D）提供竞争解释。** 可以从任一最有信息量的证据进入，随发现调整。

## 从原始计算流建立信息地图

在源码与真实保存数据中定位：

`residual+attention → norm2/AdaLN → Smooth1/Q1 → W1(5120→13824) → GELU(tanh) → Smooth2/Q2 → post-Q2 gain/cast → W2(13824→5120) → gate/residual → 下一层`。

优先核查决策关键的张量形状、dtype、预处理和量化顺序、scale/code 状态的生产者与消费时机、residual 的旁路路径，以及 step/expert/block/latent 位置。对照两条计算语义：

- **已有真实 Wan 路径：** A100 HiF4 的 4/8/64 层级 QDQ → BF16 GEMM。内部 `lv0/lv1/lv2` 属于 CUDA 内临时尺度。
- **面向更广的原生格式：** MXFP4 E8M0/32、NVFP4 E4M3/16 + 全局因子等的公开表示与硬件执行机制。把格式提供的信息、读取成本与实际采集字段分别列出。

原始入口：[W1/Q1/Q2/W2 参考](../../tools/packed_ffn_reference.py)、[Wan 适配](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/wan_i2v.py)、[HiF4 CUDA](../../upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu)和 [2026-10-25 E0 原生采集](../20261025/E0_NATIVE_OBSERVATIONS.json)。

## 四类可以互相竞争的解释

| 解释入口 | 研究重点 | 最有区分力的现象 |
| --- | --- | --- |
| **A：量化信息** | scale/exponent 是否只是便宜幅度摘要？符号、层级 amax 与舍入边界包含什么附加结构？ | 幅度匹配条件下的码字/分块差异与 Q2 前后响应 |
| **B：FFN 几何** | W1、GELU、W2 有符号组合、有效方向与残差的关系 | 相同激活能量下不同输出方向和 downstream 响应 |
| **C：蒸馏与时空** | 四步专家切换、共同均值、latent 邻接与真实变化部分 | raw/去均值余弦、同/跨专家与时空位置的分歧 |
| **D：下游功能几何** | 归一化切向/径向、门控、残差旁路与后续传播 | 局部理想 LN 抑制与真实 nextnorm/flow 响应的差异 |

经典/近期工作对这些解释的启发应来自广泛深入的 [Web 文献与源码检索](../agent_protocol/DEEP_WEB_RESEARCH.md)：读原论文公式、实现、消融和反例，沿后向与前向引用延伸，再回到 Wan 源码验证假设。文献检索的成果是更好的现象解释和新自由度。

## 现有资产能支持哪些判断

| 资产 | 有根据的使用方式 |
| --- | --- |
| [CAL24 旧 operand 复核](../20261025/E0_CAL24_STRUCTURE.json) | 保存的 QDQ 后输入/输出分布和采样行结构；与新增 E0 六来源有输入重合 |
| [六来源 E0 发现](../20261025/E0_DISCOVERY.md) | 真实 Q1/Q2 前后、GELU、gate、residual、固定 W2 面板与局部 FP64 LN 几何 |
| [径向分析](../20261025/E0_RESIDUAL_GEOMETRY.json) | 对实际采样的五个固定组解释 pre/post/leave-out 的方向能量分解 |
| [跨步时空](../20261025/E0_TEMPORAL_STRUCTURE.json) | 固定 latent 坐标、同/跨专家、共同均值与变化分量 |
| [高杠杆异常行](../20261025/E0_PROFILE_OUTLIERS.json) | 保留 row1981 和其对聚合结果的敏感性 |
| 历史 HP1/HP2/NP1、C0/C1、R12/R13 | 各自历史动作、局部 proxy、合成数学与参考路径的**已绑定范围** |

尤其值得独立研究 [Block 0 残差径向对齐](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md)：能否从真实下游残差分支解释这一现象？如果只是下一次 LN 的短程投影，对信息结构意味着什么？如果在后续 flow 仍有意义，又揭示了哪种真实计算规律？

## 本轮交付：一份有说服力的科学发现

一份简短发现记录应使另一 Agent 可以独立复核：

1. **现象及来源：** 真实源码、数据、数学结构或论文启发；说明最有价值的张力。
2. **竞争解释：** A 与 B 的因果机制；找出它们预测不同的具体条件。
3. **数学或源码判别：** 一个条件恒等式、构造性反例、保存数据关系或精确消费路径。
4. **下一项观察：** 在现有资料中优先核查什么；需要新原始字段时，提出采集合同草案。
5. **它开启的自由度：** 如自然引出算子、新信息来源或其他表征，记录为 `idea seed`，保留更多可能方向。

独立发现在 [E0 目录](../method_discussion/discoveries/E0/README.md) 保存；评议由其他 Agent 从源码/数学与深度文献检索独立完成。

**执行范围：** E0 科学讨论、源码、数学、公开文献及已获准的保存数据分析。2026-10-25 六来源被动原生采集已经完成；后续 CPU/GPU/model 实验与资源开销根据用户新的明确指令另行确定。
