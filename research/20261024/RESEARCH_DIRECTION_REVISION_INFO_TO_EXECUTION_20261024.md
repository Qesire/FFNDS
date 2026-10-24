# FFNDS 研究方向再校正：从量化元数据评分，转向信息—算子—输出几何的联合探索

2026-10-24 · AutoResearcher pinned `5bd7ec537ae63f2b6a6dd579f0e886aff58a177c`

**一句话目标：** 在真实 Wan2.2 I2V 四步蒸馏模型中，构造能在视频质量近似不变时真正减少原生 FFN / 全生成执行时间的计算图；量化元数据只是可供低成本决策复用的信息之一，不是方法形态或论文贡献的先验定义。

> 本文是用户继续要求的科学方向修订与研究判断，不替代根目录 AGENTS.md 或 `research/ACTIVE_RESEARCH.json` 的单一活跃权威，不代替其它 agent 的审核，不改变原始实验暂停与此前仅限归档统计的 CPU 离线授权。

## 新的科学中心：寻找「物理有意义的算子」和「足以选择它的信息」

我们反复遇到三种不同的事实：

1. **Q1 数据已经完整包含本地 deterministic FFN 结果所需的信息**，但读全向量并计算 W1 太贵；原先的 scale、码字、pre-norm residual、attention 更新可能以较低增量成本透露一小部分*足以选算子*的信息。量化没有创造关于固定输入的额外 Shannon 信息；其潜在优势是**已经编码好且相对廉价**。
2. **局部特征分数不等于省算价值。** G6D 最初预测 `E||W1_g q||²`，但 `K=a bᵀ` 或 `q_t=λ_t q0` 的条件下，任意时间尺度变化仍有静态 top-k。旧 HP1 在固定 108/216 组下 step2 retained-C 从 `0.5348`→`0.5604`（posthoc oracle `0.6357`），说明源留出 local proxy 有部分信息；它的原始 gate 失败，且 `C` 不是下游自由视频决策标签。旧增益不是新 metadata 方案的实测结果。
3. **同一个 W1 局部误差可以具有不同的 downstream 价值。** C0 合成的归一化/残差尺度反例说明 `q` 相同而残差旁路状态不同，有限动作排序可以翻转；R7 的 signed交互+有限传播项可互相抵消；C1 的写回 SSE/nextnorm SSE 最佳四分之一行集合只有约 `0.783` 重叠。原始全生成的光照/背景细节稳定性不能用一个局部能量指标替代。

研究应保持三类对象的互相反馈：
`信息对象 S_t(时机,物理代价)` ↔ `实际动作 A(数据流,计算单元)` ↔ `评价 H(残差旁路,后续block,step,视频)`。

## 1. 先定义可改变的真实算子，不先限定通道 mask

实际路径（按 archived native `tools/packed_ffn_reference.py` 和 Wan/LightX2V 源）：  
`residual+attention→norm2/AdaLN→Smooth1→Q1→W1 (K=5120,N=13824)→GELU_tanh→Smooth2→Q2→post-Q2 alpha/BF16→W2 (K=13824,N=5120)→gate+residual→后续层/四步输出`。

| 算子方向 | 对齐的便宜信息 | 改变的物理维度 | 两投影理想 MAC 收益上限（省去比例 p） | 难点 |
| --- | --- | --- | --- | --- |
| **W1 input-K panel** | Q1 输入微缩放 scale / 低位码的统计 | W1 的 K 轴，天然与 Q1 的 16/32 输入块对齐 | `p/2` | 扰动经过 GELU/Q2 后可能翻大量代码；同 tile 的 keep 并集会趋于满 |
| **W1 hidden-N + W2 hidden-K grouped path** | Q1 便宜 sketch、离线权重、pre-LN/attention/gate | W1 的 N、W2 的 K 同一隐藏组 | `p` | 跨轴路由打包及 Q2 的 block/global scale 耦合，动态输出影响 |
| **W2 input-K panel** | W1 后已产生的 Q2 scale/码字 | W2 K 轴 | `p/2` | W1 已付费，Q2 side-state 需要在 W2 dispatch 前廉价读取，幅度界可能太松 |
| **非 mask 结构** | 权重结构、时序和条件状态，也可能无需新增观测 | 重基、合并、共享、离线/少量后训练的直接算子 | 由实际算子定 | 必须真正改变 tensor-core 执行及比特级数值/视频行为 |

`p` 必须是**实际 tile 计算指令或 panels 已取消的比例**，不是按 token 自报的逻辑稀疏比；式中仅两次相同形状投影的理想 MAC，不是整个 FFN 或全视频加速比。当前 HiF4 QDQ BF16 与 Blackwell FP4 native 是两条不同的被测计算图。

重要的轴区分：**Q1 的 scale-block 轴沿 5120 输入 K，不是 W1 的 13824 输出组**。64 隐藏组是 HiF4 历史的自然 Q2 独立组（216 组），也恰是 MXFP4 的两个 32 块、NVFP4 的四个 16 块；但不意味着 NVFP4 tensor-global 动态 scale 不跨组耦合。选择组后需重新核查实际 Q2 global/local scale 规则，特别是省掉的 W1 hidden 行可能是 global amax 决定者。

## 2. 将信息价值从「相关性」改成「特定动作的增量收益」

对完整 action set `A` 与现有信息 `Z`，定义 `R(Z)=E[min_{a∈A} E[L(a)+λ C(a)|Z]]`（对固定的 source/horizon/action 单位）；引入量化 side-info `S_Q` 的零读成本价值为 `R(Z)-R(Z,S_Q)`。实际净效用还必须扣除：捕捉 Q1 scale / 码字统计 → 搬运/同步 → policy → packed routing → GEMM occupancy → QDQ/native scale semantics。

**额外信息的成功门槛不是提高预测 R²，而是改变了实际动作选择，并在同成本下改善 finite-action 损失**。这是经典条件决策理论在本任务的应用，不宣称 Bayes 公式本身是新发明。

分开比较四种情况：
- 本身为条件固定可达的 `step/expert/block/temporal-bin` + static weight action；
- 已有廉价 `input RMS / pre-LN mean/rstd / attention innovation / gate`；
- Q1 scale SHAPE 与真实码字最少统计（解释相对增益，且不预设已存储可读）；
- 上述因素与一个具体的 native K/N panel 动作共同作用的效用。

特别注意：`q` 完整已给定时，额外 upstream 状态不能为本地 deterministic FFN 再增加数值信息；**但下游 residual bypass 没有被 q 覆盖**，所以 upstream 状态可能对选取 FFN action 的后续传播价值有增量信息。在给定 cheap sketch(q) 时，外部 state 对本地 hidden 预测也可能有帮助。两个问题不能混作一个。

## 3. 将「量化特征」升级为计算传输的中间机制

原有 G6D `P_{t,g}=Σ_b ν_b||W1[g,b]||² s_{t,b}²` 是有条件成立的输出组 W1 能量期望。它的作用可作为静态+尺度 baseline，但**不能从公式直接得到动态价值或应当执行的组**。

真正值得检查的量化特异路径：**W1 省算引入的预激活扰动是否引发 Q2 码字/尺度状态的离散跃迁**，以及这种跃迁经 W2/gate/残差如何传播。

令 `z=Σ_b A_b q_b +bias`，若省略一组 Q1 K 面板 `S`，则 `z'=z−Δz_S`，`Δz_S=Σ_{b∈S}A_bq_b` 是确切的解码实数算子恒等式。随后的最终差异必须按 `Q2(GELU(z))−Q2(GELU(z'))` 计算，且对每个 changed group 包含 scales、post-Q2 gain 和 BF16 次序。

从 Q1 已有的 `s_b` 可得到保守的 `||Δz_S||₂≤Σ_{b∈S}6 s_b√k_b ||A_b||_F`（E2M1、固定有效解码权重）；它是低成本投影扰动上界，**不是 Q2 code-flip 或视频安全证书**。对任意小的 `||Δz||`，若有隐藏项接近 Q2 code/scale 变化边界，仍可能产生大量 discrete changes。

新研究动作 [G6F](../method_discussion/proposals/MD-20261024-QMETA-KPANEL-TRANSITION-G6F/v1.json)：将**Q2 跨编码状态跃迁的条件风险**作为 W1 K-panel 省算的中介变量，研究该风险能否从 Q1 native metadata + 前置信息低成本预测；Q2 真值只充当离线标签而非免费在线 look-ahead。这个想法的启发在于把量化从“标签为无关的幅度指标”改成“省算行为经过的离散传输机制”，同时保留 G6D N 路由、G6E W2 路由和非 mask 算子竞争。

可能结果很有信息量：如果所有轻微的 K-panel 减算都触发大量 Q2 scale changes，则该误差通道不适于压缩；如果多数 action 落入量化平台期，则不需要补算也能保持局部 Q2 操作数。两个结果都应从**真实 source、scale layout 与同一数值路径**获得，而不是由总体码字轨道数量推论。

## 4. 格式差异不再被当成无关附注

**平台 A：归档 Wan 原生证据** = HiF4 的 QDQ→BF16 GEMM。HiF4 QDQ CUDA 源内有 64/8/4 层级局部最大值 `lv0/lv1/lv2`，但没有把这些值作为永久有效的 packed scale tensors 传递给 BF16 `F.linear`。凡要使用 scale 作 policy，应记下 export/recompute/producer fusion 的真实开销。

**平台 B：未来原生 FP4 执行** = MXFP4（E8M0/32）或 NVFP4（E4M3/16 + global FP32）等硬件实际支持格式。原生 GEMM 需要 scale，**表示中有 scale ≠ 路由在 GEMM 前可免费取出 scale**。需要核查 kernel API、scale swizzle、global scale static/dynamic、实际权重量化和 MMA physical tile。TensorRT-LLM 官方兼容矩阵中原生 NVFP4/MXFP4 是 Blackwell 或后续硬件路径；不能以 A100 HiF4 的数值/性能结论直接替代 Blackwell native 结论。

跨平台比较仅用于确定是否值得迁移研究机制；**算法质量/省算的严格比较必须在同格式、同 checkpoint、同有效量化状态及本机 dense 基线上完成**。HiF4 的 64组与 native MX/NV 的 microblock 不可简单互换。

OCP MX 规格：<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>；NVIDIA TE NVFP4：<https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html>；TensorRT-LLM 硬件支持矩阵：<https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md>。

## 5. 下一步要区分三种机制（不是先选中一个 mask）

研究首轮最值得证明或否定的机制：

1. **执行轴机制：** 相同 Q1 量化尺度信号，沿输入 K-panel、隐藏 N-group 和一个固定非 mask/固定算子具有不同的可执行 tile 密度及路由成本。对比真正的 physical K/N tile-union，而不是行级分数。
2. **量化跃迁机制：** 匹配 `||Δz||` 下，Q2 code+scale 状态变化的来源/步依赖与 gated W2/nextnorm 影响存在超出简单 W1 幅度指标的预测结构。如果只是互相关且无法改善动作价值，则这条量化特异路径要退居描述分析。
3. **旁路几何机制：** 相同 q / 相近 Q2 变化但不同 residual/attention/gate state，最终动作价值是否改变？历史 C0 已给条件反例；需要真实有限动作并按来源留出区分其实际应用程度。

中途即可问一个严格的判别问题：**在恰当的物理算子下，新增一小段已经由量化器产生的信息，是否比同成本的预归一化统计/码字 sketch/固定算子带来更高净决策价值？** 如果是，量化 Proxy 才有真正的科研意义；如果不是，也能据此转向改变计算基底或完全不需要在线路由的算子。

## 6. 使用历史资产时的可操作分界

已有 Git 原始 `OP2D`、`C1` 汇总允许检验动作后果的来源/step 异质性；`HP1/HP2/NP1` 能作为旧 proxy 的数值/特征成本对照；`R12` 证明历史 HiF4 完整64组的分离性质。不能将它们当作原生 MX/NV Q1 scale 或 Q2 code flip 的实测样本。上次离线分析只做 scalar reaggregation + FP4 alphabet exhaustion，并未访问源机 HF 3,840 tensor bytes。

**缺的不是又一个评分公式，而是一对真实操作数和动作：** 一个真正存在于 decision-time 的输入量化元数据/已有状态，及一组用于同前缀/同格式下的全动作结果。保存数据必须配对原始 step/expert/block/source、预处理/QDQ/rounding 和 GPU tile。未来实际实验需独立获得资源授权，不能从讨论 PR 或用户上次狭义离线计算许可推断已恢复模型运行。

## 7. 状态与协作

- 本轮仅改变**研究解释、方法候选及优先的判别问题**，不修改用户原定的终极目标。
- 为遵循多人讨论队列，G6F 是同一原始 agent 的独立新提案；另一 agent 可以审查，未得到支持前没有认可状态。G6D/G6E 原版本保持；可对其特定局部定理认可，不把其中任何一个设为默认总目标。
- 本科学判断不修改 `AGENTS.md`、共享 `QUEUE.md` 或 `ACTIVE_RESEARCH.json`，因并行 agent 在方法讨论中不竞争写这些共享文件。等待独立审查时仍可继续其它来源的科学构造。
- 实验权限仍按原暂停和用户仅恢复归档来源数据 CPU 离线重算的记录处理；本轮没有 CPU/GPU/model、Slurm 或全视频生成。
