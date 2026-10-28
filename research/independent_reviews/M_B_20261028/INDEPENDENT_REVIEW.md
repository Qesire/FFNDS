# FFNDS M-B 方法独立评议：数学成立之后，真正的压缩优势在哪里？

> 日期：2026-10-28  
> 身份：INDEPENDENT_CRITICAL_REVIEW / ALG + SRC-REVIEW + REPO-EVIDENCE；新增实验状态：NOT_RUN  
> 评议对象：M-B gate/suffix-sensitive coordinates，当前 METHOD_SNAPSHOT v2（2026-10-28），LightX2V Wan2.2 I2V A14B 四步、双 expert、40 blocks/expert。  
> 评议性质：独立科研判断。不是原方案修改、正式方法队列投票、跨 Agent 支持、实验运行授权或模型效果通过声明。  
> 工作方法：遵循仓库 AutoResearcher 的 mechanism-research、problem-reformulation、research-expansion、ffnds-paper-discovery、web-research-discovery 原则；本次以已核仓库数学/源码/实验和仓库列出的一手文献路径进行交叉推理，未启动原生模型、视频、GPU 或新的外部原文实验。

## 0. 决策摘要（独立意见）

**决定：CONDITIONAL_CONTINUE / B0-B1 有界推进；暂不将 M-B 视为主要可部署算法，也不因数学恒等式成立就启动全模型训练。**

- **具有较高研究价值**：以实际下游函数定义 FFN 方向重要性；同 source/state 的敏感度—输出统计配对可在数学上改变最优表示。
- **尚无真实算法增益证据**：M-B 的 B0 原生 post-write u/head/flow 捕获、B1 原生有限响应、B2 学生函数以及 B3 完整自由视频/系统时延，当前均未提供通过证据。
- **真正的科学分歧**：下游敏感的坐标是否不仅有较小的 oracle 投影误差，而且能从实际可用的 pre-W1 输入低成本预测，并在全模型学生 rollout 中比同成本紧凑 FFN 更好地保持生成行为。
- **最强竞争基线**：已有 A0 width=6016、相同训练预算下在四位置完成真实 Q2 局部恢复。因此应比较「同在线成本＋同训练预算」的 M-B 与 A0、direct-output student、PCA、marginal、shuffled-joint，而不是只对照未适配的稀疏 FFN。
- **建议定位**：B-JOINT 保持机制研究；B-H 保持末层可解析锚点；B-F 是可部署希望但有学习性和核开销风险。M-A/M-C/信息与执行线继续并行。

**单一决定性命题**：

> 条件配对敏感度构造的表示，在相同 online 成本、相同可得输入、相同训练资源下，是否比一般紧凑函数更容易保留真实视频生成行为？

该命题含三个独立因子：自然功能的重要性（NATURAL）、低成本学生可预测性（PREDICTABLE）、原生物理可执行性（COMPUTABLE）。任一维度不应由其他维度的代理结果替代。

## 1. 原生对象与作用边

主要真实计算：post-Q1 x ∈ R^(M×5120) → W1(5120→13824) → GELU(tanh) → Smooth2/HiF4 Q2(group 64→8→4) → W2(13824→5120) → pre-gate y → e5 逐通道 gate × y + residual r → 下一 block 或末层 final LN → 时间调制 → 64D head → unpatchify/flow → 下一去噪状态/视频。

- 原生对象为四步 high/high/low/low；两个 expert 各 40 blocks；M=31668 对应仓库固定分辨率/81 帧测试合同，不宜泛化为任意输入。
- teacher 的 y、J_s 和 full post-write u 都来自已付费后缀或离线测量；线上学生不得把 full teacher y / J_s 当作免费特征。
- 当前真实 HiF4 计算路径包含 fake-QDQ + BF16 GEMM；packed native FP4 的系统性能须独立测定。
- 最后一个 block 的后缀没有下一层 attention，便于严格解析；早层则存在跨 token attention 和后续多步状态依赖，需用全 M×D 的后缀对象而不是逐 token head 近似替代。
- 观测独立单位是 image/source cluster，token、行、step、block、seed 为嵌套样本。

具体源码绑定及 checkpoint 形状参阅 [末层源审计](../../method_discussion/discoveries/E0/b-final-block-head-source-audit-20261028.md) 与 [现行 B 方法快照](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)。

## 2. M-B-JOINT 的数学价值与适用条件

令 y_s 为 teacher pre-gate 输出，μ 为同一个公共中心，R=I−P，J_s 为同一 source/state 的下游局部 Jacobian，D_g 为原生门控。则

$$
G_s = D_{g_s}^{\top} J_s^{\top}J_s D_{g_s}, \quad
C_s = (y_s-\mu)(y_s-\mu)^{\top}.
$$

边际与联合代理为

$$
E_m(P)=\operatorname{tr}[\bar G R\bar C R^{\top}],\quad
E_j(P)=\mathbb E_s \operatorname{tr}[G_s R C_s R^{\top}].
$$

精确相减得到

$$
E_j(P)-E_m(P)=
\mathbb E_s \operatorname{tr}
[(G_s-\bar G)R(C_s-\bar C)R^{\top}].
$$

**独立推断 ALG**：M-B 的真正附加信息位于同一状态的敏感度—输出统计的条件配对项。其是否有实质 Wan 增益，取决于该配对项是否在合法投影集合内显著改变独立来源上的排序，而不是「使用 Jacobian」这个事实本身。

仓库已给出二维、全部 rank-1 幂等斜投影成立的反例：边际最优选择 e1，配对目标最优选择 e2，二者在该构造中的损失相差 100 倍。此为 **ALG 可能性证明**，非 Wan 中 100 倍优势或运动相关方向的 **NATIVE/VIDEO** 发现。

另一个重要条件：同一个 CAL24 (step,block) 下原 gate 对已测来源固定，gate-only 的 G 不会自动产生来源适应性。值得采集的是状态依赖的真实 suffix J_s，以及与相同状态 y_s 的配对。

**竞争解释**：如果 J_s 的有效方向大体由共同 readout 和归一化标量主导，那么 B-JOINT 可能只带来少量增益；如果状态变化造成具有泛化性的方向旋转，则配对方法应超过固定读出、marginal 和来源乱配对对照。

## 3. 末层 Jacobian 的更强结构结论：固定读出＋状态依赖秩一修正

固定对应 step 的 head modulation/gate，采用实数化 LN/head：
令 P0=I−11ᵀ/D，z=P0u，σ=sqrt(||z||²/D+ε)，T=W_head diag(1+scale)，D=5120，u=r+D_g y。

$$
F(u)=\frac{Tz}{\sigma}+c,\qquad c=W_{head}shift+b.
$$

于是对 pre-gate FFN 的局部扰动 δy：

$$
\boxed{
J_y\,\delta y =
\frac{TP_0D_g\delta y}{\sigma}
-\frac{(Tz)(z^{\top}D_g\delta y)}{D\sigma^3}
}
$$

其中第一项基于固定读出算子 TP0Dg（另受标量 σ 缩放），第二项是在单一状态下**矩阵秩至多一**的归一化耦合修正。因此：

1. 对固定 modulation/gate 的末层，后缀状态依赖未必表现为复杂的高秩方向变化；它可能主要是固定读出空间＋状态相关的低秩修正。
2. 第二项是否数值重要不能从 1/D 系数直接判断，因为 zᵀDgδy 可以随维度增长。需比较原生同状态下两项的有效幅度与实际有限响应。
3. 与其先拟合任意 5120D 敏感子空间，更有判别力的是比较 **fixed-readout、fixed+rank1、marginal、paired-joint** 的同成本来源级结果。
4. 这只是每个固定条件、单末层 token 的连续 surrogate。真实 BF16/门控写回有舍入，后续四步 video 不是同一个线性函数。

本节为从仓库已有 LN/head 公式直接导出的 **ALG 解释**；不宣称原生 delta 的数值分量已经测量。

## 4. B-H 的核心弱点：充分统计不等于高效算子，也不等于最小信息量

末层连续输出确实依赖

$$
q=TP_0u\in\mathbb R^{64},\quad v=\|P_0u\|^2/D,\quad
F(u)=q/\sqrt{v+\epsilon}+c.
$$

这提供 65 个中间充分统计，但**输出本身只有 64 维**。它不是信息最小性或计算复杂度降低定理。若共享相同合法输入，一个直接预测最终 64 维输出的学生也属于待比较的函数族。

- 若 B-H 为得到 q,v 仍完整执行 teacher W1/GELU/Q2/W2，在线没有获得所声称的省算。
- 若直接从 x,r,g,t 预测 q,v，成败取决于近似网络在成本预算下的风险；高敏感度的 target 可能恰好比一般输出方向更难预测。
- 若通过低秩 y=Φc 重建统计，精确实数式中的 Bᵀa 通常要 O(Dr)，二次 Gram 通常要 O(r²)；实际 BF16/Q2 kernel、layout、访存可能抵消简化。
- 在等输入与预算下，必须增加 direct-64 student，才能证明通过 q,v 因子化提供了额外归纳偏置、稳定性或可执行优势。

建议明确优化对象：

$$
\mathcal R(B)=
\inf_{\operatorname{OnlineCost}(h)\le B}
\mathbb E_s\|F_s(r+D_g f(x))-\widehat F_h(x,r,g,t)\|^2.
$$

注意该代理还需进一步连接到 student free-rollout/video estimand；同状态风险是先导，不是视频质量定理。

## 5. 构造性替代：学习 FFN 对已有残差读出的增量

既然 r 在当前 FFN caller 已可得，与其重新预测全部 q,v，可以测试 **residual-conditioned delta sufficient statistics**：

设

$$
a=P_0r,\quad b=P_0D_g y,\quad
q=Ta+Tb,\quad
v=(\|a\|^2+2a^{\top}b+\|b\|^2)/D.
$$

定义

$$
\Delta q=Tb,\qquad
\Delta v=(2a^{\top}b+\|b\|^2)/D.
$$

学生仅由合法在线输入计算：

$$
(\widehat{\Delta q},\widehat{\Delta v})
=h_\phi(x,r,g,t,\mathrm{expert}),\quad
\widehat F=
\frac{Ta+\widehat{\Delta q}}
{\sqrt{\|a\|^2/D+\widehat{\Delta v}+\epsilon}}+c.
$$

**新可证伪假说 H_delta**：显式保留已存在残差对读出的贡献，学习 FFN 增量，可能比从头预测完整 q,v 在相同在线成本下更易泛化，并允许分辨 FFN 主要承担 head 线性修正还是 LN 方差耦合。

对照应包括：(i) direct64，(ii) 只学 Δq，(iii) 学 Δq+Δv，(iv) 学 q+v，(v) A0／等成本低秩 FFN。特别注意 Δv 可以为负；约束最终 v+ε 的正性而不是错误要求 Δv≥0。

这仍是 **OPEN 候选**，不是已经验证的新方法：Ta 的在线成本、残差的计算时点、kernel fuse、量化及完整 head/flow 数值身份均应核算。

## 6. 最重要的推广与执行约束

### 6.1 末层加速贡献上限

四个 denoise step 各运行 40 个 block，对应 160 个 block-FFN 调用。只替换各 step 的 block39 最多涉及 4/160=2.5% 的 FFN 调用数。若 FFN 的每位置成本近似相等、替换器零成本，这也约为 **全 FFN 主计算的 2.5%**；不是实际整机加速测量。真实模型中其他模块、替换器成本与不同阶段时延使系统收益更小或不同。

**推论**：末层 B-H 适合作为可解释的机制实验，不宜独自承担 FFNDS 的全模型加速主张。前 39 个 blocks 的完整后缀 J 存在跨 token/跨步耦合；B-F 要重新设计对应 loss 和 operator。

### 6.2 事实上的强基线与真实成本

现有 A0=6016 宽普通学生的主 GEMM 理论 MAC/token 为 2Dh=61,603,840。B-FOLDED 与之相同；B-FACTORED 为 Dh+r(h+D)，r=512 时 36,503,552 MAC/token（约 A0 的 59.26%），但增加第三次 GEMM、额外中间张量（M×512 BF16 约 32.4MB）及可能的布局、量化、驻留费用。

**判定标准**应是同条件原生 online operator/FFN 和整个推理的时延、显存、质量–成本 Pareto；低 MAC/低秩代理本身并不是 SYS 通过证据。

### 6.3 与视频动力学的联系

在同状态线性化下，若第 k 步输出误差为 e_k，状态扰动可写为

$$
\delta z_{k+1}\approx A_k\delta z_k+B_ke_k,\quad
\delta z_T\approx\sum_{k=0}^{T-1}
\left(\prod_{j=k+1}^{T-1}A_j\right)B_ke_k.
$$

单层 head/flow 灵敏度只描述某个后缀和工作点，真实视频行为还受 state drift、后续去噪步、误差方向抵消/放大和观测器影响。高 Jacobian 响应不自动等于自然路径下的运动功能承担。M-B 若要声称修复运动，需与其对应的视频动作、相机运动、时间语义、纹理/RAFT 可观测性对照建立证据。

## 7. 仓库已有证据及其严格外推范围

| 证据 | 已核内容 | 对 M-B 的意义 |
| --- | --- | --- |
| UB1 consumer20 | 已暴露开发20来源：Dense RAFT 连续幅值均值 6.894，consumer 1.152（-83.28%）；成像质量 72.629→71.626（约 -1.38%），binary dynamic 6/20→0/20 | 说明出现「外观近似、运动严重缩小」这一待解释行为；并非 M-B 的成功或失败 |
| CAL24 4 sites | 已暴露24来源删一后 whole64 支持 Jaccard 均值约 0.980–0.998；同 site gate 来源间不变 | 简单来源能量支持不稳定不是这四处最强解释；必须实际测 J_s |
| A0 6016 Q2 pilot | train16 / 已暴露 heldout8；四 site energy94 gated relative SSE 1.061%–2.546%；native Q2/折叠权重验证 | 普通紧凑 FFN 已有相当强的局部恢复；是 M-B 的最强可操作匹配基线 |
| M-B v2 | 配对数学、末层 LN/head 结构和 checkpoint header 已经有 ALG/SRC；B0/B1/B2/B3 无相应新 NATIVE/VIDEO/SYS 成功数据 | 支持有界推进最小判别，不支持部署或运动质量结论 |

原报告（包含来源暴露和数据边界）：
- [UB1 consumer20 质量与运动](../../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)
- [CAL24 来源删一和 gate](../../20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md)
- [A0 四位置恢复 pilot](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)
- [B v2 方法快照](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)

## 8. 最近先例与创新边界

仓库已收集 CoBRAS（状态/梯度协方差平衡与斜投影降阶）、2ndMatch（扩散模型 Jacobian 响应匹配）、OBS-Diff（二阶/时步敏感度）和 SVDtrunc（低秩模块分配与蒸馏）等相邻方案。对应来源与版本见 [B 论文 source ledger](../../literature/20261028/B_SENSITIVITY_PRIMARY_SOURCE_LEDGER.json)。本评议只据已登记机制定位其竞争性，不声称对全部作者代码进行了本轮重新复现。

**创新性结论**：「以梯度敏感度加权选择低秩空间」及固定 G 下的广义谱解已有成熟先例，不足以单独支撑强方法贡献。潜在原创点应进一步落在以下连续链条的非平凡差异：

**同状态真实联合敏感度 → 可泛化的非边际最优表示 → 低成本可学习函数 → 完整视频行为与真实推理成本改善。**

若只得到 oracle 方向的较小局部 SSE，而 student/视频/硬件没有增益，应把成果定位为下游结构分析与有界反例，不将其包装为新加速算法。

## 9. 最小但有判别力的实验顺序（建议，未授权执行）

### E0：沿用原 B0 producer/neutral 身份

在 ACTIVE 允许且冻结 source、step、expert、条件、seed、original Q2、BF16 路径之后，捕获真实 full-M post-write u/head/flow、r/g、head modulation 与相同状态。首先做原 dtype neutral/no-op。已有旧 Y 不可代替 full u。E0 是资产与数值身份验证，不以此宣称联合机制优势。

**可复用资产计划**：source-identified full u/r/g、modulation、head/flow、合法 x、各态输入/输出精确坐标、版本化 entrypoint/config、运行 receipt、数值 neutral 误差及相应 producer。限定 teacher state，同 student rollout 不直接共用。

### E1：末层固定读出与 source joint 判别

冻结相同来源/状态、有限扰动幅度、head readout、训练/验证 source 切分。比较 fixed-readout、fixed+rank1、PCA、marginal、paired-joint、source-shuffled-joint，在相同 rank 与 head/Jacobian 测量成本下报告每独立来源的有限 flow 误差和 CI。另记录连续理论两项的有效比重、BF16 neutral noise floor 和 source-state 旋转。

**不同预测**：若状态相关配对真实有额外信息，paired 应在独立 source 上持续优于固定读出与打乱配对；若固定读出已解释绝大多数差异，昂贵来源配对可转为低优先级。

### E2：可预测性与在线预算交叉测试

用真实且前置可得的 x,r,g,t 训练 matched 学生：direct64、q+v、Δq、Δq+Δv、B-F folded、B-F factored、A0、PCA/marginal。分别报告：
(a) teacher oracle 投影误差；
(b) student 系数/统计预测误差；
(c) same-state native head/flow；
(d) online main GEMM / QDQ / head / launch / memory 的完整成本。

**不同预测**：oracle 领先但学生不能实现，说明风险在计算可预测性而不是投影选择；等成本 direct64 或 A0 更好则应调整或关闭该具体 M-B 压缩版本，保留其机制发现。

### E3：完整 40×4 free-rollout 与部署

只让 E1/E2 有明确独立来源增益和成本可行性的候选进入完整视频，比较 Dense、A0 和同成本最邻近方法；检查 prompt/动作/时间语义、主体/背景、连续运动、成像、相机与ROI，以及原生整机 wall-time/显存。独立来源为统计单位；事先冻结新确认集与来源暴露。

**判定**：同时满足质量与真实成本增益才认定部署 Pareto；不通过时保留具体工作点与反向机制，不扩写为整个函数表示家族的失败。

## 10. 可执行决策与后续分流

**当前建议**：保留 M-B 在方法研究中的位置。先完成 B0 中性 producer 和 E1 的 fixed-readout vs joint 关键反例；E2 提前加入 direct64 和残差增量学生作为最强对照。不要把 B0 重放成功、65 维数学恒等式或单一 source 的 finite 差异当作进入全80学生训练的充分条件。

**出现以下结果时的建设性转向：**

| 观察 | 最有信息价值的下一解释/方向 |
| --- | --- |
| joint 与 fixed-readout 基本等效 | 将末层看作解析低秩读出问题；把主要新机制移向早层跨 token 或其它状态依赖 |
| oracle 领先、student 不领先 | 研究输入条件充分性/可预测性，改变合法输入或低成本学生结构 |
| student 局部领先、rollout 不领先 | 将状态转移和多步误差传播纳入训练目标，保留同状态科学结果 |
| MAC 降低、实际 wall-time 不降 | 研究 fuse/packing/驻留/批形状，将 O-D 执行作为独立可创新对象 |
| RAFT 改善而语义/动作无相应改善 | 重新拆分视频观测器、纹理与真实运动，而非用光流代理定义全部视频功能 |

**终结判断**：M-B 目前是值得研究的下游感知函数压缩框架，但还不是经验证的高潜力加速算法。其最关键的价值不是「65 个统计量」，而是**配对敏感度是否能够提供超越普通紧凑函数、并能被低成本学生兑现的独立增益**。

## 11. 评议来源与不变更说明

- [现行 M-B v2](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)
- [M-A/M-B/M-C 整合方案 v2](../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)
- [原有独立评议（支持 ALG/SRC 的确切版本）](../../method_discussion/discoveries/E0/b-sensitive-coordinates-method-independent-review-20261028.md)
- [B0/B1 Agent 后续任务](../../agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)
- [配对敏感度数学](../../method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)
- [仓库 AutoResearcher 基础](../../../AGENTS.md)

本文独立存放于 research/independent_reviews/M_B_20261028，保留与既有 METHOD_SNAPSHOT/QUEUE/ACTIVE 的角色分离。正文是 **独立意见＋可证伪研究设计**，不修改原方法、不给其他 Agent 代投票，也不暗示本轮完成了未运行的实验。
