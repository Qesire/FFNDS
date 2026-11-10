# G20 · FFN 终端可观测增量算子：G19 Q1-r2 的数学重审

**日期：2026-11-10 · 身份：DESIGN_ONLY / ALG+SRC+LIT / REVIEW_REQUIRED / NO_NEW_RUN**

**固定对象：** Wan2.2-I2V-A14B-LightX2V-Distill-4step，HHLL、无 CFG、BF16、A100 80GB、832×464×81 帧；高低噪声专家各 40 层，FFN 输入维度 `d=5120`、隐藏维度 `m=13824`。本文件记录基于既有 G19 的理论重推导与下一轮研究决策，引用既有真实结果而不改变其身份，不构成实验执行授权。

**关联入口：** [G19 原始方法和全部实测](../20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [G19 证据和源码](../../reports/20261107/g19/README.md) · [G20 可辨识实验设计](G20_FUNCTIONAL_OPERATOR_EXPERIMENT_DESIGN_20261110.md) · [FFN-only 基线](../20261107/ffn_baseline_reproduction/README.md)。

## 1. 新的科学判断

G19 的问题应当从“更准确地拟合一张 FFN 替代矩阵”表述为：

> 给定真实视频生成轨迹与硬件预算，哪些 FFN 的输入响应—输出写入作用需要在当前去噪步重新计算，哪些可由锚点历史保持，哪些能够用更低维、共享或结构化算子更新？

原先讨论的“最终输出拟合 → 时间差分 → 从后层向前层拟合 → 所有层低秩 A”构成四个可比较的候选手段。新的顺序应为：**真实增量函数 → 终端可观测性及跨层耦合 → 稳定有效维度与可用的在线信息 → 执行模式及拟合目标 → 自由学生视频与真实延迟**。

中心科学对象写作 `(\mu,\bar J_F,\mathcal T,\mathcal C)`：
- `\mu`：在同一生成轨迹上匹配的锚点—命中输入、gate、来源和时空坐标的分布；
- `\bar J_F`：沿实际两时刻输入线段的精确 FFN 平均 Jacobian；
- `\mathcal T`：局部写入对去噪输出、采样末端 latent 与视频时空观测的传播；
- `\mathcal C`：原生、全 A、低秩、Rank-0 历史复用等模式在实际硬件上的成本。

这些对象的含义互相独立。低局部 MSE、终端低可观测性、低秩和真实省时是分别需要证明的性质。

## 2. 文献如何组成计算理论，而非直接移植算法

1. **Geva 等，Transformer Feed-Forward Layers Are Key-Value Memories（EMNLP 2021）**：经典 GELU FFN 具有键值式的条件读取—写入形式。这与当前 Wan GELU FFN 的代数结构直接相容。
2. **Geva 等，Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space（EMNLP 2022）**：FFN 写入可逐神经元加性分解；在词表解码下得到加性的 logit 更新，softmax 概率本身由全部 logits 耦合决定。
3. **Dar 等，Analyzing Transformers in Embedding Space（ACL 2023）**：embedding/LM-head 提供权重的一个可解释观测系；中间层的静态投影与它对最终输出的真实传播需分别分析。
4. **Toy Models of Superposition（2022）、ConceptVectors（Hong、Geva 等，2025）、Flash Interpretability（2025）、ROTATE／Disentangling MLP Neuron Weights in Vocabulary Space（COLM 2026）**：物理神经元可以混合多种功能，静态概念方向可用于候选定位，并通过实际激活和干预检验；ROTATE 使用词表 logit 峰度搜索多个相关方向，并未自动提供唯一正交的函数分解。[ROTATE 原文](https://arxiv.org/html/2604.06005v2)。
5. **Engram、STEM、DeepEmbed**：架构层面探索静态检索与条件动态计算的分工。它们启发对计算时点和存储位置的重新设计，但其语言模型词表检索机制并非 Wan 视频 FFN 的现成模块。

**可迁移的数学共同点：** 读取的状态依赖、写入方向的叠加，以及后续消费者决定的功能。Wan 没有供 ROTATE 原样使用的语言 LM-head；其自然的功能观测空间是去噪输出、latent 时空变化和后续消费者响应。

## 3. 真实计算图和精确恒等式

采用仓库的**行向量约定**。对单个 token `u\in\mathbb R^{1\times d}`：

$$
F(u)=\phi(uW_1^\top+b_1^\top)W_2^\top+b_2^\top,\quad
W_1\in\mathbb R^{m\times d},\quad W_2\in\mathbb R^{d\times m}.
$$

`\phi` 是 `GELU(approximate='tanh')`。`u` 是原生 norm2 与条件 scale/shift 后的 FFN 输入；FFN 输出后才乘当前 gate，再加入残差流。Wan 此处属于 GELU FFN，不能把语言模型 SwiGLU 的 gate/up/down 对应关系直接搬进该计算图。

令 `w_j\in\mathbb R^d` 为 `W_1` 第 `j` 行的列表示，`v_j\in\mathbb R^d` 为 `W_2` 第 `j` 列。则：

$$
F(u)=\sum_{j=1}^m \phi(uw_j+b_{1,j})v_j^\top+b_2^\top.
$$

定义锚点 `s`、命中 `t` 的 `\Delta u=u_t-u_s`；对每个隐藏神经元的精确割线斜率：

$$
\beta_j(u_s,u_t)=\int_0^1
\phi'(u_sw_j+\tau\Delta u\,w_j+b_{1,j})\,d\tau.
$$

于是得到**精确代数恒等式**：

$$
\Delta F=F(u_t)-F(u_s)=
\Delta u\,\bar J_F,\qquad
\bar J_F=\sum_{j=1}^m\beta_j w_jv_j^\top.
$$

当前 G19 Q1 使用 `A(\alpha)=\sum_j\alpha_jw_jv_j^\top=W_1^\top\mathrm{diag}(\alpha)W_2^\top`；锚点缓存和命中输出为：

$$
C_s=F(u_s)-u_sA,\qquad
\hat F(u_t)=u_tA+C_s=F(u_s)+\Delta u A.
$$

因此其**同输入函数误差**为：

$$
E_F=F(u_t)-\hat F(u_t)
=\Delta u(\bar J_F-A)
=\sum_j(\beta_j-\alpha_j)(\Delta u\,w_j)v_j^\top.
$$

这表明三个可分离的误差来源：神经元割线斜率随输入变化、真实输入在读取方向上的投影、写入方向的叠加／抵消。`C_s` 保存的是锚点的完整非线性残差基准，不是固定通用 bias。锚点原生计算只保证**同输入 FFN 替代误差为零**；先前步骤导致的 latent 漂移继续存在。

当前 Q1 的正规方程已经包含两个神经元的读取与写入交互：

$$
H_{jk}=\mathbb E_{\mu}\left[
(\Delta u\,w_j)(\Delta u\,w_k)
v_j^\top D_g^2v_k\right],
$$

其中 `D_g` 表示当前 gate 的对角算子。其 Frobenius 矩阵先验的 Gram 为 `K_{jk}=(w_j^\top w_k)(v_j^\top v_k)`。因此 Q1 已经联合处理**同层神经元耦合**；新路线关注的是跨层、跨去噪步与终端功能耦合。

## 4. 终端可观测性：函数空间而非权重空间

对站点 `i=(t,\ell)` 定义 `X_i=\Delta U_i`、`Y_i=\Delta F_i`。门控后的局部替代误差（候选减精确）为：

$$
e_i(A_i)=G_i\odot(X_iA_i-Y_i).
$$

沿冻结的参照轨迹，设 `\mathcal J_i` 将该站点注入的误差映射到某个终端观测 `y`，则一阶模型为：

$$
\delta y\simeq\sum_i\mathcal J_i[e_i(A_i)].
$$

对于完整自由学生，需要将**去噪网络、采样器与锚点缓存**都视为增广动力系统。`\mathcal J_i` 因而依赖选择的参照轨迹，误差交互与高阶反馈必须通过重新运行学生检验。

若固定 `X_i,G_i,\mathcal J_i` 及正半定目标度量 `M`，终端二次目标为：

$$
L_{\rm terminal}=
\Big\|\sum_i\mathcal J_i[e_i(A_i)]\Big\|_M^2,
\qquad \|z\|_M^2=z^\top Mz.
$$

其分块 Hessian 形如 `H_{ik}=\mathcal T_i^\ast M\mathcal T_k`，其中 `\mathcal T_i[\Delta A_i]=\mathcal J_i[G_i\odot(X_i\Delta A_i)]`。对角块表示单站点敏感性，非对角块表示联合误差叠加和抵消。固定传播模型、线性参数化与二次正则时是凸二次式；真实闭环与因子化低秩优化则一般为非凸问题。

**终端等价类：** 若 `\mathcal T_i[\Delta A_i]=0`，该方向在当前样本和线性化观测下不可辨识；换来源、状态、输出定义可能打破等价。故终端小误差与内部稳定性是两个不同的约束。

**样本维度反例：** 仅给一条非零行增量 `x` 和目标行输出 `y`，矩阵 `A=x^\top y/\|x\|^2`（秩至多 1）即可令 `xA=y`。单样本 Rank-1 精确拟合仅说明观测约束有限；要判断稳定有效秩，必须进行来源外、状态外以及完整自由视频验证。

## 5. 全局目标应加入哪些路径约束？

建议将完整四步末端潜在状态作为主要终端评价，同时使用同输入去噪输出与真实学生自由轨迹构建不同用途的监督：

$$
L=\lambda_f L_{\rm final}+\lambda_pL_{\rm path}
+\lambda_{\ell}L_{\rm local}
+\lambda_rL_{\rm prior}
+\lambda_\tau L_{\rm temporal}.
$$

- `L_{\rm final}=\|\hat Z_4-Z_4^D\|^2`：相同初始噪声和 scheduler 的完整学生四步末端 latent。
- `L_{\rm path}`：各调用的最终去噪预测误差；需记录同一 latent 上的 Dense-vs-学生比较（模型替代效应），也可单独记录各自自由轨迹的状态差（累积漂移）。
- `L_{\rm local}=\sum_i\|e_i(A_i)\|^2`：限制大幅内部偏差仅靠跨层抵消得到相同终端输出。
- `L_{\rm prior}=\sum_i\|A_i-A_i^{Q1r2}\|_F^2`：提供现有执行上已验证的起点；允许研究其他功能度量替代绝对 Frobenius 先验。
- `L_{\rm temporal}`：沿视频 latent 时间轴的误差差分。上述 λ 的大小及归一化须在独立验证前冻结；这里列出候选的含义，并未给出已校准的权重。

一阶时间项应作用于**候选与 Dense 的误差** `e=\hat Z-Z^D`，即 `\|D_\tau e\|^2`。其 Fourier 权重在周期边界近似下为 `4\sin^2(\omega/2)`，强化高频误差但并不直接等于物体运动正确性。若加入参考运动场的 warping，必须固定 Dense 参考对应关系并明确遮挡、可见性和时空网格映射；RAFT、CoTracker 保留独立验收地位。考虑到蒸馏 Dense 本身可能带有高频闪烁，单纯教师模仿还应配合独立画质及闪烁评价，避免把已有瑕疵当作目标优点。

## 6. 重新审查 A 的低秩与计算模式

**低秩前提：** FFN 在实际输入—终端功能联合度量下具有稳定衰减的有效谱，而非只有在少量校准行上可拟合。

反例：当 `\Delta u` 的协方差为单位阵、真实目标算子 `A_*=I_d`、终端保留全部方向时，任意秩不超过 `r` 的矩阵满足

$$
\mathbb E\|\Delta u(A_r-I_d)\|^2=\|A_r-I_d\|_F^2\ge d-r.
$$

因此 FFN 增量可被某张固定矩阵精确表示，与该矩阵容易低秩压缩，是两条不同的命题。

若下游度量可以近似分离为输入协方差 `\Sigma` 和输出 PSD 度量 `G`，可研究奇异值谱 `\Sigma^{1/2}A G^{1/2}`（行向量约定）；实际跨 token、跨层 Jacobian 的一般 Hessian `H` 不必是 `G\otimes\Sigma`。功能加权 SVD 因而适合作为初始化与诊断，最终需要在真实目标中复核。

**计算模式应包含：** Rank-0 历史输出复用（`A=0`）、Rank-r 两级 GEMM、完整 A GEMM、原生 `W_1`–GELU–`W_2`。不同专家/层可以选择不同模式。Rank-0 仍使用当前 gate，意味着它是历史 FFN 输出复用而不是旧门控残差复用。

以当前四步且两次锚点、两次命中、所有步均按相同 rank-r 表示参与投影的方案，四步 FFN 主 GEMM MAC / Dense 约为：

$$
\rho_{\rm MAC}(r)=\frac12+\frac{r}{m},
\qquad m=13824.
$$

完整 `A` 时 `\rho=0.685185`，rank-512 时 `\rho=0.537037`，rank-0 时 `\rho=0.5`。80 张 BF16 完整 A 的逻辑存储约 3.90625 GiB；rank-512 因子为约 0.78125 GiB。现有单残差缓存仍约 12.0804 GiB，另行分配中间投影缓存时应重新核算。上述为**理论 MAC/逻辑字节**，速度与显存峰值需要原生 kernel、offload 与真实 A100 实测，特别注意 rank-r GEMM 的布局与启动开销。

计算预算问题可写为：

$$
\min_{\{m_\ell,A_\ell\}}L_{\rm functional}(\{A_\ell,m_\ell\})
\quad {\rm s.t.}\quad
T_{\rm measured}(\{m_\ell\})\le B.
$$

其中 `m_\ell` 在本式为执行模式标签（与隐藏宽度 `m=13824` 区分）。可先用逐层边际收益初始化，再考虑跨层联合效果与实际 wall-time。

## 7. 拟合顺序的重新定位

“从最后一层逐步向前”是一个可以比较的**求解策略**，并非由终端损失推出的最优性定理。固定参考轨迹下，任意层的下游敏感性均能通过反向传播/JVP-VJP 得到；无需先拟合所有后层。前向扫描有利于让后层吸收早期残差，后向扫描先稳定下游组合；跨层耦合强时应迭代或分组联合拟合。

建议对照四种求解器：前向一轮、后向一轮、后向初始化+交替重扫、跨层联合拟合。控制相同初始化、目标、层集合与优化预算，比较来源外终端误差及自由学生视频。固定强凸二次目标的充分收敛分块法可到同一最优点；真实学生轨迹、低秩分解、有限预算条件下顺序可能影响结果。

**四步去噪必须按真实 0→1→2→3 重放。** Step 1 的近似会影响 Step 2 的原生锚点状态，再影响 Step 3 的输入。跨层后向优化与跨去噪步时间上的正向状态传播不是同一操作。可采用“固定轨迹的终端线性化 → 有界更新 → 学生完整重放 → 更新线性化”的信赖域迭代，并记录预测/实际改善比。

## 8. 现有证据与新结论的分界

- **ALG（本文件）：** 平均 Jacobian 精确恒等式；Q1 误差读取—写入分解；单样本 Rank-1 反例；全秩映射的低秩误差界；固定传播模型下的跨层 Hessian 结构。
- **SRC（已核对）：** Q1 按真实 gate、来源等权拟合全部斜率，含矩阵先验和 PCG；Q2 是从 Q1 追加一次秩一方向，最终 A 依然完整；`fused_post` 保留 BF16 舍入及缓存生命周期。
- **VIDEO / NATIVE（既有 2026-11-07 开发集）：** Q1-r2 的 CAL BF16 MSE 0.01668745→0.01668569（相对旧 Q1 约 −0.011%）；七个已暴露来源等权 RAFT 对 Dense 方向余弦 0.81664→0.8083，运动模长比 0.89507→0.9017，六/七来源方向余弦下降。真实 Q1-r2 新生成 14 段视频；自定义质量面板与官方完整 benchmark 不同。
- **SYS（历史 G0）：** 非冷启动顺序配对 Dense 39.4345s→G0 37.6135s，少约 4.62%；不是 Q1-r2 新配对速度证明。两项打包/省 W1 预取实验数值一致，但未取得稳定总时间收益。
- **OPEN：** 功能低秩谱与来源外稳定性、全局目标对运动/画质的提升、运动对齐损失、真实每层模式最优分配、终端线性化在自由学生上的有效性、ICC 的当前同配置实测与论文新颖性评估。

## 9. 决策、停止分支与相关材料

优先以[配套可辨识实验设计](G20_FUNCTIONAL_OPERATOR_EXPERIMENT_DESIGN_20261110.md)执行 **E1 终端传播诊断**，使用当前已保存的 Q0/Q1-r1/Q1-r2 及真实学生/教师资产作为起点。它检验终端一阶传播、跨层非加性及状态漂移，决定如何构造 E2 全局目标。

- 若下游一阶预测和真实扰动显著失配，优先建立信赖域/真实学生重放，而不是增加全局二次代理的求解轮次。
- 若局部误差与终端敏感性排序基本一致，当前 Q1 的局部目标仍有价值；重点转向执行模式及降低真实系统开销。
- 若功能谱在来源外趋于平坦，推迟统一低秩假设，优先研究层选择和原生/缓存模式组合。
- 若时间差分只改善其代理而无独立视频指标收益，转向明确的运动对应与真实视觉评价。

**最近邻工作的研究边界：** ICC（增量校准与 SVD）、GOC（缓存误差梯度传播）、Q-VDiT（时空蒸馏）、Chipmunk（FFN 激活增量动态稀疏）分别已经包含相关组成部件；本研究的新颖性需由**实际可观测函数、条件计算单元及其同预算视频-系统收益**证明，不以组件拼接和新损失名称认定。

**执行治理：** 本文件是新的设计与数学版本，没有新生成、校准、GPU/CPU 模型实验、独立审查结论或 ACTIVE 更新。任何新 RUN 先按 [实验前独立复审规则](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)完成主审→不同上下文独立评议→修订→recheck，再建立独立冻结 scope；旧收据、失败与资产保持只读。
