> **2026-11-12 当前执行设计：** [G20路线v3](../20261112/G20_EXPERIMENT_ROADMAP_V3_20261112.md)已吸收R1/R1b证据并指定图文规模、辨识实验、资产与顺序。本文保留11/11文献/机制综合身份；其中ICC未运行和旧Next不覆盖后续已发表结果及新路线。

# G20 / ICML：正向函数机制与反向文献思考的统一研究决策

**日期 2026-11-11｜状态：DESIGN_ONLY / ALG+LIT+ARCHIVED_OBS / REVIEW_REQUIRED / NO_NEW_RUN**

**入口**：[G19 方法和真实测试](../20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [G20 原推导](../20261110/G20_FFN_FUNCTIONAL_OPERATOR_REDERIVATION_20261110.md) · [G20 E0–E4](../20261110/G20_FUNCTIONAL_OPERATOR_EXPERIMENT_DESIGN_20261110.md) · [新增论文 L99–L109](../literature/20261111/G20_FEW_STEP_AND_INCREMENTAL_ACCELERATION_PRIMARY_REVIEW_20261111.md) · [Chipmunk 旧证据/ICC 尚缺](../20261107/ffn_baseline_reproduction/G19_BASELINE_AND_ASSETS.md)。

## 0. 中心命题与真实痛点

当前已经具备能实际部署的四步 Wan2.2 I2V FFN 增量算子，其线性折叠/锚点缓存都有前例。**ICML 需要回答的不是还能如何调整一张 A，而是：蒸馏与少步采样在何处改变了可重复计算的函数结构？在合法廉价信息下，哪些变化确实可以预测，并跨 FFN/Attention 换来真实 wall-time 收益？**

四个决策性事实：

1. **现有 G19 是真实执行算法**：HHLL / no CFG / BF16 / A100 / 832×464×81；每专家40层，160个 FFN 站点，0/2 anchor、1/3 hit。在线 `A` 固定，门控与残差使用当前状态，80 张 `5120×5120` BF16 A；当前槽单残差约12.08GiB。历史 G0 11对非冷启动 wall Dense 39.4345s → G0 37.6135s，减少4.62%，不等于新Q1-r2实时系统收益。
2. **历史全层 FFN 原生输入诊断（αH，不是新版 Q1）**：四 held sources 中固定斜率隐藏误差的83.45%–85.02%来自来源内、同通道内部变化；来源间斜率漂移仅约8%–10%。GELU 的激活中点符号形成严格的冻结斜率误差下界；W2/gate 后残差通道间交叉项常为正。对照 [本仓 2026-11-05 历史报告](../20261105/G19_METHOD_AND_RESULTS_20261105.md)，不能把该 hidden αH 分解当作 Q1、终端指标或全局最优下界。
3. **真实学生路径敏感**：历史 high-only/low-only/both 的同输入与后缀视频质量排序不完全一致；新版 Q1-r2 的局部 BF16 CAL MSE 改善约0.011%，七开发来源RAFT方向均值却从0.81664到0.8083（6/7下滑），同时幅值比0.89507→0.9017。不能据局部增量 SSE 推断动作方向与末端视频结果。
4. **FFN-only 端到端加速上限**：若实际 FFN 时间占比为 p_F，且四步两次原生锚点不可省、hit 计算变为零且无附加开销，则 S_max=1/(1-p_F/2)。完整 A 的历史 FFN 主 GEMM MAC 比例约0.685185，真实墙钟受 SDPA/模型offload/缓存搬运约束；p_F 尚需同硬件分解实测。故**更大收益须涉及非FFN计算或改变锚点成本**。

用户近期报告 G19 已优于 ICC/Chipmunk，作为 **USER_REPORTED_NEEDS_MATCHED_RECEIPT** 研究动机；目前公开归档的 Chipmunk P64 是旧两暴露来源的全160成功结果，ICC 的同四步 FFN-only 移植在 2026-11-07 合同仍为 NOT_RUN。本稿不补造新结果，不将不匹配的源/硬件/Kernel拼成名次。

## 1. 先识别前人已经建立了什么

**直接同构构件**：
- C17 TARDIS 已做 GELU 等 FFN 激活的逐通道仿射近似与 W1/W2 权重折叠，在线异常恢复。
- L99 ICC 的外层形式是 `old output + low-rank input delta`，只是 FFN 的中间 GELU 与两级低秩校正仍在。
- L09 Chipmunk 动态选择隐藏列真实更新；实际 Wan 默认作者例子的 MLP 是关闭的，另需区分作者整系统与本仓 FFN 移植。
- L52 L²P / L39 SVD-Cache 已占据历史特征线性预测、主/残差子空间分离；L102 LearniBridge 占据跨 prompt 共享低秩校正；L101 RFC 占据当前输入估计输出变化。
- L104 DisCa 已将少步蒸馏模型的特征缓存困难形成 CVPR 2026 论文；L105 AutoTarget 已研究 few-step solver-dependent cache target；L106 Chorus 已研究四步跨请求复用。
- L103 BudCache 占据固定预算终端输出质量，L108 SenCache 占据敏感度触发，L107 RECAP 占据稀疏 Attention + 缓存残差 + 在线反馈。简单拼接/加新损失名不会独立构成新颖性。

**反向录用论文的证据链**：从已发表工作的明确失效条件出发，寻找此前没被有效利用的可观测结构，给出有区别的计算动作，并在相同预算下对比原Dense、最近邻和其直接组合。不能反推审稿人的个人录用原因。少步模型已有人研究，*难*不自动等于*新*；不必为好看速度回到50步，也不能只在单一四步模型上声称普适。

## 2. 精确的基础：G19 不是单纯静态 FFN 替换

使用行向量。真实 FFN 为

$$
F(u)=\phi(uW_1^\top+b_1^\top)W_2^\top+b_2^\top.
$$

定义两端输入 `u_s,u_t`，`δ=u_t-u_s`，`w_j` 是 W1 第 j 行的列表示，`v_j` 是 W2 第 j 列。精确有限割线给出

$$
\begin{aligned}
F(u_t)-F(u_s)&=\delta\,\overline J_F,\\
\overline J_F&=\sum_j\beta_j w_jv_j^\top,\\
\beta_j&=\int_0^1\phi'(u_sw_j+b_{1,j}+\tau\,\delta w_j)\,d\tau.
\end{aligned}
$$

G19 Q1 采用 `A(α)=Σ_j α_j w_j v_j^T`，原生锚点 `F(u_s)` 与单残差 `C_s=F(u_s)-u_sA`，命中 `Fhat(u_t)=u_tA+C_s`。于是候选减真实：

$$
\begin{aligned}
e_F&=\delta(A-\overline J_F)=R_A(u_s)-R_A(u_t),\\
R_A(u)&=F(u)-uA.
\end{aligned}
$$

上式是**精确代数恒等式（ALG）**，不是新算法的免费在线 oracle。G19 Q1 的隐藏斜率基底、W2 输出写入、当前 gate 的联合 Gram 已处理同层神经元交互；下一步的不同信息是**同一来源内状态变化、两时刻缓存状态与下游终端传播**。

## 3. 新的首要判别：蒸馏改变了哪一种冗余？

在匹配的真实调用对上，彼此独立地测：
- **隐藏增量可稀疏更新性** $E_{\mathrm{sparse}}(k)=\min_{|S|\le k}\|g\odot(\Delta h_{\bar S}W_{2,\bar S}^\top)\|^2$（固定同一 token/组预算与门控输出读出；理想后验 subset oracle，仅诊断，真实 mask/gather 成本单计）。不能以 top-k 幅值代替最优有符号输出误差。
- **低秩输入/中间增量可表示性**：ICC 的两级 rank-r 保留的 FFN 增量误差，分别报告 W1 输入投影误差与 W2 读出误差及缓存 4/2 态内存；rank-r 不是随样本重算的全自由矩阵 oracle。
- **全维线性增量可预测性**：在同一 source/train/holdout、当前 gate 与预激活条件下，比较 Q0、Q1、无结构受限的 ridge 完整 A，以及时间/状态分组。
- **终端可保持性**：同状态注入的去噪输出与完整学生自由轨迹分别报告，视频动作/运动方向、纹理、时序、主体一致性和真实 wall。这里的终端误差不由同层 SSE 排名决定。

竞争机制：
H_sparse：蒸馏使重要隐藏增量的支持更稠密，但输入→完整输出仍可用 A 预测 → Chipmunk 类 subset 在相同计算预算更难；
H_rank：重要增量的输入/输出有效秩提高，ICC 两级低秩误差上升，而完整 A 保持；
H_curve：少步大区间和状态偏移使有限割线斜率变化加剧，固定 A 的遗漏在 GELU 读出方向/曲率而非支持或秩；
H_suffix：本地误差相似，差别来自下游传播、锚点漂移或 BF16；
H_sys：质量代理相近，但某类方法付费 mask/index/HBM/cache、offload 改变全系统成本。

**“蒸馏导致什么”是因果问题**：分别记录原/蒸馏权重、各自真实轨迹、相同输入交叉评估、步数/实际timestep；不要把 50→4 的采样间隔、权重变化、CFG 变化、高低专家切换当作同一个因素。少步原模型的四步强行推理可能成为 OOD 诊断而非公平质量基线。

## 4. 非线性数学空间：有限增量依赖位移和中点

令 `m=(u_t+u_s)/2`，`δ=u_t-u_s`，

$$
\mathcal D_F(m,\delta)=F(m+\delta/2)-F(m-\delta/2).
$$

对共同状态位移 `a` 和端点差位移 `b`，分别有（行向量导数作用）：

$$
\begin{aligned}
D_m\mathcal D_F[a]&=a\big(J_F(u_t)-J_F(u_s)\big),\\
D_\delta\mathcal D_F[b]&=\frac b2\big(J_F(u_t)+J_F(u_s)\big).
\end{aligned}
$$

G19 的 `δA` 对中点的导数恒为0。通过两次微积分基本定理，

$$
\mathcal D_F(m+a,\delta)-\mathcal D_F(m,\delta)
=\int_0^1\int_{-1/2}^{1/2} D^2F(m+\lambda a+\tau\delta)[a,\delta]\,d\tau\,d\lambda.
$$

对 GELU FFN，`D²F(u)[a,δ]=Σ_j phi''(uw_j+b1_j)(a w_j)(δ w_j)v_j^T`。这将真实观测的符号中点/通道内部变化转化为明确的读写-曲率作用。**数学恒等式已有微积分先例；新意仅可能来自可廉价测量的结构与部署算法。**

缓存双时刻敏感性（固定 A，当前候选状态）：

$$
\begin{aligned}
\mathrm d\widehat F&=(\mathrm du_t)A+(\mathrm du_s)(J_F(u_s)-A),\\
\mathrm d(\widehat F-F(u_t))&=(\mathrm du_t)(A-J_F(u_t))+(\mathrm du_s)(J_F(u_s)-A).
\end{aligned}
$$

它解释为什么锚点本身随前序学生漂移会带来额外作用；应通过分别扰动 anchor/hit 和完整自由 rollout 的剂量响应检验，避免仅在 teacher 捕获输入上判高精度。

**候选数学命题**：完整/线性主响应可满秩，*非线性变化部分*的可观测曲率或输入方向可能低维；试验 `F(u)≈uA+b+g(uP)Q^T`，其中 P/Q 是固定小基底，g 为小函数。若全部二阶变化在输入 `ker P^T` 消失且输出落在 Q 方向，在连通且路径适当的定义域上存在该形式；这是函数分解的已知几何条件，并非本仓已测低秩事实。需把重建函数的**绝对值、有限跨步增量和状态方向响应**三个目标分开测，不把单样本 rank-1 插值当稳定秩证据。

## 5. 信息可用性的严谨命题

令真实增量 `Y=F(u_t)-F(u_s)`，在线可负担特征 `Z=Ψ(u_t,历史,条件)`，其购买成本 `C(Ψ)` 必须列明首次可用时点。平方预测损失的最佳风险为

$$
R(Z)=\mathbb E\|Y-\mathbb E[Y\mid Z]\|^2
=\mathbb E\,\mathrm{tr}\big(\mathrm{Cov}(Y\mid Z)\big).
$$

新增合法特征 S 的理论最优收益：

$$
R(Z)-R(Z,S)=
\mathbb E\left\|\mathbb E[Y\mid Z,S]-\mathbb E[Y\mid Z]\right\|^2\ge0.
$$

它来自经典条件期望正交投影，**不是新颖定理**。只有定义受限成本的特征类 `C(Ψ)≤B` 才有科研意义；完整当前输入与任意昂贵模型可确定性地产生 Y，零条件方差不代表能省时。元信息/符号/幅值/gate/浅层统计仅作为候选，先以来源外容量受控的预测器估算增量信息价值，再决定是否购买。若必须先执行完整 W1/GELU 才得到代理，应将真实成本计入、可能直接出局。

## 6. 从 FFN 扩展到 Attention：应以读写变化而非缓存组合为起点

对一个 Attention 头、固定匹配的 key/query 坐标和 mask/归一化定义，`O=PV, P=softmax(QK^T/√d_h)`，有

$$
\Delta O=\overline P\,\Delta V+\Delta P\,\overline V.
$$

它精确地区分内容 value 变化与读取分配变化；仅有 P 的重叠度、QK 范数或漂移标量不足以决定 `ΔO`。token 间耦合意味着 FFN token-local 的 A 不应原样替代完整 Block。第一步保留原生 QKV/RoPE/输出投影，仅探索主要 Attention 聚合的低成本实际增量；RECAP 已有 sparse+coarse+detail cache+online error，SLA / BWCache 也已有强系统先例，故不能以“组合稀疏和缓存”宣称原创。

**可执行额外价值**只存在于：同一在线低成本状态信息能否预测终端重要的读写变化，或能否在相同质量和时间预算下优于 G19 + 强 Attention 加速的独立组合。每种候选算子记录当前 Q/K/V、token索引、log-mass/分母、gate、BF16舍入和真实 kernel 代价，后验得到完整 QK/P 的特征只能当付费 oracle。

## 7. 两条必须保留的限制性结论

1. **仅更换终端正定输出度量，不保证自由矩阵 A 会改变。** 固定下游 R>0、单模块自由最小二乘且输入协方差可逆时，`min_A E[(Ax-y)^T R (Ax-y)]` 的极小值仍为 `A*=E[yx^T]E[xx^T]^{-1}`。G20 只有在受限的 W1-斜率-W2 基底、低秩/共享参数、样本依赖传播度量、跨模块联合或真实学生闭环中才获得新的优化自由度。G19 已考虑随样本变化 gate，终端敏感性要证明新增信息。
2. **时间差分不等于运动正确。** 视频 latent 时间高频惩罚可能压制误差，也可能复制 Dense 本身的闪烁；RAFT 同像素光流、CoTracker 共同可见集合、相机补偿、语义事件和人工连续播放分开验收。

## 8. 与 ICML 对应的实际研究路线及出局条件

| 优先 | 判别实验（计划） | 信息与候选对照 | 足以改变决策的结果 |
|---|---|---|---|
| **P0** | 旧资产重审：Chipmunk/ICC/G19 在相同 preW1 输入和四步条件的增量稀疏性、ICC-rank、全维 A 风险曲线 | 保留作者原实现、简化相同代价 oracle、同gate及source；校验 SHA/数值族/缓存/实际时间 | 若匹配证据不存在，先补最小同源对照；若三种结构都退化，重新定位状态/下游 |
| **P1** | **G20 E1：单点、双时刻、双层有限幅注入 + JVP/VJP**；同参考状态/学生自由轨迹分别记录 | 新旧Q1、Rank0、原生、冻结 `η=0,0.5,1`；source为独立单位 | 传播敏感度可预测才进入终端目标；若失配明显，转向真实学生重放和局部信赖域 |
| **P2** | **中点-曲率与廉价信息可预测性**：在固定真实 `δ` 下扰动中点，估计 `J_t-J_s`、有符号read/write、条件风险下降 | 原A、无结构ridge、A+小型g、局部原生回退；对可用前置信息量进行成本计量 | 外来源稳定解释足够多**重要误差**，且在线额外费用低，才推进状态核 |
| **P3** | 训练/蒸馏/步长因素区分：原权重、蒸馏权重各自轨迹与相同输入交叉测量，时间条件对齐 | 不把强行少步原模型与正式部署基线混用 | 证明是权重结构、步长还是专家/条件变化导致异质性 |
| **P4** | Attention 聚合局部机制：`PbarΔV` 与 `ΔP Vbar`、读取-写入关联与实际 kernel 成本 | Attention 原生、强稀疏/缓存、G19+强Attention独立组合 | 新算法必须在质量约束下产生独立组合之外的 wall-time 增益 |
| **P5** | 函数/终端表示 + 真实质量—时延 Pareto；多来源正式 CONFIRM | Dense、G19、Chipmunk作者忠实FFN、ICC作者忠实FFN、TARDIS static/full、L102/104/107相关系统；相同后端/预算/质量 | 无独立来源增益则将 G19 作为可复现基线，局部理论以分析论文主张界定 |

**停止/收缩分支**：① 源外中点效应低：维持共享A，不新增昂贵条件网络；② 非线性谱不衰减：放弃统一低秩g，尝试少量原生回退/算子分组；③ E1 JVP失效：限制更新并自由重放，不大规模求冻结Hessian；④ Attention联合不比独立组合好：保留 FFN 方法，不声称模型级原创联合；⑤ 只有理论MAC收益：先复核内核/缓存/系统时间，再谈加速。

**实验治理**：截至本文只做静态论文/公式/仓库审查，**没有**新 CPU/GPU 数据任务、独立 Agent 评议票、ACTIVE 授权、RUN/EXIT。任何新的模型、校准、视频及计算测试按 [独立预审](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md) 的自审→真实独立评议→修订→recheck→ACTIVE 冻结→执行，不复写旧收据。

## 9. 投稿论证：从已有现象到真正的新问题

可形成的 Introduction 主线是：少步蒸馏缩短可复用轨迹；ICC/Chipmunk 等方法在原作者设置已有效，但在 four-step Wan 的适用性与结构假设必须重新检查；G19 表明非线性锚点+全函数线性增量可在真实四步视频执行；在相同输入变化下，隐藏增量稀疏性、低秩性、线性可预测性以及终端可观测性并不等价。因此 **研究少步重复神经计算在给定信息—预算下可保留的函数增量结构**，并在真实系统 Pareto 上证明何种机制有价值。

需要正面比较已录用/新预印本：L104 DisCa（distilled+learned predictor）、L105 AutoTarget（cache target）、L106 Chorus（inter-request serving）、L102 LearniBridge（50-step Wan low-rank calibration）、L103 BudCache（budget）、L107 RECAP（Attention+cache coupling），以及 L39/L52/ C16/C17 和本仓 ICC/Chipmunk。**“比较原始Dense并取得×倍加速”不足以说明新增机制；与强先例同预算、同约束以及直接组合对照才是最关键证据**。

**可发表贡献现在是研究假说，不是既定结论**：① 蒸馏对三种不同计算冗余的影响具有可重复的结构判据；② 状态依赖有限增量的有意义部分存在可负担的共享表示；③ 该表示跨 FFN/Attention 产生实测模型级收益。若 ②/③ 不成立，继续以 G19 的真实强基线和机制分析形成有限主张，不为了完整故事捏造通用加速。

