# M-B 二阶独立评议：从末层读出到前39层的函数等价、后缀传播与四步可观测性

> 2026-10-28 · 独立扩展评议 / ALG + UPSTREAM-SRC + REPO-EVIDENCE + PRIMARY-LIT + OPEN。  
> 对象：FFNDS LightX2V Wan2.2 I2V A14B 4-step high/high/low/low，两 expert，每 expert 40 blocks；标准 FFN 写回与后缀。  
> 身份：**独立理论/机制评议及最小实验建议**，不是 M-B 方法快照更新、正式方法投票、ACTIVE 运行授权、已有实验结果或部署通过声明。  
> 原评议：[INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)；原方法：[B v2](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)；方法优先级以 [RESEARCH_NAVIGATION](../../RESEARCH_NAVIGATION.md)、[ACTIVE](../../ACTIVE_RESEARCH.json) 为准。  
> 本轮实际完成：阅读 FFNDS 现行文档、仓库已登记的源码审计与原生结果；对照 Wan2.2、LightX2V 上游公开源码及相关论文；建立条件数学命题和反例。**本轮没有执行新模型、BF16 有限差、JVP/VJP、GPU、训练、自由视频或 SYS benchmark。**

## 0. 执行摘要：哪些新结论有用？

1. **ALG-1 / 前层共同规范自由度（gauge）**：对固定条件、无破坏分支的 Pre-LN Wan Block，任意逐 token 的通道公共平移 $E=b\mathbf1_D^\top$ 满足 $B_l(X+E)=B_l(X)+E$；最终 LN/head 消去这一平移。它是跨所有后续 Block 的**精确实数函数不变性**，不同于末层近似径向不变性。原生 BF16 与 FFNDS 分支仍须做有限差核验。
2. **ALG-2 / 巨大但状态依赖的局部零空间**：一个去噪步内任意 FFN 写回到 Flow 的 $J_l:\mathbb R^{MD}\to\mathbb R^{64M}$ 有 $\dim\ker J_l\ge(D-64)M$。$D=5120$ 时单工作点至少 98.75% 的**输入维数**在一阶零空间内，但这些方向会随 source/state 改变，并可跨 token 耦合；**绝不等价于可固定删除98.75%的 FFN 通道或计算**。
3. **ALG-3 / 下游机制分解**：标准 Cross-Attention 在上下文 K/V 固定、视频 Q 按 token 生成时，视频 token 间 Jacobian 是块对角的；Self-Attention 通过 Q/K/V 对多 token 耦合。由此可将后缀敏感度拆成本地条件变换与真正的跨空间—时间传播，而不把两者混成“全局敏感度”。
4. **ALG-4 / 四步有限时域目标**：将 Block 后缀、scheduler 和后续步连起来，得到 $H_{t,l}$ 及 $G^{terminal}_{t,l}=H_{t,l}^\top QH_{t,l}$。它是沿**指定 Dense 轨迹、指定终端读出**的局部二次形式；完整多位置误差还含有**有符号交叉项**，不能单凭逐层 SSE 确定全视频偏移。
5. **方法方向**：B-GAUGE（全层不变量，优先静态/CPU）＋ B-OBS（全层/四步可观测性，CoBRAS/微分 Gramian/VJP）＋ B-JOINT（跨位置有符号有限响应，已有 E0 为先验）。三条都是**候选研究子题**，不是替换现有正式 B0/B1/M-A/M-C 队列的指令。
6. **科研优先级**：近期仍以至少一个能全程生成、保留视频能力且真实减少 FFN 计算的可用稀疏/紧凑方法为主线（现有 A0 是强基线）；本评议的新机制以离线推导、已有数据复用和少量辨别实验并行，不占用已冻结主任务。

## 1. 原生计算图、变量与源码绑定

一去噪步在当前研究合同下处理 $M=31668$ 个 latent patch token，每 token $D=5120$ 通道；每个 FFN $H=13824$ hidden，末端 head 输出 $64$/token。$M$ 对应仓库固定 81 帧、配置分辨率，并非任意生成参数。

令 $X_l\in\mathbb R^{M\times D}$ 为 Block $l$ 的输入残差。单步固定条件 $c=(\mathrm{source},\mathrm{seed},t,\mathrm{expert},\mathrm{context},\mathrm{embed},\mathrm{dtype})$，省去 $c$ 下标：

$$
\begin{aligned}
U_l&=S_l(X_l)=X_l+G_{a,l}\,\mathcal A_l(N_{1,l}(X_l)),\\
V_l&=C_l(U_l)=U_l+\mathcal C_l(N_{3,l}(U_l);K_c,V_c),\\
X_{l+1}&=F_l(V_l)=V_l+G_{f,l}\,f_l(N_{2,l}(V_l)),\\
B_l&=F_l\circ C_l\circ S_l.
\end{aligned}
$$

- $N_1,N_2$ 是 token-wise LN 加固定 AdaLN 仿射调制；$N_3$ 是 Cross-Attention 的 token-wise LN，可带 learned affine；$G_a,G_f$ 是相应广播门控（对具体张量维度需取原生合同）。
- $f_l$ 真实执行涉及 post-Q1、$W_1$、GELU(tanh)、HiF4 Smooth2/Q2(group64→8→4)、$W_2$、e5 写回与 BF16 取整。式子是**连续结构代理**，不是 BF16 逐指令身份式。
- $\mathcal A_l$ 的 $Q,K,V$ 都从当前视频状态获得，带 Q/K RMSNorm 和 RoPE；$\mathcal C_l$ 使用视频 Q 和固定文本/图像 K/V。若实际分支重新计算或修改 context、激活 adapter，则需重建这条边。
- 下游一个 step 的后缀：$H_l=\mathrm{Head}\circ B_{39}\circ\cdots\circ B_{l+1}$（干预位于 Block $l$ 的 **post-FFN/writeback**；空积为恒等）。后续 flow 还经真实 unpatchify 和 scheduler 影响下一去噪状态。

**源码核对**：
- [Wan2.2 upstream model.py](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)：$WanLayerNorm$ 对各 token 通道规范化；$WanAttentionBlock$ 的 $norm1$、$norm2$ 是 LN，$norm3$ 在 $cross_attn_norm=true$ 为带 affine 的 LN、否则是 Identity；标准 $cross_attn_norm$ 构造默认 true。$Head$ 又有 final LN。
- [LightX2V upstream transformer_infer.py](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/networks/wan/infer/transformer_infer.py)：$infer_block$ / $infer_self_attn$ / $infer_cross_attn$ / $infer_ffn$ / $post_process$ 支持上述顺序；存在 $before_proj$、$after_proj$、post-adapter、sequence parallel、sensitive dtype 等**条件分支**，必须按冻结 worker 配置核实开关。
- [仓库冻结末层审计](../../method_discussion/discoveries/E0/b-final-block-head-source-audit-20261028.md)：记录特定 runtime 源文件 SHA、head 形状、BF16 分支；本轮核查的是**上游公开源码，不是对冻结 .runtime 文件的重新逐字节验收**。二者不等同。

### 1.1 可审计的 Block Jacobian

在连续、固定条件、当前 Dense 工作点，$A_l=D B_l$，按复合求导：

$$
A_l=
(I+G_{f,l}J_{f_l}J_{N_{2,l}})
(I+J_{\mathcal C_l}J_{N_{3,l}})
(I+G_{a,l}J_{\mathcal A_l}J_{N_{1,l}}).
$$

这里 $I$ 是 $MD\times MD$ 恒等；门控在当前固定条件下冻结。时序/文本条件若随输入变化，需要增加对应导数项；有副作用 adapter、额外直读 residual 的分支也要单独建模。注意 FFN 注入在本 Block **完成写回后**，所以其第一个后续因子是 $A_{l+1}$，而不是错误地再乘一次 $A_l$。

## 2. 定理 A：所有后续 Pre-LN Block 的逐 token 公共平移等变性

**定义**：$\mathbf1_D=(1,\ldots,1)^\top$，对任意 $b\in\mathbb R^M$ 定义 $E=b\mathbf1_D^\top$。每一个 token 的通道统一偏移 $b_p$ 独立可取。

逐 token LayerNorm 具有精确实数身份：

$$
LN(x+\beta\mathbf1_D)=LN(x).
$$

它对 $\epsilon>0$ 仍严格成立；LN 后固定仿射增益/偏置和 AdaLN 也不会改变该身份。因此 $N_i(X+E)=N_i(X)$。在固定上下文/参数、没有未归一化直读支路时：

$$
S_l(X+E)=S_l(X)+E,\quad
C_l(U+E)=C_l(U)+E,\quad
F_l(V+E)=F_l(V)+E,
$$

故对所有满足条件的 Block：

$$
\boxed{B_l(X+E)=B_l(X)+E.}
$$

通过后续 Block 归纳，再由最终 LN 的平移不变性：

$$
\boxed{H_l(X+E)=H_l(X).}
$$

**推论 A1**：连续 $J_l=D H_l$ 必有 $J_l\,\mathrm{vec}(E)=0$；每 token 的通道公共平移构成跨所有后续层和来源的至少 $M$ 维公共结构零空间（条件保持一致时）。若该 step 的 head/flow 字节级相同，后续确定性 scheduler 也得到同一 latent；这个后续传播结论是条件推论，不是已执行视频验证。

**推论 A2：gated FFN 表示**。若将干预放在 pre-gate $y_l$，要实现 post-write $E$ 需解 $G_{f,l}\,\delta y_l=E$。仅在 gate 各相关通道非零且逆条件数可接受时才可写 $\delta y_l=G^{-1}_{f,l}E$；某通道 gate 为零、Q2 组限制或有限秩 $W_2$ 可导致不可实现或代价极高。零空间的存在**不推导出**可低成本跳过一个真实 64-group Q2、W1/W2 GEMM tile。

**定理适用边界**：
1. $norm3=Identity$ 时，Cross-Attention 可能直接看到 $U+E$，等变链失效；非 token-wise normalization 或外部 direct residual 分支同理。
2. 若开启 $before_proj$ 等对原始 $X$ 的额外投影、post-adapter 或缓存状态回写，须检查其是否满足 $P(X+E)=P(X)+E$。
3. BF16 中逐元素取整的 $fl(fl(x+E)+r)$ 不保证等于 $fl(fl(x+r)+E)$；LN FP32 reduction、量化及多次原位加法会引入有限误差。这是 **ALG 实数定理 / NATIVE 未验证**，不是原生 bitwise equivalence。
4. **径向/尺度不变性不能推广到早层**。末层 LN 对 $x\mapsto \alpha(x-\bar x)+\bar x$ 仅在 $\epsilon=0,\alpha>0$ 精确径向不变（$\epsilon>0$ 近似）；残差分支的加法一般破坏早层尺度等变性。前层可靠的共同不变量是上面的通道平移，而非整体缩放。

**首个验证点**：选普通路径中后段 Block，固定 source/step/expert 与 context，在 post-write 处按 token 注入 3 类 $E$（常数、跨 token 非常数、同范数随机通道方向）；比较 FP32 代理与 BF16 的 $flow$、后续 Block 状态，先验证 neutral；同时取 cross_attn_norm/adapter 的真实配置 receipt。该检查不需要全 Jacobian。

## 3. 定理 B：任意前层 Flow Jacobian 的秩上界与可利用性区别

固定工作点，将整个 post-write state 展开成 $MD$ 维。当前 step 的最后 head/unpatchify 输出每 token $64$ 维，因此：

$$
J_l^{flow}:\mathbb R^{MD}\to\mathbb R^{64M},
\quad \mathrm{rank}(J_l^{flow})\le64M,\quad
\boxed{\dim\ker J_l^{flow}\ge(5120-64)M.}
$$

以 $M=31668$ 代入，输入 $162,140,160$ 标量、输出 $2,026,752$ 标量；单一 Dense 工作点的局部零度至少 $160,113,408$，即 **98.75% 输入维数**。

**本结论的严格解释**：
- 是线性代数的**输出维数上界**，不是已测的数值 Jacobian singular spectrum；不是“每个 token 有固定 5056 个独立可删除通道”。
- 早层的 $J_l$ 跨所有 token 耦合，$\ker J_l$ 一般不是 $\ker(J_{\mathrm{token}})\otimes\mathbb R^M$ 这样的逐 token 可分离结构。
- 不同 $s$ 的核会旋转：部署固定基需要研究 $\cap_s\ker J_{l,s}$ 或容忍一定目标风险的近零方向，而不是把各自 98.75% 算进共同免算容量。
- 线性化的零方向可能存在二阶有限响应；原生 BF16/Q2 还可能跳变；即使作用空间稳定，学生从合法 pre-W1 输入低成本生成表示也需要独立证明。

**可证伪候选 B-ROBUST-KERNEL**：在已经严格剔除公共平移规范方向之后，是否还存在跨独立 source×seed×step 的稳定低风险共享表示，能以低成本后训练 FFN 表达？最强竞争解释是各状态零空间大但互相不对齐，共同有效维数仍高。

## 4. 定理 C：后缀 Attention 导数的空间—时间结构

令一 head $O=\mathrm{softmax}(QK^\top/\sqrt d)\,V W_O$，记 $A=\mathrm{softmax}(S)$：

$$
dO=(dA\,V+A\,dV)W_O,\qquad
dA_i=(\mathrm{diag}(a_i)-a_ia_i^\top)dS_i,\quad
dS=(dQ\,K^\top+Q\,dK^\top)/\sqrt d.
$$

**Cross-Attention**：文本/图像 $K_c,V_c$ 在同一个 frozen step/context 下固定，$dK_c=dV_c=0$；视频 $Q_i$ 来自各自 token 的 $LN3$，因此其**视频 token 到输出 token**的局部 Jacobian 是块对角的（每 token 的文本/图像注意力本身不能直接改变另一个视频 token）。这不意味着文本条件语义不重要，也不排除后续 self-attention 传播；如果其他分支重算 context 或读取视频 KV，结论条件改变。

**Self-Attention**：$dQ,dK,dV$ 都来自视频 state，$dO_i$ 中有 $\sum_j a_{ij}dV_j W_O$，还有由 $dK_j$ 影响其它 query 的 softmax 项，形成真正跨 token 耦合。其支持范围取决于当前实现的 attention kernel/window、RoPE、mask、分组和缓存。

**假设 H-ATTN-MEDIATION**：某些 W2/FFN 误差方向的本地能量不高，但对后续 Self-Attention 的 Q/K logits 或 V 传输有较大影响，从而改变时空运动。竞争解释是后续 FFN、scheduler 自身更主要地放大状态误差。最小判别需保存相同状态的中介响应（Q/K/V 或 attention output）与完整后缀有限差，不把单次 patch 解释成“运动神经元”。

## 5. 定理 D：四步后缀与有限时域终端可观测性

对确定性的当前四步采样器写抽象状态转移：

$$
z_{t+1}=\mathcal S_t(z_t,\,f_t(z_t,c)),\quad t=0,1,2,3.
$$

这里 $f_t$ 包含该 step 全部 40 Blocks、final head 和 unpatchify，$\mathcal S_t$ 是真实 scheduler 更新。若 scheduler 依赖历史 noise/velocity、RNG/缓存或外部状态，就将这些并入扩充 $z_t$，不能默认一阶 Markov。

在某条固定 Dense 轨迹上线性化：

$$
\begin{aligned}
A_t &= D_z\mathcal S_t+D_f\mathcal S_t\,D_z f_t,\\
B_t &= D_f\mathcal S_t,\\
\delta z_{t+1}&\simeq A_t\,\delta z_t+B_t\,\delta f_t.
\end{aligned}
$$

令 $J^{flow}_{t,l}$ 为该 step 第 $l$ FFN 的 **post-write** 到当前 flow 的 Jacobian；$G_{t,l}$ 是 pre-gate FFN 扰动到 post-write 的 gate。其到最终 latent 的一阶传播：

$$
\boxed{
H_{t,l}=
\left(A_3 A_2\cdots A_{t+1}\right)
B_tJ^{flow}_{t,l}G_{t,l},
}
$$

$t=3$ 时空乘积为 $I$。若最终目标是解码视频/动作或预定义行为 $b(z_4)$，把 $D b$ 合并进 $H_{t,l}$，或采用目标正半定度量 $Q$；不能把原始 latent MSE 无理由等同 RAFT、主体动作或 prompt 语义。

$$
\boxed{
G^{terminal}_{t,l}=H_{t,l}^\top QH_{t,l},\quad
\mathcal R_{t,l}(e)=\mathbb E_s\|H_{t,l,s}e_{t,l,s}\|_Q^2.
}
$$

这是给定 trajectory/output 的**局部终端敏感度**；可视为有限时域微分可观测性的一种读出相关构造，但**不自动满足控制理论论文的可达性、系统形式及全局误差界条件**。

### 5.1 联合有符号风险不是逐层风险之和

把所有 160 个执行位置按实际次序统一为 $i$，Dense 工作点局部线性化：

$$
\delta z_4\simeq\sum_i H_i e_i,\qquad
\mathbb E\|\delta z_4\|_Q^2
=
\sum_i\mathbb E\|H_ie_i\|_Q^2+
2\sum_{i<j}\mathbb E\langle H_ie_i,H_je_j\rangle_Q.
$$

**直接后果**：如果跨位置误差在终端空间存在稳定的有符号配对结构，独立最小化每层 SSE 就未必最优；若该结构在 Student rollout 或新来源上漂移，依赖相消的方案反而可能脆弱。因此“跨层联合补偿”目前只是**竞争机制假说**，不是默认要训练的算法。

[仓库 E0 Whole-Flow](../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) 两来源同状态有限反事实的正负交叉项表明，已发生过“已有状态偏差的 Dense 传播”与“当前候选净作用”的有符号增强/抵消。**E0 这些项并非不同 FFN 位置 $H_i e_i$ 的直接实测**；它们是提出跨位置机制的经验入口，不能在论文中当作 B-JOINT 已被证实。

### 5.2 有限混合模型望远镜分解：无需 Jacobian 的精确身份

将拟替换的 $K$ 个 FFN 位置按固定顺序排列。$Y^{(k)}$ 是前 $k$ 个采用 Student、其余 Dense 的**完整确定性同初始状态**输出，则：

$$
\boxed{Y^S-Y^D=\sum_{k=1}^K\big(Y^{(k)}-Y^{(k-1)}\big).}
$$

此式对非线性、量化和真实 scheduler 都严格成立（同一状态空间/比较对象、每次完全可复现）。但每项取值依赖替换顺序，且更换早层会改变后层看到的 state；它是**路径分解**而非顺序无关的神经元因果贡献。完整逐站 $K=160$ 的代价太高，先以按 step/层段 grouped-hybrid 反事实作判别，必要时重复两种预冻结顺序。

## 6. 论文先例：真实可迁移的定理、工具与边界

| 一手论文 | 确切原始机制 | Wan 可用的数学/工程工具 | 必须保留的迁移条件 |
|---|---|---|---|
| [CoBRAS — Otto et al. 2023](https://arxiv.org/html/2207.14387v4) | 平衡状态与 adjoint 梯度协方差，快照交叉 SVD 构造斜投影/模型降阶；解决低方差高敏问题 | 同预算固定边际协方差强比较器，snapshot-SVD 比形成完整 $MD\times MD$ Gram 便宜 | 其边际平衡不等于 M-B 同 source $G_s,C_s$ 联合风险；原文性能/界不适用于 BF16 Wan |
| [Empirical differential Gramians — Kawano & Scherpen, Automatica 2021](https://doi.org/10.1016/j.automatica.2021.109534) | 固定非线性系统轨迹的变分可达/可观测 Gramian 与经验近似，作者原定理涉及常数输入向量场 | 定义 trajectory-wise 对应的 $H^\top QH$，用有限扰动/JVP/VJP 构造 sketch | 只继承方法学，不声称该论文在 Wan 的结构/扰动下给出全局平衡误差界 |
| [2ndMatch — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html) | 随机投影匹配 teacher/student 的 $J^\top J$ 局部响应度量，后训练剪枝扩散模型 | 多层 VJP/JVP sketch，局部学生敏感度一致性损失 | $J^\top J$ 不是 Hessian 全部，也丢失 $Je$ 的输出符号：例如 $J$ 与 $-J$ 具有同一个 $J^\top J$ |
| [OBS-Diff — arXiv 2510.06751](https://arxiv.org/abs/2510.06751) | Diffusion pruning 的 timestep-aware 二阶近似与 module-packages 顺序校准、早期 step 较大权重 | 现有 CAL24/CAL-v2 的分 step 来源配额、实验预算与模块化收集 | 其对其它模型的 log-decrease schedule 不是本 Wan4-step 的最优权重定理；实际测终端敏感度再竞争 |
| [SVD-Cache — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html) | 低能残差可能变化快、主子空间较可预测；分别 forecast/reuse | 在变差能量之外加入 time predictability，避免把低能视作低功能 | 作者是缓存方法；不能把其视频加速或残差策略当作 Wan 冻结 FFN 原生最优 |
| [Wan2.2 官方源码](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py) | Pre-LN / Cross-LN / residual / final-LN 实际拓扑 | 定理 A 的结构前提 | FFNDS 当前部署用的是其 LightX2V 分支，须检查 runtime SHA / adapter / dtype |
| [LightX2V 上游源码](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/networks/wan/infer/transformer_infer.py) | Q/K/V、Cross-Attention、FFN、post-process 具体位置及条件分支 | B0/B1 producer 与 hook 放置的可审查入口 | 上游 $main$ 可能不同于冻结 runtime，不能只用上游当前代码替代本地审计 |

**原创性审计**：低方差高敏/加权谱截断与斜投影（CoBRAS）、梯度/Jacobian metric 匹配（2ndMatch）、时间加权二阶剪枝（OBS-Diff）均有强先例。新的可证伪对象应是**固定 Wan 原生计算流中的跨状态共同规范结构、配对/有符号后缀相互作用，以及可编译的低成本函数**。仅改论文名词不构成新方法；真正方法贡献仍需通过匹配的 A0、边际/paired、direct student、原生成本与完整视频结果支持。

## 7. 可以立刻做什么：从静态证明到最小判别

| 阶段/优先级 | 输入与产物 | 主要预测/竞争解释 | 判定与不越界条款 |
|---|---|---|---|
| **P0 / 静态源码身份** | 冻结 runtime 的 $norm1,norm2,norm3$ 类型、$cross_attn_norm$、adapter/before_proj、Q2、sensitive dtype、head 带 SHA 的简明审计 | $norm3$ 若为 Identity/其它直读会打断 gauge | 静态 SRC，不宣称原生数值 |
| **P0 / 纯数学与 CPU 玩具** | 小维 FP32 Block 的 $B(X+E)−B(X)−E$、$H(X+E)−H(X)$、finite-difference $J E$ | gauge 一条严格恒等、径向不变量只限末层 | ALG/CPU；不需要伪装 Wan checkpoint 跑通 |
| **P1 / B0/B1 完整原生** | 现有 B 计划的 post-write u/head/flow neutral producer；在固定 source/step 做同状态 gauge/radial/control 有限差 | 量化舍入是否主导与扰动大小有关 | 用实际原 dtype/影响闭包，按 source 为独立单位 |
| **P1 / 复用 E0/UB1** | 保存的 Dense/Student $P/N/cross$、source/seed 动作分类、系统数据字典；无新增生成 | Signed 传播在 source 间改变 vs 固定权重解释；识别最有信息量的站点 | E0 只证明既有状态响应×候选作用，不能重命名为跨位置响应 |
| **P2 / multi-layer sketch** | 选定已暴露开发来源，固定 readout Q，整步或局部后缀 $v^\top f$；一次 VJP 提取多个层梯度 $J_l^\top v$ | 去掉 gauge 后的主要方向跨来源是否稳定；能否打败边际/PCA | $E[vv^\top]=Q$ 时 $E[(J^\top v)(J^\top v)^\top]=J^\top QJ$；VJP 需要可微代理，Q2/BF16另作真实 finite 对照 |
| **P2 / grouped-hybrid finite** | 两段/两位置 Dense/Student 混合全后缀、预冻结两种顺序；目标为合法最终 latent/flow/行为读出 | 非线性交互是否与一阶预测一致；是否存在稳定方向符号 | 混合必须在同 seed/conditioning/fresh dense 起点；报告顺序依赖与真实完整成本 |
| **P3 / 低成本学生和全自由生成** | A0 width6016 vs M-B student、PCA/marginal/paired + group-structured candidate，同输入/训练预算 | oracle 领先是否能被学生且硬件兑现；视频功能是否保留 | 完整 40×4 无 Dense 补算、主 GEMM/QDQ/layout/wall/VRAM、独立 source、视频全指标；通过后才能宣称 SYS/VIDEO |

### 7.1 关于导数求值的工程特别说明

- LightX2V 常用 $torch.no_grad()$ 推理入口；若做 VJP/JVP，必须有**独立的诊断入口**启用梯度、限定 checkpointing/显存与 source，并确保 Q2 的导数约定与实际 HiF4 非连续行为分别记账。
- 梯度的 gauge 检查是：每个 token $p$ 的 $\sum_c \partial \phi/\partial X_{p,c}\approx0$；在具有该性质的**连续代理**下成立，并不承诺 BF16 autograd 与实数定理 bitwise 一致。
- 多 step 残差方向可用结构化随机探针和 adjoint 反向传播计算；每次终端读出 $Q$ 必须先定义。在尝试昂贵视频语义梯度前优先使用 latent/Flow 向量读出。
- 累计样本以 image/source cluster 为独立单元，token、step、block、seed 都属于嵌套样本；已暴露 CAL24/consumer20 是开发集，不当作独立确认。
- 每一条方法的在线成本要核算：合法 pre-W1 观测、学生网络、W1/GELU/Q2/W2 真正执行条目、kernel launch、QDQ、packed FP4 vs fake-QDQ、驻留/中间写回、完整视频。

## 8. 研究决策与和原评议的关系

**与 [原独立评议](INDEPENDENT_REVIEW.md) 的一致部分**：M-B-JOINT 有数学价值，但 B0/B1/B2/B3 尚无原生可部署证明；同成本 A0 6016 是强基线；末层 65 统计不是最低信息量/省算定理；B-H 不应独自承担全模型加速主张。

**本轮对研究自由度的实质拓展**：
- B-GAUGE：从 block39 的状态局部“归一化作用”推广到所有后续 Pre-LN Block 的公共精确实数不变方向；
- B-OBS：从单层 $J_s^\top J_s$ 转向具有 scheduler 和后续步的轨迹条件终端读出；
- B-JOINT：从同一 source 上 $G_s,C_s$ 的配对扩展到多个 site 的有符号输出交叉项，结合有限混合模型分解。
- 原 v2 的固定投影仍是成熟、可比较的模型类，而非必须预设为唯一构造；依次比较**自然功能保留、前置可预测性、物理可执行性**。

**有界决策**：维持 **CONDITIONAL_CONTINUE**，不把“98.75% Jacobian 零维度”包装为可删通道数量，不把静态 gauge 证明包装为真实 speedup，不把 E0 P/N cross 当成跨层压缩因果证明。M-A 可用稀疏方法、M-C 编码计算和现有 M-B B0/B1 继续并行；本目录是独立的方案评议，不修改 ACTIVE/QUEUE/G14 事实登记或自动触发 GPU 任务。

## 9. 证据追踪与最短阅读链

1. [M-B 首份独立评议](INDEPENDENT_REVIEW.md)，[B v2 方法快照](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)，[末层代码/SRC 审计](../../method_discussion/discoveries/E0/b-final-block-head-source-audit-20261028.md)。
2. [M-A0 6016 local recovery](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)：四位置、16 train/已暴露 heldout8 的 native-Q2 局部恢复，**非**完整视频/整端速度。
3. [E0 Whole Flow](../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[G13 函数定位](../../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)：区分 Dense 状态传播、候选当前作用、前后量化、后缀与功能身份。
4. [.agents/skills/mechanism-research/SKILL.md](../../../.agents/skills/mechanism-research/SKILL.md)、[problem-reformulation](../../../.agents/skills/problem-reformulation/SKILL.md)、[research-expansion](../../../.agents/skills/research-expansion/SKILL.md)、[ffnds-experiment-evidence](../../../.agents/skills/ffnds-experiment-evidence/SKILL.md)、[web-research-discovery](../../../.agents/skills/web-research-discovery/SKILL.md)。
5. 外部一手主要来源：Wan2.2/LightX2V 源码、CoBRAS 2023、Kawano–Scherpen 2021、2ndMatch CVPR 2026、OBS-Diff、SVD-Cache CVPR 2026（精确 URL 见第6节）；仅核对了与本轮命题相关的方法/官方结构，**没有复现作者代码或将其质量/速度指标移植到 Wan**。

> 附注：以上含有独立数学演绎、源码分支适用条件和后续可证伪实验。新增算法名称只是讨论标签，不代表已被其它 Agent 支持或成为正式登记的 FFNDS 方法。
