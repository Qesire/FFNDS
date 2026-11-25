# 从当前交互出发：两组正权摘要、完整教师与单残差锚点

2026-11-25。起点是 FFNDS `873e1a879b91d544fdae11894d68bbc6acc854d1` 的实际代码，以及用户对六来源 C/A 成片“基本没有差别”的新判断。本文件推进一个具体数学构造；没有执行新模型、CUDA 内核或成片生成。

使用 operator-tensor-analysis 的张量语义、矩与协方差、有限误差和执行成本工具，以及 active-abstraction 的信息充分性、更新闭合与构造性判据。数值核验使用 NumPy/SymPy。合成激活与真实资产、静态尺寸严格区分。

## 1. 新构造保留什么作用

当前 high OUT-ABS 的非 P 行由 selected 输出 B 构造逐位置特征。上轮已证明，同一 head/调用内相同的 B 行会得到相同预测。一个能改变这个信息接口的构造，应直接读取当前 query 与遗漏 K/V 的关系。

只保存每个遗漏块的 K/V 均值，已经能加入部分当前关系；进一步需要保留 K 与 V 在块内如何一起变化。本轮采用一个可执行的最小细分：每块按初始化时 K 在固定方向上的中心化投影分为两组，计算两组各自的有效数量、当前 K 均值和当前 V 均值。无锚调用使用当前分组；带锚更新冻结组标签，但均值始终读取当前 K/V。

这保留每个 query，保留 selected blocks 的逐 key 精确计算，增加当前遗漏交互。它不是在 B 派生的同一个特征族里重新选择 CAL 坐标。

## 2. 将 attention 看成可相加的未归一化响应

固定 layer、head 和当前调用。query 轴 i、key 轴 j、key 通道 α、value 通道 ν。令有效操作数 Q/K 已包含实际 norm/RoPE 等处理，d=128：

\[
w_{ij}=\exp(q_i^\alpha k_j^\alpha/\sqrt d),\qquad
Z_i=\sum_j w_{ij},\qquad
N_i^\nu=\sum_j w_{ij}v_j^\nu,\qquad
A_i^\nu=N_i^\nu/Z_i.
\]

对任何不交的 key 分区，(Z,N) 可以相加。单独的归一化输出 A 则缺少合并各部分所需的质量 Z。实现中使用 LSE 与稳定缩放完成同一归约，不显式计算可能溢出的 Z。

令 S_i 为已有 selected key blocks。它们给出精确 (Z_i^S,N_i^S)，B_i=N_i^S/Z_i^S。其余 key 由下述正权摘要补入。

## 3. 一个有辨识力的例子，以及能够恢复它的构造

令 selected keys 和 values 均为零，所以所有 query 的 B_i=0。遗漏部分只有 k=±a、v=±b 两个对应的 token。令 t=q·a/√d，selected key 数为 n_S，则

\[
A(q)=\frac{2\sinh(t)}{n_S+2\cosh(t)}b.
\]

遗漏部分的 K/V 均值都为零，但真实响应随 q 改变方向和幅度。把 ± 两组分别保留为正权摘要即可精确重建。该例指出具体需要补入的关系：**K/V 的共同变化决定了 query 如何选择不同的 value 响应。**

### 3.1 固定、可执行的分组

对每个原始 block b，令其有效 key 索引集合为 I_b，计算当前均值 μ_b^K。取已归档 P_k 的第一列作为方向 r，归一化后冻结。定义

\[
I_{b,+}=\{j\in I_b:(k_j-\mu_b^K)^\top r\ge0\},\qquad
I_{b,-}=I_b\setminus I_{b,+}.
\]

固定方向使本构造在现有资产下完整可实现；它并非关于最佳分组方向的主张。无锚调用按当前 K 构造分组，标签只在本次调用临时使用。带锚路径在 anchor 确定标签并冻结到 hit，后续仅重新计算各组的当前 K/V 均值，详见 §6。空组跳过；末块按实际有效数量计算，本模型为 52 keys。主选中支持仍以原 block 为单位，某 block 被 selected 时，它的两个摘要均排除，原有逐 key 计算全部保留。

对非空组 c=(b,±)，保存

\[
n_c=|I_c|,\quad \bar k_c=n_c^{-1}\sum_{j\in I_c}k_j,\quad
\bar v_c=n_c^{-1}\sum_{j\in I_c}v_j.
\]

对每个 query，令

\[
\bar z_{ic}=n_c\exp(q_i^\top\bar k_c/\sqrt d),\qquad
Z_i^0=Z_i^S+\sum_{c:\,b(c)\notin S_i}\bar z_{ic},
\]

\[
N_i^0=N_i^S+\sum_{c:\,b(c)\notin S_i}\bar z_{ic}\bar v_c,
\qquad
G_i=N_i^0/Z_i^0.
\]

原始 block 的支持、分组数量与 current K/V 都在需要时已知。该构造不需要未来的原生全行输出。

### 3.2 可以证明的保持关系

1. **每个粗组内 K 完全相同时，G 对该组贡献精确，V 可任意。** 此时每个 key 的指数权相同，数量与 V 均值足够。
2. **正权与凸包保持。** G 是 selected values 与粗组 value 均值的正权平均，因而位于全部实际 V 的凸包内。
3. **value 平移与线性映射保持。** 分组仅依赖 K。V→VL+c 时，G→GL+c。
4. **logit 共同平移保持。** 某 query 的全部 logits 加同一常数时，N、Z 与其摘要同倍缩放，G 不变。全局 key 平移 k→k+a 不改变中心化分组，并产生这样的共同 logit 平移。
5. **质量随分区细化单调逼近。** 由 exp 的凸性，对任意粗组的细分，有

\[
n_c e^{q\bar k_c/\sqrt d}
\le\sum_{u\subset c}n_u e^{q\bar k_u/\sqrt d}
\le\sum_{j\in c}e^{qk_j/\sqrt d}.
\]

所以固定 selected 支持下，单均值→两组→逐 key 的 Z⁰ 单调增加，并始终不超过完整 Z。输出响应的误差由下文的质量与 value 相关项共同控制；质量单调关系只对 Z 作保证。

### 3.3 两组保留的响应有明确协方差含义

在一个 parent block 内，q=0 时的 conditional attention 对 q 的 Jacobian 为

\[
\left.\frac{\partial A_b}{\partial q}\right|_{q=0}
=\frac1{\sqrt d}\operatorname{Cov}_b(V,K).
\]

总协方差可以精确分解为

\[
\operatorname{Cov}_b(V,K)
=\sum_c\omega_c\operatorname{Cov}_c(V,K)
+\sum_c\omega_c(\bar v_c-\bar v_b)(\bar k_c-\bar k_b)^\top.
\]

两组摘要精确保留第二项，即组间共同变化；需要进一步恢复的是组内共同变化和有限 score 波动。这里只用 q=0 导数说明所保存的局部响应；下面的误差式对任意有限 q 成立。

## 4. 剩余误差的精确有限形式

在粗组 c 内写 k_j=bar k_c+δk_j、v_j=bar v_c+δv_j，设 t_ij=q_i·δk_j/√d。组内均匀平均满足 E_c[t]=0、E_c[δv]=0。定义

\[
a_{ic}=\mathbb E_c[e^{t_{ij}}]-1\ge0,
\qquad
c_{ic}=\mathbb E_c[(e^{t_{ij}}-1)\delta v_j].
\]

则完整组贡献为

\[
Z_{ic}=\bar z_{ic}(1+a_{ic}),\qquad
N_{ic}=\bar z_{ic}[(1+a_{ic})\bar v_c+c_{ic}],
\]

而完整 attention 与 G 的误差是

\[
\boxed{
A_i-G_i
=\frac{
\sum_c\bar z_{ic}\{a_{ic}(\bar v_c-G_i)+c_{ic}\}
}{Z_i^0+\sum_c\bar z_{ic}a_{ic}}.
}
\]

求和仅涵盖未 selected 的粗组。第一项是遗漏组内部 score 波动引起的质量变化；第二项是 score 与 value 的相关响应。这与上轮单均值恒等式相同，但分组改变了实际计算底座与剩余项：在 ±a/±b 例中，两组内两个项都为零。

还有一个直接的有限界。若 ε_ic=max_j |t_ij|，则

\[
0\le a_{ic}\le\tfrac12 e^{\epsilon_{ic}}\mathbb E_c[t_{ij}^2],\qquad
\|c_{ic}\|\le(e^{\epsilon_{ic}}-1)\mathbb E_c\|\delta v_j\|.
\]

代入上式并用三角不等式即可得到输出误差上界。本次合成数值验证通过该界。这个界的价值在于明确分组希望减少的对象：组内 score 变化与 value 偏离共同决定剩余响应。其数值可能保守，不能作为现有视频质量分数。

## 5. P64 教师如何观测剩余质量与响应

令

\[
a_i=Z_i/Z_i^0-1,\qquad c_i=(N_i-N_i^0)/Z_i^0.
\]

则

\[
A_i=\frac{G_i+c_i}{1+a_i},\qquad a_i\ge0.
\]

对任意预测 â、ĉ，有精确误差式

\[
\widehat A_i-A_i
=\frac{(\widehat c_i-c_i)-A_i(\widehat a_i-a_i)}{1+\widehat a_i}.
\]

这解释了质量与 value 响应为何应在同一归一化关系下考虑。只有原生输出 A 的 P 教师时，a 与 c 不能各自识别：固定 A 后，任取 a 都可令 c=(1+a)A-G。若同时取得原生 LSE，便可以用相对 LSE 确定 a，从而确定 c。

旧 FIN-C1 已经使用过 P 上的质量标签，因此“增加质量监督”本身不是本轮的新变化。本轮变化是先用当前两组 K/V 交互形成新的 G 与 Z⁰，再把剩余作用交给 P 教师。旧 FIN-C1 的部分质量标签效应弱，也正说明仅更换标签不足以替代当前信息入口的改变。

### 5.1 数值有界的坐标

Z/Z⁰ 很大时，a、c 的数值尺度会变大。可以改用

\[
g_i=Z_i^0/Z_i=\exp(\ell_i^0-\ell_i)\in(0,1],
\qquad
h_i=A_i-v_* -g_i(G_i-v_*),
\]

其中 v_* 是当前全部 V 的均值。于是

\[
\boxed{A_i=v_*+g_i(G_i-v_*)+h_i.}
\]

无需除以很小的遗漏率。h 对所有 values 的共同平移不变。若预测 ĝ 限制在 [0,1]，其误差精确为

\[
\widehat A_i-A_i=(\widehat g_i-g_i)(G_i-v_*)+(\widehat h_i-h_i).
\]

这套坐标可以作为质量/响应归属的观测语言，或后续结构化恢复的接口。它不是以新的符号重新宣称 FIN-C1 的质量监督有效。

### 5.2 实际 API 绑定

实际 `attention_online_structured.py` 的 P64 教师调用 `F.scaled_dot_product_attention`，当前代码只取得输出。相同文件的 `native_flash` 已调用 ATen `_scaled_dot_product_flash_attention` 并返回输出与 LSE，完整锚点正在使用该辅助函数。

若扩展 P64 教师，需要切换/封装这个接口，并核验 q64 shape、backend、有效长度和 LSE 约定。当前环境没有 torch，本轮未执行该核验，也未测量其开销。新增 LSE 净输出是 64×40×4=10 KiB；它的来源是已需计算的归一化统计，但 API 和执行成本仍应按实际实现确认。

## 6. 与已成功的 FFNDS 状态组织连接

新的 G 可以用同一种定义服务首步和后续步。保持每个位置的完整状态时，最直接的构造是：

\[
E_a=A_a-G_a,\qquad \widehat A_s=G_s+E_a+\widehat{(E_s-E_a)}.
\]

原生锚点计算 A_a，并用相同 coarse+selected 算子计算 G_a；缓存全位置 E_a 和每个 key/head 的一位分组标签。推荐以简单 uint8 保存标签，再考虑 bitpack 的额外开销。主 selected 支持与分组标签均冻结到 hit，当前 Q/K/V 及各组均值继续变化。这样有限创新不会额外混入离散重分组的变化。若每次重分组，下面的恒等式仍成立，但其 ΔE 同时包含分区变化；该选择具有另一种恢复负担。

hit 时，P64 监督的是

\[
R_{s,P}=A_{s,P}-G_{s,P}-E_{a,P}.
\]

以当前特征 ψ_s 拟合这一剩余创新，便无需存储 ρ_a 或 anchor 特征 φ_a。没有锚点时取 E_a=0，P64 监督 A_{s,P}−G_{s,P}；这仍是一个有定义的当前构造，但恢复负担与带锚更新不同。

### 一个具体的 32 列当前特征接口

为了把构造推进到可实现的候选，可使用已有 P_k/P_v 的前 15 列，定义

\[
\psi_i=
[1,\ q_iP_{k,15}/\sqrt d,\ (G_i-B_i)P_{v,15}/s_v,\ Z_i^S/Z_i^0]
\in\mathbb R^{32},
\]

其中 s_v 是当前 head 的中心化 V 元素 RMS。s_v=0 时所有 values 相同，G 已精确，此分支直接返回 G，不叠加历史 E_a，也不进行剩余拟合；实际实现给非零尺度的数值归一化指定浮点下限。

固定现有 λ=0.1、P64，以 R_P/s_v 为标签做同样大小的 ridge：

\[
W=(\Psi_P^\top\Psi_P+64\lambda I)^{-1}\Psi_P^\top(R_P/s_v),
\qquad \widehat A=G+E_a+s_v\Psi W.
\]

P 行最后覆盖原生教师。求监督标签时保留未覆盖的 G，避免把标签人为置零。这一候选的区别在于当前 q、粗细响应差和 selected 质量共同进入位置特征；它改变了当前 OUT-D 的特征族与底座，属于新方法构造。

在实数域和精确缓存下，一个有锚点的精确作用条件是 E_s=E_a：此时创新为零，底座已精确，ridge 返回零修正。实际 BF16 缓存另有舍入误差。另一个精确条件是当前每个粗组内 K 相同，此时当前 G=A；若 anchor 也满足此条件，则 E_a=0，带锚候选也直接精确。对一般 R，若 R=ΨΘ+ε，岭回归误差由表示剩余 ε、P 上的可观测性及正则偏差共同决定；这些需要真实模型验证。

上述 G 的 value 线性等变证明针对底座本身。这个具体恢复接口使用固定投影和 ridge，任意 value 换基的完整等变性需要连同投影与正则度量一起变换。共同 value 平移则由中心化响应差保持。

## 7. 真实形状的预算

N=31668，H=40，d=128，block128，共248个block，selected31个，40层。以下是每次调用的新增算术量/净载荷，非计时：

| 项 | 静态结果 |
|---|---:|
| 两组×217遗漏块的 QK/PV 成对项，相对完整 attention | 1.37047% |
| 全部两组×248块先算后mask，相同口径 | 1.56625% |
| 当前层两组 K/V 均值，FP32 | 19.375 MiB |
| 两组有效数量，int32 | 0.07568 MiB |
| 若物化全部 N×H×496 coarse scores，FP32 | 2.34057 GiB |
| 每层一个分组方向投影 | 162,140,160 MAC |
| P64 原生 LSE 的额外净输出，FP32 | 10 KiB |

归约、分组、mask、稳定合并及输出工作区需要计入实际执行；coarse scores 适合流式归约。当前已有 selected 输出、LSE 与 block support 可以复用，重算 anchor 的 G 仍要支付新增 coarse 计算。

单 residual 锚点缓存 E_a 的 BF16 净载荷为 12.08038 GiB（40层）。当前 low 缓存的 ρ_a 为 0.18876 GiB、φ_a 为 2.92572 GiB，这两类状态合计 3.11447 GiB。推荐的冻结分组还需 N×H×40 个二值标签：简单 uint8 为 **48.32153 MiB**，bitpack 净载荷为 6.04019 MiB，后者另有打包/读取成本。

采用简单 uint8 时，E_a 与标签合计 **12.12757 GiB**，相对当前 u_a+ρ_a+φ_a 净省 **3.06728 GiB**。这来自改变底座和使用当前特征，具有新的误差与计算成本；不是现有算法的位级等价压缩，也不能直接从已测峰值中扣出一个新峰值结论。无锚调用只需临时分组标签，不保留这项跨步状态。

## 8. 数学核验与证据范围

`verify_two_group_attention.py` 运行以下核验并生成 `two_group_attention_verified.json`：

在仓库 `results/joint39` 发布位置运行：

```bash
python verify_two_group_attention.py --replay-zip ../replay_evidence.zip
```

已有解包资产时也可使用 `--basis-dir /path/to/plan_assets/bases`，只读取原始 P_k.npy/P_v.npy，无需复制大张量。

- SymPy：未归一化提升、耦合误差、有界坐标误差、锚点恒等式。
- NumPy：128维 ±K/V 例。使用真实归档 P_k/P_v 方向、明确合成的 activations。t=±3 时真实响应为 ±0.135253；单均值为0；两组摘要最大绝对误差 1.39e−17。
- NumPy：一般合成 K/V，包含空组和末块不满。有限误差恒等式残差 6.11e−16；Jensen与细化关系、value平移/线性映射、凸包和总协方差分解均通过。另以冻结 anchor 分组和变化的当前 K/V 验证有限误差式。
- 实际维度：上述工作量、缓存与临时状态精确整数预算。

公开归档资产有 CAL 变换和见证汇总，没有本构造所需的真实 full Q/K/V；因此这里没有报告新的 Wan 恢复误差或质量改善。

## 9. 机制近邻与当前贡献位置

**ClusterAttention（Nordenram、Dittmann，arXiv:2608.26965）是更直接的近邻。** 已取得的原始论文摘要明确描述：当前聚类、对 excluded clusters 用 centroid 补偿，以及在 Wan2.1-14B T2V 上的测试。因此 selected 精算加遗漏 centroid 补偿应作为已有构造引用。直接 arXiv 全文访问本轮受限；不以镜像中的推导细节代替完整原文核验。

Scatterbrain（NeurIPS 2021）在未归一化 attention 中组合低秩全局近似与 sparse 修正，selected 支持上计算真实权重减低秩近似，避免双计数；其 §4.2 和 Appendix C 说明输出与归一化的构造。它保留当前全局交互的原则可作为机制参照。

本轮形成的具体研究对象是：在当前既有 block 支持上，以两组正权摘要保留 K/V 关联，再用全位置 residual 锚点和 P64 完整教师恢复有限创新。固定方向二分与 centroid 补偿本身应放在这些既有方法的范围中比较；值得继续论证的差别是当前信息入口、历史状态、教师观测和完整调用周期怎样共同工作。

原始资料：

- [ClusterAttention](https://arxiv.org/abs/2608.26965)，原始摘要已取得；全文的版本细节仍待核对。
- [Scatterbrain 原论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/9185f3ec501c674c7c788464a36e7fb3-Paper.pdf)，§4.2、Appendix C。
- [实际在线 attention](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)。
- [实际 FIN-C1 拟合](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/attention_panel_fit.py)。
