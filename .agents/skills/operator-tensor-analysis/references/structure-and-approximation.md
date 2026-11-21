# 结构与近似

用本参考区分不同结构约束，并把近似连接到实际输入与输出目标。

## 内容

- 算子族 单个算子与输入分布
- 目标相关的近似误差
- 不同秩回答不同问题
- 方法来源与可迁移经验

## 算子族 单个算子与输入分布

\[
J_\rho=\sum_{r=1}^{R}c_r(\rho)B_r
\]

这个表达约束算子随状态变化的自由度。它没有单独约束各个基算子 B_r 的矩阵秩，也没有说明输入增量分布是否集中在一个低维子空间。分别判断“算子族维数”“单个算子结构”和“实际输入方向”。

例如 J_ρ=c(ρ)B 且 B 为一般稠密满秩矩阵，状态方向只有一个基函数，作用于一般输入时仍可能需要稠密矩阵乘。



## 目标相关的近似误差

近似研究应优先考虑实际输入增量和误差的下游作用。令状态为 ρ，局部误差为 e_ρ=(J̄_ρ−A_col)Δx_ρ，下游在参考轨迹上的线性化为 L_ρ，最终输出的权重为 M≽0。定义：

\[
G_\rho=L_\rho^\top M L_\rho,
\qquad
\mathcal L(A_{\rm col})=
\mathbb E_\rho\!\left[
e_\rho^\top G_\rho e_\rho
\right].
\]

这是固定参考线性化下的目标代理；它并不把任意大的局部误差精确映射成真实终端损失。多个位置的误差同时作用时，需联合堆叠并保留相关交叉项。

固定参考轨迹及其分布，使 G_ρ 与待优化的 A_col 无关，并要求目标中的二次与交叉期望有限时，该二次目标对算子分量的曲率为：

\[
\frac{\partial^2\mathcal L}
{\partial A^a{}_i\,\partial A^b{}_j}
=
2\,\mathbb E_\rho
\left[G_{ab}[\rho]\,\Delta x_\rho^i\Delta x_\rho^j\right].
\]

这是算子空间上的四槽误差形式；它具有自己的对称性与正半定性，不因“四个指标”就成为 Weyl 张量。

对固定的实误差矩阵 K、固定 G≽0 及有限二阶矩 C=E[ΔxΔxᵀ]：

\[
\mathbb E\|K\Delta x\|_G^2
=
\operatorname{tr}(GKC K^\top)
=
\|G^{1/2}KC^{1/2}\|_F^2.
\]

C 是二阶矩；只有在均值为零时才等于协方差。若 K 或 G 随样本变化，应保留联合期望，除非另有足够的独立性或确定性条件。

这给出实际拟合的方向：在真实增量分布和输出敏感度下选择近似。使用普通 Frobenius 误差、各向同性输入或固定下游度量时，应说明这些简化如何对应研究目标。

## 不同秩回答不同问题

矩阵秩描述一个展开的输入输出子空间；多线性秩分别描述各 mode 展开；CP 秩描述最少的可分离外积项；TT 秩描述按指定次序切分各模式后的耦合维数。比较这些概念时应给出索引分组。

一个能跨越矩阵与张量表达的基本例子是：

\[
K_{i_1i_2,j_1j_2}=P_{i_1j_1}Q_{i_2j_2},
\qquad K=P\otimes Q.
\]

若 P、Q 都是 n×n 满秩矩阵，K 的矩阵秩为 n²；但按 (i₁,j₁)、(i₂,j₂) 配对分组，它具有秩 1 的两核 TT 矩阵表示。存储因子只需 2n² 个数，显式矩阵需 n⁴ 个数。

对输入数组 X：

\[
Y_{i_1i_2}
=
\sum_{j_1,j_2}P_{i_1j_1}Q_{i_2j_2}X_{j_1j_2}
=
(PXQ^\top)_{i_1i_2}.
\]

在明确的行优先展平约定下，这与 K vec(X) 一致。通过两次小矩阵乘，稠密算术量从 O(n⁴) 变为 O(n³)。收益来自可分离结构；它没有由 reshape 自动产生。

HOSVD 截断一般不是最佳给定多线性秩逼近；高阶 CP 的最佳固定秩近似还可能不存在。实际近似需检查误差目标、数值稳定性及分组选择，不能直接移用矩阵截断 SVD 的全部最优性结论。参考 [HOSVD 原文](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf)与[de Silva 和 Lim](https://arxiv.org/abs/math/0607647)。



## 方法来源与可迁移经验

- [Tucker 1966](https://three-mode.leidenuniv.nl/pdf/t/tucker1966pmet.pdf)：保留 mode 身份，用核心表示不同因子间的耦合。
- [Harshman 1970 PARAFAC](https://www.psychology.uwo.ca/faculty/harshman/wpppfac0.pdf)：多个模式共享分量；唯一性按原文条件使用。
- [HOSVD 2000](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf)：分别分析模式谱；Property 10 讨论截断的最优性边界。
- [Oseledets 2011 TT](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf)：用核心链表示各切分间的耦合，按给定模式次序讨论 TT 秩。
- [Novikov 等 2015 Tensorizing Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)：行列多索引映射影响 TT 矩阵表示及效率。
- [de Silva 与 Lim](https://arxiv.org/abs/math/0607647)：高阶固定 CP 秩最佳近似可能不存在。

将这些结果用于选择结构假设、拟合目标和收缩方式。参数压缩、准确近似和真实加速分别给出依据。
