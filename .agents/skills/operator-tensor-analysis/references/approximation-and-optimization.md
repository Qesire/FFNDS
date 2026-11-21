# 近似、目标误差与优化

把结构表示转成可拟合、可验证的算子。这里默认实坐标；先固定误差所作用的输入分布、输出度量、可用状态和计算预算。分解的代数秩、拟合自由度、优化难度与执行代价分别判断。

## 目录

- A1 由实际目标建立算子拟合
- A2 双侧加权低秩与奇异度量
- A3 Taylor 余项与可用工作域
- A4 下游传播、伴随与目标误差
- A5 Newton、Gauss–Newton 与 Hessian 作用
- A6 Fisher、自然梯度与参数化
- A7 割线、Broyden、BFGS 与 SR1
- A8 凸分析、对偶与最优性证书
- A9 近端算子与分裂算法
- A10 信赖域与模型失配
- A11 POD、插值与非线性降阶
- A12 DMD、Koopman 与动态拟合
- A13 统计与计算误差的联合验收
- 来源

## A1 由实际目标建立算子拟合

对样本状态 $\rho$，输入增量为 $x_\rho$，目标变化为 $y_\rho$，下游权重为 $G_\rho\succeq0$。用固定算子 $A$ 最小化

\[
\mathcal L(A)=\mathbb E_\rho[(Ax_\rho-y_\rho)^\top G_\rho(Ax_\rho-y_\rho)].
\]

保持 $G_\rho$ 与 $A$ 无关，且以下二次项与交叉项的期望有限时，梯度与正规方程为

\[
\nabla_A\mathcal L=2\mathbb E[G_\rho(Ax_\rho-y_\rho)x_\rho^\top],
\quad
\mathbb E[G_\rho A x_\rho x_\rho^\top]=\mathbb E[G_\rho y_\rho x_\rho^\top].
\]

对应四槽曲率 $\mathcal H_{ai,bj}=2\mathbb E[(G_\rho)_{ab}(x_\rho)_i(x_\rho)_j]$。它可通过批量作用求解，未必需要物化。在 $G_\rho$ 随输入改变且相关时，一般不能分离成 $\mathbb E[G_\rho]\otimes\mathbb E[x_\rho x_\rho^\top]$。

给 $A$ 添加低秩、稀疏、对称、TT、共享参数或硬件结构约束后，求解方法会改变。先保留无约束／高预算解作为误差基线，再比较结构约束损失。若 $G$ 本身随拟合参数改变，以上梯度还需包含其导数。

## A2 双侧加权低秩与奇异度量

固定目标算子 $J$、$G\succeq0$、$C=\mathbb E[xx^\top]\succeq0$：

\[
\mathbb E\|(J-A)x\|_G^2
=\|G^{1/2}(J-A)C^{1/2}\|_F^2.
\]

若 $G,C\succ0$，令 $B=G^{1/2}JC^{1/2}$，取其截断 SVD $B_r$，则一个最优 rank-$\le r$ 解是

\[
A_r=G^{-1/2}B_rC^{-1/2},
\quad \mathcal L(A_r)=\sum_{i>r}\sigma_i(B)^2.
\]

如果度量奇异，令 $P=G^{1/2},Q=C^{1/2}$，使用

\[
A_r=P^+ B_r Q^+.
\]

因为 $B=PJQ$ 的非零奇异空间落在 $P,Q$ 的相应支撑中，$PA_rQ=B_r$，且 $\operatorname{rank}A_r\le r$。这仍达到加权目标的最小尾谱误差。满足 $PNQ=0$ 的分量不被目标观测，但添加任意这样的 $N$ 可能破坏 rank 约束；“不可辨识”不意味着可以任意改变实现。退化或重复奇异值时也需讨论非唯一性。

**反例。** $J=\operatorname{diag}(4,1)$、$C=\operatorname{diag}(1,100)$、$G=I$。普通 rank-1 SVD 保留第一方向，目标误差为 100；加权 SVD 保留第二方向，误差为 16。矩阵元素上的最佳近似不一定是输入分布上的最佳近似。

这是固定双侧度量的闭式解。样本相关的 $G_\rho$、状态变化的 $J_\rho$ 或非 Frobenius 目标通常回到 A1 的联合问题，不能继续套单次 SVD。

## A3 Taylor 余项与可用工作域

若路径上 $DF$ 为 $L$-Lipschitz，则

\[
F(x+h)=F(x)+DF(x)[h]+r,
\qquad \|r\|\le\tfrac L2\|h\|^2.
\]

对任意候选 $A$，误差再分成 $(DF(x)-A)h+r$。若只知道局部或经验 Lipschitz 估计，不把它称为全工作域的严格上界。方向限制可给更紧的结论：只需沿实际路径控制 $D^2F(x+th)[h,h]$。

二阶模型在 $D^2F$ 为 $M$-Lipschitz 时余项至多 $M\|h\|^3/6$。ReLU、max、排序和离散门控需采用分段分析、方向导数或广义导数；跨分支的有限变化不能由某一侧的普通 Hessian 覆盖。精确割线恒等式与低成本近似的区别见 `derivatives-and-secants.md`。

**选择近似阶数。** 先比较模型项的实际方向作用和计算成本，检查增加高阶项是否降低目标误差；高阶导数存在不代表它可稳定估计，也不代表其完整张量需要被存储。

## A4 下游传播、伴随与目标误差

在参考轨迹上线性化，局部注入误差 $e_\ell$ 对终点产生

\[
\delta z_L\approx\sum_{\ell=0}^{L-1}\Phi_{L,\ell+1}e_\ell,
\qquad \Phi_{L,\ell+1}=J_{L-1}\cdots J_{\ell+1}.
\]

平方目标包含不同位置误差之间的交叉项。传播因子固定时，零交叉二阶矩（例如误差彼此独立且均值为零）足以将期望分解为逐层和；单独“独立”或单独“零均值”都不充分。若传播因子也随样本改变，须核对传播后误差的联合交叉项，不能只检查原始 $e_\ell$。共享缓存偏差可能同向累积。

**伴随／dual-weighted residual。** 对线性方程 $Ax=b$、近似 $\widehat x$、目标 $q(x)=c^\top x$，解 $A^\top z=c$ 后精确有

\[
q(x)-q(\widehat x)=z^\top(b-A\widehat x).
\]

这把残差按目标敏感度加权；用一次伴随求解可评估许多局部残差。非线性问题使用路径平均导数或局部伴随并保留余项，不能把局部恒等式当成任意大误差的全局保证。[AO5]

## A5 Newton、Gauss–Newton 与 Hessian 作用

平方残差目标 $\ell(\theta)=\tfrac12\sum_a r_a(\theta)^2$ 有

\[
\nabla\ell=J^\top r,
\qquad
\nabla^2\ell=J^\top J+\sum_a r_a\nabla^2r_a.
\]

Gauss–Newton 丢弃第二项；在零残差点与精确 Hessian 相同，远离该区域时可能漏掉重要曲率。$J^\top J$ 正半定但可能奇异。Levenberg–Marquardt 求 $(J^\top J+\lambda I)p=-J^\top r$、$\lambda>0$；正阻尼控制局部模型和病态方向。

对一般复合损失 $\ell=\ell_{\rm out}(F(\theta))$：

\[
H=J^\top H_{\rm out}J+\sum_a(\partial_a\ell_{\rm out})D^2F_a.
\]

当输出损失凸时，第一项为 PSD 的 generalized Gauss–Newton。其作用 $v\mapsto J^\top[H_{\rm out}(Jv)]$ 依次用 JVP、输出曲率作用、VJP 实现。完整 HVP 则还要保留第二项；Newton–CG 在 Hessian 不定时需处理负曲率。

## A6 Fisher、自然梯度与参数化

对概率模型 $p_\theta(y|x)$，固定输入分布 $q$，并要求 score 的二阶矩有限，条件模型的 Fisher 为

\[
F(\theta)=\mathbb E_{x\sim q,\,y\sim p_\theta(\cdot|x)}[s_\theta s_\theta^\top],
\qquad s_\theta=\nabla_\theta\log p_\theta(y|x).
\]

自然梯度方向解 $Fp=-\nabla\ell$。奇异时精确可解需 $\nabla\ell\in\operatorname{Ran}F$；在可辨识切空间／商空间建立问题，或明确使用坐标伪逆、投影及阻尼分别改变了什么。Fisher 是概率分布变化的局部 KL 度量；普通数据梯度外积 $\frac1n\sum s(x_i,y_i)s(x_i,y_i)^\top$ 为 empirical Fisher，通常不等于模型 Fisher 或 GGN。[AO1]

若输出采用指数族的自然参数，负对数似然对这些输出的 Hessian 等于相应输出 Fisher，可推出相应 GGN 与 Fisher 的等价；换一种输出参数化或损失后重新核对。期望下 Hessian 与 Fisher 的等式还要求交换求导积分、支持等正则条件。

当 $F\succ0$ 时，Fisher 度量与微分在光滑可逆重参数化下协同变换，使连续自然梯度向量场具有相应不变性。奇异 $F$ 的普通 Moore–Penrose 伪逆选出坐标欧氏最小范数解，该参数向量一般不协变；若讨论可辨识商空间或诱导的分布切向量，须单独证明其定义良好。有限步更新、近似求解、对角／Kronecker 因子化和加 $\lambda I$ 的坐标依赖阻尼也可能破坏不变性。采用 K-FAC 类 Kronecker 近似时，应明确它如何把联合四槽期望近似成两个二阶因子的乘积，不能宣称一般精确。

## A7 割线、Broyden、BFGS 与 SR1

给端点差 $s=x_{k+1}-x_k$、$y=F(x_{k+1})-F(x_k)$，割线条件仅要求 $B_{k+1}s=y$。未约束的方向仍有自由度。

**Broyden。** 最小 Frobenius 改动满足割线的 rank-1 更新为

\[
B_{k+1}=B_k+\frac{(y-B_ks)s^\top}{s^\top s},\qquad s\ne0.
\]

它不保证对称、正定或全域准确；适合更新一般 Jacobian 的方向知识。新的状态或参数改变目标函数时，需要重新判断历史割线是否仍有效。[AO2]

**BFGS。** 对标量优化取 $y=\nabla f_{k+1}-\nabla f_k$。逆 Hessian 近似更新

\[
H_{k+1}=(I-\rho sy^\top)H_k(I-\rho ys^\top)+\rho ss^\top,
\quad \rho=(y^\top s)^{-1}.
\]

$H_k\succ0$ 且 $y^\top s>0$ 时保持正定。L-BFGS 只保留少量 $(s,y)$ 对，用两轮递推求方向，工作量与存储为 $O(mn)$。噪声或非凸场景中的曲率条件、阻尼、跳过更新和线搜索均影响结论。

**SR1。** 从对称 $B_k$ 出发，令 $r=y-B_ks$，更新 $B_{k+1}=B_k+rr^\top/(r^\top s)$。允许捕获不定曲率，但当分母过小时应跳过或保护；它不保持正定。常与信赖域配合。[AO3]

## A8 凸分析、对偶与最优性证书

proper 闭凸函数在 $x\in\operatorname{dom}f$ 的次梯度 $g\in\partial f(x)$ 满足 $f(y)\ge f(x)+g^\top(y-x)$；无约束最优性为 $0\in\partial f(x)$。Fenchel 共轭 $f^*(y)=\sup_x(y^\top x-f(x))$ 与 $f(x)+f^*(y)\ge y^\top x$ 提供对偶构造。

约束 $g_i(x)\le0,h_j(x)=0$ 的 Lagrangian 为 $\mathscr L=f+\sum_i\lambda_i g_i+\sum_j\nu_jh_j$。KKT 包含可行性、$\lambda_i\ge0$、$\lambda_i g_i=0$ 和 $0\in\partial_x\mathscr L$。

凸目标、凸不等式和仿射等式下，满足 KKT 是全局最优的充分条件；Slater 等约束资格确保相应必要性和强对偶。非凸低秩／CP 拟合一般只能把一阶条件作为局部驻点条件。对偶可行值给下界，与原始可行值的差提供证书，而训练目标下降本身不提供全局最优性。[AO4]

半正定约束 $X\succeq0$ 属于锥约束；矩阵元素非负既非必要也非充分。用 Schur 补把某些二次条件改写为 LMI 时保留正定／可逆性前提。

## A9 近端算子与分裂算法

\[
\operatorname{prox}_{\lambda g}(v)=\arg\min_x\left[g(x)+\frac1{2\lambda}\|x-v\|^2\right].
\]

闭、proper、凸 $g$ 与 $\lambda>0$ 时解唯一；非凸情形可能多值。

| 结构 | 近端／投影操作 | 保留的语义 |
| --- | --- | --- |
| $\ell_1$ 稀疏 | $\operatorname{sign}(v_i)(|v_i|-\lambda)_+$ | 作用于指定参数坐标 |
| 分组 $\ell_2$ | $(1-\lambda/\|v_g\|)_+v_g$ | 组不重叠时分离；零向量分支直接为零 |
| 核范数 | $U\operatorname{diag}((\sigma_i-\lambda)_+)V^\top$ | 软阈值奇异值，非逐元素阈值 |
| 闭凸集合指标函数 | 欧氏投影 $\Pi_C$ | 换度量后需换相应投影 |
| PSD 锥 | 对称化后将负特征值截为零 | Frobenius 投影，不是所有范数下同一解 |

prox-gradient：$x_{k+1}=\operatorname{prox}_{\eta g}(x_k-\eta\nabla f(x_k))$；凸 $f$ 的梯度 $L$-Lipschitz 时，可选 $0<\eta\le1/L$ 或满足下降条件的回溯。加速法的保证与动量、凸性和误差控制相连。

两块 ADMM 对 $\min f(x)+g(z)$、$Ax+Bz=c$ 交替最小化加二次罚的 $x,z$ 子问题，再更新缩放乘子 $u\leftarrow u+Ax+Bz-c$。验收原始与对偶残差；标准凸收敛结论要求相应闭凸性、鞍点存在及子问题条件，不能无条件推广到多块或非凸张量分解。Moreau 分解与单调算子预解式可用于推导新的 prox；用它们减少复杂子问题，而不是堆砌优化名词。[AO6]

## A10 信赖域与模型失配

构造 $m(p)=f(x)+g^\top p+\tfrac12p^\top Hp$，取对称 $H$、$M\succ0$、$\Delta>0$，限制 $\|p\|_M\le\Delta$。约束内求解，比较

\[
\rho=\frac{f(x)-f(x+p)}{m(0)-m(p)}.
\]

只有预测下降为正时才使用该比值。依据实际与预测下降调整步长、阻尼或模型；这为局部线性缓存的失效／重建判据提供思路，但重建规则仍须由其实际目标误差与成本决定。

精确二次子问题的全局最优条件为：$p^\top Mp\le\Delta^2$，存在 $\lambda\ge0$ 使 $(H+\lambda M)p=-g$、$H+\lambda M\succeq0$、$\lambda(p^\top Mp-\Delta^2)=0$。这是欧氏球定理在变量 $M^{1/2}p$ 下的形式。[AO3] 奇异的移位矩阵仍可能对应最优解，不应强制可逆。大问题可用 HVP 与截断 CG；检测负曲率后转向边界，不能把不定 Hessian 强行当作 CG 的 SPD 系统。

## A11 POD、插值与非线性降阶

**POD。** 对快照矩阵 $X=[x_1,\ldots,x_n]$，SVD 的左奇异向量给最小平方重构子空间。物理质量矩阵 $M\succ0$ 时，先对 $M^{1/2}X$ 做 SVD，再映回 $V=M^{-1/2}U_r$，使 $V^\top M V=I$。样本时间／概率权重进入快照列缩放。高能量子空间未必包含低能量但对目标敏感或不稳定的方向。

**采样插值／DEIM。** 给正交基 $U\in\mathbb R^{n\times r}$ 和坐标选择矩阵 $P\in\mathbb R^{n\times r}$，若 $P^\top U$ 可逆：

\[
f\approx\mathcal D f,
\qquad \mathcal D=U(P^\top U)^{-1}P^\top.
\]

它插值选中分量并再现 $\operatorname{col}U$。由 $(I-\mathcal D)U=0$，直接得到误差由投影残差与斜投影稳定性共同控制：

\[
\|f-\mathcal D f\|\le\|I-\mathcal D\|\,\|(I-UU^\top)f\|.
\]

这条一般界无需预设某种贪心采样算法。选点可用逐步残差最大值或 $U^\top$ 的主元 QR；检查交叉矩阵条件数。非线性降阶的速度收益需要能**只计算被选分量及其依赖**；先计算整个 $f$ 再抽取，不能得到同样收益。DEIM 作为来源已列于 AO7；此处投影性质与误差式为直接推导，不代替原论文的更紧定理。

## A12 DMD、Koopman 与动态拟合

给配对快照 $X=[x_i]$、$Y=[y_i]$，$A=YX^+$ 是数据上的最小二乘线性推进。取仅保留非零奇异值的秩约化 SVD $X=U\Sigma V^*$，其中 $r=\operatorname{rank}X$、$\Sigma\in\mathbb R^{r\times r}\succ0$，再构造小矩阵；普通 economy/thin SVD 仍可能含零奇异值，不能直接取逆。数值实现需声明秩阈值，零矩阵 $X$ 的情形单独处理。

\[
\widetilde A=U^*YV\Sigma^{-1}.
\]

非零特征对 $\widetilde A w=\lambda w$ 可提升为 $\phi=YV\Sigma^{-1}w$ 的特征模态；除以 $\lambda$ 只是不同归一化。截断 SVD 引入正则化／模型偏差，不能称为同一个精确全秩算子的完整谱。[AO8]

精确配对可由某线性映射满足的条件是 $\ker X\subseteq\ker Y$；秩亏快照不满足此条件时，不能用单一精确推进矩阵解释全部数据。DMD 的时间采样、噪声和观测变量决定拟合问题。

Koopman 算子作用于观测函数：$(\mathcal K g)(x)=g(F(x))$，即使状态推进 $F$ 非线性，$\mathcal K$ 对 $g$ 仍线性。先指定观测函数空间，检查复合后仍属于该空间；线性本身不保证有界。有限特征集合不一定在其作用下封闭；DMD／EDMD 给选定观测空间和采样分布上的近似，不自动提供全局线性化。把升维空间的线性性转成预测结论时，要验证闭包残差及外推工作域。

## A13 统计与计算误差的联合验收

将实际误差分成可辨识的来源：函数／结构近似、参数拟合、输入分布估计、截断／迭代求解、量化／舍入、状态陈旧与执行变换。相应二阶矩有限且交叉协方差为零时，可以按方差相加；均方误差还需计入均值／偏差的作用。一般先保留三角界或联合二次型。

选择最低成本、能区分机制的证据：小尺寸恒等式核验、独立随机方向、路径上的余项、留出状态的目标误差、受控的分布变化、真实完整调用周期。用同一批数据选择秩和估计其误差时记录选择偏差；历史轨迹相邻样本也不自动独立。

停止求解的容限应对齐总误差预算。统计／建模误差远大于线性求解误差时，继续降低残差未必改变决策；相反，小输出误差若会被后续非正规链强烈放大，局部平均误差也不充分。

## 来源

- **[AO1]** Martens (2020), [New Insights and Perspectives on the Natural Gradient Method](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf)，§9–12：Fisher 作用、GGN 条件等价、阻尼、empirical Fisher、参数化。
- **[AO2]** Broyden (1965), [A Class of Methods for Solving Nonlinear Simultaneous Equations](https://www.math.unm.edu/~vageli/courses/Ma576/papers/Broyden_65.pdf)，更新构造与计算流程。
- **[AO3]** Yudong Chen (2023), [BFGS and SR1](https://pages.cs.wisc.edu/~yudongchen/cs726_sp23/Lecture_22_BFGS_SR1.pdf)，作者课程讲义：割线、正定性、有限记忆、SR1 保护；[Trust-Region Methods (2024)](https://pages.cs.wisc.edu/~yudongchen/cs726_sp24/Lecture_24_trust_region.pdf)，§3 Theorem 1 核对二次信赖域子问题的全局最优条件。
- **[AO4]** Boyd & Vandenberghe，[Duality 讲义](https://web.stanford.edu/class/ee364a/lectures/duality.pdf)，对偶下界、Slater、KKT、敏感度。
- **[AO5]** Süli，[Finite Element Methods](https://people.maths.ox.ac.uk/suli/fem.pdf)，§4：对偶误差；此处有限维伴随恒等式直接推导。
- **[AO6]** Parikh & Boyd，[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)，§2–4、§6：近端、分裂、分组与谱函数。
- **[AO7]** Chaturantabut & Sorensen (2010), [Nonlinear Model Reduction via Discrete Empirical Interpolation](https://epubs.siam.org/doi/10.1137/090766498)。原文获取边界与投影式的推导依据见 `source-coverage.md`。
- **[AO8]** Tu 等，[On Dynamic Mode Decomposition: Theory and Applications](https://arxiv.org/pdf/1312.0041)，§2 的线性一致性与 Algorithm 2，以及与 Koopman 的条件联系。

双侧加权低秩、联合正规方程和示例在当前定义下直接推导；它们使用 SVD、投影和期望代数，不宣称新的原创算法或无条件泛化保证。
