# 算子理论与泛函分析

在判断可解性、稳定性、谱、积分／微分算子或离散化时使用本参考。默认实空间；复空间中 `*` 表示共轭转置，内积采用第一槽共轭、第二槽线性。公式中的范数必须与实际输入、输出和目标相容。

## 目录

- O1 空间、对偶与有界性
- O2 范数、Schatten 类与误差作用
- O3 伴随、Riesz 映射与投影
- O4 谱定理与函数演算
- O5 预解式、Neumann 级数与逆的扰动
- O6 非正规性与瞬态放大
- O7 不动点、单调性与适定性
- O8 紧算子与 Fredholm 可解性
- O9 积分核、RKHS 与表示定理
- O10 Fourier、卷积与边界条件
- O11 弱导数、Sobolev 空间与变分形式
- O12 Galerkin 与 Petrov–Galerkin
- O13 半群、Duhamel 与轨迹误差
- O14 原领域工具的选择与来源

## O1 空间、对偶与有界性

**对象与动作。** 对线性映射先声明 $A:D(A)\subseteq V\to W$。有限维数组中所有线性映射都连续；无限维微分算子通常无界，定义域、边界条件和闭性是算子的一部分。不要只凭同一微分表达式视为同一个算子。

Banach 空间对所用范数完备；Hilbert 空间的范数来自内积。$V^*$ 是连续线性泛函空间。对处处定义的有界算子 $A\in\mathcal L(V,W)$，

\[
\|A\|_{V\to W}=\sup_{\|x\|_V=1}\|Ax\|_W<\infty.
\]

若仅在 $D(A)$ 上定义，取上确界时必须限制 $x\in D(A)$。给闭算子的定义域配图范数 $\|x\|_{D(A)}=\|x\|_V+\|Ax\|_W$ 后，$A:D(A)\to W$ 有界；这不意味着它相对于原来的 $V$ 范数有界。

若把网格数组解释为连续场的采样，先给质量矩阵、积分权重或离散范数。例如 $M\succ0$ 时 $\|x\|_M^2=x^*Mx$；网格细化时未加权的 $\ell_2$ 范数可能不对应同一个连续 $L^2$ 量。

**工具边界。** 有界逆定理：Banach 空间之间的有界双射具有有界逆。闭图定理要求算子定义在整个 Banach 空间且图闭；不能据此断言一个仅在稠密子域定义的微分算子有界。统一有界原理把逐点有界的有界算子族提升为统一算子范数界，但需要 Banach 定义域。用这些结果检查“每个输入都表现稳定”能否推广为统一稳定性。

## O2 范数、Schatten 类与误差作用

对有限维 $A$，先选择与主张相符的量：

| 要回答的问题 | 量与公式 | 使用条件 |
| --- | --- | --- |
| 最坏方向误差 | $\|A\|_2=\sigma_1(A)$ | 这是欧氏坐标公式；加权范数先变换为下述 $\widetilde A$ |
| 全分量平方能量 | $\|A\|_F^2=\sum_i\sigma_i^2$ | 基为正交归一，或先变成这样的坐标 |
| 低秩凸替代 | $\|A\|_* =\sum_i\sigma_i$ | 核范数不等于秩，阈值依赖正则化 |
| 一般谱惩罚 | $\|A\|_{S_p}=(\sum_i\sigma_i^p)^{1/p}$ | $p\ge1$ 为范数；$0<p<1$ 仅拟范数 |
| 给定输入分布的误差 | $\mathbb E\|Ax\|_G^2=\operatorname{tr}(GAC A^*)$ | $A,G$ 固定，$G\succeq0$，$C=\mathbb E[xx^*]$ 有限 |

若输入、输出度量为 $G_V,G_W\succ0$，则

\[
\|A\|_{G_V\to G_W}
=\sigma_1(\widetilde A),\qquad
\widetilde A=G_W^{1/2}AG_V^{-1/2}.
\]

相应 Hilbert–Schmidt／Schatten 量也应作用于 $\widetilde A$。例如 $A=I$、$G_V=I$、$G_W=\operatorname{diag}(4,1)$ 的加权算子范数为 2，原坐标 $\sigma_1(A)=1$。奇异半正定矩阵只给半范数，需另行处理不可观测方向。

常用推导工具：相容算子范数下 $\|AB\|\le\|A\|\|B\|$；欧氏正交坐标下 $\|AB\|_F\le\|A\|_2\|B\|_F$、$|\langle A,B\rangle_F|\le\|A\|_F\|B\|_F$、$|\operatorname{tr}(A^*B)|\le\|A\|_*\|B\|_2$。有限维 $\|A\|_2\le\|A\|_F\le\sqrt{r}\|A\|_2$，$r=\operatorname{rank}A$。

无限维 Hilbert–Schmidt 与 trace-class 分别要求 $\sum_i\sigma_i^2<\infty$ 和 $\sum_i\sigma_i<\infty$；不能把所有有界算子都赋予有限 Frobenius 范数或迹。目标误差与奇异度量的处理见 `approximation-and-optimization.md`。

## O3 伴随、Riesz 映射与投影

对有界线性 $A:V\to W$，连续线性对偶的拉回为 $A':W^*\to V^*$、$A'\lambda=\lambda\circ A$。有限维若以列系数 $c$ 表示线性泛函 $\lambda(y)=c^\top y$，拉回系数为 $A^\top c$，复数情形也不加共轭。

Hilbert 伴随 $A^\dagger:W\to V$ 则由内积定义。若内积矩阵分别为 $G_V,G_W\succ0$，则

\[
\langle Ax,y\rangle_W=\langle x,A^\dagger y\rangle_V,
\qquad A^\dagger=G_V^{-1}A^*G_W.
\]

在本参考的内积约定下，Riesz 映射 $R_Vx=\langle x,\cdot\rangle_V$ 到连续线性对偶是共轭线性的，且 $A^\dagger=R_V^{-1}A'R_W$。不要将复线性对偶系数与 Riesz 向量表示混为一谈。

若 $A:D(A)\subset V\to W$ 稠密定义但无界，伴随一般也有真子域：

\[
D(A^\dagger)=
\{y\in W:\exists z\in V,\quad
\langle Ax,y\rangle_W=\langle x,z\rangle_V
\ \forall x\in D(A)\},\qquad A^\dagger y=z.
\]

稠密性保证 $z$ 唯一；等价地，关于 $x$ 的上述配对须对原 $V$ 范数有界。此时不能直接写 $A^\dagger:W\to V$，形式分部积分得到的微分表达式也未确定伴随的边界条件。

微分 $df$ 是协向量；实坐标下 $\operatorname{grad}_{G_V}f=G_V^{-1}\partial f$。因此 VJP 的拉回语义不要求先选择梯度度量，而“梯度下降方向”要求。复变量实值目标按实配对求 VJP；一般非全纯映射还含共轭增量，见微积分参考 C8。

满列秩基 $U$ 的 $G$ 正交投影为

\[
P_U=U(U^*GU)^{-1}U^*G,
\quad P_U^2=P_U,
\quad P_U^\dagger=P_U.
\]

一般幂等矩阵只是斜投影，其范数可大于 1。数值上先对 $G^{1/2}U$ 做 QR 或稳定求解，避免显式求 Gram 逆。闭子空间的投影定理与 Riesz 表示把这些公式推广到 Hilbert 空间；闭性是最近点存在的关键。

## O4 谱定理与函数演算

**选择条件。** 有限维复正规矩阵 $AA^*=A^*A$ 可酉对角化；自伴矩阵有实谱；正定矩阵有正特征值。一般复矩阵用 Schur 分解 $A=QTQ^*$，$T$ 上三角；坚持实坐标时用正交实 Schur 分解，$T$ 为含 $1\times1$／$2\times2$ 对角块的准上三角。Jordan 形式适合精确分类，但在数值扰动下不稳定。

紧自伴算子在可分 Hilbert 空间中具有正交特征展开，非零特征值只能在 0 聚集。一般有界自伴算子则可能有连续谱，应使用谱测度 $A=\int\lambda\,dE(\lambda)$，不能假设存在离散的特征向量基。

对正规矩阵，$f(A)=U f(\Lambda)U^*$。对一般矩阵，不可逐元素计算 $f(A)$；使用 Schur、矩阵函数算法或直接计算 $f(A)v$。局部 resolvent 积分

\[
f(A)=\frac{1}{2\pi i}\oint_\Gamma f(z)(zI-A)^{-1}\,dz
\]

此处针对有限维矩阵或有界算子，要求 $\Gamma$ 位于预解集、正向围绕谱一周，且 $f$ 在相应区域解析。无界自伴算子的谱演算还要给出作用域，不能直接套包围有界谱的闭合路径。重特征值附近应比较与其余谱分离的整个子空间；稳定程度仍依赖谱间隙及正规性，见数值线代中的谱扰动工具。

## O5 预解式、Neumann 级数与逆的扰动

本节的级数与扰动界先限定为同一 Banach 空间上的有界算子。仅当 $z$ 在预解集时定义 $R_A(z)=(zI-A)^{-1}$；实算子的复谱通过复化理解。对 $\|K\|\le q<1$：

\[
(I-K)^{-1}=\sum_{j=0}^{\infty}K^j,
\qquad
\left\|(I-K)^{-1}-\sum_{j=0}^{m}K^j\right\|
\le\frac{q^{m+1}}{1-q}.
\]

这是可验证的充分条件，并非可逆性的必要条件。若 $A$ 可逆且 $\|A^{-1}E\|<1$，则

\[
(A+E)^{-1}-A^{-1}=-(A+E)^{-1}EA^{-1},
\quad
\|(A+E)^{-1}-A^{-1}\|
\le\frac{\|A^{-1}\|^2\|E\|}{1-\|A^{-1}E\|}.
\]

用它分析近似 Jacobian、预条件器和隐式层的误差。实现优先解线性系统；写出逆不意味着物化逆矩阵。

## O6 非正规性与瞬态放大

当 $A$ 非正规时，特征值不能单独控制 $\|A^k\|$ 或 $\|e^{tA}\|$。对 $A\in\mathbb C^{n\times n}$、$\varepsilon>0$，采用开集版本的二范数伪谱

\[
\Lambda_\varepsilon(A)=\{z\in\mathbb C:\sigma_{\min}(zI-A)<\varepsilon\}
=\bigcup_{\substack{E\in\mathbb C^{n\times n}\\\|E\|_2<\varepsilon}}\sigma(A+E).
\]

这等于 $\sigma(A)\cup\{z\in\rho(A):\|R_A(z)\|_2>1/\varepsilon\}$。即使 $A$ 是实矩阵，上述并集也允许复扰动；限制 $E$ 为实数或指定结构会得到不同的扰动集合。例如实标量 $A=0$ 的复伪谱包含 $i\varepsilon/2$，实标量扰动却只能产生实特征值。可用移位系统求解或最小奇异值估计检查预解式响应，而非只画特征值。

**反例。** $A=\begin{pmatrix}-1&K\\0&-1\end{pmatrix}$ 的两个特征值都为 $-1$，但

\[
e^{tA}=e^{-t}\begin{pmatrix}1&Kt\\0&1\end{pmatrix},
\qquad \|e^{A}e_2\|_2=e^{-1}\sqrt{1+K^2}.
\]

大 $K$ 产生强瞬态放大，随后衰减。对残差链、缓存误差传播或深层 Jacobian 乘积，检查奇异值、非正规耦合及方向对齐；即使每层谱“看起来稳定”，乘积也可能放大误差。此例是直接推导，谱诊断依据 Trefethen 的原领域方法 [OF3]。

## O7 不动点、单调性与适定性

**Banach 不动点。** 在非空完备集合上，$T$ 映回该集合且具有统一 Lipschitz 常数 $0\le q<1$，则存在唯一不动点，迭代收敛；残差证书为

\[
\|x-x_*\|\le\frac{\|x-T(x)\|}{1-q}.
\]

单个点的 Jacobian 谱半径小于 1 只支持适当条件下的局部结论，不能冒充整个工作域上的压缩映射。非扩张 $q=1$ 也不自动保证直接迭代收敛，例如 $T=-I$。

**强单调性。** 实 Hilbert 空间中若 $\langle F(x)-F(y),x-y\rangle\ge\mu\|x-y\|^2$、$\mu>0$，则解唯一且对右端扰动的逆稳定常数至多 $1/\mu$（在解存在的范围内）。存在性还需连续性、定义域与适当 coercivity／最大单调性条件。

对 proper、下半连续凸 $f$ 与 $\lambda>0$，$\operatorname{prox}_{\lambda f}=(I+\lambda\partial f)^{-1}$ 将优化与单调算子联系起来；其坚定非扩张性是稳定分裂算法的工具，见优化参考。不要把非凸 prox 的多值问题省去。

## O8 紧算子与 Fredholm 可解性

本节在 Hilbert 空间上讨论紧算子。紧性把有界集合映为相对紧集合，并等价于可在算子范数下由有限秩算子逼近；对一般 Banach 空间不能省略有限秩逼近所需的空间条件。对紧自伴 $K$ 和非零实数 $\lambda$，$\operatorname{Ran}(K-\lambda I)$ 闭且

\[
(K-\lambda I)u=f\ \text{可解}
\iff f\perp\ker(K-\lambda I).
\]

一般紧算子的 Fredholm alternative 对 $I-K$ 使用伴随核：$f\perp\ker(I-K^*)$；$\ker(I-K)$ 与余核有限维。在 $\lambda=0$ 处不能无条件沿用上述紧自伴可解性结论，像空间可能不闭。求解奇异系统前检查相容条件；最小二乘、伪逆和正则化回答的是不同问题。

## O9 积分核、RKHS 与表示定理

在 $\sigma$-有限测度空间上，对 $Tf(x)=\int K(x,y)f(y)\,d\mu(y)$：若 $K\in L^2(\mu\times\mu)$，则 $T:L^2(\mu)\to L^2(\mu)$ 为 Hilbert–Schmidt，$\|T\|\le\|K\|_{L^2}$。若 $\operatorname{ess\,sup}_x\int|K(x,y)|d\mu(y)\le a$、$\operatorname{ess\,sup}_y\int|K(x,y)|d\mu(x)\le b$，Schur 测试给出 $\|T\|_{2\to2}\le\sqrt{ab}$。

离散求积得到 $T_{ij}=K(x_i,y_j)w_j$；权重属于算子，不能把纯采样数组 $K_{ij}$ 的谱直接当成积分算子的谱。Nyström 近似也需区分求积离散化、低秩列抽样及正则化。

正定核定义 RKHS。按首槽共轭、次槽线性的内积约定，令 $k_x=k(\cdot,x)$、$K_{ij}=k(x_i,x_j)$，再生关系是 $f(x)=\langle k_x,f\rangle$。对于仅通过 $f(x_1),\ldots,f(x_n)$ 依赖数据的目标，加上严格递增的 $\Omega(\|f\|_\mathcal H)$，任何存在的最小解可写成

\[
f(\cdot)=\sum_{i=1}^n\alpha_i k(\cdot,x_i).
\]

依据是正交分解 $f=f_\parallel+f_\perp$：$f_\perp$ 不改变采样值，只增加范数。表示定理降低搜索空间，但不保证极小值存在、唯一或 Gram 矩阵良态。样本值向量为 $K\alpha$；复 Hermitian 核不能把两个核参数直接交换。核 ridge 采用 $\sum_i|f(x_i)-y_i|^2+\lambda\|f\|_{\mathcal H}^2$、$\lambda>0$ 时，可取系数解 $(K+\lambda I)\alpha=y$；若损失改为样本均值，同一正则参数对应 $K+n\lambda I$。核矩阵的非负定性指二次型，而非每个元素非负。[OF4]

## O10 Fourier、卷积与边界条件

循环卷积 $y_i=\sum_j k_{(i-j)\bmod n}x_j$ 由 DFT 对角化。采用未归一化前向变换 $\widehat x_\omega=\sum_{j=0}^{n-1}x_j e^{-2\pi i\omega j/n}$、逆变换含 $1/n$ 时，$\widehat y_\omega=\widehat k_\omega\widehat x_\omega$；若前向 DFT 为酉归一化，右侧多一个 $\sqrt n$。多通道卷积在每个频率上保留通道矩阵 $\widehat y_\omega^a=\widehat K^a{}_{b}(\omega)\widehat x_\omega^b$，并非整体都变成标量乘法。

**使用。** 识别平移对称性、快速作用、频率响应与条件数。FFT 的优势需要卷积边界、填充、核大小及转换成本配合。有限区间零填充形成 Toeplitz 结构；可嵌入较大的循环卷积，但不能把原矩阵的谱直接等同于较大 circulant 的谱。

连续微分的 Fourier 符号由 $\partial_j\mapsto i\xi_j$ 给出，周期／整空间与合适正则性下适用。变系数、边界或非线性会耦合频率。用 Parseval 比较能量时保留 FFT 归一化约定。

## O11 弱导数、Sobolev 空间与变分形式

弱导数通过 $\varphi\in C_c^\infty(\Omega)$ 定义：$\int u\,\partial_j\varphi=-\int(\partial_j u)\varphi$。$H^1(\Omega)$ 要求函数及一阶弱导数平方可积；$H_0^1$ 定义为 $C_c^\infty(\Omega)$ 在 $H^1$ 范数下的闭包。在 Lipschitz 等适当边界正则性下，它也等于零迹子空间。

对零 Dirichlet 问题 $-\nabla\cdot(a\nabla u)=f$，取 $V=H_0^1(\Omega)$，积分分部得到

\[
a(u,v)=\int_\Omega \nabla v^\top a\nabla u,
\qquad \ell(v)=\int_\Omega f v,
\qquad a(u,v)=\ell(v)\quad\forall v\in V.
\]

Lax–Milgram：在实 Hilbert 空间中，双线性形式连续 $|a(u,v)|\le M\|u\|\|v\|$、coercive $a(v,v)\ge\alpha\|v\|^2$、$\alpha>0$ 且 $\ell$ 连续，则唯一解满足 $\|u\|\le\|\ell\|_{V^*}/\alpha$。复空间可采用对 $u$ 线性、对 $v$ 共轭线性的形式与共轭线性 $\ell$，coercivity 改为 $\operatorname{Re}a(v,v)\ge\alpha\|v\|^2$。Poincaré 不等式把梯度半范数转成满足边界约束后的范数；对纯 Neumann 问题需处理常数核和右端相容性。弱式会降低导数要求，不能据此忽略边界积分。[OF2]

## O12 Galerkin 与 Petrov–Galerkin

选试探空间 $V_h=\operatorname{span}\{\phi_j\}$ 和测试空间 $W_h=\operatorname{span}\{\psi_i\}$：

\[
u_h=\sum_jc_j\phi_j,
\quad A_{ij}=a(\phi_j,\psi_i),
\quad b_i=\ell(\psi_i),
\quad Ac=b.
\]

相容 Galerkin 取 $V_h=W_h\subset V$。在沿用连续问题的双线性形式与右端、无额外积分或几何误差时，连续且 coercive 的问题给出 Céa 界 $\|u-u_h\|\le(M/\alpha)\inf_{v_h\in V_h}\|u-v_h\|$。非相容空间、近似积分或近似边界还要计入一致性误差，不能直接套此式。

Petrov–Galerkin 允许不同空间，必须检查离散 inf–sup：

\[
\beta_h=\inf_{v_h\ne0}\sup_{w_h\ne0}
\frac{|a(v_h,w_h)|}{\|v_h\|_V\|w_h\|_W}>0.
\]

有限维、基线性无关且 $\dim V_h=\dim W_h$ 时，$\beta_h>0$ 保证对任意右端有唯一离散解；若维数不同，单侧 inf–sup 只保证试探方向无核，仍需检查右端相容性／伴随核。例如 $A=(1,0)^\top$ 的单侧常数为 1，但 $Ac=(0,1)^\top$ 不可解。稳定近似界涉及 $\beta_h$，不能沿用对称正定 Galerkin 的能量最优性。降阶模型 $W^*AVc=W^*b$ 正是同一结构；配对矩阵小不代表 $\beta_h$ 大。**Petrov–Galerkin 与 Weyl 张量的 Petrov 分类是不同工具。**

## O13 半群、Duhamel 与轨迹误差

有限维 $\dot x=Ax+f(t)$ 有

\[
x(t)=e^{tA}x(0)+\int_0^te^{(t-s)A}f(s)\,ds.
\]

误差系统 $\dot e=A(t)e+r(t)$ 使用状态转移算子 $\Phi(t,s)$；固定 $A$ 时回到上式。若 $\|\Phi(t,s)\|\le M e^{\omega(t-s)}$，即可界定注入误差的累积与衰减。无限维无界生成元需要半群条件，不能只形式展开指数级数。

对于实微分方程，若 $\langle F(x)-F(y),x-y\rangle\le L\|x-y\|^2$，可由 Grönwall 控制轨迹差；$L$ 可为负。对线性欧氏系统，$L=\lambda_{\max}((A+A^*)/2)$ 给出瞬时能量增长界。用这个量与谱、伪谱一起分析稳定性。

离散求解还受稳定域控制：前向 Euler 的标量放大因子是 $1+h\lambda$；连续稳定 $\operatorname{Re}\lambda<0$ 并不保证任意步长下离散稳定。缓存近似、时间积分和算子分裂的误差必须放进真实递推关系。

## O14 原领域工具的选择与来源

Hahn–Banach／分离用于证明最优性或构造分离泛函；弱收敛与弱紧性用于存在性；紧性、谱理论用于可解性与离散逼近。它们不是每次分析都要展示的前置课。调用定理前写出当前对象满足的假设，再给它改变了哪个结论。

- **[OF1]** Casey Rodriguez, MIT 18.102 (2021), [课程讲义](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/)：空间、有界算子、对偶、投影／Riesz；[Lecture 22](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/57596554e180e442c9487f580303147b_MIT18_102s21_lec22.pdf) 的 Fredholm 与紧自伴谱定理。另核对同课程 [Melrose 2020 原讲义](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/3d4cc88026d44a01f936cd6a0aa995cb_MIT18_102s20_lec_FA.pdf) 第 1 章 §11 式 (1.108) 的图范数；无界伴随域在 O3 按定义与 Riesz 表示给出。实际核对定位见 `source-coverage.md`。
- **[OF2]** Endre Süli, [Finite Element Methods for PDEs](https://people.maths.ox.ac.uk/suli/fem.pdf), 2025 讲义，§1.1–1.2、§2.4、§4、§5：Sobolev、变分解、Lax–Milgram、Céa、对偶误差、演化稳定性。
- **[OF3]** Trefethen (1997), [Pseudospectra of Linear Operators](https://people.maths.ox.ac.uk/trefethen/publication/PDF/1997_72.pdf)，§1 与算子例：谱以外的响应与非正规性。
- **[OF4]** Schölkopf, Herbrich & Smola (2001), [A Generalized Representer Theorem](https://www.cise.ufl.edu/~arunava/Teaching/cap6617sp21/Readings/SchHerSmo01.pdf)，Theorem 1 及正交分解证明。
- **[OF5]** Parikh & Boyd, [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)，§2.3、§3.2：坚定非扩张、平均算子与次微分预解式。

本参考的简单矩阵反例、投影公式和误差恒等式按给定定义直接推导；引用用于界定所属理论、成立条件与进一步求证的原始位置。
