# 导数与割线

用本参考分析导数的槽位、方向作用、有限变化及锚点状态。FFN 是可迁移范例；对新的计算图重新应用链式法则与乘积法则。

## 内容

- 一般导数与直接作用
- FFN 的导数与有限变化
- 动态修正与锚点缓存
- 方法来源

## 一般导数与直接作用

先写明映射 F:V→W 以及保持固定的参数。把一阶导数视为 DF(x):V→W，把高阶导数视为对多个输入方向的多线性作用。

\[
D^kF(x)[u_1,\ldots,u_k]\in W.
\]

仅需方向作用时直接收缩。JVP 为 DF(x)[u]；VJP 将输出协向量沿 DF 拉回到输入对偶空间。对实线性或复线性映射，线性对偶拉回的列系数表达都是 \(J^{\mathsf T}\lambda\)。将协向量用内积识别为向量后，才使用相应的度量伴随；复 Hermitian 内积 \(h(u,v)=u^{\mathsf H}Mv\) 下为 \(M_V^{-1}J^{\mathsf H}M_W\)。一般实可微的复映射还包含 \(\delta\bar z\) 项，按 [张量微积分 C8](tensor-calculus-and-autodiff.md) 处理。先声明配对、实／复可微性及框架梯度约定，再决定转置与共轭。

对依赖同一输入的逐元素乘积，应用完整乘积法则。若 F(x)=B[a(x)⊙b(x)]：

\[
DF(x)[u]=B\{Da(x)[u]\odot b(x)+a(x)\odot Db(x)[u]\}.
\]

对条件化映射 F(x;c)，明确研究的是固定 c 的输入导数，还是 x、c 同时变化的总差分。方向求导、对参数求导和时间差分使用各自的依赖关系。

## FFN 的导数与有限变化

### 统一列向量和批处理约定

令输入空间为 V=Rᵈ，输出空间为 W=Rᵒ，隐藏宽度为 m。单 token 采用列向量：

\[
F(x)=W_2\phi(W_1x+b_1)+b_2,
\quad W_1\in\mathbb R^{m\times d},
\quad W_2\in\mathbb R^{o\times m}.
\]

批处理数组 X 采用行存样本的数学约定，X 的 shape 为 N×d：

\[
F(X)=\phi(XW_1^\top+\mathbf1b_1^\top)W_2^\top+\mathbf1b_2^\top.
\]

这里的“行存样本”规定矩阵语义，并不单独规定物理内存 stride。后续单 token 的近似矩阵 A_col 与批处理的 A 满足 A=A_colᵀ。

### 导数及其类型

设 w_h 是 W₁ 的第 h 行所定义的输入协向量，v_h 是 W₂ 的第 h 列所定义的输出向量，z_h=w_h(x)+(b₁)_h。若激活在相关区域为 Cᵏ：

\[
DF(x)=\sum_h\phi'(z_h)\,v_h\otimes w_h,
\qquad
D^2F(x)=\sum_h\phi''(z_h)\,v_h\otimes w_h\otimes w_h,
\]

\[
D^kF(x)=\sum_h\phi^{(k)}(z_h)\,v_h\otimes w_h^{\otimes k}
\in W\otimes\operatorname{Sym}^k(V^*).
\]

k 阶导数具有 k+1 个张量槽，其中 k 个输入槽对称。这给出至多 m 项的部分对称 CP 表示；“至多 m 项”本身不保证有效秩小，也不保证存在准确的极低秩近似。

计算二阶方向作用时可直接收缩：

\[
D^2F(x)[u,v]
=
W_2\!\left[
\phi''(z)\odot(W_1u)\odot(W_1v)
\right].
\]

无需构造 o×d×d 的完整导数数组。类似地，JVP 计算 Jv，VJP 拉回输出协向量；仅在目标确实需要时才物化 Jacobian。

若所研究的算子输入在归一化之前，还必须应用完整链式法则：

\[
D^2(G\circ F)[u,v]
=
D^2G(F)[DF[u],DF[v]]
+
DG(F)[D^2F[u,v]].
\]

对独立逐 token 的 FFN，其跨 token Jacobian 为块对角；周围 attention 的跨 token 耦合仍需在完整网络中考虑。

### 精确割线与局部线性化

固定权重、层、专家和其他条件。令 Δx=x_t−x_s，Δz_h=w_h(Δx)。定义：

\[
\beta_h=
\begin{cases}
\dfrac{\phi(z_{t,h})-\phi(z_{s,h})}{\Delta z_h},&\Delta z_h\ne0,\\[6pt]
\phi'(z_{s,h}),&\Delta z_h=0\ \text{且导数存在}.
\end{cases}
\]

零位移处导数不存在时，可取任意有限 β_h，因为它乘的 Δz_h 为零。平滑情形也可写为：

\[
\beta_h=\int_0^1\phi'(z_{s,h}+\tau\Delta z_h)\,d\tau,
\qquad
\bar J_{s,t}=W_2\operatorname{diag}(\beta)W_1.
\]

于是精确有：

\[
F(x_t)-F(x_s)=\bar J_{s,t}\Delta x.
\]

对于 ReLU 等分段线性激活，沿线段的绝对连续性和几乎处处导数也可给出相应积分关系；不能把跨拐点的有限差分直接替换为单个端点的导数。

端点对只固定算子在 Δx 上的作用，不能唯一辨识整个矩阵。上述路径积分选择了一个具体算子。获得精确 β 可能仍需新的隐藏投影和激活，因此这个恒等式本身不是低成本算法。

### 锚点近似和误差

固定 A_col，定义锚点预测：

\[
\widehat F_t=F_s+A_{\rm col}\Delta x,
\qquad
e_t=F_t-\widehat F_t=(\bar J_{s,t}-A_{\rm col})\Delta x.
\]

若 A_col=W₂diag(α)W₁：

\[
e_t=\sum_h(\beta_h-\alpha_h)\,v_h\,w_h(\Delta x).
\]

误差由系数偏差、实际输入方向和输出读出共同决定。若 F 为 C²，另一种精确分解是：

\[
e_t=
(J_s-A_{\rm col})\Delta x+
\int_0^1(1-\tau)D^2F(x_s+\tau\Delta x)[\Delta x,\Delta x]\,d\tau.
\]

这将固定算子失配与轨迹方向的非线性分开。对于真实缓存，层、专家、条件、token 顺序或权重改变时，要判断是否仍在同一个函数的分析范围。

若条件 c 同时变化，可先使用恒等分解：

\[
F(x_t;c_t)-F(x_s;c_s)
=
[F(x_t;c_s)-F(x_s;c_s)]
+
[F(x_t;c_t)-F(x_t;c_s)].
\]

第二项不能无声地并入只对 x 求得的割线算子。

## 动态修正与锚点缓存

保持批处理约定。令 A 为 d×o，U 为 d×r，V 为 o×r，G_t 为 N×r。考虑作用于输入增量的候选模型：

\[
\widehat F_t=
F_s+\Delta X A+
[(\Delta XU)\odot G_t]V^\top.
\]

其可复用状态与执行顺序是：

\[
C_s=F_s-X_sA,\qquad P_s=X_sU,
\]

\[
P_t=X_tU,\qquad
R_t=(P_t-P_s)\odot G_t,\qquad
\widehat F_t=X_tA+C_s+R_tV^\top.
\]

这既保持 X_t=X_s 时回到锚点输出，又将动态中间量控制为 N×r。若直接加入 (X_tU⊙G_t)Vᵀ，会破坏这个锚点条件。

若要表达“条件改变了整个低秩函数”，则需要另一种差分：

\[
(P_t\odot G_t-P_s\odot G_s)V^\top
=
[(P_t-P_s)\odot G_t+
P_s\odot(G_t-G_s)]V^\top.
\]

这与“只修正输入增量”的模型不同，应由目标决定采用哪一种。

对全局共享增益，可以缩放小因子的列。若显式构造 A_t=A+Udiag(g_t)Vᵀ，则缓存也要相容地改为 C_t=C_s−(P_s⊙g_t)Vᵀ，或保留等价的锚点减项。逐 token 的 G_t 通常不能折叠成单个共享 A_t；直接物化 N×d×o 算子会引入巨大中间量。

对于 A=W₁ᵀdiag(α)W₂ᵀ，缓存还满足含偏置的恒等式：

\[
Z_s=X_sW_1^\top+\mathbf1b_1^\top,
\qquad
C_s=
[\phi(Z_s)-(Z_s-\mathbf1b_1^\top)\odot\alpha]W_2^\top
+\mathbf1b_2^\top.
\]

它提供另一条缓存构建路径。锚点输出 F_s 仍需供后续网络使用，因此缓存成本应按实际共享、融合或多路累加调度计算。不能仅凭这个改写把缓存构建成本视为零。

## 方法来源

参考 [Laue 等 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/0a1bf96b7165e962e90cb14648c9462d-Paper.pdf)与 [2020 张量求导工作](https://arxiv.org/abs/2010.03313)，保留导数输入输出关系并化简表达。本文 FFN 恒等式按给定定义直接推导；历史框架的性能比较按原实验范围理解。
