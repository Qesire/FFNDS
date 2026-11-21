# 张量代数、对称性与群作用

本文件提供从“一个多轴数组”走向“有明确作用空间的数学对象”的工具。一般 ML 计算继续使用命名轴、显式求和或带输出标签的 einsum；只有声明空间、对偶和变换律之后，才把上下指标解释为几何类型。来源与阅读位置见 [几何来源台账](sources-geometry.md)。

## 内容

- TA01 张量积、对偶与多线性化
- TA02 收缩、迹、偏迹与合法复合
- TA03 置换、对称化与反对称化
- TA04 外代数、楔积与可分解性
- TA05 度量、升降指标与伴随
- TA06 对称张量、多项式与极化
- TA07 Young 对称性与代数曲率投影
- TA08 群作用、轨道、稳定子与不变量
- TA09 等变映射与群平均
- TA10 不可约分解、Schur 引理与耦合
- TA11 应力、弹性张量与 Mandel/Kelvin 表示
- 选用与迁移

## TA01 张量积、对偶与多线性化

**解决什么。** 判定哪些槽可以收缩，区分“张量作为对象”和“张量的数组表示”，把多线性算子化为普通线性映射。

有限维实或复空间中，

\[
\operatorname{Mult}(V_1,\ldots,V_k;W)
\simeq\operatorname{Hom}(V_1\otimes\cdots\otimes V_k,W)
\simeq W\otimes V_1^*\otimes\cdots\otimes V_k^*.
\]

一个 \((r,s)\) 型张量属于 \(V^{\otimes r}\otimes(V^*)^{\otimes s}\)。上指标记录向量槽，下指标记录对偶槽；形状相同不提供这些识别。复空间的线性对偶与共轭对偶要分开，Hermitian 配对含共轭，不能用普通双线性配对代替。

**操作。** 对双线性层写

\[
y^a=T^a{}_{ij}x^iz^j,
\qquad \widetilde T:V\otimes U\to W.
\]

把 \((i,j)\) 展平可获得 \(\widetilde T\) 的矩阵，但这种展平没有把原函数变成对拼接向量 \((x,z)\) 的线性函数。选择一组输出向量和输入对偶基后，数组条目才有含义。

**判据与例子。** 对每个槽都检查可加性及对标量域 \(\mathbb F\) 的齐次性，才可使用上式；只有可加性并不足以保证 \(\mathbb F\)-多线性。\(F(x)=\operatorname{softmax}(Ax)\) 不满足这个条件，但对实向量空间上的 \(C^k\) 映射，\(D^kF(x)\in W\otimes\operatorname{Sym}^k(V^*)\)。复向量空间上只有复可微／全纯情形才能作相同的复多线性声明；仅实光滑的 \(F\) 应使用实化空间，或保留 \(z,\bar z\) 混合槽。例如 \(F(z)=\bar z\) 满足实线性却不满足复线性。无限维时完成张量积、连续性和所用交叉范数另行声明。[复数微分 C8](tensor-calculus-and-autodiff.md#c8-复数微分)给出相应传播规则。

**阅读。** Etingof 等，§2.11–2.12；微分对象见 [导数与割线](derivatives-and-secants.md)。

## TA02 收缩、迹、偏迹与合法复合

**解决什么。** 将轴消去解释为配对，检查算子复合是否类型正确，区分全迹与某组子系统上的迹。

自然配对是 \(V^*\otimes V\to\mathbb F\)，\(\alpha_iv^i\)。它不需要度量。若 \(A:V\to W\)、\(B:W\to U\)，

\[
(BA)^c{}_i=B^c{}_aA^a{}_i.
\]

只有 \(A:V\to V\) 才有不依赖附加识别的 \(\operatorname{tr}A=A^i{}_i\)。对于 \(T\in\operatorname{End}(V\otimes U)\)，

\[
(\operatorname{tr}_U T)^i{}_j=T^{i\alpha}{}_{j\alpha},
\quad
\operatorname{tr}_U(A\otimes B)=(\operatorname{tr}B)A.
\]

**可计算判据。** 在 \(U\) 内做同一基变换，偏迹结果不变；在 \(V\) 内换基，结果按 \(V\) 上的端同态变换。若把两个下指标直接相加，确认这是数组约定还是隐含了 \(g^{ij}\)。

**例子与边界。** 对协方差的部分变量取迹与对注意力的 token 轴求和不是同一个数学操作；后者通常只是指定数组轴的归约。\(A_{ij}B_{ij}\) 在实欧氏正交基中可代表 Frobenius 内积，一般基中对两个协变二槽对象须用相应逆度量收缩；复数 Hermitian 内积还须共轭第一个对象。张量阶数、矩阵秩和 CP 秩分别回答不同问题。

**阅读。** Etingof 等，§2.11、§4.4；坐标与数组收缩见 [类型与指标](types-and-indices.md)。

## TA03 置换、对称化与反对称化

**解决什么。** 利用交换对称压缩自由度、隔离某类响应，并验证导数、核和物理张量的槽对称性。

对同类型的 \(k\) 个槽定义置换作用 \(P_\pi\)。在特征为零的域上，

\[
\operatorname{Sym}T=\frac1{k!}\sum_{\pi\in S_k}P_\pi T,
\qquad
\operatorname{Alt}T=\frac1{k!}\sum_{\pi\in S_k}\operatorname{sgn}(\pi)P_\pi T.
\]

圆括号与方括号采用上述含阶乘的归一化。例如

\[
T_{(ij)}=\tfrac12(T_{ij}+T_{ji}),\quad
T_{[ij]}=\tfrac12(T_{ij}-T_{ji}).
\]

**判据。** 两个算子都幂等；在由正定内积诱导的张量积内积下，它们是正交投影。逐个交换生成元检查即可验证完全对称或完全反对称，不必枚举 \(k!\) 项。不同空间的槽只能在明确给出交换目标空间后置换，不能默认交换后还是原类型。

**例子。** 二阶张量分成对称与反对称两部分；三阶以上还存在混合 Young 对称分量，因此一般 \(T\ne\operatorname{Sym}T+\operatorname{Alt}T\)。光滑实函数在固定向量空间上的 Hessian 对称，但带挠率联络的协变 Hessian 可有反对称部分。

**阅读。** Etingof 等，§2.12、§5.12–5.19；Meinrenken《Differentiable Manifolds》§4.2。

## TA04 外代数、楔积与可分解性

**解决什么。** 表示有向面积、多维体积、交替多线性响应和二向量作用空间。

\(\Lambda^kV^*\) 是交替 \(k\) 形式空间，\(\dim\Lambda^kV^*=\binom nk\)。采用标准微分形式归一化，

\[
\alpha=\frac1{p!}\alpha_{i_1\ldots i_p}
dx^{i_1}\wedge\cdots\wedge dx^{i_p},
\qquad
\alpha\wedge\beta=\frac{(p+q)!}{p!q!}\operatorname{Alt}(\alpha\otimes\beta).
\]

所以 \(\alpha\wedge\beta=(-1)^{pq}\beta\wedge\alpha\)，而奇数阶形式有 \(\alpha\wedge\alpha=0\)。向量的内乘为

\[
(\iota_X\alpha)_{i_2\ldots i_p}=X^{i_1}\alpha_{i_1i_2\ldots i_p},
\quad
\iota_X(\alpha\wedge\beta)=\iota_X\alpha\wedge\beta+(-1)^p\alpha\wedge\iota_X\beta.
\]

**可计算判据。** 非零二形式 \(B\) 可分解成 \(u^*\wedge v^*\)，当且仅当 \(B\wedge B=0\)，等价于其反对称矩阵秩为 2。更高阶可分解性由相应 Plücker 关系判断，不能仅看一个展平秩。

**例子与反例。** \(dx^1\wedge dx^2\) 是一个二维平面的面积形式；\(dx^1\wedge dx^2+dx^3\wedge dx^4\) 的平方为非零四形式，不能代表单一平面。四阶曲率因每对槽反对称而自然作用在 \(\Lambda^2\)，普通 Hessian 的对称输入槽则自然作用在 \(\operatorname{Sym}^2\)。

**阅读。** Meinrenken《Differentiable Manifolds》§4.1–4.5；Bär §6.1、§6.5。

## TA05 度量、升降指标与伴随

**解决什么。** 明确长度、角度、梯度、谱与奇异值使用的几何，避免把相同数组下的不同对象混为一谈。

非退化双线性度量给出 \(\flat:V\to V^*\)、\(v_i=g_{ij}v^j\)，及其逆 \(\sharp\)。对实空间映射 \(A:V\to W\)，

\[
A^\dagger=G_V^{-1}A^\mathsf T G_W,
\qquad
g_W(Au,w)=g_V(u,A^\dagger w).
\]

在正定度量下，广义奇异值的平方是

\[
A^\mathsf T G_W A v=\sigma^2G_Vv
\]

的本征值。实现时可用 Cholesky 白化后做普通 SVD；不定度量下不能从自伴直接推断实谱、正范数或正交可对角化。

复空间中另声明使用双线性还是 Hermitian 结构。若 \(h(u,v)=u^HGv\)（第二槽线性），则 \(v\mapsto h(v,\cdot)\in V^*\) 是共轭线性的，且 \(A^\dagger=G_V^{-1}A^HG_W\)。它不同于不带共轭的线性对偶拉回 \(\alpha\mapsto\alpha\circ A\)，后者在对偶基中用 \(A^T\)。复双线性曲率配对、Hermitian 能量与实损失梯度不得共用未经声明的升降规则。

**体积对象。** 定向坐标下体积形式为

\[
\mathrm{vol}_g=\sqrt{|\det g|}\,dx^1\wedge\cdots\wedge dx^n,
\qquad
\varepsilon_{i_1\ldots i_n}=\sqrt{|\det g|}[i_1\ldots i_n].
\]

\([i_1\ldots i_n]\) 是数值置换符号；单独把它当任意坐标中的普通张量会遗漏行列式因子。Hodge 星还需要定向，详见 [微分几何 DG06](differential-geometry.md)。

**判据与反例。** 同时变换坐标、度量和 \(A\)，检查配对值及广义奇异值保持。复对称 \(Q^\mathsf T=Q\) 不等于 Hermitian \(Q^*=Q\)；\(\begin{psmallmatrix}1&i\\i&-1\end{psmallmatrix}\) 非零而平方为零。

**阅读。** Meinrenken《Riemannian Geometry》§12、§15；Bär §6.2–6.4。

## TA06 对称张量、多项式与极化

**解决什么。** 在高阶导数、矩、能量和多项式响应中，用方向作用重建对称多线性信息。

对 \(T\in\operatorname{Sym}^k(V^*)\)，令 \(p(v)=T(v,\ldots,v)\)。实数域的极化恒等式给出

\[
T(v_1,\ldots,v_k)=\frac1{2^kk!}
\sum_{\epsilon_i\in\{-1,1\}}
\Big(\prod_i\epsilon_i\Big)
p\Big(\sum_i\epsilon_i v_i\Big).
\]

二次情形是 \(2T(u,v)=p(u+v)-p(u)-p(v)\)。对 \(f\) 的 \(k\) 阶 Taylor 项，系数是 \(D^kf/k!\)，不要重复或漏掉阶乘。

**计算判据。** 对选定方向比较混合方向作用与极化结果；利用对称性只存独立分量。重复方向值只能确定完全对称部分：对任意 \(T\)，\(T(v,\ldots,v)=(\operatorname{Sym}T)(v,\ldots,v)\)。因此该方法不能恢复任意非对称张量。

**例子。** 二次型只保留矩阵对称部分，\(x^\mathsf T A x=x^\mathsf T(A+A^\mathsf T)x/2\)。高阶激活响应可用方向多项式分析，但其对角取值不能揭示反对称通道。

**阅读。** Etingof 等 §2.12、Lemma 5.18.3；Absil–Mahony–Sepulchre §5.5 的 Hessian 极化。

## TA07 Young 对称性与代数曲率投影

**解决什么。** 处理“部分对称、部分反对称”的高阶张量，构造满足某类代数恒等式的分量。

对 Young 表 \(t\)，令 \(a_t\) 为同行置换的和，\(b_t\) 为同列置换的带符号和，固定顺序 \(y_t=a_tb_t\)。在特征零下适当按钩长积归一化可得到幂等元。它通常不是给定欧氏数组内积下的正交投影；不同表的幂等元也不应未经验证就当成互相正交的完整分解。

一个可直接实现的例子是任意四槽协变数组的代数曲率投影：

\[
A_{abcd}=T_{[ab][cd]},\quad
B_{abcd}=\tfrac12(A_{abcd}+A_{cdab}),\quad
R_{abcd}=B_{abcd}-B_{[abcd]}.
\]

这样得到每对反对称、成对交换对称和代数 Bianchi 恒等式。再按 DG05 去掉 Ricci 与标量迹即可得到代数 Weyl 部分。

**验算。** 检查 \(R_{abcd}+R_{acdb}+R_{adbc}=0\)、\(R_{abcd}=R_{cdab}\) 及再次投影结果不变。\(n\) 维代数曲率空间维数是 \(n^2(n^2-1)/12\)，可作为独立参数计数检查。

**使用边界。** 投影定义了原对象的一个分量，并不证明它来自所选度量的 Levi-Civita 曲率，也不证明该分量控制网络误差。若原 Hessian 在待反对称的槽上对称，投影变零只说明投影不匹配。

**阅读。** Etingof 等 §5.12–5.19；Bär §6.1；微分可实现性需另看 [微分几何](differential-geometry.md)。

## TA08 群作用、轨道、稳定子与不变量

**解决什么。** 在讨论“同一个算子”“内在特征”或分类之前，明确允许改变什么。

输入输出坐标改变 \(x'=Sx,y'=Ry\) 时，线性映射是

\[
A'=RAS^{-1},
\]

协变二次型是 \(G'=S^{-\mathsf T}GS^{-1}\)。只有输入输出已识别且使用同一变换，才是相似变换 \(A'=SAS^{-1}\)。张量群作用在每个向量槽用对应表示、每个对偶槽用逆转置表示。

**轨道判据。** \(\mathcal O_T=\{g\cdot T:g\in G\}\)，稳定子 \(G_T=\{g:g\cdot T=T\}\)。光滑李群作用下，轨道切空间为 \(\{X\cdot T:X\in\mathfrak g\}\)，所以可通过求线性系统 \(X\cdot T=0\) 得到稳定子李代数，并计算

\[
\dim\mathcal O_T=\dim G-\dim G_T.
\]

**例子。** 普通相似下的迹、行列式和 Jordan 结构是不变量；一般左右独立可逆变换只保留矩阵秩等较粗信息。欧氏奇异值在左右正交作用下不变。\(\operatorname{tr}A^k\) 等标量不变量不能区分所有非半单轨道：非零幂零矩阵与零矩阵的所有这些迹都为零。

**ML 条件。** 被动隐藏坐标改变必须同时改变非线性：\(\phi'=S\circ\phi\circ S^{-1}\)。原实现的逐坐标 \(\phi\) 通常不满足 \(\phi(Sz)=S\phi(z)\)，因此隐藏旋转不是自动成立的网络对称性。

**阅读。** Meinrenken《Group actions on manifolds》§1.1–1.5；Etingof 等 §2.3、§2.14。

## TA09 等变映射与群平均

**解决什么。** 推导参数共享与合法通道耦合，把对称约束转成可求解的系数空间。

给定表示 \(\rho_V,\rho_W\)，\(F\) 等变要求

\[
F(\rho_V(g)x)=\rho_W(g)F(x).
\]

线性层的条件是 \(A\rho_V(g)=\rho_W(g)A\)。向量化后得到线性系统；有限生成群只需检查生成元。对连通李群，可检查李代数生成元

\[
A\,d\rho_V(X)=d\rho_W(X)A.
\]

不连通群还需检查其它连通分支的代表。多线性层把输入表示替换为 \(\rho_{V_1}\otimes\cdots\otimes\rho_{V_k}\)。

有限群或紧群的归一化 Haar 测度给出 Reynolds 投影：

\[
\mathcal R(A)=\int_G\rho_W(g)A\rho_V(g)^{-1}\,dg.
\]

它把任意 \(A\) 投到 intertwiner 空间，满足 \(\mathcal R^2=\mathcal R\)。使用不变正定内积时该投影正交；非紧群一般没有可直接用来这样平均的有限 Haar 概率测度。

**例子与验算。** 对 \(n\ge2\) 的坐标置换表示，所有等变线性层形如 \(aI+b\mathbf1\mathbf1^\mathsf T\)。\(SO(3)\) 向量到向量的线性等变层只有标量倍恒等；非线性 \(F(x)=h(\|x\|^2)x\) 等变，逐坐标 ReLU 一般不等变。数值抽样残差可发现实现错误，但证明需满足表示约束本身。

**阅读。** Etingof 等 §2.3、§4.1、§4.4–4.7；Meinrenken《Group actions》§1 的紧群不变结构。

## TA10 不可约分解、Schur 引理与耦合

**解决什么。** 将对称性转成算子分块、独立参数数目和允许的张量积通道。

对完全可约的有限维复表示，写

\[
V=\bigoplus_\lambda \mathbb C^{m_\lambda}\otimes V_\lambda,
\qquad
W=\bigoplus_\lambda \mathbb C^{n_\lambda}\otimes V_\lambda.
\]

若 \(V_\lambda\) 两两非同构不可约，则

\[
\operatorname{Hom}_G(V,W)
\simeq\bigoplus_\lambda
\operatorname{Hom}(\mathbb C^{m_\lambda},\mathbb C^{n_\lambda})\otimes I_{V_\lambda}.
\]

因此复维数为 \(\sum_\lambda m_\lambda n_\lambda\)。这是对称性约束的参数数目，不等于每个参数实例的矩阵秩。Schur 的“不可约自同态为标量”需要代数闭域及有限维；实不可约表示的对易代数还可能是 \(\mathbb C\) 或四元数。

**可计算工具。** 有限群的等型分量投影为

\[
P_\lambda=\frac{d_\lambda}{|G|}\sum_{g\in G}
\overline{\chi_\lambda(g)}\rho(g),
\]

张量积重数由字符内积计算；紧群把和换成 Haar 积分。Clebsch–Gordan 系数把张量积基变到不可约基，它们的归一化及实/复基约定需要固定。

**具体例子。** 三维向量的二重张量积在 \(SO(3)\) 下分成维数 \(1+3+5\)：迹、反对称部分、对称无迹部分。

\[
T_{ij}=\tfrac13\delta_{ij}T_{kk}
+\tfrac12(T_{ij}-T_{ji})
+\left[\tfrac12(T_{ij}+T_{ji})-\tfrac13\delta_{ij}T_{kk}\right].
\]

用 \(\varepsilon_{kij}\) 把反对称部分映为三分量对象时，它在 \(O(3)\) 下是轴向量；忽略反射奇偶性会得到错误等变层。Schur–Weyl 对偶进一步把 \(V^{\otimes k}\) 的 \(GL(V)\) 与槽置换 \(S_k\) 结构同时组织起来。

**阅读。** Etingof 等 Proposition 2.3.9、Corollary 2.3.10，§3.1、§4.5–4.9、§5.18–5.19。

## TA11 应力、弹性张量与 Mandel/Kelvin 表示

**解决什么。** 在连续介质中分析四阶算子的真实作用空间、谱与能量；为“高阶张量可以变成小矩阵”提供有物理含义的例子。

以下取三维欧氏空间的正交基，小应变、无偶应力的线弹性。应变 \(\epsilon\) 和 Cauchy 应力 \(\sigma\) 都在 \(\operatorname{Sym}^2(V^*)\) 中，借度量识别槽后

\[
\epsilon_{ij}=\tfrac12(\partial_i u_j+\partial_j u_i),
\quad \sigma_{ij}=C_{ijkl}\epsilon_{kl},
\quad E(\epsilon)=\tfrac12\epsilon_{ij}C_{ijkl}\epsilon_{kl}.
\]

\(C_{ijkl}=C_{jikl}=C_{ijlk}\) 是 minor symmetries。若本构由二次能量势导出，即 \(\sigma=\partial E/\partial\epsilon\)、\(C=D^2E\)，才由 Hessian 对称性得到 major symmetry \(C_{ijkl}=C_{klij}\)。仅写出标量 \(E=\frac12\epsilon:C:\epsilon\) 不够：它只看到 \(C\) 在 \(\operatorname{Sym}^2\) 上的自伴部分，其导数一般为 \(\frac12(C+C^\dagger)\epsilon\)。满足上述保守条件时，\(C\) 是六维对称二阶张量空间上的自伴算子，至多 21 个独立参数。一般有限变形切线、非保守本构或不同应力度量不能直接继承全部假设。

**压缩表示的度量。** 固定顺序 \((11,22,33,23,13,12)\)。原始六元坐标

\[
v(T)=(T_{11},T_{22},T_{33},T_{23},T_{13},T_{12})^\mathsf T
\]

满足 \(T:S=v(T)^\mathsf T\operatorname{diag}(1,1,1,2,2,2)v(S)\)。Mandel/Kelvin 的正交坐标为

\[
m(T)=D v(T),\qquad
D=\operatorname{diag}(1,1,1,\sqrt2,\sqrt2,\sqrt2),
\quad T:S=m(T)^\mathsf Tm(S).
\]

其六个基分别为三个 \(e_i\otimes e_i\) 和三个 \((e_i\otimes e_j+e_j\otimes e_i)/\sqrt2\)。\(C^M_{AB}=E_A:C:E_B\) 给出 \(m(\sigma)=C^Mm(\epsilon)\)，可以在普通六维欧氏内积下求真实算子谱和检查能量正定。

**与工程 Voigt 的转换。** 若 \(\sigma^V=v(\sigma)\)，\(\epsilon^V=D^2v(\epsilon)\) 采用工程剪应变，\(\sigma^V=C^V\epsilon^V\)，则

\[
C^M=DC^VD,
\qquad \sigma:\epsilon=(\sigma^V)^\mathsf T\epsilon^V.
\]

不能把未经说明的六元应变、六元应力和同一个欧氏内积混用。软件中 Voigt 轴顺序也可能不同，应显式记录置换。

**各向同性分解。** 令

\[
\mathbb I_s{}_{ijkl}=\tfrac12(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}),
\quad J_{ijkl}=\tfrac13\delta_{ij}\delta_{kl},\quad K=\mathbb I_s-J.
\]

则 \(J^2=J,K^2=K,JK=0\)，且

\[
C=3\kappa J+2\mu K,
\quad \kappa=\lambda+\tfrac23\mu,
\quad C_{ijkl}=\lambda\delta_{ij}\delta_{kl}
+\mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}).
\]

**可计算判据。** 严格正能量要求 \(\mu>0,\kappa>0\)；Mandel 矩阵的谱为体积通道 \(3\kappa\) 和五维偏量通道 \(2\mu\)。逆算子是 \(C^{-1}=(3\kappa)^{-1}J+(2\mu)^{-1}K\)。仅检查若干坐标方向的正应力不足以保证正定。纯剪切 \(\epsilon_{12}=a\) 有 \(\epsilon:\epsilon=2a^2\)，可一行检出遗漏的剪切权重。

**与 Petrov 的区别。** 弹性 \(C\) 作用于 \(\operatorname{Sym}^2\)，Weyl 作用于 \(\Lambda^2\)；二者都有四槽，依据的对称性、度量与分类群不同。

**阅读。** López-Pamies《Mechanics of Solids》§4.3、式 (4.39)–(4.54)；Helnwein 2001 的压缩表示空间解释，全文阅读状态见来源台账。

## 选用与迁移

先声明对象与问题，再选能减少未知量或提供判据的工具。要证明特定激活对称性，用 TA09；要分析一个能量四阶响应，先看 TA03、TA07 或 TA11 的作用空间；要从网络导数构造几何，先到 [微分几何](differential-geometry.md) 指定度量和联络。把群作用、度量或投影写清之后，标准矩阵表示往往是最便于计算的最终形式。
