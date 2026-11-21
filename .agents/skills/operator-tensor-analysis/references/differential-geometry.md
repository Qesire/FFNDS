# 微分几何、张量场与流形计算

使用这些工具时，先给出流形、所比较的切空间、度量及联络。网络的特征数组不自动提供它们。下文统一取
\(R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z\)，
\(R^a{}_{bcd}Z^bX^cY^d=(R(X,Y)Z)^a\)，
\(R_{abcd}=g_{ae}R^e{}_{bcd}\)。换用其他书的约定时先翻译槽次序和整体符号。来源见 [几何来源台账](sources-geometry.md)。

建立图册、约束子流形、丛或商空间时，先读[微分流形与丛](smooth-manifolds-and-bundles.md)。固定秩、统计模型及跨点有限变化见[流形建模与计算](manifold-modeling-and-computation.md)；全局拓扑、Hamilton 动力学及辛结构见[全局与辛几何](global-and-symplectic-geometry.md)。本文件集中提供度量、联络和曲率的计算接口。

这些条目也按几何概念自身的关系提供学习入口：微分与拉回、联络与运输、曲率与可积性、度量与优化。已有 AI 连接包括 DG11–DG13 的约束优化、表示子空间及正定参数，和 DG01、DG03–DG04 的参数响应与跨点比较。可从其他条目探索新的建模与分析方式；得出目标模型的结论时，再明确结构映射及相应假设。

## 内容

- DG01 切映射、拉回与诱导度量
- DG02 微分形式、外微分与 Cartan 演算
- DG03 联络、协变导数与协变 Hessian
- DG04 平行移动、测地线与 Jacobi 场
- DG05 曲率、Bianchi 恒等式与 Weyl 分解
- DG06 Hodge 星、余微分与 Hodge 分解
- DG07 活动标架、Cartan 结构方程与规范变换
- DG08 Frobenius 可积性、Killing 场与守恒量
- DG09 子流形、第二基本形式与 Gauss 方程
- DG10 度量和曲率的一阶变分
- DG11 流形优化的梯度、Hessian、回缩与运输
- DG12 Stiefel 与 Grassmann 的可计算接口
- DG13 正定矩阵几何、均值与张量场平滑
- DG14 零测地线丛、光学张量与聚焦
- 与算子分析的连接

## DG01 切映射、拉回与诱导度量

**解决什么。** 在数据坐标、潜在坐标与观测空间之间正确传递方向、协向量和距离。

光滑 \(F:M\to N\) 的微分为 \(dF_p:T_pM\to T_{F(p)}N\)。对 \(N\) 上协变 \(k\) 张量，

\[
(F^*T)_p(v_1,\ldots,v_k)
=T_{F(p)}(dF_pv_1,\ldots,dF_pv_k).
\]

所以 \(F^*\alpha\) 总有定义；一般映射只把向量场推成沿 \(F\) 的向量，若要成为 \(N\) 上单值向量场，还需要可投射性等条件。微分同胚可自然推前或拉回所有类型的张量。

**计算。** 若观测空间度量矩阵为 \(H(F(x))\)，

\[
g_{ij}(x)=\partial_iF^a\,H_{ab}(F(x))\,\partial_jF^b.
\]

\(H\) 正定且 \(dF\) 满列秩时，\(g\) 正定；否则可能只是半正定的二次型。协向量的拉回就是 VJP，方向的推前就是 JVP，但“梯度”还需要用度量将协向量升指标。

**例子与判据。** 局部微分同胚 \(F:\mathbb R^d\to\mathbb R^d\) 拉回欧氏度量得到平坦度量；更高维观测空间中的弯曲嵌入可以有内禀曲率。对拟设的潜在度量，先验算 \(v^ig_{ij}v^j=\|dF(v)\|_H^2\) 及局部秩，再解释它在任务中度量什么误差。

**阅读。** Meinrenken《Differentiable Manifolds》§1.3–1.5、§4.5；Tong §2.1.3、§2.2–2.3。

## DG02 微分形式、外微分与 Cartan 演算

**解决什么。** 将梯度、旋度、散度、守恒定律和可积性写成不依赖坐标的运算。

对 \(p\) 形式 \(\alpha\)，

\[
(d\alpha)_{i_0\ldots i_p}=(p+1)\partial_{[i_0}\alpha_{i_1\ldots i_p]},
\quad d^2=0,
\quad d(\alpha\wedge\beta)=d\alpha\wedge\beta+(-1)^p\alpha\wedge d\beta.
\]

外微分不需要度量，也不需要联络。对任何光滑 \(F\)，\(dF^*\alpha=F^*d\alpha\)。向量场的内乘和 Lie 导数满足

\[
\mathcal L_X\alpha=d\iota_X\alpha+\iota_Xd\alpha,
\qquad
[\mathcal L_X,d]=0,
\qquad
[\mathcal L_X,\iota_Y]=\iota_{[X,Y]}.
\]

**可计算判据。** 对光滑一形式 \(\alpha_i dx^i\)，\(d\alpha=0\) 检查局部交叉偏导相等；在可缩邻域上 Poincaré 引理保证正次数闭形式局部恰当。全局势还受拓扑约束。Stokes 定理

\[
\int_Md\alpha=\int_{\partial M}\alpha
\]

这里采用定向 \(n\) 维流形与 \(\alpha\in\Omega_c^{n-1}(M)\) 的版本，边界取诱导定向；紧致 \(M\) 上光滑形式自动紧支撑。非紧且无紧支撑时，须另证穷竭过程中的额外边界通量消失等条件，不能只凭积分存在就删去这些条件。例如 \(M=(0,1)\)、\(\alpha=x\) 时，\(\alpha,d\alpha\) 均绝对可积，但 \(\int_Md\alpha=1\)，而该无边界流形的 \(\partial M\) 为空。

**例子与反例。** 圆周上的角一形式局部是 \(d\theta\)，沿整圈积分非零，因此不可能是全局单值函数的微分。\(d^2=0\) 是连续微分结构事实；两个无关数值差分算子未必继承它。网格/复形上用相容的 coboundary 矩阵可代数地保证相邻外微分复合为零。

**阅读。** Meinrenken《Differentiable Manifolds》§4.2–4.5、§4.9、§5；Tong §2.4。

## DG03 联络、协变导数与协变 Hessian

**解决什么。** 比较不同点上的向量、使高阶微分具有张量变换律，并区分坐标变化和真实几何变化。

对 \((r,s)\) 张量，协变导数在每个上槽加一个 \(\Gamma\) 项、每个下槽减一个 \(\Gamma\) 项。例如

\[
\nabla_iT^a{}_b=\partial_iT^a{}_b
+\Gamma^a{}_{ic}T^c{}_b-\Gamma^c{}_{ib}T^a{}_c.
\]

Levi-Civita 联络由无挠率及 \(\nabla g=0\) 唯一确定：

\[
\Gamma^k{}_{ij}=\tfrac12g^{k\ell}
(\partial_i g_{j\ell}+\partial_jg_{i\ell}-\partial_\ell g_{ij}).
\]

\(\Gamma\) 自身不是张量；两个联络的差是 \((1,2)\) 张量。挠率为 \(T(X,Y)=\nabla_XY-\nabla_YX-[X,Y]\)。

**Hessian。** 标量函数的协变 Hessian 是

\[
(\nabla^2f)_{ij}=\partial_i\partial_jf-\Gamma^k{}_{ij}\partial_kf.
\]

无挠率时对称。一般有 \(\nabla^2f(X,Y)-\nabla^2f(Y,X)=-df(T(X,Y))\)。对流形间映射 \(F:M\to N\)，其 \(\nabla dF\) 还含输出联络：

\[
(\nabla dF)^a{}_{ij}
=\partial_i\partial_jF^a-\Gamma^k{}_{ij}\partial_kF^a
+\widetilde\Gamma^a{}_{bc}(F)\partial_iF^b\partial_jF^c.
\]

**检查例。** \(x=z^2,z>0,f(x)=x\)。在 \(z\) 坐标中 \(\partial_z^2f=2\)，欧氏度量拉回为 \(4z^2dz^2\)，\(\Gamma^z{}_{zz}=1/z\)，所以 \(\nabla^2f=2-(1/z)(2z)=0\)。原始二阶偏导的变化不是曲率证据。

**阅读。** Meinrenken《Riemannian Geometry》§15、§19；Tong §3.2.1–3.2.3；Absil 等 §5.3、§5.5。

## DG04 平行移动、测地线与 Jacobi 场

**解决什么。** 将不同点的方向放在可比较的空间中，构造局部直线和曲率敏感的偏离方程。

沿曲线 \(\gamma\)，平行场满足

\[
\dot V^a+\Gamma^a{}_{bc}(\gamma)\dot\gamma^bV^c=0.
\]

Levi-Civita 平行移动保存内积；不同路径一般给出不同结果。测地线满足

\[
\ddot\gamma^a+\Gamma^a{}_{bc}\dot\gamma^b\dot\gamma^c=0,
\quad\operatorname{Exp}_p(v)=\gamma_v(1).
\]

\(\operatorname{Log}_p\) 是合适正常邻域中的局部逆；跨越 cut locus 时可能多值，不能默认全局唯一。

**曲率作用。** 对一族仿射参数测地线的变分场 \(J\)，

\[
\nabla_{\dot\gamma}^2J+R(J,\dot\gamma)\dot\gamma=0.
\]

这是可测的方向聚焦/散开判据；它比单看坐标 Christoffel 的大小更有内在意义。

**例子与验算。** 单位球面的测地线是大圆；两个反足点间最短测地线不唯一。数值计算应核验 \(g(\dot\gamma,\dot\gamma)\) 守恒及平行运输的内积守恒。正定情形的局部最短性质不能原样用于 Lorentz 度量的所有因果类型。

**阅读。** Meinrenken《Riemannian Geometry》§15–17；Tong §3.3；Absil 等 §5.4。

## DG05 曲率、Bianchi 恒等式与 Weyl 分解

**解决什么。** 检测内禀非平坦性，分离迹信息与共形信息，并为代数分类构造正确对象。

按本文件约定，

\[
R^a{}_{bcd}=\partial_c\Gamma^a{}_{db}-\partial_d\Gamma^a{}_{cb}
+\Gamma^a{}_{ce}\Gamma^e{}_{db}-\Gamma^a{}_{de}\Gamma^e{}_{cb}.
\]

Levi-Civita 情况有

\[
R_{abcd}=-R_{bacd}=-R_{abdc}=R_{cdab},
\quad R^a{}_{[bcd]}=0,
\quad\nabla_{[e}R^a{}_{|b|cd]}=0.
\]

Ricci 与标量曲率为 \(\operatorname{Ric}_{bd}=g^{ac}R_{abcd}\)，\(s=g^{bd}\operatorname{Ric}_{bd}\)。收缩 Bianchi 给出 \(\nabla^a(\operatorname{Ric}_{ab}-\tfrac12sg_{ab})=0\)。

**可实现分解。** \(n\ge3\) 时定义 Schouten 张量

\[
P_{ab}=\frac1{n-2}\left(\operatorname{Ric}_{ab}-\frac{s}{2(n-1)}g_{ab}\right),
\]

则

\[
C_{abcd}=R_{abcd}
-(g_{ac}P_{bd}-g_{ad}P_{bc}-g_{bc}P_{ad}+g_{bd}P_{ac}).
\]

验算 \(g^{ac}C_{abcd}=0\) 和重构 \(R\)。\(n\ge4\) 时 \(C=0\) 等价于局部共形平坦；\(n=3\) 的 Weyl 恒为零，应检查 Cotton 张量 \(\nabla_cP_{ab}-\nabla_bP_{ac}\)。二维需使用单独的标量曲率公式。

**不变量与例子。** 常截面曲率 \(K\) 的空间满足 \(R_{abcd}=K(g_{ac}g_{bd}-g_{ad}g_{bc})\)、\(\operatorname{Ric}=(n-1)Kg\)、\(C=0\)。所以 Weyl 为零不等于曲率为零。共形缩放 \(\widehat g=e^{2\omega}g\) 下，\(C^a{}_{bcd}\) 不变，而全下指标版本按 \(e^{2\omega}\) 缩放。

**阅读。** Meinrenken《Riemannian Geometry》§18；Tong §3.2、§3.4；Bär §6.1–6.3。Petrov 判别见 [几何分类](geometric-classification.md)。

## DG06 Hodge 星、余微分与 Hodge 分解

**解决什么。** 将 \(k\) 形式与余维形式联系起来，构造自对偶分解、Laplace 算子和全局势的分解。

在定向、非退化 \(n\) 维度量上，由

\[
\alpha\wedge *\beta=\langle\alpha,\beta\rangle_g\,\mathrm{vol}_g
\]

定义 Hodge 星。若度量有 \(q\) 个负方向，

\[
*^2|_{\Lambda^k}=(-1)^{k(n-k)+q}I.
\]

例如四维 Lorentz 的二形式有 \(*^2=-I\)，需复化才能用 \(\pm i\) 扇区；四维正定及 \((2,2)\) 签名有 \(*^2=I\)，可在实空间用 \(\pm1\) 扇区。

**正定情形的分析工具。** 在 \(k\) 形式上

\[
\delta=(-1)^{n(k+1)+1}*d*,
\quad\Delta_H=d\delta+\delta d.
\]

这是非负的 Hodge Laplacian，标量上等于 \(-\nabla^i\nabla_i\)。在紧、定向、无边界的 Riemannian 流形上，

\[
\Omega^k(M)=d\Omega^{k-1}(M)\oplus\delta\Omega^{k+1}(M)\oplus\mathcal H^k(M),
\quad\mathcal H^k=\ker\Delta_H\simeq H^k_{\mathrm{dR}}(M).
\]

边界情形需要边界条件；非紧情形要指定函数空间及闭值域条件。Lorentz 情形不具有同一椭圆正定理论。

**验算与反例。** 检查星的两次作用符号和 \(\langle\Delta_H\alpha,\alpha\rangle=\|d\alpha\|^2+\|\delta\alpha\|^2\)。环面上的非零调和一形式可既闭又余闭，因此“局部无旋且无散”不推出全局为零。

**阅读。** Bär §6.2；Tong §3.1.4；Auroux，MIT 18.966 Lecture 15。

## DG07 活动标架、Cartan 结构方程与规范变换

**解决什么。** 把复杂坐标中的曲率计算化为低阶微分形式的收缩，并跟踪局部基选择。

取局部标架 \(e_a\) 与对偶 coframe \(\theta^a\)，定义 \(\nabla e_b=\omega^a{}_b\otimes e_a\)。Cartan 方程为

\[
T^a=d\theta^a+\omega^a{}_b\wedge\theta^b,
\qquad
\Omega^a{}_b=d\omega^a{}_b+\omega^a{}_c\wedge\omega^c{}_b,
\]

\[
\Omega^a{}_b=\tfrac12R^a{}_{bcd}\theta^c\wedge\theta^d.
\]

正交标架中的 metric compatibility 是 \(\omega_{ab}=-\omega_{ba}\)；Levi-Civita 时 \(T^a=0\)。非坐标标架满足 \(d\theta^a=-\tfrac12c^a{}_{bc}\theta^b\wedge\theta^c\)，不得把这项漏掉。

**向量丛版本。** 在局部纤维基下 \(\nabla=d+A\)、\(F=dA+A\wedge A\)。若基变成 \(e'=eS\)，

\[
A'=S^{-1}AS+S^{-1}dS,
\qquad F'=S^{-1}FS.
\]

联络多出的导数项正好保证曲率的协变性；Bianchi 是 \(d_AF=0\)。

**例子与判据。** 平面极坐标的正交 coframe 是 \(\theta^1=dr,\theta^2=r\,d\varphi\)。取 \(\omega^2{}_1=d\varphi,\omega^1{}_2=-d\varphi\)，结构方程无挠率且 \(\Omega=0\)。联络分量非零来自基旋转，不能单凭其数值判断曲率。

**阅读。** Tong §3.4.2、§3.4.4；Meinrenken《Riemannian Geometry》§19。

## DG08 Frobenius 可积性、Killing 场与守恒量

**解决什么。** 判断一组局部方向能否组成子流形，以及某方向是否产生保持度量的变换。

常秩分布 \(\mathcal D\subset TM\) 局部可积，当且仅当其截面对 Lie bracket 闭合。余维一且 \(\alpha\ne0\) 时，

\[
\mathcal D=\ker\alpha\text{ 可积}\quad\Longleftrightarrow\quad\alpha\wedge d\alpha=0.
\]

Killing 场满足

\[
\mathcal L_Kg=0\quad\Longleftrightarrow\quad\nabla_iK_j+\nabla_jK_i=0.
\]

沿仿射参数测地线，\(g(K,\dot\gamma)\) 守恒。若单位非零范数向量场 \(u\) 满足 \(g(u,u)=\epsilon\in\{-1,1\}\)，令 \(a=\nabla_u u\)。局部条件

\[
\nabla_i u_j=\epsilon u_i a_j,\qquad da^\flat=0
\]

保证存在局部因子 \(f\) 使 \(K=fu\) 是超曲面正交 Killing 场，其中 \(d\log f=-\epsilon a^\flat\)。全局因子还需要相应闭形式恰当。

**例子与判据。** 平面上的旋转场是 Killing 场，但其轨线不是一般位置的测地线；二者不能等同。检查 Killing 方程、Frobenius 三形式和所需积分周期，比凭坐标是否显式出现某变量更可靠。

**阅读。** Meinrenken 的向量场与群作用讲义；Tong §4.3；Ferrando–Morales–Sáez §5 将这些判据与 Weyl 提取的方向结合。

## DG09 子流形、第二基本形式与 Gauss 方程

**解决什么。** 区分特征嵌入的外在弯曲和诱导度量的内在曲率。

对浸入 \(F:M^d\to\mathbb R^D\)，

\[
g_{ij}=\langle\partial_iF,\partial_jF\rangle,
\qquad
\mathrm{II}_{ij}=P_N\partial_i\partial_jF.
\]

其中 \(P_N\) 是法空间正交投影；切向部分由 Levi-Civita 联络吸收。欧氏环境中的 Gauss 方程是

\[
R_{ijkl}=\langle\mathrm{II}_{ik},\mathrm{II}_{jl}\rangle
-\langle\mathrm{II}_{il},\mathrm{II}_{jk}\rangle.
\]

**可计算判据。** 用 \(J\) 的 QR/SVD 找切空间及法投影，独立从 \(g\) 和 \(\mathrm{II}\) 两条路径计算曲率。满秩且 \(D=d\) 的局部微分同胚没有法方向，拉回欧氏度量必平坦。

**例子。** 圆柱有非零第二基本形式但高斯曲率为零；球面有非零内禀曲率。把非线性特征映射的 Hessian 范数直接命名为“内禀曲率”会混淆这两种对象。非浸入点的降秩还会使上述 Riemannian 构造退化。

**阅读。** Tong §3.3 的内外几何区分；Absil 等 §3.3、§5.3.3 及 Proposition 5.3.2 的嵌入子流形联络。Gauss 公式按本文件曲率约定给出，可由切/法分解直接核验。

## DG10 度量和曲率的一阶变分

**解决什么。** 对几何模型的度量参数求灵敏度，线性化曲率与几何约束。

令 \(g(t)=g+th+O(t^2)\)，在 \(t=0\) 处

\[
\dot g^{ij}=-g^{ia}h_{ab}g^{bj},
\qquad
\frac{d}{dt}\mathrm{vol}_{g(t)}=\tfrac12\operatorname{tr}_g(h)\,\mathrm{vol}_g.
\]

Levi-Civita 联络的变分是张量：

\[
\dot\Gamma^k{}_{ij}=\tfrac12g^{k\ell}
(\nabla_i h_{j\ell}+\nabla_jh_{i\ell}-\nabla_\ell h_{ij}),
\]

\[
\dot{\operatorname{Ric}}_{ij}
=\nabla_k\dot\Gamma^k{}_{ji}-\nabla_j\dot\Gamma^k{}_{ki},
\qquad
\dot s=-\langle h,\operatorname{Ric}\rangle_g
+\nabla_i\nabla_jh^{ij}-\nabla^i\nabla_i\operatorname{tr}_g h.
\]

**验算与边界。** 先在常数缩放 \(h=cg\) 下核验 \(\dot\Gamma=0\)、\(\dot s=-cs\)。对由微分同胚产生的 \(h=\mathcal L_Xg\)，相应几何对象变化应为其 Lie 导数，不能解释为新几何。分部积分求目标函数变分时保留边界项或说明边界条件。

**阅读。** Bär §5.1；Tong §4.1。这里仅提供几何线性化接口，不替代原文关于约束 PDE 的存在性论证。

## DG11 流形优化的梯度、Hessian、回缩与运输

**解决什么。** 在张量秩、正交性、子空间或正定性等约束上更新参数，同时保持导数的空间类型。

度量梯度由

\[
g_x(\operatorname{grad}f,\eta)=df_x[\eta]
\]

定义。在坐标中 \(\operatorname{grad}f=G^{-1}\partial f\)；对带诱导度量的欧氏嵌入子流形，\(\operatorname{grad}f=P_{T_xM}\nabla\bar f\)。Riemannian Hessian 是

\[
\operatorname{Hess}f(x)[\eta]=\nabla_\eta\operatorname{grad}f,
\]

在诱导度量下可写 \(P_{T_xM}D(\operatorname{grad}f)[\eta]\)。这里只投影欧氏 Hessian 一次一般会遗漏投影算子本身的导数。

**回缩与运输。** 回缩要求 \(R_x(0)=x,DR_x(0)=I\)，更新用 \(x_+=R_x(\eta)\)。向量运输 \(\mathcal T_\eta:T_xM\to T_{R_x(\eta)}M\) 是随 \((x,\eta)\) 光滑、对被运输向量线性且满足 \(\mathcal T_0=I\) 的映射。它负责把动量、共轭方向等迁到新的切空间，不自动等于平行移动或等距映射。两个可实现选择是 \(\mathcal T_\eta\xi=DR_x(\eta)[\xi]\)，或嵌入情形的 \(P_{T_{R_x(\eta)}M}\xi\)。

**计算判据。** 检查可行性、\(DR_x(0)[\eta]=\eta\)、梯度与任意切方向的配对及 Hessian 的度量自伴性。一般回缩也可建立二阶模型，但对 \(c(t)=R_x(t\eta)\) 必须计入
\[
D^2(f\circ R_x)(0)[\eta,\eta]
=g_x(\operatorname{Hess}f(x)[\eta],\eta)
+df_x\!\left[\left.\nabla_{\dot c}\dot c\right|_{t=0}\right].
\]
二阶回缩令初始协变加速度为零，因而保证任意点、任意目标的 pullback Hessian 与 Riemannian Hessian 一致；指数映射满足这一条件。临界点上 \(df_x=0\)，任意回缩都给出该一致性。对固定目标或方向，额外项也可能因配对为零而消失，不能把二阶回缩当成所有二阶模型的必要条件。

**例子。** 球面 \(S^{n-1}\) 上 \(P_x=I-xx^\mathsf T\)、\(R_x(\eta)=(x+\eta)/\|x+\eta\|\)。原始欧氏梯度的径向部分只会违反约束，切向部分才改变可行参数。

**阅读。** Absil–Mahony–Sepulchre §3.6、§4.1、§5.3–5.5、§8.1；回缩的二阶项另见 Boumal §5.10 式 (5.28)、Propositions 5.44–5.45（[M04](sources-manifolds.md)）。

## DG12 Stiefel 与 Grassmann 的可计算接口

**解决什么。** 区分“选择一个正交基”和“选择其张成子空间”，避免优化不可辨识的基旋转。

Stiefel 流形 \(\mathrm{St}(n,p)=\{X:X^\mathsf TX=I\}\) 在诱导 Frobenius 度量下满足

\[
T_X\mathrm{St}=\{\eta:X^\mathsf T\eta+\eta^\mathsf TX=0\},
\quad P_X(Z)=Z-X\operatorname{sym}(X^\mathsf TZ).
\]

可用正对角 QR 的 \(\operatorname{qf}(X+\eta)\) 或 polar 回缩

\[
R_X(\eta)=(X+\eta)(I+\eta^\mathsf T\eta)^{-1/2}.
\]

Grassmann 把 \(X\sim XQ,Q\in O(p)\) 识别为同一子空间。正交代表中的水平切向量满足 \(X^\mathsf T\eta=0\)，投影是 \((I-XX^\mathsf T)Z\)。

**例子与判据。** 若 \(f(XQ)=f(X)\)，可在 Grassmann 上建模；例如 \(f(X)=-\operatorname{tr}(X^\mathsf TAX)\) 只依赖子空间。若目标逐列区分 \(X\)，则这种商空间识别会改变问题。比较两个子空间可对 \(X^\mathsf TY\) 求 SVD，奇异值 \(\cos\theta_i\) 给出主夹角。

**实现边界。** Stiefel 的 canonical metric 和诱导 Frobenius metric 不相同，投影、梯度、测地线公式必须跟随所选度量。QR 的符号约定会影响局部光滑实现；先固定对角符号，再讨论导数。

**阅读。** Absil 等 §3.3–3.6、Example 4.1.3、§5.3。

## DG13 正定矩阵几何、均值与张量场平滑

**解决什么。** 在协方差、扩散张量和正定预条件器之间计算距离、插值与均值，保持所选不变性。

\(\mathrm{SPD}(n)\) 的切空间是对称矩阵。仿射不变度量

\[
g_P(U,V)=\operatorname{tr}(P^{-1}UP^{-1}V)
\]

在 \(P\mapsto APA^\mathsf T\) 下保持。对应指数、对数与距离为

\[
\operatorname{Exp}_P(U)=P^{1/2}\exp(P^{-1/2}UP^{-1/2})P^{1/2},
\]

\[
\operatorname{Log}_P(Q)=P^{1/2}\log(P^{-1/2}QP^{-1/2})P^{1/2},
\quad d(P,Q)=\|\log(P^{-1/2}QP^{-1/2})\|_F.
\]

若 \(\nabla_Ef\) 是对称欧氏梯度，\(\operatorname{grad}f=P\nabla_EfP\)。这些公式在严格正定时成立，半正定边界不在此流形内部。

**均值与插值。** 非负且和为 1 的权重下，Fréchet 均值极小化 \(\tfrac12\sum_iw_id(P,P_i)^2\)，其一阶条件为 \(\sum_iw_i\operatorname{Log}_P(P_i)=0\)。二点插值是 \(\operatorname{Exp}_P(t\operatorname{Log}_P Q)\)。

**张量场工具。** 对欧氏定义域上的 \(P(x)\in\mathrm{SPD}(n)\)，该目标度量的张力场为

\[
\tau(P)=\sum_i\left[\partial_i^2P-(\partial_iP)P^{-1}(\partial_iP)\right].
\]

用 \(\operatorname{Exp}_P(h\tau)\) 做几何更新可保持正定；稳定步长和离散能量仍需分析。它是映射的协变 Laplacian，不能直接与逐条目欧氏 Laplacian 等同。

**例子与边界。** 标量正数 \(a,b\) 的此度量中点为 \(\sqrt{ab}\)，不是算术平均。对数欧氏几何是另一种度量：\(\exp((\log P+\log Q)/2)\) 一般不同于上面的仿射不变中点。不要仅因都用了矩阵 \(\log\) 就互换。

**阅读。** Pennec–Fillard–Ayache，§3.3–3.4、§4、§5 与 Appendix A；这里的“tensor”明确指 SPD 二阶对象。

## DG14 零测地线丛、光学张量与聚焦

**解决什么。** 将局部曲率与一簇轨迹的形变联系起来，而不把点态代数分类误当成动力学结论。

在维数 \(n>2\)、签名 \((-+\cdots+)\) 的 Lorentz 流形上，设 \(\ell\ne0\) 是仿射参数的零测地向量场，\(\nabla_\ell\ell=0\)。屏空间是 \(\ell^\perp/\langle\ell\rangle\)，具有正定度量，维数 \(n-2\)；一般不定签名不保证其正定。\(X\mapsto\nabla_X\ell\) 在屏空间上定义光学算子 \(B\)，分成

\[
B=\frac{\Theta}{n-2}I+\sigma+\omega,
\quad\Theta=\operatorname{tr}B,
\quad\sigma=B_{(ij)}-\frac{\Theta}{n-2}\delta_{ij},
\quad\omega=B_{[ij]}.
\]

在沿射线平行的屏标架中，\(\dot B=-B^2-\mathcal R\)，其中 \(\mathcal R_{ij}=R_{a b c d}e_i^a\ell^be_j^c\ell^d\)。取迹得到

\[
\dot\Theta=-\frac{\Theta^2}{n-2}-\|\sigma\|^2+\|\omega\|^2
-\operatorname{Ric}(\ell,\ell).
\]

四维有些文献用 \(\theta=\Theta/2\)，复剪切的归一化也可能不同；比较公式先转换定义。这里屏空间张量的范数由其正定度量定义。

**判据与例子。** 无扭且 \(\operatorname{Ric}(\ell,\ell)\ge0\) 时有 \(\dot\Theta\le-\Theta^2/(n-2)\)。若 \(\Theta(0)<0\)，积分得到的聚焦上界为 \(t_*=(n-2)/|\Theta(0)|\)：在测地线可延伸至该范围、曲率光滑且上述条件持续成立时，横截 Jacobi 映射须在不晚于 \(t_*\) 时退化，光束不能一直保持正则。若测地线提前离开所给流形或不可延伸，不能据局部不等式断言该流形内已有焦点。Petrov 类本身没有规定 \(\Theta,\sigma,\omega\) 的演化，需要联络、曲率与初始光束。

**阅读。** Bär §6.6 的屏丛及光学分解；Riccati 与取迹式可由 DG04 的 Jacobi 方程推得。延伸条件与聚焦比较见 Galloway §6、Propositions 6.3–6.4、式 (6.18)–(6.21)（[来源 G03 补充](sources-geometry.md)）。若要使用 Goldberg–Sachs 等把剪切与重复主零方向联系起来的定理，需回到该定理的四维 Einstein 等明确假设，不能仅凭同名 Petrov 类迁移到网络。

## 与算子分析的连接

这些工具分别提供坐标一致的求导、保持结构的优化、场的平滑及曲率分类入口。网络中选择 Fisher、Jacobian 拉回或其它度量时，应说明它定义了什么统计或误差关系；度量选择改变梯度与曲率。探索时可以并列研究不同选择，推导各自保留的结构与可区分预测，再考察它们与任务的联系。尚待建立的目标对应保持为研究假设；已证的几何关系按其自身条件保留在知识库中。
