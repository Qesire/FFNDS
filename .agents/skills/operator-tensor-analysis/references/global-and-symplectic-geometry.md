# 全局拓扑、流与辛几何

用本文件处理局部公式不能独自决定的问题：流是否存在到目标时刻、局部势能否拼成全局势、约束交集是否仍是流形、拓扑是否迫使临界点存在、动力学和离散步是否保持几何结构。先建立[微分流形与丛](smooth-manifolds-and-bundles.md)中的对象与正则性；外微分、Cartan 演算、联络及 Hodge 工具沿用[微分几何](differential-geometry.md)。

默认流形有限维、Hausdorff、第二可数，映射光滑；未另说明时无边界。边界流形、非紧流形及奇异商必须重新列出附加条件。下面的数值验算与反例用于诊断具体模型，不替代全域证明。

## 内容

- GS01 局部流、完备性与流的线性化
- GS02 横截性、Sard 与参数扰动
- GS03 定向、映射度与有符号解计数
- GS04 de Rham 上同调、拼接与全局势
- GS05 Morse 临界点与子水平集
- GS06 辛形式、余切丛与 Darboux 坐标
- GS07 Lagrangian 子流形与生成函数
- GS08 Hamiltonian 向量场、Poisson 括号与守恒
- GS09 矩映射、约束与辛约化
- GS10 辛积分器、误差与稳定性
- GS11 接触形式、Reeb 场与非完整约束
- 调用与实现检查
- 来源与实际阅读范围

## GS01 局部流、完备性与流的线性化

**解决什么。** 把连续深度模型、输运或优化方向场变成有定义域的演化算子，并判断可逆性和灵敏度声明在什么时间范围成立。

对 \(X\in\Gamma(TM)\)，先写初值问题及其最大流域：

\[
\dot x(t)=X(x(t)),\quad x(0)=x_0,\qquad
\Phi:\mathcal D\subset\mathbb R\times M\to M.
\]

\(\mathcal D\) 是包含 \(\{0\}\times M\) 的开集。\(\Phi_{t+s}=\Phi_t\circ\Phi_s\) 仅在两侧都有定义时使用；\(\Phi_t\) 在相应开域之间是微分同胚，逆为 \(\Phi_{-t}\)。只有 \(\mathcal D=\mathbb R\times M\) 时才称 \(X\) 完备，并得到全局一参数群。紧致无边界流形上的光滑向量场、或无边界流形上的紧支撑光滑向量场都完备。[GS-S1]

若 \(X(p)\ne0\)，可取局部流盒坐标使 \(X=\partial/\partial u^1\)。构造方法是选与 \(X(p)\) 横截的小截面，再用流的时间作为第一坐标。这是局部直化，不保证同一坐标覆盖闭轨道或平衡点。[GS-S2]

**计算。** 在同一坐标图中，对初值求导得到变分方程

\[
A^a{}_i(t)=\frac{\partial\Phi_t^a}{\partial x_0^i},\qquad
\dot A^a{}_i=(\partial_bX^a)(\Phi_t)A^b{}_i,
\quad A(0)=I.
\]

由 Jacobi 行列式公式直接得到

\[
\frac{d}{dt}\log|\det A|=\partial_aX^a(\Phi_t).
\]

一般体积形式 \(\nu=\rho(x)\,dx^1\wedge\cdots\wedge dx^n\) 对应
\(\operatorname{div}_\nu X=\rho^{-1}\partial_i(\rho X^i)\)。比较密度时要同时变换体积形式，不能把一个坐标图中的普通 Jacobian 行列式当作内禀体积比。

**反例与验算。** \(\dot x=x^2\) 的流为 \(\Phi_t(x_0)=x_0/(1-tx_0)\)，对 \(x_0>0\) 在 \(t=1/x_0\) 爆破；向量场处处光滑仍不足以保证完备。若状态域是有边界的约束集，还要检查边界切向性或离开域的事件。对时变 \(X_t\) 使用两时刻传播子 \(\Phi_{t,s}\)，不能直接套自治流的 \(\Phi_{t+s}\) 群律。

## GS02 横截性、Sard 与参数扰动

**解决什么。** 判断约束、等值面和模型族的交集能否稳定地当作光滑流形处理，区分维数计数与实际秩条件。

给定嵌入子流形 \(S\subset N\) 和 \(f:M\to N\)，在每个 \(x\in f^{-1}(S)\) 检查

\[
df_x(T_xM)+T_{f(x)}S=T_{f(x)}N.
\]

成立时记 \(f\pitchfork S\)，且非空原像是余维 \(\operatorname{codim}_N S\) 的子流形。令 \(S=\{y\}\) 就得到正则值条件。具体约束的秩计算见[流形基础](smooth-manifolds-and-bundles.md)。[GS-S4]

Sard 定理在这里采用有限维、\(C^\infty\) 版本：临界值集在目标坐标图中为 Lebesgue 零测集，因此正则值稠密。它不保证某个预先指定值正则，也不保证该值有原像。缺少 proper 性等条件时，正则值集合未必开。[GS-S3]

**参数化调用。** 对 \(F:P\times M\to N\)，一个易检验的充分条件是 \(F\) 为下沉映射。此时对固定 \(S\subset N\)，几乎所有参数 \(a\in P\) 的切片 \(F_a\) 横截 \(S\)。更一般的 \(F\pitchfork S\) 版本可由对投影 \(F^{-1}(S)\to P\) 使用 Sard 得到；说“随机扰动消除退化”前先验证整个参数族的横截条件，以及采样分布对参数坐标 Lebesgue 测度绝对连续。[GS-S4]

**反例与验算。** \(F(a,x)=x^2-a\) 对 \((a,x)\) 的微分处处满射，零等值面在 \(a\ne0\) 的切片均正则；\(a<0\) 时原像为空。若参数族只是 \(F(a,x)=x^2\)，所有切片在零点退化。有限个采样点上最小奇异值非零只能作为数值证据，不能证明全流形上的横截性。

## GS03 定向、映射度与有符号解计数

**解决什么。** 为局部可逆性增加全局检验，追踪变换的绕行次数、解的存在性及积分方向。

先选定 \(M,N\) 的定向。设二者紧致、连通、无边界且维数同为 \(n\)，\(F:M\to N\) 光滑。对正则值 \(y\)，

\[
\deg F=\sum_{x\in F^{-1}(y)}\operatorname{sgn}\det(dF_x),
\qquad
\int_MF^*\eta=(\deg F)\int_N\eta,
\quad\eta\in\Omega^n(N).
\]

行列式符号在与所选定向一致的基中计算。度数是整数且同伦不变；非零度数蕴含满射。非紧或有边界版本需要 proper 映射、相对度数或对边界的条件，不能删除上述假设后照抄公式。[GS-S1]

**例子与诊断。** \(F(e^{i\theta})=e^{ik\theta}\)，\(k\in\mathbb Z\setminus\{0\}\)，处处局部可逆且 \(\deg F=k\)。当 \(|k|>1\) 时仍有多个原像，因此满秩 Jacobian 不能证明全局单射。度数为零也可能有原像，只是定向符号相消；度数为一仍不能单独证明单射。

**算子接口。** 微分形式的拉回保留有符号 Jacobian；概率密度的换元使用绝对值，并在非单射情形对各原像求和。先辨明被积分对象是定向顶形式还是正密度，再判断实现中的 `det`、`abs(det)` 和分支求和。

## GS04 de Rham 上同调、拼接与全局势

**解决什么。** 判断局部梯度场、守恒量或规范势能否全局实现，找出无论怎样局部拟合都不能消除的拓扑障碍。

把闭形式的全局剩余自由度记录为

\[
H^k_{\mathrm{dR}}(M)
=\ker(d:\Omega^k\to\Omega^{k+1})/
\operatorname{im}(d:\Omega^{k-1}\to\Omega^k).
\]

对闭一形式 \(\alpha\)，全局写成 \(df\) 当且仅当所有闭曲线周期 \(\oint_\gamma\alpha\) 都为零；仅在可缩图内求势不能回答这一问题。[GS-S1]

**计算入口。** 优先找形变收缩；仍需拼接时，取开覆盖 \(M=U\cup V\)，用 Mayer–Vietoris 长正合列

\[
\cdots\to H^{k-1}(U\cap V)\xrightarrow{\delta}H^k(M)
\to H^k(U)\oplus H^k(V)
\xrightarrow{r_U-r_V}H^k(U\cap V)\to\cdots.
\]

好覆盖要求每个非空有限交集可缩；此时可通过覆盖的 Čech 复形计算实系数 de Rham 上同调。只有开集本身可缩、交集却未检查，不足以宣称是好覆盖。若已有紧致无边界 Riemannian 结构并需要数值代表，转用[DG06 的 Hodge 工具](differential-geometry.md)，保留其边界与正定性条件。[GS-S1]

**反例与验算。** 在 \(\mathbb R^2\setminus\{0\}\) 上

\[
\alpha=\frac{-y\,dx+x\,dy}{x^2+y^2},\qquad
d\alpha=0,\qquad\oint_{x^2+y^2=1}\alpha=2\pi.
\]

局部可写 \(d\theta\)，全局不存在单值实函数势。要保留此场，可以使用重叠坐标与转移数据；强制单一全局势会删去这一自由度。离散上同调若用于推断连续拓扑，还须说明网格或复形为何正确表示目标空间；单纯检查 \(D_{k+1}D_k=0\) 只证明形成了复形。

## GS05 Morse 临界点与子水平集

**解决什么。** 从驻点的局部二次型推断损失景观或能量子水平集可能发生的拓扑变化，并判断这种推断何时失效。

先求 \(df_p=0\)，再检查临界点 Hessian 非退化。因为此时 \(df_p=0\)，二阶偏导矩阵在换坐标下按合同变换，惯性指数有内禀意义。令 \(\lambda\) 为负特征值个数，Morse 引理给出某组局部坐标中的**精确等式**

\[
f=f(p)-\sum_{i=1}^{\lambda}(u^i)^2
+\sum_{i=\lambda+1}^{n}(u^i)^2.
\]

这比固定原坐标中的二阶 Taylor 近似强；该坐标变换通常不保持原度量或数值条件数。[GS-S5]

**全局使用条件。** 记 \(M_a=\{f\le a\}\)。若 \(f^{-1}([a,b])\) 紧致且其中无临界点，则 \(M_a,M_b\) 微分同胚。若 \(a,b\) 正则、该带紧致且仅含一个非退化临界点，则 \(M_b\) 与给 \(M_a\) 附加一个 \(\lambda\) 维胞腔所得空间同伦等价。胞腔陈述是同伦类型陈述，不是把开胞腔直接当作光滑边界拼接。[GS-S6]

对紧致无边界流形上的 Morse 函数，临界点计数 \(c_k\) 与 Betti 数满足 \(c_k\ge b_k\)，且 \(\chi(M)=\sum_k(-1)^kc_k\)。它给出必要的驻点数量约束，不给出算法找到全局最优的时间界。[GS-S6]

**反例与验算。** \(f(x)=x^4\) 在零点有严格极小值，却不满足非退化条件。连续参数对称性常使临界点形成子流形、Hessian 出现零方向；先检查是否满足 Morse–Bott 的法向非退化条件，或先建立适当商空间，不能直接把零特征值丢弃后使用 Morse 引理。数值判断惯性时记录容差和谱隙。

## GS06 辛形式、余切丛与 Darboux 坐标

**解决什么。** 为相空间建立与内积不同的结构，使共轭变量、守恒动力学和保持结构的变换有统一接口。

对 \(2n\) 维流形，检验二形式 \(\omega\) 是否满足

\[
d\omega=0,\qquad
\omega^\flat:TM\to T^*M,\quad X\mapsto\iota_X\omega
\quad\text{处处可逆}.
\]

这两条定义辛结构。仅有反对称非奇异矩阵场还缺少闭性；在坐标中需检查 \(\partial_a\omega_{bc}+\partial_b\omega_{ca}+\partial_c\omega_{ab}=0\)。辛形式不是正定度量，不能用 \(\omega(v,v)\) 定义长度。[GS-S7]

**统一约定。** 在余切丛 \(T^*Q\) 中取

\[
\theta=p_i\,dq^i,\qquad
\omega=-d\theta=dq^i\wedge dp_i,
\qquad z=(q,p),\qquad
J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.
\]

\(\theta_{(q,p)}(v)=p(d\pi(v))\) 是不依赖坐标的定义。Darboux 定理保证任意辛流形在每个点附近都能写成上述 \(\omega\)；它不提供全局正则坐标，不消除 Hamiltonian 的非线性，也不使额外指定的度量自动平坦。[GS-S7]

**变换验算。** 对局部相空间映射 \(F\)，检验

\[
(DF)^\top J(DF)=J.
\]

它蕴含局部可逆与相空间体积保持，但在 \(2n\ge4\) 时，行列式等于一远不足以证明辛性。例如在 \((q^1,q^2,p_1,p_2)\) 次序下，\(A=\operatorname{diag}(2,\tfrac12,1,1)\) 有 \(\det A=1\)，却未保持 \(dq^1\wedge dp_1+dq^2\wedge dp_2\)。

## GS07 Lagrangian 子流形与生成函数

**解决什么。** 把保辛变换、作用量与势函数联系起来，构造具有结构保证的可逆模块或隐式更新。

对子流形 \(i:L\hookrightarrow(M^{2n},\omega)\)，同时检查 \(\dim L=n\) 和 \(i^*\omega=0\)。二者成立才称 Lagrangian。若一形式 \(\alpha\) 的图像是 \(T^*Q\) 中的截面，则其拉回辛形式为 \(-d\alpha\)，因此图像 Lagrangian 当且仅当 \(d\alpha=0\)。全局写成 \(\alpha=dS\) 仍要通过 GS04 的上同调检查。[GS-S7]

对变换 \((q,p)\mapsto(Q,P)\)，可用二类生成函数 \(S(q,P)\) 定义

\[
p_i=\frac{\partial S}{\partial q^i},\qquad
Q^i=\frac{\partial S}{\partial P_i}.
\]

由 \(dS=p_i\,dq^i+Q^i\,dP_i\)，再取外微分即得 \(dq^i\wedge dp_i=dQ^i\wedge dP_i\)。要让这些关系实际定义局部映射，检查混合 Hessian \(\partial^2S/\partial q^i\partial P_j\) 可逆，从而能从旧 \((q,p)\) 解出 \(P\)。生成函数存在与所选投影的横截性相关，不是每个全局保辛映射都有同一种全局生成函数。[GS-S8]

**例子与验算。** 取 \(S(q,P)=q^iP_i+hH(q,P)\)，得到
\(p=P+hH_q(q,P)\)、\(Q=q+hH_P(q,P)\)，正是一个隐式辛 Euler 更新。隐式求解误差会转化为结构误差；给出残差与迭代容差。若混合 Hessian 接近奇异，先评估解分支与条件数，不把形式上的生成函数当作稳定可逆性的保证。

## GS08 Hamiltonian 向量场、Poisson 括号与守恒

**解决什么。** 从标量能量产生动力学，区分能量守恒、体积保持、辛性，以及一般反对称动力系统。

沿用 GS06 并固定

\[
\iota_{X_H}\omega=dH,\qquad
X_H=J\nabla H,\qquad
\dot q^i=\partial_{p_i}H,\quad
\dot p_i=-\partial_{q^i}H.
\]

定义

\[
\{f,g\}=\omega(X_f,X_g)
=\partial_{q^i}f\,\partial_{p_i}g
-\partial_{p_i}f\,\partial_{q^i}g.
\]

则 \(X_Hf=\{f,H\}\)，并且本约定下
\([X_f,X_g]=-X_{\{f,g\}}\)。如果参考资料把 \(X_f\) 定义为 \(\{f,\cdot\}\)，其向量场符号与此处相反，先翻译再调用。[GS-S7, GS-S9]

**直接推导。** 由 Cartan 公式，

\[
\mathcal L_{X_H}\omega=d^2H+\iota_{X_H}d\omega=0,\qquad
X_HH=\{H,H\}=0.
\]

因此自治 Hamiltonian 流在存在的时间区间内保持 \(H\)、\(\omega\) 和体积形式 \(\omega^n/n!\)。时间依赖的 \(H_t\) 仍产生保辛传播子，但 \(dH_t(x(t))/dt=\partial_tH_t\)，不能宣称原能量守恒。全时间存在性另用 GS01。

**拓展到 Poisson 张量。** 写

\[
\{f,g\}=\Pi^{ab}(z)\partial_af\,\partial_bg,
\qquad\dot z^a=\Pi^{ab}\partial_bH.
\]

除反对称性外，必须验证 Jacobi 条件

\[
\Pi^{a\ell}\partial_\ell\Pi^{bc}
+\Pi^{b\ell}\partial_\ell\Pi^{ca}
+\Pi^{c\ell}\partial_\ell\Pi^{ab}=0.
\]

允许 \(\Pi\) 退化；若 \(\Pi^{ab}\partial_bC=0\)，则 \(C\) 是 Casimir，对所有 Hamiltonian 都守恒。一般反对称 \(\Pi\) 也可使 \(dH/dt=0\)，但不一定满足 Jacobi，更不自动提供辛体积。[GS-S9]

**全局障碍。** 保辛向量场满足 \(d\iota_X\omega=0\)；存在全局 \(H\) 还要求 \(\iota_X\omega\) 恰当。在二环面上取 \(\omega=dq\wedge dp\)、\(X=\partial_q\)，则 \(\iota_X\omega=dp\) 周期非零，不能成为全局实函数的微分。这正是 GS04 在动力学中的调用。

## GS09 矩映射、约束与辛约化

**解决什么。** 同时固定守恒量与除去对称冗余，获得仍有辛结构的较小状态空间。

给定保辛左作用 \(G\curvearrowright M\)，取基本向量场
\(\xi_M(x)=\left.\frac{d}{dt}\right|_{t=0}\exp(t\xi)\cdot x\)，并约定
\(\langle\operatorname{Ad}^*_g\mu,\xi\rangle=\langle\mu,\operatorname{Ad}_{g^{-1}}\xi\rangle\)。
矩映射 \(\mu:M\to\mathfrak g^*\) 要满足

\[
d\mu^\xi=\iota_{\xi_M}\omega,
\qquad\mu^\xi=\langle\mu,\xi\rangle,
\qquad\mu(gx)=\operatorname{Ad}^*_g\mu(x).
\]

若 \(H\) 在群作用下不变，则 \(\xi_MH=\{H,\mu^\xi\}=0\)，所以 \(d\mu^\xi/dt=\{\mu^\xi,H\}=0\)。一般保辛作用未必有全局矩映射。[GS-S7]

**正则约化。** 一个明确的充分条件组合是：\(G\) 紧致、\(\mu\) 等变、\(0\) 为正则值、\(C=\mu^{-1}(0)\ne\varnothing\)，且 \(G\) 在 \(C\) 上自由作用。此时

\[
M_{\rm red}=C/G,\qquad
\dim M_{\rm red}=2n-2\dim G,\qquad
\pi^*\omega_{\rm red}=i^*\omega.
\]

先限制后取商，因为 \(i^*\omega\) 的核恰为群轨道切向。仅把约束代入并删除变量可能留下退化二形式。紧致性可由适当 proper 性条件替代；商空间的基础判据见[微分流形与丛](smooth-manifolds-and-bundles.md)。[GS-S7]

**反例与边界。** 在 \(T^*\mathbb R^2\) 中限制 \(p_2=0\)，拉回形式仅余 \(dq^1\wedge dp_1\)，其核含 \(\partial_{q^2}\)；还要除去相应冗余方向才能得到二维辛空间。若群作用有非平凡稳定子，商可能是 orbifold 或分层奇异空间，不能继续按普通流形求逆。非零值 \(\eta\) 通常要商去余伴随稳定子 \(G_\eta\)，不能直接除以整个 \(G\) 或照搬零值维数公式。

## GS10 辛积分器、误差与稳定性

**解决什么。** 将连续 Hamiltonian 算子变成离散步，并把形式保辛与实际数值行为分别验收。

对可分离 \(H(q,p)=T(p)+V(q)\)，固定步长 \(h\) 的辛 Euler 为

\[
p_{k+1}=p_k-h\nabla V(q_k),\qquad
q_{k+1}=q_k+h\nabla T(p_{k+1}).
\]

它是一阶方法。对称组合得到二阶 Störmer–Verlet：

\[
\begin{aligned}
p_{k+1/2}&=p_k-\tfrac h2\nabla V(q_k),\\
q_{k+1}&=q_k+h\nabla T(p_{k+1/2}),\\
p_{k+1}&=p_{k+1/2}-\tfrac h2\nabla V(q_{k+1}).
\end{aligned}
\]

每个 kick 或 drift 都是一个 Hamiltonian 子系统的精确流，故组合保辛。一般不可分离 \(H\) 的对应公式通常隐式，不能继续把上式当作任意 \(H\) 的显式格式。[GS-S10]

**可复算的反例。** 对 \(H=(q^2+p^2)/2\)，辛 Euler 的一步矩阵为

\[
A_h=\begin{pmatrix}1-h^2&h\\-h&1\end{pmatrix},
\qquad A_h^\top JA_h=J.
\]

它保持 \(\widetilde H_h=(q^2+p^2-hqp)/2\)，一般不逐步保持原 \(H\)。\(\widetilde H_h\) 仅在 \(|h|<2\) 时正定；超出此区间不能期待该振子的离散稳定性。普通显式 Euler 的矩阵 \(I+hJ\) 则使原能量乘以 \(1+h^2\)。这些结论可直接由二维矩阵乘法验证，不能从“阶数高”或“看起来不漂移”替代推导。

**实现与验收。** 分开记录 \(\|(DF)^\top JDF-J\|\)、轨道误差、能量误差与约束误差；大维度可随机探测 \(\omega(DFu,DFv)-\omega(u,v)\)，但抽样不构成严格证明。`autodiff` 求导后端、隐式求解容差、低精度舍入和额外投影都可能引入结构残差。每步预先规定不同 \(h_k\) 仍可组成保辛映射；令步长依赖当前状态时，链式法则出现额外项，一般不再保辛。长时间能量误差界还需光滑性、步长和有界轨道等假设，不能仅由保辛性给出。

## GS11 接触形式、Reeb 场与非完整约束

**解决什么。** 处理奇维状态空间中的超平面约束，并识别局部允许速度能否积分为配置子流形。

在 \(2n+1\) 维流形上，接触形式满足

\[
\alpha\wedge(d\alpha)^n\ne0,
\qquad\mathcal H=\ker\alpha.
\]

给定 \(\alpha\) 后，Reeb 场由

\[
\alpha(R)=1,\qquad\iota_Rd\alpha=0
\]

唯一确定。接触结构可只局部写成 \(\ker\alpha\)；全局接触形式需要共定向条件。缩放 \(\alpha\) 不改变超平面分布，却一般改变 Reeb 场。这里不把选定分布自动解释成耗散模型，具体动力学仍需单独给定。[GS-S7]

**直接验算。** 在 \((q,p,z)\) 上取 \(\alpha=dz-p\,dq\)，则 \(d\alpha=dq\wedge dp\)、\(R=\partial_z\)。两个允许速度场

\[
X=\partial_q+p\partial_z,\qquad
Y=\partial_p
\]

都满足 \(\alpha(X)=\alpha(Y)=0\)，但 \([X,Y]=-\partial_z\notin\mathcal H\)。因此这个秩二分布不能成为二维叶片的切丛。这是非完整速度约束的最小例子；一般分布用[DG08 的 Frobenius 判据](differential-geometry.md)检查。非完整约束系统不自动等价于 GS09 的辛约化。

需要偶维辛表示时，可对已选全局接触形式构造辛化
\(\widetilde M=M\times\mathbb R\)、\(\widetilde\omega=d(e^t\alpha)\)。这是增加维度和结构的明确建模步骤，不能凭奇偶维数就认定某个特征空间天然具有接触几何。[GS-S7]

## 调用与实现检查

| 要回答的问题 | 优先调用 | 必须交付的证据 |
|---|---|---|
| 连续模型是否可逆、能否运行任意时间 | GS01 | 最大定义域或完备性条件；边界和爆破检查 |
| 扰动后约束交集是否仍光滑 | GS02 | 实际横截/秩条件；参数族而非单点的检查 |
| 局部无奇异点能否推出全局一一对应 | GS03 | 度数、覆盖或单射性证据；不能只给 Jacobian |
| 闭场是否能用一个全局势表示 | GS04、GS08 | 周期或上同调类；局部势的转移关系 |
| 驻点与景观拓扑怎样关联 | GS05 | 非退化性、Morse 指数、紧致带及正则端点 |
| 变换是否保辛、能否构造成保辛模块 | GS06–GS08 | 形式与符号约定；拉回或 Jacobian 恒等式 |
| 能否删除对称冗余并保持动力学 | GS09 | 矩映射、正则值、自由 proper 作用和约化形式 |
| 离散迭代是否保持连续结构 | GS10 | 一步映射、精度/残差、稳定步长与独立能量检查 |
| 速度约束能否积分成位置约束 | GS11 | 分布秩、Lie 括号闭合性或接触条件 |

把几何结构映射到实现时，显式记录坐标次序、共轭变量配对、约束和可用时间区间。矩阵 \(J\)、Jacobian 与投影是选定坐标下的表示；布局、分块与融合可以改变执行方式，但不能擅自改变张量槽、辛形式或隐式解分支。

## 来源与实际阅读范围

下列来源用于核对定义、定理前提及符号。卡片中的模型选择、实现诊断和明示的反例为独立推导；只把已读章节列为核对范围，不把来源目录中的主题算作已覆盖。

| 编号 | 一手作者资料 | 本文件实际核对的位置 |
|---|---|---|
| GS-S1 | Eckhard Meinrenken, [Differentiable Manifolds](https://www.math.toronto.edu/mein/teaching/LectureNotes/Man.pdf) | §3.2，定理 3.8、3.10 与习题 3.9；§5.1–5.2；§6 的 Mayer–Vietoris 构造；§10 定理 10.2、推论 10.3；§12.2 定理 12.4。只调用这些范围，不采用该旧稿其他章节的未核实断言。 |
| GS-S2 | Brian Conrad, [Local and global Frobenius theorems](https://math.stanford.edu/~conrad/210CPage/handouts/frobthm.pdf) | §5，定理 5.3 与横截切片构造；非零向量场的局部直化。 |
| GS-S3 | Tomasz Mrowka, MIT 18.965, [Lectures 10–11](https://ocw.mit.edu/courses/18-965-geometry-of-manifolds-fall-2004/lecture10_11.pdf) | 定理 12.3 的有限维光滑 Sard 定理；不调用讲义其余引理来扩大到任意可测像。 |
| GS-S4 | Tomasz Mrowka, MIT 18.965, [Lecture 20](https://ocw.mit.edu/courses/18-965-geometry-of-manifolds-fall-2004/lecture20.pdf) | 定义 18.1、引理 18.2、定理 18.3；参数族下沉映射的充分条件及对原像投影使用 Sard 的论证。 |
| GS-S5 | Brian Conrad, [The Morse Lemma](https://math.stanford.edu/~conrad/diffgeomPage/handouts/morselemma.pdf) | §1 的 Hessian 与非退化性；§2 定理 2.1、推论 2.4。使用一般函数时明确补回常数 \(f(p)\)。 |
| GS-S6 | Joa Weber, [Lecture Notes on Morse and Conley Theory](https://www.math.stonybrook.edu/~joa/PUBLICATIONS/MORSELEC.pdf), 2018 manuscript | 导论定理 1.0.1、1.0.2，弱 Morse 不等式与 Morse–Bott 段落。该链接可见内容不是目录所列全部章节；不据目录声称已读完整 Conley/Floer 理论。 |
| GS-S7 | Ana Cannas da Silva, [Lectures on Symplectic Geometry](https://people.math.ethz.ch/~acannas/Papers/lsg.pdf), 2006 website text | §§1–3，§8.1 Darboux，§§10–11 接触/Reeb/辛化，§18.1、18.3 Hamiltonian 与括号，§22.1 矩映射，§23.1 正则约化；并核对作者 [2022 网站版勘误](https://people.math.ethz.ch/~acannas/Papers/lsg_errata_website.pdf)。避免把旧稿中的猜想状态当成当前事实。 |
| GS-S8 | Ernst Hairer, [Geometric Numerical Integration, Lecture 1](https://www.unige.ch/~hairer/poly_geoint/week1.pdf), 2010 | §5 定理 4–5 与式 (22)：生成函数及局部保辛变换；其变量次序 \((p,q)\) 已转换为本文件的 \((q,p)\)。 |
| GS-S9 | Eckhard Meinrenken, [Introduction to Poisson Geometry](https://www.math.toronto.edu/mein/teaching/MAT1341_PoissonGeometry/Poisson7.pdf), 2017 | §§1.3–1.5：Jacobi、Poisson 双向量及 Casimir。该稿 \(X_f=\{f,\cdot\}\) 的约定与 GS08 相反，引用时已明确翻译。 |
| GS-S10 | Ernst Hairer, [Geometric Numerical Integration, Lecture 2](https://www.unige.ch/~hairer/poly_geoint/week2.pdf), 2010 | §1 定理 1–3：辛 Euler、Störmer–Verlet、隐式中点法；本文采用可分离 Hamiltonian 的显式子情形。 |
