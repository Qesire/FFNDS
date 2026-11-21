# 流形建模与算子计算

用于把流形结构落实为可辨识维数、约束方向、近似误差与可执行更新。图册、切丛、正则值、商空间的基础见[微分流形与丛](smooth-manifolds-and-bundles.md)；联络、回缩、曲率及 Stiefel／Grassmann／SPD 接口见[微分几何](differential-geometry.md)。自然梯度与 Fisher／GGN 的区别沿用[近似与优化 A6](approximation-and-optimization.md)。

本文件均采用实数有限维空间；复数问题需明确实流形结构或 Hermitian 度量。矩阵转置对应所声明的欧氏／Frobenius 内积。出现流形距离、正交投影或运输时，须保留相应度量、联络与路径。

## 目录

- GM01：参数、模型与数据的局部维数
- GM02：固定秩矩阵的切投影与回缩
- GM03：低秩因子的规范与表达维数
- GM04：固定 TT 秩流形及唯一切向表示
- GM05：正则约束、KKT 与约束曲率
- GM06：概率单纯形、Fisher 拉回与 softmax 冗余
- GM07：沿流形的有限变化与 Taylor 误差
- GM08：跨点方向、算子运输与缓存

## GM01　参数、模型与数据的局部维数

**对象与目标。** 给定参数流形 \(\Theta\)、观测／模型映射 \(\Phi:\Theta\to W\) 和当前参数 \(\theta\)，判断哪些参数变化能改变指定观测。下述有限维结论中的 \(W\) 可以是输出向量空间，也可以是有限维算子空间 \(\operatorname{Hom}(V,U)\)；须明确观测是有限训练集还是某个条件下的算子。若观测是全部输入上的模型函数，先指定其函数空间、拓扑和微分，再另证适用的局部定理，不能直接把下面的有限维坐标标准形搬入任意函数空间。有限样本观测的可辨识性也不等于完整模型函数的可辨识性。

局部线性变化为

\[
\delta y^a=J^a{}_i(\theta)\,\delta\theta^i,
\qquad J=d\Phi_\theta.
\]

若 \(d\Phi\) 在邻域内恒秩 \(r\)，常秩定理给出局部坐标中的标准形

\[
\Phi(u^1,\ldots,u^p)=(u^1,\ldots,u^r,0,\ldots,0).
\]

于是局部像有 \(r\) 个独立方向，局部等观测纤维有 \(p-r\) 个方向，且其切空间为 \(\ker d\Phi\)。这是指定局部观测映射的结论；不能据此推出全局唯一参数或整个像集没有自交。

**区分三个量。** 单个矩阵算子的秩 \(\operatorname{rank}A_\theta\)、算子族的局部维数 \(\operatorname{rank}dA_\theta\)、数据变化的维数分别描述不同对象。例如 \(A_t=tI_n\) 在 \(t\ne0\) 时秩为 \(n\)，算子族只有一个参数方向；它们都不能确定输入分布是否集中在低维集合附近。

**退化与证据边界。** 单点 Jacobian 秩下降不决定像集类型：\(t\mapsto t^3\) 在零点导数为零而像为光滑直线；\(t\mapsto(t^2,t^3)\) 在零点形成尖点。对 ReLU 等分段光滑模型，经典微分流形推断首先限于光滑区域；激活边界需单独处理。

数据流形假说应写成可检验的近似对象，例如

\[
\mathbb E_{X\sim P}\operatorname{dist}(X,M)^2\le\varepsilon,
\quad \dim M=d,
\]

并说明样本机制、噪声、允许的流形复杂度及误差尺度。\(X=\psi(Z)+\epsilon\) 若包含独立、非退化高斯噪声，则 \(X\) 的支持可为整个环境空间；低维潜变量仍可解释集中结构，却不意味着观测严格落在低维子流形上。Fefferman–Mitter–Narayanan 的检验问题明确限制维数、体积和 reach，并针对均方距离与置信度给结论；不能从一次 PCA 或 Jacobian 谱直接继承那些保证。见来源 GM-S6。

**可执行用法。** 对参数方向使用 JVP／VJP，估计选定度量下的奇异谱；同时记录数值阈值、采样输入与工作邻域。对数据使用局部邻域尺度、留出重构误差、噪声敏感性与跨邻域一致性。谱衰减支持该尺度下的近似低维性；精确常秩和全局流形结构需要额外依据。

## GM02　固定秩矩阵的切投影与回缩

**对象与条件。** 令 \(0<r<\min(m,n)\)，采用 Frobenius 诱导度量：

\[
\mathcal M_r=\{X\in\mathbb R^{m\times n}:\operatorname{rank}X=r\},
\qquad \dim\mathcal M_r=r(m+n-r).
\]

在薄 SVD \(X=U\Sigma V^\top\) 中，\(U^\top U=V^\top V=I_r\)、\(\Sigma\succ0\)。切向量可唯一写成

\[
\eta=UMV^\top+U_\perp V^\top+UV_\perp^\top,
\quad U^\top U_\perp=0,\quad V^\top V_\perp=0,
\]

其中 \(U_\perp\in\mathbb R^{m\times r}\)、\(V_\perp\in\mathbb R^{n\times r}\) 是增量，不是完整正交补基；\(M\in\mathbb R^{r\times r}\)。自由度为 \(r^2+(m-r)r+(n-r)r\)，得到上述维数。

**切投影。** 对任意环境增量 \(Z\)，令 \(P_U=UU^\top\)、\(P_V=VV^\top\)，则

\[
\mathsf P_XZ=P_UZ+ZP_V-P_UZP_V,
\qquad (I-\mathsf P_X)Z=(I-P_U)Z(I-P_V).
\]

因此 \(\operatorname{grad}f(X)=\mathsf P_X\nabla\bar f(X)\)。按 \((U,U^\perp)\) 与 \((V,V^\perp)\) 分块，切空间恰好令右下块为零；这同时证明投影公式、自伴性与幂等性。Vandereycken 的原文定位见 GM-S1。

**局部回缩。**

\[
R_X(\eta)=\operatorname{SVD}_r(X+\eta)
\]

取截断 SVD 重构。局部需要截断处的间隔
\(\sigma_r(X+\eta)>\sigma_{r+1}(X+\eta)\)，且保留的第 \(r\) 个奇异值为正。一个充分条件是 \(\|\eta\|_2<\sigma_r(X)/2\)：由奇异值扰动界，保留部分下界为 \(\sigma_r(X)-\|\eta\|_2\)，舍弃部分上界为 \(\|\eta\|_2\)。内部保留奇异值重合可以使 SVD 因子不唯一，但不破坏有截断间隔时的重构矩阵唯一性与局部光滑性。

**计算与退化。** 采用上面的切向因子，\(X+\eta\) 的行列空间分别包含在 \([U,U_\perp]\)、\([V,V_\perp]\) 的张成空间中。先 QR，再对至多 \(2r\times2r\) 核做 SVD，可把回缩成本组织为 \(O((m+n)r^2+r^3)\)，无需形成稠密 \(m\times n\) 增量；梯度生成、数据读取和残差计算的成本仍须另外计入。

\(\mathcal M_r\) 不闭；其闭包 \(\mathcal M_{\le r}\) 含较低秩层。这里 \(0<r<\min(m,n)\) 时，较低秩点是该秩约束代数集合的奇异点，不能延用同一固定维数切空间。\(\sigma_r(X)\to0\) 意味着正在接近秩下降；这与固定秩流形内部两个非零奇异值重合不同。自适应改秩应作为跨层操作另行定义。端点需另说：\(r=\min(m,n)\) 时，满秩集是开流形，而 \(\mathcal M_{\le r}\) 是整个矩阵空间；\(r=0\) 时两者均为零矩阵这个单点。

## GM03　低秩因子的规范与表达维数

**对象。** 用满列秩因子 \(L\in\mathbb R^{m\times r}\)、\(R\in\mathbb R^{n\times r}\) 表示 \(X=LR^\top\)。纤维中的等价表示为

\[
(L,R)\sim(LQ,RQ^{-\top}),\qquad Q\in GL(r).
\]

这解释了为什么因子参数量 \((m+n)r\) 不等于可表达矩阵集合的维数。相关商表示及度量选择见 GM-S2。

**从微分得到冗余。** 对 \(\pi(L,R)=LR^\top\)，

\[
d\pi_{(L,R)}(\delta L,\delta R)
=\delta L R^\top+L\delta R^\top.
\]

沿 \(Q(t)=I+t\Omega+O(t^2)\) 的规范变化给出垂直方向

\[
\mathcal V_{(L,R)}
=\{(L\Omega,-R\Omega^\top):\Omega\in\mathbb R^{r\times r}\}
=\ker d\pi_{(L,R)}.
\]

核等式可直接证明：先把 \(d\pi=0\) 左乘 \(I-LL^\dagger\)，利用 \(R\) 满列秩推出 \(\delta L=L\Omega\)，再代回得到 \(\delta R=-R\Omega^\top\)。故

\[
\dim\operatorname{im}d\pi=(m+n)r-r^2=r(m+n-r).
\]

**选择几何后再优化。** 水平空间是垂直空间在所选总空间度量下的补空间。简单的乘积 Frobenius 度量不对一般 \(GL(r)\) 规范保持不变：\((L,R)\mapsto(cL,c^{-1}R)\) 已会改变相同矩阵方向的因子范数。可采用规范不变度量，例如

\[
\bar g_{(L,R)}(\xi,\eta)
=\operatorname{tr}((L^\top L)^{-1}\xi_L^\top\eta_L)
+\operatorname{tr}((R^\top R)^{-1}\xi_R^\top\eta_R),
\]

并由此构造水平提升。代入 \(Q\) 变换并用 trace 循环性即可核对不变性。这是一种商度量选择；它一般不等于 GM02 的环境 Frobenius 诱导度量。

**算子分析用法。** 以 \(\delta X=d\pi[\delta L,\delta R]\) 判断真实更新，而不能把大的因子变化直接解释为大的算子变化。因子欧氏 SGD、加入 \(\|L\|_F^2+\|R\|_F^2\) 的正则化、以及商流形梯度法，定义不同的更新机制；这个正则项本身不对上述规范不变。进入因子秩不足区域后，核维数可能变化，不能继续套用满秩商结构和自由度公式。

## GM04　固定 TT 秩流形及唯一切向表示

**对象与前提。** 固定模式次序及可实现的精确展开秩
\(\mathbf r=(r_1,\ldots,r_{d-1})\)，取 \(r_0=r_d=1\) 且张量非零。定义

\[
\mathcal T_{\mathbf r}
=\{X:\operatorname{rank}X_{(1:k)\mid(k+1:d)}=r_k,\ k=1,\ldots,d-1\}.
\]

这里的精确秩由实际展开矩阵确定；仅给各核预留这些尺寸，不保证张量具有该秩。满秩 TT 表示写成

\[
X_{i_1\ldots i_d}
=\sum_{\alpha_1,\ldots,\alpha_{d-1}}
G^{(1)}_{1i_1\alpha_1}\cdots G^{(d)}_{\alpha_{d-1}i_d1}.
\]

Holtz–Rohwedder–Schneider 的固定 TT 秩流形与切向表示见 GM-S3。维数为

\[
\dim\mathcal T_{\mathbf r}
=\sum_{k=1}^d r_{k-1}n_kr_k-\sum_{k=1}^{d-1}r_k^2.
\]

**把维数接回收缩。** 每个内部 bond 上可插入 \(Q_kQ_k^{-1}\)，保持张量不变。若以矩阵切片 \(G^{(k)}(i_k)\) 表示核，则

\[
G^{(k)}(i_k)\mapsto Q_{k-1}^{-1}G^{(k)}(i_k)Q_k,
\quad Q_0=Q_d=1,
\]

相应无穷小垂直变化是
\(\delta G^{(k)}=-\Omega_{k-1}G^{(k)}+G^{(k)}\Omega_k\)。把各核变化代入全收缩，邻接项逐对消去。满秩前提下，去掉的规范自由度为 \(\sum_{k<d}r_k^2\)，不能当成张量本身的表达方向。

**唯一切向坐标。** 用 QR／SVD 取得左正交核，对 \(k<d\) 令左展开 \(G_k^L\in\mathbb R^{(r_{k-1}n_k)\times r_k}\) 满足 \((G_k^L)^\top G_k^L=I\)。每个切向量唯一表示为

\[
\eta=\sum_{k=1}^d
\operatorname{contract}(G^{(1)},\ldots,\delta G^{(k)},\ldots,G^{(d)}),
\quad (G_k^L)^\top\delta G_k^L=0\quad(k<d),
\]

最后一核不施加该条件。它删除了在相邻核之间重复分配同一张量变化的自由度。环境 Frobenius 内积仍由左右环境收缩确定；单纯左正交并不自动使所有 \(\|\delta G^{(k)}\|_F^2\) 之和等于 \(\|\eta\|_F^2\)。

**实现边界。** 可用核收缩与环境复用计算方向导数、梯度投影和局部更新；具体复杂度需给出核尺寸、收缩顺序与重正交成本。固定秩流形与“秩至多为 \(\mathbf r\)”的集合不同；秩下降时要改用相应层的描述。TT 截断的误差与算法见[张量分解](tensor-decompositions.md)，不能仅凭存储格式就把任意截断宣称为当前流形上的光滑回缩。模式重排也会改变这些展开秩。

## GM05　正则约束、KKT 与约束曲率

**对象与前提。** 在欧氏环境中，设

\[
M=\{x\in\mathbb R^n:c(x)=0\},\qquad
A=Dc(x)\in\mathbb R^{k\times n},
\]

\(c\) 在邻域光滑，\(A\) 在当前可行点满行秩。于是局部 \(\dim M=n-k\)、\(T_xM=\ker A\)，切投影为

\[
P=I-A^\top(AA^\top)^{-1}A.
\]

只求投影作用时解 \((AA^\top)\alpha=Av\)，返回 \(v-A^\top\alpha\)；数值条件差时优先考虑 QR／SVD 或结构化约束求解，不必显式形成逆和稠密 \(P\)。

**梯度与 Hessian。** 对环境延拓 \(\bar f\) 及限制 \(f=\bar f|_M\)，定义

\[
\lambda(x)=(AA^\top)^{-1}A\nabla\bar f(x),
\qquad \operatorname{grad}f=P\nabla\bar f.
\]

对 \(\eta\in T_xM\)，

\[
\operatorname{Hess}f(x)[\eta]
=P\left[\nabla^2\bar f(x)-\sum_{a=1}^k\lambda_a(x)\nabla^2c_a(x)\right]\eta.
\]

推导：对 \(\operatorname{grad}f=\nabla\bar f-\sum_a\lambda_a\nabla c_a\) 求方向导数，再投影；\(D\lambda_a[\eta]\nabla c_a\) 为法向项，投影后消失。该公式在正则可行点成立，不只在驻点成立。这里 Hessian 中的 \(\lambda(x)\) 作为该点系数使用；不是对复合函数 \(x\mapsto\bar f(x)-\lambda(x)^\top c(x)\) 直接取环境 Hessian。来源 GM-S4，§7.7 式 (7.77)–(7.81)。

**与 KKT、第二基本形式连接。** 驻点条件 \(P\nabla\bar f=0\) 等价于 \(\nabla\bar f=A^\top\lambda\)。取拉格朗日函数 \(\mathcal L(x,\lambda)=\bar f(x)-\lambda^\top c(x)\)，其固定乘子 Hessian 限制在切空间上就是上述 Riemannian Hessian。严格正定给出严格局部极小的充分条件，半正定是局部极小的必要条件；半正定本身不保证极小。

沿可行曲线 \(\gamma\)，两次微分 \(c(\gamma)=0\) 得

\[
A\ddot\gamma+\big(\dot\gamma^\top\nabla^2c_a\dot\gamma\big)_{a=1}^k=0.
\]

这决定曲线加速度的法向部分，即嵌入的第二基本形式；对应的 \(\lambda\nabla^2c\) 项解释了为何环境 Hessian 为零也可以有约束曲率。

**球面验算。** 取 \(c(x)=(\|x\|^2-1)/2\)、\(\bar f(x)=a^\top x\)，则 \(\lambda=a^\top x\)，

\[
\operatorname{grad}f=a-(a^\top x)x,
\qquad\operatorname{Hess}f[\eta]=-(a^\top x)\eta.
\]

在 \(x=-a/\|a\|\)（\(a\ne0\)）处 Hessian 为 \(\|a\|I\) 作用于切空间，符合严格极小。若约束 Jacobian 不满行秩，须区分约束冗余、约束写法退化和集合奇异性；不能仅将逆替换为伪逆便宣告以上正则推导仍成立。

## GM06　概率单纯形、Fisher 拉回与 softmax 冗余

**对象。** 在固定有限支持上的严格正概率单纯形

\[
\Delta_{n-1}^{\circ}=\{p_i>0,\ \sum_i p_i=1\},
\qquad T_p\Delta^{\circ}=\{u:\sum_i u_i=0\},
\]

Fisher 度量为

\[
g_p(u,v)=\sum_i\frac{u_iv_i}{p_i}.
\]

由 categorical score \(u_i/p_i\) 的期望乘积直接得到此式，它在切空间正定。设 \(\psi_i(p)=2\sqrt{p_i}\)，则 \(\|\psi(p)\|^2=4\) 且

\[
\langle d\psi_p[u],d\psi_p[v]\rangle
=\sum_i u_iv_i/p_i=g_p(u,v).
\]

因此可把单纯形内部等距表示为半径 \(2\) 球面上各坐标均为正的区域。这是本例的直接构造；概率边界 \(p_i=0\) 不属于上述图册和度量域。

**可直接执行的梯度。** 若 \(a_i=\partial\bar f/\partial p_i\) 是环境延拓的梯度分量，则

\[
(\operatorname{grad}^{F}f)_i
=p_i\left(a_i-\sum_jp_ja_j\right).
\]

该向量的分量和为零，且对任意 \(u\in T_p\Delta^\circ\)，有 \(g_p(\operatorname{grad}^{F}f,u)=a^\top u=df[u]\)，故不依赖延拓选择。有限步相加仍须保证正概率，或选定该流形上的回缩／指数映射。

**softmax 的冗余不是新增表达维数。** 对 \(p_i=e^{z_i}/\sum_j e^{z_j}\)，

\[
J_z=\operatorname{diag}(p)-pp^\top,
\qquad
F_z=J_z^\top\operatorname{diag}(p)^{-1}J_z
=\operatorname{diag}(p)-pp^\top.
\]

\(\ker F_z=\operatorname{span}\{\mathbf1\}\)，对应 \(z\sim z+\alpha\mathbf1\)。可固定 \(z_n=0\)，或固定 \(\sum_i z_i=0\)，得到 \(n-1\) 个可辨识坐标。需区分这个精确不变方向与概率极小时出现的数值病态。

对任意光滑参数化 \(p=p(\theta)\)，令 \(J_\theta=Dp(\theta)\)，则

\[
F_\theta=J_\theta^\top\operatorname{diag}(p)^{-1}J_\theta,
\qquad \ker F_\theta=\ker J_\theta.
\]

因此只有在参数微分单射时，拉回才是整个参数空间上的正定 Riemannian 度量。恒秩区域可用局部纤维／可辨识坐标处理；全局商空间是否光滑仍要验证。一般条件模型还包含对输入分布的期望，不能把单个输入下的核直接当作总体 Fisher 的核。

**与 A6 的分工。** 本卡提供分布流形和 softmax 的具体构造。Fisher、经验 Fisher、GGN 的区别，奇异线性系统可解性、伪逆选择、阻尼及有限步重参数化边界见 A6；不得由 \(F_\theta^{\dagger}\nabla f\) 自动断言参数向量按任意重参数化协变。来源 GM-S5，§5–6、§12。

## GM07　沿流形的有限变化与 Taylor 误差

**标量函数。** 设 \(f\in C^2(M)\)，选定 Riemannian 度量，令 \(\gamma(t)=\operatorname{Exp}_x(tv)\) 在 \([0,1]\) 上有定义，\(y=\gamma(1)\)。若要由端点唯一取得 \(v=\operatorname{Log}_x(y)\)，还需选在相应正常邻域内。沿 \(\gamma\) 的 Levi–Civita 平行运输记为 \(P_t:T_xM\to T_{\gamma(t)}M\)，于是 \(\dot\gamma(t)=P_tv\)。一维 Taylor 积分公式给出

\[
f(y)=f(x)+df_x[v]
+\int_0^1(1-t)\operatorname{Hess}f_{\gamma(t)}(P_tv,P_tv)\,dt.
\]

若沿曲线 \(\|\operatorname{Hess}f\|_{\mathrm{op}}\le H\)，余项绝对值不超过 \(H\|v\|_x^2/2\)。这需要沿实际路径的界，而非只测量锚点 Hessian。原领域接口见 GM-S4，§10.4。

对一般曲线则有

\[
\frac{d^2}{dt^2}f(\gamma(t))
=\operatorname{Hess}f(\dot\gamma,\dot\gamma)
+df\left[\nabla_{\dot\gamma}\dot\gamma\right].
\]

因此用回缩曲线替换测地线时，要保留加速度项；二阶回缩使该项在初点为零，但不意味着整条回缩曲线都是测地线。驻点的 \(df=0\) 也只消掉该点的相应项。

**输出在固定向量空间。** 对 \(F:M\to W\)，\(W\) 为已固定坐标识别的向量空间，端点差和积分均在 \(W\) 中合法：

\[
F(y)-F(x)
=\left[\int_0^1 dF_{\gamma(t)}\circ P_t\,dt\right]v
=\bar J_{x,v}v,
\quad \bar J_{x,v}:T_xM\to W.
\]

于是锚点算子 \(A_x:T_xM\to W\) 的精确误差为

\[
F(y)-\big(F(x)+A_xv\big)=(\bar J_{x,v}-A_x)v.
\]

这把[导数与割线](derivatives-and-secants.md)中的方向误差推广到曲线路径。欧氏直线与恒等运输恢复原来的平均 Jacobian 公式；参数、外部条件和逻辑轴身份必须在该比较中保持定义一致。

**输出本身也是流形。** 一般 \(F:M\to N\) 的微分具有类型

\[
dF\in\Gamma(T^*M\otimes F^*TN),
\qquad dF_q:T_qM\to T_{F(q)}N.
\]

把输入、输出纤维运输到锚点后，Jacobian 可放入同一个 Hom 空间比较。但沿输出曲线平行拉回速度再积分通常得到曲线的 development，不能自动写成 \(\operatorname{Log}_{F(x)}F(y)\)。若输出曲线始终处于基点 \(b=F(x)\) 的一个正常邻域，可明确取向量空间值函数 \(G=\operatorname{Log}_b\circ F\)，再由链式法则得到

\[
\operatorname{Log}_bF(y)
=\int_0^1
(d\operatorname{Log}_b)_{F(\gamma(t))}
\circ dF_{\gamma(t)}\circ P_t[v]\,dt.
\]

这里 \(d\operatorname{Log}_b\) 一般不等于平行运输。若 \(N\) 已嵌入环境向量空间，也可计算环境端点差，但必须把它解释为环境误差，不能直接当作内蕴对数向量。

## GM08　跨点方向、算子运输与缓存

**对象与类型。** 取连接 \(x\) 与 \(y\) 的指定曲线，令 \(P:T_xM\to T_yM\) 为平行运输。比较向量场 \(V\) 时使用

\[
P^{-1}V(y)-V(x)\in T_xM.
\]

对端点切空间上的算子 \(A_x:T_xM\to T_xM\)、\(A_y:T_yM\to T_yM\)，使用

\[
\widetilde A_y=P^{-1}A_yP,
\qquad \Delta A=\widetilde A_y-A_x.
\]

一般输入、输出丛不同，则用 \(S^{-1}A_yP\)，其中 \(P\) 运输输入纤维，\(S\) 运输输出纤维。路径、联络、基点与标架是缓存语义的一部分；这些识别规则改变时，数组差不一定反映同一个几何差。

**可计算的变化界。** 对光滑算子场沿曲线 \(\gamma\)，记 \(P_t\) 为从起点到当前点的平行运输，则

\[
\frac{d}{dt}\left(P_t^{-1}A_{\gamma(t)}P_t\right)
=P_t^{-1}(\nabla_{\dot\gamma(t)}A)P_t.
\]

当运输为等距映射时，

\[
\|P_1^{-1}A_yP_1-A_x\|_{\mathrm{op}}
\le\int_0^1\|\nabla_{\dot\gamma(t)}A\|_{\mathrm{op}}\,dt.
\]

所以若缓存的是几何算子，刷新判据可针对运输后的变化、实际输入方向和下游误差建立。使用便宜的向量运输替代平行运输时，需另计其失真；运输不等距时不能直接照搬上面的范数等式和界。参见 GM-S4，§10.3–10.4、§10.7。

**球面反例可核验实现。** 在单位球面上，对非对跖端点 \(x,y\)，沿短大圆弧的平行运输为

\[
P_{x\to y}v
=v-\frac{v^\top y}{1+x^\top y}(x+y),
\qquad x^\top v=0.
\]

可通过将 \(v\) 分解为大圆切方向与其正交方向推导，检查 \(y^\top Pv=0\) 且 \(\|Pv\|=\|v\|\)。令 \(x=e_1,y=e_2,v=e_2\)，得到 \(Pv=-e_1\)；直接投影 \((I-yy^\top)v=0\)。因此“投到新切空间”可以是算法选用的向量运输，却不是这个平行运输，也不保持方向长度。

**接回普通 ML 张量。** 固定环境空间中的 activation、权重或 Jacobian 数组，只要轴身份和输入输出空间一致，普通差分已经合法；不应为它们强加一个未经建模的联络。只有当对象确实取值于变化的切空间／丛，或需要参数化不变的比较时，才加入运输。几何近似是否节省时间还须核算坐标恢复、QR／SVD、运输、状态更新与实际 kernel 成本；本卡不据几何结构推断硬件加速。

## 已核读来源与定位

以下是本文件使用的领域来源与实际阅读范围；工具卡中的小例、核空间证明、奇异值间隔充分条件和链式法则改写按本文约定直接推导，不是逐段转述原文。访问日期：2026-11-16。

| 编号 | 一手来源与所读版本 | 核读位置与用途 |
| --- | --- | --- |
| GM-S1 | Bart Vandereycken, *Low-Rank Matrix Completion by Riemannian Optimization*, SIAM J. Optim. 23(2), 1214–1236 (2013)，[作者托管正式论文](https://www.unige.ch/math/vandereycken/papers/published_Vandereycken_2013.pdf)，DOI 10.1137/110845768 | §2.1 Proposition 2.1，式 (2.3)、(2.5)：固定秩维数、切空间、投影；§2.2 式 (2.7)–(2.8)：局部截断 SVD 回缩。GM02 的截断间隔按保留／舍弃边界 \(\sigma_r>\sigma_{r+1}\) 表述。 |
| GM-S2 | B. Mishra, G. Meyer, S. Bonnabel, R. Sepulchre, *Fixed-rank matrix factorizations and Riemannian low-rank optimization*, Computational Statistics 29, 591–621 (2014)，[arXiv 记录](https://arxiv.org/abs/1209.0430)，[核读作者稿镜像](https://citeseerx.ist.psu.edu/document?doi=d673e4dac43c9a21d3bd48ea6084d3b4c40db3fa&repid=rep1&type=pdf) | 所读稿首页日期 2014-05-02，并带 arXiv:1209.0430v2 标记。§3.1 式 (4)–(6)：满秩因子的 GL 商；§4.2–4.4：水平表示与商度量；式 (20)–(21)：欧氏因子度量与规范不变度量的区别。 |
| GM-S3 | Sebastian Holtz, Thorsten Rohwedder, Reinhold Schneider, *On Manifolds of Tensors of Fixed TT-Rank*，[原始预印本全文](https://citeseerx.ist.psu.edu/document?doi=8f33b04869733a5ecc92e9b634f95546b674e8f6&repid=rep1&type=pdf)；正式论文 Numer. Math. 120, 701–731 (2012)，DOI 10.1007/s00211-011-0419-7 | 核读版本为 DFG SPP 1324 Preprint 61，2010-12-22。Theorem 3.1 与 Remark 3.2：精确 TT 秩、正交表示与模式次序；Lemma 4.1 式 (4.2)：维数；Theorem 4.2、Remark 4.3 式 (4.4)–(4.7)：唯一切向规范。 |
| GM-S4 | Nicolas Boumal, *An Introduction to Optimization on Smooth Manifolds*, Cambridge UP (2023)，[作者公开稿](https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf) | 核读稿封面标明勘误／澄清更新至 **2026-10-08**，不是未经更新的 2023 原稿；其章节和公式编号与出版版相同，页码不同。§7.7 式 (7.72)–(7.81)：正则约束、投影与 Hessian；§10.4 Definition 10.49、Propositions 10.50–10.53、式 (10.24)–(10.25)：运输后的算子比较与 Taylor；§10.3、§10.7 是进一步使用运输和张量场的入口。 |
| GM-S5 | James Martens, *New Insights and Perspectives on the Natural Gradient Method*, JMLR 21(146), 1–76 (2020)，[正式论文](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf) | §5 式 (3)–(4)：Fisher 定义；§6：KL 的局部二次形式与几何解释；§12 Corollary 2、§12.1：可逆性前提、有限步和重参数化的范围。GM06 的 categorical／softmax 公式直接由这些定义计算。 |
| GM-S6 | Charles Fefferman, Sanjoy Mitter, Hariharan Narayanan, *Testing the Manifold Hypothesis*, JAMS 29(4), 983–1049 (2016)，[作者托管正式论文](https://mitter.lids.mit.edu/publications/121_Testing_Manifold.pdf)，DOI 10.1090/jams/852 | §1，正式页 984–986：均方距离目标 \(L(M,P)\)、i.i.d. 采样、维数／体积／reach 限制以及有容差的检验陈述。这里借用的是问题定义与结论范围，没有声称核读其全部算法证明。 |
