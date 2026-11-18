# 全 80 读出的状态与变化张量：从已有视频观察到可计算的新算子

日期：2026-11-18。分析基点：c9e231e5fbce231ddfb3be08ed9b5cacd2634e75；当前原生读出源码：3e4f7ba806644f5aaaafce1c3c78df8eebc3f035。本文使用 operator-tensor-analysis，给出 ALG/SPEC 构造与静态成本。视频依据是用户对既有完整视频的连续观察；本轮新增模型、CAL 拟合与视频运行均为 0。

**核心进展：保留全 80 的锚点、完整输出观测和共享读出，把新的表示预算用于“状态差异与共同变化如何相互作用”。一个可直接写入现有接口的构造是在 32 列原生割线后追加 16 列极化双线性割线，仍使用 64 行完整 FFN 观测；其新增主要 MAC 为当前读出的约 0.2105%。较宽原生响应压缩与 Dense 锚点上下文，为同一类状态调制算子提供另外两种具体取材方式。**

## 1. 修正研究判断，并从已有成功能力出发

撤回前报告依据 geometry64 的局部 MSE 降低其视频修正优先级的判断，也撤回把缺少普遍终端保证作为关系构造进入研究的门槛。原始数值、统计对象和计算成本继续保留；它们用于描述或构造算子参数。视频方案的质量判断依据其真实视频表现。

用户观察提供了明确目标：总体运动幅度相近，主要损坏表现为本应保持一体的区域在共同运动中抖动、解离；0392 的全画幅交叠运动尤其明显。P64 与全 80 在线读出较好保留了这些变化，相对其他近似方法损坏较小，仍有可见损坏。Dense 本来的交叠、重影和晕眩感属于应保留的目标内容。[人审补充](../../reports/20261118/observed_secant_ffn_operator/analysis/HUMAN_REVIEW_ADDENDUM.json)

全 80 已经提供三个可继续发展的结构：全部专家层位置采用同一种修正方式；当前 64 行完整 FFN 响应决定输出适配；同一 hit 的整个 latent 视频张量共享一个完整输出映射。由这些结构出发，本轮研究以下正面问题：

> 怎样用少量共享坐标表达“共同变化作用于不同区域状态”所需的相对响应，并继续使用现有完整观测对其校准？

## 2. 固定对象、轴和现有割线

每一专家、每一层分别讨论。token 轴 n 展平 latent 时间与空间，N=21×29×52=31668；d=5120，隐藏宽度 m=13824。去噪端点 0、1 与 latent 视频时间轴是不同对象。以 token 行向量记输入：

\[
F(U)=\phi(UW_1^\top+b_1)W_2^\top+b_2,
\quad W_1:[m,d],\quad W_2:[d,m].
\]

这里的 \(\phi\) 是当前原生 GELU(tanh)。令 \(w_h=W_{1,h,:}^{\top}\) 为输入列方向，\(b_h=W_{2,:,h}^{\top}\) 为输出行方向，则

\[
F(u)=\sum_{h=1}^{m}\phi(uw_h+b_{1,h})b_h+b_2.
\]

现有全 80 算子为

\[
\widehat F(u_1)
=F(u_0)+\Delta u A+\Delta\Phi_{\rm old}V_s,
\quad
\Delta\Phi_{\rm old}=\Phi_{\rm old}(u_1)-\Phi_{\rm old}(u_0).
\]

\(\Phi_{\rm old}:[d]\to[32]\) 由 16 个原生通道的配对特征形成；\(V_s:[32,d]\) 由当前 64 行完整 FFN 响应适配。s 表示本次去噪 hit 的条件，所有 N 行共享 V_s。32 表示当前有效宽度的上限；后文 48 维与缓存按这一满维预算核算。

定义 \(f_s(u)=uA+\Phi_{\rm old}(u)V_s\)，则

\[
\widehat F(u_1)=F(u_0)+f_s(u_1)-f_s(u_0).
\]

这个表达保留当前已经有效的锚定有限变化。新的特征可以直接加入同一个 f_s，再通过同一完整输出读出执行。[原设计](OBSERVED_SECANT_OPERATOR_DESIGN.md) · [当前算子与成本](READOUT_COST_COHERENCE_REASSESSMENT.md)

## 3. 关键对象：状态差异与共同变化的混合响应

对两个 token 的四个真实端点，定义

\[
\mathcal C_{ij}[F]
=F(u_{1,i})-F(u_{0,i})
-F(u_{1,j})+F(u_{0,j}).
\]

它表示两处 FFN 写入的相对变化。为展开张量结构，先取共同输入增量 v，起点状态差 a：

\[
\mathcal C[F]
=F(u+a+v)-F(u+a)-F(u+v)+F(u).
\]

两次微积分基本定理给出

\[
\boxed{
\mathcal C[F]
=\int_0^1\!\!\int_0^1
D^2F(u+\alpha a+\beta v)[a,v]\,
d\alpha\,d\beta.
}
\]

完整 FFN 的相应收缩为

\[
\boxed{
D^2F(u)[a,v]
=\sum_h\phi''(uw_h+b_{1,h})
(aw_h)(vw_h)b_h.
}
\]

这里两个输入槽分别承接区域状态差 a 和共同变化 v，输出槽保留完整 d 维响应。完整 FFN 已经通过这类乘积决定：相近的变化在不同状态处应怎样产生有区别而协调的响应。

因此可以把下一步的表示目标明确为：用少量可计算特征承载这个混合张量的作用。真实视频的一般情形直接使用四端点；共同 v 是展示张量结构的分析情形。

## 4. 主构造：极化双线性割线

### 4.1 算子与精确有限差分

冻结 CAL 中心 \(\mu:[d]\) 与两组方向 \(P,Q:[d,b]\)，取 b=16 作为本次具体预算实例，定义

\[
\psi(u)=((u-\mu)P)\odot((u-\mu)Q):[b].
\]

扩展现有读出：

\[
\boxed{
\widehat F(u_1)=F(u_0)+\Delta uA
+\Delta\Phi_{\rm old}V_s+\Delta\psi B_s,
\quad B_s:[b,d].
}
\]

令 \(\bar u=(u_1+u_0)/2-\mu\)。乘积差分给出精确恒等式

\[
\boxed{
\Delta\psi
=(\Delta uP)\odot(\bar uQ)
+(\bar uP)\odot(\Delta uQ).
}
\]

新特征直接表达“本次变化 × 路径中点状态”。对固定的当前 B_s，每个新增割线方向为

\[
r_j(\bar u)
=P_j(\bar uQ_j)+Q_j(\bar uP_j),
\]

随当前两个端点而变化。它映射到完整输出的方式由 B_s 在线决定。

参数取 B_s=0 时，扩展函数族包含现有全 80 函数族；在线实际值由扩展系统联合拟合。两个端点相同时，所有特征差为零，实数算术中的锚点一致性保持。

### 4.2 对两个相关区域，新增能力可以精确写出

若两处具有共同增量 v，起点差为 a，则

\[
\boxed{
(\Delta\psi_i-\Delta\psi_j)B_s
=\big[(aP)\odot(vQ)+(vP)\odot(aQ)\big]B_s.
}
\]

这给原有状态差与共同变化建立了一个共享的有符号响应关系。保持区域关系所需的非零相对响应，直接由该式表达。

对实际可能不同的增量 \(v_i,v_j\)，记

\[
\delta v=v_i-v_j,\quad \bar v=(v_i+v_j)/2,\quad
a=\bar u_i-\bar u_j,\quad m=(\bar u_i+\bar u_j)/2.
\]

则

\[
\begin{aligned}
\Delta\psi_i-\Delta\psi_j
={}&(\delta vP)\odot(mQ)+(mP)\odot(\delta vQ)\\
&+(\bar vP)\odot(aQ)+(aP)\odot(\bar vQ).
\end{aligned}
\]

前两项承接“相对变化与共同状态”，后两项承接“共同变化与状态差”。它们共同给出区域协同变化的可计算表示。

### 4.3 用原生 FFN 张量初始化 P、Q

令 S 为当前 16 个原生输入方向张成的空间，\(\Pi_S\) 是其正交投影。将完整 FFN 的任一输入方向分解为

\[
w_h=s_h+t_h,\qquad
s_h=\Pi_Sw_h,\quad t_h=(I-\Pi_S)w_h.
\]

取

\[
P_h=t_h,\qquad Q_h=s_h+\tfrac12t_h,
\]

则

\[
\boxed{
P_hQ_h^\top+Q_hP_h^\top
=w_hw_h^\top-s_hs_h^\top.
}
\]

所以一个极化特征精确承载该原生通道在 S 外部及跨 S 的补充曲率几何。输出系数和状态强度继续由 CAL 与当前完整观测确定。新分支的三阶张量为

\[
\mathscr H^{\rm add}_{abc}
=\sum_{j=1}^{b}
(P_{aj}Q_{bj}+Q_{aj}P_{bj})(B_s)_{jc}.
\]

实际计算只收缩两个 d×b 因子和一个 b×d 读出。

这里 P 位于 S 的正交补是一种有明确来源的初始化。当前 native 曲率占据的是 \(\operatorname{span}\{w_hw_h^\top:h\in\mathcal K\}\)；S 内不同方向之间的交叉项也是可加入的结构。因此后续 CAL 可以允许 P、Q 在全输入空间内优化，或专门加入 S 内交叉因子，把外部和内部混合作用一起纳入表示。

### 4.4 一个可执行的参数起点

构造原生互补原子

\[
R_h=w_hw_h^\top-s_hs_h^\top,\qquad
\mathscr T_h=R_h\otimes b_h.
\]

它们的内积可直接计算：

\[
\boxed{
\langle\mathscr T_h,\mathscr T_g\rangle
=\big[(w_h^\top w_g)^2-(s_h^\top s_g)^2\big]
(b_hb_g^\top).
}
\]

因此可以在全原生通道字典上按需计算 Gram 列，做确定性主元 Cholesky 选择得到 b 个不同的响应方向，无需物化 d×d×d 张量，也无需建立完整 m×m Gram。若使用已有 CAL 状态的曲率签名 \(\kappa_h=\phi''(z_h)\)，再乘来源等权的 \(\langle\kappa_h,\kappa_g\rangle_{\rm CAL}\)，就把状态分布也放入原子内积。选中通道给出上面的 P、Q；中心和尺度由 CAL 固定。

可先对旧方向做 QR 得正交基 E，预计算 \(W_1E:[m,\dim S]\)。一个 Gram 列只需 \(W_1w_h\)、\((W_1E)(E^\top w_h)\) 和 \(W_2^\top b_h^\top\) 三组收缩。按主元剩余平方范数选择，平分时使用原生通道 ID 定序；数值零秩按实际有效列数处理。基础初始化只需模型权重与旧通道集合，CAL 加权版再使用相应激活曲率。新的 Dense 输出标签由现有在线 64 行继续提供。

初始化保留旧 prior，并令新增 prior 为零：

\[
\widetilde V_0=\begin{bmatrix}V_0\\0\end{bmatrix}.
\]

这提供无需先完成非凸张量拟合的起点。进一步离线优化可以拟合实际四端点混合差分，或上述二阶方向作用；新特征在 CAL 上先消去与旧特征重复的列分量，再定标，也可把预算更多用于新增结构。这些计算负责生成参数，视频方向的取舍继续依据视频。

### 4.5 缓存和在线解

anchor 计算并缓存 \(\psi(U_0):[N,16]\)；hit 计算 \(\psi(U_1)\) 后相减。无需显式保存中点或完整 U0。

令

\[
\widetilde X=[\Delta\Phi_{\rm old},\Delta\psi]:[N,48],
\quad
\widetilde V_s=\begin{bmatrix}V_s\\B_s\end{bmatrix}:[48,d].
\]

新增特征用 CAL 的正尺度定标；零 RMS 列使用预先定义的正尺度底。一个完整的数值定义是

\[
\Gamma_{\rm CAL}
=\operatorname{blockdiag}(\Gamma_{\rm old},I_b)\succ0,
\qquad \lambda>0,\quad\tau>0.
\]

旧分支保留已有有效 CAL 坐标，新分支保留所构造坐标；实际系统宽度为 \(r_{\rm old}+b\le48\)。这个正定度量使联合求解直接适用于有限 CAL 面板中相关或重复的特征列。它是新算子的明确正则定义；若采用全联合 CAL 换基，则同时变换先验与度量，保持所定义的目标。

设 \(\mathcal P\) 为 64 行观测位置，\(D_{\mathcal P}\) 是相对于扩展静态先验的完整 FFN 创新。沿用当前共享读出形式：

\[
H=\widetilde X^\top\widetilde X/N+\tau^2\Gamma_{\rm CAL},
\]

\[
\left(
\widetilde X_{\mathcal P}^{\top}\widetilde X_{\mathcal P}/p
+\lambda H
\right)\Delta\widetilde V
=\widetilde X_{\mathcal P}^{\top}D_{\mathcal P}/p.
\]

这是一个 48×48 系统、d 个右端。两类特征拼成一次输出 GEMM，使新增分支复用当前完整输出的写回路径。生产实现要把扩展特征尺度、先验、CAL 度量与现有 BF16 舍入位置一起定义；本文给出实数算子与接口，尚未产生新的参数包。

## 5. 第一种互补取材：较宽原生非线性先计算，再联合压缩

P64 的原生非线性计算提示了一个可迁移结构：让较多原生激活共同参与有限响应，再压缩它们的联合输出表示。这里可把感知宽度 q 与缓存/读出宽度 r 分开设计。

固定 \(\mathcal J\supset\mathcal K\)，取 q=256、r=32 作为具体预算实例：

\[
z_{\mathcal J}=uW_{1,\mathcal J}^{\top}+b_{\mathcal J},\quad
\rho_{\mathcal J}=\phi(z_{\mathcal J})-\beta_{\mathcal J}\odot z_{\mathcal J},
\]

\[
\boxed{
\Phi_{\rm wide}(u)
=[z_{\mathcal J},\rho_{\mathcal J}]
\begin{bmatrix}C_z\\C_\rho\end{bmatrix},
\quad C_z,C_\rho:[256,32].
}
\]

每个压缩坐标可以汇集 256 个原生 GELU 响应。运算顺序是先逐通道 GELU，再线性压缩。旧构造通过把新增通道对应的压缩矩阵行置零嵌入该函数族。

设原生有限斜率

\[
\bar\beta_h=\int_0^1\phi'(z_{0,h}+t\Delta z_h)\,dt.
\]

则

\[
\boxed{
\Delta\Phi_{\rm wide}
=\Delta uW_{1,\mathcal J}^{\top}
\big[C_z+\operatorname{diag}(\bar\beta-\beta)C_\rho\big].
}
\]

256 个方向通过各自当前的有限斜率，共同形成 32 个随状态变化的有效输入方向。固定当前 V_s 后，一个输出坐标的 Hessian 为

\[
\nabla^2 f_{s,c}(u)
=\sum_{h\in\mathcal J}
\phi''(z_h)(C_\rho V_s)_{hc}w_hw_h^\top.
\]

**输入曲率的原生覆盖宽度和在线读出的坐标数是两个独立自由度。** 一个压缩坐标也可以汇集许多个原生曲率项；其状态依赖由实际两个端点的 GELU 计算提供。

一种 CAL 初始化是按来源等权的平方根权重堆叠宽原生差分 Z 与 Dense 割线残差 R，对组合矩阵 \(B=CV\) 求解

\[
\min_{\operatorname{rank}(B)\le r}
\|ZB-R\|_F^2/M+\eta\|B\|_F^2,\quad\eta>0.
\]

令

\[
G=Z^\top Z/M+\eta I,\quad D=Z^\top R/M,\quad
G^{-1/2}D=L\Sigma Q^\top,
\]

\[
C_{\rm init}=G^{-1/2}L_r,\qquad
V_{\rm init}=\Sigma_rQ_r^\top.
\]

此式对应共同欧氏输出度量的低秩岭回归初始化。随后在实际 BF16 特征程序上，按现有来源 gate 拟合读出与 CAL 度量。宽字典从全原生通道构造，包含旧 16 通道；其新增部分可用原生响应原子的按需 Gram 主元选择，联合压缩保留不同通道的有符号作用。

这一路线直接使用原 FFN 非线性形状。§4 的极化特征直接构造状态与变化的混合作用。二者可以分别研究，也可以拼接为 \([\Phi_{\rm wide},\psi]\) 后共享完整输出读出。

## 6. 第二种互补取材：让 Dense 锚点成为响应上下文

Dense anchor 已经为全视频面板产生 \(Y_0=F(U_0)\)。取冻结投影 \(R_c:[d,4]\)，形成

\[
c_0=(Y_0-\mu_Y)R_c:[N,4].
\]

给现有割线 \(X:[N,32]\) 增加交互

\[
J_{na}=\sum_{r=1}^{32}\sum_{q=1}^{4}
X_{nr}c_{0,nq}T_{rqa},
\quad T:[32,4,16].
\]

用 \([X,J]:[N,48]\) 做同一个共享读出。这给出“当前割线方向 × 锚点完整响应上下文 × 输出通道”的三阶张量。在线仅更新 48×d 读出，R_c 与 T 由 CAL 冻结。

其两位置关系满足

\[
\begin{aligned}
X_i\otimes c_i-X_j\otimes c_j
={}&(X_i-X_j)\otimes(c_i+c_j)/2\\
&+(X_i+X_j)/2\otimes(c_i-c_j).
\end{aligned}
\]

第一项表达共同上下文中的相对变化；第二项表达共同变化中应保留的状态差。Dense 锚点因此从保存函数值的状态，进一步成为调制当前响应的连续坐标。

在未压缩、使用普通欧氏特征内积时，

\[
\langle[X_i,X_i\otimes c_i],[X_j,X_j\otimes c_j]\rangle
=\langle X_i,X_j\rangle(1+\langle c_i,c_j\rangle).
\]

这说明共同响应可以按“变化与上下文的联合关系”组织。部署中的 CAL 压缩与正则继续定义实际读出几何。

CAL 可先从 Y0 得到四维连续坐标，再在 \(X\otimes c_0\) 中投影掉 X 已覆盖的成分，用与完整响应/混合差分的交叉协方差提取 16 个方向，得到 T。只缓存四列 c0，已有完整 Y0 在 anchor 使用后即可按原生命周期释放。

## 7. 让完整观测同时校准点值和真实相对响应

新增张量也允许在同一 64 行观测中，明确拟合 Dense 的有符号关系。令 B_P 为这些观测之间的差分矩阵，W 为非负边权，\(\gamma\ge0\)：

\[
M=I+\gamma B_P^\top WB_P.
\]

这里 \(\Psi\) 表示当前 hit 的完整差分设计矩阵：极化分支用 \([\Delta\Phi_{\rm old},\Delta\psi]\)，宽原生分支用 \(\Delta\Phi_{\rm wide}\)，上下文分支用 \([X,J]\)。各候选在自己的 CAL 定标坐标中指定正定的 \(\Gamma_{\rm CAL}\)，并取正的 \(\lambda,\tau\)。分别定义

\[
R_P=F(U_{1,P})-Y_{{\rm base},P},\quad
D_P=R_P-\Psi_PV_0,\quad
V=V_0+\Delta V,\quad
H=\Psi^\top\Psi/N+\tau^2\Gamma_{\rm CAL}.
\]

在线更新为

\[
\boxed{
\left(\Psi_P^\top M\Psi_P/p+\lambda H\right)\Delta V
=\Psi_P^\top M D_P/p.
}
\]

关系项的目标是

\[
B_P(\Psi_PV-R_P),
\quad R_P=F(U_{1,P})-Y_{{\rm base},P}.
\]

因此每条关系都保留 Dense 的非零相对响应。对一对观测的误差，定义

\[
e_+=(e_i+e_j)/\sqrt2,\qquad
e_-=(e_i-e_j)/\sqrt2.
\]

取这条边的权重为 1、仅计此对的点值与关系项，对应损失精确等于

\[
\|e_+\|^2+(1+2\gamma)\|e_-\|^2.
\]

一般边权 w 时，反对称分量的系数为 \(1+2\gamma w\)。

它把额外拟合能力分配给相对响应。64 行可在选点时组织成 32 对，或者在已观测位置间根据已知坐标与锚点上下文构造关系；每条边的真实目标都来自两端完整 FFN 标签。具体配对是表示与采样设计的一部分，边数不增加独立标签数。

几何选点可以相应作用于扩展特征，覆盖不同混合响应方向；关系配对可以同时覆盖共同上下文和不同上下文。两者都是可继续发展的正面构造，由其实际视频作用决定质量价值。

## 8. 成本与执行落点

沿用当前成本报告的主要 GEMM 口径，一次 MAC 表示一次乘加。单位是一对 Dense anchor 与近似 hit；一个视频有 80 对。以下 q=256、b=16、上下文维数 4 是具体构造实例，用于核算资源，不代表已经选出的质量最优值。

| 构造 | pair 主要 MAC，T | 两次 Dense 的比例 | 完整 observer 行数 | 40 层 anchor 缓存，GiB |
|---|---:|---:|---:|---:|
| 两次 Dense | 8.965702 | 100% | — | — |
| Q1 | 6.143166 | 68.5185% | — | 12.08038 |
| Global P64 | 6.147696 | 68.5690% | 64，W1-only | 44.69742 |
| 当前全 80 在线读出 | 6.162603 | 68.7353% | 64，完整 FFN | 12.23139 |
| 追加 16 个极化特征 | 6.175574 | 68.8800% | 64，完整 FFN | 12.30689 |
| 宽原生配对响应 256→32 | 6.241468 | 69.6149% | 64，完整 FFN | 12.23139 |
| 4 维锚点上下文→16 个交互 | 6.165846 | 68.7715% | 64，完整 FFN | 12.25026 |

极化分支新增：

\[
\Delta C=4Ndb+Nbd=5Ndb
=12,971,212,800\ {\rm MAC}.
\]

前四项来自两个端点各两组输入投影，最后一项是新增完整输出读出。它比当前读出增加约 0.2105%；新增 40 层 FP32 特征缓存为 0.07550 GiB。按 BF16 存储，80 位置的 P/Q/B 合计新增 37.5 MiB。

宽原生版本显式计入两端的 512→32 压缩：

\[
C_{\rm wide}
=2Ndm+2Nd^2+2Ndq+4Nqr+Nrd+2pdm.
\]

上表现有路径与极化/上下文路径沿用旧报告的主项口径，未计原有窄 CAL 变换；r32 满维时该共同变换为 0.064856 G MAC/pair。宽版本已计其 1.037697 G 压缩，以免隐藏新增收缩。GELU、逐元素、Gram 和求解另计。

上下文版本新增主要项为 \(Nd\times4+Nd\times16=3.242803\) G MAC；通用 T 收缩另为 0.064856 G，48 维替代 32 维的 Gram 增量为 0.040535 G。新增 40 层 c0 缓存 19.3286 MiB。

输出端将特征拼接后只进行一次完整输出 GEMM，并沿已有 base 写回融合方向实现。所有数字是算术和 payload 计数；对应部署延迟尚待实际实现计时。[原成本账本](../../reports/20261118/observed_secant_ffn_operator/analysis/READOUT_COST_AUDIT.json)

## 9. 本轮形成的研究进展

主推导把问题推进到了可执行的状态调制割线：

1. 由完整 FFN 的混合张量，明确了“状态差异与共同变化”的乘积结构。
2. 极化特征把该结构写成精确有限差分，给出原生权重初始化、48 维在线解和小幅新增成本。
3. 宽原生压缩把输入非线性覆盖宽度与缓存/读出宽度分离，允许 32 个坐标汇集 256 个原生响应。
4. 锚点上下文把已付费获得的完整响应转成条件坐标，进一步表达区域状态与当前变化的交互。
5. 真实成对 Dense 标签可以直接校准相对响应；MSE 与条件数继续服务于参数计算和数值描述，视频质量按真实视频判断。

这几项具有不同的结构预测：极化分支适合显式表示低维混合作用；宽原生分支保存较宽激活的真实有限变化；上下文分支利用 Dense anchor 中已有的状态区分。它们给出具体、可组合的算子方向，当前 0392 的交叠场景为“区域关系与运动连续性是否得到改善”提供明确的视觉目标。

讨论来源：/root/positive_secant_tensor_design 提出极化特征；/root/positive_research_synthesis 提出宽原生压缩；/root/positive_joint_structure 提出锚点上下文和真实关系拟合；/root 完成统一推导、原生曲率空间修正、成本口径核对与主张整理。本文记录构造贡献，视频收益身份为待实际视频判断。
