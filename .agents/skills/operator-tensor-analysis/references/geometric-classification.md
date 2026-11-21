# 几何分类：Weyl、Petrov、主方向与局部等价

本参考按原有数学对象组织已收集论文中的几何分类知识，可独立学习其作用空间、群作用、不变量及重构方法，并由此寻找 AI 中的新问题与构造。GC12 连接这些结构与机器学习，供形成迁移主张时核对条件；探索无需先匹配既有 AI 场景。模型参数、激活或高阶导数的数组形状本身不构成调用 Petrov 分类的依据。

这里分类的是声明了空间、度量、对称性和群作用的对象。先使用 [张量代数](tensor-algebra-and-symmetry.md) 确认类型，若需要从度量构造曲率，再使用 [微分几何](differential-geometry.md)。本文件保持三维自对偶算子上的迹归一化，不将不同论文的因子直接混合。来源及精确阅读位置见 [几何来源台账](sources-geometry.md)。

## 内容

- GC01 代数 Weyl 对象与作用空间
- GC02 Hodge 扇区与不同签名
- GC03 不变量、最小多项式与 Petrov 判别
- GC04 电磁分解与复对称算子
- GC05 Newman–Penrose、自旋量与主零方向四次式
- GC06 无坐标主零方向与 Bel–Sachs 判据
- GC07 Ferrando–Morales–Sáez 的协变重构
- GC08 从主方向到微分几何特征
- GC09 Boost weight 与高维 alignment
- GC10 Cartan 不变量与局部度量等价
- GC11 数值分类与退化
- GC12 向机器学习迁移的具体条件
- GC13 Ricci/二次型的 Segre 与因果分类

## GC01 代数 Weyl 对象与作用空间

**解决什么。** 判断 Petrov 是否真的适用，并把四槽张量压成有数学意义的算子。

经典 Petrov 是四维 Lorentz 空间上一点处的代数 Weyl 分类。给定非退化度量 \(g\)，要求

\[
C_{abcd}=-C_{bacd}=-C_{abdc}=C_{cdab},
\quad C_{a[bcd]}=0,
\quad g^{ac}C_{abcd}=0.
\]

在二形式空间上定义

\[
(\mathcal C\Omega)_{ab}=\tfrac12C_{ab}{}^{cd}\Omega_{cd},
\qquad \langle A,B\rangle=\tfrac12A_{ab}B^{ab}.
\]

成对交换对称使 \(\mathcal C\) 对这个双线性配对自伴。六维 \(\Lambda^2\) 上的矩阵来自对称性和度量，并非任意展开。

**可计算判据。** 分别核验反对称、成对交换、Bianchi 和迹残差，再构造 \(\binom42=6\) 维基上的矩阵。点态代数分类不要求 Einstein 方程、真空或全局流形解。若来自 \(g\) 的曲率，先按 DG05 去掉 Ricci 和标量部分。

**反例。** 多维数组的四个轴不等于四维切空间；\(W\otimes\operatorname{Sym}^3(V^*)\) 中的网络三阶导数有四槽，却没有 Weyl 的槽对称性。

**阅读。** Batista §II–III；Bär §6.1–6.3。

## GC02 Hodge 扇区与不同签名

**解决什么。** 确定可用的实/复算子、允许的标准形，以及“有多少类”的计数约定。

四维定向空间中 \(\mathcal C\) 与 Hodge 星可交换。Lorentz 签名下 \(*^2=-I\)，在复化空间取

\[
P_+=\tfrac12(I-i*),\qquad P_-=\tfrac12(I+i*),
\quad\Lambda^2_\mathbb C=\Lambda^2_+\oplus\Lambda^2_-.
\]

每个扇区复维数为 3。限制 \(\mathcal C_\pm=P_\pm\mathcal CP_\pm\)；在 Lorentz 自对偶扇区选择归一化复双线性正交基，使 \(\langle\mathcal U_i,\mathcal U_j\rangle=-\delta_{ij}\)，此时其矩阵满足 \(Q^\mathsf T=Q\)，一般不满足 \(Q^*=Q\)。无迹性在任意基中成立；任意基中的自伴条件是 \(Q^\mathsf T B=BQ\)，其中 \(B\) 是配对的 Gram 矩阵。只有正交而未统一归一化的基不能直接保证普通矩阵对称。

| 基底空间/签名 | 二形式上的星 | 两个扇区的关系 | 可用粗类型 |
| --- | --- | --- | --- |
| 实 Lorentz 四维 | \(*^2=-I\) | 两个复扇区互为共轭 | 由一个扇区的 I、II、D、III、N、O 决定 |
| 实正定四维 | \(*^2=I\) | 实扇区独立，诱导度量正定 | 每个扇区只有 I、D、O |
| 实 \((2,2)\) 四维 | \(*^2=I\) | 实扇区独立，诱导度量不定 | 每个扇区可有六种粗类型 |
| 复四维 | 按所选体积约定分裂 | 两个复扇区独立 | 每个扇区可有六种粗类型 |

**计数条件。** 固定定向时记录有序 \((X_+,X_-)\)：正定粗分有 \(3^2=9\) 种，复/neutral 粗分有 \(6^2=36\) 种。若允许交换定向，从而识别 \((X,Y)\sim(Y,X)\)，分别为 6 和 21；Batista 使用后一计数。neutral 情形还可按实主自旋量/实谱结构细分，六个字母不是完整的实轨道分类。

**例子与边界。** 正定自伴扇区不能出现非零幂零算子；neutral 的不定自伴扇区可以非对角化。两者虽然都有 \(*^2=I\)，标准形结论并不相同。

**阅读。** Batista §II–VII；Bär §6.2–6.4。

## GC03 不变量、最小多项式与 Petrov 判别

**解决什么。** 给定三维自对偶 Weyl 算子，得到可计算的精确代数判别。

固定

\[
I=\tfrac12\operatorname{tr}Q^2,
\quad J=\tfrac16\operatorname{tr}Q^3,
\quad \chi_Q(\lambda)=\lambda^3-I\lambda-2J,
\quad\Delta=4(I^3-27J^2).
\]

| 类型 | 精确判据 | 最小多项式/补充信息 |
| --- | --- | --- |
| I | \(\Delta\ne0\) | 三个互异本征值 |
| D | \(\Delta=0,(I,J)\ne(0,0)\)，且 \((Q-\rho I_3)(Q+2\rho I_3)=0\) | \((\lambda-\rho)(\lambda+2\rho)\)，\(\rho=-3J/I\ne0\) |
| II | 同一非零重复谱，但上面的二次多项式不为零 | \((\lambda-\rho)^2(\lambda+2\rho)\) |
| III | \(Q^3=0,Q^2\ne0\) | \(\lambda^3\)，秩 2 |
| N | \(Q^2=0,Q\ne0\) | \(\lambda^2\)，秩 1 |
| O | \(Q=0\) | \(\lambda\)，秩 0 |

若 \(I=J=0\)，Cayley–Hamilton 给出 \(Q^3=0\)，但不能据此区分 III、N、O。仅看重复本征值也不能区分 D、II；最小多项式或核空间维数提供缺少的信息。

需要显式本征值时，可用 Cardano：选择 \(u^3=J+\sqrt{J^2-I^3/27}\)、\(v^3=J-\sqrt{J^2-I^3/27}\)，并使 \(uv=I/3\)，则 \(\lambda_k=\omega^ku+\omega^{-k}v\)，\(\omega=e^{2\pi i/3}\)。两立方根不能独立随意取支；遇到零分母或重根直接回到多项式和核空间判据。

**验算例。** 以下都是复对称无迹矩阵：

\[
Q_D=\operatorname{diag}(1,1,-2),\quad
Q_{II}=\begin{pmatrix}2&i&0\\i&0&0\\0&0&-2\end{pmatrix},
\]

\[
Q_N=\begin{pmatrix}1&i&0\\i&-1&0\\0&0&0\end{pmatrix},\quad
Q_{III}=\begin{pmatrix}0&0&1\\0&0&i\\1&i&0\end{pmatrix}.
\]

\(Q_D,Q_{II}\) 都有 \(I=3,J=-1\)，但二次最小方程只对前者成立。\(Q_N^2=0\)，\(Q_{III}^2=Q_N\ne0\)，\(Q_{III}^3=0\)。这组例子适合验证实现是否错误使用 Hermitian 特征分解。

**阅读。** Ferrando–Morales–Sáez §3、式 (10)–(11) 及判别图；Bär §6.4。该论文用 \(a=2I,b=6J\)。

## GC04 电磁分解与复对称算子

**解决什么。** 将 Weyl 的十个实分量与一个指定观察者测得的两个空间对称无迹对象联系起来。

取签名 \((-+++)\)、单位类时 \(u\)，定义

\[
E_{ab}=C_{acbd}u^cu^d,
\qquad B_{ab}=(*C)_{acbd}u^cu^d.
\]

\(*\) 作用在 Weyl 的第一对槽。\(E,B\) 都与 \(u\) 正交、对称且无迹。令正交空间基为 \(e_i\)，\(F_i=u^\flat\wedge e_i^\flat\)，\(\mathcal U_i=(F_i-i*F_i)/\sqrt2\)。有 \(\langle\mathcal U_i,\mathcal U_j\rangle=-\delta_{ij}\)，因此与 GC01 算子约定一致的矩阵为

\[
Q_{ij}=-\langle\mathcal U_i,\mathcal C\mathcal U_j\rangle
=-E_{ij}+iB_{ij}.
\]

若换度量签名、星的定向或曲率整体符号，此关系也要相应转换；直接从基收缩构造 \(Q\) 可以消除因子歧义。

**判据与例子。** 存在某个类时 \(u\) 使 \(B=0\) 时，\(Q\) 为实对称矩阵，故只能属于 I、D、O。给定观察者的 \(E,B\) 不是观察者无关的两个张量场选择；“某帧一个分量为零”也不能推出纯电性。

**阅读。** Ferrando–Morales–Sáez §2、§5.1–5.2；自对偶转换和正交基见 Bär §6.2–6.3。

## GC05 Newman–Penrose、自旋量与主零方向四次式

**解决什么。** 将 Petrov 问题变为一个射影四次多项式的根重数，并利用适应主方向的标架简化微分方程。

在 Lorentz 四维可取复零标架 \((\ell,n,m,\bar m)\)，其中 \(g(\ell,n)=-1\)、\(g(m,\bar m)=1\)，其余配对为零。五个复 Weyl 标量是 \(C\) 在该标架上的指定收缩。它们依赖标架，不能单个视为几何不变量。

为固定四次式而避免不同 NP 符号混用，直接定义一个归一化自旋 dyad \((o,\iota)\)，令完全对称 Weyl 自旋量的分量为

\[
\Psi_r=\Psi_{ABCD}\,o^{A}\cdots o^{\cdot}\iota^{\cdot}\cdots\iota^D,
\qquad r=0,\ldots,4,
\]

其中恰有 \(r\) 个 \(\iota\)。于是展开本身保证

\[
P(s,t)=\Psi_0s^4+4\Psi_1s^3t+6\Psi_2s^2t^2
+4\Psi_3st^3+\Psi_4t^4.
\]

它的根在 \(\mathbb{CP}^1\) 上；不能丢掉 \(s=0\) 的无穷远根。根自旋方向 \(\xi\) 给出实零方向 \(k^{AA'}=\xi^A\bar\xi^{A'}\)。自旋结构只在做全局自旋场时增加拓扑要求，点态分类可用零标架或二形式独立完成。

| 类型 | 四个根的重数 |
| --- | --- |
| I | \(1+1+1+1\) |
| II | \(2+1+1\) |
| D | \(2+2\) |
| III | \(3+1\) |
| N | \(4\) |
| O | 多项式恒零，所有零方向均满足条件 |

**可计算判据。** 用齐次多项式或两张射影图求根，用 \(P\) 与导数的最大公因子/结式判断重数。选定 \(o\) 为某主方向后，该方向重数至少 1、2、3、4 分别对应 \(\Psi_0=0\)、\(\Psi_0=\Psi_1=0\)、再加 \(\Psi_2=0\)、再加 \(\Psi_3=0\)。D 型可让只有 \(\Psi_2\) 非零。

**微分接口。** NP 的 \(D=\nabla_\ell,\Delta=\nabla_n,\delta=\nabla_m,\bar\delta=\nabla_{\bar m}\) 与 spin coefficients 是活动零标架中的方向导数和联络分量。对 Weyl 标量求导时保留标架变化项，不能把方向偏导直接当协变导数。

**阅读。** Batista 式 (13)–(20)、§IV 的分量消去表；其签名为 \((+---)\)，作用在二形式的算子另有因子约定。四次式与二形式的等价关系见 Ferrando §3、Bär §6.5；上述二项式系数由对称四自旋量直接展开。

## GC06 无坐标主零方向与 Bel–Sachs 判据

**解决什么。** 在已有几何方向时验证其主性与重数，或者避开某一套 NP 坐标约定。

自对偶二形式 \(H\) 表示主零二形式，当

\[
H\ne0,\quad\langle H,H\rangle=0,
\quad\langle H,\mathcal CH\rangle=0.
\]

与其对应的实零方向由实部和 Hodge 双对偶的公共零方向确定。对非零 Weyl，零范数（null）的本征二形式对应重复主方向；这里“零范数”不要求其本征值为零。一般主二形式未必是本征二形式。

**Bel–Sachs 的实际判据。** 固定实零向量 \(\ell\ne0\)：

| 主方向重数至少 | 需要为零的收缩 |
| --- | --- |
| 1 | \(C(\ell,x,\ell,y)\)，所有 \(x,y\perp\ell\) |
| 2 | \(C(\ell,x,\ell,y)\)，所有 \(x\perp\ell\)，任意 \(y\) |
| 3 | \(C(\ell,x,\cdot,\cdot)\)，所有 \(x\perp\ell\) |
| 4 | \(C(\ell,\cdot,\cdot,\cdot)\) |

**计算。** 将 \(\ell\) 补成一个零标架，只需在相应基向量上逐项检查这些多线性式，不必对无限个向量试验。两个不同二重方向给 D，一个三重加一个单重给 III。

**反例。** 对 type I，特征二形式的主平面一般不等于四条 Debever 主零方向。只把 \(Q\) 对角化然后把三条本征轴叫作四条主零方向会混淆两套几何。

**阅读。** Bär Proposition 6.40、Proposition 6.44、Corollary 6.45；Ferrando §3、式 (12)–(16)。

## GC07 Ferrando–Morales–Sáez 的协变重构

**解决什么。** 分类后实际构造特征二形式、主零方向与 canonical frame，而不止给出一个字母标签。

以下 \(W\) 表示自对偶空间上的算子；\(\langle\ ,\ \rangle\) 是复双线性配对，\(\mathbf1\) 是恒等算子。把 \(u\otimes v\) 理解为 \(z\mapsto u\langle v,z\rangle\)，而非无度量的数组外积。单位二形式满足 \(\langle U,U\rangle=-1\)。对 \(\langle z,z\rangle\ne0\) 定义归一化 \(N(z)=z/\sqrt{-\langle z,z\rangle}\)。平方根支的改变给允许的符号自由度。

### A. 从算子多项式提取二形式

**N 型。** 选择 \(X\) 使 \(WX\ne0\)：

\[
H=\frac{WX}{\sqrt{\langle X,WX\rangle}},
\qquad W=H\otimes H,
\qquad\langle H,H\rangle=0.
\]

**III 型。** 选择 \(X\) 使 \(W^2X\ne0\)：

\[
H=-\frac{W^2X}{\sqrt{-\langle X,W^2X\rangle}},
\quad
U=\frac{2\langle H,X\rangle WX-\langle X,WX\rangle H}
{2\langle H,X\rangle^2}.
\]

所得 \(\langle U,U\rangle=-1\)、\(\langle U,H\rangle=0\)，并有 \(W=U\otimes H+H\otimes U\)。

**D 型。** 令 \(\rho=-\operatorname{tr}W^3/\operatorname{tr}W^2\ne0\)，选择 \(P X\ne0\)，

\[
P=W-\rho\mathbf1,\quad U=N(PX),
\quad W=3\rho U\otimes U+\rho\mathbf1.
\]

**II 型。** 定义

\[
P=(W-\rho\mathbf1)^2,
\quad K=\frac{(W-\rho\mathbf1)(W+2\rho\mathbf1)}{3\rho}.
\]

选 \(PX\ne0,KY\ne0\)，取

\[
U=N(PX),\quad H=\frac{KY}{\sqrt{\langle Y,KY\rangle}},
\quad W=3\rho U\otimes U+\rho\mathbf1+H\otimes H.
\]

**I 型。** 对互异本征值 \(\rho_i\)，

\[
P_i=\prod_{j\ne i}(W-\rho_j\mathbf1)
=W^2+\rho_iW+(\rho_i^2-\tfrac12\operatorname{tr}W^2)\mathbf1,
\quad U_i=N(P_iX_i).
\]

选择 \(P_iX_i\ne0\)，则 \(W=-\sum_i\rho_iU_i\otimes U_i\)。\(P_i\) 是提取特征线的多项式；真正的幂等谱投影是 \(P_i/\prod_{j\ne i}(\rho_i-\rho_j)\)。不要把未经归一的 \(P_i\) 本身当成 \(P_i^2=P_i\)。

**验算。** 检查归一化、零范数/正交关系、本征方程和原算子的重构。更换合法试验向量，输出的特征线或平面应保持，仅剩规定符号和基自由度。接近退化时分母可能病态，转用 GC11 的残差和子空间方法。

### B. 二形式给出的主方向

为区分 A 中的复对象与作用在实切空间上的对象，以下将 A 的自对偶形式记作 \(\mathcal U,\mathcal H,\mathcal V\)。对每个 \(\mathcal F\)，另定义实形式 \(F_R=\sqrt2\operatorname{Re}\mathcal F\)，并以 \((F_R)^a{}_b=g^{ac}(F_R)_{cb}\) 作为实端同态。取任意未来类时 \(x\)；下面的平方作用都使用带 \(R\) 下标的实端同态。

对于单位简单实形式 \(U_R\)，两个主零方向为

\[
\ell_\pm\propto(U_R^2\pm U_R)x.
\]

对于上述非零 null 自对偶形式 \(\mathcal H\) 对应的实二形式 \(H_R\)（满足 \(\langle H_R,H_R\rangle=\langle H_R,*H_R\rangle=0\)），唯一 fundamental 方向及一个规范化代表是

\[
\ell\propto H_R^2x,
\qquad \ell=\frac{H_R^2x}{\sqrt{-g(x,H_R^2x)}}.
\]

N 型的 \(H_R\) 给四重方向；III 型的 \(H_R\) 给三重方向，\(U_R\) 给三重与单重方向；D 型的 \(U_R\) 给两条二重方向。II 型先在复自对偶空间中构造

\[
\mathcal V_\pm=\mathcal U\pm\frac{i}{\sqrt{3\rho}}\mathcal H;
\]

再取 \((V_\pm)_R=\sqrt2\operatorname{Re}\mathcal V_\pm\)，各实形式的主方向包含同一二重方向与一个单重方向。I 型也先在复自对偶空间中取

\[
\mathcal V_\pm=c\mathcal U_1\pm s\mathcal U_2,
\quad c^2=\frac{\rho_3-\rho_1}{\rho_2-\rho_1},
\quad s^2=1-c^2,
\]

再取对应的实形式，其两两主方向给四条单重 Debever 方向。若系数为复数，必须先完成自对偶线性组合再取实部，不能把同一复系数直接用于先前取出的实形式。所有输出再用 GC06 的收缩判据验证。

### C. 从二形式恢复标架

沿用 B 的实形式约定。若 \((U_R,H_R)\) 已按定向选取相容符号，并满足所需 null-frame 归一化，则

\[
n=-\frac{(U_R^2+U_R)x}{2g(x,\ell)},
\quad e_2=H_R(n),\quad e_3=-(*H_R)(n).
\]

核验 \(g(\ell,n)=-1\)、\(e_2,e_3\) 空间正交单位和定向。I 型取 \(U_{i,R}=\sqrt2\operatorname{Re}\mathcal U_i\)，三个相容定向单位特征二形式还给

\[
P_0=\tfrac12\left(\mathbf1-\sum_iU_{i,R}^2\right),
\quad e_0=-\frac{P_0x}{\sqrt{g(x,P_0x)}},
\quad e_i=U_{i,R}(e_0).
\]

标架余自由度也是结果：I 型剩有限的 24 个定向/时间定向选择，II 型有两个规范选择，III 型在相容约定下可唯一固定，D 与 N 型仍有两个实连续参数。O 型的 Weyl 不提供这种标架固定。

**阅读。** Ferrando–Morales–Sáez §2 的 Lemma 1–2；§4.1–4.5 的 Propositions 1–15，尤其式 (19)、(24)、(29)、(37)、(42)–(45)。以上以统一的算子、配对和归一化重写，实际使用仍需检验定向与符号约定。

## GC08 从主方向到微分几何特征

**解决什么。** 把点态 Weyl 代数提取的方向用于检验整个度量的微分性质。

**I 型静态性。** 在类型保持 I 的邻域中，从 GC07 选未来单位主观察者 \(u\)，令 \(a=\nabla_u u\)。局部静态的检验为

\[
\nabla_i u_j=-u_i a_j,\qquad da^\flat=0.
\]

若满足，局部求 \(d\log f=a^\flat\)，\(K=fu\) 为超曲面正交类时 Killing 场。这里先从 Weyl 固定候选方向，再核验微分条件，避免搜索所有未知 Killing 分量。

**III 型超曲面正交 Killing 方向。** 候选零方向是 GC07 的三重主方向 \(\ell\)，候选类空方向由 canonical frame 的 \(e_3\) 给出。类空单位 \(s\) 检查

\[
\nabla_i s_j=s_i(\nabla_s s)_j,
\qquad d(\nabla_s s)^\flat=0.
\]

对零方向，检查是否存在一形式 \(a,b\) 使

\[
\nabla_i\ell_j=a_i\ell_j+\ell_i b_j,
\qquad d(a+b)=0.
\]

这在每一点先是可解的线性分解，随后是微分闭合条件；局部取 \(d\log f=-(a+b)\) 即得 \(K=f\ell\)。等价的无辅助一形式公式见原文 Lemma 4。

**例子与边界。** 点态为 I 或 III 不等于存在这些 Killing 场，决定性信息来自 \(\nabla C\) 及方向导数。跨越 Petrov 类型变化处，需要重新选择光滑分支；闭一形式的全局积分还受拓扑影响。

**阅读。** Ferrando–Morales–Sáez §5、式 (46)–(51)、Propositions 16–18；几何运算见 DG08。

## GC09 Boost weight 与高维 alignment

**解决什么。** 将张量分量按零标架的伸缩权重分层，并在高维 Lorentz 空间寻找 aligned null directions。

对 null frame 做 \(\ell\mapsto\lambda\ell,n\mapsto\lambda^{-1}n,m_i\mapsto m_i\)。若一个标架分量变为 \(\lambda^bT\)，其 boost weight 是 \(b\)。协变分量收缩中，每个 \(\ell\) 贡献 \(+1\)，每个 \(n\) 贡献 \(-1\)。Weyl 的权重范围为 \(-2,\ldots,2\)。

**计算流程。** 以某零方向为 \(\ell\)，求最高权重 \(+2\) 分量 \(C_{0i0j}\)。若可全部消掉，该方向是 WAND；进一步消掉 \(+1\)、0、\(-1\) 层给更高 alignment。高维主类型对应：I 最高剩 \(+1\)，II 最高剩 0，III 最高剩 \(-1\)，N 只剩 \(-2\)，O 全零；不存在 WAND 时为 G。D 还要求存在两个相对方向使只剩权重 0。

四维 NP 的 \(\Psi_r\) 具有权重 \(2-r\)，以上退化为经典 Petrov。高维的 WAND 方程通常是过定多项式系统，可用零旋转参数化、消元或 Gröbner 基寻找全部解，并报告实解、复解和解簇维数。

**边界与反例。** 四维中复自对偶三维谱分类与 alignment 等价，不代表高维 Weyl 的任意 bivector 矩阵 Jordan 类已经等价于 CMPP 分类。Euclid 情形没有实零向量；若通过复化做 boost 分类，最后必须施加正确实条件。neutral 可有多个独立 boost，权重应是向量。

**阅读。** Batista §VII；Coley–Milson–Pravda–Pravdová 2004，式 (1)–(9) 及 Principal/Secondary Classification。

## GC10 Cartan 不变量与局部度量等价

**解决什么。** 判断两个度量是否只是坐标写法不同，超过点态 Petrov 字母与少量标量不变量的辨别力。

从定向正交/零标架中的

\[
\mathcal I_q=\{R,\nabla R,\ldots,\nabla^qR\}
\]

开始，每阶用仍允许的标架变换规范化分量，记录剩余 isotropy 群 \(H_q\) 及独立函数数目 \(t_q\)。继续求导，直至在正则邻域中这些数据稳定，并比较规范分量之间的函数关系。这里比较的包括残余群作用与微分关系，不是只比较两个坐标数组。

**可计算判据。** 用规范分量的微分秩确定 \(t_q\)，用使所有分量保持的群/李代数方程确定 \(H_q\)。先去除残余标架选择引入的伪坐标依赖，不把任意帧中分量的函数数目直接当作 \(t_q\)。在适用的正则局部情形，等距群维数由 \(n-t_q+\dim H_q\) 给出。停止时检查剩余群本身及函数秩的稳定，离散 isotropy 也可能改变。实际停止阶及等价性须满足所用版本的正则条件，不给所有张量或所有维数套一个固定阶数。

**例子与反例。** Lorentz 度量可能所有标量多项式曲率不变量都为零但 Riemann 非零；因此“标量不变量与 Minkowski 全相同”不保证平坦。Petrov N 与 O 也已经展示了迹不变量为零时仍可有非零算子。

**阅读。** Karlhede 1980 原论文的算法来源；可公开逐步核查的作者论文 Mergulhão–Batista 2020 §II–III。GC07 的 Weyl canonical frame 是这一流程的一种标架固定入口，完整等价判断还需 Ricci 和协变导数。

## GC11 数值分类与退化

**解决什么。** 防止把浮点噪声、近重根和不稳定 Jordan 结构当作精确几何结论。

**操作。** 在声明的标架及数值范数下，先记录原 Weyl 对称残差；如做代数投影，同时记录投影大小。先取 \(q=\|Q\|\)：精确 \(q=0\) 直接判为 O，不进行归一化；含噪输入若 \(q\) 已在绝对误差量级内，先报告与零算子相容，不能由归一化后的噪声判定精确类型。仅在 \(q>0\) 且尺度可分辨时，用 \(\widehat Q=Q/q\) 计算尺度一致的 \(\widehat I,\widehat J\) 和

\[
r_\Delta=\frac{|\widehat I^3-27\widehat J^2|}
{|\widehat I|^3+27|\widehat J|^2+\varepsilon},
\quad
r_D=\|(\widehat Q-\widehat\rho I)(\widehat Q+2\widehat\rho I)\|.
\]

另检查 \(\widehat Q^2,\widehat Q^3\) 及相关多项式的奇异值，以输入误差和舍入规模决定秩阈值。\(I,J\) 很小时不使用 \(\rho=-3J/I\) 的不稳定除法。

**输出判据。** 精确符号/有理输入可以给出精确类；噪声输入应报告“与某标准形相容的残差、阈值及其它仍可能类型”。一般微扰会把退化类型变成 I，单个近零判别式既不能稳健分开 D/II，也不能定量保证接近某个特定 Jordan 轨道。

**说明。** 这里的矩阵范数是指定计算表示下的误差尺度；任意 Lorentz boost 不保持普通 Frobenius 范数。几何不变量与选定标架下的数值条件数应分别报告。验证时使用 GC03 的 D/II 与 III/N/O 例子，及其小扰动。

**阅读。** Bär §6.4 的退化关系；一般谱扰动与数值秩另由本技能的数值线性代数工具处理。

## GC12 向机器学习迁移的具体条件

**可复用的一般结构。** 识别槽对称、选择自然作用空间、声明群作用、构造不变量、使用最小多项式与谱投影、利用稳定子和 residual symmetry。这些结构提供分析高阶导数、协方差、弹性响应、等变层和张量网络的线索；新的对象与组合继续按其数学条件研究。

**需要建模后迁移。** 若定义了统计流形或特征嵌入及其度量、联络，可以计算其曲率；若目标对象恰为四维 Lorentz 代数 Weyl，则再作 Petrov 分类。正定 Fisher 度量通常进入 Riemannian 几何，不能原样称为 Lorentz Petrov 分类。

**由差异提出问题。** 缺少原分类所需结构时，可以研究适合目标对象的作用空间、允许变换、投影或不变量，也可以有意识地构造新的几何模型。分别记录哪些性质由定义保证、哪些需要证明、哪些尚属启发。原定理的前提约束其结论的使用；前提不满足也可以成为修改构造或发展新判据的起点。

**一个明确的排除计算。** 光滑 FFN 的 Hessian 属于 \(W\otimes\operatorname{Sym}^2(V^*)\)。任意反对称 \(\Omega^{ij}\) 都有

\[
H^a{}_{ij}\Omega^{ij}
=\tfrac12(H^a{}_{ij}-H^a{}_{ji})\Omega^{ij}=0.
\]

因此把 Hessian 送进 bivector 反对称通道只会丢掉它的真实响应。四槽三阶导数也不会自动成为 Weyl；投影后为零不意味着原网络是 Petrov O。

**已有连接与进一步组合。** DG11–DG13 提供梯度／距离，DG05、DG09 连接曲率响应，张量分解与对称投影处理高阶 multilinear 结构。这些连接可继续组合，也可由新的结构对应扩展。Petrov–Galerkin 是另一类数值离散方法，名称中的 Petrov 不提供与 Weyl/Petrov 分类的数学联系。

## GC13 Ricci/二次型的 Segre 与因果分类

**解决什么。** 处理 Riemann 曲率的 Ricci 部分或其它对称二阶张量，避免把它们误送进 Weyl 的六类表。

给定对称 \(S_{ab}\) 和非退化 \(g\)，先形成

\[
A^a{}_b=g^{ac}S_{cb},\qquad g(Au,v)=g(u,Av).
\]

对 \(A\) 求本征值、各广义本征空间的核链 \(\ker(A-\lambda I)^k\) 和最小多项式，再记录对应子空间的度量限制及实/复共轭结构。Segre 记号编码 Jordan 块大小与本征值相等关系；在不定度量中，因果性质和度量信息使分类比普通无度量相似更细。

**可计算反例。** 取 \(g=\operatorname{diag}(-1,1)\)、\(A=\begin{psmallmatrix}1&1\\-1&-1\end{psmallmatrix}\)。有 \(gA=(gA)^\mathsf T\)、\(A^2=0\)、\(A\ne0\)。所以不定自伴不保证对角化。取 \(A=\begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix}\) 仍满足相同自伴条件，却有共轭虚本征值。正定 \(g\) 下这些情况被谱定理排除。

**例子。** 对实际度量的 Ricci 张量做上述操作，得到与 Weyl/Petrov 互补的代数信息；用于局部度量等价时两部分都需要。对普通网络 Jacobian，若无 \(gA=A^\mathsf T g\) 的已证结构，不能自动借用自伴或因果分类。

**阅读。** Batista §I 对 Ricci/Segre 与 Weyl 分类的区分；方法是带度量的端同态及核链的直接计算，Jordan 工具见 Bär §6.4 与本技能线性算子参考。这里不声称枚举所有维数、所有签名的 Segre 标准形。
