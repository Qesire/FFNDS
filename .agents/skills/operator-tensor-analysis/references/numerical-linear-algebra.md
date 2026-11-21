# 数值线性代数与算子计算

识别到矩阵结构后使用本参考。先保留张量的索引分组和所选范数，再选分解、迭代或随机方法；矩阵只需可作用时，不要求先物化。复 Euclidean 内积中的伴随使用共轭转置 $A^*$，但代数对偶拉回、按列 vec 等纯索引恒等式仍可能使用普通转置 $A^\top$；复变量实值微分还需区分实线性与复线性，不能机械替换所有转置。

## 目录

- N1 分解选择
- N2 截断 SVD 与数值秩
- N3 最小二乘、伪逆与正则化
- N4 Kronecker、Sylvester 与 Lyapunov
- N5 Schur 补与低秩更新
- N6 矩阵函数及其 Fréchet 导数
- N7 条件数、后向误差与舍入
- N8 谱与子空间扰动
- N9 Krylov、CG、GMRES 与预条件
- N10 随机范围寻找与随机 SVD
- N11 随机迹、Frobenius 范数与 Hutch++
- N12 子空间嵌入、ID、CUR 与 Nyström
- N13 算子接口与计算成本
- 来源

## N1 分解选择

| 对象或目的 | 首选结构 | 必须核对的条件与代价 |
| --- | --- | --- |
| 一般方阵求解 | 带选主元 LU | 检查奇异性、增长因子与残差；稠密约 $O(n^3)$，多右端复用分解 |
| Hermitian 正定 | Cholesky $A=LL^*$ | 正定性、对称存储与缩放；不可因“应该正定”忽略失败 |
| Hermitian 不定 | 带主元 $LDL^*$ | $D$ 可含 $2\times2$ 块；保留惯性，不强行 Cholesky |
| 最小二乘／正交基 | Householder QR | 满列秩的过定问题可三角回代，避免形成 Gram；欠定最小范数问题可用 $A^*$ 的 QR 或 SVD |
| 数值秩／最小范数解 | SVD | 阈值依赖噪声及目标，奇异值不应按机器零机械截断 |
| 一般谱与矩阵函数 | Schur $A=QTQ^*$ | 复 Schur 为上三角；实 Schur 含 $1\times1$／$2\times2$ 对角块；Jordan 形式只作精确代数分类 |
| 巨大稀疏或隐式算子 | Krylov／随机草图 | 以 matvec、伴随作用、通信与工作内存计成本 |

列主元 QR $A\Pi=QR$ 是实用的秩揭示启发；普通主元 QR 的最坏情况不等于 strong rank-revealing QR 的界。对满列秩 $A$，$\kappa_2(A^*A)=\kappa_2(A)^2$；秩亏或列数多于行数时 $A^*A$ 奇异。形成 Gram 矩阵还可能丢失小奇异方向。带状、稀疏和层次低秩结构需记录填充、排序与重压缩，不能只使用稠密 FLOP 模型。

## N2 截断 SVD 与数值秩

若 $A=U\Sigma V^*$，则 $A_r=\sum_{i=1}^r\sigma_i u_iv_i^*$ 是最佳 rank-$\le r$ 的谱范数和 Frobenius 近似：

\[
\min_{\operatorname{rank}B\le r}\|A-B\|_2=\sigma_{r+1},
\qquad
\min_{\operatorname{rank}B\le r}\|A-B\|_F^2=\sum_{i>r}\sigma_i^2.
\]

Frobenius 解在切分处有严格奇异值间隙时唯一；谱范数最佳解一般不唯一。这个最优性不自动推广到 CP、HOSVD 或某种特定 TT 秩。

**操作。** 给定误差容限，先选范数，再按尾谱选秩。若目标是 $\mathbb E\|(A-B)x\|_G^2$，且 $A$ 与 $G\succeq0$ 固定、$C=\mathbb E[xx^*]$ 有限，则对 $G^{1/2}AC^{1/2}$ 做 SVD，并按近似参考 A2 的逆／伪逆规则映回原空间；若目标是特定方向 $x$，直接检查 $(A-B)x$。不要把参数矩阵的普通尾谱当作任意任务的误差保证。

**代价。** 满 SVD 通常 $O(mn\min(m,n))$；只需前 $r$ 项时考虑部分 SVD／随机 SVD。若输入只能通过 JVP／VJP 使用，先提供双侧作用接口。

## N3 最小二乘、伪逆与正则化

对 $\min_x\|Ax-b\|_2$，SVD 给出最小范数解

\[
x_*=A^+b=\sum_{\sigma_i>0}\frac{u_i^*b}{\sigma_i}v_i.
\]

小奇异值把噪声放大；最小二乘残差小不保证解误差小。截断 SVD 舍弃 $\sigma_i\le\tau$ 的方向；Tikhonov

\[
\min_x\|Ax-b\|_2^2+\lambda\|x\|_2^2,
\qquad
x_\lambda=\sum_i\frac{\sigma_i}{\sigma_i^2+\lambda}(u_i^*b)v_i
\]

采用连续谱滤波，$\lambda>0$。一般罚项 $\|Lx\|^2$ 对应堆叠最小二乘 $[A;\sqrt\lambda L]x\simeq[b;0]$，唯一性要求 $\ker A\cap\ker L=\{0\}$。可用 QR、LSQR 或预条件迭代求解，避免显式求逆。

右端、测量与参数度量分别决定加权最小二乘、统计噪声模型与先验；这三者不要混用。同一数据上选择截断阈值和报告拟合误差可能乐观，应以目标误差和独立方向核验。

## N4 Kronecker、Sylvester 与 Lyapunov

本节明确采用**按列 vec**：

\[
\operatorname{vec}(AXB)=(B^\top\otimes A)\operatorname{vec}X.
\]

转置 $B^\top$ 来自索引重排，即使复数也不是 $B^*$。其他展平顺序须重新推导排列。对 $AX+XB=C$：

\[
(I\otimes A+B^\top\otimes I)\operatorname{vec}X=\operatorname{vec}C.
\]

当 $\sigma(A)\cap\sigma(-B)=\varnothing$ 时有唯一解。Kronecker 式揭示线性结构；实现通常用 Schur／Bartels–Stewart 或低秩迭代，避免物化 $n^2\times n^2$ 矩阵。[NL3]

连续 Lyapunov 方程 $A^*P+PA=-Q$：若 $A$ Hurwitz 且 $Q\succ0$，则

\[
P=\int_0^\infty e^{tA^*}Qe^{tA}\,dt\succ0.
\]

$V(x)=x^*Px$ 给出稳定性证书。若 $Q\succeq0$，正定性还需可观测性等条件。离散方程 $P-A^*PA=Q$ 对应 $\rho(A)<1$ 和级数解；不要混用连续与离散稳定条件。

## N5 Schur 补与低秩更新

对块系统 $\begin{pmatrix}A&B\\C&D\end{pmatrix}\binom{x}{y}=\binom{f}{g}$，若 $A$ 可逆，消去 $x$ 得

\[
(D-CA^{-1}B)y=g-CA^{-1}f,
\qquad x=A^{-1}(f-By).
\]

Schur 补适用于约束优化、条件高斯、块预条件与消元。对 Hermitian 块矩阵，$A\succ0$ 时整体正定当且仅当 $D-B^*A^{-1}B\succ0$。实际通过解 $A$ 的系统实现。

Woodbury 的无需 $C^{-1}$ 版本为

\[
(A+UV^*)^{-1}
=A^{-1}-A^{-1}U(I+V^*A^{-1}U)^{-1}V^*A^{-1}.
\]

要求 $A$ 与中间小矩阵可逆。收益取决于 $A$ 的分解／作用可复用，以及低秩更新维度确实小。接近奇异的中间矩阵会放大误差；小更新秩不等于小扰动范数。

## N6 矩阵函数及其 Fréchet 导数

导数 $L_f(A,E)$ 是关于方向矩阵 $E$ 的线性映射，满足 $f(A+E)=f(A)+L_f(A,E)+o(\|E\|)$。对谱邻域解析的 $f$：

\[
f\!\begin{pmatrix}A&E\\0&A\end{pmatrix}
=\begin{pmatrix}f(A)&L_f(A,E)\\0&f(A)\end{pmatrix}.
\]

这是计算与验证入口，未必是最省的实现；优先使用专门的矩阵函数方向导数算法。[NL4]

| 函数 | 正确方向作用 | 条件／风险 |
| --- | --- | --- |
| $A^{-1}$ | $-A^{-1}EA^{-1}$ | $A$ 可逆；两侧因子不能随意交换 |
| $e^A$ | $\int_0^1e^{(1-s)A}Ee^{sA}\,ds$ | $AE=EA$ 是可简化为 $e^AE$ 的充分条件；一般不可省略两侧指数 |
| $A^2$ | $AE+EA$ | 非交换乘积规则 |
| 主平方根 $S=A^{1/2}$ | 解 $SX+XS=E$ | 若 $\sigma(A)\cap(-\infty,0]=\varnothing$，主根解析且 $\operatorname{Re}\sigma(S)>0$，解唯一；含零特征值的半正定平方根不能据此声称 Fréchet 可微 |
| $\log\det A$ | $\operatorname{tr}(A^{-1}E)$ | 实 SPD 时无分支歧义；一般复情形须选局部分支 |

对 Hessian 或高阶方向导数，继续求导这些**作用**，不要默认构造 $n^2\times n^2$ Jacobian。直接求 $f(A)v$ 可使用 Krylov 或多项式／有理逼近；先检查谱域、非正规性与目标精度。

## N7 条件数、后向误差与舍入

相对条件数把输入扰动映到输出扰动。以下界针对非奇异方阵 $A$、$b\ne0$ 及相容的向量／诱导矩阵范数；二范数下 $\kappa_2(A)=\sigma_{\max}/\sigma_{\min}$。对计算解 $\widehat x$、残差 $r=b-A\widehat x$，可用

\[
\frac{\|x-\widehat x\|}{\|x\|}\le\kappa(A)\frac{\|r\|}{\|b\|},
\qquad
\eta=\frac{\|r\|}{\|A\|\|\widehat x\|+\|b\|}
\]

分别表达前向误差上界和范数后向误差；$\eta$ 对应同时允许无结构扰动 $\|\Delta A\|\le\eta\|A\|$、$\|\Delta b\|\le\eta\|b\|$ 的问题。若只扰动一项、要求保持结构或讨论最小二乘，应使用对应条件数和后向误差。后向稳定算法得到邻近输入问题的精确解；若原问题病态，前向误差仍可能大。

标准舍入模型 $\operatorname{fl}(a\circ b)=(a\circ b)(1+\delta)$、$|\delta|\le u$ 需排除溢出及不适用的下溢情况。顺序点积的基本绝对误差界为

\[
|\operatorname{fl}(x^\top y)-x^\top y|
\le\gamma_n\sum_i|x_iy_i|,
\qquad \gamma_n=\frac{nu}{1-nu},\quad nu<1.
\]

实际界随乘法、累加格式、FMA、树形归约与量化而变。强抵消时相对误差可大。迭代精化用较高精度残差 $r=b-A\widehat x$ 和复用的低精度分解解修正；收敛要求近似逆足够好，$\kappa(A)u_{\rm low}\ll1$ 是常用诊断而非对所有实现的完整保证。

## N8 谱与子空间扰动

**Weyl。** 对 Hermitian $A,E$，有序特征值满足 $|\lambda_i(A+E)-\lambda_i(A)|\le\|E\|_2$；奇异值满足同样的谱范数扰动界，不需要 $A$ 为方阵。

**Bauer–Fike。** 若 $A=V\Lambda V^{-1}$ 可对角化，则 $A+E$ 的每个特征值距 $\sigma(A)$ 不超过 $\kappa_2(V)\|E\|_2$。正规情形 $V$ 酉，常数为 1；病态特征向量会破坏“权重改得小，谱就改得小”的判断。

**Davis–Kahan。** 对实对称 $A,\widehat A=A+E$，将两组特征值均按降序排列，选同序号 $r:s$ 的 $d=s-r+1$ 维特征子空间 $V,\widehat V$。若原矩阵的外部间隙

\[
\delta=\min(\lambda_{r-1}-\lambda_r,\lambda_s-\lambda_{s+1})>0
\]

（缺失边界用 $+\infty$），则可使用 Yu–Wang–Samworth 的版本：

\[
\|\sin\Theta(V,\widehat V)\|_F
\le\frac{2\min(\sqrt d\|E\|_2,\|E\|_F)}{\delta}.
\]

这里比较子空间主角，允许簇内部重特征值；逐个特征向量可能任意旋转。不要删掉间隙条件。[NL5]

**奇异子空间／Wedin 路线。** 用残差和分离条件的 Wedin 定理，或对 Hermitian dilation $\begin{pmatrix}0&A\\A^*&0\end{pmatrix}$ 应用相应子空间定理。其谱为 $\pm\sigma_i$ 与可能的 0，必须针对所选簇检查正负谱及零谱的分离；仅有奇异值误差界并不足以保证奇异向量稳定。

## N9 Krylov、CG、GMRES 与预条件

对 $Ax=b$，构造 $\mathcal K_m(A,r_0)=\operatorname{span}\{r_0,Ar_0,\ldots,A^{m-1}r_0\}$。无需存储 $A$，但 matvec 的语义、精度和固定性必须明确。[NL1]

**CG：** 要求 Hermitian 正定。每步通过残差内积选择 $\alpha_k=(r_k^*r_k)/(p_k^*Ap_k)$，更新 $x,r$，再用 $\beta_k=(r_{k+1}^*r_{k+1})/(r_k^*r_k)$ 构造共轭方向。精确算术下

\[
\|e_m\|_A\le2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^m\|e_0\|_A.
\]

每步一次 $A$ 作用、少量内积与向量操作，工作空间 $O(n)$；全局归约通信可能主导。

**GMRES：** 一般非奇异方阵用 Arnoldi $AV_m=V_{m+1}\overline H_m$，解小最小二乘 $\min_y\|\beta e_1-\overline H_my\|_2$。最小化当前 Krylov 空间中的残差；$m$ 步需 $O(nm)$ 基存储，正交化通常 $O(nm^2)$。重启节省空间但可停滞，不能沿用未重启的收敛结论。

Hermitian 不定问题可选 MINRES；矩形最小二乘可选 LSQR／LSMR，以 $A$ 和 $A^*$ 作用避免显式正规方程。Lanczos 是自伴的短递推谱工具；有限精度中注意正交性丢失与伪重复 Ritz 值。

**预条件。** 左预条件残差通常是 $M^{-1}(b-Ax)$，验收仍检查真实残差。CG 的预条件要保持相容正定结构。预条件器每步改变时，应使用适合变动预条件的算法，例如 flexible GMRES。比较 Jacobi、块 Jacobi、ILU、稀疏近似逆、multigrid 时计入 setup、apply、内存与通信；不以单步迭代数代替总成本。

## N10 随机范围寻找与随机 SVD

目标 rank $k$、超采样 $p$，令 $\ell=k+p$：

1. 抽取 $\Omega\in\mathbb R^{n\times\ell}$，形成 $Y=A\Omega$。
2. 正交化 $Y=QR$；在慢衰减谱下，用交替 $A^*$、$A$ 作用及每轮 QR 做稳定子空间迭代。
3. 形成 $B=Q^*A$，对小矩阵做 SVD；返回 $Q\widetilde U_k\Sigma_kV_k^*$。
4. 用独立探测方向估计 $\|(I-QQ^*)A\|$，不以训练草图自身残差代替独立检查。

对实矩阵、标准 Gaussian $\Omega$、$k\ge2,p\ge2,k+p\le\min(m,n)$，范围投影满足

\[
\mathbb E\|(I-QQ^*)A\|_F
\le\sqrt{1+\frac{k}{p-1}}\left(\sum_{j>k}\sigma_j^2\right)^{1/2}.
\]

这是 rank-$\ell$ 投影的期望界，不是每次实现的确定性 rank-$k$ 误差保证；最终截断要另计误差。[NL2, Theorem 10.5]

稠密 Gaussian 草图的乘法约 $O(mn\ell)$；不要把结构化随机变换的 $O(mn\log k)$ 复杂度直接安到它身上。$q$ 轮幂迭代和形成 $B$ 合计约 $2q+2$ 次对 $A/A^*$ 的遍历，另计 QR 与小 SVD。磁盘、网络和跨设备遍历可能比 FLOPs 更重要。

## N11 随机迹、Frobenius 范数与 Hutch++

若 $\mathbb E[zz^*]=I$，则 $\mathbb E[z^*Az]=\operatorname{tr}A$。对实对称 $A$ 与 $s$ 个独立样本：

\[
\widehat t=\frac1s\sum_{j=1}^s z_j^\top A z_j.
\]

Gaussian 探针方差为 $2\|A\|_F^2/s$；Rademacher 探针为 $2\sum_{i\ne j}a_{ij}^2/s$。对非对称实矩阵，二次型只感知对称部分。相对误差保证需处理迹接近零的情形；PSD 假设不能无声删除。

标准 Hutch++ 用与残差探针独立的草图建立 $Q$，将迹分解为

\[
\operatorname{tr}A=\operatorname{tr}(Q^*AQ)
+\operatorname{tr}\big((I-QQ^*)A(I-QQ^*)\big),
\]

再只对残差部分随机估计。它降低低秩主空间造成的方差；对 PSD，在固定成功概率下可将相对误差 $\varepsilon$ 所需 matvec 数从标准估计的 $O(\varepsilon^{-2})$ 改善到 $O(\varepsilon^{-1})$，详细置信参数按原定理。[NL6]

同理 $\mathbb E\|Az\|^2=\|A\|_F^2$，可估计隐式 Jacobian 的能量；配合 HVP 可以估计 Hessian 迹，但 Hessian 未必 PSD，不能因此给出正值或相对精度承诺。

## N12 子空间嵌入、ID、CUR 与 Nyström

**子空间嵌入。** 对 $0<\varepsilon<1$，若草图 $S$ 对 $\operatorname{col}[A\ b]$ 中所有 $v$ 满足 $(1-\varepsilon)\|v\|^2\le\|Sv\|^2\le(1+\varepsilon)\|v\|^2$，则草图最小二乘解 $\widetilde x$ 满足

\[
\|A\widetilde x-b\|^2\le\frac{1+\varepsilon}{1-\varepsilon}\min_x\|Ax-b\|^2.
\]

这是直接比较两个目标得到的保证。随机 JL、SRHT、CountSketch 的维数与失败概率各依赖构造；只保持有限个点的距离不自动等于保持整个子空间。稀疏嵌入能降低草图构建成本，后续小系统仍需稳定求解。

**ID／CUR。** ID 使用原列 $C=A[:,J]$ 和系数 $T$ 表示 $A\approx CT$；CUR 同时保留原列和原行 $A\approx CUR$。这些形式适合可解释选择、采样与隐式访问，稳定性取决于选取与中间求解，不能把任意交叉子矩阵求逆称为可靠低秩逼近。

**Nyström。** 对 PSD $K$，选列 $C=K[:,J]$，交叉块 $W=K[J,J]$，构造 $\widehat K=CW^+C^*$。需处理 $W$ 的病态、正则化及采样分布；有噪声时正则化不是代数恒等式。若 $K$ 来自连续核，还要检查积分权重。

## N13 算子接口与计算成本

大规模工具的最小接口通常是：形状、$v\mapsto Av$、$u\mapsto A^*u$、多右端作用、dtype、是否随调用变化，以及可接受误差。反向 AD 可以提供伴随作用，但需使用相同状态、随机性和边界定义。

选择工具时对比：矩阵物化量、matvec 次数、访问次数、正交化与全局归约、工作内存、分解复用范围、预条件器／草图构建成本。所有“快”都要指向这个成本模型或真实计时。张量网络、分解因子与稀疏格式可实现这些作用，但额外的重压缩误差必须进入算法误差预算。

## 来源

- **[NL1]** Barrett 等，[Templates for the Solution of Linear Systems](https://www.netlib.org/templates/templates.pdf)，第 2 章迭代方法、第 3 章预条件、第 4 章相关技术。实际核对 CG、GMRES、停止标准。
- **[NL2]** Halko, Martinsson & Tropp (2011), [Finding Structure with Randomness](https://tropp.caltech.edu/papers/HMT11-Finding-Structure.pdf)，§3–6、Theorem 10.5：范围寻找、后处理、幂迭代与误差；ID／Nyström 见相应后处理节。
- **[NL3]** Higham，[What Is the Sylvester Equation?](https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/)，谱分离、Kronecker 表示与平方根导数。
- **[NL4]** Higham，[What Is a Fréchet Derivative?](https://nhigham.com/2020/06/23/what-is-a-frechet-derivative/)，定义、条件数与非交换导数；Al-Mohy & Higham (2009), [Computing the Fréchet Derivative of the Matrix Exponential](https://eprints.maths.manchester.ac.uk/1218/1/alhi09.pdf)，Theorem 2.1、§2–3 给出块公式与作用计算；Higham & Al-Mohy (2010), [Computing Matrix Functions](https://eprints.maths.manchester.ac.uk/1451/1/paper8.pdf)，§2.3 Theorem 2.5 核对主平方根的谱域。
- **[NL5]** Yu, Wang & Samworth，[A useful variant of the Davis–Kahan theorem](https://arxiv.org/pdf/1405.0680)，Theorem 2、Theorem 4；[Demmel 讲义 19](https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Lectures/Lecture_19.html) 用于原领域的特征／SVD 扰动与算法选择。
- **[NL6]** Meyer, Musco, Musco & Woodruff (2021), [Hutch++](https://arxiv.org/pdf/2010.09649)，Algorithm 1、§2–3、Theorem 1。

其他基本恒等式在本参考的维度、范数与可逆性条件下可直接验证；逐篇阅读范围与扩充来源见 `source-coverage.md`。
