# 共享输入排列的联合函数界：精确线性支持、GELU断点与HiF4闭包的独立审阅

**日期：2026-10-27。独立评议对象：G12 单行重排界与 I-P1 共享输入机制。**

同一输入排列的共享结构，允许把任意**固定的有符号 W1 线性功能**先合并为一个输入方向，再通过重排给出精确支持函数。这一结果直接暴露出新的研究对象：哪些非线性残差保留这种相关性、哪些 HiF4 整组约束可以廉价复用。线性部分属于成熟的 permutahedron／majorization 与重排理论；有意义的 OPEN 问题在真实 tanh-GELU、Q2 与执行代价的交界处。

本审阅的主要判定为：

- **ALG-SUPPORT**：`conv{b + W P x} = b + W Perm(x)`，且固定方向的支持函数可精确化为对 `Wᵀc` 的重排。
- **ALG-CORRECTION**：`b + W Perm(x)` 一般是离散可达域的精确凸包，不能把它称为离散联合精确可行域。
- **ALG-COUNTEREXAMPLE**：真 tanh-GELU 可严格反转线性排序的极值；凸化输入再过 GELU 也会引入虚假的最坏输出。一般精确非线性支持问题包含 PARTITION 判定。
- **SRC + ALG**：HiF4 Q2 的 64→8→4 动态尺度会广播依赖；即使固定尺度，舍入也能破坏 GELU 的正负配对线性恒等式。
- **OPEN**：应先比较“固定方向精确投影”和“CROWN／DeepPoly 的关系型传播＋输入重排支持 oracle”，再研究更便宜的非线性联合证书。

本文件为本人独立推导与一手原文／源码核验。起始仓库 HEAD 为 `0ca46a2e98e096461cc9556bea274992d337766e`；未修改 ACTIVE、图谱、QUEUE 或其他 Agent 文件。新增数值枚举、真实张量分析、CPU 模型、GPU／视频运行、安装及提交均为 **0**。以下小例子为符号构造，不是运行结果。

## 1. 先固定信息对象：共同排列、有序权重与尺度

令输入 multiset 为 `x ∈ R^n`，`P` 遍历所有 `n×n` 排列矩阵，`W ∈ R^{H×n}` 是**按真实输入坐标排列的已知权重矩阵**，bias 为 `b`。定义

\[
\mathcal O(x)=\{Px:P\in\mathfrak S_n\},\qquad
\Pi(x)=\operatorname{conv}\mathcal O(x),\qquad
\mathcal Z=b+W\mathcal O(x).
\]

实际未知的是同一个 `P`，所有 W1 行共用它。把每行各自选择的 `P_j` 拼在一起，会改变问题的信息集合。

这里的排列是“只保留量化后输入 multiset 时尚未读取的坐标配对”的数学不确定性，并非自然视频或原模型输入真的按这些排列分布。若已经读取当前完整有序 Q1 向量，实际输入是其中一个确定点；证书只能通过减少信息读取或复用统计获得经济意义。

对真实量化尺度有两种合法写法：

1. 若 `x` 是已解码实数输入，`W` 是固定解码权重，所有固定 weight scale 已经包含在 W 内，线性投影不再要求各行 weight scale 相同。
2. 若 `x` 表示 code multiset，则允许的排列块必须与已知 input scale／code 对应关系一致。块内共同正尺度 `s_G` 可以提出重排；不同坐标的动态 scale 不能随意丢弃。

HiF4 的 4／8／64 层级元数据还可能约束哪些 code assignment 能产生观测到的尺度。仅把相同 scale 的坐标划组、允许组内所有排列，可能仍是合法外包络；要宣称离散域精确，须证明这些排列都与保留的量化元数据一致。原 G12 同时使用两侧 code histogram 的单行推导，仍须遵循其 scale 交集条件。

## 2. 凸包精确、离散域通常不凸：一个必要区分

**命题 1（ALG）。**

\[
\boxed{\operatorname{conv}(\mathcal Z)=b+W\Pi(x).}
\]

证明只用线性映射与凸组合交换：

\[
\sum_r\lambda_r(b+WP_rx)=b+W\sum_r\lambda_rP_rx,
\quad\lambda_r\ge0,\quad\sum_r\lambda_r=1.
\]

这给出的是联合域的**最小凸外包络**。离散域 `𝒵` 是有限集，若它等于凸集 `b+WΠ(x)`，它必须只有一个点。因此非平凡情况下二者不能相等。例如

\[
x=(1,-1),\qquad
W=\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad b=0
\]

只有 `(2,−2)` 和 `(−2,2)` 两个可达点；`WΠ(x)` 还包含 `(0,0)` 等线段内部点。

若 x 非常数，则排列差向量的线性张成空间为 `1^⊥`。证明：选出两个不同的 x 值，排列后把它们放在任意两个坐标 i、j，交换这两项便生成非零倍数的 `e_i−e_j`。故联合域为单点的充要条件是 W 消灭 `1^⊥`，即每个 W 行在本组所有输入坐标上恒定。x 本身为常数也是退化例外。

对于互相独立的输入组 `G`，定义 `Π = ∏_G Π(x_G)`。有

\[
\operatorname{conv}\{b+\sum_GW_G P_G x_G\}
=b+\sum_GW_G\Pi(x_G).
\]

等式右侧是 Minkowski 和。若组之间另有位置／尺度约束，应把约束保留在输入域中；不能自动套用乘积域。

经典 majorization 描述给出 `Π(x)={u:u≺x}`，等价于总和相等及所有 k 个坐标子集的和不超过 x 最大 k 项之和。也可用 `u=Dx`，D 为 doubly stochastic 矩阵。**同一个 D 必须被全部 W 行共用。** 这些是经典结构，不是本审阅的新算法。[Chong 1978 原始证明](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D2B7C6EE869A682FFAB2BF32B64D3657/S1446788700011551a.pdf/a-note-on-rados-theorem.pdf)，[Postnikov §2，Definition 2.1／Proposition 2.5](https://arxiv.org/pdf/math/0507163)。

## 3. 固定 signed 方向的联合支持恰为一次重排

给定事先固定的 `c ∈ R^H`，令 `p=Wᵀc`。于是

\[
\begin{aligned}
h_{\mathcal Z}(c)
&=\max_P c^\top(b+WPx)\\
&=c^\top b+\max_Pp^\top Px\\
&=c^\top b+\sum_{k=1}^{n}p_{(k)}x_{(k)},
\end{aligned}
\]

其中 `(k)` 表示升序，保留数值符号。下支持／最小值是

\[
\min_{z\in\mathcal Z}c^\top z
=c^\top b+\sum_{k=1}^{n}p_{(k)}x_{(n+1-k)}.
\]

证明的核心是交换一对逆序配对：当 `p_i≤p_j`、`a≤b` 时，同序相对反序增加 `(p_j−p_i)(b−a)≥0`。逐步消除逆序得到同序最优；反序给最小值。重复值只造成多个极值排列，不影响公式。线性函数在集合与其凸包上的支持相同，所以此处凸化**没有支持间隙**。

独立分组时，对每组分别排序 `p_G=W_Gᵀc` 与 `x_G`，把支持相加。这保留所有输出行共享输入排列的事实；它不要求构造 H 维联合多面体。

### 3.1 和逐行最紧区间的差距来自什么？

若 `l_j,u_j` 为每行的精确范围，则逐行 box 的 signed 上界为

\[
U_{\rm box}(c)=\sum_{c_j\ge0}c_ju_j+
\sum_{c_j<0}c_jl_j.
\]

共享支持满足 `h_𝒵(c)≤U_box(c)`。去掉 bias 后，该不等式是支持函数的次可加性：

\[
h_{\mathcal O(x)}\left(\sum_jc_jw_j\right)
\le\sum_jh_{\mathcal O(x)}(c_jw_j).
\]

**等号充要条件**：所有 `c_j≠0` 的 signed 行目标 `c_jw_jᵀPx` 存在同一个最大化排列。若 x 各项互异，存在共同排序顺序是直观表述；存在重复值／weight ties 时，应直接使用 argmax 集合有共同交点的条件。

在第 2 节的两行反对称例子中，`c=(1,1)` 使 `Wᵀc=0`，故真实支持为 0；逐行 box `[-2,2]^2` 的支持为 4。这里的收紧由不可同时实现的行极值产生。

更一般地，`rᵀW=0` 蕴含精确联合等式 `rᵀ(z−b)=0`。当前 H=13824、D=5120 的矩形 W1 使这种联合约束成为自然对象，但在 GELU 之后这些等式一般改变；不能直接搬运隐藏线性子空间。

### 3.2 单行 weight histogram 不包含跨行配对

取相同输入 `x=(1,−1)`、`c=(1,−1)`，比较

\[
W_A=\begin{pmatrix}1&-1\\1&-1\end{pmatrix},\qquad
W_B=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
\]

两者每行 signed histogram 完全相同；但 `W_Aᵀc=0`，联合支持为 0，而 `W_Bᵀc=(2,−2)`，联合支持为 4。

因此，“保存全部行 histogram”与“知道有序 W 后离线形成 Wᵀc”是不同的信息预算。后者可以成立，读取／预计算／储存的代价必须登记。单靠原先边缘 histogram，无法唯一恢复所需的跨行权重结构。

## 4. 真 tanh-GELU：排序反例、凸化间隙和可保留的线性方向

以本项目实际使用的 tanh 公式为数学对象：

\[
\phi(t)=\frac{t}{2}\{1+\tanh[\kappa(t+a t^3)]\},
\quad \kappa=\sqrt{2/\pi},\quad a=0.044715.
\]

原始 GELU 论文 §2 同时给出 erf 定义和此 tanh 近似；本项目 `F.gelu(..., approximate='tanh')` 选择后者。下述恒等式针对这个**真实被选择的 tanh 函数**，没有用 ReLU 或 erf 代替。[Hendrycks–Gimpel 原论文 §2](https://arxiv.org/pdf/1606.08415)。

### 4.1 严格极值反转：单个 GELU 已足够

取 `x` 的 multiset 为 `{−2,−1,+1}`，`W=e₁ᵀ`，`b=0`，输出方向 `c=−1`。线性目标 `−z` 的重排最优排列把 `−2` 放到第一坐标。实际目标是 `−φ(z)`，却有

\[
\boxed{-\phi(-1)>-\phi(-2)>0,\qquad -\phi(1)<0.}
\]

严格证明：设 `u_t=κ(t+a t³)`，则 `−φ(−t)=t/(1+exp(2u_t))`。`u₂>2u₁`。令 `Y=exp(2u₁)`，由 `u₁>1/2` 得 `Y>e>1+√2`，所以 `Y²>1+2Y`，进而

\[
e^{2u_2}>Y^2>1+2Y,
\quad \frac1{1+Y}>\frac2{1+e^{2u_2}}.
\]

所以真实最坏排列把 `−1` 放在第一坐标。这个例子使用的输入值都属于常见 FP4 可表示值；它没有借助小数枚举或数值拟合。零填充可嵌入更大的排列块。

### 4.2 “先凸化再过非线性”与“非线性输出的凸包”不同

令 `x=(1,0)`，`z₁=(Px)₁−1/2`，`z₂=−z₁`，`c=(−1,−1)`。两个排列都给出 `z₁=±1/2`。由于

\[
\phi(t)+\phi(-t)
=t\tanh[\kappa(t+a t^3)]>0\quad(t\ne0),
\]

离散域上的最大目标严格为负。但 `u=(1/2,1/2)∈Π(x)` 给出 `z₁=z₂=0`，目标为 0。

因此

\[
\max_{P}c^\top\phi(WPx+b)
<\max_{u\in\Pi(x)}c^\top\phi(Wu+b)
\]

可以严格发生。`conv{φ(WPx+b)}` 仍对**非线性输出的线性方向**给精确支持，但 `φ(WΠ(x)+b)` 没有这种保证。应明确凸化发生在哪个边界。

### 4.3 有用的例外：反对称配对精确保留线性分量

由于 tanh 内部是奇函数，

\[
\boxed{\phi(t)-\phi(-t)=t.}
\]

写成

\[
\phi(t)=\frac12t+r(t),\qquad
r(t)=\frac12t\tanh[\kappa(t+a t^3)],\qquad r(-t)=r(t),
\]

便能看出自由度：若两行满足完整预激活关系 `z_k=−z_j`，且功能系数为 `(d,−d)`，偶残差精确抵消，贡献为 `d z_j`，可继续用第 3 节的支持公式。相反，同号系数会消掉线性分量、保留偶残差。

充分条件包括 `w_k=−w_j`、`b_k=−b_j`；省略 bias 会误认配对。Smooth2 若两侧不同，须把功能系数也按各自平滑尺度调整，才有相同恒等式。实际是否有这样的可用结构属于 OPEN，未读取真实权重张量判断。

另一个窄例外：若 W 只是同一增益下的坐标选取／排列，bias 也相同，GELU 先逐元素作用于 multiset 后仍可排序。但任意稠密 W 混合坐标时，`φ(WPx)` 一般不是某个固定向量的排列。

### 4.4 一般精确非线性支持包含 PARTITION

取任意正整数 `a₁,…,a_n`，令 `S=Σa_i`。输入 multiset 含 n 个 1 与 n 个 0；令

\[
w=(a_1,\ldots,a_n,\underbrace{0,\ldots,0}_{n}),\qquad
t=w^\top Px-S/2.
\]

令两个 W1 输出为 `(t,−t)`，目标系数为 `(−1,−1)`。则

\[
F(P)=-\phi(t)-\phi(-t)
=-t\tanh[\kappa(t+a t^3)]\le0,
\]

等号当且仅当 `t=0`。n 个零权重坐标允许任意大小的原 a 子集被填入 1，其余 1 放到零权重位置。因此

\[
\max_PF(P)=0
\iff \exists A\subseteq[n]:\sum_{i\in A}a_i=S/2.
\]

这是从 PARTITION 到本支持判定的多项式归约。它说明一般问题已有组合困难，即使仅两个相反的 affine 隐藏行也如此；一个通用的精确“再次排序”公式需要额外结构。

此处的复杂度结论针对输入维度及有理权重位长可增长的数学问题，属于一般问题的弱 NP-hard 性。它不证明固定 B=64、冻结 HiF4／BF16 权重的实用求解必然昂贵，也不证明任何近似证书无价值。此归约为独立推导；未进行全面复杂度文献优先权宣称。

## 5. HiF4 Q2：广播尺度和固定尺度舍入是两个不同断点

### 5.1 本次直接读到的 SRC 路径

读取的本地源为：

- `.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py` 的 `HIF4CacheRuntime.apply`：先 FP32 除 Smooth scale、cast 回输入 dtype，再调用 activation QDQ，然后执行缓存权重的线性操作。
- `.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py`：`QType('hifx4')`，`force_fp32`，metadata 明确是 float QDQ、BF16 GEMM。
- `.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py#quant_hifx`：reshape 为 `(-1,8,2,4)`，由绝对值逐级取得 max4、max8、max64，随后计算共享尺度、8／4 子尺度与舍入。
- 同目录 `QType.py`：`hifx4` 的 `man_bits=3`。

外层尺度 σ 来自 max64/7 及 BF16、E6M2 形式的取整；reciprocal 也 cast 到 BF16。两个子尺度由 max8 和 max4 决定。若暂把 reciprocal／cast 误差移到独立误差项，固定尺度分支中的量化格距为 `σ·s₈·s₄/4`，并带上界饱和。其真实代码包含的所有 cast、clip、floor 与 round 仍须在原生证书中保留。

历史文件 [E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json](../../../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json) 已记录两个**原 Python reference** 调用：单项 `7→14` 令所有 64 个 QDQ 输出改变，未改变输入的 index8 输出 `1.25→1.5`。本次只阅读此保存结果并核对同 SHA 源码，没有复跑。该结果身份仍是 synthetic Python reference，非新增 CUDA 等价验证。

### 5.2 即使尺度已固定，GELU 的精确配对也会消失

构造一个合法 Q2 输入组：index0 固定为 7；index8、9 为 `φ(t),φ(−t)`，`t=1/8`；其他项为 0。两项均满足 `|φ(±t)|<1/8`。正 anchor 7 可由某个固定正 GELU 输入产生，故这与 GELU 输出范围相容。

按原 reference 的运算常量，`BF16(1/7)=73/512`，`7×73/512=511/512` 是 BF16 相邻值的中点，round-to-nearest-even 得到外层 σ=1；后续外层取整及 reciprocal 均保持 1。index8、9 所在 8／4 子组的子尺度都是 1，所以两项按 1/4 格距舍入为 0：

\[
Q_2(\phi(t))=Q_2(\phi(-t))=0,
\qquad
Q_2(\phi(t))-Q_2(\phi(-t))=0\ne t.
\]

这是依原 reference 方程得到的符号反例，未执行 reference。输入 BF16 cast 对这两项仍保持严格小于 1/8，量化为零的结论不依赖它们的小数近似。连续实数 GELU 恒等式与原生 BF16 GELU 的逐位等价是不同主张；本反例只需展示 Q2 舍入本身可额外破坏配对。

从误差角度看，`Q₂(v)=v+e(v)`；成对 GELU 的差变成 `t+e(φ(t))−e(φ(−t))`。量化器全向量符号对称 `Q₂(−v)=−Q₂(v)` 也不能令这两项误差自动相同，因为 `φ(−t)` 不是 `−φ(t)`。

### 5.3 保留哪些条件才可能继续做便宜联合界？

一个可操作的条件分支是先证明某 64 组所有允许输入落在固定 scale／code cell 内。如果整组输出 code 都确定，Q2 输出可以当常量；若只有尺度确定，仍需对各 rounding cell 合法处理。若尺度尚未确定，必须联合约束 max64、max8、max4 及相应取整，不能只对单 neuron 删除值加一个独立噪声。

满幅饱和、BF16 reciprocal 与 casts 使“每项误差总是半个量化间距”需要附带条件。只有已证明未饱和且算术语义匹配的固定分支，才能直接使用最近舍入半格距界。

这些 scale／code 信息首次出现于真实 post-GELU／pre-Q2；若目标是省 W1，应从已有信息证明它们，而不是先运行完整 GELU/Q2 后再把它们计作免费前置信息。

## 6. 最强成熟基线：关系型仿射传播后，最后只做一次输入支持

CROWN 的核心做法是用合法 affine 上下包络替代非线性，并按下游系数正负选择上／下包络，把最终方向反传到输入。DeepPoly 同样保留 affine 关系并作 substitution。输入域改成 permutation domain 后，最后的线性优化可以使用第 3 节精确 oracle。[CROWN §3.1，Definition 3.1／Theorem 3.2](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf)，[DeepPoly §4.4](https://ggndpsngh.github.io/files/DeepPoly.pdf)。

具体地，若对每行的合法区间有

\[
\ell_j(z)=\alpha_j^Lz+\beta_j^L
\le \phi(z)/s_j
\le \alpha_j^Uz+\beta_j^U=u_j(z),
\]

为了界定 `Σc_j φ(z_j)/s_j`，当 `c_j≥0` 时选上包络，当 `c_j<0` 时选下包络。把所选系数记为 α、β，令 `d_j=c_jα_j`，`η=Σc_jβ_j`，则

\[
\boxed{\max_P\sum_jc_j\phi((WPx+b)_j)/s_j
\le\eta+d^\top b+h_{\Pi(x)}(W^\top d).}
\]

这是成熟 affine bound propagation 加上特殊输入域支持函数的组合。它保留共享输入相关性，仍可能因非线性包络产生松弛；不应宣称为新型通用验证算法。GELU 的包络必须覆盖负侧非单调和全部相关曲率区间。CROWN 原文为简便列出非负斜率，并明确说明定理可扩展负斜率；不能直接把 tanh 的单调 S 形包络表代入 GELU。

Q2 可通过经过证明的 group 误差集合增加支持罚项，或直接对整组构造

\[
c_G^\top Q_{2,G}(\phi(z_G)/s_G)
\le d_G^\top z_G+\eta_G.
\]

各组的 `d_G` 与 `η_G` 应先相加，再只对合并后的 `Wᵀd` 做一次输入支持。若先逐组最大化然后相加，会再次丢失不同 hidden 组共享同一个输入排列的相关性。

**真正更近的“多 neuron”先例**是 k-ReLU／PRIMA：它们显式处理组内输入相关性，构造联合非线性约束。PRIMA 的 SBLM 会划分输入区域、加入该区域合法包络、lift 并近似合并凸包；其一般性依赖提供覆盖域的有效局部 bounds。HiF4 的 max／尺度／舍入组合仍需自己的合法分区与 bounds，原论文并未提供 Wan HiF4 算子。[k-ReLU 原论文 §1–2](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)，[PRIMA §5 Algorithm 3 与 §5.1](https://arxiv.org/pdf/2103.03638)。

### 6.1 作者代码实际核到了什么？

- [原 CROWN `get_bounds_general.py`](https://github.com/CROWN-Robustness/Crown/blob/07c7c0f567a50790e451c20c9614d4e276e2d5a3/get_bounds_general.py)：`get_layer_bound_relax_adaptive_matrix_huan_general_optimized` 按 A 的符号挑选 upper／lower slopes、intercepts，然后执行 `A * diag` 与 `A @ W`。`general_fit.py` 给 tanh／sigmoid／arctan 的包络构造；不是现成的 tanh-GELU＋Q2 实现。
- [ERAN `krelu.py`](https://github.com/eth-sri/eran/blob/8771d3158b2c64a360d5bdfd4433490863257dd6/tf_verify/krelu.py)：`KAct.type` 直接列出 ReLU／Tanh／Sigmoid；`get_ineqs_zono` 对 `{-1,0,1}^k` 非零方向取得联合线性界。此方向库正是联合相关性查询的先例；代码中的 k 维方向数增长也有实际成本。
- [auto_LiRPA `gelu.py`](https://github.com/Verified-Intelligence/auto_LiRPA/blob/e3930353a3589f7d10879f1e71223d1eefb57196/auto_LiRPA/operators/gelu.py)：`BoundGelu.forward` 调用默认 `F.gelu(x)`；导数是 erf／Gaussian exp 公式。它界定的是 erf GELU 分支，不能直接替代项目 tanh 公式。本次同时读取 current master 与 pinned blob，关键语义一致，行号因版本布局变化而不同。

这些是源码语义核验，无运行或安装。移植至少需要 tanh-GELU 的有效包络、HiF4 嵌套尺度／舍入语义，以及原生 BF16／FP32 累加误差；一个通用“GELU 已支持”的名称不能替代这三项。

## 7. 物理成本：线性投影已可精确廉价求值，非线性适配会改变预算

固定 c 时预存 `p=Wᵀc`。若当前完整有序输入 x 可读，则

\[
c^\top(Wx+b)=p^\top x+c^\top b
\]

在线只需 O(D) 算术和 p／x 读取便得该线性功能的**精确值**。因此 histogram 支持的第一竞争对象应为这个 collapsed exact projection，不能只与完整 W1 的 O(HD) 对照。这个基线仅解决所指定的线性功能，不等于求出了整个 nonlinear FFN。

若输入本来就提供 B 组内 K 档 code counts，且每组 input scale 是共同正因子，则可以离线把每个 `p_G` 排序，储存其前缀和 `S_G(k)`。设 code 值 `v₁<…<v_K`，counts 累计为 `N_l`，则

\[
h_G=\sum_{l=1}^{K}v_l
\{S_G(N_l)-S_G(N_{l-1})\}.
\]

此式使用 signed prefix sums，直接给第 3 节上支持；乘上组共同正 input scale 即可。它是经典重排在有限码本上的计数实现，并非独立算法首创。

| 情形 | 离线／共享代价 | 在线每 token 的主要代价 | 决策意义 |
| --- | --- | --- | --- |
| 固定 r 个线性方向，精确投影 | O(rHD) 形成 p，储存 O(rD) | O(rD) 点积，输入读取 | 强精确基线 |
| 固定方向，histogram 支持 | 上述投影＋每组排序／前缀和 | hist 产生 O(D)；查询 O(rK·D/B) | 仅当 metadata／访存复用合算才有潜力 |
| 逐行 W1 histogram 界 | 每行组统计及元数据 | O(HK·D/B)，另有 input hist | 可能收紧单行范围，仍须全量遍历 H |
| 每 token 自适应 GELU 包络 | 合法区间、斜率和截距 | 新 `d=c⊙α` 后，通常 O(HD) 形成 `Wᵀd`，外加排序／约束 | 紧界可能重新付出 GEMM 量级工作 |
| 少数共享斜率／group 模板 | 模板 p／prefix 缓存与存储 | 选择／认证模板、有限 queries | OPEN：共享程度、界宽与真实成本共同决定 |

这张表给算术／存储规模，不给任何测量速度。真实 fake-QDQ 路径未必直接产生可免费读取的 packed code histogram，额外编码、binning、launch 和 metadata byte 都要收费；动态坐标尺度还可能改变 p 排序。native packed FP4 和原 HiF4 BF16 路径属于不同格式／执行基线，二者均需保留身份。

`c=W₂ᵀ diag(e5) a` 可以代表指定 residual 方向 a。若 gate、a 或 nonlinear envelope 随 token 改变，相应投影更新也要收费；来自 dense 后验 Jacobian 的 a 属于已付费信号。即使一个局部方向已界定，下游 attention／flow／scheduler 仍是非线性响应，完整视频质量并未随之认证。

## 8. 一手阅读账本与实际检索范围

**检索日期／途径**：2026-10-27；Web search、正式期刊／会议原 PDF、作者代码 raw、GitHub API 只读获取 commit。代表词族为 `Rado majorization permutahedron support rearrangement`、`DeepPoly CROWN affine bound correlation`、`single neuron convex barrier k-ReLU PRIMA`、`GELU tanh approximation bound code`。由 Rado 追至 Chong／Postnikov，由 CROWN／DeepPoly 追至 Salman／k-ReLU／PRIMA，再核对 2026 的多神经元表达力工作。

| 一手资料 | 本人实际阅读位置／深度 | 本次用途与边界 |
| --- | --- | --- |
| [Rado, **An Inequality**, JLMS 1952, DOI 10.1112/jlms/s1-27.1.1](https://londmathsoc.onlinelibrary.wiley.com/doi/10.1112/jlms/s1-27.1.1) | 正式书目信息；未取得全文逐页阅读 | 历史原出处；没有把搜索初现的另一篇 Rado 1952 文章当作本定理 |
| [Chong, **A note on Rado's theorem**, 1978, pp.86–88](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D2B7C6EE869A682FFAB2BF32B64D3657/S1446788700011551a.pdf/a-note-on-rados-theorem.pdf) | 全部 3 页，尤其 p.88 的 theorem／proof；另核正式卷期 | 一手再证明，凸组合／majorization 关系；区分 1978 发表与 2009 上线 |
| [Postnikov, **Permutohedra, associahedra, and beyond**, 2005 preprint](https://arxiv.org/pdf/math/0507163) | §2 Definition 2.1、Proposition 2.5 及 Rado 引文 | 只深读相关几何命题，未声称读完 59 页 |
| [Hendrycks–Gimpel, **Gaussian Error Linear Units**, arXiv v5](https://arxiv.org/pdf/1606.08415) | §2 两种公式与 tanh 系数 | 固定本审阅的 GELU 对象；未援用其其他任务成绩 |
| [Zhang et al., **CROWN**, NeurIPS 2018](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf) | §3.1 Definition 3.1、Theorem 3.2、§3.2 包络结构；作者代码的符号选择与矩阵反传 | 一手方法＋代码；不声称原代码含 HiF4 或 tanh-GELU |
| [Singh et al., **DeepPoly**, POPL 2019](https://ggndpsngh.github.io/files/DeepPoly.pdf) | §4.2–4.4，尤其 affine substitution；浮点 soundness 讨论 | 关系型 bound 的成熟对照；本次未审 ELINA 全部 C 源 |
| [Salman et al., **A Convex Relaxation Barrier…**, 2019](https://arxiv.org/pdf/1902.08722) | 摘要、附录中 bound propagation／CROWN／DeepPoly 的对应式 | 成熟方法统一视角；未复现其 LP benchmark，也未逐项证明全文定理 |
| [Singh et al., **k-ReLU**, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf) | §1–2 的联合反例与机制；ERAN KAct／联合方向代码 | 非线性相关性已有直接先例；未声称 HiF4 可直接运行 |
| [Müller et al., **PRIMA**, POPL 2022 / arXiv v3](https://arxiv.org/pdf/2103.03638) | §1–3 的几何机制、§5 Algorithm 3 与 §5.1 的覆盖域／有效 bounds 前提 | 对真 GELU／多变量 Q2 分支的最近框架；未审所有 PDDM 证明或原生实现 |
| [Mao et al., **Expressiveness of Multi-Neuron Convex Relaxations…**, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/8b7d8779c020fa3671ad267152362d56-Paper-Conference.pdf) | §1–2 定义、§4 相关 barrier 段落 | 当前理论追踪；定理对象是其网络／分层松弛，未把它当作冻结 Wan 的失败证明 |
| CROWN 作者仓库 `07c7c0f567a50790e451c20c9614d4e276e2d5a3` | `get_bounds_general.py` 相关完整 backward 段及 `general_fit.py` 包络函数 | 代码静态阅读，未运行 |
| ERAN 作者仓库 `8771d3158b2c64a360d5bdfd4433490863257dd6` | `tf_verify/krelu.py` KAct、方向查询、grouping；`deeppoly_nodes.py` input／affine 接口 | 前者 pinned 内容再次核对；后者是 current master 阅读 |
| auto_LiRPA 作者仓库 `e3930353a3589f7d10879f1e71223d1eefb57196` | `operators/gelu.py` forward、erf 导数、包络／interval 相关段 | pinned 语义与 current master 对照；未运行 bound |

本地源码静态 SHA256：

```text
hifx.py                       6a8d1f1b8ea49ce099439cff251309b4d9b3eeeab6a7ee0a83bdcf85b24704f7
QType.py                      aec909792bbeb59a1a441bd0f3677585bc9597389c10519185e00a706b56e0fc
lightx2v_wan_distill.py        076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3
```

证据标签：第 2–4 节为 **ALG／MATH_CONDITIONAL**；第 5 节源行为为 **SRC**，历史 JSON 为 **ARCHIVED_SYNTHETIC_REFERENCE**；外部工作为 **PRIMARY_PAPER／OFFICIAL_CODE**。没有新增 NATIVE、VIDEO 或 SYS 结论，也没有把独立数学推导认定为算法优先权已通过。

## 9. 真正的竞争解释与最低成本的下一判别

**最强竞争解释 H-V（成熟关系型验证的域适配）**：线性阶段的“联合增益”由经典支持函数严格解释，非线性阶段的可得收紧可首先由 CROWN／DeepPoly／k-ReLU／PRIMA 类型的关系型松弛解释。新科学价值应落在 Wan 的可复用 signed 功能结构、合法 Q2 closure 以及付费信息预算上。

与之竞争的 **H-S（可复用结构）** 是：实际参数／输入统计使少量固定方向、反对称偶残差抵消或稳定 Q2 group cells 能覆盖大量 token／step，因此用很少的在线适配保留有用方向。**H-C（成本抵消）** 则预测：真正有用的紧包络必须大量 token 自适应，`Wᵀd`、group 分支或额外读取占去所省成本。

三者有不同预测：

- 若只在线性 W1 上出现大幅 box→joint 收紧，而通过 true GELU／Q2 后差距消失，支持 H-V 对阶段性的解释。
- 若某些固定 signed 方向在非线性后仍由同一小族合法约束控制，而且模板可跨大量 token 复用，支持 H-S。
- 若保持非线性紧度所需的方向／scale-cell 数随 token 快速增长，或精确 `pᵀx` 已更便宜，则支持 H-C。

**当前不运行实验的最小后续工作**是：先由独立 reviewer 检查第 4.1／4.2／4.4 的三个符号构造和第 5.2 的 reference 舍入常量；再静态确认真实 Q1 可见的是哪些 code／scale 元数据、其合法排列群与 cost。在这两个对象固定后，才有意义比较相同信息预算下的 row box、CROWN＋permutation oracle、少量 group constraints 和 collapsed exact projection。任何数值枚举／真实张量／原生模型判别由主线程依实际授权另行决定，本审阅不启动。

可供串行 Graph Agent 选择的概念关系是：G12 单行 certificate → **ALG** 共享输入线性支持；共享支持 → **LIT_EQUIVALENT** permutahedron／majorization；共享支持 → **ALG_COUNTEREXAMPLE** GELU 非交换凸化；真实 Q2 closure → **SRC_DEPENDENCY** 整组尺度与 rounding cell；低成本 full-FFN 证书仍为 **OPEN**。本审阅未直接登记图节点或变更既有 RUN 身份。

## 10. 主线程新生成的有限软件对照：独立只读审阅

主线程随后提交了以下三个对象供本 Agent 独立审查：

- [固定协议](../../../../config/e0_shared_permutation_discriminator_20261027_v1.json)，SHA256 `508faee7f6943b467183c47dd88513e8fca9ec47935935494046d67fafe8ce23`。
- [保存结果](../../../20261027/E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.json)，状态为 `PASS_EXACT_FINITE_LINEAR_CHECKS_AND_SCALAR_COUNTEREXAMPLES`。
- [执行源码](../../../../tools/e0_shared_permutation_discriminator.py)，SHA256 `4e1c65b005a04ad8fd72800cd2a64a7cd70d0048d14e41d1fd2a29acae84b287`；对应 tests SHA256 `8e22a72bddad807178ed29f7c470b107e06c4f6e19b6619fc4e62d2d001be753`。

本人只读 protocol、results、tool、tests，并静态核对文件 SHA；**未执行该脚本或 tests，未重新枚举**。以下的运行数值属于主线程保存产物，和本 Agent 的 ALG 推导分开记账。

**审阅结论：对这些固定 case，没有发现阻断数学结论或越过声明范围的问题。**

1. `collapse` 对同坐标 W 列按 c 合并；`linear_check` 在每一个保存的离散输入排列内核验 `cᵀ(Wx)=(Wᵀc)ᵀx`。其整数输入全部使用 Python integer 运算，所以线性范围的等式比较不依赖浮点容差。
2. `support` 按组分别同序／反序排序；rowbox 对 c 的负号正确交换 row 上下端点。本协议四组区间都按顺序、无遗漏、无重叠覆盖输入，因此 `orbit` 的拼接正好实现声明的独立组内排列；该结论针对冻结协议，没有将 helper 当作任意未验证 group 布局的通用 API。
3. 用符号重排逐项核对，三行例 `p=(−1,1,−3,8)` 的范围为 `[-23,23]`，而 rowbox 为 `[-35,31]`；两组例第二组 `p=(3,4,5,−3)` 的范围为 `[-8,17]`，与第一组相加为 `[-31,40]`，rowbox 为 `[-62,67]`。结果中对应上下 witness 都具有合法组内 multiset，并达到所列端点。
4. 固定 case 的排列数为 `2+2+24+24²=604`，与 protocol 和 results 一致。`distinct_hidden_states=541` 等额外枚举统计在此保留为主线程产物；本人没有独立复算这些统计，主要证明也不依赖它们。
5. `nonlinear_checks` 明确计算 stdlib 的连续 tanh-GELU；negative-tail 的两个极值与 §4.1 严格证明一致。antipodal 的正系数版本给两个正离散值，凸化输入 midpoint 给 0；它直接展示的是**虚构更小输出／下支持间隙**。若把同一例子叙述成“上支持的虚假最坏值”，应把方向 c 取负，正如本文件 §4.2。该符号区别不影响“非线性前凸化可引入不可达输出”的结论。
6. 协议和结果均明确区分 `GPU=0`、`model_calls=0`、`actual_Wan_observations=0`、无新视频、`quality_accepted=false`。GELU 标量是双精度软件佐证，不是 BF16／HiF4 原生数值或视频功能结果；本文件 §5.2 的 Q2 构造仍保持符号／源码证据身份。

该实验支持经典数学在这些整数输入上的软件实现一致性，并为阶段性断点提供可检查的例子。它没有测试真实 Q1 元数据可达域、metadata 成本、自适应 CROWN 斜率开销或 Wan 全 FFN／视频 Pareto。

**交付与所有权**：此独立文件由 `/root/joint_permutation_independent_review` 独占编辑，已完成。后续 QUEUE、图谱、ACTIVE、协议运行状态及收尾提交由主线程串行维护。gate 的来源虽可沿 modulation／embed0 追踪，当前四步配置下是否可作为 input／seed 无关的离线常量尚未在本审阅证明；全文只在明确固定 c 的条件下使用预计算。
