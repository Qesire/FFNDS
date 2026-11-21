# 收缩与执行

用本参考把数学表达落实到迭代、数据访问、状态、布局和数值语义；按当前问题选择静态分析或执行证据。

## 内容

- 第一层 数学语义
- 第二层 迭代域与访问
- 第三层 存储和执行映射
- 布局上的二进制线性代数
- 状态 稀疏与量化
- 成本与验收
- 编译方法的来源

将数值精度和状态作为贯穿条件，分三层分析实现。

## 第一层 数学语义

\[
Y_{pa}=\sum_i X_{pi}A_{ia}+C_{pa}.
\]

先说明 p、a 是输出轴，i 是归约轴，C 是什么状态。一般 einsum 只规定值之间的关系；相同公式可以对应不同收缩路径和中间量。

## 第二层 迭代域与访问

\[
\mathscr D=\{(p,a,i):0\le p<N,\ 0\le a<o,\ 0\le i<d\},
\]

\[
\pi_X(p,a,i)=(p,i),
\qquad \pi_A(p,a,i)=(i,a),
\qquad \pi_Y(p,a,i)=(p,a).
\]

p、a 对应可并行输出，i 对应归约。归约仍可通过 split-K 等策略并行，只是必须处理部分和。这里与 [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) 的 indexing maps、parallel/reduction iterator 直接相连。

分块是对索引域的分解：

\[
p=b_pB_p+u,\qquad a=b_aB_a+v,\qquad i=b_iB_i+w,
\]

\[
Y^{(b_p,b_a)}_{uv}\mathrel{+}=
\sum_w X^{(b_p,b_i)}_{uw}A^{(b_i,b_a)}_{wv}.
\]

每个输出 tile 先用对应的 C 初始化一次，再遍历全部 b_i 累加；尾块须限制有效的 u、v、w，不能对每个归约块重复添加 C。由此可计算局部输入、权重和累加器大小，并权衡复用、寄存器、共享内存、并发度与尾块。tile 较大不自动更快。

## 第三层 存储和执行映射

对简单、未打包的 strided array：

\[
\operatorname{addr}_X(p,i)=\operatorname{base}_X+
b_X(p\,s_p+i\,s_i).
\]

其中 b_X 是每元素字节数，s_p、s_i 以元素为单位；若 stride 已以字节计，不再乘 b_X。地址映射与元素归属不同。元素归属还需要：

\[
\lambda:(\operatorname{register},\operatorname{lane},\operatorname{warp})
\longrightarrow (u,v).
\]

逻辑坐标相同的数组可以使用不同 stride 或不同线程归属。transpose 可能只改变元数据，也可能因消费方布局要求产生拷贝。是否使用高效 MMA 路径还取决于目标架构、数据格式、累加格式、指令形状与布局；维度对齐本身不足以给出性能结论。参考 [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) 和 [Triton Gluon layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)。

## 布局上的二进制线性代数

[Linear Layouts](https://arxiv.org/html/2505.23819v5)将一类硬件索引位到逻辑坐标位的映射写成：

\[
u=Lb,\qquad L\in\mathbb F_2^{q\times p}.
\]

这里改变的是索引位，张量数值仍属于原来的数值域。对源布局 L_s 和目标布局 L_t，寻找从目标物理位置到源位置的转换 T，应满足：

\[
L_sT=L_t.
\]

只有 L_s 双射时才可直接写 T=L_s⁻¹L_t。若源布局有广播副本，通常存在多解；应利用 ker(L_s) 选择保持线程或 warp 归属的解。

例如源布局 L_s=[1 0]、目标布局 L_t=[1 1]，物理位为 (r,w)。源的两个 warp 都持有同一组值。以下转换保留 warp：

\[
T=
\begin{pmatrix}1&1\\0&1\end{pmatrix},
\qquad L_sT=L_t,
\qquad w_s=w_t.
\]

所以目标可由各 warp 内的寄存器重排实现。任意选择一个把所有读取指向 warp 0 的右逆，会夸大通信需求。这个框架适用于所定义的布局族，不能把任意整数地址函数都当作二进制线性映射。

## 状态 稀疏与量化

### 可合并状态与 FlashAttention

对一行 attention 的有效 key，假设未屏蔽分数有限，使用状态 m、l、z，分别保存当前最大分数、归一化质量及加权 value。两个非空块状态可合并为：

\[
m=\max(m_1,m_2),\qquad
l=e^{m_1-m}l_1+e^{m_2-m}l_2,
\]

\[
z=e^{m_1-m}z_1+e^{m_2-m}z_2,
\qquad O=z/l.
\]

至少有一个有效 key 时，在实数运算下这保留全局 softmax 的正确归一化。不能分别求各块 softmax 后简单平均输出。全空或全 mask 块需要单独的空状态约定，以避免 −∞−(−∞)；若整行都无有效 key，还必须另定输出语义，不能执行 O=z/l。NaN 或正无穷分数也不在上述有限分数假设内。

这种表示允许分块生成并消耗 score，避免完整 N×N 矩阵落入 HBM；dense attention 的二次交互仍然存在。[FlashAttention 原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)提供相应 IO 分析。可迁移的问题是：哪些状态足够继续计算，中间量在哪个内存层级生成和消失，是否可以用重算减少存储。

### 稀疏支持与决策时点

将激活乘 mask 后继续执行同形状 dense GEMM，改变了数值非零量，却没有据此减少实际乘加。

固定行批量约定 $\operatorname{FFN}(X)=\phi(XW_1+\mathbf1b_1^\top)W_2+\mathbf1b_2^\top$，其中 $W_1\in\mathbb R^{d\times m}$、$W_2\in\mathbb R^{m\times o}$。若共享隐藏支持集合 S 在第一层投影前已知，且激活按隐藏坐标独立作用，可选择 W₁ 的对应列、b₁ 的对应元素和 W₂ 的对应行，精确执行该 masked FFN。只有删去分量本来就不贡献时，它才等于原未屏蔽 FFN；跨隐藏坐标的归一化等操作还需另查。若 mask 依赖已经算出的完整当前激活，就不能把第一层已完成的计算计为节省。

逐 token 支持集合还需处理索引表示、gather、packing、分组、负载不均和输出累加。应说明实际执行了哪些 tile。目标硬件的结构稀疏格式需要另外匹配；例如特定 2:4 路径要求规定归约分组与压缩元数据，一般 50% 零值不足以证明匹配。具体格式按[目标库的类型说明](https://docs.nvidia.com/cuda/cusparselt/types.html)核对。

### 量化分组也是算子定义

对一个明确的对称分组量化例子，把归约轴写成 i=gB_q+u。以解码后的低精度数值表示 q̃：

\[
X_{pgu}\approx s^X_{pg}\widetilde X_{pgu},
\qquad
A_{gua}\approx s^A_{ga}\widetilde A_{gua},
\]

\[
(XA)_{pa}\approx
\sum_g s^X_{pg}s^A_{ga}
\sum_u\widetilde X_{pgu}\widetilde A_{gua}.
\]

这只描述两边分组相容、零点为零的 GEMM 部分；完整仿射输出还要加 C。非对称量化要保留零点修正；真实格式还要明确码字、scale、打包与解码方式。

排列量化数据时，要保持每个逻辑元素与 scale、zero-point 的对应。重新分组并重新校准会改变近似。量化块与执行 tile 是不同对象；低比特存储、硬件原生低比特乘法和实际提速也应分别判断。

### 浮点执行

记录会影响结论的存储、乘法、累加格式和舍入位置。例如：

\[
Q_{\rm out}\bigl(\phi(Q_z(z))\bigr)
\quad\text{与}\quad
Q_{\rm out}\bigl(\phi(z)\bigr)
\]

具有不同的中间舍入。结合律改写、融合、缓存精度或归约树可能改变结果。[NVIDIA 浮点说明](https://docs.nvidia.com/cuda/floating-point/index.html)给出不同合规求和方式产生不同结果的例子。

验证应围绕用途选择绝对误差、相对误差、方向作用或最终输出质量，并说明零分母和近零值的处理。只有目标明确要求时才追求逐位一致。

## 成本与验收

### 同时记录运算 中间量和复用

批处理大小为 N，输入 d，输出 o，隐藏 m，因子宽度 r。这里的 FFN 采用上述两次投影、无门控分支的结构；按一次乘加计 2 FLOPs，仅计主要 GEMM：

| 计算路径 | 主要 FLOPs | 重要状态或中间量 |
| --- | --- | --- |
| 两次投影 FFN | 2Nm(d+o) | N×m 隐藏量 |
| 稠密 XA+C | 2Ndo | N×o 缓存 |
| 完整低秩替代 | 2Nr(d+o) | N×r 中间量及 N×o 缓存 |
| 稠密 A 加动态低秩修正 | 2Ndo+2Nr(d+o) | 额外 N×r 状态、增益及预测成本 |

若改为 $\bigl[\phi(XW_g)\odot(XW_v)\bigr]W_o$，两个输入投影均为 $d\to m$ 时，主要 GEMM 为 $2Nm(2d+o)$，不能沿用两次投影的数值。中间容量与实际 HBM 流量分别计算。后者还依赖缓存命中、重复读取、spill、packing、融合和实际调度。

以 N=31668、d=o=5120、m=13824、r=64 为静态算例，权重和缓存按每元素 2 字节：

| 路径或对象 | 算术量或容量 |
| --- | --- |
| 上述两次投影 FFN | 约 8.966 TFLOP |
| 稠密 A | 约 1.660 TFLOP |
| 完整 rank 64 替代 | 约 41.508 GFLOP |
| 稠密 A 加 rank 64 修正 | 约 1.702 TFLOP |
| 单个 N×d 缓存 | 309.258 MiB |
| 40 个这样的缓存同时存活 | 12.080 GiB |

这些是公式直接计算的量。TFLOP 表示运算量，TFLOP/s 才表示吞吐。额外低秩修正的 GEMM FLOPs 约为基础 GEMM 的 2.5%，并不意味着耗时只增加 2.5%；分开执行可能增加大张量的读写。

### 锚点周期按实际调度计算

令一次 dense 调用主要运算量为 F_D，一次稠密替换为 F_A。对于“dense 锚点，再额外执行 X_sA 建缓存，随后命中一次”的指定调度：

\[
\frac{\text{基线两次调用的 FLOPs}}
{\text{候选两次调用的 FLOPs}}
=
\frac{2F_D}{F_D+2F_A}
=
1.459\ldots
\]

单次命中的 F_D/F_A=5.4 不能代表整个周期。不同缓存构建调度会有不同账本。运行时间需计入真实锚点、缓存、命中、预测与更新路径：

\[
T_{\rm cycle}=T_{\rm anchor}+T_{\rm cache}+K\,T_{\rm hit}
\]

这个加法仅适用于这些阶段按所述方式串行、且计时范围不重叠的情况。

### 使用 Roofline 定位可能瓶颈

\[
I_{\rm HBM}=\frac{F}{B_{\rm HBM}},
\qquad
T\ge
\max\!\left(
\frac{F}{P_{\rm peak}},
\frac{B_{\rm HBM}}{\operatorname{BW}_{\rm peak}}
\right).
\]

这是理想下界，不能替代计时。launch、同步、索引、布局转换、通信和利用率由实际执行决定。公式见 [Roofline 原始报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)，GEMM 形状和分块影响参见 [NVIDIA 矩阵乘性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

| 剩余问题 | 足够且有区分力的证据 |
| --- | --- |
| 等价式是否正确 | 代数推导，必要时小规模数值反例 |
| 实现是否对应公式 | 真实调用路径、轴、状态与数值对照 |
| 完整候选是否更快 | 同一配置下覆盖完整路径的可靠计时 |
| 为何变慢 | 围绕未排除解释选择流量、利用率、launch 或布局证据 |

完整计时已能回答是否更快时，不必自动追加全面 profiling。只给出静态分析的任务也无需等待 GPU。

## 编译方法的来源

[TACO 2017](https://tensor-compiler.org/files/kjolstad-oopsla17-tensor-compiler.pdf)将索引公式与稀疏／稠密存储格式一起用于循环生成。[Tensor Comprehensions 2018](https://arxiv.org/abs/1802.04730)把扩展索引表达与形状推断、融合、代码生成和调优连接起来。

从这些系统提取“公式、迭代、存储和调度分别表达”的方法。具体硬件能力、库接口和性能结论绑定到实际设备与版本；需要实施时核对当前官方文档。
