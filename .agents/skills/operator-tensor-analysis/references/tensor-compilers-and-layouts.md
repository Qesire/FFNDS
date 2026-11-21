# 张量编译、布局与硬件数学

将同一算子分别表示为数值关系、迭代域、调度、访问映射、存储及线程归属。每一步给出保持的语义和引入的成本。常用 FFN 成本、动态状态与量化分组例子在 [收缩与执行](contractions-and-execution.md)，CS 来源定位见 [台账](sources-calculus-and-systems.md)。

## 目录

- [E1 收缩图、路径与物化](#e1-收缩图路径与物化)
- [E2 多面体域、访问与合法调度](#e2-多面体域访问与合法调度)
- [E3 稀疏格式、迭代图与 merge lattice](#e3-稀疏格式迭代图与-merge-lattice)
- [E4 分块、融合、访存与算术强度](#e4-分块融合访存与算术强度)
- [E5 在线 softmax 与反向重算](#e5-在线-softmax-与反向重算)
- [E6 布局代数与 F₂ 线性映射](#e6-布局代数与-f₂-线性映射)
- [E7 广播、向量化、shuffle 与 swizzle](#e7-广播向量化shuffle-与-swizzle)
- [E8 数值精度、量化与稀疏指令](#e8-数值精度量化与稀疏指令)
- [E9 自动调优、状态摊销与验收](#e9-自动调优状态摊销与验收)

## E1 收缩图、路径与物化

**对象。** 用超图表示 einsum：张量是结点，轴标签是连接其出现位置的超边，输出标签是开放轴。某标签只有在不再被剩余操作数及最终输出需要时才能消去。先合并两个操作数时保留这个规则，避免过早求和。

**成本。** 稠密二元收缩的直接工作量由参与轴大小的乘积决定；结果存储由保留轴大小的乘积决定。最少 FLOP、最小峰值内存、最少通信和最短延迟未必是同一路径。矩阵链可用动态规划；一般网络需要启发式、分区、slicing 或路径搜索，网络宽度见 [张量网络与统计](tensor-networks-and-statistics.md)。

例 `Y_ij=Σ_ab A_ia B_ab C_bj`，A:m×r、B:r×s、C:s×n：

\[
(AB)C:\quad 2mrs+2msn,\ \text{中间量 }m\times s;
\qquad
A(BC):\quad 2rsn+2mrn,\ \text{中间量 }r\times n.
\]

m=n=4096、r=s=32 时，两条路径的主 FLOP 同量级。跨调用缓存 AB 要求 A、B 都保持有效，缓存 BC 要求 B、C 都保持有效；仅共享其中一个因子并不足够。针对一次调用和跨调用分别分析生命周期。einsum 可表达等价收缩，不承诺优化器会找到所需路径。

**融合与重算。** 若中间量被单次消费，融合可能省写回；多消费者时融合可能重复计算。表达共享、数值舍入点、缓存可用性和消费者布局一起决定是否物化。把逐元素激活穿过求和一般不合法。GEMM epilogue 可融合广播、激活及必要辅助输出；辅助输出仍占写回流量。[CS25]

## E2 多面体域、访问与合法调度

**形式化。** 对每条语句 S 定义整数点域 `D_S={i∈Z^d:A i+B p+c≥0}`、读写访问 `f_A:D_S→coordinates(A)`、调度 `θ_S:D_S→Z^k`。p 为循环中不变的参数。若存在必须保持的依赖 `(S,i)→(T,j)`，要求

\[
\theta_S(i)\prec_{\rm lex}\theta_T(j).
\]

先定位读后写、写后读、写后写和别名，再判定 loop interchange、fusion、skewing、tiling、parallelization 是否合法。归约重排还需要结合性假设或允许的数值误差。静态 affine 模型不直接涵盖任意数据依赖的索引与分支。[CS11]

**范围推断与访问安全。** 给定输入合法域 T，检查访问 footprint `f_A(D_S)⊆T`；必要时对未知界进行约束求解/投影。卷积 `X[i+r]` 的输出范围来自 i、r 的联合约束，不应由“两个轴大小相同”推断。padding、clamp、mirror 都改变访问函数，须显式建模。

**调度树。** band 记录一个或多个部分调度维；只有通过依赖检查的 permutable band 才能任意交换成员，band 本身不保证可排列或可并行。sequence 规定子域次序，set 不规定子域间次序，filter 限制语句实例，extension 插入拷贝等辅助语句。把 tile 分为块坐标与块内坐标，再对合法的并行维选择 block/thread 映射；把 footprint 搬入共享内存时加入拷贝和同步依赖。[CS11, CS26]

**MLIR 对接。** Linalg 的 indexing maps 描述循环点到各操作数的映射，iterator types 指出 parallel/reduction，region 描述标量负载。保留其结构可支持分块、融合、向量化和向库算子的 lowering。`parallel` 表示逻辑迭代语义，不自行确定某个 GPU lane。[CS16]

## E3 稀疏格式、迭代图与 merge lattice

**格式层。** 把张量视为坐标树：每层对应一个模式及其存储顺序。dense 层保存大小并支持直接寻址；compressed 层用 `pos[parent]:pos[parent+1]` 给出 `idx` 片段，叶子定位 vals。CSR、CSC 是不同层序；显式零、重复坐标和排序状态必须另定。[CS10]

**迭代图。** 为每个访问构建依照物理格式的轴路径；兼容路径共同决定可顺序遍历的循环森林。CSR 与 CSC 的冲突可能产生有向环。TACO 2017 的生成算法要求无环迭代图；不能把论文“任意表达式”的概括误读为无需格式转换即可支持所有遍历。[CS10]

**支持集合代数。** 对同一坐标域上的逐元素乘加，在通常有限数值且零为乘法吸收元的语义下，

\[
\operatorname{supp}(b+c)\subseteq S_b\cup S_c,\qquad
\operatorname{supp}(bc)\subseteq S_b\cap S_c.
\]

使用包含关系，因为抵消可让候选位置变零；NaN、Inf、非零 fill value 或特殊半环需重新推导。对 `a_i=b_i+c_i d_i`，候选域为 `S_b∪(S_c∩S_d)`。合并有序游标，遇到共同坐标求完整表达，某些流耗尽后转入剩余有效子表达。merge lattice 从表达树自底向上合成这些活动流组合与循环分支。[CS10]

**格式优化。** 两个 dense 域只需一个计数器；sparse×dense 可按 sparse 支持遍历并随机访问 dense；相加则仍可能需要完整 dense 域。对

\[
A_{ij}=M_{ij}\sum_k B_{ik}C_{kj},
\]

已知 sparse mask M 时只在其候选位置计算点积（SDDMM）。若 mask 要在该点积之后才能决定，则不能先扣除点积成本。

**assembly 与 compute。** 先形成输出 pos/idx，再计算数值，可在结构不变而数值变化时复用 assembly；结构变化必须重建。输出重复写入涉及 workspace、原子操作或归约，需计算其成本。TACO 2017 的外层 OpenMP 并行规则有额外限制：不合并游标、不产生未同步 scatter、输出各层 dense；这些是该版本实现限制，不是稀疏计算的数学限制。[CS10]

## E4 分块、融合、访存与算术强度

**tile 容量。** GEMM tile `B_m×B_n×B_k` 至少需考虑 `B_m B_k+B_k B_n` 个输入元素和 `B_m B_n` 个累加器；乘以各自字节数，再加流水缓冲、padding、scale 和同步开销。增大 B_k 不必提高整个程序的复用，且可能挤占并发度。

**IO 模型。** 红蓝 pebble 把计算图结点值分配到容量为 M 的快存储和大慢存储：操作数在快存储才能计算，跨层读写计数。它区分“有多少运算”与“为运算搬多少次值”。对普通方阵乘法计算图，输入最初位于慢存储、输出最终写回慢存储，且 M 足够容纳基本计算时，相应双层模型给出 `Ω(n³/√M+n²)` words 的通信下界；它绑定普通乘加算法和存储假设，不能作为所有快速矩阵乘法或任意硬件的无条件下界。[CS15]

**Roofline。** 对指定内存层级 h，流量 Q_h 以 bytes 计，以 β_h、π 分别表示该模型采用的带宽与计算吞吐上界：

\[
I_h=F/Q_h,\qquad P\le\min(\pi,\beta_h I_h),\qquad
T\ge\max(F/\pi,Q_h/\beta_h).
\]

多层分析分别计算 HBM、L2、共享内存等流量。若 β_h 或 π 来自某个基准的实测持续值，它们是经验校准的 roof，不能无条件视作所有 kernel 的严格上界。原 Roofline operational intensity 以缓存过滤后的 DRAM 流量为分母。数组容量、逻辑访问次数、实际事务量与测得流量不能互换。[CS14]

**瓶颈判断。** 低于 roof 不自动说明“带宽瓶颈”：小 kernel 的 launch、依赖链、占用率、访存延迟、同步与格式转换也可支配时间。减半 FLOP 而增加 Q_h 可能更慢；跨线程复用是否省全局事务取决于合并访问和缓存。[CS18, CS21]

## E5 在线 softmax 与反向重算

**归约状态。** 对一个 query 和非空的已处理、未屏蔽 key 集 J，假设这些分数有限，维护

\[
m=\max_{j\in J}s_j,\quad
\ell=\sum_{j\in J}e^{s_j-m},\quad
z=\sum_{j\in J}e^{s_j-m}v_j.
\]

对不相交集合的状态 `(m₁,ℓ₁,z₁)`、`(m₂,ℓ₂,z₂)`，设 m=max(m₁,m₂)，并以 `exp(m_k−m)` 同时缩放 ℓ_k、z_k 后相加。最终 ℓ>0 时输出 z/ℓ。空块须使用专门恒等状态分支，避免 `−∞−(−∞)`；若整行没有有效 key，普通 softmax 的分母为零，必须另定零输出或报错等算子语义，不能把空块分支当作已定义的行输出。NaN／正无穷分数也需单独约定。这种状态支持分块、树形合并和流式计算。[CS12]

**反向公式。** 无 dropout 时，S=τQKᵀ、P=softmax(S)、O=PV，给定 barO：

\[
\bar V=P^\top\bar O,\quad
\bar P=\bar O V^\top,\quad
D_i=\sum_a\bar O_{ia}O_{ia},\quad
\bar S_{ij}=P_{ij}(\bar P_{ij}-D_i),
\]

\[
\bar Q=\tau\bar S K,\qquad \bar K=\tau\bar S^\top Q.
\]

保存每行归一化统计量与 O，在反向按 tile 重算 P；D_i 的式子避免再次对长度 N 的 P、barP 作完整归约。dropout 必须重放同一 mask/RNG 对应关系，且在相应乘法处加入 mask 与缩放。[CS12]

**IO 条件。** 原论文在 `d≤M≤Nd`、M 以元素容量计的双层模型中，选 `B_c=Θ(M/d)`、`B_r=Θ(min(M/d,d))`，得到 `Θ(N²d²/M)` HBM 访问和 `Θ(N²d)` 工作。论文下界排除的是在整个 M 区间均渐近改进的算法，不能改写为对每个固定 M 都达到全局最优。block-sparse 扩展仅遍历允许块，相对该 masked 算子是精确实现；若从 dense attention 删除块，则是模型近似。[CS12]

## E6 布局代数与 F₂ 线性映射

**先区分映射。** CuTe Layout 使用 shape/stride 组成逻辑坐标到索引空间的映射，并提供 composition、product、division 等组合操作。其整数坐标/层级 stride 代数不能直接替换为二进制无进位加法。[CS17]

对满足模型的二进制 tile，Linear Layout 用

\[
L:\mathbb F_2^{p_{reg}}\oplus\mathbb F_2^{p_{lane}}
\oplus\mathbb F_2^{p_{warp}}\to\mathbb F_2^{q_{logical}},
\qquad x=Lh.
\]

列是每个硬件坐标 bit 的逻辑作用；加法是 XOR。组合用矩阵乘，独立扩展用直和。论文的 sliced layout 指归约后删除逻辑维，对应投影并删除矩阵行；它与提取某个固定非零坐标的数组切片不同，后者若有物理原像，该原像是仿射陪集，需要保留偏移。一般布局含进位、非二幂域或数据依赖映射时，不一定是这一域上的线性映射。[CS13]

**秩与复本。** 若 `rank L=r`，则每个可达逻辑点有 `2^(p−r)` 个物理原像；只有 r=q 时才覆盖全部逻辑坐标。零列表示该 bit 不改变逻辑位置；一般 L 的复本由整个 ker L 描述，不能只检查零列。论文特定 distributed-layout 家族的非零列为互不重复的标准基，故零列检测足够。[CS13]

**转换。** 固定方向：目标物理位置 h_t 读取源物理位置 `h_s=T h_t`。在同一逻辑坐标的源复本确实保存同一个值的前提下，索引正确性是

\[
L_sT=L_t.
\]

若 L_s 满射，求右逆 R 满足 L_sR=I；所有线性解可写作

\[
T=RL_t+KZ,\qquad \operatorname{im}K=\ker L_s.
\]

用 Z 选择复本，优先保留目标的 warp/lane 坐标。R 非唯一，且 RL_s 一般不等于 I。对一般二进制系统，“把自由变量全设零”不保证最小通信或最小 Hamming weight；须依据当前布局族与约束判定。[CS13]

**可直接检查的小例子。** `L_s=[1,0]`、`L_t=[1,1]`，坐标顺序为 `(register,lane)`。取

\[
T=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad L_sT=L_t.
\]

目标 lane 保持不变，只交换该 lane 的 register。机械选择 `R=(1,0)ᵀ` 会得到 `RL_t=[[1,1],[0,0]]`，把所有读取送到 lane 0；利用 ker L_s 可消除不必要通信。

## E7 广播、向量化、shuffle 与 swizzle

**连续访问。** 将逻辑到内存 offset 的映射与线程布局组合，检查相邻 lanes / registers 的地址差、对齐、尾部 mask 和有效元素。在适用的位布局模型内，可分离的低位恒等直和因子支持连续向量化；不能只检查一个恒等子块而忽略同一 register bit 对高 offset bits 的作用。例如 $r\mapsto(r,r)$ 的最低位恒等，但两个地址为 0、3，并不连续。shape 的最后一维大小为 1 也不应自动否定其它维的连续性。[CS13, CS18]

**硬件指令匹配。** 若 memory layout M 采用“内存坐标→逻辑坐标”的双射约定，而 distributed layout L 采用“线程坐标→逻辑坐标”，则 `M⁻¹L` 给出每线程的内存坐标。若 M 采用 CuTe 的“逻辑坐标→offset”约定，则地址映射直接为 `M L`。先检查定义域和值域再复合，不能只照搬字母顺序。若所需小指令 tile T 可以作为映射的直和因子分离，则可使用相应 SIMD load/store；必要时先在寄存器内重排。此处 layout 的 division 指结构分解，不是标量除法。[CS13]

**通信分级。** 若转换 T 保持 lane 和 warp，仅需寄存器重排；保持 warp 但改变 lane 时考虑 warp shuffle；改变 warp 时通常需要共享内存及同步，或者重新选择源复本。对无广播的标准基布局，共同 register 基可组成向量化子空间，共同 lane 基无需交换，剩余基成对 XOR 后可划分 shuffle 轮次。指令每次可传输的位宽必须计入轮数。[CS13]

**bank 冲突。** 把共享内存位分成 vector、bank、segment。构造置换或 XOR swizzle 以保持向量化，同时让并发 lane 的 bank 映射尽量单射。在“每 lane 每次访问一个完整 bank word、无广播”的简化模型中，若 bank 位映射在活动 lane 子空间的核维数为 k，则同 bank 竞争度为 2^k。较宽或较窄事务、多段地址、同址广播须用对应指令模型重算。优化写与读两端的布局，不能只优化一侧。[CS13]

**混合精度与 gather。** scale 的 reshape/transpose/broadcast 必须同数据块对应，布局传播可选择共同 scale 的装载者；预 shuffle 也须保持元素配对，且计入预处理摊销。gather 若源和索引涉及的全部元素都在同一 warp，可由 shuffle 完成；只要有跨 warp 的候选位置，就不能仅凭输出 shape 小而采用这一策略。[CS13]

## E8 数值精度、量化与稀疏指令

**浮点模型。** 在无溢出/显著下溢等条件下，可用 `fl(a op b)=(a op b)(1+δ)`、`|δ|≤u`。顺序点积的标准绝对误差界形如

\[
|\widehat{x^\top y}-x^\top y|\le\gamma_n\sum_i|x_i y_i|,
\qquad \gamma_n=nu/(1-nu),\quad nu<1.
\]

严重抵消时相对误差可很大。树形归约、补偿求和、FMA 和高精度累加改变舍入路径，需按具体算法建立界；FMA 是一次舍入，不等同于分别乘加。数学结合律不能保证位级相同。[CS19, CS23]

调用 [CS23] 的独立舍入误差模型时，需要说明独立、零均值等假设，并给出相应概率；不能无条件把确定性 n 因子换成 √n。重复操作数、相关计算和确定性舍入都可能破坏这些假设。其他概率模型可使用不同条件，但须援引其对应结论，不能把独立性当作所有随机舍入分析的必要条件。

**混合精度。** 分别记录存储、解码/乘法、累加和最终写回格式。低精度输入误差 `ΔA,ΔB` 与算法舍入共同贡献

\[
\widehat C-AB=(\Delta A)B+A(\Delta B)+(\Delta A)(\Delta B)+E_{round}.
\]

消去中间低精度写回或更换归约序，可能提高误差也可能降低误差；用任务定义的容限判断，保留 NaN/Inf、signed zero 等真正影响语义的分支。

**量化分组。** `A_i=s_{g(i)}(q_i-z_{g(i)})` 中 g 是计算语义的一部分。变换或分块必须同时变换 g、scale、zero-point、padding 与打包位置。仅排列 q 而保持旧分组元数据通常改变 A。分组缩放可否移入/移出收缩取决于它是否在归约轴上恒定。

**结构化稀疏。** 稀疏指令需要具体的模式、数据类型、元数据编码、对齐和操作数方向。不能把元素稀疏率直接乘到 dense FLOP；应记录被执行的块/指令、压缩和重排成本。实现时核对实际 cuSPARSELt / CUDA / 设备版本支持范围。[CS22]

## E9 自动调优、状态摊销与验收

**调优变量。** 合法调度空间内比较 tile、线程布局、pipeline stage、fusion、split reduction、向量宽度、格式和库调用。Tensor Comprehensions 的可复用思想是把语义与调度参数分开，在目标 shape 与设备上编译、计时并保存配置；历史 genetic/random search 不构成今天最优搜索法的保证。[CS11]

**缓存键。** 需要包含影响正确性的算子语义、shape/stride、dtype/精度模式、设备/编译器、静态条件、mask 或量化结构。数值锚点缓存和编译缓存是不同状态；其失效条件分别建立。

**摊销式。** 构建成本 C₀、每次复用成本 C₁、原成本 C_base，K 次调用后收益要求

\[
C_0+K C_1<K C_{base}.
\]

若每周期刷新一次，按真实周期累计。packing、layout conversion、mask prediction、assembly、同步和额外输出都纳入对应路径。仅在其后确实可复用时才摊销。

**验收。** 先核验逻辑轴及索引等价，再检查边界形状、尾块、重复 gather、空归约与状态变化；用适合 dtype 的误差标准比较参考输出。性能测量包含预热、异步完成、可重复输入及真实调用方式；报告测得延迟和配置，区分静态模型、生成代码与实机证据。硬件峰值与旧论文 speedup 只用于限定范围内的估计。[CS20, CS21]
