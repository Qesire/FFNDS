# 张量微积分与系统来源覆盖

此台账记录论文中的可复用数学机制及其在技能中的落点。C 编号对应 [张量微积分与自动微分](tensor-calculus-and-autodiff.md)，E 编号对应 [张量编译、布局与硬件数学](tensor-compilers-and-layouts.md)。核对日期：2026-11-16。

## 目录

- [既有论文的方法覆盖](#既有论文的方法覆盖)
- [补充原领域常用工具](#补充原领域常用工具)
- [既有官方实现资料](#既有官方实现资料)
- [来源边界与继续阅读规则](#来源边界与继续阅读规则)

“已覆盖”表示工具的对象、核心关系、使用方式与条件已进入参考；不表示复刻全文证明、所有实验配置或逐条搬运附录。发现任务需要尚未展开的分支时，沿本台账读取原节，再补充适用条件与工具卡。

## 既有论文的方法覆盖

| ID 与原始来源 | 实际核对范围 | 可复用工具及落点 | 必须保留的边界 |
| --- | --- | --- | --- |
| CS01 Laue, Mitterreiter, Giesen, 2018, *Computing Higher Order Derivatives of Matrix and Tensor Expressions*；[NeurIPS 原文](https://proceedings.neurips.cc/paper_files/paper/2018/file/0a1bf96b7165e962e90cb14648c9462d-Paper.pdf) | §2 表达语言及 Ricci 翻译；§3 DAG、前反向传播、二阶求导例子；表 1–3 | 向量/协向量/线性映射槽位；δ 恒等张量；表达 DAG 的路径链式累积；重复求导与化简；回译矩阵表达。C1、C2、C4、C5 | 论文使用的逐元素扩展须与严格几何 Ricci 记号区分；欧氏度量识别不能视为自然的 V≅V*；历史框架性能不外推。补充材料中的每个实验图未复现。 |
| CS02 Laue, Mitterreiter, Giesen, 2020, *A Simple and Efficient Tensor Calculus for Machine Learning*；[arXiv v1](https://arxiv.org/html/2010.03313v1) | §2 扩展 Einstein 乘法；§3.1–3.2 的通用乘法/一元函数/逐元素函数前反向规则及证明结构；§3.3 cross-country 与压缩 | 显式输出标签确定求和集；链式法则的收缩表示；图切面上的混合 AD；按结构重排；恒等因子移除；矩阵分解目标的块 Hessian 压缩。C1、C2、C5 | 求解小块仅省去大矩阵分解，仍有多右端成本；网络 Hessian 压缩需逐项核对共享/线性结构。未把 §4 旧框架速度比较加入默认事实。 |
| CS03 Chiang, Rush, Barak, *Named Tensor Notation*；[作者规范正文](https://namedtensor.github.io/)、[论文记录](https://arxiv.org/abs/2102.13196) | §2 非正式算子；§5 records、shapes、部分索引、广播扩展、common operations；§6 输入/输出轴微分 | 命名轴兼容性；切片；作用域内广播；收缩与归约；轴重命名；导数轴分离；attention 的轴语义。C1–C3，types-and-indices | 轴名不是物理布局；同大小不同身份仍须改名。其 §3 各模型/概率例子作为用法库，技能提取共用操作而不重复所有模型公式；其 soft argmax 极限在并列极值时分配质量，不是某个库的离散索引返回规则。 |
| CS10 Kjolstad 等, 2017, *The Tensor Algebra Compiler*；[作者机构原文](https://commit.csail.mit.edu/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf) | §3 dense/sparse 格式树；§4 iteration graphs；§5 merge lattice 构建与三种 dense 简化；§6.1–6.4 生成、表达提升、assembly 与并行限制；§7 assemble/compute 接口 | 格式层组合；pos/idx；遍历依赖；支持集合交并；表达树生成游标合并；公共子表达式提升；输出结构复用。E3 | §4/§10 明示原生成器不支持有环迭代图；§6.4 并行有额外限制。输出候选非零与实际值非零不同。未将基准结果当作当前 sparse backend 性能。 |
| CS11 Vasilache 等, 2018, *Tensor Comprehensions*；[arXiv 原文](https://arxiv.org/pdf/1802.04730) | §5 逻辑到访问模型、范围与足迹；schedule trees；§5.2 调度、§5.3 不完美循环嵌套分块、§5.4–5.6 GPU 映射与 memory promotion；§6 cache/autotuning | 域/访问/依赖/调度四元组；范围推断与越界判定；合法融合/分块；共享内存 footprint 与同步；语义和调度分离；编译配置缓存。E2、E4、E9 | affine/static-control 范围必须声明；数据间接访问需要额外处理；§5.6 不能泛化为自动生成完备寄存器调优。附录的具体 API 及全部实验不写死进技能。 |
| CS12 Dao 等, 2022, *FlashAttention*；[NeurIPS 版](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)、[含附录 arXiv v2](https://arxiv.org/html/2205.14135v2) | §3 分块、重算、IO 与 block-sparse；附录 B.1–B.4 前/反向关系与 RNG 重放；C 中 Theorem 2 和 Proposition 3 的参数及证明；D.1 的支持块 | max/normalizer/weighted-sum 状态合并；按 tile 重算概率；`D_i=barO_i·O_i`；线性中间存储；跨内存层流量；masked block 选择。E5 | 实数 exact 不承诺逐位相同；dense 工作仍二次；IO 的“对所有 M 无一致小 o 改进”不等于逐点最优。只保留完成的方法，未把 D.2 中未来扩展当作已实现能力。 |
| CS13 Zhou 等, *Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F₂*；[arXiv v5, 2026-03-06 / ASPLOS 2026](https://arxiv.org/html/2505.23819v5) | §4.1–4.4 定义、复合/直和/左除/右逆、特定布局家族与 shape-operation 闭包；§5.1–5.5 连续性、广播、混合精度、SIMD、shuffle、swizzle、gather；附录 §9.1–9.2 闭包及 bank 模型 | 硬件位到逻辑位的线性映射；rank/kernel；右逆与复本选择；形状传播；指令 tile 分解；通信分层；基空间分割；bank 冲突；scale 广播与 gather 条件。E6–E7 | “全部布局”限定论文所定义的家族；零列判重只在特定分布布局充分；一般 F₂ 线性求解自由变量归零不保证全局最小 Hamming weight 或通信。论文最优性须绑定其事务与布局模型；未复现硬件 benchmark。 |
| CS14 Williams, Waterman, Patterson, 2008 report / CACM 2009, *Roofline*；[原技术报告入口](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)、[作者版本](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)、[机构存档](https://escholarship.org/uc/item/5tz795vq) | 获取作者版本的检索正文，核对 §3 operational intensity、性能 roof 与瓶颈含义，§7 常见误解及 A.1 流量/带宽测量说明；部分 PDF 直取超时 | 带宽—计算上界、分层算术强度、测量与静态流量区分、优化 ceiling。E4 | bound/bottleneck 模型不等于延迟预测；分母是指定层级的实际流量。没有逐图重读所有机器实验，不引用其中 speedup 作为本机结论。 |

## 补充原领域常用工具

### CS04 矩阵算子的方向微分与伴随

[Mike Giles, 2008, *An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation*](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)。核对 §2 乘积/逆/行列式及 trace 配对、§3 特征/奇异向量的谱间隙分母、§4 Cholesky 与 §5 范数。落点 C2–C3。使用其逐算子线性化方法；不能因采用矩阵记号而宣称求导不可能。重根附近要更换目标或使用不变子空间。

### CS05 Hessian 的直接作用

[Barak Pearlmutter, *Fast Exact Multiplication by the Hessian*, 1994；作者机构保存的 1993 稿](https://mural.maynoothuniversity.ie/id/eprint/5501/1/BP_fast%20exact.pdf)。核对 §2 差分关系、§3 R 算子与传播规则、§4.1 前馈网络应用。落点 C4。提取 HVP 机制；全文其它随机/循环网络变体未逐式展开。

### CS06 稀疏导数的图着色

[Gebremedhin, Manne, Pothen, 2005, *What Color Is Your Jacobian?*，作者 PDF](https://eecs.wsu.edu/~assefaw/CSCAPES-src/coloringpage/abstracts/sirev.pdf)。核对导论的结构正交列、前/反向分组，§3.5 的二部图 distance-2 对应与 Theorem 3.5，§4 Hessian 对称结构、§5 双向划分入口及 §6.1 / §7.2 两色森林恢复。落点 C6。提供列着色种子/恢复、Hessian 的 star 直接恢复与 acyclic 叶子消元。求最少颜色需专门启发式；并未把所有双向着色、部分矩阵恢复和最优图算法逐个实现。

### CS07 隐式微分

[Blondel 等, 2022, *Efficient and Modular Implicit Differentiation*](https://proceedings.neurips.cc/paper_files/paper/2022/file/228b9279ecf9bbafe582406850c57115-Paper-Conference.pdf)。核对 §2.1 root、fixed-point、JVP/VJP 线性系统与求解器边界，表 1 的最优性条件类型。落点 C7。近似根的 Jacobian 精细误差定理未写成已验证通用常数；当前工具要求原问题与伴随残差分别验收。

### CS08 复数/实微分

[Kreutz-Delgado, 2009, *The Complex Gradient Operator and the CR-Calculus*](https://arxiv.org/pdf/0906.4835)。核对 §1–3 实值复变量目标、全纯条件及 Wirtinger 导数，并补核 §4.1–4.2 式 (22)–(26)（PDF pp.14–15）的向量 Wirtinger 导数、Jacobian／共轭 Jacobian 与 `dw=A dz+B dbar(z)`。落点 C8。固定 `dℓ=Re(gᴴdw)` 后，实损失的输入梯度 `Aᴴg+Bᵀbar(g)` 由配对独立推导；不能把非全纯导数简化为单一复线性 Jacobian。完整复 Hessian 和 Newton 分块公式未在技能中逐条复制。

### CS09 程序反演与 checkpoint

[Griewank & Walther, 2000, *Optimal program execution reversal*](https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/download/616/484)；[Revolve 原论文记录](https://users.fmi.uni-jena.de/~autodiff/?id=griewank2000ara&module=Publications&submenu=list+publications)。核对前者 §1 的固定状态大小、§2 的切向/伴随传播、§3 的离散反演问题，并核对 Revolve 论文身份。落点 C9 的分块重算推导与进一步调度入口；不将尚未逐式移植的 Revolve 最优控制器宣称为已实现脚本。

### CS15 IO 复杂度

[Hong & Kung, 1981, *I/O Complexity: The Red-Blue Pebble Game*](https://www.cs.rochester.edu/~cding/Teaching/573Spring2005/ur_only/HongKungSTOC81.pdf)。核对 §2 的操作规则/双层内存解释及普通矩阵乘法下界的对象范围。落点 E4。用于明确模型与通信下界；FFT、排序网络、图乘积等原论文其它案例没有逐个扩成专用工具。

### CS23 舍入误差的确定性与概率模型

[Higham & Mary, 2018 preprint, *A New Approach to Probabilistic Rounding Error Analysis*](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf)。核对引言的 γ_n、浮点模型、Model 2.1、Lemma 2.2 及 Theorem 2.4。落点 E8。确定性界与基于独立零均值舍入假设的概率界分开；不能把“随机误差通常抵消”当作任意程序的保证。矩阵分解各专门概率界未逐条展开。

## 既有官方实现资料

以下是实现语义和参数核对入口，不将某个版本的约束写成永久硬件定律。

| ID | 一手文档与已核对内容 | 工具落点 |
| --- | --- | --- |
| CS16 | [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)：structured op、indexing maps、iterator types 与 scalar region | E2，把轴与归约语义保留到编译表示 |
| CS17 | [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)、[Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)：coordinate→index、shape/stride、coalesce、composition/product/divide | E6；与 F₂ 布局区分，计算真实地址及分区 |
| CS18 | [Triton Gluon layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)：register/lane/warp 分布、blocked 参数、tiling/broadcast、寄存器占用与向量访问 | E4、E7；不以逻辑 shape 替代物理复本数 |
| CS19 | [NVIDIA Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)：结合序、FMA、rounding mode、编译选项 | E8；只使用运算与舍入语义。文档中的历史 CPU/GPU 产品概括不作为当前事实 |
| CS20 | [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：block matmul、stride 指针、尾部 mask、program ordering/L2 与 autotune | E4、E9；具体 API 使用时按安装版本核对 |
| CS21 | [NVIDIA GEMM performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：FMA/FLOP、算术强度、tile 及形状影响 | E4、E9；不把示例设备的经验阈值推广到所有设备 |
| CS22 | [cuSPARSELt Data Types](https://docs.nvidia.com/cuda/cusparselt/types.html)：数据类型对应的结构模式、compute mode、batch stride 与 epilogue 属性 | E8；设备、dtype、操作数方向、metadata 必须共同核对 |
| CS24 | [A100 June 2021 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)：第一页不同容量/封装的规格表、稀疏峰值脚注 | E4 成本估计输入。只按指定 SKU 读取理论峰值，不视为实际 kernel 吞吐；不重复历史产品营销比较 |
| CS25 | [CUTLASS custom epilogue fusions](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)：GEMM 后融合激活/广播/辅助输出、示例设备要求 | E1、E9；融合保留必要辅助输出，并计入流量和改变的舍入点 |
| CS26 | [isl 0.28 Manual — Schedule Trees](https://libisl.sourceforge.io/user.html#Schedule-Trees)：band 的部分调度、sequence/set 的次序含义，band 的 permutable／coincident 属性及查询接口 | E2；band 不自动保证可排列或可并行，先核对依赖与属性，再选择硬件映射 |

## 来源边界与继续阅读规则

1. 优先调用正文已有的公式、条件和可核验例子。若需要超出正文的定理或算法分支，按此处章节进入原文，核对符号与量词后再使用。
2. 论文中对“所有”“最优”“精确”“压缩”的限定必须随工具一起保留。分别检查表达语言、数组/几何类型、计算图家族、存储模型、容量区间和精度。
3. 原文有不严谨概括或特定实现限制时，不把它升格为数学定理。本文右逆一般解、简单 checkpoint 成本、归约伴随、LayerNorm 方向导数等是按定义重推的可复用关系。
4. 历史论文提供机制；软件文档提供具体版本的接口与约束；实际速度必须由目标程序和设备测量。来源相符不替代代码执行证据。
