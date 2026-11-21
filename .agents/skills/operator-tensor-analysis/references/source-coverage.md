# 来源覆盖与知识维护

本表连接已收集的论文、讲义与知识条目，并记录新增原领域来源。核对日期：2026-11-16。覆盖单位是定义、运算、算法、结构判据、适用条件和误差机制，也保留概念之间的联系与尚待发展的迁移线索；实验复现、全文逐条证明和原文列出的所有参考文献另行判断，不能从“已入库”推出这些工作已经完成。

## 目录

- 既有 17 篇论文与讲义的覆盖入口
- 四组详细来源台账
- 算子、数值与优化的新增来源
- AI 学习、生成与交互的来源入口
- 概念关系与知识迁移的来源入口
- 公式依据与当前读取边界
- 新论文进入知识库的方式

## 既有 17 篇论文与讲义的覆盖入口

这 17 项由 6 篇张量方法、8 篇微积分／系统论文和 3 篇几何论文／讲义组成。下表是逐篇核对入口，详细台账给出真实阅读节段和没有展开的分支。

| 已收集材料 | 提取的工具群 | 正文与详细定位 |
| --- | --- | --- |
| Tucker, 1966, *Some Mathematical Notes on Three-Mode Factor Analysis* | 三模式核心、mode 因子、旋转自由度、展开、共享／独有变异与估计路径 | 张量分解 D01、D04；张量方法来源 |
| Harshman, 1970, PARAFAC | 可分离分量、ALS、松弛、条件唯一性、乘性交互与模型扩展 | 张量分解 D02–D04；张量方法来源 |
| De Lathauwer, De Moor & Vandewalle, 2000, *A Multilinear Singular Value Decomposition* | mode-SVD、全正交核心、mode 秩、截断误差、对称／共轭槽、一阶扰动 | 张量分解 D05、D08；张量方法来源 |
| Oseledets, 2011, *Tensor-Train Decomposition* | 切分秩、TT-SVD、规范化、rounding、基本运算与秩增长、算子迭代 | 张量分解 D09–D11；张量方法来源 |
| Novikov 等, 2015, *Tensorizing Neural Networks* | 多索引映射、TT 矩阵层、收缩顺序、前向与参数导数、存储／计算量 | 张量分解 D12；张量方法来源 |
| de Silva & Lim, 2008, *Tensor Rank and the Ill-Posedness of the Best Low-Rank Approximation Problem* | CP 非闭集、border rank、退化序列、低维轨道、实复域差别与判别式 | 张量分解 D13；张量方法来源 |
| Laue 等, 2018, *Computing Higher Order Derivatives of Matrix and Tensor Expressions* | 槽位类型、Ricci／数组表达、表达 DAG、前反向 AD、高阶求导与 δ 化简 | 张量微积分 C1–C5；CS01 |
| Laue 等, 2020, *A Simple and Efficient Tensor Calculus for Machine Learning* | 显式输出轴代数、通用求导规则、cross-country、收缩重排、精确 Hessian 结构压缩 | 张量微积分 C1–C5；CS02 |
| Chiang, Rush & Barak, *Named Tensor Notation* | records、命名 shape、索引、广播、归约、rename、导数轴分离及模型表达 | 类型与指标；张量微积分 C1–C3；CS03 |
| Kjolstad 等, 2017, *The Tensor Algebra Compiler* | 坐标格式层、迭代图、支持集合、merge lattice、表达提升、assembly／compute 与并行条件 | 张量编译 E3；CS10 |
| Vasilache 等, 2018, *Tensor Comprehensions* | 多面体域／访问／依赖／调度、范围推断、调度树、分块、GPU 映射、memory promotion、调优 | 张量编译 E2、E4、E9；CS11 |
| Dao 等, 2022, *FlashAttention* | 在线 softmax、状态合并、分块重算、完整前反向关系、IO 复杂度、block sparse | 张量编译 E5；CS12 |
| Zhou 等, *Linear Layouts*, arXiv v5／ASPLOS 2026 | F₂ 线性映射、rank／kernel、复合／直和／除法、右逆、布局闭包、SIMD／shuffle／swizzle／gather | 张量编译 E6–E7；CS13 |
| Williams, Waterman & Patterson, 2008 report／2009 CACM, *Roofline* | operational intensity、计算／带宽上界、层级流量、优化 ceiling 与测量 | 张量编译 E4；CS14 |
| Batista, *Weyl Tensor Classification in Four-dimensional Manifolds of All Signatures* | Weyl 在二形式上的作用、Hodge 扇区、各 signature 的实性条件、代数类型、boost-weight 与 spinor 联系 | 几何分类；几何来源 |
| Ferrando, Morales & Sáez, 2001, *Covariant Determination of the Weyl Tensor Geometry* | 谱不变量、幂零／重根判据、主二形式、主零方向、不同 Petrov 类型的几何恢复 | 几何分类；几何来源 |
| Christian Bär, *General Relativity* 讲义 | Lorentz 几何、联络、测地线、曲率、标架、因果与光学结构的必要条件 | 微分几何；几何来源 |

## 四组详细来源台账

- [张量方法](sources-tensor-methods.md)：上述 6 篇的定义、算法、定理／性质定位；补充 CP 唯一性、HOOI、ST-HOSVD、HT、网络收缩、矩方法、ICA、TT-cross、非负与缺失观测分解等原领域来源。
- [微积分与系统](sources-calculus-and-systems.md)：上述 8 篇逐项映射、补充一手来源及官方实现资料。官方资料覆盖 MLIR、CuTe、Triton、浮点、GEMM、cuSPARSELt、A100 规格、CUTLASS epilogue 与 isl 调度树。
- [几何](sources-geometry.md)：上述 3 篇的具体定位；补充张量代数、表示论、流形优化、连续介质、Hodge 与几何计算来源。
- [微分流形与丛](sources-manifolds.md)：图册、常秩／正则值、约束子流形、丛、李群和商空间的原文定位；配套的[流形建模与计算](manifold-modeling-and-computation.md)及[全局与辛几何](global-and-symplectic-geometry.md)在文末记录各自的一手来源、版本与实际阅读位置。

由这些入口区分“已经写入可调用条目”“只有定理／算法入口”“需要针对当前问题继续读原文”。工具正文里的公式、条件和反例优先于论文标题中的概括。

## 算子、数值与优化的新增来源

下列条目补足原领域常见的可解性、稳定性、求解与误差工具，并与上述张量方法衔接。

| 来源 | 实际核对位置与依据 | 已入库工具 |
| --- | --- | --- |
| [MIT 18.102, Rodriguez, 2021](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/) | 课程目录；Lecture 2 的有界／连续与算子空间，Lecture 17 的投影与 Riesz，Lecture 22 的 Fredholm alternative 与紧自伴谱定理 | O1、O3、O4、O8。目录中的其余 Banach 定理只作为准确条件下的标准工具入口，未声明全课逐页重读 |
| [Melrose, Functional Analysis 讲义](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/3d4cc88026d44a01f936cd6a0aa995cb_MIT18_102s20_lec_FA.pdf) | Ch.1 §11，式 (1.108)，图范数 | O1：区别定义域的图范数与原空间范数；O3 无界伴随的域按定义及 Riesz 表示另行构造 |
| [Süli, Finite Element Methods, 2025](https://people.maths.ox.ac.uk/suli/fem.pdf) | §1.1–1.2，Theorem 1 的连续／coercive 条件；§2.4 的 Céa；§4.1 的对偶问题；§5.3 的能量稳定式 | O11–O13、A4：弱式、Sobolev、Galerkin、目标残差、连续与离散稳定性 |
| [Trefethen, 1997, Pseudospectra of Linear Operators](https://people.maths.ox.ac.uk/trefethen/publication/PDF/1997_72.pdf) | §1–2 的非正规性、预解式与伪谱；算子例的定义域／边界含义 | O5–O6：谱不足以决定瞬态；正文 $2\times2$ 例为独立推导 |
| [Schölkopf, Herbrich & Smola, 2001, Representer Theorem](https://www.cise.ufl.edu/~arunava/Teaching/cap6617sp21/Readings/SchHerSmo01.pdf) | §1 核／Hilbert 表示，Theorem 1 的严格递增惩罚与正交分解证明 | O9：RKHS、表示定理、有限样本搜索空间与存在性边界 |
| [Barrett 等, Templates for Linear Systems](https://www.netlib.org/templates/templates.pdf) | 第 2 章 CG／MINRES／GMRES 的适用对象与算法，第 3 章预条件入口，§4.2 的停止准则 | N1、N7、N9、N13：方法选择、真实残差、前／后向误差、计算接口 |
| [Halko, Martinsson & Tropp, 2011](https://tropp.caltech.edu/papers/HMT11-Finding-Structure.pdf) | §3 的 SVD／QR／ID；§4 的 range finder、误差探测与幂迭代；§5 的后处理；Theorem 10.5/10.6 的 Gaussian 范围误差条件 | N2、N10、N12：随机 SVD、超采样、稳定幂迭代、ID／Nyström；最终截断与范围投影分开 |
| [Higham, Sylvester Equation](https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/) | 作者文章的谱分离、Kronecker 与平方根导数 | N4、N6：Sylvester／Lyapunov、直接作用 |
| [Higham, Fréchet Derivative](https://nhigham.com/2020/06/23/what-is-a-frechet-derivative/) | 定义、Gâteaux 区别、条件数；$X^2$ 的 $XE+EX$、乘积规则与指数导数的积分作用 | N6–N7：方向导数与条件数；知识迁移 KT2 的非交换关系 |
| [Al-Mohy & Higham, 2009](https://eprints.maths.manchester.ac.uk/1218/1/alhi09.pdf) | Theorem 2.1 的块矩阵恒等式，§2 的迭代导数关系 | N6：矩阵函数 Fréchet 导数；未移植完整 Padé 参数选择器或其性能比较 |
| [Higham & Al-Mohy, 2010](https://eprints.maths.manchester.ac.uk/1451/1/paper8.pdf) | §2.3，Theorem 2.5，主平方根的谱域条件 | N4：主矩阵函数及平方根分支；不把任选平方根视为主平方根 |
| [Yu, Wang & Samworth, arXiv:1405.0680](https://arxiv.org/pdf/1405.0680) | Theorem 2 的原矩阵外部谱隙、常数 2 与 Frobenius 主角界；Theorem 4 的奇异空间联系 | N8：Davis–Kahan 的具体可用版本；与条件不同的定理版本分开 |
| [Demmel, MA221 Lecture 19](https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Lectures/Lecture_19.html) | symmetric eigenproblem／SVD 扰动与算法部分 | N1、N8：正交／谱结构与求解方法入口 |
| [Meyer 等, Hutch++, 2021](https://arxiv.org/pdf/2010.09649) | Algorithm 1、§2–3、Theorem 1 及非 PSD 情形边界 | N11：低秩主空间加随机残差的迹估计；固定成功概率与带置信参数的界分开 |
| [Martens, 2020](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf) | §5–6 的 Fisher、局部 KL 与几何解释；§9.1–9.2 的 Fisher 作用与 GGN 条件等价、§10 阻尼、§11 empirical Fisher、§12 参数化 | A5–A6：可操作曲率与自然梯度；知识迁移 KT1 的拉回联系；其全部渐近统计证明未展开 |
| [Broyden, 1965](https://www.math.unm.edu/~vageli/courses/Ma576/papers/Broyden_65.pdf) | Jacobian 更新构造与迭代流程 | A7：一般割线 rank-1 更新；本文最小改动式可由约束投影直接验证 |
| [Chen, BFGS/SR1 讲义, 2023](https://pages.cs.wisc.edu/~yudongchen/cs726_sp23/Lecture_22_BFGS_SR1.pdf) | §1 BFGS、正定性与有限记忆；§2 SR1 分母与保护 | A7：对称优化曲率、正定条件与不定更新 |
| [Chen, Trust-Region Methods, 2024](https://pages.cs.wisc.edu/~yudongchen/cs726_sp24/Lecture_24_trust_region.pdf) | §3，Theorem 1，球约束二次问题的全局最优条件 | A10：正定度量版本由变量换元得到；移位矩阵可奇异 |
| [Boyd–Vandenberghe, Duality 讲义](https://web.stanford.edu/class/ee364a/lectures/duality.pdf) | Lagrangian／对偶下界、Slater、KKT、敏感度 | A8：全局最优证书与局部驻点的区别 |
| [Parikh & Boyd, Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf) | §2.3 平均／坚定非扩张；§3.2 次微分预解式；§4.2、4.4 迭代；§6.5、6.7 分组／谱函数 | O7、A9：prox、投影、分裂、核范数／分组约束 |
| [Chaturantabut & Sorensen, DEIM, 2010](https://epubs.siam.org/doi/10.1137/090766498) | 论文身份与方法摘要已确认；作者演讲 PDF 超时、机构论文下载返回拒绝，未声称精读原定理 | A11：插值算子、再现性和一般斜投影误差式由直接代数推导；更紧常数与原贪心定理须继续核对 |
| [Tu 等, DMD, arXiv:1312.0041](https://arxiv.org/pdf/1312.0041) | §2、Algorithm 2、Theorem 1、Definition 2 与 Koopman 联系 | A12：低维谱求解、非零模态提升、线性一致性；ERA/LIM 的全部等价分支未展开 |

## AI 学习、生成与交互的来源入口

[AI 分析对象](ai-analysis-objects.md)按研究问题组织层、整网、参数／函数、表示、训练、生成、数据分布、图与集合、离散机制、有状态推理及 RL／Agent。它提供对象和工具之间的连接，不声明覆盖每类架构的全部文献。

[学习动力学与概率模型](learning-dynamics-and-probabilistic-models.md)包含 LK01–LK10 的可调用关系，文末记录其实际读取的一手来源：经验 NTK 与梯度流、优化器状态、随机梯度、Markov 核、score SDE 与概率流、flow matching、耦合误差、非光滑机制、Bellman 更新和图消息传递。原始理论、按定义直接推导的关系和实际模型的适用条件分别标明。

知识库主要服务深度学习与人工智能研究，知识来源保持开放。具体问题、尚缺的分析能力、原领域的重要结构、不同理论之间的关系，以及可能产生新问题的类比，都可以驱动文献扩充。新知识无需先有已知 AI 应用；分别保留原领域结论、已建立的目标对应和待验证的迁移假设。

## 概念关系与知识迁移的来源入口

[数学知识地图](mathematical-knowledge-map.md)按数学对象及定义依赖、特例／推广、对偶、结构保持、近似和极限组织现有知识，具体定理与来源继续由各领域正文承担。[知识迁移与研究发现](knowledge-transfer-and-discovery.md)给出有类型的对应、缺失条件的研究改写，以及误差传播、拉回二次型和非交换顺序效应的推导。其末尾 KT1–KT2 连接上述 Martens 与 Higham 来源。

其中交换缺陷界、扰动传播式、有限类别 Fisher／KL 关系及残差块顺序公式是所述条件下的直接推导。由它们提出的压缩、缓存、度量或更新顺序研究方向属于候选迁移，不代表已经验证任务收益或学术新颖性。

## 公式依据与当前读取边界

**按定义直接推导的内容。** FFN 导数／割线、动态锚点、联合加权误差曲率、奇异度量的双侧低秩解、加权伴随／投影、简单非正规矩阵、Schur／Woodbury、有限维伴随残差、DEIM 斜投影、布局的一般核空间解。它们由明确假设下的代数给出；数值检查辅助确认实现，不能替代证明。

**依赖定理的内容。** CP 唯一性、HOSVD／TT／HT 截断、谱扰动、随机方法概率界、Lax–Milgram、Céa、紧自伴谱定理、Weyl 分类、常秩与正则值、自由 proper 商、管状邻域、Sard／Morse、辛约化、硬件 IO 下界等。保留原定理的对象、域、量词、范数、概率、signature 或计算模型；需要更强结论时回到原文。

**实现资料。** 对 API、支持 dtype、稀疏模式、设备峰值、编译选项和布局族，记录当前版本／设备并在实际使用时复核。历史 benchmark 只说明原实验范围，不成为当前设备的预测。

**尚未成为独立工具条目的内容。** 各论文完整实验、所有证明细节，以及补充来源中与当前工具主干不同的专用分支，在详细台账中定位。需要它们时继续读取并补入，不能用总表的一行冒充已经实现所有分支。

## 新论文进入知识库的方式

1. 记录原文、版本、作者资料及实际读到的节／定理／算法；区分摘要、方法正文与实验。
2. 保留原领域的对象、问题与定义，再提取基本运算、算法步骤、结构／不变量、误差／唯一性／稳定性、计算／存储模型；按材料实际内容组织，不强求每篇都具有所有项目。
3. 记录与既有知识的关系：等价、特例、推广、对偶、离散化、近似、冲突或共同假设。同一名字可以有不同条件，同一数学结构可以来自不同领域。
4. 将知识写入所属正文，至少留下明确关系、成立条件和一个可检查的例子或反例；提供概念入口及可用的交叉链接。尚无直接应用的条目仍可按原结构收录，迁移想法另标证据状态。
5. 当提出目标模型的迁移结论时，用独立推导、反例或不同于正文例子的任务检查它；校验公式和数值时保留独立依据。探索中的未知条件作为后续研究内容保留。
6. 更新本表与对应详细台账。只新增参考链接不算完成知识扩充；未读的部分准确注明。新结构超出现有分类时扩展目录，保持已有索引可继续使用。
