# 张量方法：原始来源、调用位置与覆盖边界

本清单用于按问题返回原文，不把“在参考文献中出现”当成“全部定理已实现”。D 编号对应 [分解工具](tensor-decompositions.md)，N 编号对应 [网络与统计工具](tensor-networks-and-statistics.md)。正文以统一轴约定重写公式；明确标成自导的式子由给定假设推出，不冒充原文编号公式。

## 目录

- [既有六篇论文的工具覆盖](#既有六篇论文的工具覆盖)
- [T01 Tucker 1966](#t01-tucker-1966)
- [T02 Harshman 1970](#t02-harshman-1970)
- [T03 De Lathauwer 等 2000：HOSVD](#t03-de-lathauwer-等-2000hosvd)
- [T04 Oseledets 2011](#t04-oseledets-2011)
- [T05 Novikov 等 2015](#t05-novikov-等-2015)
- [T06 de Silva 与 Lim 2008](#t06-de-silva-与-lim-2008)
- [T07 Kruskal 1977](#t07-kruskal-1977)
- [T08 De Lathauwer 等 2000：最佳近似](#t08-de-lathauwer-等-2000最佳近似)
- [T09 Vannieuwenhoven 等 2012](#t09-vannieuwenhoven-等-2012)
- [T10 Grasedyck 2010](#t10-grasedyck-2010)
- [T11 Markov 与 Shi 2008](#t11-markov-与-shi-2008)
- [T12 Anandkumar 等 2014](#t12-anandkumar-等-2014)
- [T13 Comon 1994](#t13-comon-1994)
- [T14 Gray 与 Kourtis 2021](#t14-gray-与-kourtis-2021)
- [T15 Sidiropoulos 与 Bro 2000](#t15-sidiropoulos-与-bro-2000)
- [T16 Oseledets 与 Tyrtyshnikov 2010：TT-cross](#t16-oseledets-与-tyrtyshnikov-2010tt-cross)
- [T17 Kim 与 Park 2012：非负 CP](#t17-kim-与-park-2012非负-cp)
- [T18 Acar 等 2011：缺失观测 CP](#t18-acar-等-2011缺失观测-cp)
- [公式复核和继续扩充入口](#公式复核和继续扩充入口)

## 既有六篇论文的工具覆盖

| 论文 | 本技能已经展开的可调用方法族 | 使用边界 |
| --- | --- | --- |
| Tucker 1966 | 组合索引/Kronecker、三模核心、旋转自由度、mode Gram、Methods I–III、共同度/独有方差、载荷与核心恢复、因子得分不定性 | 共同度沿用普通因子分析的估计问题；原文未给出三模专用全局收敛算法 |
| Harshman 1970 | 比例轮廓与共享条件、system/object variation 区分、CP、ALS/MTTKRP、松弛、全列秩唯一性、非线性交互 | 原文实证样例不是一般唯一性证明；新交互不能直接继承线性 ALS |
| HOSVD 2000 | mode SVD、核心正交/能量、秩与谱、截断误差、重根自由度、对称/共轭结构、HOEVD 区分、一阶扰动 | 普通 HOSVD 核心不必超对角；截断不保证全局最优 |
| TT 2011 | 切分秩、TT-SVD、准最优、规范化、rounding、CP→TT 思路、基本运算、Kronecker 和/二体交互秩界、压缩幂迭代 | 显式初始分解、已有 TT 运算与增秩后 rounding 成本分开 |
| Tensorizing NN 2015 | 行列轴双射、TT 矩阵、参数量、前向、输入与核心梯度、左右环境复用 | 固定低 TT 秩限制表达类；原文压缩实验不代表当前硬件加速 |
| de Silva–Lim 2008 | CP/mode/边界秩、闭性、发散反例、weak solution、判别式、实/复域差别、2×2×2 轨道判据、半代数判定入口 | 一般张量分类与高效 CP 秩求解不能由低维判别式直接推广 |

“覆盖”指这些方法族有对象、公式/算法及条件；不表示整篇论文的每一引理、历史讨论、实验表和全部证明都已复制进技能。遇到需要精确定理常数、非标准域或特殊退化分支的问题，应按下面定位继续读原文。

## T01 Tucker 1966

L. R. Tucker, *Some Mathematical Notes on Three-Mode Factor Analysis*, Psychometrika 31, 279–311 (1966).

- [原文 PDF](https://three-mode.leidenuniv.nl/pdf/t/tucker1966pmet.pdf)
- [DOI](https://doi.org/10.1007/BF02289464)
- 调用：D01、D04。
- 核对范围：原文开头的 mode/组合变量记号；Methods I、II、III，尤其式 (46)–(71)；“Model and Analysis with Unique Variance for Combination Variables”的式 (72)–(82) 及 pp. 307–311 的载荷、核心、得分不定性与斜变换讨论。直接 PDF 部分请求超时后，通过可读全文补齐了这一章节。
- 已提取：mode 独立性、核心耦合、Gram 特征子空间、先压缩小 mode 与组合变量协方差路径；D04 以统一矩阵式重推 Method III，并检查 \(A^TA=I\)。共同方差部分补入对角替换、偏迹、Kronecker 投影、核心恢复及因子得分不能从观测唯一恢复的条件。
- 估计边界：式 (81) 将共同度估计归结为普通因子分析；原文未给出一个独立的三模专用共同度迭代收敛定理。不同共同度估计器的统计性质应另行核对，不能把原文未提供的保证补写成既有结论。

## T02 Harshman 1970

R. A. Harshman, *Foundations of the PARAFAC Procedure: Models and Conditions for an “Explanatory” Multimodal Factor Analysis*, UCLA Working Papers in Phonetics 16, 1–84 (1970).

- [作者原文 PDF](https://www.psychology.uwo.ca/faculty/harshman/wpppfac0.pdf)
- 调用：D02、D03。
- 核对范围：§III 三模 PP 模型与 system/object variation（约 pp. 18–26）；§IV 的最小二乘快速算法与松弛（pp. 31–33，式 (10)–(13)）；§V Jennrich 唯一性（pp. 61–62）；§VI 的乘性交互/非线性推广（pp. 63–68 的核心模型段）。没有重做整篇语音数据与所有合成实验。
- 已提取：\(X_{ijk}=\sum_rA_{ir}B_{jr}C_{kr}\) 的共享关系、逐块回归、因子归一化与尺度自由度、全列秩唯一性、松弛的目标验收、乘性交互对应受约束的新外积项。
- 阅读定位：原文 pp. 31–32 已把 Gram 的逐项乘积与数据—双因子收缩写出；D02 把它整理成现代 MTTKRP/Khatri–Rao 形式，不把该术语追溯为原文原词。
- 复核补充：原文 p. 33、式 (13) 取松弛因子在 1 与 2 之间。D02 由固定其他块的二次最小二乘目标独立推出 \(-\omega(2-\omega)\|\Delta Z^T\|_F^2/2\) 的目标差；因此不能把所有 \(\omega>1\) 都归为无下降保证。该式限于精确块解，不是整体加速定理。复数 CP 正规方程亦按无共轭外积与 Hermitian 残差范数独立推导。
- 边界：解释性需要生成机制支持；算法停止或多次重启得到同一结果不替代唯一性定理。

## T03 De Lathauwer 等 2000：HOSVD

L. De Lathauwer, B. De Moor, J. Vandewalle, *A Multilinear Singular Value Decomposition*, SIAM J. Matrix Anal. Appl. 21(4), 1253–1278 (2000).

- [原论文 PDF](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf)
- [DOI](https://doi.org/10.1137/S0895479896305696)
- 调用：D01、D05、D08；N04 的复数累积量槽结构入口。
- 核对范围：§3–4 的 HOSVD 定义、Property 4 的重根自由度、Properties 6–12 中的秩/结构/能量/截断/SVD 联系；§5 对称性、Theorems 3–4 与 Example 7；§6 的非零孤立谱、重根优选基和 Theorem 7 一阶扰动。
- 已提取：完整正交分解、all-orthogonality、mode 奇异谱与秩、Frobenius 能量守恒、截断尾部平方和界、对称与共轭基协调。
- 条件定位：Property 10 的截断误差公式不等于 Eckart–Young 的最优性；Theorem 4 的 HOEVD 谱定义也不等于 Z-eigenvalue/正交 CP。
- D08 的 Gram 微分形式是对实数简单谱的独立重推；复数、重根分裂和高阶导数需要按原文 §6 重新选择规范，不能把该简式无条件外推。

## T04 Oseledets 2011

I. V. Oseledets, *Tensor-Train Decomposition*, SIAM J. Scientific Computing 33(5), 2295–2317 (2011).

- [原论文 PDF](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf)
- [DOI](https://doi.org/10.1137/090752286)
- 调用：D09–D11。
- 核对范围：§2 Theorems 2.1–2.2、Corollary 2.4 与 Algorithm 1；§3 正交环境/QR、rounding 和 CP 转换的算法段；§3.1 Laplace/Scholes 型秩结构；§4 加法、标量乘、Hadamard、内积/范数、TT 矩阵乘；§5 压缩特征问题的应用入口。未独立复现实验计时。
- 已提取：指定次序的切分秩、TT-SVD、全局截断预算、最优有界秩解存在性、规范形与小矩阵 rounding、直接基本运算及秩增长。
- 成本定位：rounding 的 \(O(dnr^3)\) 指已有 TT、截断前秩上界；直接 TT 矩阵乘 TT 向量的 \(O(dn^2r^4)\) 之后还可能需要昂贵 rounding。
- 结构迁移：D09 二体交互式的系数块秩上界由原文切分论证细化而来；D10 跨迭代误差传播是独立线性递推推导，未声称来自原文某个平方和定理。

## T05 Novikov 等 2015

A. Novikov, D. Podoprikhin, A. Osokin, D. Vetrov, *Tensorizing Neural Networks*, NeurIPS (2015).

- [会议原文 PDF](https://proceedings.neurips.cc/paper_files/paper/2015/file/6855456e2fe46a9d49d3d3af4f57443d-Paper.pdf)
- 调用：D12。
- 核对范围：§3.1 行列多索引映射，§4 TT 层，§5 反向传播、部分核积与核心梯度，Table 1 的特定实现复杂度。未把 §6 的实验压缩倍数写为一般性能保证。
- 已提取：配对行列轴、TT 矩阵参数量、稠密输入前向、\(W^Tg\)、直接核心梯度、左右环境复用。
- 条件定位：原文式 (3)、(5)、(7)–(12)；D12 采用显式四轴核心 \(G_k[a,i_k,j_k,b]\) 重写链式法则。
- 边界：不能用 TT 向量输入的运算复杂度替代 dense-input 层成本；低 TT 秩允许满矩阵秩，但不表示任意稠密矩阵都可由固定小 TT 秩精确表达。

## T06 de Silva 与 Lim 2008

V. de Silva, L.-H. Lim, *Tensor Rank and the Ill-Posedness of the Best Low-Rank Approximation Problem*, SIAM J. Matrix Anal. Appl. 30(3), 1084–1127 (2008).

- [作者 arXiv 版本与全文入口](https://arxiv.org/abs/math/0607647)
- [可检索原文 HTML](https://arxiv.org/html/math/0607647v2)
- 调用：D13，并为 D04/D09 的秩区分提供背景。
- 核对范围：§4.1 闭性/存在性，§4.2–4.3 rank-3 极限与发散；§5.3 判别式、§5.4 病态、§5.5 weak solution 入口；§6 半代数秩分层；§7 Table 1、§7.2 子式判定、§7.4 标量域依赖、§7.5 低维嵌入不变量入口；§8 的 Theorems 52、55 陈述已核读并将 Frobenius 下的正体积失败条件加入 D13，未逐条重证其完整证明。
- 已提取：rank-2 序列趋近 rank-3 的显式反例；CP 秩与边界秩；有界多线性秩的闭集；判别式的 GL 相对不变量性质；八个实 \(2\times2\times2\) GL 轨道的 mode-rank/符号判据。
- 定位提醒：HTML 的定理编号可能与期刊排版编号不同；按章节标题及式 (26)–(27)、Table 1 定位。
- 边界：D13 的完整判定表限于实 \(2\times2\times2\)；一般实张量的半代数表示提供存在性/符号工具入口，不等于已实现的多项式时间分类算法。

## T07 Kruskal 1977

J. B. Kruskal, *Three-Way Arrays: Rank and Uniqueness of Trilinear Decompositions, with Application to Arithmetic Complexity and Statistics*, Linear Algebra and its Applications 18, 95–138 (1977).

- [含 1984 更正的原文扫描 PDF](https://three-mode.leidenuniv.nl/pdf/k/kruskal19771984laa.pdf)
- [DOI](https://doi.org/10.1016/0024-3795(77)90069-6)
- [原论文文本副本](https://www.academia.edu/5013906/Three_way_arrays_Rank_and_uniqueness_of_tri_linear_decompositions)
- 调用：D03。
- 核对范围：原文摘要及三线性等价/唯一性部分，尤其 Theorems 4a–4e 的条件层次；扫描 PDF 不含可检索文字，正文条件另由原论文文本副本与 T15 的 Theorem 1 交叉核对。没有复述全部排列引理证明。
- 已提取：列置换/尺度等价、k-rank 与普通 rank 的差别、三阶唯一性充分条件。
- 边界：精确 k-rank 可能需检查多个列子集；带噪分解的稳定性不能仅凭代数唯一性得出。

## T08 De Lathauwer 等 2000：最佳近似

L. De Lathauwer, B. De Moor, J. Vandewalle, *On the Best Rank-1 and Rank-(R1,R2,…,RN) Approximation of Higher-Order Tensors*, SIAM J. Matrix Anal. Appl. 21(4), 1324–1342 (2000).

- [原论文 PDF](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/delathauwer-2000-rank-approx.pdf)
- [DOI](https://doi.org/10.1137/S0895479898346995)
- 调用：D06。
- 核对范围：§3.2–3.4 高阶幂法、双 mode 联合 SVD 更新与局部极值；§4.1–4.2 投影能量目标、Algorithm 4.2 HOOI。
- 已提取：rank-1 最优幅值、逐块归一化方向更新、双模更新、固定多线性秩的投影能量最大化、HOOI 主子空间更新。
- 边界：原文一般问题是非凸优化；HOOI 的块优化和残差下降不能被写成全局最优保证。原文特殊低维多项式求根分支未做通用实现。

## T09 Vannieuwenhoven 等 2012

N. Vannieuwenhoven, R. Vandebril, K. Meerbergen, *A New Truncation Strategy for the Higher-Order Singular Value Decomposition*, SIAM J. Scientific Computing 34(2), A1027–A1052 (2012).

- [作者原文 PDF](https://people.cs.kuleuven.be/~nick.vannieuwenhoven/papers/01-STHOSVD.pdf)
- [DOI](https://doi.org/10.1137/110836067)
- 调用：D07。
- 核对范围：§3 正交投影记号、Theorem 5.1 的误差拆分、§6 ST-HOSVD 定义与 Algorithm 1、处理顺序比较段。
- 已提取：在当前压缩核心上做逐 mode 截断；局部舍弃残差的平方和；顺序对成本和误差的影响。
- 边界：论文的应用计时属于其数据和实现；不将实证“常更好”写成对任意输入误差都更小的定理。

## T10 Grasedyck 2010

L. Grasedyck, *Hierarchical Singular Value Decomposition of Tensors*, SIAM J. Matrix Anal. Appl. 31(4), 2029–2054 (2010).

- [作者机构论文入口](https://www.mis.mpg.de/publications/preprint-repository/article/2009/issue-27)
- [机构原稿 PDF](https://files-www.mis.mpg.de/mpi-typo3/preprints/2009/preprint2009_27.pdf)
- [Göttingen 图书馆保存的修订原稿 PDF](https://webdoc.sub.gwdg.de/ebook/serien/e/MPI_Math_Nat/preprint2009_27.pdf)
- [作者原论文文本副本](https://www.academia.edu/9248954/Hierarchical_Singular_Value_Decomposition_of_Tensors)
- [DOI](https://doi.org/10.1137/090764189)
- 调用：N01、N07。
- 核对范围：机构摘要；§1 张量积函数空间；§3 维度树、嵌套基、存储、截断界与 Algorithms 1–2；§4 互补环境 Gram、正交化与截断；§5.3 树形状与并行性。机构 PDF 的 octet-stream 请求受阻后，已通过 Göttingen 图书馆保存的作者修订原稿（2009-07-07）直接核读截断和构造正文。
- 版本定位：期刊版 Theorem 3.11/Remark 3.12 对应所链接修订原稿 Theorem 17/Remark 18（印刷 pp. 9–10）；Algorithms 1–2 的 root-to-leaves/leaves-to-root 构造见原稿 pp. 12–13。原稿的统一连续编号与期刊按节编号不同，不应混用为同一 PDF 的定位。
- 已提取：HT 层次秩、转移张量、正确嵌套投影、两种构造的差别、小环境 Gram 递推、已有 HT 的截断成本。
- 条件定位：\(\sqrt{2d-2}\) 是直接计数界；\(\sqrt{2d-3}\) 需将根的两个互补切分联合处理。N07 的质量矩阵式由张量积内积独立推导。

## T11 Markov 与 Shi 2008

I. L. Markov, Y. Shi, *Simulating Quantum Computation by Contracting Tensor Networks*, SIAM J. Computing 38(3), 963–981 (2008).

- [作者 arXiv 版本](https://arxiv.org/abs/quant-ph/0511069)
- [原文 PDF，v7](https://arxiv.org/pdf/quant-ph/0511069)
- 调用：N02–N03。
- 核对范围：§3 网络收缩表示，§4 收缩复杂度、线图和 Proposition 4.2；Theorem 1.1/4.6 的有界局部门维电路条件。
- 已提取：收缩顺序对应边消去；\(cc(G)=\operatorname{tw}(L(G))\)；由树分解构造路径。
- 边界：线图树宽等式使用原文定义；任意超边、不等边维或硬件布局需额外建模，不能直接替换为原图树宽。

## T12 Anandkumar 等 2014

A. Anandkumar, R. Ge, D. Hsu, S. M. Kakade, M. Telgarsky, *Tensor Decompositions for Learning Latent Variable Models*, JMLR 15, 2773–2832 (2014).

- [期刊原文 PDF](https://jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf)
- 调用：D03、N05–N06。
- 核对范围：§4 正交分解与驻点性质，§4.3 Condition 4.1、白化和 Theorem 4.3；§5 Algorithm 1、Theorem 5.1 及扰动说明；§6.1 隐式矩访问；附录 D 联合对角化的模型与矩阵切片入口。没有逐项复制 §3 所有 HMM/LDA/混合模型的修正矩，也未逐行核证附录 B 的全部概率证明。
- 已提取：对称条件独立视图的矩结构；白化/反白化；正交张量幂法、多次启动与消去；算法依赖扰动界；样本级方向收缩。
- 条件定位：\(\epsilon\lesssim\lambda_{\min}/R\) 只是完整 Theorem 5.1 条件的一部分，严格高概率主张还须采用其迭代与重启条件。经验矩、白化误差、反白化误差属于同一误差链。
- 复核补充：Theorem 5.1（p. 2796）要求 \(R\) 维空间中的 \(R\) 个正交基方向及完全对称扰动，N06 已明确这一版本。N05 的经验跨视图矩对称化由其总体矩模型直接推出，不是额外的正定性保证。
- 继续读取：若任务是 HMM 或 LDA 识别，先回到 §3 对应生成模型推导具体修正矩，再调用 N05；不能用本卡简单 cross-moment 直接替代。

## T13 Comon 1994

P. Comon, *Independent Component Analysis, a New Concept?*, Signal Processing 36(3), 287–314 (1994).

- [作者原文 PDF](https://www.gipsa-lab.grenoble-inp.fr/~pierre.comon/FichiersPdf/como94-SP.pdf)
- [DOI](https://doi.org/10.1016/0165-1684(94)90029-9)
- 调用：N04。
- 核对范围：§2 独立成分/尺度置换可辨识背景；§3 多线性累积量变换与 Theorem 16；§4.1 的成对处理、Algorithm 18；附录中固定阶对比函数的正交不变/判别条件段。未复现 Edgeworth 展开的完整高阶系数。
- 已提取：累积量张量与线性混合、白化后正交旋转、固定阶累积量对比函数、Jacobi 方向、零累积量条件的失败反例。
- 边界：固定阶至多一个零累积量的条件强于“至多一个 Gaussian 源”；低阶统计独立不等于一般统计独立。
- 复核补充：原文 Theorem 16 与附录 §7.A.13 分别给出对比函数和辨别条件；调用第 \(p\) 阶量仍需该阶矩存在。N04 的 \(K(I,u,u,u)=\mathbb E[\widetilde x(\widetilde x^Tu)^3]-3(Cu)(u^TCu)\) 由实四阶累积量定义直接收缩，数组维数与累积量阶数分别记为 \(n,p\)。

## T14 Gray 与 Kourtis 2021

J. Gray, S. Kourtis, *Hyper-Optimized Tensor Network Contraction*, Quantum 5, 410 (2021).

- [期刊原文与软件入口](https://quantum-journal.org/papers/q-2021-03-15-410/)
- [原文 PDF](https://quantum-journal.org/papers/q-2021-03-15-410/pdf/)
- 调用：N02–N03。
- 核对范围：§2–3 收缩树/成本与路径搜索框架；§4.7.1 切片；§4.7.2 原文区分路径搜索/编译与运行时间的说明。没有重跑全部电路 benchmark。
- 已提取：收缩树整体成本、带预算路径搜索、切片枚举与并行、切片内存收益和重复计算代价、搜索成本的摊销。
- 边界：原文启发式的高质量结果不是任意输入的最优性证书。N03 的局部残差—环境误差界由线性收缩直接推导，不归给该文未提供的普遍近似保证。

## T15 Sidiropoulos 与 Bro 2000

N. D. Sidiropoulos, R. Bro, *On the Uniqueness of Multilinear Decomposition of N-Way Arrays*, Journal of Chemometrics 14, 229–239 (2000).

- [作者机构论文入口](https://researchprofiles.ku.dk/en/publications/on-the-uniqueness-of-multilinear-decomposition-of-n-way-arrays/)
- [原论文全文副本](https://www.academia.edu/165102616/On_the_uniqueness_of_multilinear_decomposition_of_N_way_arrays)
- 调用：D03。
- 核对范围：§2 Definition 1、Theorem 1 的三阶条件、Lemma 1 的 Khatri–Rao k-rank 下界；§3 Theorems 2–3 和重复 mode 的讨论。原稿 PDF 下载受阻，使用原论文全文副本；未逐行重证附录排列引理。
- 已提取：\(k_{A\odot B}\ge\min(k_A+k_B-1,R)\) 的非零列前提，以及不可约 N 阶模型的 \(\sum_k k_{A^{(k)}}\ge2R+N-1\) 充分条件。
- 复核补充：Theorem 3（pp. 234–235）明确要求不可约，即真实 CP 秩为 \(R\)；D03 已将该前提从来源台账补回正文。四阶 \(R=3\)、一个因子含零列而另三个为 \(I_3\) 的反例说明，不能仅凭不等式数值排除零外积造成的不唯一。
- 边界：额外重复 mode 的 k-rank 为 1 时，判据两边同时增加 1，不增加可辨识信息。

## T16 Oseledets 与 Tyrtyshnikov 2010：TT-cross

I. Oseledets, E. Tyrtyshnikov, *TT-cross Approximation for Multidimensional Arrays*, Linear Algebra and its Applications 432(1), 70–88 (2010).

- [期刊与 DOI](https://doi.org/10.1016/j.laa.2009.07.024)
- [作者研究组介绍](https://oseledets.github.io/news/black-box-cross-approximation-for-multidimensional-arrays/)
- [原论文全文副本](https://www.academia.edu/52266103/TT_cross_approximation_for_multidimensional_arrays)
- 调用：D15。
- 核对范围：原论文 §3.1 的矩阵骨架式 (9)、嵌套索引构造与 Theorem 3.1；§3.2 的 QR/maxvol、Algorithm 2、双向 TT 扫描、停止与重压缩；pp. 79–82 的查询／算术成本及误差分析边界。使用原论文全文副本读取，未重跑 §5 积分实验。
- 已提取：oracle 查询模型、交叉插值式、可逆子块与精确切分秩条件、近最大体积选取、批量查询和迭代开销。
- 边界：原文在近似情形明确把一般误差增长分析留待后续；没有给任意初值必定发现交叉的保证。D15 单支撑张量反例为直接构造，用于说明发现条件和秩条件不同。

## T17 Kim 与 Park 2012：非负 CP

J. Kim, H. Park, *Fast Nonnegative Tensor Factorization with an Active-Set-Like Method*, in *High-Performance Scientific Computing: Algorithms and Applications*, 311–326 (2012).

- [作者原稿 PDF](https://faculty.cc.gatech.edu/~hpark/papers/2011_paper_hpscbook_ntf.pdf)
- [DOI](https://doi.org/10.1007/978-1-4471-2437-5_16)
- 调用：D16。
- 核对范围：§3 的 ANLS、式 (3.1)–(3.4) 与唯一块解下的驻点说明；§4 的 KKT (4.2)、活动集合／块主元 (4.3)–(4.6)、避免循环的备用规则、多右端共享计算与 Algorithm 1；§5 的 ridge 形式 (5.1)–(5.2)。未复现实验计时与全部稀疏正则变体。
- 已提取：非负 CP 的凸块子问题、互补条件、活动集求解、全观测 Gram/MTTKRP 复用、约束与正则改变目标。
- 边界：块 NNLS 的全局最优与联合 CP 的全局最优不同。正文含缺失权重的逐行 NNLS 是按统一目标自行推导的扩展，未归为原文已经分析的采样恢复定理。

## T18 Acar 等 2011：缺失观测 CP

E. Acar, D. M. Dunlavy, T. G. Kolda, M. Mørup, *Scalable Tensor Factorizations for Incomplete Data*, Chemometrics and Intelligent Laboratory Systems 106(1), 41–56 (2011).

- [作者论文页](https://www.mathsci.ai/publication/acdukomo11/)
- [作者完整原稿 PDF](https://www.mathsci.ai/publication/CILS-CPWOPT-preprint.pdf)
- [DOI](https://doi.org/10.1016/j.chemolab.2010.08.004)
- 调用：D16。
- 核对范围：§2 的二元支持权重目标；§3.2 的 EM-ALS 对照；§4 的目标／梯度式 (5)–(8)、§4.2 支持集向量化与 Fig. 3；§5.3.1 的生成方式和每片至少一个观测条件。未把所有应用数据及缺失机制实验转录成工具。
- 已提取：缺失与观测零的区分、仅在支持集评估 CP、稀疏残差梯度、无需稠密重建、因子恢复与缺失项预测的不同验收对象。
- 边界：高缺失率成功是实验证据；原稿明确没有一般 CP 补全样本量结论。D16 的正权重、ridge 联合目标和对角观测反例是明确写出的推导／构造，不是假借原文的恢复定理。

## 公式复核和继续扩充入口

每次改变索引约定时，优先重查以下关系，而非机械复用印刷次序：

- 展开的复合索引顺序决定 Kronecker/Khatri–Rao 因子的排列；以逐项定义核对 D01–D02。
- HOSVD、ST-HOSVD、TT-SVD 和 HT 的误差常数来自不同投影/切分集合；局部尾部、原始展开尾部与跨迭代误差不能混用。
- TT/HT 规范化确保环境等距；一般网络的局部残差还受环境放大。
- 同一张量的普通矩阵秩、mode 秩、CP 秩、边界秩与树/链切分秩回答不同问题。
- 正交三阶谱法的常数依赖其模型和算法；一般 CP、HOSVD-HOEVD 和 Petrov 分类各自有不同对象。

D15–D16 已提供 TT 交叉插值及非负／缺失观测 CP。进一步添加约束/耦合分解、带严格采样恢复定理的张量补全、张量流形优化或网络近似时，应各自取得原论文的目标函数、识别/采样条件、算法和误差依据，再加工具卡。不要把“该领域常见”写成所有输入上可用的统一能力。
