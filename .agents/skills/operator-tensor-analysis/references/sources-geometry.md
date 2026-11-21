# 几何与张量代数：原始来源及工具覆盖

本台账对应 [张量代数与对称性](tensor-algebra-and-symmetry.md)、[微分几何](differential-geometry.md)、[几何分类](geometric-classification.md)。工具卡是按统一约定重推的可执行表达；本文件记录原始材料的阅读位置、已覆盖内容及不能据此声称的覆盖范围。

## 内容

- 已收集几何材料的覆盖映射
- G01 Batista：四维任意签名 Weyl 分类
- G02 Ferrando–Morales–Sáez：Weyl 几何的协变重构
- G03 Bär：广义相对论讲义的曲率与 Petrov 工具
- 张量代数、表示论和微分几何的基础来源
- 流形计算与连续介质的领域来源
- 高维与微分分类的领域来源
- 统一约定与已完成的代数核验
- 阅读范围和后续补充规则

## 已收集几何材料的覆盖映射

| 原始材料中的方法 | 对应工具卡 | 覆盖形式与条件 |
| --- | --- | --- |
| Batista：二形式、Hodge、Weyl 的分块作用 | TA04–TA05，GC01–GC02 | 明确作用空间、复化、度量签名及定向 |
| Batista：六种代数类型、不同签名实条件 | GC02–GC03 | 最小多项式、重根、幂零阶；正定/neutral 分开 |
| Batista：零标架和 Weyl 标量 | GC05 | 分量消去与齐次四次式；源文符号需转换 |
| Batista：boost weight、alignment 与签名推广 | GC09 | 四维等价、高维扩展及实零方向限制 |
| Ferrando：实/自对偶二形式、主平面和标架 | TA04–TA05，GC04，GC07B–C | 复双线性配对、简单/零二形式、主方向提取 |
| Ferrando：迹不变量和特征多项式 | GC03 | 特征值、Cardano 的支配对、最小多项式判别 |
| Ferrando：两个二次曲面的相交、Debever/Sachs 条件 | GC05–GC06 | 主二形式的两方程、零方向重数、坐标检验 |
| Ferrando：N、III、D、II、I 的全部协变提取分支 | GC07A | 算子多项式、合法试验向量、归一化与重构残差 |
| Ferrando：canonical frame 和残余自由度 | GC07C | 离散选择、连续 isotropy、相容定向与时间定向 |
| Ferrando：静态性及 hypersurface-orthogonal Killing 判据 | DG08，GC08 | 点态方向提取加微分闭合条件；局部与全局分开 |
| Bär：代数曲率、Ricci/Weyl 分解 | TA07，DG05，GC01 | 代数投影、Bianchi、迹核验 |
| Bär：Hodge、复线性自伴作用、Petrov 标准形 | DG06，GC02–GC03 | 签名、非 Hermitian 性、Jordan/核链 |
| Bär：principal null bivector/vector、Bel–Sachs | GC05–GC06 | 主方向重数表及可直接收缩的等价条件 |
| Bär：optical scalars、screen 空间 | DG14 | 光学矩阵分解；Riccati/Raychaudhuri 从 Jacobi 推出 |
| Bär：几何量的一阶变分 | DG10 | 度量逆、体积、联络、Ricci 和标量曲率变分 |

这里“全部分支”针对 Ferrando §4 的代数提取算法。Bär 是整本讲义，本台账不把“已核对 Petrov 相关体系”扩大为“其全部宇宙学、黑洞解和双曲 PDE 章节均已转成工具卡”。

## G01 Batista：四维任意签名 Weyl 分类

**来源。** Carlos Batista，*Weyl Tensor Classification in Four-dimensional Manifolds of All Signatures*，arXiv:1204.5133，使用 v4（2013-02-07）。

- [版本和摘要](https://arxiv.org/abs/1204.5133)
- [作者论文全文](https://arxiv.org/html/1204.5133v4)

**阅读位置。** §II Bivectors in 4 Dimensions；§III Complex Weyl Tensor；§IV Lorentzian Case, The Petrov Classification；§V Euclidean Signature；§VI (2,2) Signature；§VII Boost Weight Method。核对了式 (1)–(20)、标准形、消失分量表和各签名实条件。

**实际提取。** 六维二形式到两个三维扇区的分解、迹零复对称算子、Jordan 类型、零标架适应、四维 alignment 的对应关系。Ricci/Segre 在 §I 中用于区分不同分类对象；GC13 的核链示例是独立直接计算。

**约定风险。** 文中 Lorentz 零标架满足 \(\ell\cdot n=1,m_1\cdot m_2=-1\)，与本技能采用的 \((-+++)\) 不同；Weyl 对二形式的作用也有整体因子差异。文中的 6/21 类把两个 Hodge 扇区交换视作等价；固定定向时应记录有序对，粗分分别为 9/36。neutral 的实细分类还需保留实根或实自旋方向信息。

## G02 Ferrando–Morales–Sáez：Weyl 几何的协变重构

**来源。** Joan Josep Ferrando、Juan Antonio Morales、Juan Antonio Sáez，*Covariant determination of the Weyl tensor geometry*，Classical and Quantum Gravity **18** (2001)，4939–4959。

- [作者机构全文 PDF](https://www.uv.es/moralesa/Dos-fluids/CQG-CovariantDeterminationWeylTensorGeometry(2001).pdf)

**阅读位置。** §2–§5；Lemmas 1–4；Propositions 1–18；式 (1)–(51)。各型算法在 §4.1–§4.5，静态性/Killing 应用在 §5。

**按段落映射。**

| 原文位置 | 在技能中的可执行内容 |
| --- | --- |
| §2、Lemmas 1–2 | 实二形式与自对偶形式、简单与零形式、主方向、标架恢复：GC04、GC07B–C |
| §3、式 (10)–(16) | \(\operatorname{tr}W^2,\operatorname{tr}W^3\)、特征方程、两个二次型、Sachs 重数：GC03、GC06 |
| §4.1、Propositions 1–2 | N 型 rank-one 形式与 fundamental direction：GC07A–B |
| §4.2、Propositions 3–6 | III 型用 \(W^2\) 及 \(W\) 重构 null/unit forms 和标架：GC07 |
| §4.3、Propositions 7–9 | D 型简单本征二形式、两条重复主零方向及自由度：GC07 |
| §4.4、Propositions 10–12 | II 型两个算子多项式、三条主方向和 canonical frame：GC07 |
| §4.5、Propositions 13–15 | I 型三个特征过滤多项式、Debever 形式、有限标架选择：GC07 |
| §5、Lemmas 3–4、Propositions 16–18 | I 型静态性、III 型特定 hypersurface-orthogonal Killing 候选：DG08、GC08 |

**约定风险。** 论文的 \(a=\operatorname{tr}W^2,b=\operatorname{tr}W^3\) 对应本技能 \(a=2I,b=6J\)，重复根为 \(\rho=-b/a\)。配对是复双线性的，不带复共轭；单位自对偶二形式范数为 \(-1\)。§2、Lemma 1 使用归一化正交基；GC02 中的普通矩阵对称性也以此为前提，任意基中应核验 \(Q^\mathsf T B=BQ\)。I 型的 \(P_i\) 原式是特征线过滤多项式，只有除以本征值差的乘积才是幂等谱投影。上述工具卡显式给出这一归一化。

## G03 Bär：广义相对论讲义的曲率与 Petrov 工具

**来源。** Christian Bär，*General Relativity: A Mathematical Introduction*，Potsdam 作者讲义，版本 2023-03-23。

- [作者官方全文 PDF](https://www.math.uni-potsdam.de/fileadmin/user_upload/Prof-Geometrie/Dokumente/Lehre/Lehrmaterialien/GR.pdf)

**阅读位置。** §5.1 几何量的一阶变分；§6.1（印刷页 163）代数曲率；§6.2（167）Hodge；§6.3（173）复线性端同态；§6.4（175）Petrov；§6.5（183）主零方向；§6.6（189）光学量。重点核对了这些部分的定义、分类条件和相关推导，尤其 Propositions 6.40、6.44 和 Corollary 6.45。

**使用方式。** Bär 的直接多线性 Bel–Sachs 判据为 GC06 提供容易实现的检查；Hodge、代数曲率和 Jordan 体系是 GC01–GC03 的独立交叉核验。DG14 的屏空间与光学张量沿用该体系，演化方程由本技能声明的 Jacobi/曲率约定推导。

**DG14 条件的补充核对。** Gregory J. Galloway，[*Notes on Lorentzian causality*](https://www.math.miami.edu/~galloway/vienna-course-notes.pdf)，2014-08-04，§6、Propositions 6.3–6.4、式 (6.18)–(6.21)，印刷页 27–29。已核读屏丛 Riccati、无扭 Raychaudhuri 不等式及在未来完整生成元上的比较论证；原文维数是 \(n+1\)，故其屏维数 \(n-1\) 对应 DG14 的 \(n-2\)。这里仅补足已有聚焦判据的延伸条件，不调用其后全部因果或奇性定理。将平直时空中的收敛光束限制到焦点之前的开域，即可独立说明负扩张不保证焦点落在当前流形内。

**范围。** 讲义前言明确假定微分几何背景。不能仅引用它来表示已读完外代数、联络或流形优化的教材体系；这些基础分别由下面的作者讲义补足。§5.2–§5.5 的约束方程和 PDE 存在性不是此处已完整抽取的章节。

## 张量代数、表示论和微分几何的基础来源

### G04 Etingof 等：表示论

**来源。** Pavel Etingof、Oleg Golberg、Sebastian Hensel、Tiankai Liu、Alex Schwendner、Dmitry Vaintrob、Elena Yudovina，*Introduction to Representation Theory*，AMS 2011；使用作者公开版本。

- [作者全文 PDF](https://math.mit.edu/~etingof/representationtheorybook.pdf)
- [作者书目与版本入口](https://math.mit.edu/~etingof/books.html)

**阅读位置。** §2.3 表示及 Schur 引理；§2.11–§2.12 张量积/张量代数；§2.14 对偶和 Lie 表示；§3.1 半单分解；§4.1 Maschke；§4.4–§4.9 对偶、张量积、字符、正交关系与重数；§5.12 对称群、§5.17 hook 公式、§5.18–§5.19 Schur–Weyl。读取了相关定义和结论，不把整个教材的所有分类定理列作已实现工具。

**映射。** TA01–TA03、TA06–TA10。复数域、半单性/完全可约性及紧群条件要写在使用处；实域 Schur 交换子代数可能为 \(\mathbb R,\mathbb C,\mathbb H\)，不能直接套复数域的标量结论。紧群 Haar 平均是有限群平均的积分推广，需有归一化不变测度。

### G05 Meinrenken：微分流形

**来源。** Eckhard Meinrenken，*Differentiable Manifolds*，University of Toronto，Fall 2001。

- [作者全文 PDF](https://www.math.toronto.edu/mein/teaching/LectureNotes/Man.pdf)
- [作者讲义目录](https://www.math.toronto.edu/mein/teaching/lectures.html)

**阅读位置。** §1.3–§1.5 光滑映射和切向量；§3.1–§3.2 向量场和流；§4.2 外代数、§4.3 形式、§4.4 外微分、§4.5 拉回与 Cartan 演算、§4.6–§4.9 定向/积分/Stokes；§5 de Rham。映射 TA03–TA04、DG01–DG02、DG08。

**条件。** 外微分与拉回不依赖度量；Hodge 和梯度则依赖度量。局部 Poincaré 引理和全局恰当性由不同条件控制。Stokes 使用 §4.9 Theorem 4.22 的紧支撑版本；DG02 的 \((0,1)\) 反例直接说明一般非紧流形上不能只检查绝对可积性。

### G06 Meinrenken：Riemannian 几何

**来源。** Eckhard Meinrenken，*Riemannian Geometry*，University of Toronto，Spring 2002。

- [作者全文 PDF](https://www.math.toronto.edu/mein/teaching/LectureNotes/rieall.pdf)

**阅读位置。** §3 子流形；§4–§5 切向量和切映射；§8–§9 流与 Lie 括号；§11 Frobenius；§12 度量；§15 联络和平行移动；§16 测地线；§17 完备性；§18 曲率；§19 向量丛联络。映射 DG01、DG03–DG09。

**条件。** 正定 Riemannian 的完备性/距离结论不能不加修改搬到 Lorentz；坐标偏导、沿曲线导数与协变导数按其定义区分。

### G07 Meinrenken：群作用

**来源。** Eckhard Meinrenken，*Group Actions on Manifolds*，University of Toronto，Spring 2003。

- [作者全文 PDF](https://www.math.toronto.edu/mein/teaching/LectureNotes/action.pdf)

**阅读位置。** §1.1–§1.5 的 Lie 群/代数作用、轨道、稳定子、proper action 和 orbit type；Corollaries 1.10、1.27。映射 TA08–TA09。用 Lie 代数方程推出群等变性时需要处理群的连通分支；只验生成元导数会遗漏离散对称。

### G08 Tong：流形、曲率与场论接口

**来源。** David Tong，*General Relativity*，Cambridge Part III，Michaelmas 2019。

- [作者课程入口](https://davidtong.org/teaching/general-relativity/)
- [作者全文 PDF](https://davidtong.org/pdfs/teaching/general-relativity/gr.pdf)

**阅读位置。** §2.1–§2.3 流形、向量、张量与 Lie 导数；§2.4 微分形式；§3.1 度量/Hodge；§3.2 联络、挠率、曲率；§3.3 平行移动、测地线、Jacobi；§3.4.1–§3.4.2 Ricci/Einstein 与标架形式；§4.1 变分、§4.3 Killing、§4.4.1 conformal。这里用相应公式和定义作交叉核对，不声称逐段提取整本相对论课程。

**映射。** DG01–DG10；尤其连接指标表达与 Cartan 结构方程。子流形 Gauss 公式在工具卡中依切/法向分解及统一曲率约定书写，不直接拼接不同书的槽次序。

### G09 Auroux：Hodge 理论的分析条件

**来源。** Denis Auroux，MIT 18.966 *Geometry of Manifolds*，Spring 2007，Lecture 15。

- [MIT 官方课程资源](https://www.ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/resources/lect15/)
- [讲义 PDF](https://www.ocw.mit.edu/courses/18-966-geometry-of-manifolds-spring-2007/d6848bb391c032ef27993e984fef4558_lect15.pdf)

**阅读位置。** 讲义中的 Hodge 星、形式内积、余微分、Hodge Laplacian 及 Hodge decomposition theorem。映射 DG06。使用紧致、定向、无边界、正定的版本；边界条件、非紧谱理论和 Lorentz 双曲问题需要另立定理，不由这张卡保证。

## 流形计算与连续介质的领域来源

### G10 Absil–Mahony–Sepulchre：矩阵流形优化

**来源。** P.-A. Absil、Robert Mahony、Rodolphe Sepulchre，*Optimization Algorithms on Matrix Manifolds*，Princeton University Press，2008。

- [出版社官方完整章节入口](https://press.princeton.edu/absil)
- [第 3 章](https://press.princeton.edu/chapters/absil/Absil_Chap3.pdf)：流形、子流形、切空间、Riemannian gradient
- [第 4 章](https://press.princeton.edu/chapters/absil/Absil_Chap4.pdf)：Definition 4.1.1 retraction，Stiefel 的 QR/polar 例子
- [第 5 章](https://press.princeton.edu/chapters/absil/Absil_Chap5.pdf)：§5.3.3 / Proposition 5.3.2 嵌入联络，§5.5 Hessian 和二阶 retraction
- [第 8 章](https://press.princeton.edu/chapters/absil/Absil_Chap8.pdf)：§8.1 vector transport，式 (8.6)、(8.10) 的两种构造

**映射。** DG09、DG11–DG12。这里提取几何计算接口：切投影、Riemannian grad/Hess、retraction、vector transport、Stiefel/Grassmann；完整优化算法和收敛性由技能的优化工具另行处理。

**条件。** 更换 Euclidean/canonical/quotient 度量会改变梯度、连接和距离；把约束集合写出来还不足以确定优化几何。

**二阶回缩的补充核对。** Boumal 的作者书稿（[流形来源 M04](sources-manifolds.md)）§5.10 式 (5.28)、Propositions 5.44–5.45 给出一般回缩的加速度项及其消失条件；DG11 据此区分“可以建立二阶模型”与“可直接使用 Riemannian Hessian 作为 pullback Hessian”。独立例子 \(M=\mathbb R\)、\(R_x(v)=x+v+av^2\)、\(f(x)=x\) 满足回缩一阶条件，却有 \(D^2(f\circ R_x)(0)=2a\)、\(\operatorname{Hess}f=0\)；一般回缩的二阶模型仍然存在。

### G11 Pennec–Fillard–Ayache：SPD 张量计算

**来源。** Xavier Pennec、Pierre Fillard、Nicholas Ayache，*A Riemannian Framework for Tensor Computing*，International Journal of Computer Vision **66**(1)，41–66，2006；作者稿日期为 2005-02-08。

- [INRIA 作者全文 PDF](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Pennec.IJCV05.pdf)

**阅读位置。** §3.3 不变度量和测地线、§3.4 Exp/Log；§4 均值/插值；§5 扩散；Appendix A 的变分与流形 Laplacian。映射 DG13。

**条件。** 论文称为 tensor 的对象明确是 SPD 矩阵，原始领域是 diffusion tensor imaging 等。不能把其 SPD 距离套到任意高阶数组。工具卡把空间场导数和 SPD 目标流形的连接项分别写出。

### G12 López-Pamies：连续介质与弹性

**来源。** Oscar López-Pamies，*Mechanics of Solids: A Continuum Perspective*，Edition 2021，Illinois 作者教材。

- [作者官方全文 PDF](https://pamies.cee.illinois.edu/assets/pdf/Mechanics_of_Solids.pdf)

**阅读位置。** §4.3，式 (4.38)–(4.54)：由储能 Hessian 定义模量、major/minor symmetries、弹性能及其梯度给出的 Hooke 定律、材料对称性、各向同性 Lamé 表达及正定条件；Hessian 与本构关系位于印刷页 57–59（PDF 页序 61–63）。映射 TA11。

**条件。** major symmetry 来自模量为势函数的 Hessian：原文式 (4.38) 定义 \(L=D^2W(I)\)，式 (4.45) 将应力定义为应变能梯度。仅写一个标量二次式 \(E(\varepsilon)=\tfrac12\langle\varepsilon,C\varepsilon\rangle\)，只能确定 \((C+C^\dagger)/2\)，不能由此推出原本构算子 \(C\) 自伴。正定性作用于对称应变空间；由于 minor symmetries，反对称子空间被消掉，不能说四阶算子在全部 \(V^*\otimes V^*\) 上严格正定。工具卡使用 \(\operatorname{Sym}^2\) 的体积/偏量投影给出谱判据。

### G13 Helnwein：压缩六维表示的来源状态

**来源。** Peter Helnwein，*Some Remarks on the Compressed Matrix Representation of Symmetric Second-Order and Fourth-Order Tensors*，Computer Methods in Applied Mechanics and Engineering **190**(22–23)，2753–2770，2001；DOI 10.1016/S0045-7825(00)00263-2。

- [作者机构存档与摘要](https://repositum.tuwien.at/handle/20.500.12708/174886)
- [原出版社记录](https://www.sciencedirect.com/science/article/pii/S0045782500002632)

**实际阅读范围。** 已核对机构元数据和摘要，未取得全文。该来源确认压缩表示应理解为六维空间中的坐标/基与度量问题；不把未读的正文公式当成已核验依据。TA11 的 Mandel/Kelvin 权重、Voigt 转换和剪切反例由所列正交基直接推导，并作独立代数验算。

## 高维与微分分类的领域来源

### G14 Coley–Milson–Pravda–Pravdová：高维 Weyl alignment

**来源。** A. Coley、R. Milson、V. Pravda、A. Pravdová，*Classification of the Weyl Tensor in Higher Dimensions*，Classical and Quantum Gravity **21**，L35–L42，2004；arXiv:gr-qc/0401008。

- [原论文版本入口](https://arxiv.org/abs/gr-qc/0401008)
- [原论文全文](https://arxiv.org/html/gr-qc/0401008v2)

**阅读位置。** 零标架变换和 boost 权重的式 (1)–(4)，Weyl 分层与式 (9)，Principal Classification 和 Secondary Classification。映射 GC09。

**条件。** 高维 WAND 方程可能过定而无实解，因此存在 G 型。bivector 算子的 Jordan 分类和 alignment 分类在高维不自动等价；需要声明所采用的分类定义。

### G15 Karlhede 与可公开核读的算法论文

**历史原始论文。** Anders Karlhede，*A review of the geometrical equivalence of metrics in general relativity*，General Relativity and Gravitation **12**，693–707，1980。

- [原出版社记录](https://link.springer.com/article/10.1007/BF00771861)

**实际阅读范围。** 原论文取得元数据/摘要，未取得全文；只用来注明算法出处，不声称逐步依据其未读正文实现。

**已核读的作者论文。** Thiago M. Mergulhão、Carlos Batista，*On the Equivalence of Spacetimes, the Cartan-Karlhede Algorithm*，arXiv:2007.04123，使用 v2（2020-07-31）。

- [版本入口](https://arxiv.org/abs/2007.04123)
- [作者论文全文 PDF](https://arxiv.org/pdf/2007.04123)

**阅读位置。** §II 的算法；§III 的函数依赖和 isotropy 例子；§IV 的 Schwarzschild 标架固定例；导言的标量不变量不足反例。映射 GC10。该文是作者对算法的技术阐释和实例推导，不替代 Karlhede 原文全部证明。

**条件。** 先排除任意标架带来的伪函数依赖；追踪整个剩余群，包括离散部分；局部正则性和可继续微分的条件不能省略。

## 统一约定与已完成的代数核验

本技能的几何参考统一使用以下接口；导入其他论文时按对象转换，而不是只修改符号名称。

| 位置 | 统一选择 | 已做的针对性检查 |
| --- | --- | --- |
| 曲率 | \(R(X,Y)=[\nabla_X,\nabla_Y]-\nabla_{[X,Y]}\)；槽顺序见 DG 文件开头 | 代数曲率投影的反对称、成对对称、Bianchi 与幂等性 |
| Weyl 二形式作用 | \((\mathcal C\Omega)_{ab}=\tfrac12C_{ab}{}^{cd}\Omega_{cd}\) | Weyl 去迹及三维矩阵无迹、复对称 |
| Lorentz 自对偶配对 | 复双线性，单位范数 \(-1\) | 非 Hermitian 的 N/III/II 具体矩阵例子 |
| Hodge | \(*^2=(-1)^{k(n-k)+q}\) | 四维正定、Lorentz、neutral 的二形式显式计算 |
| 电磁分解 | 指定基下 \(Q=-E+iB\) | 从实四槽 Weyl 的收缩独立得到同一 \(Q\) |
| Petrov 不变量 | \(I=\tfrac12\operatorname{tr}Q^2,J=\tfrac16\operatorname{tr}Q^3\) | D/II 同迹不变量而不同最小多项式；III/N/O 逐级幂零 |
| Ferrando 重构 | 含度量的 \(u\otimes v:z\mapsto u\langle v,z\rangle\) | N、III、D、II、I 五种分支对精确矩阵全部重构成立 |
| 弹性六维坐标 | Mandel 剪切乘 \(\sqrt2\)，工程 Voigt 应变乘 2 | 内积保持、\(C^M=DC^VD\)、各向同性谱的 1+5 重数 |

这些核验排除了具体符号/因子风险，不构成所有输入、所有流形和所有数值误差条件下的自动证明。工具卡中的前提与残差判据仍需针对当前任务检查。

## 阅读范围和后续补充规则

当前覆盖强调可调用的方法：对象和作用空间、恒等式、算法分支、判据、反例、边界条件。原论文里的数值实验设置、完整物理解或教材的全部练习不因此自动成为本技能内容。

新增材料时，对每个有实质用途的数学方法记录“输入对象与类型 → 前提 → 公式/构造 → 可计算验收 → 适用边界”，并将它落到工具卡；单独把论文名或方法名加进书目不计作覆盖。若某新论文需要未收录的表示论、PDE、拓扑、统计或数值工具，应明确增补相应体系，而不是把张量记号本身当成这些工具已经存在。

只有摘要可读的条目保留为出处线索；用于精确计算的公式必须来自已核读的技术正文或在卡内给出可独立检查的推导。本文对 Helnwein 和 Karlhede 原文的阅读限制按此记录。
