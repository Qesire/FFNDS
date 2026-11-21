# 流形基础：来源、核读范围与工具映射

此台账支持 [光滑流形与丛](smooth-manifolds-and-bundles.md) 的 MF01–MF18。记录的是本次实际核读的章节与可调用工具，不将一本书的全部主题视为已提取。2026-11-16 核查作者或授课机构公开版本；页码采用书内页码时明确注明，PDF 页序可能不同。

## 内容

- M01：Meinrenken，微分流形与向量丛。
- M02：Meinrenken，群作用、商与主联络。
- M03：Etingof，李群、李代数与指数。
- M04：Boumal，流形基础及计算商几何。
- M05：Cohen，主丛、关联丛、拉回与管状邻域。
- M06：Gualtieri，局部映射定理与边界。

## M01 Meinrenken：流形、图册、切空间与向量丛

**来源。** Eckhard Meinrenken，*Differentiable Manifolds*，University of Toronto，Fall 2001。

- [作者讲义](https://www.math.toronto.edu/mein/teaching/LectureNotes/Man.pdf)
- 与已有 [几何来源](sources-geometry.md) G05 为同一讲义，本次增加下列已核读范围。

**实际核读。** §1.1–§1.5：图册、光滑映射、浸入/浸没的局部模型、以导子定义切空间及切映射；§2，尤其 Theorems 2.2、2.5 的局部有限覆盖与单位分解；§13.1–§13.3：局部平凡化、截面、向量丛、过渡函数、切丛、法丛与逐纤维代数运算。

**映射。** MF01–MF03、MF07、MF09–MF12、MF18。图与流形定义、边界/角点约定以及过渡函数分别落到具体工具卡；纤维运算继续连接 TA 与 DG，而不复制另一套张量代数。

**核对与限制。** 使用正确类型 \(\varphi:U\subset M\to\mathbb R^n\)。不采纳 §1.2(a) 关于所有高维球面复结构的过强总括；不采纳 §13.3(c) 将 tautological 向量丛总空间称为 Stiefel 流形的表述。MF16 明确区分向量纤维与标架纤维。这些旁支不构成本模块的定理依据。

## M02 Meinrenken：自由 proper 商与主丛联络

**来源。** Eckhard Meinrenken，*Group actions on manifolds*，University of Toronto，Spring 2003。

- [作者讲义](https://www.math.toronto.edu/mein/teaching/LectureNotes/action.pdf)
- 与既有 G07 为同一来源；此处补足商流形和主联络的调用条件。

**实际核读。** §1.1 的光滑作用与齐性空间；§1.2 的无穷小生成元及符号约定；§1.4 proper action 的定义、轨道性质与 Theorem 1.21 的自由 proper 商；§5.1 的垂直丛、联络形式、水平/基本形式和曲率定义。

**映射。** MF13–MF17；连通群与无穷小等变性的接口沿用 TA08–TA09。

**约定。** 原文统一左作用，并对生成向量场使用负时间。本技能 MF16–MF17 改用右主作用 \(p\mapsto pg\)，因此写 \(R_g^*\omega=\operatorname{Ad}_{g^{-1}}\omega\)。转写时同时核对生成元重现性质和曲率符号，不直接拼接左右约定。

**范围。** 这里只提取上述基础接口。整章 Chern–Weil、classifying spaces、谱序列和 localization 不因引用这份讲义而自动成为本模块已实现工具。

## M03 Etingof：李群、李代数与指数

**来源。** Pavel Etingof，*18.745: Lie Groups and Lie Algebras, I*，MIT 作者讲义。

- [作者公开讲义](https://math.mit.edu/~etingof/lnlg.pdf)

**实际核读。** 第一部分 §2.1 的李群定义；§3.1 齐性空间、§3.5 左右平移与伴随作用；§5.3，尤其 Proposition 5.12、Definition 5.13、Theorem 5.16 的一参数子群、指数、局部可逆性及共轭相容性；§6.1 的群交换子与李括号；§7.3 的向量场李代数。

**映射。** MF13、MF14。矩阵 \(SO(n)\) 指数保正交性由反对称生成元直接验算；Lie 指数与 Riemannian 指数的区分衔接 DG04。

**范围。** 半单李代数分类、根系、最高权、BCH 的高阶展开等，未在这张基础卡中逐项提取。已有 TA 的表示论来源仍独立使用，不能以此替代具体表示的条件。

## M04 Boumal：抽象流形、约束正则性与商计算

**来源。** Nicolas Boumal，*An Introduction to Optimization on Smooth Manifolds*，Cambridge University Press，2023；使用作者公开 PDF，其首页标明截至 2026-10-08 的修订注释，章节/定理编号与出版版一致。

- [作者书稿](https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf)
- [作者图书页](https://www.nicolasboumal.net/book/)

**实际核读。** §3.10 对约束表示与交叉集合的讨论；§5.10 式 (5.28)、Propositions 5.44–5.45 的回缩二阶项与 Hessian 一致性；§5.12 的 metric projection，重点 Theorem 5.53；§8.1–§8.6 的图册、拓扑、切向量、微分、切丛；§8.14 的嵌入子流形，重点 Definitions 8.72–8.73、Theorem 8.75、Corollary 8.76、Proposition 8.77 和 Lemma 8.78；第 9 章中 §9.2–§9.3 群作用商与商映射、§9.4–§9.7 垂直/水平、提升和商度量。

**映射。** MF01–MF03、MF05–MF08、MF14–MF15、MF18；§5.10 亦用于 DG11 的二阶回缩条件。

**关键条件。** Proposition 8.77 要求等值集的一个开邻域内恒秩；只沿等值集恒秩不够。水平空间在书中按度量正交补定义，MF15 同时说明一般补空间是额外选择；只有满足代表元相容性的内积才能定义商度量。

**独立推导。** MF06、MF15 的加权投影矩阵分别由约束最小二乘和正交分解推导。例子的有限维验算不等于整本书中的优化算法、收敛界或全局几何已全部实现。

## M05 Cohen：主丛、拉回、标架与管状邻域

**来源。** Ralph L. Cohen，*Bundles, Homotopy, and Manifolds*，作者课件；公开于 Stanford Math 215B（Winter 2020）课程页。

- [作者课程页](https://math.stanford.edu/~ralph/math215b/)
- [课程所链接书稿](https://math.stanford.edu/~ralph/math215b/book.pdf)

**实际核读。** §2.1 的自由群作用、主丛、关联向量丛和 frame bundle，尤其 Theorem 2.5、Corollary 2.6 对向量丛与标架丛的关系；§2.2 “Pull Backs and Bundle Algebra”，包括 §2.2.1 与 Proposition 2.8 的纤维积构造及随后给出的纤维代数运算；§8.1，尤其 Theorems 8.1–8.3 的管状邻域及欧氏法丛映射 \((p,n)\mapsto p+n\) 的证明。

**映射。** MF03、MF11–MF12、MF16、MF18。

**约定与范围。** 课件中部分丛构造在拓扑范畴陈述；工具卡在光滑底流形、光滑过渡函数和光滑映射下使用其光滑版本。管状邻域卡采用闭嵌入、无边界的常用充分条件，并把可变小邻域与全局定义域区分开。书中的同伦群、谱序列、特征类和配边不在本次基础抽取范围。

## M06 Gualtieri：局部映射定理、正则值与边界

**来源。** Marco Gualtieri，*Topology I: smooth manifolds*，University of Toronto，MAT1300F，2017 作者课程讲义。

- [课程与讲义索引](https://www.math.toronto.edu/mgualt/courses/17-1300/)
- [Part 5：Local structure of smooth maps](https://www.math.toronto.edu/mgualt/courses/17-1300/docs/17-1300-notes-5.pdf)
- [Part 11：Partitions of unity and Whitney embedding](https://www.math.toronto.edu/mgualt/courses/17-1300/docs/17-1300-notes-11.pdf)

**实际核读。** Part 5 §2.5：Theorem 2.15 逆函数定理、Theorem 2.16 常秩定理及其构造证明、Propositions 2.18/2.20 的等值集结论、Definition 2.22 的 immersion/submersion/embedding；§2.6，尤其 Proposition 2.28 的边界正则值条件。Part 11 §3.6：局部有限覆盖、单位分解以及紧致源上单射浸入成为嵌入的论证。

**映射。** MF04–MF07、MF09–MF10。隐函数导数、活跃约束角点模型和坏约束对照由这些局部模型直接构造。

**范围。** Part 5 的完整常秩证明是 MF05 局部标准形的直接依据。单独的手写 IFT/CRT PDF 只确认了可用入口，未据其无法抽取的正文新增结论；使用上面可核读的排版讲义。Part 11 的全局嵌入维数界未扩写成新的工具卡。

## 可重算的数学检查

以下检查验证工具卡内的实现型公式与关键区分；它们不替代一般定理的证明。

- 对可逆坐标 Jacobian 同时变换切/余切系数，验证配对值不变。
- 对正定 \(G\) 和满行秩 \(J\)，验证 MF06 的 \(JP_T=0\)、幂等性和 \(G\)-自伴随。
- 对独立垂直生成元 \(K\)，验证 MF15 的 \(P_{\mathcal H}K=0\)、\(K^{\mathsf T}GP_{\mathcal H}=0\) 和幂等性。
- 对非交换矩阵变标架，核对 \(e'=eS\)、\(s'=S^{-1}s\)、\(A'=S^{-1}AS+S^{-1}dS\) 保持协变导数一致。
- 对 \(y^2,xy,x^2y^2\) 的约束梯度及尖点参数导数直接求导；以局部拓扑/嵌入条件补充秩检验。
- 对反对称矩阵指数核对正交性；圆上平坦线丛的 holonomy 直接由平行移动 ODE 求得。
