# DISCOVERY · 用联合输入草图区分“矩表示有用”与“矩信息可得”

**一个小的输入 token 草图可以保留 W1 投影方差中的带符号协方差项，同时避免逐 token 求全部 W1。** 这是从[新四 panel 均值结果](../../../20261026/E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.md)产生的候选信息机制：投影后边际矩有解释力，而输入对角方差只保留约53–57%的真实投影总方差。竞争解释是：可预测均值只来自特定采样面板；即使联合方差取得准确，非高斯尾部、原Q2及逐token创新仍决定不可替代功能。最便宜的判别是原四个保存面板上的固定草图和激活公式对照，先区分信息取得与非线性闭合；本文仅推导，没有执行这个新数值范围。

## EVIDENCE · 先写精确对象，再引入两种独立近似

令 Q 为某个固定 token 集的实际 post-Q1 矩阵，大小 M×D；其行均值 μ，中心化矩阵 A=Q−1μᵀ。第j个原W1行向量为 w_j，b1仍在GELU内部。忽略原生矩乘舍入时，

\[
m_j=w_j^T\mu+b_{1,j},\quad
v_j=\frac1M\|Aw_j\|^2
=w_j^T\left(\frac{A^TA}{M}\right)w_j.
\]

这不要求 Q 高斯，也没有输入独立性假设。用 `diag(AᵀA)` 替代全协方差会删除 `2∑(i<k)w_ji w_jk Cov(q_i,q_k)`。已有四panel的方差缺口与此机制相容，BF16原生W1舍入的独立方差界仍缺失。

**近似一：取得联合投影方差。** 在观察本次标签/输出之前固定 s 列独立Rademacher token探针 R∈{−1,+1}^{M×s}，令

\[
U=\frac{A^TR}{\sqrt M},\qquad
\widehat v_j=\frac1s\|w_j^TU\|^2.
\]

由 E[RRᵀ]=sI 可直接得 `E_R[v̂_j|Q,W1]=v_j`。对于 a=Aw_j/√M，独立探针给出

\[
\operatorname{Var}_R(\widehat v_j)
=\frac2s\left(\|a\|^4-\sum_m a_m^4\right)
\le\frac2s v_j^2.
\]

这是条件于固定数据/权重的随机方差估计，不是未来token的生成预测。共享探针使不同neuron的方差估计误差相关；W2投影后不应按独立误差求和。s=16仅给出最坏约35.4%的单neuron相对标准差上界，并不保证足够精确。为了同时覆盖13,824个坐标或自适应挑选坐标，还需要相应集中界/选择控制。

**近似二：从矩闭合 GELU 均值。** 对 erf-GELU，若该neuron在当前token集的分布可用 N(m_j,v_j)替代，已有精确公式

\[
g(m,v)=m\Phi\!\left(\frac m{\sqrt{1+v}}\right)
+\frac v{\sqrt{1+v}}\varphi\!\left(\frac m{\sqrt{1+v}}\right).
\]

用 g(m,v̂) 既有草图误差，也有分布闭合误差。无偏方差不保证非线性 g 的无偏性。实际 tanh-GELU、FP32 Smooth2除法/BF16 cast、原HiF4 Q2必须继续分别说明；不能将 g/smooth 称为原post-Q2均值。

连续系数均值的 signed W2 readout是一个有意义的功能坐标，但不是完整tokenwise FFN。即使精确取得全部neuron均值，令两个token输出偏差为(+u,−u)就能让均值完全正确而每个token均错误。已有低运动和边缘变化也提醒：均值、空间创新、运动相位及条件遵循应各自验证。

## 原始文献如何改变了这个提案

2026-10-27通过公开Web检索并亲读以下一手材料：

- [Wright等，AISTATS2024，§3、附录B](https://proceedings.mlr.press/v238/wright24a/wright24a.pdf)以及[作者源码入口](https://github.com/omwright/cov-prop-nn)：其均值/协方差传播与Gaussian GELU公式是已有数学；本提案不把公式发明权归给FFNDS。它促使这里明确区分取得 `diag(WΣWᵀ)` 与对激活分布作高斯替代。其目标是神经网络不确定性传播，不是原生Wan逐token FFN省算。
- [Kuang与Lin，2026预印本，§2、§4及附录D/H/N](https://arxiv.org/html/2601.22307v1)：非对角Gaussian层矩和多层非Gaussian误差是分开的对象。这里最有用的反例方向是即使输入联合矩准确，非线性后续层也可能失配；因此没有从四panel均值外推完整视频质量。
- [Ghashami等，Frequent Directions，原论文§1–3](https://arxiv.org/pdf/1501.01711)与[作者实现](https://github.com/edoliberty/frequent-directions)：另一条合法信息路线是流式联合协方差草图。其谱误差界给出所有w上的投影方差控制，但反复分解/更新成本与随机token草图不同。这里选择随机方差估计作为最小判别，未实测或选择FD部署。

检索关键词包括Gaussian/GELU moment propagation、joint covariance、diagonal estimator、Frequent Directions及neural covariance。另定位到Bekas–Kokiopoulou–Saad对角估计文献；上述Rademacher方差式由本文展开四阶探针矩直接推得，没有把未精读材料列作已验证源码。这个构造使用已有随机线性代数，新的待测对象是**真实FFN所需矩信息的取得代价与其功能价值**。

## 最强反例和有辨识力的下一观察

一个相关输入反例是 Q的两列为(x,−x)，w=(1,1)：真实投影方差严格为0；对角近似产生正方差；任意token草图经原w投影仍为0。这区分联合信息与输入独立性。

另一个对称稀有尾分布在0处取概率1−ε，在±1/√ε各取ε/2，保持均值0、方差1；其平均GELU可随ε趋零，而Gaussian矩闭合仍给非零值。这区分“方差估计已修好”与“分布闭合成立”。任意固定低维草图也存在对探针正交的数据方向；概率保证针对固定数据、独立随机探针，不能抵抗事后构造或输出驱动选seed。

拟议最小判别保持原block10step1的四panel、同权重、同signed readout：在前置固定探针下比较对角、联合草图与真实z矩；另加入经验erf-GELU均值以区分Gaussian闭合与tanh/GH32差异。报告方差、signed readout、不同panel和seed敏感性，不搜索最优探针/新bank。若联合草图也无法恢复有价值的功能坐标，停止这项信息表示的扩展；若有效，再研究它对tokenwise必要计算的关系，而非直接扩大40层视频替换。

## EXECUTION / 物理成本

本文是ALG+CONJECTURE，0新数值/模型/GPU/视频。未来真实分析须另冻有限scope并按当前临时持续授权登记。

理想主MAC为 `MD(s+1)+(s+2)DH`：输入均值/草图、W1的s+1个向量、W2均值readout各计费；中心化、非线性/量化、归约、探针存储和layout另算。完整dense tokenwise双投影为2MDH。这个比较只计算**一个token集合的均值坐标**，不代表同等完整FFN功能，也不能据此声称同等压缩比。实际H=13,824、D=5,120，W1/W2权重至少合计283,115,520bytes BF16，s很小时可能受带宽、小M利用率和额外Q读取主导。

图锚点 F04→F06→F08R/F08→F09：Q1之后、W1之前可以采集输入草图；`W1U`是明确支付的新投影，不是免费信息。新边是conditional，输出为集合均值而非逐token真实post-Q2。其与跨neuron共享函数、已有posterior集中、时空组织及四步视频之间的联系均保持OPEN。
