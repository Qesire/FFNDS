# SparsePR 重复性：条件等价、历史信息与仿射读出的边界

日期：2026-11-27。以下是算子推导与 CPU FP64 构造核验；没有新增 Wan 数据、模型执行或 GPU 性能结果。它不是 SparsePR 作者实现的复现。

## 1. 最简判断

“稀疏输出＋少量原生 query＋affine ridge＋原生行覆写”与 SparsePR 属于同一个核心恢复框架。将拟合标签称为“创新”本身只是在已有回归中更换先验，并不自动产生新的恢复算子。

当前 low 方法真正可以超出这一当前响应读出的是：锚点全位置状态提供当前 sparse 输出及其 affine 特征未保留的逐 query 差异，并按当前 value 坐标运输。此差异能够用实际 attention 公式构造出来，而不只靠命名或应用场景区分。

SparsePR 的标准化 affine 残差回归本身，在固定路由、探针、coverage 权重、正对角 value 变换及非退化标准化条件下，也具有 value affine 等变性。此前与 Chipmunk 静态残差的尺度反例，不能作为本方法相对 SparsePR 独有等变性的证据。

## 2. 同一读出算子下的严格等价条件

令完整输出为 \(Y\in\mathbb R^{N\times d}\)，稀疏底座为 \(S\)，探针选择为 \(P\in\mathbb R^{p\times N}\)，\(PP^\top=I_p\)。固定当前恢复器 \(K\in\mathbb R^{N\times p}\)，并记

\[
M=I-P^\top P,\qquad L=M(I-KP).
\]

最终覆写原生探针的恢复为

\[
\mathcal R_B(Y_P)=P^\top Y_P+M\{B+K(Y_P-PB)\}.
\]

有历史先验 \(R\) 的底座是 \(B=S+R\)。精确地，

\[
\boxed{\mathcal R_{S+R}(Y_P)-\mathcal R_S(Y_P)=LR.}
\]

因此，两个方法在本次输入上输出完全相同，当且仅当 \(LR=0\)。历史状态的有效新信息由这个算子作用后的部分体现，而不是由缓存字段的存在、标签改名或先验本身的范数体现。

对字典 \(\Phi\in\mathbb R^{N\times q}\)、\(A=P\Phi\)、正权重 \(W\)、系数惩罚 \(\Lambda\)，

\[
K=\Phi(A^\top WA+\Lambda)^{-1}A^\top W.
\]

若 \(\Lambda=0\)、\(A\) 满列秩，且 \(R=\Phi C_0\)，则 \(KPR=R\)，有无历史的输出严格相同。这里“满列秩”必不可少：若特征列数超过探针数或 \(A\) 秩亏，使用最小范数伪逆时，字典中仍可存在探针无法识别的方向。

若 \(R=\Phi C_0\) 但 \(\Lambda\ne0\)，则

\[
LR=M\Phi(A^\top WA+\Lambda)^{-1}\Lambda C_0.
\]

此时函数空间可以完全相同，变化只来自正则化中心。令总系数 \(D=C+C_0\)，以历史为底座拟合的目标等价于

\[
\min_D\;\|W^{1/2}(P\Phi D-(Y_P-PS))\|_F^2
+\|\Lambda^{1/2}(D-C_0)\|_F^2.
\]

这是有历史系数先验的同一回归问题。相反，\(R\) 若含有 \(\Phi\) 无法表达或探针无法识别的全位置方向，就会保留额外信息。例如 \(PR=0\) 时，差异恰为 \(MR=R\)。

## 3. 不同特征、正则和拟合对象的作用

实际 ours 和 SparsePR 的读出并不共用同一 \(K\)。若分别为 \(K_1,K_0\)，仍共用 \(P,S\)，则

\[
\Delta\widehat Y=
M\{(I-K_1P)R+(K_1-K_0)(Y_P-PS)\}.
\]

第一项是历史在新读出后仍保留的作用，第二项是恢复器本身改变的作用。两者须分开解释。

将“拟合 residual”改成“拟合 dense”也有同类等价问题：

\[
M\{S+K(Y_P-PS)-KY_P\}=LS.
\]

若 sparse 底座能够被无正则、探针可辨识的字典精确重建，两种写法等价；有 ridge 时常体现为 identity/zero 先验中心的变化。

根据本轮实际源码核查，high 的理想实数特征为 \([1,S C]\)，其中 \(C\in\mathbb R^{128\times31}\) 为固定 CAL 投影。其恢复形式

\[
S+(SC)T+\mathbf1b^\top
=S(I+CT)+\mathbf1b^\top
\]

属于当前 sparse 输出的 affine 函数类，且相较 full-channel affine 有固定输入子空间限制。SparsePR v2 默认无输出截断，标准化、均值中心化后的 full-channel affine 特征在非退化条件下覆盖这一实数函数类。特征降维、权重、正则与 BF16 舍入产生具体实现差别；这些差别不改变两者共享当前响应 affine 恢复框架的事实。

low 的历史有两个不同入口：逐位置运输底座，以及当前与锚点投影之差的特征字典。忽略舍入时，后者为 \(\Phi_\Delta=[\mathbf1,(S_t-S_a)C]\)。固定 \(S_a\) 时，它对整个 \(S_t\) 矩阵是 affine 映射，但 \(-S_aC\) 是逐 query 不同的偏置，一般不属于 SparsePR 的逐行共享 affine\((S_t)\) 函数族。若 \(S_{t,i}=S_{t,j}\) 而 \(S_{a,i}C\ne S_{a,j}C\)，差分字典仍能区分两行。

例如令当前四行 sparse 输出全部为 0，锚点输出为 \((-1,0,1,2)^\top\)，\(C=1\)，probe 取首尾两行，并令目标为锚点输出的相反数。没有历史底座时，差分字典已能恢复全部四行；当前共享 affine\((S_t)\) 只能在两个未观测行给出相同值。脚本给出输出差范数 \(0.70710678\)。这只证明差分字典可独立带来历史区分，未对实际模型里底座和字典的贡献大小作归因。

## 4. SparsePR 式当前回归也可 value affine 等变

固定完整和 selected attention 权重、路由、探针与 coverage。令

\[
V'=V\Gamma+\mathbf1c^\top,
\qquad\Gamma>0\text{ 为对角阵}.
\]

则 \(Y'=Y\Gamma+\mathbf1c^\top\)、\(S'=S\Gamma+\mathbf1c^\top\)，残差 \(Y'-S'=(Y-S)\Gamma\)。对每通道减均值、除标准差的特征，若尺度非退化并忽略固定 epsilon 的影响，标准化特征保持不变。固定权重和正则下 \(K'=K\)，因而

\[
\widehat Y'=
S'+K\{P(Y'-S')\}
=\widehat Y\Gamma+\mathbf1c^\top.
\]

原生探针覆写也保持此关系。这个性质是输出等变性，不等于完整输出精确性；已有近似误差同样随 \(\Gamma\) 变换。

当前路由若依赖响应并随变换改变，或尺度 epsilon、截断等改变了特征，该固定路由命题不直接推出整个动态系统等变。相同限定也应放在本方法的局部命题旁。

## 5. 一个普通 attention 就能实现的严格区分见证

构造三 query、三 key 的完整概率矩阵

\[
\Pi=
\begin{bmatrix}
0.7&0.2&0.1\\
0.7&0.15&0.15\\
0.7&0.1&0.2
\end{bmatrix},
\qquad V_a=(0,0,1)^\top.
\]

它可以用标准 scaled-dot-product attention 精确实现：取 \(K=I_3\)、\(Q=\sqrt3\log\Pi\)，则 \(\operatorname{softmax}(QK^\top/\sqrt3)=\Pi\)。所有 query 的 top-1 都是 key 0。

selected 输出三行均为 0，遗漏质量三行均为 0.3；完整输出却为 \((0.1,0.15,0.2)^\top\)。probe 只选择中间行。任何共享的 affine(S) 恢复在另外两行只能给出相同值，锚点完整响应则保存了两行的差别。

令 \(V_t=1.7V_a-0.3\)。当前完整输出为

\[
Y_t=(-0.13,-0.045,0.04)^\top.
\]

当前 affine(S) 拟合给三行相同的 \(-0.045\)。centered 逐位置历史运输则精确给出上述三行。该构造中 sparse 输出、anchor sparse 输出与 rho 全部相同，差别明确保存在逐位置 Z 中。

两种方法都满足这一条件下的 value affine 等变性，只有保存了该历史场的方法在此构造中能精确还原完整响应。区分点是历史提供了当前 affine(S) 读出未保留的逐 query 信息。所有完整 Q/K/V 都是已知输入，所谓“不可辨识”仅针对受限读出接口，不是对完整 attention 的信息论不可能性主张。

脚本另保留更小的两 key 构造。它们只证明存在严格不同的行为，并不声称这种构造代表 Wan 的典型分布或视频质量收益。

## 6. 本轮数值核验

- 一般输出差 \(LR\) 恒等式最大误差：\(9.99\times10^{-16}\)。
- 无正则且可辨识字典内的历史被完全吸收，输出差：\(3.55\times10^{-15}\)。
- ridge 同字典、不同先验中心的差式误差：\(2.89\times10^{-15}\)。
- 不同恢复器的分解差式误差：\(1.33\times10^{-15}\)。
- 固定路由、标准化 affine 回归的 value affine 等变误差：\(1.08\times10^{-15}\)。
- 三 key attention 构造的历史运输最大误差：\(2.78\times10^{-17}\)。

这些数值均为 FP64 构造的代数证书。对应脚本为 `verify_equivalence.py`，完整输出为 `EQUIVALENCE_VERIFIED.json`。
