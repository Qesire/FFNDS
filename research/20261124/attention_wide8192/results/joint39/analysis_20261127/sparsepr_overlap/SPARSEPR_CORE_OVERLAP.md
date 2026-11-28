# 我们是否与 SparsePR 的核心完全重复？

日期：2026-11-27。当前执行源码固定于 FFNDS `d3503514dfc1a747b07c2811c8f38d30b3ba2596`；比较 SparsePR v1（2026-08-19）及 v2（2026-09-27）。本轮完成原文、实际源码和 CPU 代数核查，新增模型/GPU执行为0。

## 1. 直接判断

**当前原生 probe 观测、稀疏输出残差拟合、affine/ridge 全场恢复和 probe 覆写，这个核心与 SparsePR 直接重合。当前 high OUT-ABS 是其 v2 默认仿射恢复函数族的受限输入特征变体。** 上一轮只说“共享恢复组件”，对这一重合的强调还不够。

**完整当前方法通常不等价。** low centered 在全位置历史底座和跨步恢复字典两处引入历史信息；wide8192 FFN 的割线底座、16增益/探针核联合解也具有不同结构。可成立的研究增量应落在这些具体操作及其实际质量/成本收益上。

| 层次 | 核查后的关系 | 可以承担的研究表述 |
|---|---|---|
| 原生P64、当前输出残差、ridge、探针覆写 | SparsePR直接已有 | 明确引用的共同恢复机制 |
| high OUT-ABS | 共享 affine(S) 家族的受限特征变体 | CAL输入子空间、估计器与执行成本的具体取舍 |
| low 的全位置历史底座 | SparsePR当前输出恢复之外的输入 | 保存并运输当前稀疏读出丢失的位置差异 |
| low 的跨步特征字典 | 历史决定逐位置特征 | 用变化特征约束剩余更新 |
| wide8192 FFN | 共享原生观测思想，底座与读出算子另有结构 | 具体算子扩展及其与attention的联合作用 |
| 四步蒸馏、Wan模型、P64数量 | 应用条件或参数 | 实证范围与资源条件 |

## 2. SparsePR 的恢复目标与实际步骤

固定一个 head，记完整输出为 A∈R^(N×d)，selected 内重新归一化的输出为 S，P选取原生probe。SparsePR的目标确实是 R_P=A_P−S_P。它以当前 S 的完整输出通道作回归特征，按probe coverage加权，用当步统计标准化特征、中心化残差，拟合共享的仿射修正并在P行写回原生值。

其 v2 对未观测行的算子是

\[
\widehat A=S+\mathbf1\mu_R+\bar S B_\lambda,
\qquad
B_\lambda=(\bar S_P^\top W\bar S_P+\lambda I)^{-1}
\bar S_P^\top W\bar R_P.
\]

v1另将中心化预测投影到当步probe残差的rank16输出子空间；v2默认取消该截断。输入降维与输出方向截断发生在不同轴上，需分别判断。原文定义、版本差异和附录覆盖见[原文审查](primary/SPARSEPR_PRIMARY_AUDIT.md)。

论文C.1会复用centroid/factor作为迭代初值，因此准确的区别是其输出恢复没有完整历史响应残差场，而非整个算法完全无状态。[SparsePR v1](https://arxiv.org/html/2608.18484v1)；[SparsePR v2 §4.2、A.3、C](https://arxiv.org/html/2608.18484v2)。

## 3. high：可以严格定位到同一函数族

当前Wan每头d=128，实际实现将S经固定投影降到31维，并添加常数列。忽略BF16舍入，令C=P_vU_CAL∈R^(128×31)，则

\[
\Phi=[\mathbf1,SC],\qquad
\Theta=(\Phi_P^\top\Phi_P+6.4I_{32})^{-1}\Phi_P^\top(A_P-S_P).
\]

把系数拆开，有

\[
\widehat A=S+\mathbf1\theta_0+SC\Theta_1
=S(I+C\Theta_1)+\mathbf1\theta_0.
\]

因此修正仍然是当前S的共享仿射函数，固定C限制了其输入方向。它属于SparsePR v2默认无输出截断的完整通道仿射函数族的受限子类。两者的实际估计器有差异：我方使用固定CAL输入投影、等权、连同bias一起正则化；SparsePR使用当步coverage权重和标准化、单独处理bias。BF16舍入也使实际路径不同。

这些差异具有真实的估计和执行含义，但更准确的定位是同一恢复原理中的结构取舍。将“我们也使用64原生行恢复残差”作为新核心，会与SparsePR重合。当前源码、真实形状和完整求解式见[执行路径审查](ours/OURS_RECOVERY_CODE_AUDIT.md)。

## 4. low：底座与字典两处携带额外历史

当前low保存

\[
Z_a=[A_a-\operatorname{diag}(1-\rho_a)S_a-\rho_a\mu_a^\top]D_a^{-1},
\]

并在当前value坐标解码

\[
b_t=\operatorname{diag}(1-\rho_a)S_t+\rho_a\mu_t^\top+Z_aD_t.
\]

其恢复字典也依赖历史。在理想实数中，

\[
\Phi_t=[\mathbf1,(S_t-S_a)C_{low}],
\qquad
\widehat A_t=b_t+\Phi_t\Theta_t,
\]

其中Theta由当前原生probe与b_t之差拟合，最后覆写probe。实际代码分别舍入S_tC和S_aC再相减，并保留零尺度处理。

这里存在两个额外的信息入口：

1. Z_a保存每个位置的历史完整响应，可以区分当前S相同的两个位置。
2. −S_aC是逐位置不同的特征偏置，可以让当前S相同、历史S不同的两行具有不同特征。

固定锚点后，差分字典对整个S_t矩阵是仿射映射，但通常不属于SparsePR逐行共享的affine(S_t)函数族。不能忽略位置轴而将二者混为一谈。实际模型中两个历史入口各有多大作用，需要配对分解；本轮不将贡献预先全部归给某一项。

## 5. “拟合创新”何时只是改写？

先固定同一P、底座S和恢复算子K，以隔离历史底座R的作用。令M=I−P^T P，最后覆写原生probe：

\[
\mathcal R_B(A_P)=P^\top A_P+M\{B+K(A_P-PB)\}.
\]

有无历史的输出差严格为

\[
\boxed{\mathcal R_{S+R}(A_P)-\mathcal R_S(A_P)=M(I-KP)R.}
\]

因此，是否更换了标签名称、是否增加缓存字段，都不是充分判据。

- 若R能被当前字典表示、probe足以唯一识别系数，并采用无正则求解，则历史被恢复器完全吸收，两种输出相同。
- 在同一字典内采用ridge，加入历史可等价为将正则中心从零改到历史系数；这仍是同一函数族中的先验选择。
- 若历史包含当前字典或probe不能识别的位置方向，它便提供额外信息。

实际ours与SparsePR的恢复器也不同。令其分别为K_1、K_0，则

\[
\Delta\widehat A=M\{(I-K_1P)R+(K_1-K_0)(A_P-PS)\}.
\]

第一项隔离历史底座在恢复后的作用，第二项包含字典、权重和正则等恢复器变化。low的差分字典属于后者的历史依赖。完整证明和数值见证见[等价性审查](math/EQUIVALENCE_REVIEW.md)。

## 6. 用标准attention构造真正不同的行为

取三query、三key的完整概率

\[
\Pi=\begin{bmatrix}.7&.2&.1\\.7&.15&.15\\.7&.1&.2\end{bmatrix},
\qquad V_a=(0,0,1)^\top.
\]

取K=I_3、Q=√3 log(Pi)，标准scaled-dot-product attention恰好产生Pi。每行top-1都是key0，selected输出全为0，遗漏质量全为.3；完整输出却为(.1,.15,.2)^T。只将中间行作为probe，任何共享affine(S)都只能对另两行给出同一个值。

历史Z保留了这两行的区别。令V_t=1.7V_a−.3，当前完整输出变为(−.13,−.045,.04)^T：当前共享affine(S)给出三行同为−.045，历史运输恢复上述三行，FP64最大误差2.78e−17。仅扩query轴、将中间query复制成64行，可得到66-query/3-key且含64个probe的矩形attention见证。

另一个核验只改变差分字典、不加入历史底座：当前S四行全0、锚点S_a=(−1,0,1,2)^T，首尾为probe。差分字典可恢复目标−S_a，FP64误差2.22e−16，而当前共享affine(S)不能区分两个留出行。

这些是由真实attention公式或线性字典构造的代数见证，不是Wan典型分布、画面质量或性能结果。“不能识别”只针对指定的恢复输入和函数族；完整Q/K/V作为原始输入当然足以计算原生attention。

## 7. 尺度等变性也要与SparsePR分别核对

上一轮针对Chipmunk静态历史残差的尺度构造成立。对SparsePR v2还可以从其标准化回归推出：固定完整/selected权重、支持、P、coverage与lambda，非退化尺度并忽略epsilon时，对于V'=VΓ+1c^T、Γ正对角，其估计满足

\[
\widehat A'=\widehat A\Gamma+\mathbf1c^\top.
\]

本轮FP64核验误差1.08e−15。固定epsilon、输出截断或动态路由改变时需分别处理。该性质表示输出等变，原有误差也随Gamma变化；它不等于拥有完整历史后对上述变化类的精确恢复。

因此，相对SparsePR的区分应落在历史保留的信息与更新法则。单独强调适应value尺度不足以说明独有性。

## 8. 对研究主张的调整

将P64在线残差恢复明确作为有直接前例的共同机制。当前high的贡献定位为该机制的结构与执行选择；low的研究主张集中为：**在短复用周期中保存逐位置历史响应，按当前value统计解码，并通过跨步特征及原生观测更新剩余变化。** FFN的割线底座与联合读出另外按实际算子论证。

现有成片和配对计时证明完整路线已经可用；本轮推导给出了超出当前affine(S)的明确行为。新增结构的贡献大小应由同一底座、同一probe预算和完整锚点周期下的比较确定。评价分别观察恢复后、后继FFN/块出口与整片动作/纹理，让空间信息的保留体现为实际质量或资源收益。

最有区分力的比较应将历史底座与跨步字典分开控制，并保留作者式当前S全通道恢复作为近邻。这样能判断哪些历史信息值得其状态和锚点成本，而不把任何非等价的实现细节都升级为独立核心。

## 本轮资产

- [原文审查](primary/SPARSEPR_PRIMARY_AUDIT.md)
- [当前源码审查](ours/OURS_RECOVERY_CODE_AUDIT.md)与[固定文件身份](ours/SOURCE_IDENTITIES.json)
- [数学等价性与两个历史入口](math/EQUIVALENCE_REVIEW.md)
- [CPU核验脚本](math/verify_equivalence.py)与[结果](math/EQUIVALENCE_VERIFIED.json)

核验脚本需要Python和NumPy，可从任意目录运行，结果写回脚本所在目录。本轮未执行作者未发布的完整实现，未提交GPU生成作业。
