# 从模块误差到联合状态：完整块缺陷的有限变化

2026-11-25。用户新增判断：“6源基本没有差别。”当前六来源成片证据可据此记为用户报告视觉基本相当；下文发展新数学对象，不把这条反馈解释为对任意扰动的全局收缩定理。

分析采用 operator-tensor-analysis 的有限割线、读出与状态语义，以及 active-abstraction 的任务充分表示、更新闭合和构造成本。源码依据 FFNDS@873e1a879b91d544fdae11894d68bbc6acc854d1 的生产图与在线恢复接口。新增计算是历史真实 Gram 的正交分解、SymPy 恒等式和明确标记的合成非线性块验证；新增模型/GPU/视频执行均为0。

## 1. 真实资产进一步说明：联合影响产生了不同于单模块误差的新方向

第39层历史四臂实验给出 H448 上的原生输出 Y00、仅 attention 替换 Y10、仅 FFN 替换 Y01、两者同时替换 Y11。先将已输出张量转 FP64 再做有限差分：

\[
E_A=Y_{10}-Y_{00},\qquad E_F=Y_{01}-Y_{00},\qquad
I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

所以联合误差精确为 E_A+E_F+I。这里所有输出位置和通道共同作为一个向量场；在 flow 层面，每个场具有448×64个标量。

本轮从归档3×3 Gram，求出 I 在 span(E_A,E_F) 上的正交投影：

\[
G_2=\begin{bmatrix}\langle E_A,E_A\rangle&\langle E_A,E_F\rangle\\
\langle E_F,E_A\rangle&\langle E_F,E_F\rangle\end{bmatrix},\quad
\beta=G_2^{-1}\begin{bmatrix}\langle E_A,I\rangle\\\langle E_F,I\rangle\end{bmatrix},
\quad I=\beta_AE_A+\beta_FE_F+I_\perp.
\]

因此：

\[
\|E_A+E_F+I\|^2
=\|(1+\beta_A)E_A+(1+\beta_F)E_F\|^2+\|I_\perp\|^2.
\]

用真实归档重新计算，BF16 wide8192 背景得到：

| 历史站点 | βA | βF | I⊥能量 / I能量 | I能量 / 联合误差能量 | I⊥能量 / 联合误差能量 |
|---|---:|---:|---:|---:|---:|
| 0392 high | −0.11759 | −0.03167 | 94.02% | 6.64% | 6.25% |
| 0392 low | −0.48306 | −0.07295 | 71.02% | 18.32% | 13.01% |
| 1013 high | −0.20773 | −0.05739 | 85.56% | 11.42% | 9.77% |
| 1013 low | −0.45150 | −0.03817 | 72.31% | 12.12% | 8.76% |

这使联合机制更具体：交互中确有沿两个单独误差场反向的部分，同时大部分交互能量位于这两个场的二维张成空间之外。对两个单独误差只作全局标量重配，无法产生这部分方向。它在总联合误差中的能量份额为6.25%–13.01%，具有实质大小。

这一结论严格限定于历史第39层、四臂、H448 和已选具体替换。I 包含原生 tail、非线性 head 以及固定恢复器随输入变化的联合效应；它不是纯 FFN Hessian，也不是当前六源 C 的机制实测。所用两维空间由两个完整场张成，不能推广为“任意逐位置/逐通道重配都只能达到这个界”。

## 2. 新的恢复对象：完整块的缺陷锚点

固定层、专家和输出空间，令当前完整原生块为 F_s:X→X。时间条件、文本和真实门控明确计入 s。构造一个确实可运行的廉价复合块 G_s：

1. 用当前选中或粗细 attention 得到便宜 self 输出；
2. 在该输出形成的残差上，实际执行 cross 和归一化；
3. 在相应 FFN 输入上，执行固定线性底座与选中非线性。

例如行表示中可取

\[
G_{F,s}(U)=U A_0+\phi(UW_{1,S}^{\top}+b_{1,S})W_{2,S}^{\top},
\]

其中 A0、选中权重及激活可直接来自现有实现；其余偏置或恒定项是否显式放入 G，必须固定为同一个定义，剩余项统一归入缺陷。该式是新复合底座的明确选择，不宣称等同于现有完整 wide8192 恢复器。

定义完整块缺陷

\[
D_s=F_s(X_s)-G_s(X_s).
\]

在原生 anchor 步，保存全位置字段

\[
C_a=D_a=F_a(X_a)-G_a(X_a)\in\mathbb R^{N\times5120}.
\]

最终P覆盖前的当前预测与误差为

\[
\boxed{\widehat Y_s=G_s(X_s)+C_a+\mathcal R_s(P(D_s-C_a)),}
\qquad
\boxed{\widehat Y_s-F_s(X_s)=\mathcal R_s(P\Delta D)-\Delta D.}
\]

若最后将P行覆盖为原生教师，令M_H=I−PᵀP，则最终误差为M_H[𝓡_s(PΔD)−ΔD]。

取 R=0 就得到一个完全可构造的“廉价复合块＋完整块缺陷锚点”基线；加入恢复器时，待恢复量明确为联合缺陷变化 ΔD。该目标同时包含 attention、cross、FFN、gate 与残差的有限联合作用，不需要先把每个部分的误差假设为独立，也不需要小步长。

在相同完整输入与条件回返锚点时，D_s=D_a，预测回返原生块出口。条件/gate 改变时，其作用直接进入 ΔD；这里没有假定锚点时刻的门控可以替代当前门控。

### 2.1 它具体保存了什么

记 T_s(A) 为 attention 写入残差及 cross 的结果，U_s(A) 为 FFN 输入，gF为当前门控。用 A 表示原生 attention，B 表示廉价 attention，Ff/Gf 表示原生/廉价 FFN，则

\[
D_s=
[T_s(A)-T_s(B)]
 +g_F[F_f(U_s(A))-F_f(U_s(B))]
 +g_F[F_f(U_s(B))-G_f(U_s(B))].
\]

保存 D_a，将这三种在 anchor 处已经发生的全位置作用作为一个联合字段保留。当前需要估计的是这一字段怎样变化。它为“联合恢复”给出了一个不同于直接用终端字典重新拟合整个输出的对象。

### 2.2 和历史 D176 的关系

历史 D176 用当前有限字典，从 P64 的64维 flow标签恢复 H448；完整H表示空间与P可达空间之间已有明确差距。这里保存的是每层全位置5120维联合缺陷，当前教师也是块出口5120维。它保留了高维历史字段，并将目标改为 ΔD。

这个区别解释了新构造为何值得单独推导。它没有自动证明 ΔD 容易恢复。后续若选择具体 R，应继续保留原生 P 链中已经计算的逐头 attention/FFN 中间观测，用它们限制允许的恢复结构；完整块/flow目标决定这些结构对任务的作用。不能仅以“块输出维度5120”宣称P64可识别，也不应将已计算的模块教师先压成一个64维终端标签。

## 3. P64 如何实际观测新的目标

当前模型的 self P64 必须读取全部当前 K/V；self 后的 cross、LN、FFN 与 residual 在视频query轴上逐行。因此，从同一当前块输入 X_s 出发，可以用一次原生 self P64 加原生尾部得到 F_s(X_s)_P。廉价全行路径已经给出 G_s(X_s)_P，缓存给出 C_a,P，三者构成真实创新标签。

**观测必须在原生 P 覆盖之前构造。** 如果把原生教师先覆盖进 G_P，则 G_P=F_P，当前缺陷被人为设为零。若锚点也这样覆盖，C_a,P=0，PΔD就全部为0，而非P的实际ΔD仍可能显著。此时恢复器看到的零来自执行中的覆盖操作。

在现有 C 代码中，attention输出先在P行被原生覆盖，FFN probe因而沿原生attention输入运行。若复用这一结构来分析未覆盖廉价复合块，必须保留覆盖前的 attention P，并实际计算其 cheap shadow tail。非线性tail依赖该输入，仅保存覆盖前attention值还不能得到G_P。

新构造的一种清楚执行顺序是：先完成无原生覆盖的全行 G；另外做原生 P 链取得教师；计算 PΔD；再执行恢复并决定是否在块出口覆盖P。原生中间教师都可保留，约1.25MiB的单个FP32 P64×5120张量不是主要状态成本。

## 4. 和整个后缀相容的商空间

当前普通 Wan 生产图的 residual branches 都先沿通道 LayerNorm，门控只依赖时间，末端 flow head 也先作 LayerNorm。固定这些图条件，且没有额外直接读取 raw hidden 的 adapter 时，令 E_i=c_i·1 为逐位置通道常数平移，则实数域中每块满足

\[
F_s(X+E)=F_s(X)+E.
\]

若 G 也按上述归一化分支构造，同样满足 G_s(X+E)=G_s(X)+E。因此其缺陷 D_s 对这种平移不变。原生后缀的所有块保持这个差异，末端 LayerNorm 消去它；近似后缀还要求近似器及缓存更新保持同一平移关系。

这给出一个可检验的完整后缀充分条件：校正后块出口误差若属于逐位置通道常数空间，则原生后缀的flow相同。它把“下游会读什么”变成了精确性质，保留了attention跨位置传播；没有将任意误差都视作被LN消除。

这里的常数方向作用于完整块出口的全部5120通道，位置在输出投影、门控和块内残差之后。它不是每个attention head的128通道均值方向；不能把head空间的平移直接视为相同的后缀等价类。

相应地可以用通道中心化投影 Π 定义任务等价的缺陷字段 ΠD；若整个后续近似和缓存更新都保持同一平移关系，则保存 ΠC_a 并恢复 ΠΔD 已足以保留该后缀。每位置只少一个通道，因此这个商本身几乎不节约容量；其价值是明确联合恢复的合法等价方向，而不是提供大幅压缩。

BF16 cast、缓存量化和分支重排有各自舍入误差，以上严格等价属于实数算术。脚本使用明确标记的合成非线性块验证结构等式，误差约10^-15；没有把这个数值写成真实Wan的BF16一致性。

## 5. 构造和成本都必须闭合

真实尺寸 N31668、5120通道、一个驻留专家40层，单个BF16完整字段占 **12.0804 GiB**。联合缺陷字段有机会替代多个分别存储的输出缺陷字段，但它不能同时充当原算法的 FFN u0、选中激活或未知未来读出所需状态。

若 G 使用上面的无锚点线性＋selected非线性FFN，就定义了新的模块组合；若 G 继续调用现有 wide8192 hit，它仍需要原有 FFN 状态，联合字段只会增加容量。两种情况应分开计算。

anchor 的 G_a 需要从廉价attention输入继续执行完整cheap tail。原生anchor已经计算的FFN输入来自原生attention，两者不同，所以 G_a 不能直接由已有原生tail免费获得。这项全行shadow计算，是该构造的主要新增成本。

四步 HHLL 中，每个专家只有一个anchor和一个hit。若记廉价块、P教师、恢复和状态维护的成本分别为 T_G、T_P、T_R、T_C，基本周期为

\[
T_{\rm macro}=T_{F,a}+T_{G,a}+T_{G,s}+T_P+T_R+T_C.
\]

额外shadow只由一次hit摊销。真正比较时，需要与当前C的完整anchor+hit成本逐项相减；已有C的13.48% GPU收益不能直接移植给这个新构造。

## 6. 当前可以作出的选择

这条推导形成了三个可用结果：

1. 真实历史交互场具有实质的二维误差场之外分量，支持继续保留联合过程产生的新作用方向。
2. 完整块缺陷锚点是一个可构造的新恢复对象，能把联合作用存为全位置场，并把当前目标改成其有限变化。
3. P观测位置、shadow尾部和一次复用回本条件都已经确定，可以据此判断任何后续实现是否真的计算了所声称的对象。

当前C已经拥有六来源的成片基本相当反馈与四步GPU时间收益证据，可作为新构造的固定参照。是否把联合缺陷作为下一执行路线，应由可取得的ΔD结构和shadow成本决定；这里交付的是对象、证明、已有资产的新几何和可计算性分析，没有新增GPU实验任务。

## 复核与来源

- [verify_joint_defect.py](verify_joint_defect.py) 与 [joint_defect_verified.json](joint_defect_verified.json)：原始Gram重新读入、Schur正交分解、符号与合成检查、真实尺寸静态预算。
- [0392 原始回执](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/RESULT_0392.json)、[1013 原始回执](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/RESULT_1013.json)。
- [四臂定义与H448 Gram代码](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/attention_joint39.py)。
- [在线attention及P覆盖](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)。
- [生产计算图审查](https://github.com/Qesire/FFNDS/blob/873e1a879b91d544fdae11894d68bbc6acc854d1/research/20261124/attention_wide8192/results/JOINT_GRAPH_AUDIT.md)绑定原始 LightX2V@ed146db64d0b6130c669285fc1aa0b280a6a4823 的 transformer 与scheduler，并确认各LayerNorm、时间条件和无额外adapter的当前配置。该构造尚未替换生产实现。

在仓库 joint39 目录可运行 `python verify_joint_defect.py --source-dir .`。本轮只读取已归档结果，没有加载真实模型权重或原始四臂向量。
