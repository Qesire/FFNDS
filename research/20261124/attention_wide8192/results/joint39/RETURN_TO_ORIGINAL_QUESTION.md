# 从 D176 回到原问题：FFNDS 怎样扩展到注意力与组合计算

日期：2026-11-24。证据固定于 [FFNDS@0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c][head]。分析使用 evidence-analysis、operator-tensor-analysis 与 active-abstraction；具体调用受限观测恢复、正交投影、有限变化、目标相关误差和计算图闭合的数学工具。

本次完成的是代码核查、真实发布收据的独立 CPU 复算、数学推导及一个明确标记的合成代数检查。学校作业 193683 已完成真实 FFN 核补采；本次分析新增真实模型/GPU执行数为0。学校封存的 joint39 向量和 kernel safetensors 当前未在本地，因此下文新提出的 P 可达空间数值仍待使用这些既有资产计算。

> 后续已在学校用封存向量执行第 9.1 节。[固定 P64 可达空间实测与完整奇异谱](D176_OBSERVABILITY_REPORT.md)记录八站点结果，分清了独立底座的系数估计缺口与便宜底座的低噪声表示缺口。

## 1. 原问题现在可以怎样回答

**现有证据支持继续把 FFNDS 的结构化恢复机制扩展到 attention，并把完整块作为观测与作用分析的对象。** 可迁移的机制是：锚点保存已有状态，便宜当前计算保留真实有限变化，少量原生查询约束剩余缺陷，恢复器按算子结构把这些约束传到其他位置。

这轮结果把实现条件进一步明确了：

- 注意力已有可执行的原生 P64 查询入口；本模型内，这些查询能够继续经过 cross、归一化、FFN 和 head，形成同输入下的完整块教师。
- 原生后续 FFN 和终端 head 确实改变注意力扰动的作用。恢复目标应包含这些联合作用。
- 原注意力和 FFN 的按头分解、当前非线性方向、模块教师与核几何，提供了当前 D176 终端线性字典尚未覆盖的作用。下一构造应把这些结构保留在恢复过程中。

因此，联合目标应决定恢复什么，原生中间教师和模块结构应决定怎样用有限查询恢复。当前 D176 适合继续作为表示与观测的诊断对象；下一阶段的方法路线应围绕这种结构化组合展开。

## 2. 最新实验给出的直接证据

本轮已覆盖 DEV0392、DEV1013，Dense/BF16 wide8192 两背景，high/low 两专家，共8个第39层站点。四条背景轨迹与既有四步状态及解码 RGB 逐位一致；新增 FFN 核的入块状态、原生 attention、块出口和 q16 也有逐位匹配收据。统计单位为两个来源，背景和专家属于来源内重复条件。[补采收据][export]、[逐站点结果][replay]

下面以同一站点的原独立读出误差为1，再对两个来源的能量比取几何平均。P64 列表示仅用 P64 拟合、在 H448 评价；H448 下限表示直接用 H448 全标签拟合并在 H448 评价。

### 2.1 保留独立结构，追加终端校正

| 表示 | high：P64 点误差 | low：P64 点误差 | high：H448 点下限 | low：H448 点下限 |
|---|---:|---:|---:|---:|
| Φ48 | 1.22824 | 1.29263 | 0.69041 | 0.73849 |
| Φ48 + 头间对比核 | 1.21053 | 1.27685 | 0.46197 | 0.54970 |
| Φ48 + FFN 核 | 1.14481 | 1.20124 | 0.43925 | 0.51375 |
| D176：两种核均加入 | 1.14077 | 1.20056 | 0.29984 | 0.38168 |

只加 FFN 核，相对 Φ48 的 P64 点误差下降6.79%/7.07%。在此之上加入头间对比，实际点误差再降0.35%/0.06%，完整 H448 表示下限却再降31.74%/25.71%。这组配对说明头间对比提供了与终端残差相关的表示方向；固定 P64 和当前读出利用这些方向的程度，需要单独解释。[学校报告][report]、[独立复算](joint39_round3_recomputed.json)

D176 保留独立底座时，0392/1013 的实际点误差比分别为 high 1.13029/1.15134，low 1.17080/1.23108。空间与时间边误差在两源、两专家上也都高于原独立读出。现有独立路径继续作为此轮的有效参考。

### 2.2 恢复对象的变化，比字典宽度更关键

| D176 的底座 | high：P64 点误差 | low：P64 点误差 | high：H448 点下限 | low：H448 点下限 |
|---|---:|---:|---:|---:|
| cheap：SP/ANCH + FFN 便宜底座 | 1.48894 | 9.81573 | 0.36519 | 3.41988 |
| 原独立读出，覆盖前 B11_fixed | 1.14077 | 1.20056 | 0.29984 | 0.38168 |

低噪声下，即使给当前 cheap→D176 字典全部 H 标签，其点误差仍为独立读出的3.420倍。保留独立路径后，同一字典的 H 下限降到0.382倍；两个下限之比为8.960倍。改变底座实际改变了待恢复函数的复杂度和方向，而不仅是给同一误差换一个初值。

H448 同 λ=0.1 的 D176 拟合，在独立底座上得到0.63212/0.67412倍点误差。这说明完整 H 字典与具体正则化解也存在差别；它尚未把 P64 的位置、观测、当前读出子空间和正则效应分别识别出来。

上述“下限”均限于固定面板、固定字典和点平方误差。对应的空间/时间边分数是点最优系数的边评分，不是边损失各自的最优下限，也不是分布外误差保证。

## 3. 原 FFNDS 与 attention 类推，依赖哪些真实计算关系

### 3.1 FFN：有限非线性底座、模型方向与稀疏残差延拓

当前 wide8192 先利用同位置锚点、折叠线性变化和2560个选中非线性通道构造底座，再采用：

\[
\widehat F_i
=b_i+q_i\operatorname{diag}(g)V+\tfrac12 K_F(i,P)C.
\]

这里 q 有16个坐标，V 是已知输出方向，g 是方向增益；核坐标来自当前选中激活相对锚点的变化。原生 P64 FFN 输出同时约束这些结构。实际求解使用原核和 probe inverse metric，包含通道方向约束；它在计算图中保留了非线性响应与模型权重的关系。[原分叉实现][jointcode]

### 3.2 attention：省略质量与当前 value 方向的有限耦合

固定一个 query/head。令当前选中 key 输出为 \(O_s^S\)，未选中 key 的总概率质量为 \(\rho_s\)，其对完整输出的加权 value 贡献为 \(u_s\)。有：

\[
A_s=(1-\rho_s)O_s^S+u_s.
\]

锚点 a 提供 \(\rho_a,u_a\)，从而得到当前底座：

\[
b_s=(1-\rho_a)O_s^S+u_a.
\]

固定锚点参考 \(v_*\)，定义 \(w=u-\rho v_*\)、\(c_s=O_s^S-v_*\)。直接展开可得：

\[
\boxed{A_s-b_s=\Delta w-\Delta\rho\,c_s.}
\]

这是有限变化恒等式，不要求 anchor→hit 是无穷小变化。原 FIN-C0 用：

\[
\widehat A_{ih}
=b_{ih}+X_{ih}B_h-(X_{ih}a_h)c_{ih},
\]

把质量变化沿当前 value 方向作用的关系保留在恢复器中。它按 head 组织，当前方向 \(c_{ih}\) 随 query 改变。当前特征、锚点状态及原生 P64 输出分别对应明确的计算时点。[注意力拟合实现][attentioncode]

此前全键 self 见证的有限割线恒等式最大绝对误差为 \(9.68\times10^{-14}\)，而完整 anchor→hit 变化上的一阶切线相对 L2 为115.39%。这支持使用实际有限创新作为迁移的数学对象；它并未把有限恒等式本身等同于廉价可计算的恢复算法。[原始见证与结果][firstresults]

### 3.3 D176 恢复了哪些变量，改变了哪些机制

D176 的 FFN 核确实调用原 ffn_operands，但输入明确为 cheap_U，两种底座均使用这份坐标。独立 FFN 路径在 H 使用 candidate attention 产生的 a_op，在 P 拟合时使用 native attention 的 n_op。三种输入具有不同语义。[核导出实现][exportcode]

此外，D176 将 \(K_F(i,P)\) 归一化后作为64列特征，再统一做 ridge。令 P-RMS 的对角尺度为 \(S_F\)，该部分诱导的 query 核为：

\[
\widetilde K_F(i,j)
=K_F(i,P)S_F^{-2}K_F(P,j).
\]

它与原 FFN 的直接核、固定方向 gains 和 inverse metric 共同构成的恢复算子不同。头间对比核同样保留了部分逐头坐标，但没有自动保留 FIN-C0 的当前输出向量方向及其乘性关系。[D176 实现][d176code]

这使本轮的实验对象非常具体：若干原方法坐标对终端线性字典的增量价值。上述输入、约束、核几何和监督接口共同改变；现有结果还不能把总差额归因到其中某一项。详见[实现语义审查](D176_IMPLEMENTATION_SEMANTICS.md)。

## 4. 同样 P64，保留的教师信息不同

| 恢复路径 | P64 实际使用的教师 | 恢复结构 |
|---|---|---|
| attention OUT-ABS | 40头，各64×128的原生输出 | 每头32维设计分别拟合 |
| attention FIN-C0 | 同上 | 每头有限创新与当前方向乘性耦合 |
| wide FFN | 64×5120原生 FFN 输出 | 固定方向 gains + 直接激活核 |
| 终端 D176 | 64×64原生 flow | 176列统一线性读出 |

固定 query 数量约束了原生查询位置；它没有约束算法必须丢弃或保留哪些中间输出。完整 P 路径中间张量已经被计算出来，终端 flow 是这些张量通过后续图后的任务观测。标签尺寸不是互信息计数，但观测接口确实发生了变化。

在简化的受限线性模型中，令：

\[
R_{ihc}=\sum_{a=1}^{32}\Phi_{iha}\Theta_{hac}.
\]

i 是 query，h 是 head，a 是特征坐标，c 是 head 输出通道。若每个 \(\Phi_{P,h}\) 满列秩，逐头教师可以分别确定各 \(\Theta_h\)。同一64个 query 能约束40个具有各自结构的恢复问题。

若只观察合成的 flow，则：

\[
Y_{iv}=\sum_{h,c}L_{ihcv}R_{ihc}.
\]

把两种教师映射分别记为 \(M_{\rm int}\)、\(M_{\rm flow}\)，H端目标记为 \(T_H\)。在固定已知线性下游的条件下：

\[
\ker M_{\rm int}\subseteq\ker M_{\rm flow}.
\]

目标能够由相应教师唯一确定，要求：

\[
\ker M_{\rm teacher}\subseteq\ker T_H.
\]

于是，中间教师可能区分一些在 P 的终端输出中相抵、却在 H 产生不同作用的变化。最小例子是两头在 P 的特征均为1，在 H 分别为1与−1；终端 P 只知道 \(\theta_1+\theta_2\)，H 所需的却是 \(\theta_1-\theta_2\)。逐头教师分别给出两个参数。

这是一项条件推导。真实 FIN-C0 还需检查包含当前方向的耦合 Schur 观测算子，不能只凭 \(\Phi_P\) 的秩判定；真实非线性图应检查“教师观测相同的允许状态，目标作用是否相同”的纤维条件。

**64维 flow 输出降低了单个 query 的输出宽度，沿 query 轴的恢复函数仍可能复杂。** 当前目标只有64个通道，并不自动意味着一个64维 query 函数空间足够，也不意味着逐模块教师可以等价压成终端教师。

## 5. 新的关键辨析：完整字典与当前 P 读出的可达空间

D176 实际使用 P-only RMS 归一化后的设计矩阵 \(D\in\mathbb R^{512\times176}\)。定义当前底座残差 \(R=Y-B\)，则 P64 岭读出的 H 预测是：

\[
\widehat R_H
=D_HD_P^\top(D_PD_P^\top+64\lambda I)^{-1}R_P.
\]

因此它落在：

\[
\mathcal V_P=\operatorname{col}(D_HD_P^\top),\qquad
\dim\mathcal V_P\le64.
\]

现有 H448 OLS 下限则允许任意系数，使用：

\[
\mathcal U=\operatorname{col}(D_H),\qquad
\mathcal V_P\subseteq\mathcal U.
\]

### 5.1 已发布 rank 已经允许一项明确推论

8个站点中，Φ48 在 P 上的秩均为48；head112、ffn112、D176 均为64。[逐站点结果][replay]

对 Φ48，\(D_P\) 满列秩，因此：

\[
\operatorname{col}(D_HD_P^\top)=\operatorname{col}(D_H).
\]

它的完整字典下限与 P 诱导空间下限相同，差距位于该空间内的系数恢复/正则化环节。

对 D176，P 满行秩只说明其176维系数空间有112维 P 零空间。并非所有这些系数方向都在 H 有独立作用，也并非真实残差都需要它们。当前收据没有 H 字典秩和目标对齐谱，故应实际检查这一中间空间。

### 5.2 可以精确相加的三项能量

令 \(\Pi_U,\Pi_V\) 为上述两空间的正交投影。对当前 FP64 预测及同一点平方误差：

\[
\boxed{
\|R_H-\widehat R_H\|_F^2
=
\|(I-\Pi_U)R_H\|_F^2
+\|(\Pi_U-\Pi_V)R_H\|_F^2
+\|\Pi_VR_H-\widehat R_H\|_F^2.
}
\]

三项依次是：

1. 完整字典仍未表示的分量；
2. 当前 P 与岭坐标选择形成的可达空间，未覆盖的字典分量；
3. 可达空间内，当前 P 标签与求解规则实际选出的系数差距。

这里相加来自正交性，无需假设随机误差独立。第三项混合了有限标签、正则和拟合规则的作用，不应直接命名为采样方差。现有 FP32 输出 cast 产生的能量差另列。

需要补算的中间数值为：

\[
E_{Pspan}=\min_A\|R_H-D_HD_P^\top A\|_F^2.
\]

当前0.29984/0.38168是第一项所对应的完整 H 字典下限；它还不是同一 P64 读出可以实现的精度。

### 5.3 可识别性与当前读出限制是两个问题

要从 P 标签恢复任意线性字典函数的 H 值，充要条件为：

\[
\ker D_P\subseteq\ker D_H.
\]

对于有结构先验的允许缺陷空间，应在该空间上检查。这个条件在可逆特征重参数化下不变。

\(\mathcal V_P\) 则依赖当前 P-RMS 坐标和欧氏岭先验；非正交重参数化、不同正则度量或更强的模块先验可能改变它。因此 \(E_{Pspan}\) 是当前读出与 P 的诊断下限，不是所有 P64 方法的共同信息论下界。

新增字典列保持完整 H 函数族嵌套，却可能旋转 P 所诱导的可达空间。只应要求完整 H 点下限单调，不能要求 \(E_{Pspan}\) 或实际 P64 误差也单调。

## 6. 从现有真实结果直接算出的结构作用下界

即使没有原始向量，也能对独立路径中“当前字典无法表达的作用”给出严格范数界。

固定同一 D176，令 \(Q=I-\Pi_U\)，两底座残差分别为：

\[
R_c=Y-B_{\rm cheap},\quad
R_i=Y-B_{\rm independent},\quad
\Delta B=B_{\rm independent}-B_{\rm cheap}=R_c-R_i.
\]

已发布 OLS 点误差给出 \(E_c=\|QR_c\|_F^2\)、\(E_i=\|QR_i\|_F^2\)。由三角不等式：

\[
(\sqrt{E_c}-\sqrt{E_i})^2
\le\|Q\Delta B\|_F^2
\le(\sqrt{E_c}+\sqrt{E_i})^2.
\]

对每个真实站点先计算，再除以该站点独立路径的最终点误差：

| BF16 wide8192 站点 | 字典外底座变化的能量下界 | 上界 |
|---|---:|---:|
| 0392 high | 0.002639 | 1.373366 |
| 1013 high | 0.003832 | 1.281985 |
| 0392 low | **2.200052** | 7.731247 |
| 1013 low | **1.031664** | 4.807520 |

低噪声两源都给出有实质大小的字典外作用，而且保留这条独立路径后，投影残差明显减小。这个结果支持把原结构路径作为恢复机制的一部分继续研究。

该界约束当前 cheap_U 上构造的 D176、当前 H448 和线性输出族。它不把误差差额全归因于某个非线性项，也不对原始特征经任意非线性读出的信息量作结论。计算见[复算程序](recompute_joint39_round3.py)与[逐站点结果](joint39_round3_recomputed.json)。

## 7. 后续模块的联合作用在第二来源复现

| BF16 wide8192 站点 | 原生 FFN 后 attention 扰动能量下降 | head 对 FFN/attention 方向的相对能量增益 |
|---|---:|---:|
| 0392 high | 28.52% | 4.567 |
| 1013 high | 25.18% | 3.263 |
| 0392 low | 15.96% | 4.078 |
| 1013 low | 11.32% | 4.293 |

能量下降在同一5120维残差空间中计算。记 FFN 前后扰动为 \(\delta T,\delta B\)，有效 FFN 响应为 \(\delta F_{\rm eff}=\delta B-\delta T\)，则：

\[
\|\delta B\|^2-\|\delta T\|^2
=2\langle\delta T,\delta F_{\rm eff}\rangle+\|\delta F_{\rm eff}\|^2<0.
\]

故所测响应与原 attention 扰动具有反向内积。BF16 下的有效响应包含实际门控与舍入；这描述本批有限扰动，不是任意方向的全局收缩保证。

四臂定义 \(E_A=B_{10}-B_{00}\)、\(E_F=B_{01}-B_{00}\)、\(I=B_{11}-B_{10}-B_{01}+B_{00}\)。在 flow 空间，两背景共8站点均有：

\[
\langle E_A,E_F\rangle>0,\quad
\langle E_A,I\rangle<0,\quad
\langle E_F,I\rangle<0.
\]

有利的抵消方向具体来自组合计算产生的交互项。在 wide 低噪声，交互能量是 attention 单独误差的1.026倍和0.845倍，已经处于同一尺度。与此同时，总误差相对两个单独误差能量之和的比例在1013/low为1.01996，说明交互项自身能量也必须计入，不能只看负交叉项。

以上结果支持让真实后续模块参与恢复和评价。head 的增益比描述所测误差方向，不是全局谱范数；中间模块 MSE 的排序也不能直接替代最终任务作用的排序。[0392][r0392]、[1013][r1013]、[复算](joint39_round3_recomputed.json)

## 8. 向哪些计算图单元扩展，条件已经具体到什么程度

| 对象 | 可利用的原生教师入口 | 当前便宜信息/作用 | 关键闭合条件 |
|---|---|---|---|
| FFN | P64行原生 FFN | 同位置锚点、选中激活有限变化、已知输出方向 | 逐行结构；当前输入与锚点身份一致 |
| self-attention | P64 query × 全部31668 keys | 当前 selected attention、质量/矩关系、当前 value 方向 | 教师保留全键依赖；完整 K/V 等构建成本计入 |
| 固定文本 cross-attention | 少量原生视频 query × 固定文本 K/V | 当前 query 的条件读取变化 | 文本 K/V、mask、分支身份固定；有限步长另验 |
| self→cross→FFN 的单块 | self P64 教师继续经过原生 tail | 保留模块结构的当前便宜复合路径 | 后续算子在视频 query 轴上独立 |
| 多块/完整后缀 | 需共享全局当前状态或额外可更新摘要 | 任务相关状态及其传播 | 下一层 self 的 K/V 重新依赖全部 query，需建立新的闭合构造 |

当前模型内，单块教师链已经由真实代码和 P64 全键核验支持；它为更大恢复对象提供了可计算入口。该对象的低成本恢复质量还需具体算法和自身轨迹证明。[生产计算图分析][graph]

P64/31668约为0.2021%，只描述 attention core 的 query-key 交互比例。Q/K/V、输出投影、selected 统计、缓存、原生 tail、求解和写入都有实际成本。HHLL 仍有两个原生锚点时，即使两个 hit 的替换工作趋零，该部分四步周期的理想加速上界也是2倍；整网收益还取决于未替换计算。这是调用周期算术，不能替代端到端计时。

## 9. 下一步如何直接回答原问题

### 9.1 先闭合本轮的信息判断

使用现有封存资产运行 [diagnose_joint39_observability.py](diagnose_joint39_observability.py)，复用原 D176 的加载、哈希、P-only 归一化和字典构造。输出：

- \(E_{\rm full}\)、\(E_{Pspan}\)、实际 P64 误差及 FP32 cast 差额；
- \(D_P,D_H,D_HD_P^\top\) 的秩和奇异值；
- 采样不可见的函数维度、P→H 放大及正则有效自由度。

它不更换 P、不调 λ、不改变候选预测、不新增模型执行。H 标签只用于诊断。

若 \(E_{Pspan}\) 仍高于独立底座，当前读出空间需要改变；只在其中改系数不能达到目标。若它已明显低于底座，则把问题定位到可用空间内的观测/估计，随后应依据具体谱和结构决定恢复方式。该判别避免把完整 H oracle 的改善直接解释为某个在线方法的能力。

### 9.2 下一方法构造保留三类已经有依据的东西

第一，保留 attention 的逐头有限变化关系、当前 value 作用方向，以及 FFN 的实际当前输入、模型方向和原核。新的联合方法若替换这些部分，应给出对应结构怎样被重现或怎样被更强的任务先验替代。

第二，在同一原生 P 查询链上保留中间 attention/FFN 教师。它们可以约束模块恢复器；块出口/flow 教师用来确定哪些恢复误差真正影响任务。一个联合目标可以拥有多个中间观测接口，不必压成终端单一标签。

第三，把便宜复合路径通过真实 cross/FFN/head 后的作用作为评价对象，同时保留空间/时间差分。第39层建立了方便的完整输出观察点，后续全层候选还需自身轨迹与完整片段的人审，才回答动作、轮廓和纹理是否保留。

这定义的是下一构造应满足的计算和观测关系，并未把某个尚未实现的结构化联合优化器计为已验证方法。

### 9.3 当前候选决定

当前独立 attention + wide8192 路径继续作为参考。cheap→D176 和独立底座上的 D176 追加校正保留为机制实验；本轮没有形成新的自身轨迹候选。下一项最小证据是上述既有资产的可达空间诊断，此后才据其结果形成具有清楚结构约束的算法变更。

同 P 上再拟合剩余误差会重新利用已有教师，不会自动新增精确观测。在冻结线性读出的条件下，两次校正写作：

\[
\widehat R_H=[S_{HP}+T_{HP}(I-S_{PP})]R_P.
\]

覆盖前 B11_fixed 正确避免了 P 残差被覆盖为零，但剩余 P 误差仍受第一阶段拟合影响。该条件模型说明需区分训练残差与未观测残差；当前数据尚未把它独立识别为附加校正变差的原因。

## 10. 方法品味与新颖性应落在哪里

少量采样加子空间延拓已有降阶与经验插值传统。DEIM 的原始论文明确把降低状态维度与降低非线性求值成本区分开来；因此“低维/少量查询”本身不是充分的新颖性主张。[Chaturantabut–Sorensen 2010][deim]

当前值得形成具体算法与判别实验的主张是：在同一实时预算下，利用锚点和当前有限创新，保留 attention/FFN 的不同算子关系及中间教师约束，恢复它们共同的下游作用。其区别要由信息时点、监督接口、当前方向、实际输入、恢复算子与完整成本来证明。

这次 D176 结果的研究价值，正是把“特征相关”与“结构保持后的可识别恢复”分开。今后的新颖性比较应逐项比较这些过程与原理，而不是按同为加速或近似的目的归并方法。

## 11. 可复现资产与执行状态

- [完整统计复算](joint39_round3_recomputed.json)：逐来源结果、配对对照、有符号 Gram、结构作用范数界及源文件 SHA256。
- [统计复算程序](recompute_joint39_round3.py)：只依赖 Python 标准库，同目录放置原始 JSON 收据后直接运行。
- [实现语义审查](D176_IMPLEMENTATION_SEMANTICS.md)：具体输入、核、监督和函数族的差别。
- [P 可达空间诊断](diagnose_joint39_observability.py)：复用原 D176 producer 和封存资产，需要其已有 NumPy/Torch/safetensors 环境。
- [合成代数核验](observability_algebra_check.json)：明确为合成数据，不是模型结果。

在仓库 joint39 目录重算已发布统计：

    python recompute_joint39_round3.py

诊断脚本使用原 D176 producer 的五个实参，不使用虚构学校路径。以该 producer 原来能运行的环境执行，最后一个实参必须指向新的输出文件：

    python diagnose_joint39_observability.py REDUCED FIT EXPORT INDEX NEW_OUTPUT

REDUCED、FIT、EXPORT、INDEX 分别对应原 producer 的 reduced、fit、export、index。若源码未并排放置，通过 --implementation-dir 指定原 joint39 实现目录，并沿用原 producer 的依赖搜索路径。

当前已执行统计复算：发布摘要/对照的最大差为 \(4.44\times10^{-16}\)，8站点 Gram 闭合最大差为 \(4.55\times10^{-13}\)。已执行的合成检查验证了 full-H 与 P 可达下限的区别及正交分解，闭合误差约 \(4.4\times10^{-16}\)。真实 D176 新谱和 \(E_{Pspan}\) 仍待封存向量 CPU 重放；本文件未给它们填入推测数值。

[head]: https://github.com/Qesire/FFNDS/commit/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c
[report]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/D176_REPORT.md
[export]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/D176_EXPORT.json
[replay]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/D176_REPLAY.json
[jointcode]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/attention_joint39.py
[attentioncode]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/attention_panel_fit.py
[exportcode]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/attention_joint39_export_ffn.py
[d176code]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/attention_joint39_d176.py
[firstresults]: https://github.com/Qesire/FFNDS/blob/b1e15c2df4d1dad9a285798e552192ee4fca0bd8/research/20261124/attention_wide8192/results/README.md
[graph]: https://github.com/Qesire/FFNDS/blob/29639e148244e19752c0acbe40ed0928291a8263/research/20261124/attention_wide8192/results/JOINT_GRAPH_AUDIT.md
[r0392]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/RESULT_0392.json
[r1013]: https://github.com/Qesire/FFNDS/blob/0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c/research/20261124/attention_wide8192/results/joint39/RESULT_1013.json
[deim]: https://epubs.siam.org/doi/abs/10.1137/090766498
