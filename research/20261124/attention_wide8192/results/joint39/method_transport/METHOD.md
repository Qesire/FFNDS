# 已形成的方法原型：当前交互、标准化缺陷运输与探针恢复

2026-11-25。依据 FFNDS 研究提交 `7c8196fa5380dbe67dd9772f942553085fde55b3`。本轮继续回答“现在形成了新方法吗？继续研究”。

**已经形成一个方法原型：输入、锚点状态、当前运算、教师、恢复规则、零尺度分支和输出都已闭合，并有独立可运行的CPU参考。** 相对上一轮，核心推进是由attention对V的作用规律决定历史缺陷怎样更新：保存标准化缺陷，使用当前V的逐通道尺度还原，P64恢复剩余的标准化创新。

这使恢复目标从任意输出差，转为路由缺陷变化与标准化value形状变化。当前均值与尺度直接进入输出。主候选是逐通道标准化版本；一般通道混合运输作为已推导的推广，不加入本轮默认参考路径。

六源成片基本相当和已有计时属于冻结C；本候选的新增证据是数学定理、源码依赖核查与明确合成激活上的数值执行。完整Q/K/V真实激活没有随当前replay归档提供，本轮没有新Wan成片或GPU运行。

## 1. 与已有方法的关系已进一步查清

本轮直接阅读原论文或作者源码，把上轮构件放回准确的位置：

| 已有机制 | 取得的原始依据 | 本方法继承与改变的部分 |
|---|---|---|
| 当前遗漏组centroid补偿 | ClusterAttention实际`backend→sage_query_mean_compensation` | 给定相同分组/支持，当前G的理想实数算子对应；本方法发展它剩余缺陷的状态和更新 |
| 完整dense−sparse残差缓存 | MC-Sparse的`_full_cal`与`reuse` | 提供完整历史场；本方法根据当前V运输历史作用，再用当前教师恢复创新 |
| 当前P64仿射残差拟合 | SparsePR v2 §4.2及附录A/B | 继承真实行观测与回归机制；目标改为标准化缺陷的有限变化，特征同时读取当前query、遗漏补偿响应和质量 |
| 少量分量恢复全场 | DEIM/gappy POD原文 | 提供经验恢复与稳定性工具；可恢复对象由当前底座、历史状态和观测共同决定 |
| 通道统计匹配 | AdaIN原文 | 尺度匹配是已有运算；这里依据R1=0，用输入V统计运输输出缺陷，并保留当前输出单位 |

当前可提出的具体方法主张是**由V等变性规定的缺陷状态运输与当步观测更新**。在已核验的ClusterAttention、MC-Sparse和SparsePR主路径中未见这一特定过程；OnlineCache学习式corrector的全文仍未取得，作为查新开放项。尚未据此宣称完整领域首创。[机制逐项比较与固定来源](MECHANISM_NOVELTY.md)

## 2. 从零质量作用推导状态

每层每头的有效操作数Q、K已包含实际norm/RoPE，V为当前value。query轴i、key轴j、value通道ν。令S_s和S_s⁰分别表示完整attention与当前分组底座在原key轴上的概率矩阵：

\[
A_s=S_sV_s,\quad G_s=S_s^0V_s,\quad
R_s=S_s-S_s^0,\quad E_s=R_sV_s.
\]

两种概率矩阵均逐行归一化，故R_s1=0。这给出两条精确关系：共同value平移在缺陷中抵消；缺陷的有限变化分成路由变化和值变化：

\[
E_s-E_a=(R_s-R_a)V_s+R_a(V_s-V_a).
\]

当前值变化中的一些作用可以直接读取。令μ_s、σ_s为V在key轴上的逐通道均值、标准差，D_s=diag(σ_s)，并令D_s⁺对零标准差取0：

\[
\widetilde V_s=(V_s-\mathbf1\mu_s)D_s^+,\qquad
C_s=E_sD_s^+=R_s\widetilde V_s.
\]

零标准差列的V为常量，缺陷列严格为0，因此总有E_s=C_sD_s。**锚点保存完整C_a；当前使用G_s+C_aD_s。** D_a只参与构建，可随后释放。

P64的剩余目标变成

\[
\boxed{\Delta C_s=(R_s-R_a)\widetilde V_s+
R_a(\widetilde V_s-\widetilde V_a).}
\]

第一项是路由缺陷变化，第二项是标准化value形状变化。均值由当前正权底座G_s携带，通道尺度由当前D_s解码，两者都被写回真实输出。

## 3. 完整算法与执行接口

### 3.1 当前底座

沿用当前128-key blocks与每个query block选31个key blocks的支持。每个原key block以归档P_k第一列对中心化K作符号二分；空组跳过，末block按52个有效keys计数。anchor冻结成员标签与selected支持，hit用当前K/V更新组均值。

selected块逐key精算，得到B_i和Z_i^S。遗漏组c以数量n_c、均值k̄_c/v̄_c产生正权响应：

\[
G_i=\frac{Z_i^SB_i+\sum_{c\notin S_i}n_ce^{q_i^\top\bar k_c/\sqrt d}\bar v_c}
{Z_i^S+\sum_{c\notin S_i}n_ce^{q_i^\top\bar k_c/\sqrt d}}.
\]

selected块的摘要被排除。实现使用LSE分块合并，保留真实softmax归一化，不物化全部coarse scores。

### 3.2 锚点与命中

anchor取得原生A_a，计算G_a、D_a，保存C_a=(A_a−G_a)D_a⁺及分组标签/支持，原生输出继续进入后续块。

hit计算G_s、D_s，并构造32列当前特征：

\[
\Psi_i=[1,\ q_iP_{k,15}/\sqrt d,\
((G_i-B_i)D_s^+)P_{v,15},\ Z_i^S/Z_i^0].
\]

归一化在value投影之前完成。对原有64个P位置执行完整attention行（每行读取全K/V），先形成标签

\[
T_P=(A_{s,P}-G_{s,P})D_s^+-C_{a,P}.
\]

固定λ=0.1，使用实际α=64λ=6.4：

\[
W_s=(\Psi_P^\top\Psi_P+6.4I)^{-1}\Psi_P^\top T_P,
\]

\[
\boxed{\widehat A_s=G_s+(C_a+\Psi_sW_s)D_s.}
\]

最后将P行覆盖原生教师。当前σ=0的列直接返回G，相应历史项与系数可置0。anchor的常量V列缓存定义为0。无锚调用取C_a=0。

实现入口为`build_anchor`与`recover_hit`，在 [reference_attention.py](reference_attention.py)。生产绑定对象是四步HHLL中0/2原生anchor、1/3命中；FFN接现有BF16 wide8192，共用原P64与40层活动槽生命周期。

## 4. 两个有区分力的定理

### 4.1 新增的精确有限演化类

若R_s=R_a且V_s=V_aΓ+1b，其中Γ为正对角，则D_s=D_aΓ、标准化V保持，C_s=C_a。因此G_s+C_aD_s=A_s。

该条件允许各通道大幅、不同幅度的正增益和偏置；原raw缺陷底座在同一情形的误差是E_a(I−Γ)。固定selected索引本身不等于R固定。有效Q/K的对偶变换Q_s=Q_aL^{-T}、K_s=K_aL+1κ在冻结分组/支持下是保持两种概率矩阵的一种充分构造。

E_s恒定与C_s恒定是两种不同的作用假设，新旧算法的全部精确输入集合一般不相互包含。此处建立的是一个具体新增的精确响应能力。

### 4.2 完整算法的两时刻独立正仿射等变

分别对两个时刻施加V'_r=V_rΓ_r+1b_r，r∈{a,s}，Γ_a和Γ_s可以不同。每次变换前后保持其有效Q/K、partition/support与P，允许Q/K在anchor和hit之间变化。

标准化C_a、C_s与当前Ψ分别不变，因此标签、ridge系数均不变，最终

\[
\boxed{\widehat A'_s=\widehat A_s\Gamma_s+\mathbf1b_s.}
\]

这在非零创新、非零近似误差时同样成立。该性质属于attention value子图的等变关系，最终解码保留当前幅度和均值，真实W_O及FFN收到对应变化。它不把完整Wan输出视为对value变化不敏感。

数学定义采用零分支。参考实现的非零尺度地板为1e−12；上述严格等变限定为地板未激活，生产浮点/缓存另有误差项。

## 5. 误差和联合计算图

若标准化cache为C_a+ε_C、当前特征固定，令K_s=Ψ_s(Ψ_PᵀΨ_P+αI)^{-1}Ψ_Pᵀ、M_H=I−PᵀP，最终误差为

\[
\boxed{\widehat A_s-A_s=M_H(K_sP-I)(\Delta C_s-\varepsilon_C)D_s.}
\]

它把剩余创新、观测传播、缓存舍入和当前输出尺度联系起来。P64是否充分取决于ΔC的允许空间与P限制在该空间上的稳定性；新坐标本身不替代这一条件。

还可把运输本身的贡献完全分离。两条路径共用同一个当前G、Ψ、P教师、ridge和最终P覆盖，仅把历史项从E_a换成C_aD_s，则

\[
\boxed{\widehat A_{\rm transport}-\widehat A_{\rm raw,same\ feature}
=M_H(I-K_sP)(C_aD_s-E_a).}
\]

这个式子在含P恢复的完整流程中隔离了历史更新。参考接口用同一个标准化state，并通过`transported_anchor_defect=E_a`构造未运输对照，两个当前特征和教师数组逐位相同。

实际源码还给出一个组合性质：固定当前块输入、时间/文本条件与FFN缓存，attention与FFN共用P，attention在P写入相同原生值后，逐行的projection/gate/cross/LN使FFN的P输入相同。当前wide8192的gain、coeff、xp只读取这些P输入、P缓存及固定参数，所以它们也相同。非P误差继续通过真实非线性尾部传播，当前FFN共享拟合则受到P链保护。

该条件适用于一块内给定状态；后续self-attention会重新混合位置，实际不同kernel的P舍入差也需单列。详见 [共享探针的组合推导](PROBE_CLOSED_COMPOSITION.md)。它解释了为何可以保持模块边界和原生教师来研究联合作用。完整块缺陷是另一种目标，需要未覆盖cheap shadow尾部，不能混用其教师标签。

## 6. 已执行的计算与真实资产范围

数学参考使用NumPy float64，真实P_k/P_v字节与明确合成Q/K/V；主体为H2、N190、d128、P64。独立运输见证为96 query、768 key、d128。

| 本轮核验 | 结果 |
|---|---:|
| 同一anchor输入回返 | 最大误差2.78e−17 |
| 固定路由、正对角V演化，raw未拟合底座留出L2 | 4.00535 |
| 同例标准化底座与最终输出 | 最大误差1.11e−15 |
| 两时刻采用不同增益/偏置、且有非零routing创新 | 最终等变误差2.66e−15 |
| 上项独立实现复核 | 最终等变误差4.00e−15 |
| 标准化缓存有限误差式 | 最大差8.33e−17 |
| 固定同一特征/教师，仅改变运输，最终差分恒等式 | 最大差2.36e−16 |
| 同特征正对角演化，未运输/运输后的留出L2 | 3.97776 / 2.38e−14 |
| 同特征一般随机有限变化，未运输/运输后的留出L2 | 7.33626 / 7.37903 |
| P教师接口 | 每次64 query、全部190 key |
| 合成联合块中覆盖P后的FFN输入/读出/P出口差 | 0 |

raw与normalized的主接口还改变了特征坐标。表中raw未拟合底座对照与另行执行的同特征对照，都直接隔离了状态运输；直接比较两个主模式的拟合结果则仍包含特征差异。一般随机变化见证中，运输后误差约增加0.58%；它具体限定了收益所依赖的演化结构。正对角演化的精确恢复由定理保障，真实轨迹上的相应结构需要实际输入检验。

真实replay归档16个文件仅含投影、CAL变换、索引、见证和metrics，没有完整anchor/hit Q/K/V。学校历史轻面板约10.15GB的存在记录没有替代其输入字节。当前程序可直接接收真实有效Q/K/V、固定支持、P及原生anchor输出；需要这些匹配数据才能给出本候选的Wan回放结论。[实现记录与资产清点](REFERENCE_METHOD_IMPLEMENTATION.md)

## 7. 成本与更一般的运输

主候选的C_a直接替换上一轮E_a，不增加完整历史字段，也无需保留D_a。按真实N31668、H40、d128，40层BF16缺陷与uint8分组标签合计12.12757GiB；已有selected支持另计，与上轮相同。当前通道均值/方差需要实际归约，每层decode有162,140,160个乘法。可与组统计/读出共享读取或融合，实际融合和时延尚待实现。anchor计算G_a的成本同样保留。

一般通道混合也已推进到可识别作用。若V_s=V_aT+1b+r，则

\[
E_s=E_aT+(R_s-R_a)V_s+R_ar.
\]

固定组的anchor/current value均值提供Y_s≈XT。由Y_s唯一识别目标E_aT的必要充分条件为

\[
\boxed{\ker X\subseteq\ker E_a.}
\]

它允许T本身不唯一，但实际需要的作用唯一。可用T̂=I−X⁺X+X⁺Y_s。固定anchor的X、E_a时，对当前组观测Y_s的扰动，目标作用敏感度由||E_aX⁺||决定。合成秩亏例中T仍相差12.20656，E_aT作用误差为3.55e−15。

完整T版本在40层额外保存X⁺约0.37842GiB，秩亏完成矩阵约0.09766GiB；当前拟合与应用约为full attention QK/PV成对MAC的0.20526%。它还有3200次anchor小分解及完整字段读取/写回。作为一般化保留，主参考仍采用无需新矩阵求解的逐通道规则。[完整运输推导与成本](DEFECT_TRANSPORT_DERIVATION.md)

## 8. 当前判断与复算入口

本轮已经从“组合框架”推进到**有明确状态语义、更新规律、精确作用类、整套等变性质和可运行实现的方法原型**。其最有区分力的科学问题是：标准化缺陷的路由/形状创新，在真实四步轨迹上是否更稳定、更容易由P64恢复。六源C作为已有人审参照，新方法保留单独的执行证据。

真实回放须在相同输出单位比较：原始创新J_raw=E_s−E_a，运输后创新J_transport=(C_s−C_a)D_s=E_s−C_aD_s。不能直接以||ΔC||比||ΔE||小宣称收益，因为二者单位不同。同特征对照使用同一个恢复算子，再经过实际W_O、gate与FFN尾部读出；这同时检查历史运输的作用与联合模块对剩余误差的响应。

```bash
OPENBLAS_NUM_THREADS=1 python verify_reference_attention.py --replay-zip ../../replay_evidence.zip
OPENBLAS_NUM_THREADS=1 python verify_defect_transport.py --basis /path/to/P_k.npy
OPENBLAS_NUM_THREADS=1 python verify_probe_composition.py
```

只执行CPU数学参考。若从仓库目录调用，`replay_evidence.zip`是上两层results中的既有归档；第二条也可省略basis使用固定合成方向，输出会记录实际来源。完整输入输出约定在 [METHOD_SNAPSHOT.json](METHOD_SNAPSHOT.json)。
