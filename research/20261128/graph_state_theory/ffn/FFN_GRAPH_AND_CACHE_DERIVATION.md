# FFN 当前图的资源界：把算子因子化与充分缓存一起设计

2026-11-28。对象为冻结的 wide8192：N=31668、d=5120、m=13824、K=2560、P=64、模式数16、每专家40层，HHLL中每锚点仅对应一次命中。分析只读当前调用链、封存原编译器和既有证据；本轮无模型推理、GPU作业或计时。小尺寸操作数核验明确采用合成CPU数组，真实模型形状用于静态算量与容量。

## 1. 本轮最有价值的构造

**对 A0 因子化的同时，只保存旧输入在因子上的投影，可以同时减少命中计算和跨步历史状态。** 它避免旧截距折叠必须在每个anchor另算一次完整 UaA0 的代价。其可用秩必须由当前恢复器之后的剩余作用决定；两次小GEMM和建缓存的全周期盈亏点为 r<d/3，计入窄J投影后约 r<1701。

这条构造仍需真实 A0/输入谱来确定可用秩。本地现有归档没有完整A0或DeltaU张量，因此本文给出精确的作用误差、可行成本区间和静态数值，未声称 rank512/1024 保持了成片质量。

另外两项判断已经明确：

- 当前P64核读出采用N×64的因子化路径，是很经济的表示；直接展开成K×d输出权重会把这部分GEMM算量增加约26.72倍。
- 当前三缓存不是信息量下界。只存Ua便可通过重算原生anchor恢复Ya、Za，但要增加717.26TFLOP/片。应比较状态和计算的联合前沿，而非仅按缓存数量判定理论极限。

## 2. 真实代码所实现的对象

采用行批量记号。W∈R^(d×K)是selected W1，B∈R^(K×d)是selected W2，A=A0∈R^(d×d)，Z=zmap∈R^(K×16)，J=jmap∈R^(d×16)，V∈R^(16×d)。实数参考图为

\[
z_t=U_tW+b,
\quad\Delta h=\phi(z_t)-\phi(z_a),
\quad\Delta U=U_t-U_a,
\]

\[
b_t=Y_a+\Delta U A+\Delta hB,
\quad Q=\Delta hZ-\Delta UJ,
\quad X=(\Delta h-\Delta z\odot\beta_k)\oslash s.
\]

当前源代码实际执行 du_FP32、dub=BF16(du)、linear=BF16(dub@A0)，selected GELU差也另舍入至BF16；以下实数恒等与这些具体舍入路径分开陈述。

原生P64教师为Y_P，目标T=Y_P−Pb_t。令

\[
S=X_PX_P^\top/K+M^{-1},\quad
G=(Q_P^\top S^{-1}Q_P)\odot(VV^\top),
\]

\[
g(T)=G^{-1}\bigl[\langle(S^{-1}Q_P)_{:j},Tv_j^\top\rangle\bigr]_j,
\quad C(T)=S^{-1}\{T-Q_P\operatorname{diag}(g(T))V\}.
\]

全场恢复为

\[
\mathcal R_F(T)=Q\operatorname{diag}(g(T))V
+\tfrac12(XX_P^\top/K)C(T).
\]

P行最后覆写原生Y_P。固定当前U、Q、X、V和metric时，**整个16模式＋P64核GLS对T是线性的**；G可逆是已执行求解的条件。

## 3. A0旁路的正确误差对象

仅把底座中的A换成Ar，而Q/X保持既有定义。此时重新用原生P64拟合，新底座变化

\[
E_B=\Delta U(A_r-A).
\]

定义M_H为去掉P行的掩码。则完整近似FFN输出变化在实数图中严格为

\[
\boxed{\delta\widehat Y=M_H(I-\mathcal R_FP)\,[\Delta U(A_r-A)].}
\]

这里R_F作用于P×d矩阵，不能随意替换成一个与输出通道无关的N×P核，因为16个模式与V耦合。上式保留了真实GLS结构及0.5核系数。

因此合理目标是让因子化舍弃的作用经过当前P64恢复后仍小：

\[
\mathbb E\left\|M_H(I-\mathcal R_FP)[\Delta U(A_r-A)]\right\|_F^2.
\]

若再连接后续目标，可在这一实际误差上作用相应的后缀Jacobian/有限差分，而不直接用A0的权重Frobenius误差替代。Q/X随来源变化，期望保留这种联合依赖。

一个正面的精确性质是：若某误差恰为Q diag(c)V，GLS模式分支会将它从P完整恢复，核残差为零。故A0中能由现有模式解释的作用可以更大胆地交由当前观测恢复。这是作用等价的方向；是否足以给真实A0带来低秩压缩，要由其实际投影判断。

## 4. 算子和缓存联合因子化

令A_r=L_rR_r，L_r∈R^(d×r)、R_r∈R^(r×d)。anchor保存

\[
\boxed{Y_a,\quad z_a,\quad a_a=U_aL_r,\quad j_a=U_aJ.}
\]

hit执行

\[
b_t^{(r)}=Y_a+(U_tL_r-a_a)R_r+\Delta hB,
\quad Q=\Delta hZ-U_tJ+j_a.
\]

这正是上一节Ar底座的充分状态，旧Ua的5120列被r+16列取代。Za保留全部2560列，故GELU有限变化和原kernel特征的当前信息仍可计算。

### 容量与完整周期

用BF16保存Ya、Za与a_a，FP32保存j_a，40层槽分时复用：

\[
C_r=2LN(d+K+r)+4LN\cdot16.
\]

下表只描述静态候选；还未测其真实模型逼近误差。

| r | BF16投影＋FP32 J历史 | a_a也用FP32 | 额外anchor投影/站点 | hit线性路径/站点 | 全片净省TFLOP，含额外J建锚 |
|---|---:|---:|---:|---:|---:|
| 512 | 19.404 GiB | 20.612 GiB | 0.1660 | 0.3321 | **92.563** |
| 1024 | 20.612 GiB | 23.028 GiB | 0.3321 | 0.6641 | **52.715** |
| 1536 | 21.820 GiB | 25.444 GiB | 0.4981 | 0.9962 | **12.867** |
| 2048 | 23.028 GiB | 27.860 GiB | 0.6641 | 1.3283 | **−26.980** |

当前历史30.201GiB，hit的A0路径每站点1.660315TFLOP。候选完整一次anchor+hit需要6Ndr FLOP，另有2Nd·16的anchor J投影，因此盈亏点为

\[
6Ndr+2Nd\cdot16<2Nd^2
\iff r<(d-16)/3=1701.33\ldots.
\]

只计算hit会得到r<2560，这会漏掉当前短周期中的建锚费用。r2048正好提供了一个可由静态计算判断的例子：它确实省缓存、hit更便宜，但完整周期GEMM反而增加。

还有静态参数容量：两个专家共80份A0，BF16原值约3.906GiB；r512的L/R共0.781GiB，r1024共1.562GiB。只有运行时不继续保留原A0的GPU副本，这项减少才成立。

### 有限精度和P64

若a_a编码误差为ε_a，则线性底座新增−ε_aR_r；还要计入当前投影、第二个GEMM及旧dub/linear舍入差。可以将这些全部汇成实际测得的E_B，沿上一节同一恢复算子作用。

j_a的重排改变FP32求和次序。保留P64的原Ua仅需额外0.024414GiB，可使同一当前输入上的Q_P按旧路径计算；Za已保留，X_P不变。对于非P行Q的微小变化，若Q_P保持，g/C保持由新底座目标确定，额外输出项为δQ diag(g)V。此处原生Y_P仍重算和最终覆写，A_r的近似误差仍进入P残差，不把P底座偷偷换回旧A0。

## 5. 已有折叠与本轮构造的关系

11月25日的SUFFICIENT_STATE_DERIVATION已完整推导C_a=Y_a−U_aA、J_a=U_aJ及Za的充分状态，连同BF16差分缺陷、P保护和全周期成本。该旧构造约18.245GiB（含FP32 J与P原值），额外anchor工作约133.240TFLOP/片。它不是本轮的新发现。

本轮改变的是**先让未来线性作用通过r维坐标，再保存这些坐标**。如此anchor无需展开完整UaA，低维状态与低维GEMM由同一L_r生成。若r不足以保留恢复后作用，则回到完整A；如果r在约1701以下且误差适合当前恢复，便存在同时减少算量与状态的具体空间。

仅以信息充分性衡量，Ua本身决定Ya与Za；只存Ua为12.080GiB。hit按同一native核与形状重算完整anchor可恢复旧状态，但要额外717.256TFLOP/片。因此当前30.20GiB从来不等于信息理论下界；它是在省重算的条件下保存了多种昂贵函数值。

## 6. compiler中J的因子关系能够进一步整理表达

真实compile_sites.py构造

\[
J_*=W\operatorname{diag}(\beta_{\rm CAL})Z.
\]

于是未舍入图中

\[
Q=(\Delta h-\Delta z\odot\beta_{\rm CAL})Z.
\]

它使16模式也只依赖selected预激活的有限变化，未来不必读取UaJ。beta_CAL用于首轮浮点CAL拟合，kernel_beta来自BF16预激活/激活差的第二轮拟合，二者在源码中明确不同。不能把Q直接认作X·diag(scale)·Z；正确式还包含

\[
Q=X\operatorname{diag}(s)Z+
\Delta z\operatorname{diag}(\beta_k-\beta_{\rm CAL})Z.
\]

针对当前冻结J和已舍入Za/Zt，设E=diag(beta_CAL)Z，则用Delta z替代Delta U的数值差为

\[
Q_{z}-Q_{U}
=\Delta U(J-WE)-(\Delta z-\Delta UW)E.
\]

这分别定位了编译因子化舍入和预激活舍入。保留J_a只需0.0755GiB，因此应将这项更简洁的表示视作窄映射的整理，而非主要显存或加速来源。

## 7. 当前核读出已有良好收缩顺序，跨步摘要还受未来查询约束

核输出0.5(XX_P^T/K)C按现有顺序需要2NKP+2NPd FLOP。若先展开X_P^TC再乘X，则需2KPd+2NKd。真实形状下后者为前者的**26.72倍**。当前N×80联合读出已有效利用64个probe和16个模式，不宜因追求“一个矩阵”而展开为K×d。

跨步压缩则不同。定义f_b(z)为BF16 GELU LUT值，ξ(z)=(f_b(z)−beta_k z)/s。当前lookup实际满足

\[
X=\xi(z_t)-\xi(z_a)+e_{\Delta h}/s,
\quad e_{\Delta h}=Q_{BF16}(f_b(z_t)-f_b(z_a))-(f_b(z_t)-f_b(z_a)).
\]

因此固定anchor端点ξ(z_a)的可分性属于忽略额外差分舍入的参考图；现有逐位图还有这项依赖当前/anchor共同值的缺陷。

对该参考图，可以研究未来probe查询子空间。若未来要读Phi_a T_t，编码Phi_aE能够精确回答全部查询当且仅当所有T_t的列空间包含在range(E)内。给定正交E，误差为

\[
-\Phi_a(I-EE^\top)T_t.
\]

这里T_t可以包括X_P^TC/2K，因此应关注实际未来输出读出，而非只看anchor自身能量。一轮查询的秩至多64，不意味着整个未来查询族只有64维；原K=2560个坐标可能都被不同查询区分。CPU构造验证：两个anchor的压缩码相同，仍可在后续另一查询上产生不同答案。

这一结论给出合理的进一步简化方向和准确条件，但当前归档缺少该未来查询族谱，本文不据此选择压缩维数。

## 8. 16个模式提供一个精确的底座等价类

在固定Q、X、J、Z、V和原生P64教师的实数图中，对任意固定c∈R^16，定义

\[
A_c=A_0+J\operatorname{diag}(c)V,
\qquad B_c=B-Z\operatorname{diag}(c)V.
\]

这里仅改近似分支的A0与selected W2K，原生教师仍调用原模型W2，Q与X也沿用原定义。于是

\[
b_c=b-Q\operatorname{diag}(c)V,
\qquad T_c=T+Q_P\operatorname{diag}(c)V.
\]

由GLS模式复现性质，

\[
\boxed{g_c=g+c,\qquad C_c=C,\qquad\widehat Y_c=\widehat Y.}
\]

因此模式基线的某些成分可以在固定底座与在线gain之间精确转移。相应的有效线性作用

\[
A_c-J\operatorname{diag}(g_c)V=A_0-J\operatorname{diag}(g)V,
\]

\[
B_c+Z\operatorname{diag}(g_c)V=B+Z\operatorname{diag}(g)V
\]

保持不变。这是明确的表示自由度，保留全部P64拟合、kernel和最终输出。若c选为本次−g，形式上可把模式gain移入当前A/B；但在P拟合后动态生成大矩阵会增加工作，因此该规范选择本身不被计作提速。

更有价值的是在静态近似前选择c，然后以A_r逼近A_c，同时保留配对B_c。完整输出差严格为

\[
\delta\widehat Y
=M_H(I-\mathcal R_FP)
\left[\Delta U\left(A_r-A_0-J\operatorname{diag}(c)V\right)\right].
\]

由此可以研究

\[
\min_{c,\,\operatorname{rank}(A_r)\le r}
\mathbb E\left\|M_H(I-\mathcal R_FP)
[\Delta U(A_r-A_0-J\operatorname{diag}(c)V)]\right\|_F^2,
\]

并使用第4节的投影缓存与完整周期预算。这将普通矩阵逼近变为对当前恢复器等价类的作用逼近。

该自由度只有16个标量参数，更新矩阵秩至多16；由rank不等式，rank(A_c)≥rank(A0)−16。因此若A0满秩5120，这个变换本身最多将精确秩降到5104，不能据此声称获得rank512的精确表示。其价值是把可精确转移的成分明确移出逼近目标，剩余压缩依赖真实作用结构。

有限精度下，改变B_c只限近似分支t["w2k"]，不能覆盖phase.ffn_2的原生teacher权重。BF16存储A_c/B_c会引入误差；而且实际base使用dub=BF16(DeltaU)，Q使用FP32 DeltaU。即使权重不舍入，基本额外缺陷已包含

\[
(\operatorname{BF16}(\Delta U)-\Delta U)J\operatorname{diag}(c)V,
\]

加上两条A0 GEMM输出舍入之差、权重重编码与累加顺序项。可以将这些合并为实际底座偏差e_g；固定Q/X时，最终增加M_H(I−R_FP)e_g。CPU例子分别核验实数恒等和BF16矩阵重编码缺陷，没有将其称为现有执行的逐位等价。

## 9. 验证、来源与交付范围

[verify_ffn_graph_rewrites.py](verify_ffn_graph_rewrites.py)实际执行了16项CPU恒等核验：旧折叠、J因子式、联合因子缓存、GLS线性、因子化误差经恢复传播、模式误差完全恢复、BF16端点/因子缺陷和查询子空间误差。并追加配对模式规范变换、gain平移、kernel系数不变、规范后的因子化商误差及BF16权重缺陷，最大绝对误差为**1.33e−15**。结果与真实形状预算在[FFN_GRAPH_REWRITES_VERIFIED.json](FFN_GRAPH_REWRITES_VERIFIED.json)。

本轮另核对当前仓库的[fourstep_direct_hit.py](https://github.com/Qesire/FFNDS/blob/9bf7eceba0b5af483625886e8965351110b9d125/tools/ffnds_fourstep/fourstep_direct_hit.py)。仓库已存在合并dub/dh输入与A0/W2K权重的直接GEMM候选；它删除了原A0中间BF16舍入，源码也明确称其为数值候选。因此“将两个base GEMM合成一个”不是本轮新算法，且其乘加总数本身不变。

数学工具来自operator-tensor-analysis的任务读出状态证书与真实周期收缩成本。完整源码字节SHA-256记在JSON。未修改生产方法、未提交新实验、未声称当前数据已证明任意低秩候选有效。
