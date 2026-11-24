# 从固定 P64 诊断转向完整方法：attention、FFN 与生成闭环

日期：2026-11-24。最新实测固定于 [FFNDS@cc01675282db463fc8983f9113ea3a4ff8c309ef][evidence]。分析采用 evidence-analysis、operator-tensor-analysis 与 active-abstraction，实际使用受限采样恢复、正交投影、有限变化、计算图依赖和张量收缩工具。

本轮独立复算了发布的八站点完整谱和64组读出记录；另用真实投影基、真实模型维数和合成 Q/K/V 验证在线统计的代数重排。新增真实模型/GPU执行数为0。以下分别标明发布实测、数学构造与后续验证。

## 1. 整体判断与推进对象

**已有证据支持把现成的结构化 attention 路径接入 wide8192 的全层、四步自身轨迹。** 当前所谓“独立读出”就是一个具体扩展候选：high OUT-ABS、low FIN-C0，串联原 wide8192 FFN。它已经使用当前原生 P64 attention 教师、逐头恢复、真实 cross 和当前输入上的 FFN 恢复。

D176 探索的是怎样进一步合并恢复对象。最新诊断已经回答其完整表示空间、固定 P 读出空间和实际系数之间的差距。接下来应让已有结构路径接受完整生成质量与成本的检验；D176 的新增字典和正则搜索可在此收束。

从这次计算依赖审查，还得到一个更直接增加可替换计算的后续构造：high OUT-ABS 不依赖完整 attention 锚点输出，因此 attention 可以采用自己的锚点时序，而 FFN 保留自身所需的锚点。它属于清楚分开的新调度，先完成原有候选在线闭环，再检验这一步扩展。

## 2. 最新结果已经明确的三层关系

### 2.1 当前 P 诱导空间含有有用作用，实际读出尚未选中它

下表来自新实测；每个站点均除以同站点原独立恢复的 H448 flow 点平方误差。P/H 分别为64个观测位置与448个留出位置，输出是64维 flow patch。

| wide8192 站点 | 完整 D176 下限 | 当前 P 诱导空间下限 | 实际 P64 读出 | P 空间内目标能量占原误差 |
|---|---:|---:|---:|---:|
| 0392 high | 0.3139 | 0.6168 | 1.1303 | 38.32% |
| 1013 high | 0.2864 | 0.5992 | 1.1513 | 40.08% |
| 0392 low | 0.4207 | 0.6927 | 1.1708 | 30.73% |
| 1013 low | 0.3463 | 0.6503 | 1.2311 | 34.97% |

这项结果给出了积极而具体的判断：保留结构路径以后，当前固定 P 诱导的函数空间已经与约31%–40%的目标残差能量相交。该投影使用 H 标签来识别，在线 P 标签能否选中相同作用仍由观测接口和模型关系决定。[学校报告][report]、[独立复算](observability_recomputed.json)

当前校正到有用投影的平方距离，为该投影能量的1.34–1.66倍。因此当前实际系数离有用投影比零校正更远。这个事实定位了恢复模型与教师的匹配问题；已有标量无法进一步确定其是否同向过冲，不能把该差距全部归于某一个正则参数。

cheap 底座的 low 则具有不同的负担：两源完整 D176 下限为4.5449/2.5733，P 空间下限为7.0954/4.2118。原独立路径已经完成了一部分当前 cheap→D176 函数族难以重建的作用。因此，保留按头恢复、当前方向和 FFN 原核有直接的结果依据。

### 2.2 扩大字典改变了 P 到其他位置的延拓几何

以下为两来源的误差比几何平均。

| 独立底座上的字典 | high 完整下限 | high P 空间下限 | low 完整下限 | low P 空间下限 |
|---|---:|---:|---:|---:|
| FFN 核112 | 0.4393 | 0.6044 | 0.5138 | 0.6684 |
| D176 | 0.2998 | 0.6080 | 0.3817 | 0.6712 |

加入头间对比使完整下限降低31.74%/25.71%，当前 P 空间下限却提高0.5846%/0.4181%；四个 wide 站点方向一致。实际误差只再降低0.3534%/0.0560%。

原因可以从当前算法直接看出。若增加的特征块为 E，则代表核从 D Dᵀ 变为 D Dᵀ+E Eᵀ。完整 H 列空间包含原空间，但64列的 H–P 代表矩阵由 D_H D_Pᵀ 变为 D_H D_Pᵀ+E_H E_Pᵀ，其列空间不必包含原空间。新增相关特征同时改变了教师如何延拓到其他 query。

因此，结构方向的价值应与其进入恢复器的方式一起评价。逐头身份、当前 value 方向和原 FFN 直接核，正是已有路径保留的恢复约束。

### 2.3 谱把问题定位到恢复关系，而非浮点精度

D176在全部八站点实测：

\[
\operatorname{rank}D_P=64,\quad
\operatorname{rank}D_H=176,\quad
\operatorname{rank}(D_HD_P^\top)=64,\quad
\operatorname{rank}(D_H\ker D_P)=112.
\]

这112个在 P 上无法区分的系数方向，在 H 上确实对应不同函数。当前代表空间与这些方向的像一般是斜补关系；不能把“字典到 P 空间”的正交投影差直接称作112个不可观测方向的能量。

按每query RMS统一 P64/H448 的查询数量，D176 的 P→H 算子增益为1.48–2.03；报告中的总范数3.91–5.38包含 H 行数是 P 七倍的尺度。有效自由度41.65–47.01描述当前岭的谱收缩，目标在这些方向上的分量仍需通过恢复关系理解。

本轮复算中，原评分逐项完全一致，三项能量独立算术闭合差最大6.57e−14，FP32输出舍入的误差变化最多为独立基准的2.06e−7。数值求解精度已足以服务当前判断。[详细数值辨析](NUMERIC_GEOMETRY_ANALYSIS.md)

## 3. 信息充分性应如何严格表述

### 3.1 固定便宜表示，研究受限恢复

原生模型是确定性的；实际问题带有计算预算。应先固定在线已经计算的特征、锚点和允许的恢复族，再考察稀疏教师能区分哪些未计算作用。

设允许变化用参数 θ 表示，P 教师为 M_P θ，目标作用为 T_H θ。线性条件下，目标由教师唯一确定的充要条件是：

\[
\ker M_P\subseteq\ker T_H.
\]

稳定恢复还要求可见的目标相关方向具有合适的最小奇异值。一般非线性模型则使用同一条件的纤维形式：所有给出相同教师观测的允许状态，都必须给出相同目标作用。

对当前完整 D176 族，M_P=D_P、T_H=D_H，上述全族唯一性条件不成立。这精确限定了当前模型。对于更有结构、更小或具有不同度量的函数族，可以重新建立该条件；当前 P 代表空间内的实际目标投影也已由实测给出。

### 3.2 “可达空间中有目标”与“P 教师能估计目标”之间的确切关系

令 Z_H=D_H D_Pᵀ、G=D_P D_Pᵀ，二者分别为448×64和64×64。α 的形状为64×64，后一个64是flow输出通道。记

\[
\alpha_H=\arg\min_\alpha\|r_H-Z_H\alpha\|_F^2,
\qquad
\alpha_\lambda=(G+\tau I)^{-1}r_P,
\quad\tau=64\lambda=6.4.
\]

定义 P 教师相对 H 最佳函数的兼容性差：

\[
\eta_P=r_P-G\alpha_H.
\]

直接代入得到精确等式：

\[
\boxed{\alpha_\lambda-\alpha_H
=(G+\tau I)^{-1}(\eta_P-\tau\alpha_H).}
\]

空间内误差因此同时取决于同一个有限模型能否解释 P/H，以及正则如何改变它，二者还存在交叉项。一个充分条件也很清楚：残差在所有相关位置属于同一代表函数族、P标签精确且τ=0时，满秩G允许P教师确定正确系数。

这使下一构造的目的具体化：通过模块教师、有限变化和模型方向，让同一组恢复参数在观测位置与未观测位置保持有意义的作用关系。

### 3.3 同一组 P64 已经提供了多个模块的原生教师

| 路径 | 已使用的 P 教师 | 恢复约束 |
|---|---|---|
| attention | 40头各64×128的原生输出 | 每头32维设计；FIN另含当前方向耦合 |
| wide8192 FFN | 64×5120原生输出 | 16个已知模型方向、直接激活核及探针度量 |
| D176终端读出 | 64×64原生flow | 176列设计的统一岭延拓 |

这里的张量尺寸描述实际观测接口，不作为互信息数量。对逐头输出模型，每头拥有自己的参数；同一64个query可以同时约束40个恢复问题。FIN-C0的充分性还涉及耦合方向的Schur算子，不能仅凭32列的秩保证。

终端flow是这些中间响应经后续计算后的任务观测。保留中间教师，有机会区分在P终端输出中抵消、在其他位置又产生不同作用的参数变化。因此，完整块适合评价共同效果，原生中间教师适合约束恢复过程。

## 4. 注意力与 FFN 的联合作用怎样进入方法

### 4.1 attention 的有限变化保留了什么结构

对一个query/head，令当前 selected 输出为 O_s^S，省略keys的完整概率质量为ρ_s，省略部分对完整输出的加权贡献为u_s。则

\[
A_s=(1-\rho_s)O_s^S+u_s.
\]

锚点a给出ρ_a、u_a。使用当前selected输出的底座为 b_s=(1−ρ_a)O_s^S+u_a。固定锚点参考v_*，定义w=u−ρv_*、c_s=O_s^S−v_*，得到有限恒等式

\[
\boxed{A_s-b_s=\Delta w-\Delta\rho\,c_s.}
\]

真实FIN-C0保留这一乘性关系：

\[
\widehat A_{ih}=b_{ih}+X_{ih}B_h-(X_{ih}a_h)c_{ih}.
\]

它让质量变化沿当前value方向作用。原实测的全键有限割线恒等式误差达到9.68e−14，而完整anchor→hit的一阶切线相对误差为115.39%。有限变化因此是有实测根据的迁移对象；当前廉价特征与P求解决定其可实现精度。[初轮结果][initial]

### 4.2 后续 FFN 响应必须属于评价对象

在固定当前入块状态下，把attention后的残差/cross合成为T(A)，FFN输入记为U(A)，FFN门控为G_F。完整块为

\[
B(A)=T(A)+G_FF(U(A)).
\]

候选attention为Ahat，FFN自身近似误差为 e_F(Uhat)=Fhat(Uhat)−F(Uhat)。在实数计算图中精确地，

\[
\widehat B-B^*
=\underbrace{T(\widehat A)-T(A^*)}_{\text{attention直接作用}}
+G_F\underbrace{[F(\widehat U)-F(U^*)]}_{\text{真实FFN响应}}
+G_Fe_F(\widehat U).
\]

这三个向量的内积决定它们怎样叠加。attention单独的平方误差不能代替完整块误差；两个模块的误差也不能简单按能量相加。实际BF16门控与残差写回需另计舍入项，或按有效写入定义对应响应。已有四臂分叉对应这些有限作用关系。全层在线验证将进一步纳入下一层self-attention读取全部候选状态的反馈。

### 4.3 一个已有实现语义的澄清：B11_fixed可以对应标准在线P覆盖

在线先把attention的P行覆盖为当前原生A_P*。本块cross的文本K/V固定，LN/AdaLN与FFN按query作用，故

\[
U_P^{online}=U_P^*,\qquad
U_H^{online}=U_H^{candidate}.
\]

wideFFN因此在原生P输入和原生P标签上拟合，而在H使用候选attention后的实际输入。这与现有分叉代码的B11_fixed一致。在固定同一状态、缓存与算术定义的条件下，H函数相同；512行回放与全31,668行GPU执行的batch/kernel舍入仍须按既有回放容差验证，尚无两者逐位一致的在线结果。

因此，下一实现可直接沿用这条P覆盖语义。一份原生P attention→cross→FFN链即可提供现有两模块教师；无需第二次原生P FFN。此前B11_refit使用未覆盖的候选P输入，属于另一个条件反事实。[接口与推导](ONLINE_STRUCTURED_ATTENTION.md)

本块P的“原生”以候选当前入块状态为条件。下一层self-attention会读取近似H的K/V，所以它不意味着跨层P一直等于Dense原始轨迹。

## 5. 完整四步轨迹是下一层决定性证据

真实调度器用有符号步长 a_t=σ_{t+1}−σ_t：

\[
x_{t+1}=x_t+a_tv_t(x_t),\qquad
\sigma=(1,15/16,5/6,5/8,0).
\]

令候选状态误差e_t=xhat_t−x_t，在候选自身状态和缓存s_t上定义d_t=vhat_t(xhat_t,s_t)−v_t(xhat_t)。实数域内精确有

\[
e_{t+1}=e_t+a_t[v_t(\widehat x_t)-v_t(x_t)]+a_td_t.
\]

若v在连接线段上可微，则有限变化还可写成

\[
e_{t+1}=(I+a_t\overline J_t)e_t+a_td_t,
\qquad
\overline J_t=\int_0^1Dv_t(x_t+ue_t)\,du.
\]

这条关系把当前近似、后续状态和新锚点接到同一闭环。high/low命中步的|a_t|分别为5/48和5/8，同一flow误差的直接更新幅度相差6倍；high误差又会通过中间状态改变low输入，两个作用必须同时看。

当前12条采集背景视频都保留原生attention。wide背景只替换了FFN。新增attention候选的全层自身轨迹、完整视频质量与周期成本，是尚待形成的下一层结果；已有D176投影下限无法替代它。[执行范围][initial]

## 6. 计算依赖已经给出两项明确的实现组织

### 6.1 用同一个softmax归约输出必要统计

online softmax维护最大值m、分母ℓ、加权分子z，最终得到O=z/ℓ与LSE=m+logℓ。low anchor需要的完整LSE可由同一次原生归约输出；具体后端仍需暴露这个结果及一致约定。

对selected部分，把value增广为

\[
\widetilde v_j=
[v_j,\;P_k^{8\top}k_j,\;
(P_v^{8\top}v_j)\odot(P_k^{8\top}k_j)]\in\mathbb R^{144}.
\]

同一softmax加权和给出O128、key投影均值8、混合矩8。value投影均值来自O128，因此协方差由混合矩减均值乘积得到。全键未加权混合矩可以按head先做8×128归约，再乘各query，亦无需保存全概率矩阵。

本轮用真实冻结投影基、N=31,668、P64、head128、selected3968和合成Q/K/V，独立执行direct/stream两种计算。完整LSE最大绝对差1.78e−15，selected协方差最大绝对差2.50e−16。这验证代数重排；GPU144维value实现、BF16舍入和实际延迟仍属在线实现内容。[程序](verify_online_statistics.py)、[合成核验收据](online_statistics_algebra.json)

online softmax归约的依据参见[FlashAttention原论文][flash]。这里新增的是把现有FINITE所需矩组织进其value归约，而非增加一个全键概率矩阵。

### 6.2 high OUT-ABS可以剪去不读取的anchor统计

真实代码中high OUT-ABS只读取当前selected输出、固定CAL变换和当前P教师。其anchor只需Q/K block均值用于路由，原生anchor O、LSE、u/rho及selected统计不进入读出。当前配方可以先省去未使用的high anchor selected统计，保持所需路由、tie规则、变换与BF16舍入。

low FIN-C0和low OUT-D都使用ANCH底座，仍需要low anchor的u/rho与完整LSE。C0中质量监督权重μ=0，所以当前P完整LSE不属于拟合的必要教师；当前P原生O即可。

令α为selected key密度，β=64/31,668，以四次普通全行attention core的QK+PV MAC为分母，low增广value的core系数为(128+144)/(128+128)=17/16。

| 构造 | 四步attention core理想MAC比例 | α=1/8 |
|---|---|---:|
| 历史普通V128交互估计，四步selected、未计矩增广 | 1/2+α+β/2 | 0.62601 |
| 当前B：high OUT-ABS/low C0，剪去不用的high统计 | 1/2+25α/32+β/2 | 0.59867 |
| 同专家参考C：high OUT-ABS/low OUT-D | 1/2+3α/4+β/2 | 0.59476 |

α的实际路由边界对应3892–3968个keys，详见[成本算术](extension_cost_arithmetic.json)。若low原生anchor无法返回LSE、另做一次完整QK，B/C均需增加四步core的1/8。全局矩、投影、路由、读出、布局和缓存代价另计。

完整延迟应满足成本拆分

\[
T_{new}/T_{wide}=1-f_{self\ core}(1-r_{core})+\varepsilon_{extra}.
\]

f和额外代价尚需真实计时，不能把0.59867直接解释为整片加速。原15.241GiB attention状态与47.57GiB wide峰值相加得到62.811GiB，仅是容量算术；实际同时峰值由缓存生命周期和在线工作区决定。

## 7. 新潜力：让不同模块采用自己的锚点时序

high OUT-ABS的路由可在step0由当前Q/K均值建立，并不需要原生attention输出。因此存在以下明确构造：

| 步 | attention | FFN |
|---|---|---|
| 0，high | 当前OUT-ABS＋P64 | 当前自身输入上的原生FFN，建立anchor |
| 1，high | 当前OUT-ABS＋P64 | 原wide8192命中 |
| 2，low | 原生attention，建立FIN anchor | 当前自身输入上的原生FFN，建立anchor |
| 3，low | FIN-C0＋P64 | 原wide8192命中 |

attention从两次原生anchor变为一次，FFN继续保留自己的两次anchor。每层attention的恢复调用从2次变3次，40层共120次、4800个逐头拟合。

同一α假设下，理想attention core比例为

\[
\boxed{r_{module\ schedule}
=\tfrac14+\tfrac{33}{32}\alpha+\tfrac34\beta
\approx0.38042.}
\]

这项构造的意义在于：锚点要求来自各恢复公式读取的状态，模块扩展可以同时改变缓存时序和可替换计算比例。

当前CAL变换来自已记录命中步，step0的质量尚未验证；step0近似也会改变后续轨迹。故该构造作为原有B闭环后的独立调度比较，届时仅改变high step0 attention，不改变FFN建锚/命中时序、P64、路由规则或正则；FFN锚点值仍按新轨迹上的当前输入重新建立。它是本轮信息依赖推导出的后续方法潜力，未计为现有实测收益。

## 8. 下一轮最小整体比较

登记为wide8192范围的探索性在线扩展，保留原FIN-C1双背景冻结判据的原始结论。

| 臂 | 配方 | 比较意义 |
|---|---|---|
| A | BF16 wide8192＋原生attention | 当前部署基准 |
| B | wide8192＋high OUT-ABS/low FIN-C0 | 现有结构化attention候选的完整收益 |
| C | wide8192＋high OUT-ABS/low OUT-D | 同专家分工下的输出坐标参考 |

用0392、1013原prompt、seed42、同初始噪声与原分辨率，B/C各两片，共四片新候选视频。A及Dense参考在身份完全匹配且可访问时复用；缺失的参考单独补齐。上述两源均属已反复使用的DEV，本轮用于自身轨迹与工程可行性辨析；候选稳定后再做来源独立的确认。

**A/B**回答扩展是否在自身全层轨迹中维持可接受的运动、空间关系、轮廓与纹理，并获得完整周期收益。**B/C**回答同样专家分工下有限创新耦合的增量作用。已有面板中，B相对C的wide点/空间/时间误差分别约下降2.75%/2.08%/1.50%，远小于相对统一OUT-ABS的39.78%；两种比较涉及不同机制，不混用。

C是后设的匹配机制参考，沿用其原CAL参数；原CAL low OUT-ABS分数仍优于OUT-D，不能把C改称原CAL获胜基线。该比较优先保证机制可辨识。

在线实现交付全层selected/statistics、P教师、逐头拟合、P覆盖和cache生命周期。现有512行面板代码提供数学配方，31,668行生产执行仍需接入。正常性能运行关闭捕获与影子诊断，以同卡、同条件交错配对计时，报告整片墙钟、四步GPU事件、首次准备、原生查询与恢复代价、峰值显存；具体重复口径沿用现有wide运行协议。

视觉结果由用户同步审看完整视频，局部点/时空边和flow只用于解释具体变化。先建立这一闭环，再根据实际质量/成本决定开启第7节新时序，或附录中的块出口恢复目标。

## 9. 联合恢复的后续位置

如果自身轨迹表明仍需增强联合恢复，可以保留逐头attention，把原FFN读出的目标改成真实块出口残差。q16和直接激活核在候选attention产生的实际FFN输入上计算，原生P链提供块教师，gate后的核系数直接写回块空间。该构造不增加第二次原生P FFN，且可替换原FFN读出。

这一新目标改变P核中心和函数族，须单独比较。其具体公式与实现边界保存在[结构化在线方案第4节](ONLINE_STRUCTURED_ATTENTION.md)。当前主线已有可实施候选，联合求解作为后续由整体证据触发的分支。

### 方法能扩展到哪里，取决于哪些具体关系

可迁移的对象是计算图中的子计算：便宜当前计算保留哪些真实有限变化，少量原生子查询能看到哪些剩余作用，恢复函数族如何把教师约束传到其他位置，以及写回后怎样经过真实后续模块。不同模块据其实际状态依赖决定建锚时点。

| 子计算 | 当前可用的结构 | 本轮支持的判断 |
|---|---|---|
| FFN | 逐query原生查询、选中非线性、已知模型方向、原激活核 | 现有wide8192构成完整轨迹基准 |
| self-attention | 当前全部K/V、选中支持、逐头P教师、有限质量与value耦合 | 已有明确在线候选，下一证据是全层自身轨迹 |
| text-cross | 固定文本K/V、当前Q、可取逐query教师 | 同类观测接口适用；已有面板可继续研究其有限变化 |
| 当前attention→cross→FFN块 | query局部tail让同一P链取得中间和块教师 | 可统一评价实际作用，后续可重定义结构化块目标 |

跨越下一层self-attention时，所有query通过K/V再次耦合，完整教师闭合成本随之改变。推广应沿这样的依赖关系划定子计算边界。

方法区分力仍落在具体过程：FFN保留当前非线性与模型方向，low attention保留有限质量变化和当前value方向的耦合。high OUT-ABS是现有组合中的输出坐标恢复参考；其独立锚点时序来自真实依赖，不能将整个组合的收益全部算作有限创新机制贡献。B/C因此与A/B各自保留不同的比较意义。该段是机制归属与可检验构造，本轮没有新增全面文献新颖性判定。

## 10. 本轮资产与复算

- [完整数值辨析](NUMERIC_GEOMETRY_ANALYSIS.md)、[JSON/谱复算程序](recompute_observability.py)、[复算结果](observability_recomputed.json)。
- [结构化在线接口、P覆盖等价与后续块目标](ONLINE_STRUCTURED_ATTENTION.md)。
- [真实投影基上的合成归约核验程序](verify_online_statistics.py)、[核验结果](online_statistics_algebra.json)。
- [完整周期静态算术程序](compute_extension_cost.py)、[结果](extension_cost_arithmetic.json)。

在本目录可直接运行前一项真实收据复算；脚本支持同目录gzip输入：

```sh
python recompute_observability.py
python compute_extension_cost.py
```

合成归约检查显式传入原封存P_v/P_k基目录；来源是results/replay_evidence.zip内plan_assets/bases，按收据SHA256校验。它不需要模型权重或GPU：

```sh
OPENBLAS_NUM_THREADS=2 python verify_online_statistics.py --bases /path/to/plan_assets/bases --output online_statistics_algebra.json
```

[evidence]: https://github.com/Qesire/FFNDS/commit/cc01675282db463fc8983f9113ea3a4ff8c309ef
[report]: https://github.com/Qesire/FFNDS/blob/cc01675282db463fc8983f9113ea3a4ff8c309ef/research/20261124/attention_wide8192/results/joint39/D176_OBSERVABILITY_REPORT.md
[initial]: https://github.com/Qesire/FFNDS/blob/cc01675282db463fc8983f9113ea3a4ff8c309ef/research/20261124/attention_wide8192/results/README.md
[flash]: https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf
