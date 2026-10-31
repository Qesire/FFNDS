# G16 · SICR v3：从研究框架到可检验的静态 FFN 组预算算法

2026-10-31 · AUTHOR_REVISION / SECOND_SELF_AUDIT_REQUESTED / ALG+SRC+LIT / DESIGN_ONLY / NOT_EXECUTED / NO_GPU_APPROVAL

修订对象：[G15](G15_SICR_CROSS_LAYER_STEP_CAUSAL_STATIC_QUANT_CACHE_REVIEW_20261031.md)，审查锚点为 FFNDS 465c6dbc34cef58170ed0198028a4b91ffc607db。外部输入为用户提供的《SICR 独立评议：数学正确性、工程可行性与论文新颖性》。本文件是作者采纳评议的实质修改，**不冒充正式独立Agent recheck、授权实验、模型运行或已发表新算法**。原Signed-HWG v2和历史RUN均保持原身份。

## 1. 明确采纳的批评与修正

1. **接受**：固定支持与固定增益的单残差缓存是历史输出+稀疏增量的代数重参数化，非新的函数逼近公式。参照Liu/Stoller/Teitelbaum, TOPLAS 1998, Static Caching for Incremental Computation（https://doi.org/10.1145/291889.291895），以及Chipmunk与Increment-Calibrated Caching。以后只称“single-residual sufficient cache（单残差充分缓存）”。
2. **接受**：12.08GiB是给定M=31668,D=5120,40层,BF16的**逻辑缓存**，不是完整系统峰值。首次构建可能付额外selected W2投影。独立构建的主W1/W2理论比77.7778%；若实际实现无重复组投影的分割/双累加，理想68.5185%。系统wall未知。
3. **接受并纠错**：当固定增益α≠1，缓存包含部分已保留组的历史项，G15仅以W2_unkept的rank讨论通用α不充分。正确映射和限定见§3。
4. **接受**：条件读出α(Z1)不可以免费共用一个固定历史残差；一般需要额外历史投影。S1必须用信息收益—新缓存—元信息观察—调制执行的**净价值**决定是否推进。
5. **接受**：跨步Jacobian/Gram是成熟数学工具，不是选组算法。新方法必须给出有限可付费估计器、明确更新规则、所用真实state、后缀成本和来源外判别。
6. **接受**：量化+稀疏+缓存+传播+非均匀层预算的组合已存在强先例。重新定位于“静态FFN整组可执行动作的来源外**边际视频功能价值**是否能在有界校准成本内被有效预测”这一可证伪研究问题。
7. **接受**：真实质量是多轴约束而非未定义加权总分；只用冻结功能读出标量作为离线提案代理。
8. **接受**：先开展V1缓存算子、V2边际排序、V3跨层预算三个区分明确的有限视频范围，不启动原P0–P2最大视频网格。

## 2. 新的具体候选 Q-MARG（working name，创新未确立）

**对象**：Wan2.2 I2V A14B真实4步High/High/Low/Low、每expert 40层，160次FFN调用，80个合法同expert第二次增量位点 i=(e,l)。每组整64 hidden通道，H=13824=216组。为每个位点离线冻结支持S_i；第一次同expert Dense构建缓存，第二次真正对选定组进行W1/GELU/Q2/W2计算。全视频、source×seed配对，不改变Attention或去噪步数。

### 2.1 真正的目标（不可直接免费求值）

令V(S)为**学生free-rollout**完成全部4step×40层后的真实视频，来源s、质量轴j取各自已冻结的合法指标Q_j。对增加第i位置组b的操作a=(i,b)：
\[
 \Delta^{video}_{s,j}(a\mid S)=Q_j(V(S\cup\{a\}))-Q_j(V(S)).
\]
这是有限干预的配对因果估计量，依赖已有支持、学生历史、step、source和seed。它可能在不同轴符号相反。若拿它作为优化时刻已知输入，就等于偷跑完整视频，因此**只能作为独立实验验证目标**。

真实目标为：在每轴来源级非劣界、运动双向偏移界与最坏来源尾部限制、显存上限下，最小化无observer完整系统wall；开发集制定阈值，独立P2不能重新选择阈值。

### 2.2 具体可计算的、但有条件的离线代理

先固定一个数值族Q（RAW BF16、HiF4 fake-QDQ或真正原生packed分别立项）。从匹配来源和自然完整真实状态获取：
\[
 v_{sib}=g_{si}\odot[(h^Q_{1,sib}-h^Q_{0,sib})W_{2,i,b}],
\]
h^Q各自已通过合法Q2，本式中g是第二次调用当前的pre-gate写回系数。第一次/第二次状态须同一专家、同一层，来自实际合法student或标记为teacher代理。固定α=1时，省略组同状态误差为\(-\sum_{b\notin S_i}v_{sib}\)。α≠1时必须把未保留组加上保留组权重偏差，不能套用同一边际式。

取少量**预先冻结**的最终功能可微读出R_{sq}，q=1,…,p，覆盖最终latent的时空投影、时序差、可微纹理/动作代理以及可能的语言事件近似。用真实模型/调度器及历史缓存**增广状态**，在一条合法参考轨迹上定义同状态后缀线性映射P_{si}（包括gate之后的下游Attention、后续层、High→Low去噪及scheduler反馈）。计算
\[
 z_{sib,q}=\langle R_{sq},P_{si}v_{sib}\rangle,\qquad z_{sib}\in\mathbb R^p.
\]
获得z的方式：p个经真实网络回传的VJP/后缀有限差分，配合按group streaming的adjoint×真实ΔhW2求内积。**没有假定z便宜**：模型全M≈31668，80位点×216组，VJP/回放成本可能非常高，需逐source记录真实反向/前向次数、GPU秒、峰值及读取字节。先以已有signed Gram缩小候选，再对少量代表性早/中/晚层、High/Low及争议组支付后缀观察；绝不以部分site结果冒充完整全层算法收益。

对整个静态支持S，定义**代理遗漏总向量**
\[
 r_s(S)=\sum_i\sum_{b\notin S_i}z_{sib},\qquad \hat E_s(S)=\|r_s(S)\|_2^2.
\]
注意它已经有不同层/step的signed cross-term，但仅描述同一参考状态、相同冻结线性读出下的**一阶代理**，非真实视频质量。

将组a加入支持后\(r'_s=r_s-z_{sa}\)，因此有严格恒等：
\[
 \boxed{\hat\Delta_s(a\mid S)=\hat E_s(S)-\hat E_s(S\cup\{a\})
 =2\langle r_s,z_{sa}\rangle-\|z_{sa}\|_2^2}.
\]
这就是具体的**可更新边际价值**算法内核。相对于按局部W2能量打分，它包括跨层/step当前剩余误差的有符号投影；但此公式不是数学新定理，也无视频单调性/子模性保证。高能抵消对z_a=u,z_b=-u应通过r处理，而不是盲目保留两个大组。

若采用q个功能读出联合代理，只能在开发集冻结读出后进行指定的标量化/多目标Pareto筛选，不可声称这个标量就是VBench得分。最终由源级视频多轴非劣重新判定。

### 2.3 真实层间非加性与动态偏移的校正

有限抽样互不相邻/相邻层、同expert历史跨步、High→Low、不同latent τ/空间ROI等两干预，测：
\[
 J^{video}_{s,ab}=Y_{11}-Y_{10}-Y_{01}+Y_{00},
\]
Y分别为末latent、预注册功能读出、来源级视频指标。四角各自执行**完整配对学生free-rollout**。这是整体策略交互，不等于组间局部Gram的因果识别。比较其符号与一阶交叉项的来源外排序，若失配，记录并拒绝代理；只有有独立验证的少数高危pair才考虑增加显式有限pair惩罚。

一阶模型固定参考状态而真实学生将改变input、gate、Attention及Q2编码，故代理必须分别检查teacher/学生同状态、单臂和全组合rollout、不同seed/source。原生量化切码处的导数不能当作精确全局导数；合法finite intervention为必要控制。

### 2.4 选择规则及计价

训练来源等权，保持每source的r_s，计算所有合法a的来源级边际数组。对来源级均值和尾部分位进行保守估计（来源簇bootstrap只是描述性，不能在选择时宣称分布无关置信保证），选择满足物理约束的可行组。物理成本从真实K64 tile、M维度、weight layout、packed分组/缓存构建下的逐位点分段时间表估计：T_{e,l}(K,backend,shape,cache), 不用固定“每组线性时间”。组收益除以**实际非零边际代价**可以做启发式排序，但最终在固定真实预算和质量约束下比较。每增加组，更新r_s；再采用少量固定K的1-swap及跨High/Low层的组交换。冻结完整80个支持S_i及算法参数后才做来源外视频。

存储已计算的z为O(N·80·216·p)，后续每轮朴素更新/扫描O(N·80·216·p)，但这个复杂度完全**不包含昂贵的全模型VJP/后缀/视频测量**。其可行性、来源外优于signed Gram和视频质量仍OPEN。

## 3. 单残差缓存的严格范围、最小性修正及三种构建实现

固定支持/增益时：
\[
 Y_0=b_2+\sum_b P_{0,b},\quad P_{0,b}=h_{0,b}W_{2,b},
\]
\[
 C^\alpha_0=Y_0-\sum_{b\in S}\alpha_bP_{0,b}
 =b_2+\sum_{b\notin S}P_{0,b}+\sum_{b\in S}(1-\alpha_b)P_{0,b},
\]
\[
 \hat Y_1=C^\alpha_0+\sum_{b\in S}\alpha_bP_{1,b}
 =Y_0+\sum_{b\in S}\alpha_b(P_{1,b}-P_{0,b}).
\]
在实数下精确，BF16累加次序不同可以数值不一致；缓存必须pre-gate、bias仅一次，当前e5及residual按原顺序执行。历史由**真实学生第一次同expert调用**建立；不能共享High/Low、block或来源。

若旧h任意可变、只允许**精确线性编码+线性解码**，令β_c=1（未保留）、β_c=1-α_b（保留组），则\(C^\alpha_0-b_2=h_0 B_\alpha\), \(B_\alpha=\mathrm{diag}(\beta)W_2\)。所需线性充分坐标数至少\(\operatorname{rank}B_\alpha\)，可能是D。α=1时才是\(\operatorname{rank}W_{2,\bar S}\)。这不是实际视频分布/低秩codec的最小bit数证明；0-cache预测的平方损失下界须给定合法侧信息Z，取条件协方差。

M=31668,D=5120,40 block的BF16逻辑C缓存约12.08GiB；H+Y约44.70GiB，selected h+Y约24.16GiB。动态α(Z1)在第一次构造缓存时未知，若α=α0+Uθ(Z1)，需保存C0^(0)和r个历史投影R0,j；理想缓存(r+1)×12.08GiB，另有真实构造与读取成本。

**物理首步构建三路**：
- **E1 REDO-SELECTED**：首次Dense W1+W2产生Y0，再额外一次selected W2投影构造C0；第二次选组W1+W2。两专家第二步均选K且K_H=K_L=5120时，四步主投影比=(4H+3(K_H+K_L))/(8H)=77.7778%，不含额外缓存/launch。
- **E2 SPLIT-DISJOINT-W2**：第一次完整W1后把原W2按保留/遗漏列块输入切成互斥组，两次不重叠W2 GEMM，得到P_kept、P_omitted，同时Y0=P_kept+P_omitted+b2，C0=P_omitted+(1-α)P_kept+b2；理论不重复W2 MAC，但会增加kernel launch、layout/packing、输出融合、同步和可能weight副本。第二次仅保留组正常计算。**未实现**。
- **E3 DUAL-ACCUM-W2**：首次W2每个实际乘积按β选择性加入两个累加器，同时生成Y0、C0，理论可复用乘积且不额外做selected GEMM，但寄存器压力、原生数据排布、归约顺序、kernel面积/吞吐均待验证。**未实现**。

E2/E3若原生实现无重复组投影，四步主投影理论比=(4H+2(K_H+K_L))/(8H)=68.5185%。若E2/E3实际上比H+Y及Dense慢，只能报告缓存表示收益，不能宣称加速。特别要同后端同支持比较Chipmunk H+Y/X+Y/S0 E1 E2 E3，真实CPU offload/cache build/IO/packed权重与峰值，禁止把logical GiB当物理峰值。E1额外selected-W2与X+Y额外old-W1算术比可能数值相同，**并不是同一执行路径**。

## 4. Qmeta严谨性：量化没有制造独立香农信息，但可能降低取证成本

在确定性Q1与已知W/recipe的前提下，\(Z_Q=f_Q(x,W,\rho)\)，因此\(I(A^\star;Z_Q\mid x,W,\rho)=0\)。所以“量化元信息提供额外信息”应严格解释为：**相对当前时间与成本受限的可读信息集**，sign/scale/位置与静态W几何可能给出更好的低成本组价值预测，而不是超出完整x/W的信息论新信息。

离线冻结支持的主方案：用历史量化码、scale层级、W1/W2静态分组、signed-position × W 投影、跨来源互抵统计学习S和影响代理；线上不访问动态码来改变支持。独立的结构静态S1方案：在Q1生产阶段取得当次低成本signed/scale sketch调节固定形状读出，必须同步多缓存历史投影并测NetVOI。绝不能使用尚未计算的当前Q2组代码决定本次W1 skip。

同数值身份比较 static W-only / cheap BF16 input summary / Q1 scale-only / Q1 signed histogram / ordered signed-position / W×A1 code-sketch / signed joint group；所有来源、训练预算、候选动作和读取时点固定。HiF4原runtime是fake-QDQ后BF16 GEMM，不持久直接暴露packed code/scale，sidecar/融合导出须付费。原生MXFP4 block32、NVFP4 block16的闭包需独立核实，不以HiF4 K64结论代替。

研究主命题可以是“**量化编码在同获取成本下提升边际功能价值的跨来源排序能力**”，而不是量化新增信息。若与同成本BF16摘要没有统计稳定优势，则将Qmeta降格为编码/实现选择，不视为论文主贡献。

## 5. 视频非劣、统计与最短区分性实验

最终真实约束：对每个合法质量轴j，在独立source配对seed下，来源等权\(\Delta Q_j\ge-\delta_j\)，有独立的source-tail容差；对运动/相机幅值与轨迹偏差使用**双向**容差（防衰减也防过冲），同时报告事件/文本、细物体、闪烁/拖影以及完整81帧视觉审阅。各轴阈值在开发数据/专家评审预注册，不对P2事后设定。目标为最小真实system wall与peak memory的Pareto，代理的p维标量不能取代多轴验收。

**V1 · 程序语义与成本**：Dense/no-op、同mask同α的H+Y、X+Y、S0-E1/E2/E3；先针对真实FFN pre-gate/e5/Q1/Q2/bias和student prefix逐边数值检验与无observer计时，确认E2/E3确实减少重复组投影；等价函数不需要大量重复生成，针对浮点误差抽样完整视频。

**V2 · 可预测边际价值**：固定同一个已通过V1的cache实现/总K，对比DIAG、local SIGNED、来源稳健SIGNED、Q-MARG、随机/历史P64；另做Qmeta与同成本BF16短摘要。在旧898/979/96等曝光开发来源先看来源外hold-one-out排序是否有信号，若不足重新采集，不能把3张图当盲测。少量完整free-rollout必须包含完整160 FFN（第一次Dense、第二次各层按实际支持）；V2预先规定至少一个可推翻Q-MARG的反例（视频质量不优于local signed或排序反向）。

**V3 · 全层非均匀预算**：同总K的80/80、64/96、96/64和一条事先冻结的跨层budget，配对Dense；High/Low政策差与High/Low权重效应因HHLL时段捆绑**不可因果分离**，结论只写“此冻结时间段的预算”。给出源级八轴VBench合法子集、连续RAFT40对/五时间窗、主体/背景/相机运动、语义动作、时序闪烁、视频视觉和原生system成本，记录非加性四角。后续真正未曝光P2须新image-cluster、独立seed及来源级功效/非劣界复核，不用token/τ/step当独立n。

**冻结和许可**：本轮未做新CPU科学计算/GPU/model/video。每个真正实验必须依[实验前独立审查状态机](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)：另一真实Agent评议+反例→作者修订→recheck→ACTIVE新scope→身份/资源SHA prelaunch→RUN/EXIT。现阶段DESIGN_ONLY。

## 6. 新颖性审计：与现有最接近论文具体对照

- [Static Caching for Incremental Computation, TOPLAS1998](https://doi.org/10.1145/291889.291895)：程序缓存裁剪，缓存代数本身非新。
- [Increment-Calibrated Caching, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html)：增量补偿与低秩校准。
- [CacheQuant, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CacheQuant_Comprehensively_Accelerated_Diffusion_Models_CVPR_2025_paper.html)：缓存量化耦合优化。
- [QuantSparse, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html)：视频attention稀疏+量化+二阶残差重参数化。
- [SVOO, ICML2026](https://proceedings.mlr.press/v306/luo26a.html)：视频attention的离线逐层稀疏分析。
- [MC-Sparse, 2026-10预印本](https://arxiv.org/abs/2610.06801)：KV attention sparse元信息与dense-sparse残差缓存。
- [FlowQuant, Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S0925231226020278)：特征流图的缓存+量化校准和预测。
- [Test-time Sparsity for Extreme Fast Action Diffusion, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_Test-time_Sparsity_for_Extreme_Fast_Action_Diffusion_CVPR_2026_paper.html)：动态稀疏/历史特征复用及路由成本。
- [Gradient-Optimized Cache, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Accelerating_Diffusion_Transformer_via_Gradient-Optimized_Cache_ICCV_2025_paper.html)：时间梯度缓存与误差传播。
- [SVD-Cache, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)：主/残差子空间预测与复用。
- Wan FFN公平对照还必须有仓库锁定的Chipmunk、TEAL、SharQ，与作者系统结果严格区分。

所有算法构件均有前作；现仅存在**尚未验证**的特定贡献空间：在Wan的静态K64 FFN上，量化有序码/尺度与历史残差能否以明确低成本改善**来源外最终功能边际排序**，进而在原生内核和视频质量中形成新的Pareto。没有可靠来源外指标与原生速度之前，不能以SICR或Q-MARG为独立论文创新。

## 7. 必须保留的反例与停止规则

1. 两个高能但反号的组在最终读出抵消：diagonal greedy失败，Q-MARG应能根据r区分；只限线性代理内。
2. 两个组最终latent低维投影相同，但真实视频小物体/动作不同：低维R不充分，Q-MARG的来源外质量排序会失败。
3. 本次A1元信息与普通BF16短摘要同成本同预测能力：编码信息没有额外系统收益，放弃“量化信息创新”主张。
4. 单残差与H+Y数值近似等价，但E2/E3分割后wall不降/峰值增加：只能称缓存表示实验，停止加速宣称。
5. 教师prefix的z给出准确局部排序，学生自由生成的gate/Q2/Attention漂移后失效：增量后缀代理应重新设计或退回静态Gram。
6. Low保护在曝光来源有效，独立来源不重现：不得从2源推断普遍最优的High/Low预算。
7. p个完整VJP/有限差成本远高于校准预算，且视频排序不优于cheap baseline：停止Q-MARG扩张，保留对照。
8. full video质量接受而FFN/整端wall无改善：表示探索成功的可能证据，不属于系统加速通过。

**本文件最终结论：REVISE / DESIGN_ONLY。** 从G15泛化框架收缩为具体的离线可计算边际式 + 三种明确原生缓存构建竞争路径 + 可证伪的来源外视频验证。尚未有原生算法实现、质量收益、系统速度或独立reviewer recheck。
