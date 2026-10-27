# FFNDS · 研究综合地图：有意义的输出函数、计算结构与高效执行

> **现行综合入口，2026-10-27** · G0–G13计算图、89项论文索引、既有完整原生实验及本轮I/F/C判别的证据映射。原论文正文、公式和审计账本保留精确一手证据。用户已恢复离线推进，最多同时四张空卡；新范围先冻结并在[ACTIVE](../ACTIVE_RESEARCH.json)登记，实时/流式模型实验保持暂停。



**本轮新命题（UB1运动幅值衰减，OBS/ALG/OPEN分开）**：[UB1原VBench/RAFT逐对数据与外观—运动研究](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)显示两条已暴露formal20来源的相对连续运动幅值38.13%/9.19%，38/40和40/40采样对低于Dense，而四个官方binary Dynamic Degree仍全0。A5+HWG的历史背景闪烁是**额外的非物理时序变化**，UB1的幅值衰减可能是**被抑制的时序变化**，两者都不等于静态画质SSE，亦不能视为同一物理原因。数学上在固定线性视频时间差分`D_tau`下，候选与Dense的signed cross决定时间差能量变化；RAFT非线性、blur/texture追踪可观测性和主体/相机运动需另行校验。近邻VideoJAM(ICML2025)已经将appearance-motion监督分开，Motive(ICML2026)使用motion-weighted gradient影响数据选择，DCM(ICCV2025)的semantic/detail双专家不同于Wan high/low噪声expert。以上是迁移工具，不是本仓库已经实施的 motion-aware FFN。

## 0. 统一研究主轴：信息、函数与执行并行

**科学北极星**：[统一目标与I/F/C并行研究](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。FFNDS同时争取**独立可证伪的计算机制认识**和**有真实质量–成本优势的部署方法**。三个机制入口为：

1. **I · 信息与表示**：哪些状态/码字/统计/历史在决策时实际可得，是否存在共享信息、认证界或新编码结构？G11/G12的信息价格、signed界、编码域算术可先研究其本体。
2. **F · 函数与动力学**：哪些有标签或无标签的联合FFN响应有真实视频行为作用？哪些正常前向结构承担或迁移这些作用？[G13科学专题](G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)、[O/D/Q/P](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)和[Function–Structure](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)在此使用。
3. **C · 计算与执行**：直接从W1/GELU/Q2/W2、低位packed dense/稀疏核、历史缓存、训练后函数重组及系统瓶颈建立物理代价更低的新实现；其科学动机和因果解释可反过来促成I/F的发现。

**并行选择规则**：函数识别、信息表示或物理执行均可成为一个新机制的第一结果；按具体主张追究必要的源码路径、数学预测和真实证据。功能因果结论需合法normal-forward干预，执行加速则需同条件真实wall/显存与完整视频质量，二者证据义务不同。

**时空功能的历史问题入口**：[A5+HWG背景闪烁、小物体模糊机制分析](A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)将原主体结构保持与背景局部不稳定区分，提出mask切换、低幅值但持续重要的功能、数值耦合及后缀传播的不同预测。它是历史定性观察+新ALG推理，尚非A5原生逐帧ROI因果发现；与后续Affine54/UB1的不同视频身份各自保留。背景照明的空间低频×时间高频、细节模糊的空间高频×时间低频可能并存，不能只称“一种高频丢失”。

**最重要的原生依据**：过去六来源/四新来源显示局部FFN raw误差降低与完整状态放大可能并存；source425跨seed的signed cross变号、子频带异号和背景/光照不稳定揭示当前单一局部代理与行为响应之间的张力。全M物理tile、原HiF4分组和完整四步传播提供可独立研究的数值/执行约束。旧结果对其已测方案保持有效，新候选不预设任何特定mask、补算或G13功能标签。

**一手文献分工**：L80–L89因果抽象/patching等启发F线；G7–G12的分时计算、函数复用、trainable重组、FP4算法、码本运算与物理布局直接供I/C线研究。图上的原生SRC/NATIVE、外部LIT和数学ALG仍各保留证据身份。

### 本轮改变判断的三项证据

- **I17/I18：共享输入可以保留联合signed关系，固定方向还有更强的精确投影比较器。** [联合支持与604排列](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)给出离散域的精确凸包与支持函数。实际有序Q1可读后，离线`p=Wᵀc`、在线`pᵀx`只需O(D)；证书需与此比较，并处理真实tanh-GELU、Q2与输入适应方向的成本。
- **U25–U27：直接W2项和Q2补集项在真实保存操作数上共同存在。** [原M D/Q](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)在八个条件上通过严格Q2/W2位模式基线；最大坐标动作的门控FP32补集/直接范数为13.36%–23.63%。[W2形状控制](E0_W2_SHAPE_CONTROL_RESULTS.md)解释了M160相对历史输出约0.292%的数值差异，原M恢复全部6,553,600个保存输出元素。后缀功能、可省物理工作与视频质量仍分别开放。
- **UB1：历史基线仍有值得保留的完整画面。** 用户要求的[独立全帧静态复核](UB1_INDEPENDENT_VISUAL_REVIEW.md)覆盖两个formal20配对、324帧：主体/场景保持可辨，犀牛动作幅度较dense小。对应SHA的三项原评分可追溯；[原版RAFT补评](UB1_DYNAMIC_DEGREE_RESULTS.md)显示犀牛/住宅UB1平均光流约为dense的38.1%/9.19%，而二值dynamic degree四条全0。运动量、动作正确性及完整质量是独立观察层级；全20来源及人工验收仍待对应证据。

## 1. 当前原生证据锚点：真正执行什么

原生对象是 **LightX2V Wan2.2 I2V A14B，四步蒸馏D-Q、high/high/low/low两套独立expert，40 blocks**。Wan2.1 T2V、Wan2.2 TI2V 5B、FLUX及训练后因果student作为独立的论文比较对象。研究日志中的代表性一次模型形状：M=31668个token行、D=5120、FFN hidden H=13824（216×64）、81 decoded video frames/21 latent-temporal bins。一步只有相应expert运行。

**原生依赖**：
\[
r_2=\text{cross-attention residual result},\quad
q=\mathrm{FFN\_AdaLN}(\mathrm{norm2}(r_2)),\quad
x_1=\mathrm{Q1}(\mathrm{Smooth1}(q));
\]
\[
z=W_1x_1+b_1,\quad h=\operatorname{GELU}_{\tanh}(z),\quad
\tilde h=\mathrm{HiF4\_Q2}(\mathrm{Smooth2}(h)),
\quad
y=r_2+e_5\odot (W_2\tilde h+b_2).
\]
实际数值边界按 [G2原算子与BF16写回](../../docs/98_端到端计算图与证据注解_20261025.md) 和 [原HiF4算子反例](../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json) 确定：本项目原生实现采用 **HiF4 fake-QDQ后BF16 Linear**；外部NVFP4/MXFP4 packed Tensor Core GEMM按其独立数值、内核和硬件条件比较。HiF4 Q2受64→8→4共享尺度耦合，不能在已删除hidden列的情况下自由重算其余列并声称仍是原相同FFN。

**共同科学目标**是揭示信息、函数与计算之间可复用的新机制，并推动真实省算与完整视频质量的Pareto。信息可用时间、净信息价值及可省物理操作（原Q1/Q2/Q3）相互促进；函数/结构定位、编码域代数、算子优化与训练后职责迁移可以独立起步。

### 已有实验证据提供的结构线索

- 原始 full-M 后验逐token小比例支持在 M16/K64 等合法计算tile上产生100%活跃：**该掩码与该布局仍需执行全部tile**。由此可研究输出函数的自然分组、共享输入和新的物理执行单位。[原实验报告](../20261025/E0_SHARED_TILE_PRE_W1.md)
- 对完整40层×4步的既有原生替换，局部FFN raw误差可能改善，生成latent继续偏移；历史候选有真实FFN或生成速度增益但视频存在质量失配，尤其背景光照和细物体时序稳定性。这一结果揭示了**局部算子误差与终端视频功能的独立评价轴**，促使研究从行为功能及其承载结构重新识别可减计算。[实验发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)
- 已覆盖10个独立source身份、12条source×seed完整轨迹、1920个全深度FFN观察位置；严格区分**site数量**与独立视频数；已有视频形成实际观测证据，正式VBench与人工验收继续按[协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)独立登记其状态。现行[视频与全深度指标协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)和[ACTIVE](../ACTIVE_RESEARCH.json)独立决定质量与实验许可。

## 2. 可独立启动的五种高效计算机制

| 计算层面 | 可行的物理动作/原理 | 执行该方案时实际所需的信息 | 原Wan合法性与证据 | 论文强先例 |
|---|---|---|---|---|
| **A. 冻结FFN本征函数** | 提前证实或统计预测一组gate×W2有符号函数足够小、成组共享或可复用；才考虑少算W1→W2 | 当前q/Q1/离线weight、可能少量配对或时序历史；不得读取尚未计算的完整W1/GELU/Q2当免费决策器 | 原HiF4 Q2 group closure、下游signed抵消及tile支持严格约束；目前预测可省性质未成立 | G11 Prox L69仅为付费INT4代理近邻；G12经典重排区间为ALG而非模型实测 |
| **B. 编码域直接计算表示** | 将量化码的有限组合改为bitplanes/XNOR-popcount、预计算LUT或vector-code Psumbook；无需先找到无用通道 | 当前activation、weight codes及scale、索引与足够可复用的量化codebook | LUT-GEMM/CodeGEMM公开实现只为单向量提供查表快路，多token退解权重+cuBLAS/matmul；与原fakeQDQ和native packed FP4的比较未完成 | L73 CodeGEMM、L75 BiQGEMM、L76 LUT-GEMM、L77 XNOR-Net |
| **C. 内核运算/数据布局** | 同一个低位GEMM的N:M activation pack、权重zigzag、压缩坐标、kernel融合；对真实memory和launch计费 | 当前activation稀疏性、合法N:M/组metadata、物理布局与kernel能力 | SharQ需要**sparse GEMM + dense FP4 residual GEMM**；SpQt/MACKO的速度集中在single-vector GEMV | L74 SharQ、L71 SpQt、L79 MACKO、L08 RT-Lynx |
| **D. 历史、帧、step层级省算** | 跳某block/step/frame，对整个函数值或历史部分做预测/插值/缓存 | 已产生的同expert/同权重、同条件历史，或者训练得到的专用预测器 | 四步中两次high/两次low；跨expert函数身份不同；无偿历史不能保证当前q、gate、Q2一致；要带未来latent反馈 | L34 DIFS、L35 FIS-DiT、L39 SVD-Cache、L09 Chipmunk、L52 L²P |
| **E. 训练形成的新函数** | 重新训练小路由/稀疏FFN/专家树/step子网络/时间encoder-renderer，令可跳结构真正存在 | 训练/蒸馏数据、目标与预算，新参数定义及部署内核 | 训练后质量与原冻结模型的后验天然结构是不同命题；可作为低成本后训练研究分支 | L70 Spark、L72 SQ-DM、L10 Dense2MoE、L53 tree FFF、L60 SCD、L61 LongLive2 |

五种动作能够组合，也可能给出不同的反向预测；后续以**已定位函数、实际执行收益与完整视频作用**比较这些实现，并检验与原论文机制的重叠及差异。并列参考 [论文原始索引 L01–L85](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)。

## 3. G7–G13研究如何投射到同一张计算图

**G7：时间与宽度的真实计算颗粒。** DIFS K01–03按step经过大规模蒸馏得到不同W1/W2组（训练且压紧参数）；FIS-DiT K04–06按frame×layer选择anchor并插值恢复完整状态（改变M行和attention上下文）；Mobile Video DiT K07–09训练出更窄FFN通道并部署compact dense GEMM；NerVE K10–12计算pre/post-GELU协方差特征值描述维数（后验观察，不提供推理免费坐标方向）。DIFS与FIS已核原PDF方程：[G7方法审查](../literature/20261026/PRIMARY_METHOD_EQUATION_AUDIT_20261027.md)。

**G8：FFN动态稀疏、训练MoE和跨步复用。** RT-Lynx K13–15当前GEMM输入的在线2:4 activation压缩、norm/LoRA补偿及SpTC执行（[本轮原文方法与Wan插入点审查](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)已核arXiv正文/附录，作者CUDA仓库及Wan原生复现尚未核实；旧G8账本为其当时阅读状态）；Dense2MoE K16–19训练shared+TopK normal experts/MoB连续block组（不同于Wan high/low）；SVD-Cache K20–23先对已计算的特征做主/残差分解、后续跳步主空间预测和残差复用；Chipmunk K24–27首先支付blockmean与W1 proxy的费用才能做hidden增量及W2 scatter。[G8原始审计](../literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md)

**G9：历史预测、树结构与全网职责迁移。** L²P K28–31预测50-step FLUX的**整段网络特征输出**，必须计入少量拟合和作者实现的额外FFT/历史读；FFF K32–35训练硬树符号路径，其value函数并非原Wan的W2列；“Sparsity Moves Computation” K36–37目前为一层玩具模型的作者摘要，说明与attention训练重分工的可能，不是视频实测。[G9审查](../literature/20261027/G9_CONDITIONAL_EXECUTION_TRAJECTORY_AND_FUNCTION_REALLOCATION.md)

**G10：普通视频模型转为实时流式。** S01–S25把离线bidirectional teacher→causal/AR少步student→rolling KV→新chunk的FFN→stream VAE、SLO与历史反馈表示成另一个**LIT模型图**。SCD once-per-frame encoder vs per-step renderer提供时间职能拆分的来源；LongLive2原生NVFP4是Wan2.2 **TI2V 5B**；EdenLive使用Wan2.2-TI2V-5B/Wan2.1-I2V-14B及LightX2V，不是原I2V D-Q测试。响应时延TTFF/p99与整段平均FPS不同；不能把实验G6的81帧离线延迟称实时。[14项实时模型审查](../literature/20261027/REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)

**G11：量化信息时点与已知强近邻。** Q01–Q33区别离线weight码、当前Q1后input码、逐pair代价、W1后oracle标签、Q2整个group依赖、真实kernel预算和Prox/Spark/SpQt/SQ-DM/CodeGEMM/SharQ。CodeGEMM作者代码多token反量化退dense；SharQ必须sparse+residual dense双GEMM，视频单独SharQ 480p/720p相对FP16作者测得1.23×/1.11×，+SageAttention才约1.58×，对应模型Wan2.2 **T2V A14B、4×RTX5090**。[五篇论文逐算子](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ源码和指标](../literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md)

**G12：签名直方图带来的严格但条件性的数学信息。** I01–I16针对给定单个W1行，在正尺度重叠分组、已知两侧signed码字multisets时用**经典重排不等式**得到最紧pair-agnostic内积区间，再部分读取实际pair逐步收紧。GELU负侧非单调，必须求区间全域上确界；各W1行共享同一个输入排列，故每行分别最紧的区间积**并不是整个W2有符号输出的最紧联合约束**。对M=31668、H=13824、B=32全量逐组筛查约70.04 billion次查询/FFN，还未计每组重排/内存与当前input histogram；更优机制要求先证明可共享信息或先降候选数。LUT-GEMM作者kernel（独立于CodeGEMM）同样单向量走LUT、多token走反量化cuBLAS。[G12严格证明与代码](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)

## 4. 同一数学依赖：从已知信息到整个有符号函数

核心对象是**当前FFN输出写回**而不是原先单独的后验activation或全局SSE。设每个量化共同尺度组的weight/input有符号multisets为a、b，则：
\[
s_xs_w\sum_{i}a_{(i)}b_{(B+1-i)}
\le \sum_{i}w_ix_i
\le s_xs_w\sum_i a_{(i)}b_{(i)}.
\]
这是对**该行该组**未知配对的最紧数学界；若真实码word完全读取则读到的配对信息可收紧至精确点积，但信息代价可能已接近甚至超过Tensor Core。原Wan current Q1若只输出fake-QDQ BF16，不可以假设已免费存在packed 4-bit input字节。

W1点积区间 \([L_j,U_j]\) 若处于一侧可认证解码点积的符号；继续通过 \(\sup_{z\in[L_j,U_j]}|\mathrm{GELU}_{tanh}(z)|\) 和原HiF4 Q2所有同组scale的合法集合得到最坏gate×W2影响。若选择某一个下游方向 \(a\)，预算示意为
\[
|a^\top\Delta y|
\le \sum_{j\in S_{\rm omit}}
\left|a^\top\operatorname{diag}(e_5)W_{2,:,j}\right|
\sup_{\mathrm{legal\;Q2}}|h_j|.
\]
仍须计入BF16舍入、共享Q2尺度、跨block/step受扰状态和视频。**这是候选证明框架，不是当前某些hidden可实际跳算的测量事实。**

不能只问“每个通道小吗？”；更具鉴别力的问题是“**同一当前q和共享输入配对，在多W1行之间是否产生足够稳定、可由低价编码信息预测的联合gate×W2函数方向？**” 这个问题连接G12最紧单行区间与G4/P14的共享函数方向，但旧P14四anchor/中心128的负面泛化只约束那次具体方法，不排除新信息来源下的别种结构。

## 5. 五组相互竞争、可以用最小证据区分的解释

| 假说 | 若为真，应观察到 | 最强竞争解释/廉价判别（未运行） | 原生与文献锚点 |
|---|---|---|---|
| **H1 固定FFN有便宜signed联合函数先验** | 在新source×seed×expert留出中，preW1低字节量化信息可预测同一q的group gate×W2下游方向，同时产生合法可省tile/步骤 | 若只在后验postQ2 oracle上有效，或者scale/hist界普遍跨0，则信息不足；比较只看q幅值、norm/step、random同预算预测器 | I04–I09, Q05/Q09, F04/F08R/F09/F12, P14 |
| **H2 模型自带结构弱，但少量训练可创造** | 微调/蒸馏后的router/树/窄FFN有可执行少计算、质量稳定；冻结原checkpoint预测力低 | Spark/SQ-DM/FFF/Dense2MoE训练调节了全网与attention职责；需保持相同训练预算和功能/速度公平比较 | K16–19,K32–37,Q21/Q25 |
| **H3 信息不适合筛选，但量化code能重写内积** | 对相同输入/quant模型，LUT/bitplane/码本积省去重复乘法，不需要通道或step级重要性假说 | 缓存和byte/launch花费使fullM时比低位Tensor Core慢；LUT-GEMM/CodeGEMM公开multi-token fallback是强先例 | I10–I16,Q27–29,Q14 |
| **H4 数值历史与慢快函数可分** | 同expert相邻step/stream chunk有低费用更新时段，且在自由生成long rollout维持有符号视频稳定性 | 预估的信息来自已经算完的W1，或者历史student前缀偏移、场景切换/换expert破坏预测；先比较合法信息何时可用 | K20–31,S13–20,P23/P24 |
| **H5 原生kernel与布局主导** | 在实际M/T/B、scale和weight residency约束下，规整N:M、packed dense或压缩表示明显优于额外selector/补算 | SharQ全两路GEMM、SpQt/CodeGEMM多token回退，FFNDS原tile满占；以same format/quality比较完整wall | Q13/Q14,Q24/Q29/Q32,Q33,S20,P12 |

实验设计必须先确认哪些已有保存数据足以支持纯分析；缺少postQ1代码、合法Q2 group或真实原kernel元数据时，记录观察接口并待未来授权。**不得将新数值任务、GPU作业或“必须在完整生成上跑一次”写成已经执行。**

## 6. 三种计算环境下的相同研究问题

| 环境 | 原结构与自然kernel | 信息价格/最强比较器 | 终端条件 |
|---|---|---|---|
| **Wan原I2V整段** | M≈31668每block每step W1/W2大GEMM；真实HiF4 fakeQDQ BF16 | native dense + 显存驻留实测；反复对所有H行做直方图certify代价巨大；tile完整性约束 | 40层×4步自由81帧视频、完整时序/主体/背景/光照指标 |
| **新真实因果短chunk** | 只算新chunk与历史KV，各模型NFE、M、shape不同，可能weight/launch主导 | LongLive2 NVFP4 dense、EdenLive LightX2V内核、合法cache/TTFF队列调度；不能移用当前fakeQDQ性能 | TTFF、单帧p99、jitter、交互响应、长时漂移和VAE/通信 |
| **单向量LLM decode** | GEMV权重读取与索引/缓存大，LUT-GEMM/CodeGEMM/MACKO已证实验范围 | codebook LUT、低位proxy和packed-value/delta GEMV，不能将decode speed移到Wan fullM | token延迟与PPL/下游任务，不能冒充视频表现 |

## 7. 论文的证据级别、当前未解问题及维护契约

- **原 Wan证据：** G0–G5 SRC/OBS/NATIVE-ACT/VIDEO/SYS、[EXPERIMENT_FINDINGS_ATLAS](../EXPERIMENT_FINDINGS_ATLAS.md)各结论绑定原有RUN或保存统计。原Affine/HWG方案存在具体已测问题，但“所有conditional FFN思路”仍为开放科学空间。
- **外部论文：** 89个L01–L89构成可检索的研究索引，每条按已核摘要、原始方程、作者代码及本模型实测记录各自的阅读深度。G7 DIFS/FIS v1已核原文方法；G8 Dense2MoE/SVD-Cache已核原文公式，RT-Lynx后续已读arXiv正文/附录（见[2026-10-27专题](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)），作者CUDA仓库仍待复核；G9 L54仅作者摘要；G10部分作者论文、系统码层，作者FPS均属各自协议；G11六篇核心论文已按其具体源码和公式升级；G12 L75/L77/L78阅读级较低，L76/L79已核源码；准确阅读深度逐条以 [索引](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)、[G11账本](../literature/20261027/G11_QUANTIZATION_SOURCE_LEDGER.json)、[G12账本](../literature/20261027/G12_SOURCE_LEDGER.json)为准。
- **数学：** G12重排界是经典定理在本研究编码信息层的应用；single-row tight≠joint gate×W2 tight；多token GEMM有效性≠single-vector已实现。
- **维护：** 新论文只写一遍原始方法到其审计来源，更新LID/来源等级/信息时点/训练/成本，并给机器图节点添加准确的**比较边**；综合文档仅更新**机制类别、改变判断的证据和研究问题**，图上不能把LIT边画成Wan实线。阅读指南/Agent/Navigation只放稳定指针，不用一段新研究进展覆盖旧导航。
- **执行：** 当前用户持续推进授权已登记；每个实际新范围绑定代码、资产、预算及结果，失败保留且不自动重试。既有暂停和原始失败保留历史身份，指标环境工程不等于新质量评分。

## 8. 检索与复核入口

**权威可读图** [G0–G12全图](../../docs/98_端到端计算图与证据注解_20261025.md) · **功能定位与结构候选** [FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · **机器节点/边** [GRAPH_AGENT_BINDINGS.json](../agent_protocol/GRAPH_AGENT_BINDINGS.json) · **机制交叉矩阵** [MECHANISM_GRAPH_CROSSWALK.json](../agent_protocol/MECHANISM_GRAPH_CROSSWALK.json) · **89篇论文索引** [PRIMARY_LITERATURE_INDEX.json](../literature/20261026/PRIMARY_LITERATURE_INDEX.json) · **一手文献分级** [20261027论文与源码审计目录](../literature/20261027/) · **Agent协作协议** [GRAPH_AGENT_COLLABORATION.md](../agent_protocol/GRAPH_AGENT_COLLABORATION.md) · **研究导航** [RESEARCH_NAVIGATION.md](../RESEARCH_NAVIGATION.md)。

此文件是现行研究解释的单一综合入口。旧章节继续保留原实验与原始论文审查证据，旧时代数字和来源状态只用于说明历史版本，不参与新的结论推断。
