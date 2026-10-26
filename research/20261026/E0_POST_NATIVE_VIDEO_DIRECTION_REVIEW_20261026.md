# E0 · Wan FFN 机制与全视频失败：统一科学结论及研究引导

> **权威研究解释（2026-10-26）**。合并此前 `E0_POST_AFFINE_DIRECTION_REVIEW_20261025`、Affine54方法族Skill审查、全视频后续方向审查和追踪/偏移审查的科学内容。本文件解释**已冻结证据与待验证机制**，不替代实验RUN、SHA或当前任务状态。**最新执行阶段、授权、作业、完成/失败数必须实时读取 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 和原生RUN**；不要根据本文件的写作时间推断任务仍在等待或已经通过。当前研究阶段E0开放探索，实时生成G6按用户要求暂停。

**当前科研引导的补充（2026-10-26）：** [新CPU固定坐标函数oracle、anchor/neighbor竞争解释与SM120实际packed基元的审查](E0_NEW_WORK_MECHANISM_AUDIT_20261026.md)。本报告解释既有Affine54失配；新方法从FFN结构性冗余、信息时点和生成功能出发，不把局部/有符号/频域SSE升格为机制本身。

## 一、真正已得到的结果

**研究终点**：在原生Wan2.2 I2V四步蒸馏（high/high/low/low双专家）中，减少真实必要FFN运算与端到端生成成本，同时维持完整视频中主体、运动、背景照明与细节表现。FFN的局部SSE、理想MAC、原生kernel耗时、完整wall、正式VBench和人工质量是不同证据单位。

| 对象 | 已冻结果 | 证据支持及边界 |
| --- | --- | --- |
| **源模型本征函数** | 输入宽度\(D=5120\)，hidden\(H=13824=216\times64\)；W1→tanh-GELU→原HiF4/QDQ→W2，实际caller调制、gate与BF16残差写回 | [函数重构与原生形状](E0_FUNCTION_RECONSTRUCTION.md)。真实W2输入是`q2_post`，而不是未量化的GELU hidden |
| **旧增量导数** | 两暴露来源×40层保存full-M旧GELU导数**统计**；真实输出误差只在128/32行抽样测量。0→1只有1/40层局部SSE低于5%，2→3为10/40 | [历史切线/折叠观测](../20261025/E0_AFFINE_FFN_DISCOVERY.md)。full-M统计≠全M逐行误差 |
| **跨源仿射＋余项** | 旧五层40案例中，另一来源固定仿射＋27组的local SSE均值11.27%，所测匹配MAC组能量67组为43.13% | *保存输入局部OBS*，表明具体函数重写有价值；不是任意组选法的上界，也没有视频验收 |
| **同校准全40层** | 四校准来源×每expert256行，两个已暴露held来源×四步×40层（320 case）；Affine54平均SSE **8.54%**、Whole94 **11.67%**、仿射锚定输入ridge **15.35%**；Affine54比Whole94低305/320 case | [匹配实验](E0_ALL_BLOCK_RECONSTRUCTION.md)。结果来自teacher轨迹的抽样局部值，并非独立320视频或全模型误差。单例最高24.86%，Block5/6/7等明显较难 |
| **强Dense对照** | 原样Dense权重常驻在相同phase保留18视频/72 latent hash，热态总wall约75→50s；无FFN乘加省算 | [Dense权重常驻](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)。这只是独立系统优化和更强参照，有约80GB GPU条件 |
| **完整原生宽度减小** | 两固定bank在全部40层×4步共160 FFN位置各自替换；相同卡、相同phase的warm来源12，Dense FFN9.707s→Affine54 **4.449s**（约2.18×），生成50.750s→**45.313s**（约1.12×） | [六视频真实执行](E0_RECONSTRUCTED_VIDEOS.md)。候选setup约237.089s vs Dense139.188s，另有bank构建/存储；单次热态计时不可推广到所有硬件/冷启动 |
| **完整视频失配** | 原Dense、Whole94、Affine54×来源0/12共6条81帧；候选均显示可见画面偏差、Whole94更明显破碎/重影；Affine54改善局部拟合而仍改变运动、模糊、背景/细节 | *NATIVE VIDEO/有限视觉对比*：当前固定bank与所有160处同时替换组合不满足已有项目的保真目标；正式质量测评未获通过 |
| **六视频后验指标** | 全486帧/480帧间对，修复Farneback输入灰度尺度：来源0 Affine/Dense光流比**0.091**、相邻RGB差比**0.368**；来源12约**0.846**、**1.463** | [CPU补评和逐帧资产](E0_EXISTING_VIDEO_QUALITY_BACKFILL.md)。来源0有运动估计大幅压低，来源12帧差升高且末窗口流可反超；二者**不能统一解释为“冻结”**。光流与Laplacian是编码像素代理，非真运动、语义细节或真实画质评分 |

**新增原生机制观察**：[两来源320site完整双轨](E0_PAIRED_FULL_DEPTH_RESULTS.md)已保留实际原生四臂、写回及四步状态，四视频/16latent历史指纹相同。318/320处student前缀的raw算子SSE反而更小（另外2处初始相等），但后3步flow差与状态差高度反向、latent误差持续增大。此现象削弱“每处局部误差离轨后统一变大”的解释，不识别主因；完整`v_D(z_S)`已在两来源随后实测：见[全模型Dense有限响应](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)。两来源均在已观测有限偏移上扩张，候选与Dense状态响应的有符号交互存在来源差异；最早的FFN功能缺失、跨来源因果迁移仍开放。旧8.54%为gated、保存native-q2特征的FP64重构，不能与新ungated/fullM量比较。详细数字、费用、分母及独立口径审查均见结果报告。

**质量与标注现状**：六段已有视频的核心工程指标已补齐，但VBench-I2V/VBench2、语义背景稳定性和人工全片审阅未完成；`quality_accepted=false`。四个候选报告中 `reference_identity.model_digest` 曾错误继承候选身份，已登记不可变勘误；原视频SHA和像素分析未改。新配对入口已另实现[严格原生视频身份与清单门禁](E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)，但其软件通过不追认为旧核心gate曾有同等语义检查。

## 二、哪些数学结构成立，哪些近似没有被保证

对于实数 **tanh近似GELU**，\(\phi(a)=a/2+e(a)\)，\(e(-a)=e(a)\)。这是原激活函数的精确奇偶分解（`ALG`），不是经过Smooth2、原生Q2/BF16舍入后“可以免费折叠”的逐位恒等式。固定半斜率未必是最好的数据拟合参数；训练样本上的线性热区、导数统计、signed W2抵消与真实量化余项是不同机制。

更一般地，给定**实际post-Q2系数**\(q(a)\)与任意\(\kappa,\beta\)，定义\(r(a)=q(a)-\kappa a-\beta\)，便有\(q(a)=\kappa a+\beta+r(a)\)（完整余项的**定义性恒等式**）。这一重写本身**不省算**。Affine54使用固定54个完整64组的实际非线性、截断其它组的函数余项，重新拟合读出\(V\)、截距并将仿射系数FP32合成后cast成新BF16矩阵；原生算子形态为\(\widehat f(x)=\widehat A x+Vq_S(x)+\widehat B\)。它不与原checkpoint逐位等价，也不是把整个HiF4权重精确转换为同一格式的简化。

该固定方案的理想两投影MAC相当于\((40+54)/216=94/216\)，与Whole94比较的是**理想投影工作量**，不是严格同kernel、同QDQ、同装载量或同启动开销。新BF16读出、D×D项、GPU驻留和setup均已实际计费。

**必要的数学反例**：若两处等范数局部错误\(e_1,e_2\)经过不同有限下游映射\(T_1,T_2\)，终端能量包含\(2\langle T_1e_1,T_2e_2\rangle\)，可增大也可抵消。同样的平均局部SSE并不能充分决定最终flow、latent或视频表现。反过来，这个数学事实**也不能证明本轮视频失配主要由下游传播或SSE目标导致**：真实映射/轨迹尚需配对实验识别。未经验证的QDQ普通Jacobian不是原生有限干预的替身。

## 三、结论范围：当前Affine54失败不等于整个方法族失败

- **已成立**：本次固定54组＋对应校准/正则/新BF16权重＋160位置全替换的两暴露来源完整视频，未实现满意的质量—加速联合目标；“teacher局部SSE较低足以保证视频质量”的充分性推断被实际结果反驳。
- **仍成立的正面发现**：仿射与真实非线性组的组合在所测局部任务优于两种匹配比较器，实际宽度缩小带来FFN warm加速；方法具有非平凡的函数表示与物理实现资产。
- **尚未被证明**：局部SSE“毫无意义”、SSE是**主要**失败原因、BF16部署bank每层都和连续诊断等价、Student前缀必然是主因、固定54组容量就是根因、任意Affine＋余项/条件仿射/少量后训练方法都不可能达到视频质量—计算前沿。

因此，**保留Affine54作为已测有损性能基线与机制判别对象，不把“拯救54组”设为研究目标或必须执行的算子**。问题从“仿射近似能解释多少局部输出”升级为“哪些可低成本表示、观测和执行的FFN作用，对整个视频的运动、光照、细节和四步轨迹具有不可替代的功能价值”。

## 四、可相互区分的竞争解释（目前都不是因果结论）

| 代号/机制 | 不同预测 | 判别对象与用途 |
| --- | --- | --- |
| **NUM·部署数值** | 同一真实输入下原生candidate与离线连续目标本身已有额外大偏差 | 校验所有层实际QDQ/BF16矩阵、原生same-input四臂与对应离线预测；若偏差很小削弱这一路线 |
| **CAP·函数容量/固定支持** | 在teacher轨迹本身，选定54组已丢失后缀敏感的signed功能，Student离轨不是必要前提 | 同一输入的native error与有限受控dense suffix；若teacher关键方向基本保持则削弱本解释的主导地位 |
| **SHIFT·Student前缀** | candidate在student真实输入\(x_S\)上的算子误差相较\(x_D\)明显扩大，或Dense对输入偏移反应大 | 配对四臂\(f_D(x_D),f_S(x_D),f_D(x_S),f_S(x_S)\)，保存signed交叉项与gate后写回 |
| **TRANS·后缀方向/跨层作用** | 误差局部能量相近，但不同层/步/方向的有符号传递导致不同flow响应和相消/放大 | 真实postgate/block/flow多位置迹线，随后小规模预冻hybrid/dense suffix，不以相关性冒充因果 |
| **STATE·四步调度器目标** | Student与Dense当前flow接近仍可能因已有latent偏差维持或放大下一状态差 | 记录各step的真实pre/post cast、sigma/flow/latent和\(\langle\delta,d_{flow}\rangle\)，区别teacher瞬时目标与下一状态控制 |
| **COST·端到端有效省算** | warm FFN更快，但准备/额外观测、存储/显存、非FFN时间抵消部分净收益 | 原native wall/FFN/setup/显存作为已测参照，任何新机制再与强resident Dense和相同真实质量比较 |

理论上在实际连续实数更新下有\(\delta_{k+1}=\delta_k-h_k[v_S(z_D+\delta_k)-v_D(z_D)]+\rho_k\)，\(\rho_k\)专门记录真实dtype/舍入更新缺陷。原生Wan的`step_pre`可能按当前配置对latent作BF16转换；真实数据必须分别记录pre、post和不同轨的原始dtype。完整四臂误差分解是**相关结构观察**；证明哪层对终端视频具有因果作用还需受控后缀干预。[轨迹推导](../method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md)、[双轨数据协议与dtype审计](E0_FULL_DEPTH_TRACE_TOOLING.md)给出了准确边界。

## 五、原生双轨已经测得的机制与仍开放的因果分解

**完整原生证据（NATIVE-OBS）**：[v3两来源320site](E0_PAIRED_FULL_DEPTH_RESULTS.md)在来源0/12、每来源40层×四步全部160处保存同输入FFN四臂、真实gate写回、四步flow与latent几何。318/320处Student-prefix raw FFN绝对算子误差低于Teacher-prefix，另2处相等；Student post-Q1能量/方向改变而latent差持续增大。后3步flow差与已有latent差的cos均为负，来源0约−0.870/−0.905/−0.942，来源12约−0.790/−0.801/−0.885，真实step_post有限精度缺陷另计。这表明局部operator误差降低仍可能伴随终端轨迹分离。

**从结果构造新的判别**：旧H-SHIFT中“受扰前缀让每一处raw局部operator绝对误差增大”的版本已被这两来源观测削弱；但Student后续功能未因此得到改善。对真实非零偏移状态现已补测**完整40层dense模型**的`v_D(z_S)`，利用
`v_S(z_S)-v_D(z_D)=[v_S(z_S)-v_D(z_S)]+[v_D(z_S)-v_D(z_D)]`
区分模型差与dense对状态变化的有限响应；[原生结果](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)已取得。逐FFN shadow不能代替这条完整网络流。方向传递、跨步状态、数值部署及固定bank容量是并列竞争机制；未来的受控后缀干预才能进一步指认层级因果效应。

**进一步的保存数据与数学判别**：[原生writeback followthrough](E0_WHOLE_FLOW_COUNTERFACTUAL_PREPARATION.md)也测得318小/2等/0大；实际gate/residual写回没有逆转这一观察。整步应再检查`P=δ−hR`的有限增益与`h²||e||²−2h<P,e>`的signed候选净效应，并单列真实BF16 shadow step_post；同一总流差可对应收缩或扩张的dense响应。独立radial-only理想投影界在source12 step1为555.174>输入474.421，给该一步的单方向校正家族以明确边界。完整flow固定v1现已完成：[八状态对](E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)中六个非零偏移的native dense一步能量增益3.63–7.72；候选整步交叉项来源0为负、来源12为正，净效应均正。状态来自候选历史，有限增益不等于全局Jacobian、根因占比或可相乘的恢复增益。16通道偏置+当前flow/latent两个全局尺度的数值oracle仍留下60.20%–71.14%理想目标SSE，不是形式化家族排除。该轮0新视频；正式画质仍未接纳。另有[共享投影/向量值曲线](../method_discussion/discoveries/E0/independent-signed-neuron-curve-coordinates-20261026.md)作为独立条件表示；其signed差分及Q2多变量闭包是待检验的竞争机制。

**进一步接回完整视频**：[已存末步latent的四次VAE桥](E0_SAVED_FLOW_DECODE_RESULTS.md)两dense控制逐字节一致，来源12末步换dense后RGB/时序误差下降13.58%/15.80%，来源0仅0.45%/2.32%；估计运动幅度、细节比和luma存在异质权衡，正式质量仍未验收。[周期状态频谱](E0_SAVED_FLOW_SPECTRUM.md)显示末态误差在4.79%谱维数内含53–70%能量，且signed cross可在子空间和全域反号；该观察不等于低频统一优先放大或在线可修复。独立[原权重共享曲线CPU](E0_VECTOR_CURVE_CPU_FEASIBILITY.md)所测四簇没有近共线性，曲线误差34–89%；输入流形上的固定预算非线性坐标仍是另一开放机制，下一函数空间比较须匹配rank和样本内容量。

**失败与执行版本追溯**：[v2路径故障](E0_PAIRED_V2_FAILURE_AND_RECOVERY.md)保存两条逐字节相同的DD视频但后处理未留存详细trace，记录其site证据为0；修复后v3在单卡串行完成四视频及320site，16个历史latent指纹匹配，核心全帧/严格身份门禁通过，物理峰值75.01GiB。v3的双轨观察wall包括shadow、诊断GEMM及离线核验费用，**不是候选部署速度**。正式VBench-I2V/VBench2、语义背景及人工全片仍按实际结果NOT_RUN。软件测试与冻结旧RUN各自保持身份；当前资源授权见[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)。

## 六、AutoResearcher引导：两条独立而协作的线

**A｜已完成原型的失败判别**：先分NUM/CAP/SHIFT/TRANS/STATE，选择最能切开竞争预测的已有证据/数学/源码/原生配对结果，再以冻结的、有代表性的多层×四步时空分层决定少量后缀干预；不能只用Block0/39或曾有限测的0/30代替全40层。已有v3完整观测可优先提出可区分的有限dense-flow反事实与有符号下游预测；模型层新实验按独立授权执行。

**B｜新FFN必要计算发现**：从信息可得时点、W1/GELU/W2函数基与signed值方向、时空条件、专家/step结构、可执行tile、其它FP4格式或有界后训练构造原本不存在的低成本作用单位；可完全不依赖Affine或HiF4原group64。提出新解释→实际数学与源码链路→与最接近机制不同的预测→最低成本判别→可能的终端质量/运行成本路径。

两线均遵循默认 [AutoResearcher v2.4四类Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md) 与 [Graph-Agent协议](../agent_protocol/GRAPH_AGENT_COLLABORATION.md)。G5/P21局部OBS、P22视频执行和成本、P23/P24已测双轨NATIVE-OBS与开放因果机制、P25证据政策必须保持身份分离；G6实时应用场景已暂停，不派工作。提出开放结构无须等待Affine54失败归因结束；要声称可用加速才要求同条件全视频质量与真实物理费用。

**邻近研究**：[TARDIS](https://arxiv.org/abs/2501.10054)已有局部线性化/权重折叠/动态回退；[BRECQ](https://arxiv.org/abs/2102.05426)涉及块重构及跨层相依性；[Q-VDiT](https://proceedings.mlr.press/v267/feng25q.html)提示视频时空质量区别；[CTEC](https://ojs.aaai.org/index.php/AAAI/article/view/34039)已有跨步误差校正；[Sparse VideoGen](https://proceedings.mlr.press/v267/xi25c.html)和[Sparse-vDiT](https://ojs.aaai.org/index.php/AAAI/article/view/37287)为时空/计算组织近邻。它们没有在本Wan量化四步原生算子上替我们验证结果，不能从论文名认定原创或复现。

## 七、整理与证据真伪的保留规则

本文件替代重复方向审查中的**当前解释**，但不能修改已冻结实验RUN/JSON/视频原始SHA、历史盲审/评议原记录及固定协议。错误的dense`model_digest`保留不可变勘误并在严格新门禁中拒绝，不能通过重写旧记录伪装历史没有过错误。研究叙述只引用经修正的数学/时序/数字；失效的“实时并行”“全深度collector尚未实现”“v2永远未授权”“新严格门禁未实现”等不再作为当前事实。

本文件是统一科学解释，不是新测试、生成结果或已完成独立同行评审；适用日期和运行身份见引用证据，新的原生结果应当在运行报告落地后更新此文件的*科学推断*，而不是追加新的同义方向审查。
