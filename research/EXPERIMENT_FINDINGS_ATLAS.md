# FFNDS · 实验发现谱系、机制分类与对照矩阵

> 2026-10-25 | 基于 AutoResearcher v2.4 **Ground → Derive → Discriminate → Learn** 的历史证据重组；链接至 [实际计算图 G0–G6](../docs/98_端到端计算图与证据注解_20261025.md)。这是一份**研究知识图谱**而非新增任务单，不改变原有冻结协议、失败决定、授权或当前唯一 E0 方向。

**I34 · 科学解释复审（非新增实验编号）：** [Affine54实验边界与方法族科学判断](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)。`ALG`：实数tanh-GELU奇偶及全残差恒等式；`OBS`：匹配320案例局部输出误差改善；`NATIVE/SYS`：真实窄FFN加速；`VIDEO`：两个来源视频失真而正式质量未验收；`OPEN`：数值/容量/student前缀/下游传递/调度器原因。**局部SSE不是终端质量充分条件，但不能因此断言SSE优化本身是当前失败主因或整个Affine家族无效。** 该评议不是新的测量或Agent独立认可。

**原生paired-v2执行状态入口**：已冻结[固定v2范围](../config/e0_paired_full_depth_20261026_v2.json)并完成软件采集器及严格资产身份验证。授权、队列、真实配对site、失败/成功回执与是否可升级P23/P24，始终以[活动研究状态](ACTIVE_RESEARCH.json)和真实原生RUN/质量证据为准；本图谱的E34/E35只是它们完成时的**软件/源码阶段快照**，不会根据是否已启动任务自动升级为native机制证据。

**最新质量与全层限制（2026-10-26）**：[统一每生成视频指标与全40×4真实误差政策](agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)以[全模型六视频失败](20261026/E0_RECONSTRUCTED_VIDEOS.md)为直接依据。完成了全部160处原生替换，不等于已解释每一步各层真实Student前缀误差：早先partial-native只有block0/30，8.54%局部输出SSE来自Dense前缀保存panel。六视频已有逐帧RGB/encoded-luma/差分，现已另行补全运动/结构编码像素诊断（见下方E31），官方VBench/人工指标仍未完成；不能将旧视频结果标成符合新全指标验收。今后的每段生成要求SHA/全帧评估/真实成本/质量状态，无结果则保留为未完备证据。

## 0. 研究命题与证据单位

**研究终点**：在真实 Wan 四步蒸馏 I2V 上减少实际 FFN/完整生成时间，同时保留自由四步完整视频质量。**静态/动态、通道/组/行/共享、补偿/不补偿、量化格式**都是可能的表示或实现选项，并不是目标本身。[目标复位](../docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md) 已明确 R256 与 dense restoration 不是必需路径；[当前 E0 纲领](20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) 从 W1→GELU→W2 的有符号函数价值、时空条件与真实 GEMM work 出发。

历史数据分成五种**不可随意合并**的测量层级：

| 证据层级 | 测量对象 | 典型数据/合法结论 |
| --- | --- | --- |
| **S：源码/ALG** | 固定版本与实数恒等式、shape、QDQ顺序、输入时点 | [上游源码审计](../docs/08_上游源码校对_20261020.md)、[R12 scheduler/source](20261022/R12_E0_runtime_reconstruction.json)、[packed reference](../tools/packed_ffn_reference.py) |
| **O：被动保存张量/CPU回放** | 离线 FFN/QDQ、Gram、拟合、角度、oracle | [V5/AR1/AR2](../docs/04_前期探索与证据边界.md)、[E0 dense被动观测](20261025/E0_DISCOVERY.md)：本身没有原生删算/视频动作 |
| **N：原生有限动作** | 同前缀旧输入上的完整候选算子/写回、nextnorm、flow | [R7D1](20261021/R7D1_results.json)、[C1N0](20261023/C1N0_native_results.json)、[E0 finite](20261025/E0_FINITE_GROUP_DIAGNOSTIC.json)：不是自动的自由视频质量结论 |
| **V：自由生成与质量** | 动态轨迹中的真实全四步输出、配对视频及官方分数 | [UB1历史 formal20](../docs/28_固定步间计算预算对照.md)、[R16一个来源视频](20261023/R16V0_analysis.json)：RGB 保真、官方视觉质量、人类非崩坏分别解释 |
| **T：实际效率/物理执行** | sparse GEMM 的执行形状、FFN CUDA wall、generation wall | [UB1 complete FFN](../reports/worker_20261020/UB1_complete_FFN_speed.json)、[R17 单来源系统计时](20261023/R17E0_system_timing_prefix.json)：主 MAC、FFN clock、端到端不是一个分母 |

**统计单位**：在当前 E0，6 个已暴露来源（3 subject/3 background/0 camera），不是12条重复视频/120 scopes/数百万行；R16、R17 性能各只核对过一个旧来源；历史 UB1 原 formal20 与之后不相交补充20不能当同一个验收结论。以下编号 **H、A、Q、P、T、C、E、M** 是知识分类，不是原实验名称重命名。

## 1. H 族：早期动态掩码、原生轨迹与统计有效性

| 知识ID / 原实验 | 具体观察（对照/数值） | 当前能够成立的发现；计算图位置 |
| --- | --- | --- |
| **H01 A4/A5** | [原始历史报告重审](../docs/04_前期探索与证据边界.md)：A5 s1=.7,s2=.436 相对 dense aesthetic 降 **2.401886%**、imaging 降 **0.357735%**；另有某配置画质接近 dense，但 W1 完整执行后才 mask | *V/O*。局部 masked hidden 的画质接近可能存在；**这批实现不证实 W1 省算**。F04→F06、G10 |
| **H02 R3/R14 旧采集/V5** | [V5 来源、采样与数学边界](../docs/04_前期探索与证据边界.md)：D-Q 与 D-BF16 的 192 scopes/98,112行；D-Q 后段跨自然轨迹与BF16偏离，step0/block0 FFN cosine ≈.999、step3/block38 可至 .561–.760 | *O*。自然轨迹已分叉，不等于同输入 QDQ 因果效应；F00–F13、G00–G09 |
| **H03 R8 影子** | [历史复盘](../docs/56_全路线复盘与下一步机制优先级_R7.md)：某来源 shadow FFN 约 **1.75028×**，主 GEMM 约46.875% 稀疏，未写进生成主轨迹 | *T 的局部上界*，非 VIDEO 成功；D02/D05 |
| **H04 R11/R12/R15 闭环** | [旧路线报告](../docs/04_前期探索与证据边界.md)：R15 hidden半量+rank512 FFN 约 **1.479407×**，两来源 imaging 降 **45.8%** | *V/T*。该实例完整写回损失严重、速度也未达旧1.5× FFN门槛；不否定其它算子。F04–F13 |
| **H05 原H20/A100/环境与HiF4恢复** | [恢复基线](../docs/08_上游源码校对_20261020.md)：主运行 QDQ+BF16 GEMM，CUDA stream、PTQ adapter、运行配置相互绑定 | *S*。格式字符串不能把 QDQ 参考执行升级为真实 FP4 packed 性能；F02–F09、D06 |

## 2. A 族：组/通道支持、误差修正和替代表示

| 知识ID / 原实验 | 发现、比较器及原证据 | 结论所在范围；图位置 |
| --- | --- | --- |
| **A01 R19/R22/R23、R21B/R26** | [全历史复核](../docs/56_全路线复盘与下一步机制优先级_R7.md)：做过 W1/W2 2:4、SparseGPT式拟合、权重折叠；量化前后置换不能任意交换；重编码也改变保留权重 | *S/O/部分 native*。2:4及QDQ感知不是未经探索的发明；具体旧配方未获新自由视频验收。F02–F09 |
| **A02 AR1 随机组增益** | [旧证据](../docs/04_前期探索与证据边界.md)：对侧种子固定增益使随机组解析风险下降47.426–92.610%；原生 BF16 post-Q2 CPU 回放 SSE下降47.249–92.474%，8/8 scopes、32个 mask | *O*。对既定随机组补偿可大幅改善局部值；不是动态支持、视频或部署提速证据。F08–F12 |
| **A03 AR2 A/E/C 支持测量** | [R7梳理](../docs/56_全路线复盘与下一步机制优先级_R7.md)：使用 activation 能量 A、独立贡献能量 E、真实 signed group contribution C；C 的两个来源四个组合均满足已登记局部门槛 | *O/oracle*。输出相关量比廉价幅值更能表达组合风险，但 C 不直接在 W1 前可见；不是 selector 实际有效。F15、D08 |
| **A04 PR1/PR2/PR3 廉价代理** | [R7/R10](../docs/61_全路线失败归因与组合方案复盘_R10.md)：核代理 PR1、W1-SVD PR2、input PCA PR3，在旧局部要求下未普遍通过；PR3 rank512 Fit3/4、Validation2/4 | *O*。仅关闭被冻结的代理、输出目标和成本配置；不等于「上游信号无条件信息」。D00–D03 |
| **A05 RR1/C24** | [R7结果](../docs/56_全路线复盘与下一步机制优先级_R7.md)：R256+中心化、24来源拟合，正式20 imaging相对损失约 **5.113% / 3.264%**；比更早候选有局部改善 | *V*。增强校准与低秩修正仍未使具体版本过画质；R256 为候选边，不是 dense F04–F09 的父节点 |
| **A06 VM1 输出幅值/方差校准** | [R10总结](../docs/61_全路线失败归因与组合方案复盘_R10.md)、[原VM1试验文档](../docs/21_输出方差控制实验.md) | *V/O*。匹配总能量并不保证正确有符号方向、时间稳定性或正式视频验收。F10→F13 |
| **A07 WR1/JR1** | [文档18/22](../docs/18_保留非线性特征的权重重建.md)、[R10](../docs/61_全路线失败归因与组合方案复盘_R10.md)：保留 hidden 的读出重建/与 R256 联合的冻结CV未满足各自门槛 | *O*。旧低秩读出与复原族闭合，不证明所有新基/宽度等价形式的上界。F09 |
| **A08 LA1/FA1/FA2** | [局部审计](../docs/25_折叠低秩量化适配小试.md)、[全权重实验](../docs/26_全主权重量化恢复小试.md)、[R10](../docs/61_全路线失败归因与组合方案复盘_R10.md)：FA2局部约12–14%改善但不达CV门槛，无视频 | *O*。局部可调性与实际可迁移终端质量两回事；不能把小量参数更新自动视为低成本。F04/F09 |
| **A09 OP1/OP2/OP2A/OP2D/OP2R/OP3/NF2** | [正式学生数据与控制](../docs/44_UB1真实状态的原生教师采集.md)、[完整复核](../docs/45_完整原生数据复核与回放控制.md)、[重拟合](../docs/46_真实状态重拟合与导入延迟修正.md)、[历史问题](../docs/47_复用残差隐藏量的输出修正.md)：OP2 24来源×160=3840个位置，OP3 状态匹配重拟合未胜冻结UB1 | *N/O*。**数据完整性**与**算法收益**分账；旧记录缺 pre-norm/attention 的某些配对源字段，不能从文件名推测数据存在。D01/D08 |
| **A10 HP1/HP2/RB1/RB2/NP1/NF1** | [hidden贡献拟合](../docs/32_残差隐藏特征贡献预测.md)、[跨特征读出](../docs/33_跨特征小网络贡献预测.md)、[四模式路由](../docs/30_四支持模式路由可行性小试.md)、[失败对照](../docs/31_均衡四模式路由负结果.md)、[非线性输入](../docs/41_非线性输入基的贡献预测.md) | *O*。冻结局部可预测性实验未构成可部署的NetVOI或视频收益；「学习器」不是独立科学机制类别。D00–D04 |
| **A11 R8F1/R8F2/R9A1** | [原生仿射](../docs/59_仿射骨干的完整支持原生精度检查.md)、[HiF4 Hessian修复](../docs/60_HiF4仿射矩阵的分组Hessian量化修复.md)、[全路线机制审计](../docs/61_全路线失败归因与组合方案复盘_R10.md)、[商空间结果](../docs/97_历史结果检验_门控残差商空间_20261024.md)：部分近似的误差在校准空间外增加；R9有全216组 signed Gram，未保存足以计算新投影的 Gram矩阵 | *O/N*。具体旧编码/仿射配方无跨源稳健提升；全216组联合 Gram 的历史存在不等于已验证「投影后可删」机制。F15 |

## 3. Q 族：静态预算与系统可执行的强对照

| 知识ID | 结果与实际计算范围 | 解释 |
| --- | --- | --- |
| **Q01 UB1 原正式20工程验收** | [冻结动作与审计](../docs/28_固定步间计算预算对照.md)、[原始验收RUN](../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json)：四步216个whole64组依次保留54/108/108/162，即25/50/50/75%主 FFN 中间宽度；全160位置按行数计 **50%主MAC**；R256另算。aesthetic/imaging/motion 原20相对下降 **0.873850% / 1.344750% / 0.007365%**，逐项过历史2%门槛 | *V+T（原20）*：历史静态基线真实成立。不是输入动态策略，也不是新来源画质泛化。F04/F09/D06 |
| **Q02 UB1 FFN计时** | [完整 FFN wall](../reports/worker_20261020/UB1_complete_FFN_speed.json)：Dense **10177.569ms**、UB1 **6582.689ms**，比值 **1.546111×**（40 blocks×4 steps、含 R256 等原合同边界） | *T*：不是 complete generation speed；冷启动/驻留成本不同，应独立报告 |
| **Q03 UB1 不相交补充20** | [新20质量](../docs/29_冻结候选的来源分离补充验证.md)：imaging relative drop **3.166%**，超过原2% | *V*：原正式20质量成功与新20失败同时成立；没有在补充集上通过 |
| **Q04 BO1 时序预算选择oracle/UB2** | [旧二臂schedule oracle](../reports/worker_20261020/BO1_schedule_oracle_RUN.json)、[UB2视频与末步讨论](../docs/38_固定预算下恢复末步完整FFN.md) | *O/V，动作域限定*：预算在四步中如何分配是有价值控制；对两条历史臂的 oracle 不是所有动态时间计算策略的理论上界 |
| **Q05 64-group packing/物理tile支持** | [固定子集实现](../tools/packed_ffn_reference.py)、[channel granularity](../docs/17_通道粒度与常驻计时协议.md)、[真实布局问题](../docs/69_HiF4分组边界与空间动作的可执行布局.md) | *S/T 预检*：whole64 支持可以形成真正更小的 matmul，索引/搬运、QDQ与权重预算仍须跟踪；没测过的kernel不能以 MAC 推定时延 |

## 4. P 族：局部目标与真实传播不一致

| 知识ID | 定量结果 / 原证据 | 机制上的新知识 |
| --- | --- | --- |
| **P01 V5 后期发散** | [旧 V5](../docs/04_前期探索与证据边界.md)：同源 D-Q/D-BF16 在自然轨迹上越往后越可能分离；对不同位置非单调 | *O*。末端的输出差包含**前面状态改变**，不同于同前缀一次QDQ误差 |
| **P02 TF2 同状态流** | [TF2 完整记录](../reports/worker_20261020/TF2_same_state_flow.json)、[R7机制归纳](../docs/56_全路线复盘与下一步机制优先级_R7.md) | *N*。同状态对照隔离了一部分轨迹混杂；不能直接替代真实自由行进的误差 |
| **P03 R7D1 两层有限响应** | [R7D1结果](20261021/R7D1_results.json)：四个暴露来源、24完整生成诊断，**3/4** 来源的局部较差/flow较好动作排序不同 | *N*。误差方向与后续传播权重不能用 local gated SSE 判断；G20→G21→G07 |
| **P04 R7 有限方向耦合** | [C0重算](20261023/C0_coupling_offline_results.json)、[R7D1](20261021/R7D1_results.json)：两个固定恢复动作的下游方向交叉项约 **−11859.54**，非线性传递项约 **+11657.10**，净作用约 **−202.44** | *N/ALG*。净交互小可由两个巨大项抵消产生，不能得出各块扰动独立 |
| **P05 R7D2/R11T1 局部/flow目标拟合** | [R7D2原生](20261021/R7D2_results.json)、[same deployment budget](../docs/64_相同部署预算下的局部与真实flow目标对照.md) | *N/O*。局部教师误差和真实 flow 优化目标有实际失配；仅更换损失而不识别可用算子及泛化，不自动带来视频结果 |
| **P06 R12 D0 等成本末步旧算子** | [R12 native](20261022/R12_D0_native_results.json)、[商空间回顾](../docs/97_历史结果检验_门控残差商空间_20261024.md)：四开发来源、旧三动作，oracle比最强固定仅改善 **0.2319275% flow SSE**，但仍较 UB1 差 **2.5741%** | *N*。这是旧动作集合的 headroom；不排除新无补算组算子 |
| **P07 R14N0 四token stripe** | [native](20261022/R14N0_native_results.json)、[R15纠偏](../docs/78_指标审计与已有实验映射_R15_20261023.md)：UB1采样SSE 8183.32；静态10,206.34；norm-route10,733.59；sample oracle9747.50。相对 UB1 多余损失61.33%归于动作族不足，38.67%归于selector regret | *N/O*。先改善可用的物理算子族，不能只调selector；该比值只针对冻结动作与4-token样本 |
| **P08 C1 post / nextnorm 排名** | [C1历史二目标审计](20261024/C1_boundary_target_audit.json)、[旧商空间汇总](../docs/97_历史结果检验_门控残差商空间_20261024.md)：72位点、旧swapA/swapB，约 **7.10%/7.50%** 每行相对UB1优劣符号在 post 和 nextnorm 翻转 | *N*。同一次有限动作「写回误差」不等于「下一归一化作用」；不能把每行翻转加总称作整site动态收益 |
| **P09 C0 归一化旁路反例** | [原合成结果](20261023/C0_coupling_offline_results.json)：相同post-Q1 q，放大pre-LN残差尺度时最优旧 A/B 动作**翻转** | *ALG/synthetic*。上游残差状态能提供下游价值信息，**不证明** Wan 中翻转频率或盈利路由 |
| **P10 C1 whole-site NetVOI** | [真实动作](20261023/C1N0_native_results.json)、[source-held-out分析](20261023/C1N0_analysis.json)：6来源 ×12site =72，旧补偿动作UB1/swapA/swapB；native nextnorm SSE约 **153.19m/156.84m/156.23m**，site oracle **72/72 UB1** | *N*。对这三个**整site**动作，现有尺度/attention额外信息不能改善site选择；也不推翻 P08 每行作用分歧 |
| **P11 C2 环境失败** | [C2N1故障](../docs/91_C2N1导入路径故障与环境修复验收_20261024.md)、[目标复位](../docs/93_FFNDS_目标复位_补算不是研究目标_20261024.md)：后续部分启动停在 **0 native forward / 0 video** | *工程状态*。不得登记为研究方法的负实验或视频质量测量 |

## 5. T 族：动态路由的局部收益如何折损到系统

| 知识ID | 原始结果 | 由物理执行与目标错位得到的结论 |
| --- | --- | --- |
| **T01 R16V0 160site路由** | [完整分析](20261023/R16V0_analysis.json)：旧一个来源、四个原生 block20 step，同前缀 dynamic gated-R256 row routing，local SSE比两个固定行控制中的最好者改善 **6.1276%**；相同来源完整四步视频对 dense 的 pooled RGB MSE 却比最佳固定高 **8.2257%** | *N+VIDEO*。在局部可预测性成立时，静态控制在下游轨迹仍可能更好；仅RGB保真，不宣称主观画质差 |
| **T02 R17E0 系统计时前缀** | [实际一源5技术重复/arm](20261023/R17E0_system_timing_prefix.json)：UB1 full160 FFN **1.5333×**、full generation wall **1.07568×**；dynamic_residual **1.25665× / 1.04469×** | *SYS*。路由、QDQ、gather/scatter、attention/VAE/视频存盘使 FFN 理论工作量与端到端收益脱钩；没有23个新来源正式九维质量通过的结果 |
| **T03 质量指标体系转型** | [固定九维 I2V/机制/可靠性](../docs/77_FFNDS学术指标族与评测协议_20261023.md)、[评估来源准备](../docs/82_新来源九维评测与重复原生测速_20261023.md) | *协议*。原工程三维 aesthetic/imaging/motion 与新九维/ROI/时序指标是不同覆盖；新来源质量仍需以**实际已完成的原生指标文件**为准，不能将计划中的45次metric调用写作结果 |
| **T04 视觉异常机制** | [用户观察与历史复盘](../docs/04_前期探索与证据边界.md)、[时序指标建议](../docs/77_FFNDS学术指标族与评测协议_20261023.md)：早期视频的主体有时稳定但背景光照和小物体闪动 | *描述观察*。应以稳定ROI亮度差分、motion compensated残差、背景一致性和paired视频区分来源；尚无可归因于特定通道的因果证明 |

## 6. E 族：2026-10-25 本征 FFN 结构发现（当前重点）

**最新系统证据 T05/T06：** [18视频/72latent恒等的dense常驻](20261025/E0_DENSE_WEIGHT_RESIDENCY.md)在同卡80GB A100热态75.17/75.37→50.42/50.79秒，约1.49×，物理采样显存峰值77.08GiB。算术与1600linear/QDQ不变；两暴露图片/seed组合，不是18独立来源或普适倍速。另一方面，[8案例孤立折叠计时](20261025/E0_AFFINE_FFN_DISCOVERY.md)的BF16矩阵形成约4ms、应用约8ms，但未融合旧full-M导数获取约54ms，总和约为原生after-Q1 FFN的1.26–1.28倍。后者是128行重复到31668的形状微基准，不是自然全M或完整视频，也非融合/固定斜率的成本下界。图锚点P12/D09与F06。

以下 E0 统一源为 **[6个已暴露原生来源](20261025/E0_NATIVE_OBSERVATIONS.json)**。被动 dense 共12次完整生成，捕获 120个位置×冻结行；有限组原生干预共30次完整四步生成，另10次跨worker复核在两来源做到210项hash一致；**原生干预仍完整运行密集主 GEMM**。不可把六个来源展开成数百个独立场景，也不可从这些实验报告部署加速或正式视频质量通过。

用户随后新指示继续并允许GPU，新增两批相互衔接的证据：[全hidden/完整M](20261025/E0_INTRINSIC_FULL_HIDDEN.md)为六来源180个保存数据案例+source0/12两次原生被动dense生成；[连续共享列/量化闭包](20261025/E0_SHARED_TILE_PRE_W1.md)又在source0/12各生成一次原生dense，保存真正连续的3×M128操作数，并完成额外GPU保存数据对照与原Python量化器合成反例。**新增4次原生dense被动生成，不是4个新独立来源，更不是4次稀疏视频；原12次被动、30次finite、10次跨worker记录不变。** 图解参见 [G4决策与依赖图](../docs/98_端到端计算图与证据注解_20261025.md)。

| 知识ID | 固定比较与数值发现 | 机制/强竞争解释；计算图 |
| --- | --- | --- |
| **E01 真正的非线性前稀疏性** | [E0 dense观察](20261025/E0_DISCOVERY.md)：保存操作数 GELU精确零 **0–0.0617%**，Q2后**5.42–14.00%**；Q2新置零值占原preQ2能量约 **0.0569–0.2863%** | *OBS*：零值主要是 Q2 格式结果，不是结构whole64闲置；F06→F08 |
| **E02 量化和表示轮廓** | [E0](20261025/E0_DISCOVERY.md)：组能量粗轮廓在Q2前已有差异，Q2对既定二阶解释模型增量平均 **0.08–0.19个百分点**；某source8极端坐标影响均值，已保留 | *OBS*：信息可能来自输入与 W1/GELU 的条件结构；不应把「量化附赠元信息」视为充分统计。F04–F08 |
| **E03 W2 signed方向与局部LN径向** | [E0 residual geometry](20261025/E0_RESIDUAL_GEOMETRY.json)、[完整说明](20261025/E0_DISCOVERY.md)：block0五个固定组、理想连续LN切向投影约消去 **19.82%** 组方向能量；post前 **24.71%**、leave-one-out **19.88%**，通道均值项仅0.0226% | *OBS/ALG*：真实存在方向对齐，主要径向；identity旁路与后续attention可读回，既非零视频作用也非可省计算比例。F11–F13 |
| **E04 pairing对照** | [径向配对](20261025/E0_RADIAL_PAIRING.md)：block0 raw配对19.79%、错误token配对20.54%；按位置中心化后8.84% vs3.59%；其它五来源共有方向外推cos≈**0.99645** | *OBS*：共同/固定方向解释 raw 对齐中的很大部分；不能把它全写成 token-specific 选择信号。F15 |
| **E05 几何指标是否真的新增信息** | [E0 ranking](20261025/E0_GEOMETRY_RANKINGS.md)：block0单组能量选择与有限理想LN最优在 **118/768=15.36%** 行不同，能量选择额外 ideal LN error **1.991%**；而中心化角度在全部 3072 行无最优组差异 | *OBS/ALG*：该局部几何基本由已知 angle/LN 原理解释；发现不是一个独立原创新评分准则。F15 |
| **E06 两组/三组交叉项** | [组合矩](20261025/E0_JOINT_AND_TEMPORAL.json)、[原文](20261025/E0_FOLLOWTHROUGH.md)：block0 按最小单组能量独立选两组/三组与真正组合 ideal LN 最优不同 **28.91%/30.99%**；投影后正交叉项更普遍 | *OBS*：单元素重要性不能求和近似一切组支持；要用 signed Gram 与候选整体算子对照。F15 |
| **E07 原生舍入是一个新的相互作用层** | [原生finite summary](20261025/E0_FOLLOWTHROUGH.md)、[底层 JSON](20261025/E0_FINITE_GROUP_DIAGNOSTIC.json)：固定两组作用组合非加性相对加和能量，原FFN输出 **1.76–1.93%**，BF16 residual writeback **26.64–27.87%** | *NATIVE*：不能拿FP64线性叠加与理想LN算式代替原生有限动作；对照本轮仍在dense算W1/W2。F10–F13 |
| **E08 四步有限传播与排名** | [六来源原生四臂](20261025/E0_FOLLOWTHROUGH.md)：部分组局部和最终flow排名在 **3/6** 来源不一致；[shared reference 6距诊断](20261025/E0_SHARED_REFERENCE_MOMENTS.md)：非负矩拟合的final-flow平方距离误差 **2.24–20.01%** | *NATIVE/OBS*：既不能以局部norm排序替代后续作用，也不能因终端合成项相消便断言误差独立。G20→G09 |
| **E09 跨步高角度不能直接缓存** | [E0 joint temporal](20261025/E0_FOLLOWTHROUGH.md)：同专家block39 gated输出 cosine **0.963**，旧Y直用 relative SSE **37.3%**，后验单标量7.2% | *OBS*：幅值/共同分量漂移主导raw相似；后验最佳标量用到当前真值，不能在线部署。G02/D08 |
| **E10 固定跨来源幅度规律** | [source-held-out校准](20261025/E0_TEMPORAL_CALIBRATION.md)：block39旧值reuse **37.34%**，按其它五源校准固定scale **8.32%**，固定scale+5120通道bias **4.35%** | *OBS*：共同漂移可跨已暴露来源转移，属于强**固定方法控制**；没有真实cache kernel/视频加速。D00/G04 |
| **E11 共同分量与空间位置变化** | [E0 temporal](20261025/E0_TEMPORAL_CALIBRATION.md)：block39当前Y能量中按32位置取均值的共同向量占 **88.55%**；固定仿射中心化误差相对中心化目标仍 **39.25%**，高噪声60.54%、低噪声17.96% | *OBS*：全输出高余弦受共同分量主导；相机/物体/时空功能未由这些值直接识别。F15/G10 |
| **E12 数值/QDQ位语义** | [E0 observed](20261025/E0_DISCOVERY.md)：240原生QDQ引用样本保留输入负零符号后BF16 bit一致，先前5730个零位符号不等、数值相同 | *SRC/OBS*：signed zero/scale/cast顺序是 bitwise fidelity 的必要边界，不是新「质量损失」或4-bit硬件速度证据。F02/F03/F07/F08 |
| **E13 全hidden创新与函数支持** | [完整13824通道/216组](20261025/E0_INTRINSIC_FULL_HIDDEN.md)：六来源180案例，25%单通道value对角排名的同专家创新SSE **2.44%–7.07%**；同预算连续64组后验能量排名 **32.42%–54.67%** | *GPU保存操作数/ALG*：创新比分别的绝对输出集中，但当前对角/组排名不是联合最优或前置selector；不能把通道集中直接变为视频质量结论 |
| **E14 完整M支持与原布局** | [两来源原生被动](20261025/E0_INTRINSIC_FULL_HIDDEN.json)：完整M=31668，原系数/创新每行12.5%保留时，所测 **M16/64/128×K64全部非空**；8latent、2视频、40FFN采样哈希与原dense一致 | *NATIVE被动支持*：保持这些逐行集合时原块无空闲；不排除另一共享任意列集合、packing、时空重排或条件表示。未测稀疏kernel速度 |
| **E15 W2后验更新的四步代价上界** | [本征报告计算式](20261025/E0_INTRINSIC_FULL_HIDDEN.md)：若只在同专家第二次调用减少W2，ρ=25%时最多减少 **18.75%总投影MAC** | *ALG/边界限定*：W1/GELU/Q2已付成本，前置选择或其他表示才可能突破此具体策略的上界；不是所有FFN方法的上界 |
| **E16 连续token共享列与前置观察（G4/P07–P08）** | [两来源M128跟进](20261025/E0_SHARED_TILE_PRE_W1.md)：25%列的前置micro16 GELU/value在Block39创新SSE6.55%，中间观测层42%–47%；共同创新能量首尾65%/78%、中间10%–14% | *NATIVE被动字段/GPU保存诊断*：共享后验也损失逐行优势，观察器不是唯一瓶颈；原raster分组不等于voxel重排，未获速度/质量结论 |
| **E17 W1/W2依赖边界不同（G4/P02–P11）** | [原量化组闭包](20261025/E0_SHARED_TILE_CLOSURE.json)：25%任意共享列触及所有216组；[原始Python合成](20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)只改一输入却改变64个Q2输出 | *SOURCE/保存数据/合成*：直接原生求值的W1预算不能按W2选择率计算；不是所有认证max/尺度缓存/新表示的下界；新函数/执行单位仍开放 |
| **E18 GELU、时空均值和量化的三个不同非交换环节** | [连续token机制分解](20261025/E0_SHARED_TILE_PRE_W1.md)：块均值的旧/新preactivation可能不变，但平均GELU发生变化（合成对照）；在五个观测block，线性半项与GELU余项经W2/gate投影的平均cosine **−0.64到−0.82** | *ALG/SYN/OBS*。不能用线性块均值替代平均GELU，也不能将后Q2差分的支持当成pre-Q2修改列的精确支持。实数GELU满足\`φ(a)-φ(-a)=a\`但原QDQ/BF16并不自动遵守变换等价。G4c |
| **E19 真实前置信号需要额外W1；双投影预算不同** | [实际共享列及原布局](20261025/E0_SHARED_TILE_PRE_W1.md)：原四步每专家第二次调用可用一次同专家历史；如\`ρ_W1≈1, ρ_W2=.25, o=1/16\`，160个FFN的总主投影理想MAC节约 **17.1875%**，理想主投影加速≤约 **1.208×**；若新算子真的同时缩到\`ρ_W1=ρ_W2=.25\`，才是**35.9375%** MAC节约（含相同o） | *ALG/预算*。独立书写W1、W2、selector、QDQ闭包、缓存和最终壁钟；不是新核计时或质量门槛通过。G4d |
| **E20 E0 技术可复现性** | [original 12 gens](20261025/E0_NATIVE_OBSERVATIONS.json)、[cross worker](20261025/E0_CROSS_WORKER_REPRODUCTION.json)：被动hook没有改变视频/latent SHA；10次跨worker、210项配对hash一致 | *NATIVE控制*：建立此批数据可信度；不是新的独立scene sample，也不是新的算法收益。G00–G10 |

### 历史增量核查（截至79bc9ea）：E13–E20未变，P14仍为假说

以先前计算图合并 [8df09a6](https://github.com/Qesire/FFNDS/commit/8df09a6dcb82d256a2593ef5158441f78b771ff0) 对比 FFNDS [79bc9ea](https://github.com/Qesire/FFNDS/commit/79bc9eae9f2d7c694f5536a2aa860381a9c80ee4)：**新增26次仓库提交集中于 Graph-Agent 协作、23节点初始注册表及静态测试，并无新的原生/保存数据实验报告**。该审计时点的测量依据仍是上文E13–E20；新增 [G4e P14 OPEN](../docs/98_端到端计算图与证据注解_20261025.md) 是 [一份PROPOSED、无独立评议的共享输出函数基假说](method_discussion/graph_annotations/E0/E0_GRAPH_SEED_FUNCTION_BASIS_20261025.json)，**未升级为E21证据或认可方法**。

当前可以直接从已测连续M128的共同创新能量推出一个新的**代数推论，不是新增测量**：若将每个token的创新输出都替换为本tile的后验均值向量，block0/10/20/30/39 对同专家创新能量的相对SSE案例均值依次为 **34.7%、90.2%、86.4%、85.7%、22.2%**。由于任何128×5120 token-output矩阵天然rank不超过128，单凭“维数低于5120”不构成可省算证据；需要有意义的低k谱尾、跨来源持出和足够便宜的**W1前可获得**表示。归属：E16已测共享能量 + 精确正交分解（ALG），不是新视频、kernel或验证的rank-k压缩率。新反解释为**真实token特有输出变化有较高有效秩**，参见 [当前 E0 分歧审查](20261025/E0_INTRINSIC_DIVERGENCE_AUDIT_20261025.md)。

**最新研究议程检查**：[2026-10-25 affine/非线性/同值权重常驻方向审查](20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md)，对应 [G5图层](../docs/98_端到端计算图与证据注解_20261025.md)。全层旧导数统计、affine局部重建、GELU奇偶身份、系统级同值常驻是不同对象；新的共同科研目标仍是有质量条件的FFN必要计算缩减。现有报告归属不变，未将G5开放假说标作已完成新视频算法。

## 7. M 族：将所有结果组织成可检验的机制，而不是路径偏好

后续 E21–E23 接回同一图谱：

| 知识ID | 固定比较与数值发现 | 机制边界 |
|---|---|---|
| **E21 全40层共享斜率** | [全层保存诊断](20261025/E0_AFFINE_FFN_DISCOVERY.md)：旧full-M均值斜率在uniform32上，0→1有0/40层输出SSE＜1%，2→3仅1/40 | *OBS*：full-M只指旧导数统计，误差仍在128/32行；不能以末层低误差代表模型，也不排除其他条件表示。G02/F06/F09 |
| **E22 函数基项与匹配删组** | [40案例绝对affine](20261025/E0_AFFINE_FFN_DISCOVERY.json)：另一来源固定affine＋27组uniform32 SSE11.27%，同MAC直接67组43.13%；center128为12.49%对43.64% | *ALG/OBS*：D×D基项恰等于40/216原投影组；部分非线性余项仍大、原Q2依赖/packing/下游未知；不声称发明TARDIS已有常量折叠或候选质量通过 |
| **E23 半斜率与组Gram反例** | [独立审查](method_discussion/discoveries/E0/independent-affine-residency-audit-20261025.md)：对称输入的实数tanh-GELU OLS斜率1/2；余项10v,−10v,v否定组能量排序净误差最优 | *条件ALG/反例*：将E18已有GELU奇偶关系推进为竞争预测；稳定斜率不证明窄热区，posthoc非净误差oracle；未宣称Wan对称配对或native Q2恒等。F06/F08/F15 |

| 研究问题 | 可由哪些已完成实验确认的结构推出新问题 | 真正相互竞争的机制解释 | 成本最小的下一项**信息需求**（非运行授权） |
| --- | --- | --- | --- |
| **M1 F1: signed contribution & representation** | E01/E02/E03/E06，A02/A03/A11 | (a) 静态大值方向、(b) 输入相关组合的 signed cross terms、(c) LN/共享方向弱可见性；三者不等同 | 保存组后的原生可用时点，使用已保存 W1/GELU/Q2/W2 有符号值分解；先比较 angle/signed Gram 与静态等成本 |
| **M2 F2: step–space innovation** | E04/E09/E10/E11/E13/E16/E18，H02 | (a) 按step固定共同幅度漂移、(b) 逐位置条件变化、(c) 仅Y共同均值使raw cosine高 | 针对同专家同位置，以 source-held-out 固定预测 vs 当前输入统计；明确缓存值及单层真实可减的2/160工作比例 |
| **M3 F3: functional propagation** | P03/P04/P08/P09，E07/E08 | (a) 仅局部LN短程不敏感、(b) 写回舍入非加性、(c) 下游状态/attention传播方向改变 | 冻结**同一原生前缀**与有限真实输出扰动：逐段比较 post、nextnorm、nextblock、flow及完整视频的作用向量；当前六源实验仅检验特定组 |
| **M4 F4: physical compute** | H01/H03/H04，Q01–Q05，T01/T02，E14/E17/E19 | (a) 算子确能缩窄两次主GEMM、(b) 零值仅发生在全部W1付费后、(c) gather/QDQ/selector/weight访存主导 | 先写明 W1前/后可用输入、物理tile/packet格式及每一支的实际工作量；以 R17 full generation 作不能省略的比较维度 |
| **M5 Q1: cheap information / NetVOI** | P08/P09/P10，T01，E04 | (a) residual旁路含真实下游增量信息、(b) 信息仅预测本地hidden、(c) 增益不足以改变动作或付不起cost | 必须先定义**新直接算子**，计算其完整 action value 后比较 best fixed、source-held-out、shuffle 和 selector cost；旧 C1 72/72 不支持泛化断言 |

### 三个特别容易错误迁移的“反直觉结果”

1. **局部更好 ≠ 终端更好。** R16 中动态局部 SSE 改善 6.13%，自由视频RGB保真却较固定差 8.23%；E0有限原生组还发现三来源的 local/flow rank disagreement。因此局部“预测成功”的标签不能单独驱动下游选算。
2. **方向小响应 ≠ 可删除工作。** E0 block0 大量 raw group方向位于当前LN弱敏感几何，但 wrong-pair对齐仍高、固定共同方向强；C1旧原生全site组动作下一norm均非零，native BF16写回非加性又增大。angle/LN已有严密先例；需要问真实函数改变、时序作用和物理执行能否共存。
3. **FFN 真加速 ≠ 整体真加速，更不等于画质达标。** UB1历史原20可同时得到较好三维质量与FFN1.546×，新20 imaging 3.166%失败；R17 full generation wall 仅1.07568×。旧成功是部署强对照，不是可直接迁移的论文结论。

## 8. 量化信息、算子身份及现行研究进度

原始 HiF4 code/dequant metadata、MXFP4 E8M0／NVFP4 scale 等能在对应 kernel/权重表征中提供**幅值或范围估计**；新增原HiF4根尺度闭包还证明，元信息可能在*量化运算中制造跨列依赖*，不只是供selector读取的免费幅值信号。但“元信息的存在”“W1前可读”“净决策收益”“物理skip tile”“视频质量”是五条不同边。[本地格式实现](../upstream_audit/source_deltas/Wan2.2-I2V-A14B-W4A4/files/CODE_PATH_ANALYSIS.md)、[现行首个任务](20261025/E0_FFN_FIRST_TASK_20261025.md) 要求先研究本征FFN signed计算及哪些工作值得省，再考虑元信息是否能作廉价观测。历史的 HiF4 QDQ 并未证明 MXFP4/NVFP4 真实packed kernel 上同样的信息读取代价或计算收益。

后续对每个新解释应填下列小型决策记录（模板不是实验提交）：

**原始观察 → 对应图边 → 等价/近似数学对象及假设 → 真实已算与需额外算的信息 → 区分性预测（对抗静态/既有先例）→ 最小源码/保存数据检查 → 达成后才考虑原生有限对照和自由视频。**

其中 **E0是唯一当前主动科学问题**；其他 A/Q/P/T/H 等是历史证据库，不复活已经关闭的实现枝节。[AutoResearcher 构想/研究规范](../AGENTS.md)、[active state](ACTIVE_RESEARCH.json)、[实验暂停](20261024/EXPERIMENT_PAUSE.json) 优先于过去 handoff 中的 ACTIVE / NEXT 字样。

## 9. 本档案的检索覆盖与证据链

本轮核对了：AutoResearcher v2.4 模型/问题表示/演绎/证据规则；FFNDS 主入口与 2026-10-25 E0 理论、被动、有限、跨源校准；实际 LightX2V 固定 Transformer/scheduler 与 HiF4适配；已有 **[28个重点原始文件](EXPERIMENT_EVIDENCE_INDEX.md)** 所代表的核心对照，以及 [R7/R10/前期/新指标](../docs/61_全路线失败归因与组合方案复盘_R10.md) 的按族复核。完整 [1348项文件名库存](experiment_evidence/FILE_INVENTORY.json) 用于**定位**，并未逐项独立重新执行、重读所有大张量或对所有文件做逐行复核。本档案链接到原报告/RUN JSON，尽量不让二次综述替代直接数据。

每条主张应保留 **source/checkpoint/hash → action operator → independent source groups → comparator → measurement horizon → physical cost (W1/W2/QDQ/observer分别记账) → observed number**，区分 synthetic、理想 FP64、native BF16、自由视频及硬件计时。版本识别从 [真实源码](../docs/98_端到端计算图与证据注解_20261025.md) 进入，实验状态从 [当前主入口](ACTIVE_RESEARCH.json) 进入。原实验、不变资产和归档均未被改写。

## 2026-10-26：E24–E27 函数表示与真实保留非线性

| 知识ID | 新证据 | 可成立范围与下一竞争解释 |
| --- | --- | --- |
| **E24 半斜率实际检验** | [320案例](20261026/E0_FUNCTION_RECONSTRUCTION.md)：uniform32半斜率84.80%、Q2OLS21.07%、高斯统计增益22.47%；偶部/净连续输出能量6.14、奇偶cosine−0.927 | *OBS/条件ALG*。奇偶恒等式成立但偶部不小；支持统计增益为有用解释，不证明窄热区、高斯或独立噪声。F06/F08/F15 |
| **E25 固定输出基和系数预测** | [480案例](20261026/E0_FUNCTION_RECONSTRUCTION.json)：跨源center128 rank64真实系数创新SSE72.94%；PCA-ridge86.44%、精确RRR86.45% | *OBS/ALG*。表示覆盖和迁移已是瓶颈；限制具体基/面板，不否定全部P14。未门控正交分解与门控比例分开。G18/P14 |
| **E26 四源校准的非线性重构** | [五层40案例](20261026/E0_FUNCTION_RECONSTRUCTION.md)：affine54 ridge5.89%对直接94 ridge8.47%，均为94/216理想MAC；错配对照更差 | *OBS*。同设置匹配组比较成立；相对两来源输入预测的优势有校准混杂，独立挑战促使新增匹配输入控制。新W2/bias是拟合参数。F04/F06/F09 |
| **E27 局部原生折叠** | [两层16案例](20261026/E0_FUNCTION_RECONSTRUCTION.json)：144dense＋288partial FFN，432native Q2；resident after-Q1形状计时52.69→20.77ms | *NATIVE-LOCAL/SYS*。重复160行到31668，FP32准备/BF16新矩阵未再次HiF4量化；非自然全M/视频/整网结果。block30误差仍大，F12/F13/D09 OPEN。P12 |

[独立结果复核](method_discussion/discoveries/E0/independent-function-basis-results-audit-20261026.md)保留实际SHA、反预测与校准条件挑战；[图G5b](../docs/98_端到端计算图与证据注解_20261025.md)按这一范围接回。原P14提案及E21–E23原记录不改写。

## 2026-10-26：E28–E30 匹配控制、真实视频失败与功能问题改写

| 知识ID | 新证据 | 可成立范围与下一竞争解释 |
| --- | --- | --- |
| **E28 全40层同校准对照** | [320案例](20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)：四来源256行、持出0/12；affine锚定输入15.35%对affine54 8.54%，320/320较低；旧五层标量差0 | *OBS*。这一固定正则下覆盖/先验不足以解释全部差距；不是任意线性类下界，也不匹配输入线性与非线性在线MAC。P20/P21 |
| **E29 真实全模型固定原型** | [六完整视频](20261026/E0_RECONSTRUCTED_VIDEOS.md)：原dense两视频/8latent exact；warm affine54 FFN9.707→4.449s，wall50.750→45.313s；新160包19.7GB | *NATIVE-ACT/VIDEO/SYS*。自然全M、全部160处、同卡预载强dense。whole重影，affine模糊/动作改变，均无画质验收。setup额外约237s，QDQ调用没少；理想MAC43.52%与总wall1.120×分开。P19/P22/F12/F13/D09 |
| **E30 功能/轨迹对象改写** | [原Wan源码与数学](method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md)：原生混合前缀/后缀的望远镜恒等式；理想Wan受扰目标vhat(z+delta)=v(z)+delta/h | *ALG/OPEN*。保留真实step_pre/step_post dtype与舍入（E33纠正旧FP32常驻说法），真实QDQ不可默认光滑；teacher仅离线可见。输入偏移、有限下游传递、跨步纠错不同预言。核对BRECQ作者代码、CTEC与TCEC近邻；没有新增训练/模型范围 |

四来源缺失35层补采得到560份payload；首source16进程生成前退出（原因未确认），独立冻结恢复一次通过；共四条实际dense生成，另扣一次生成前失败，实际分配峰四卡。旧12/40生成和原协议不变。固定范围全部完成，无GPU任务。E0继续开放、无方法选定。

## 2026-10-26：E31–E33 已有视频时序、完整状态账本与数值边界

| 知识ID | 新证据 | 可成立范围与下一竞争解释 |
| --- | --- | --- |
| **E31 质量异质性** | [六已有视频补评](20261026/E0_EXISTING_VIDEO_QUALITY_BACKFILL.md)：486帧/480帧对；修正光流8bit尺度，affine source0低运动估计、source12高帧差且末窗口光流反高；高Laplacian可与失真并存 | *POSTHOC-VIDEO*。两个暴露来源，未测真光流/语义背景/官方/用户完整播放；双向一致性不等于准确。四处reference model_digest勘误与原冻结报告并存。P22/P25 |
| **E32 有符号完整状态账本** | [工具](20261026/E0_FULL_DEPTH_TRACE_TOOLING.md)：四native FFN/四实际写回、完整40×4覆盖，operator-prefix交叉项、完整四维latent、六边界哈希跨步连续性；10项软件测试 | *SOFTWARE/ALG（初始阶段）*。10项夹具保留SYNTHETIC标签；后续producer与时空分层已在E34实现，**不代表任何真实site成功**。P23/P24只有原生验证后才可升级 |
| **E33 dtype与原生接口纠正** | [独立源审阅](method_discussion/discoveries/E0/independent-full-depth-native-interface-20261026.md)：继承step_pre可把FP32 latent转BF16，step_post再回转；单pipeline同block buffer交错两轨可行切口，25源码/7API静态检查 | *SRC/CONDITIONAL（初始源码设计）*。原源码的step_pre BF16条件推断不是实测迹线；已知40.70GiB只含部分权重。bounded residency后来在E34完成接线，但真实GPU峰值和中性仍要以RUN验证；本条不代表当前执行状态 |

**历史六视频**的官方/人工质量仍NOT_RUN；E31–E33对应的CPU/设计工作没有启动模型。后续另有严格固定的paired-v2独立授权和执行状态，见[活动状态](ACTIVE_RESEARCH.json)。原始六视频、RUN及冻结协议不因本文变化。

## 2026-10-26：E34 原生采集接线与有限精度写回反例

[E34实现](20261026/E0_PAIRED_NATIVE_IMPLEMENTATION.md)交付双scheduler/同buffer逐block两轨、可重入四FFN与四真实写回、当前block dense FFN驻留、唯一有限attempt账本及失败视频全帧取证。17项CPU SYN软件测试、39份源码及80份参数包checksum在初始实现阶段通过；当时真实native为0。三pair/六轨迹随后经v2独立冻结与授权，实时进度只以[活动状态](ACTIVE_RESEARCH.json)和RUN判定。初始BF16 latent与四步入口/出口哈希保持真实dtype。

有限精度反例：相同g=1、yD=0、yS=2^-9，在r=1的BF16写回误差0，在r=0误差2^-9；因此raw FFN差不变也可能改变真实写回差，student门控误差变大不必只有FFN输入失配这一解释。*SYN/ALG*，不是Wan已测机制。P23/P24仍OPEN，诊断wall包含额外observer/影子/连续/归约/双解码，不能当部署提速。旧原型质量失败和24既有scope不改。

## 2026-10-26：E35 新入口的严格视频身份与产物覆盖

[新paired严格层](20261026/E0_PAIRED_NATIVE_STRICT_EVIDENCE.md)从独立native回执与实际目录比对两视频清单，并以真实生成assets/operator重算dense model_digest；拒绝误继承候选digest、漏视频并降expected计数、删reference及无权威清单。8项新软件测试+17项paired软件测试通过，两个实际旧dense身份只读复核一致。*SRC/SOFTWARE*，仅新paired入口，原gate/旧报告字节不改，不追加质量验收。另冻v2替代未授权未执行的v1；软件门禁测试阶段模型/GPU为0。v2后续已登记有界授权/启动，不能把这里的0当作实时作业计数；P23/P24是否有有效原生迹线由实际RUN和审查决定。
