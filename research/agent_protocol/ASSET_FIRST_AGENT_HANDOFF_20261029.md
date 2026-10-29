# FFNDS · 开放式证据驱动的统一Agent交接与研究队列

**现行METHOD HANDOFF · 2026-10-29 · 探索与部署分别验收。** 科研协议：[AutoResearcher技能](../../.agents/skills/ffnds-paper-discovery/SKILL.md) → [资产与四支机制推导](../20261029/ASSET_FIRST_STRUCTURE_DISCOVERY_AND_METHOD_DERIVATION_20261029.md) → [数据交叉复算](../20261029/ASSET_FIRST_METHOD_DERIVATION_RECALC_20261029.json) → [视频终端证据](../20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)。运行状态以[ACTIVE](../ACTIVE_RESEARCH.json)为准；历史[2026-10-28任务台账](CURRENT_TASK_LEDGER_20261028.json)是任务状态登记，不再定义科研优先级。任何Agent无需从旧M-B末层handoff开始。

**本交接的“资产先行”指先了解已有证据、节约重复实验；并不要求在现有资产内完成研究。** [问题驱动开放采集](EVIDENCE_DRIVEN_ACQUISITION_20261029.md)及[新增采集队列](ACQUISITION_QUEUE_20261029.json)是**并行P0科研工作**。若缺独立source、40层×4步的时空匹配token、学生自身状态、真实Q1/Q2动态code/scale、额外领域/视频行为数据，立即建设所需数据producer和新source cohort，完成真正的原生采集后继续推导；采集可以先于具体算法。允许引入新的课题与算子，S1/S4/S2/S3及既有rank960并不是方法空间边界。

**本次用户指定的实际下一步（优先执行）**：[S2真实时空+Dense/Student状态采集与因果定位合同](../20261029/S2_NEXT_NATIVE_TEACHER_STUDENT_SPATIOTEMPORAL_CAUSAL_EXPERIMENT_20261029.md)；[必须由主Agent分出理论与实验设计子Agent](S2_THEORY_SUBAGENT_AND_EXECUTION_DISPATCH_20261029.md)。先读取最新已完成ROI坐标producer（ord898/96、maxflow并非主体）及新40条失败视频的RUN/EXIT，独立子Agent同步推导局部W2失真、student prefix偏移、运动对应、gate/attention消费者的相反预测；主Agent负责校验坐标/真实state/两卡执行范围、实现及运行pilot。pilot为2个source×Dense/value10/dcac10×高flow5×5+中心5×5+低流负对照×21τ×160site；完成后扩跨来源统计和另行冻结的same-input反事实。无原生新RUN前不得报告已完成采集/推导。

**实际接手后的推进**：[spatial52采集](../20261029/S2_SPATIAL52_DENSE_STUDENT_CAPTURE_RESULTS_20261029.md)之后，[同h四角全160 L/P/I重放](../20261029/S2_SPATIAL52_FULL160_LPI_REPLAY_RESULTS_20261029.md)已双路实际exit0，640条唯一记录、额外12条控制均闭合。gY中央AC的P²约为L²的4.7倍，但signed cross显示强抵消；[独立投影评议](../method_discussion/discoveries/E0/s2-projected-prefix-interaction-independent-20261029.md)指出I可反映滤除的状态差，真实u/rowLN及逐site有反例，不作因果占比。[12张新开发图Dense＋2条中性对照](../20261029/S2_NEW12_DENSE_BASELINE_RESULTS_20261029.md)实际exit0、20项历史状态byte-neutral；[七源28页真实连续截图](../20261029/S2_CONSECUTIVE_SCREENSHOT_VISUAL_RESULTS_20261029.md)已实际CPU exit0并有独立AI评议，明确区分拍翼/鳍肢动作、相机平移、背景变化和仍保留运动的反例。下一项为固定物理行动的正常学生forward与真实消费者/完整视频判别，先冻结代码、匹配对照与成本，不继续调整上轮AC目标权重。

## 0. 协作起手式

1. **观察与信息差距**：列出已有 *原始source-cluster、step×layer×token坐标、producer、numeric backend、teacher或student状态、latent视频时间位置*；明确一个使旧策略失效的已见现象。
2. **机制竞争**：至少提出能够产生相反预测的两个解释，附准确原论文先例与正常原生计算图；不预设补算、TopK、SSE、负交叉项或特定Kernel会有效。
3. **可构造算法**：写明新方法改变什么计算、早/晚信息何时生产、真实物理动作集合及额外观察/转换/权重/dispatch成本。给出最低费用的数学、保存原始tensor或source-held判别。
4. **终端桥梁**：探索性机制可以独立成立；申请部署方法身份时，完整四步×40个位置的真实学生状态、多source配对视频动作/运动/语义/背景/时序/画质和公平硬件成本全部要求同后端、同checkpoint。真正未暴露来源的确认应与曝光开发数据分开。
5. **资产写回**：更新本交接中该方法的证据状态与下一判别，原始实验协议/RUN与独立同行票保持版本；新的文献或运算边按G14协议更新图节点。讨论队列只记录真正的Agent提案，科学设计方案不冒充同行通过。

## 1. 可立即复用的原生资产与证据归属

| 资产 | 实测规模/语义 | 现在能回答什么 | 不能借用的身份 |
|---|---|---|---|
| [BF16原始全深度corpus](../20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.md) | 8个已暴露image clusters，6校准+2旧开发；每来源×160位置×256个真实同坐标行，1280 tensor packets；g/shift/scale同step位在8source相同 | 全层/步的teacher真实函数输入、激活、输出、门控与固定条件；S1/S4的强信息基础 | 非新来源验证；没有BF16 FP4 Q1/Q2；无student free rollout |
| [BF16 80物理函数rank960两种因子](../20261029/RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.md) | shared80/step160完整导出；source-held八锚点和两个开发source的实算 | 构造全160位BF16低秩W2候选、检查跨步共享与条件rank收益 | teacher local loss与导出文件不是视频质量/端到端加速 |
| [HiF4连续tile及整组闭包](../20261025/E0_SHARED_TILE_PRE_W1.md) | 2已暴露来源×5个Block×不同tile，真实同expert跨step | 中层共享支持劣化、层异质、旧64-group的早期执行闭包 | 非全40Block、非真实光流配对，不能转作BF16/NVFP4 FP4元信息 |
| [HiF4 UB1真实视频](../20261029/UB1_20SOURCE_FULL_DEPTH_VIDEO_STATE_EVIDENCE_RECOMPUTATION_20261029.md) | 20已暴露source，40条配对视频与四步latent有符号增量 | motion attenuation、视频终端证据应保留的动作/时序轴 | 非原BF16候选的同checkpoint对照 |
| [PRO6000原生NVFP4 FFN](../20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.md) | 实际packed native FFN对照，范围见原RUN | 确立W4A4数值/Kernel分支真实资源与成本边界 | FFN microbenchmark非全视频成功或任何HiF4兼容证据 |

## 2. 同时开放的研究任务

### A · BF16已导出因子：建立第一个完整视频方法对照

**输入**：两套原始BF16 W2 r960 factor exports、原BF16 checkpoint、full-depth Dense生成/来源manifest。**操作**：先审因子文件SHA、真实80物理函数×同expert两step覆盖、W1/GELU/b2/e5/BF16写回中性路径；冻结新的物理source cohort与每source seed；构造Dense / shared80-r960 / step160-r960三臂真实160-site自由生成，必要时分源执行，累计形成可验证的同源配对视频。

**主要输出**：来源级动作事件、连续RAFT、主体/背景、相机、闪烁、视频语义与画质向量；实际FFN、W2两GEMM、weight load/offload、observer和整段wall，记录cold/warm与调度顺序。源簇至少20个作为首个开发视频设计目标；可利用仓库中准备的10subject/7background/3camera草案，但需核查每份input的真实可用性、曝光与新scope后才执行。**判别**：这条候选若视频保真具竞争力，S1/S4才有可靠的完整网络低秩函数起点；若动作失真，精确区分函数容量、step特化和学生prefix状态，而非退回单末层拟合。

**本轮状态**：[20来源60条全160视频与实际成本](../20261029/RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md)及[官方八轴/连续RAFT](../20261029/RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.md)均已完成且实际exit0（两个V1失败前完成的六GEN单独复用）。shared/step的FFN观测约1.49倍、整段约1.057/1.042倍；RAFT来源均值比0.490/0.477，图像条件及画质也下降，当前质量未通过。公平预热重复SYS、人审与新来源盲确认待测。下一动作围绕实际时间变化损失和学生前缀机制，不重复已有60视频。

### B · S1-CSFB：跨步共享函数与收缩特化

**输入**：6校准source完整80函数的真实h/W2/g和已导出shared/step r960；只在**共同rank-r左基U**中拟合step专用的V修正。保持原W1/GELU与写回，固定step右因子可在部署前融合。

**竞争**：共享模型一般化收益来自估计方差 vs 实际存在step条件子空间且需要额外容量。**第一判别**：相同有效秩/训练来源/读出损失下，source-held先比shared、step独立、共享U+收缩；按source分报告head/tail/可达函数误差，禁止从曾见source调收缩参数。再向完整视频推进。**状态**：OBS（共享在6/6 LOO锚点、2/2历史dev有优势），新收缩为ALG/OPEN。

### C · S4-FRANK：按条件决策价值分配函数秩

**输入**：同一BF16 h、原W2与r960 factor。候选rank640基础+可选rank320块；离线高斯投影的两个误差草图预测`计算B1减少的原W2局部输出平方误差`。该probe用已完成W1/GELU后的h决定W2工作，所以不会提前省W1。

**竞争**：行动收益在同条件source或tile存在可预测异质性 vs 全rank固定执行在误差/代价Pareto上已占优。**第一判别**：对全部已存160位置、6来源LOO/2来源dev检查真正每source、每物理tile的oracle margin、k16/32/64 sketch regret、false-negative尾部、observer/route与额外rank的成本。新增学生视频需在A的同checkpoint低秩Dense比较器先建立之后冻结。**状态**：数学草图无偏性ALG，真实source-held regret和SYS为OPEN。

### D · S2-FTFD：全深度时空函数单元

**输入**：旧2来源×5 Block连续M128观测；新原BF16全160位置保存行及其token locator。**竞争**：layer/step本身决定公共创新 vs video时间/物体移动的token配对决定真正可复用结构。**第一判别**：核对21 latent时间槽×29×52空间网格、真实配对索引；先检索保存行的空间/时间匹配覆盖，缺失的paired token才另注册被动Dense采集。每层/step/source逐层记录共享均值、token-specific能量与真实运动对齐后的作用。只有发现稳定且前置廉价分组后，才发明tile Kernel。**状态**：[八来源全40×4稀疏配对OBS](../20261029/RAW_BF16_S2_TEMPORAL_PAIR_OBS_RESULTS_20261029.md)实际CPU exit0；时间同xy优势已存在于V1。[十张新图完整21τ×8xy、原BF16全160采集与CPU几何](../20261029/RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)已实际exit0，1600包全部原u byte-neutral通过。近邻复制门后AC能量约0.987而AC误差约0.855；[同输入固定因子](../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)对照也已exit0，shared DC/AC误差0.0293/0.1842。同数据、同r960、同两GEMM的[普通/时间平衡全80表示已拟合exit0](../20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)。[20图两候选40完整视频和八轴已实际exit0](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)，复用原Dense；RAFT比例0.451/0.439、质量未通过。下一观测为student内部状态及真实空间邻域，分解局部泛化与prefix放大。

### D2 · S2统计附注：时间结构、对应关系与可选FFN表示假设（不是独立路线）

**发现与定义**：[MTC/TDIC科学方案](../20261029/TEMPORAL_DISTANCE_MOTION_COMPENSATION_FFN_HYPOTHESES_20261029.md)以20来源完整BF16 r960视频的运动衰减和8来源same-xy稀疏时空距离为矛盾，提出**同一去噪step内部21个latent视频时间槽的DC/AC保留、真实运动对应、W2近似误差运输**三条区分性检验。与跨去噪step缓存严格不同，TokenFlow/FRESCO/OmniCache的特征匹配与缓存已属近先例，科研贡献必须体现在FFN函数误差和质量/成本的新增结果。

**已执行的先验**：10新source×全21τ原生采集及同Dense h重放BF16 shared/step因子均已exit0；已获得gY/u的DC/AC误差、方向、step/layer及来源分层初步结果。完整8xy采集只可做固定xy/有限横邻控制；下一步需要真实spatial-neighborhood producer、遮挡/光流离线oracle及在线match成本。**不用提前假定补算**：先检查相同rank预算的AC加权函数；任何上一τ原W2误差锚点或运动warp都必须有实际计算、可用性、重排与存储价格。若局部AC没有明显失真，转学生prefix分布漂移与Attention下游而非继续假设已证明运动补偿。图claim为[PROPOSED](../method_discussion/graph_annotations/E0/E0_GRAPH_MTC_TDIC_TEMPORAL_INNOVATION_20261029.json)，新增时序统计/空间邻域数据需求并入[队列的AQ-S2-TIME-TUBELET](ACQUISITION_QUEUE_20261029.json)，现有E1/E2是真实完成的观测，[T1–T4统计已CPU exit0](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)，gY去τ0平均r95约16.96；T5空间/学生pilot已取得960包真实采集和全160四角重放，T6正常消费者/因果变量继续按需采集，不是已执行算法或独立Agent SUPPORT。

### E · S3-QFAR：量化可见信息与早期合法动作

**输入**：旧HiF4全64组真Q2贡献、CAL24及source绑定、Q1/Q2源码；原生NVFP4以自己的W/A尺度与pack做独立lane。**竞争**：真实合法whole64组行动有source-held收益而cheap Q1前置指标能够预测 vs 真实可执行整组组束的oracle收益已经不足。**第一判别**：在真实whole64 action dictionary比较oracle、step-only、cheap preW1 BF16、Q1 scale、带有位置的signed码摘要、static W×dynamic A，以及观测/route/packing成本；输出action regret、合法闭包与用到的真实信息时间。HiF4先作为诊断，原生NVFP4组为独立待证算子。**状态**：旧任意列25%保持时需全部216组闭包和whole64能量基线已测，联合低风险动作OPEN。

**S2时序统计追加的必需工作**：[T1–T6证据需求与新增采集计划](../20261029/S2_TEMPORAL_STATISTICAL_STRUCTURE_AND_DATA_EXPANSION_20261029.md)。新10 source的全160原BF16完整21τ实际采集/CPU统计和因子E2实算已完成；当前均值/同xy/横邻数据仅能回答能量和部分方向问题，下一阶段须报告source-held lag k=1..10、时序FFT能量/相位、21×21 temporal Gram有效秩、首末窗口/突发、层/step异质与遮挡/动作分层。新[数值函数](../../tools/raw_bf16_temporal_statistics.py)和[registered-only CPU producer](../../tools/e0_s2_temporal_statistics_cpu.py)已完成静态导入修复、8项软件对照及真实CPU exit0；[1600包完整统计及ALG正交控制](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)揭示DC共享但创新宽谱/弱相关，不能直接假定少数时间基。对齐/残差运输仅是有统计收益时的可选计算构造，而不是第五条方法路线。

## 3. 有判别力的科研结论模板

按科学问题每次只输出一页：

**OBS / ALG / SRC**：这次从哪一份来源已核的真实资产发现了什么结构，单位是source而非row/step/site？
**COMPETING PREDICTIONS**：最强的两个解释分别预测什么方向、哪个层/step、哪个source的不同结果？
**NEW COMPUTABLE ACTION**：数学、线上可得信息、正常计算流、动作/量化合法单位、额外物理成本是什么？
**DISCRIMINATOR**：已复用哪份真实存档，实际新运行了哪些指令或代码，数值/运行身份与下一次最小对照？
**VIDEO/SYS BRIDGE**：如何通过全160位、多source完整视频动作与平衡实际延迟，独立确认是否有新部署能力？

旧研究中的有符号抵消、补偿、末层范数或局部SSE，只在它们解释真实竞争机制、改变可执行计算时具体展开。**负结果以明确对象和强竞争解释写成新信息，不用空洞的免责句代替下一步。**

## 4. 状态、权限与多Agent

方法研究可以并行、可读paper和保存资产；实际native/model/GPU范围严格沿[ACTIVE](../ACTIVE_RESEARCH.json)注册，核验可用GPU后先遵循适用会话/原生RUN冻结的并发约束（VIDEO20等有 `session_max_GPUs=2`），再依据[动态资源政策](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)在其内部使用有效最大资源；不以通用动态规划绕过真实执行上限。独立方法提案进入[QUEUE](../method_discussion/QUEUE.md)，须实际不同Agent review才有支持状态。不会为了保持队列整洁删除任何历史RUN、失败身份、SHA或原始论文审计。

**读取原文**：[G14 Paper Focus Graph](PAPER_METHOD_FOCUS_GRAPH.json) → [98+10论文源ledger](../literature/20261026/PRIMARY_LITERATURE_INDEX.json)；按mechanism-research / problem-reformulation / research-expansion / web-research-discovery验证每个候选最接近的论文机制及作者源码，再写创新断言。原文已核的特定物理关系优先复用，不把“再找一篇经典论文”当默认行动。
