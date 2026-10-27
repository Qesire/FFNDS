# FFNDS 技术交流：已经成功的尝试、真实机制发现与相关研究

**独立交流文档｜2026-10-27｜适合向未参加本项目的算法/系统研究同事介绍**

> **摘要**：我们围绕 Wan2.2 四步蒸馏 I2V 的 FFN，真实实现并测量了固定宽度省算、完整权重常驻优化、全模型窄 FFN 和硬件 packed FP4 基元，同时形成了跨来源、跨 step 的局部 FFN—完整生成状态联合证据。已经确认：存在实际可节省的 FFN 计算和系统时间；但**局部近似误差、完整视频功能、跨步有符号传播和物理成本**并不是同一个指标。当前最有科研价值的线索是联合函数、状态依赖、信息可得时点、量化尺度闭包以及新的执行粒度。尚无经过跨新来源完整质量验收的最终新 FFN 省算算法。

**阅读约定**：本文独立介绍对象、成果、失败范围与文献线索；正文不要求熟悉 FFNDS 内部文件结构。附带链接是可核对的实验和原始论文分析，不是未来 GPU 任务或长期 Agent 指令。

---

## 一、我们研究的具体对象

研究基准为 **LightX2V Wan2.2 I2V A14B，四步蒸馏 D-Q**。两套专家依次参与 **high/high/low/low** 四个去噪 step；每步有 **40 个 Transformer blocks**，一次完整生成包含 **160处 FFN 调用位置**。一组代表性 shape 是 token 行数 **M=31,668**、输入/输出宽度 **D=5,120**、隐藏宽度 **H=13,824＝216×64**。

FFN 的真实计算路径：

\[
\mathrm{cross\!-\!attention\ residual}\rightarrow
\mathrm{norm2/AdaLN}\rightarrow \mathrm{Smooth1/Q1}
\rightarrow W_1\rightarrow\mathrm{GELU}_{\tanh}
\rightarrow\mathrm{Smooth2/HiF4\ Q2}
\rightarrow W_2\rightarrow e_5\ \mathrm{gate/BF16\ residual}.
\]

后续 block、attention、flow 与 scheduler 消费上述写回，再产生下一步生成状态。实际 **HiF4 Q2具有64→8→4共享scale依赖**；仓库原生路径是 **fake-QDQ＋BF16 Linear**，和真正的 **packed NVFP4/MXFP4 Tensor Core GEMM** 具有不同的数值/硬件性质。

最终目标包含两种可独立成立的成果：**新的信息/函数/计算规律**，以及在完整视频质量可比的条件下**真正减少 FFN 和生成代价**。目前有 **10个独立来源、12条source×seed完整生成轨迹、1,920个嵌套 FFN 观测位置**；位置数不是独立视频样本数。详细模型和计算图：[源图G0–G13](98_端到端计算图与证据注解_20261025.md)。

---

## 二、已经成功完成的工程尝试

### 1. 冻结静态 FFN 预算：真正的历史质量–速度可行性

**UB1** 按四个step分别保留 **54/108/108/162个原始64通道组**（原宽216组），整体主FFN GEMM按位置加权密度为 **50%**。原方法包括实际 rank-256 残差及辅助计算，因此不应将它说成只有半宽矩乘。

冻结方案在**原正式20个独立来源**上，相对 Dense D-Q：

| 官方指标 | 相对下降 | 该来源集历史2%门槛 |
| --- | ---: | --- |
| aesthetic_quality | **0.874%** | 通过 |
| imaging_quality | **1.345%** | 通过 |
| motion_smoothness | **0.0074%** | 通过 |

全160处FFN边界，Dense **10,177.57 ms** → UB1 **6,582.69 ms**，**1.546× FFN加速**（三次重复、按原工程口径计入辅助与共享准备成本）。

**重要的泛化对照**：之后不相交的另外20个来源上，固定相同参数的 UB1 imaging_quality 相对下降 **3.166%**，没有达到原2%门槛。由此**既有真实历史成功，也有真实跨来源质量失配**。这是一份优秀的强静态比较器，而非经过新来源稳健验收的最终解，更不是输入动态路由。

原始证据：[20来源工程记录](../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json) · [原生FFN计时](../reports/worker_20261020/UB1_complete_FFN_speed.json) · [额外20来源](29_冻结候选的来源分离补充验证.md)。

### 2. Dense 权重常驻：在原值完全不变时提速约 1.49×

通过缓存原始解码量化线性权重、减少显式 cache flush，做到了同一A100 80GB上热态完整四步生成：

- 原策略 **75.172/75.372 s** → 全量常驻策略 **50.419/50.789 s**，分别约 **1.491×/1.484×**；
- **18条完整视频、72个scheduler latent SHA** 与对应原 dense 对照匹配；
- 额外解码权重常驻 **52.34 GiB**，物理显存采样峰值 **77.08 GiB**；
- **FFN/attention MAC数和模型数值函数均未改变**。

这是已验证的真实系统优化和未来 FFN 方法的**强 dense benchmark**。结论不能直接移植到低显存设备；也不能把省掉的权重搬运冒充为 FFN 本征少算。

原始证据：[Dense全权重常驻、18视频与反向对照](../research/20261025/E0_DENSE_WEIGHT_RESIDENCY.md)。

### 3. Affine54 / Whole94：全模型真省算的可运行原型

我们不再只在两个block模拟。对同四个校准来源、同等行数做全 **40层×4step** 的匹配局部重构。真实 FFN 宽度缩减包括：

- **Whole94**：保留94个原64组与剩余ridge输出重构；
- **Affine54**：可折叠固定仿射部分＋54个真正计算的非线性64组。

在**两个已暴露持出来源×四步×40层＝320局部case**上的 gated 相对SSE：

| 局部映射 | 平均SSE |
| --- | ---: |
| 全逐神经元Affine | 17.28% |
| 同等校准的Affine锚定输入Ridge | 15.35% |
| Whole94 + Ridge | 11.67% |
| **Affine54 + Ridge** | **8.54%** |

Affine54在 **305/320** 个case中优于Whole94。这表明在该限定的校准与正则条件下，支付一部分真实非线性计算比简单线性重构有价值；不代表“任何其它低成本表示”都有8.54%的下界。

随后的 **Dense/Whole94/Affine54三臂×两个来源＝6条完整81帧视频**均已真实生成。同一来源12热态：

| 边界 | Dense | Affine54 | 比值 |
| --- | ---: | ---: | ---: |
| 原生FFN wall | 9.707 s | **4.449 s** | **2.182×** |
| 完整生成 wall | 50.750 s | **45.313 s** | **1.120×** |

另有 setup 时间 Dense **139.188 s**、Affine54 **237.089 s**，新权重bank准备和显存也需要扣账。实际候选视频有明显模糊、运动/细节改变；Whole94呈现更严重重影或纹理退化。**两个原型均没有达到完整视频保真要求**。

这里最值得分享的是一对同时成立的事实：**窄 FFN 真的可部署并提速；相当好的局部SSE不能保证四步自由视频质量。** 这为研究功能和状态传递打开了比继续调阈值更有价值的问题。

原始证据：[320局部匹配对照](../research/20261026/E0_ALL_BLOCK_RECONSTRUCTION.md) · [6视频原生FFN/整网计时及画质分析](../research/20261026/E0_RECONSTRUCTED_VIDEOS.md)。

### 4. Packed NVFP4 原生基元已打通，但尚未集成全Wan

使用 NVIDIA CUTLASS SM120 NVFP4/BF16 GEMM 官方例程，分别执行 **128×128×128** 和 **128×256×512** 两种shape、共 **5次GPU GEMM**（约6,921万MAC）；和官方host reference逐元素匹配。该结果确认真实低位核构建、运行和数值基准链路；**不代表原Wan已改为packed NVFP4、更快或更少MAC**。

原始证据：[SM120 packed FP4基元实测](../research/20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)。

---

## 三、最有科学价值的实测发现

### 发现A：单个FFN误差变小，整体状态仍可能持续偏移

在两个来源的完整原生paired四臂中，**318/320** 个FFN site的受扰Student输入上，candidate与dense之间的raw绝对算子误差**比teacher输入时还小**（另2个初始site相等），可是 latent 状态差在随后step持续增大。

扩展到六个旧来源共960个site，计数为 **943小/6等/11大**；四个项目新来源、六条seed轨迹另外960个site是 **942小/6等/12大**。这些均是层×step的嵌套测量，不能当作1885个独立视频样本。

**科学含义**：局部算子近似的好坏、此前产生的状态偏移、后续模型的方向响应，应拆开研究。已测数据削弱“Student一旦离轨，每处局部算子误差必然变大”的特定解释；并未证明视频崩坏的唯一根因。

证据：[完整两来源配对结果](../research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md) · [六来源](../research/20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md) · [新来源](../research/20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。

### 发现B：四步状态作用存在可复核的有符号抵消与放大

完整 dense 单步有限反事实可写为：

\[
P_t=T_D(z_t^{S})-T_D(z_t^{D}),\qquad
N_t=T_S(z_t^{S})-T_D(z_t^{S}),
\]
\[
\|z_{t+1}^{S}-z_{t+1}^{D}\|^2
=\|P_t\|^2+\|N_t\|^2+2\langle P_t,N_t\rangle.
\]

**P** 是旧状态误差经过完整dense当前步的响应；**N** 是相同Student状态下候选替代dense的新增作用；交叉项保留方向、抵消或放大。真实实验保留BF16 step_post数值边界，并非未经验证的Jacobian近似。

已测两旧来源的非零偏移状态，完整dense单步的**平方距离增益为3.63–7.72**。不同来源的后三步cross符号不同；在新来源 **425**，相同图片和提示、仅seed改变，就使后三步cross由**全负**变为**全正**。

另外，六旧来源的 **source16/step2** 中，当前候选相对同Student状态dense的净增量

\[
\Delta_{\rm candidate}=\|N\|^2+2\langle P,N\rangle=-946.8824
\]

为负；新来源437在step2/3也出现负净增量。在这些有限条件下，候选当前作用可**部分抵消此前的状态响应**，虽然整个生成轨迹仍偏离dense。

**科学含义**：重要的未必是后验幅值最大的channel，而可能是随step、expert、seed改变的**联合signed输出方向与传播函数**。这些标量没有识别具体W1行、Q2组或后续attention的因果职责；这一身份仍需限定的正常计算干预。

证据：[原生完整Dense状态有限反事实](../research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md) · [source425/437跨seed与净效应](../research/20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。

### 发现C：低后验稀疏度不必产生空的物理GEMM tile

对原始连续M128 token、不同通道筛选方法的实测结果，在25%/50%隐藏列保留率下，已测试的 **60个M128 tile均触及216/216个原HiF4 64组**。因此需要按原量化闭包维持数值时，W1不能简单按保留列率缩窄。曾有低SSE的逐token mask，合成共享坐标后的SSE显著上升。

独立的原始HiF4 Python量化参考反例说明：只改一个64组系数，因共同scale变化，**其它未改输入坐标的Q2输出也能变化**。这与真正packed新格式、后Q2 W2独立列贡献或训练后改量化规则属于不同数值方案。

**科学含义**：必须同时研究信息表示、合法数值闭包、真实M/K/tile执行粒度。后验top-k评分和实际空tile/高速kernel是两个要由数据连接的命题，不存在天然蕴含关系。

证据：[连续token与物理tile](../research/20261025/E0_SHARED_TILE_PRE_W1.md) · [HiF4源码反例](../research/20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json)。

### 发现D：非线性“均值函数”和动态视频功能是两个分辨率

在保存的两来源四个panel，利用真实W1 preactivation的均值/方差，经Gaussian tanh-GELU读出预测 gate 加权的FFN**均值向量**，相对SSE为 **0.179–0.772%**；仅基于pre-W1对角方差信息的近似为 **5.039–14.294%**。这说明联合矩信息对特定非线性函数读出有价值。

但均值对token级正负抵消不敏感；这些数字既未给出全M局部误差，更没有生成视频质量或在线成本结论。

另一项已完成的“前三步候选、末步dense”原生保存latent解码反事实，在来源12使整段逐帧RGB RMSE下降 **13.58%**，来源0仅下降 **0.45%**，运动/细节代理依然出现不同方向变化。这进一步支持关注**跨步职责与条件性视频行为**，而非统一指定最后一步或某个全局频带。

证据：[Gaussian矩/均值CPU分析](../research/20261026/E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.md) · [末步dense保存latent视频桥](../research/20261026/E0_SAVED_FLOW_DECODE_RESULTS.md)。

---

## 四、最值得与同事讨论的一手论文及其真实迁移边界

项目将 **89项研究**登记到[文献索引](../research/literature/20261026/PRIMARY_LITERATURE_INDEX.json)，但阅读层级包括摘要、原论文方程、作者源码或有限复核，**不等于全部89篇都完整实现或复现**。下列是最能改变具体科学判断的机制家族。

| 方向 | 代表工作及已经核查的事实 | 在我们模型上的研究价值 |
| --- | --- | --- |
| **step/frame/width计算重组** | DIFS训练step-specific压缩，FIS-DiT按帧×层分配真实M行，Mobile Video DiT训练compact dense FFN。见[G7–G13文献综合](../research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md) | 研究真正节省H、M、step中的哪一种；冻结原模型的天然结构和重训形成的新结构要分开 |
| **跨步复用与增量** | SVD-Cache历史主/残差复用；Chipmunk源码含前置blockmean、W1差、GELU后差和索引W2；L²P还有跨时间函数预测。[原始源码对应](../research/20261025/E0_SHARED_TILE_PRE_W1.md) | 少步两expert的权重身份、历史状态费用与物理W1/W2执行单位是否允许复用 |
| **代理与训练稀疏** | Prox使用**有成本的INT4 proxy**在LLM SwiGLU上选择通道；Spark的预判-稀疏FFN通过训练获得，不能视为原Wan的天然性质。[五篇论文深读](../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md) | 决策时点、observer成本、三投影SwiGLU与当前W1/GELU/Q2/W2数值链的差异 |
| **编码域计算和物理内核** | CodeGEMM已核官方代码：公开多token分支先反量化整个权重再matmul；SpQt在prefill多token回退dense；SharQ执行稀疏及稠密残差两路GEMM。[G11研究](../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ](../research/literature/20261027/G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md) | **LLM单token GEMV ≠ Wan M≈31,668多token GEMM**；比较真正packed dense强基线和全部额外编码、layout、核启动价格 |
| **有符号量化信息界** | G12用重排不等式：对单W1行已知直方图可得到pair-agnostic严格最紧区间；多W1行共享同一个输入排列，因此各行独立极值的组合并不总可达。[G12数学与成本](../research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | 构造对联合signed输出的可付费信息界，而非独立通道score；须计入计算全部histogram及逐行查询价格 |
| **非线性函数与有限误差机制** | TARDIS已有区间线性化、常量折叠与条件纠错；BRECQ/Q-VDiT/CTEC涉及块、视频或跨步误差结构。[本仓库原论文竞争分析](../research/20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md) | 与最强现有结构区别应落实在真实HiF4闭包、signed输出、整四步作用和原生视频质量 |
| **功能因果定位** | JMLR2025 causal abstraction、ICLR2024 activation patching best practices与subspace illusion、AtP*/WISE和视频时空概念定位。[G13方法](../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md) | 从真实视频行为和写回定义条件性函数，用正确normal-forward、Q2闭包、后缀和对照定位承担者；不把单一patch或SAE标签直接叫因果功能 |

这里一个关键的研究认识是**三个独立科学入口均有正当性**：信息与表示（I）、函数与动力学（F）、计算与执行（C）。对一个原生紧凑算子可以先做数学或源码论证，随后研究功能；对函数因果现象也可以先形成科学论文级发现，再问是否可部署。G13是**函数定位专题**，其O/D/Q/P四类边界不构成I/C线进入的前置步骤。

更深的论文资料：[89项索引](../research/literature/20261026/PRIMARY_LITERATURE_INDEX.json) · [直接近邻G8](../research/literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md) · [原论文五篇源码审查](../research/literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md) · [G12严格信息界](../research/literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) · [G13原生作用和因果定位](../research/20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)。

---

## 五、下一次合作可以直接讨论的四个问题

**I：共同输入的多W1行是否存在便宜的联合信息证书？** 单行signed FP4码值直方图有严格但可能宽的上下界；多W1行共享输入排列。能否在少于dense GEMM的代价下，认证一组有符号输出方向或者真正空的计算tile？应分开研究数学界的紧度和在线查询价。

**F：背景/运动/光照稳定性究竟由哪部分正常FFN计算承担？** 跨seed有符号P/N交互与完整视频代理异质性，允许“当前FFN联合写回变化”和“既有状态经下游attention/flow运输变化”两种竞争解释。后续有限O/D/Q/P干预可以定位差别，但必须保持真实Q2闭包及后缀。

**C：能否不经代理稀疏，直接从编码算术、布局、完整tile、缓存或compact函数降低真实成本？** 这条执行主线独立于F线，可用已建成的packed基元、原真实shape与权重常驻dense强参照，研究数学可行性及实际kernel。

**X：轻量后训练能否把分布式功能迁移到新的计算结构？** 当前模型的原生因果承担结构和新模型的最优执行结构不必相同。需把训练预算、step/expert行为、自由视频质量与物理开销一起比较。

这四个开放问题在[并行证据组合](../research/20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)各有竞争解释和最低成本的判别证据；此处没有把它们宣称为已验证的算法。

## 六、交流时应怎样准确总结？

**已经成立**：静态UB1在原formal20范围通过当时的质量/FFN速度检查；可完全保持原dense数值的系统常驻加速；160处窄FFN确实能够加速但画质未保；SM120 NVFP4 GEMM基元已实测；完整状态响应、尺度耦合、source/seed条件性有符号现象均有实测数据。

**尚未成立**：UB1额外20来源imaging未过门槛；Affine54/Whole94完整自由视频存在可见质量失配；没有将局部SSE、全域频带或运动像素代理冒充官方感知指标；没有宣称找到原Wan中明确的必要神经元集合或一个新的已通过质量–速度Pareto方法。VBench-I2V/VBench2与人工完整审阅按原记录的实际执行状态区分，多项新视频研究仍为NOT_RUN/未验收。

**最值得与合作者分享的最小证据包**：[UB1原始验收](../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json)与[新增20来源失败](29_冻结候选的来源分离补充验证.md)；[权重常驻](../research/20261025/E0_DENSE_WEIGHT_RESIDENCY.md)；[全模型Affine54](../research/20261026/E0_RECONSTRUCTED_VIDEOS.md)；[原生跨步P/N](../research/20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)与[跨seed](../research/20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)；[HiF4/tile](../research/20261025/E0_SHARED_TILE_PRE_W1.md)；[论文机制综合](../research/20261027/FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)。

**证据和执行说明**：本文是已完成原生实验、数学、源码和文献调研的独立提取稿；不修改历史RUN、原始失败边界、研究授权或任何临时Agent交接设置，亦未运行新的CPU/GPU模型、视频生成或训练。
