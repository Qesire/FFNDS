# FFNDS · I/F/C并行研究路线、方法与证据组合（2026-10-27）

> **现行跨路线研究解释及方案对照入口。** 本文同步各路线的可证实结论、开放竞争解释、最小判别和可复用资产；它是科学路线图，**不是**新的GPU执行注册或Agent方法评议。共同目标以[统一研究目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)为准；执行及实时/流式范围仅由[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json)判定。原始RUN、历史方法版本、失效结果保持原证据身份。

## 1. 核心结构：三条并行入口、两种成果、一个共同物理锚点

- **I（信息／表示）**：计算之前或中途何时可获得什么，能否形成可复用的有符号证书、等价代数、低位编码或跨步共享表示。
- **F（函数／动力学）**：正常FFN/后续flow究竟承担何种外观、运动、时空协调等功能；区分当前写回、继承状态传播与观测器偏差。G13的O/D/Q/P只是这一线的定位工具。
- **C（计算／执行）**：什么真实W1/GELU/Q2/W2/weight residency、metadata和kernel工作可以重组、替代或省去，计算收益是否在强matched baseline下兑现。
- **跨线方法**：可借用固定预算稀疏、函数折叠、校准消费者、动态激活2:4、轻量后训练等。新机制科学结论与可部署质量—成本方法分别可以成立，不强迫每条路线先通过同一个局部SSE gate。

原生坐标固定为LightX2V Wan2.2 I2V A14B四步D-Q，high/high/low/low、40 blocks，原HiF4 fake-QDQ＋BF16 Linear、W1→GELU→Q2(64→8→4)→W2→e5 BF16写回；四个**去噪step**、21个**latent时间位置**与81帧**输出视频**是不同坐标。packed NVFP4/MXFP4、SpTC 2:4、其它蒸馏模型及实时模型都有独立的数值/物理部署身份。

## 2. 当前各路线的证据—竞争解释—下一判别

| 路线 / 可继续的方案 | 已完成、范围与证据等级 | 竞争解释／机制分歧 | 最有信息价值的下一判别 |
|---|---|---|---|
| **I1 · G11/G12低位有符号信息与共享W1约束** | [G12源码与定理](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)＋[604有限排列核对](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)：单行重排界、跨行相同输入排列的不可独立最优、固定方向精确投影比较器；ALG＋有限CPU软件，未形成真实加速 | 联合证书的收紧是否有足够低的在线获得代价；固定方向 `p=Wᵀc` 是否已经是更强算子 | 在共享输入多W1行的**同一对齐/尺度/符号**坐标下比较joint certificate与精确固定方向投影的计算/带宽；再决定是否值得对原GELU/Q2继续lower |
| **F1 · G13正常功能定位与有限流传播** | [跨seed signed cross](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)、[整步有限分解](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)是指定候选的NATIVE-OBS；[原M Q2直接/补集](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)提供8个保存条件下的局部数值判别 | FFN当前门控写回改变与既有Student状态经attention/后缀传播可同时存在；Q2补集scale与直接W2贡献也可能抵消 | 从[定位卡C01–C03](../agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)任选一个已存资产能分开的受控边界，配对base/donor、同状态和完整Q2闭包；先定位第一变化边，再谈视频功能承担者 |
| **F2 · 自然视频时空功能与运动可观测性** | UB1在2个已暴露开发源上连续RAFT较Dense明显低；[fixed-RR consumer六视频与八轴官方函数](UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)显示局部恢复成像及部分运动但两源未达Dense，Camera Motion未运行；[A5+HWG](A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)是不同方法的历史视觉OBS | 真正物体/镜头运动减弱、纹理模糊导致RAFT测量困难、静态prompt语义和不同post-FFN propagation需要区分；A5逐token mask切换机制不能投射到固定支持UB1 | 在**现有原视频**进行ROI/关键点/相机补偿和跨tracker一致性判别；定义motion-rich/static/camera分层。新来源、完整视频等候独立注册 |
| **C1 · 固定Whole64预算UB1与量化一致函数折叠/Affine54** | [UB1原20/新20来源尾部及系统成本](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md)：step-conditioned静态支持而非input-adaptive；[fixed-RR校准](UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)训练内gate误差恢复83.01%，[latent](UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)末步与Dense距离可反增；[Affine54方案](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)真实FFN快但六视频失配 | 局部拟合是局部目标，与QDQ部署等价、真实跨层/步函数和视频能力不同；固定组容量、分配、R256函数类、拟合数据是可正交的假说 | 优先利用[Calibration V2](CALIBRATION_V2_METHOD_SNAPSHOT.md)同预算来源×行数/teacher-student消融；保持support、alpha/R256、模型精度和算子边界可辨，先核对已有同状态原生收据再启动新的full-depth |
| **C2 · 激活2:4、packed低位与原生FFN算子** | [RT-Lynx源审查](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)是跨模型LIT；[A100合成layout/backend对照](E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)显示部分预压缩2:4 GEMM快于dense，默认预压缩亦可能慢；**在线top2+压缩全成本仍未获得Wan FFN收益证据** | 合法2:4稀疏矩阵乘与原whole64全tile是不同物理结构；SpTC布局/预打包收益可能被在线selection、packing、QDQ、weight memory抵消 | 分别定义Q1后W1输入、Q2后W2输入、两者组合S1/S2/S3；测online select→compress→SpMM与完整W1/GELU/Q2/W2成本，和best packed/dense比较，同时锁定数值格式 |
| **X · 轻量后训练与函数职责转移** | [Dense2MoE/TARDIS/RT-Lynx等先例](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)是LIT；Wan四步蒸馏/新参数形式的职责迁移仍OPEN | 固定权重功能的正常承担结构与训练后新函数类可不同；有限小训练预算能否保住罕见动态功能未测 | 先冻结训练前/后函数对象、数据分割、预算和最强已发表同约束比较；做最小训练对照需单独资产/算子/视频收据，而非默认把R256或LoRA设为必需 |

**证据身份**：ALG/SRC/LIT定义数学或既有实现；NATIVE-OBS/VIDEO/SYS只对指定本项目运行负责；STATIC_ONLY只代表代码合同或软件诊断；OPEN是下一段可证伪工作。所有上述“下一判别”均为**研究建议**，具体新模型或GPU执行以ACTIVE的独立冻结范围为准。

## 3. 四种容易被错误合并的“动态”

| 类型 | 判别标准 | 项目中的位置 |
|---|---|---|
| **离线step-conditioned固定预算** | 同step/block、不同当次输入使用同一支持 | UB1：计算预算随去噪step变化，**不构成输入条件动态路由** |
| **token/content-dependent在线选择** | 动作取决于当次可观测输入，计算观察/压缩开销 | A5+HWG旧mask、潜在2:4 activation或新selector；需记录动作时点 |
| **Student状态与跨step传播** | FFN动作可固定，但状态分布和后缀函数随轨迹变化 | UB1和Affine54相对Dense轨迹差、E0 finite P/N |
| **时间序列视频运动** | 21 latent时间槽/81帧实际动作或camera行为 | 连续motion/ROI/trajectory；不同于“动态宽度”，也不由Motion Smoothness一项决定 |

同一方法可同时涉及数项，但它们的真实来源单位、观测尺度及质量指标分别定义。

## 4. 跨路线通用的校准和评测接口

[Calibration V2](CALIBRATION_V2_METHOD_SNAPSHOT.md)按科学estimand而非预定唯一样本数配置来源与采样：support需216组能量及有符号交叉项，gain/bias需heldout signed error，rank256需实际有效秩与留出风险，teacher/student需同坐标不同状态，terminal需真实来源级视频质量和完整物理成本。12×512／24×256／48×128为相等6,144行**候选**，不是新已完成训练；其中必须先按独立image cluster抽取再登记prompt，原CAL24历史行序与新Python抽样坐标不同。

各线共享四条可重用数据流，分别登记真实producer/方法版本/来源/坐标：
1. **数字与信息**：Q1输入、W1方向、post-GELU、实际Q2 codes/scales、gate×W2 signed组贡献及完整source×step×block×state边界；有符号Gram缺失记 `NOT_MEASURED`。
2. **行为与状态**：同source/seed/frozen noise的Dense与候选前缀、FFN写回、first receiver、flow、每步latent、完整MP4；训练内局部改善不能代替自由生成。
3. **系统执行**：每算子真实输入布局/精度、QDQ、selection+compression、GEMM、补偿、gate、weight residency、latency/VRAM，包括最强对照实现。
4. **指标与分割**：image-cluster隔离历史曝光；开发集和新冻结确认集不同；VBench原函数自定义子集与官方Image Suite五次真生成不同；连续运动与相机/语义轴另量化。两条已暴露source不能作为新确认集。

[实验Skill](../../.agents/skills/ffnds-experiment-evidence/SKILL.md)与[可执行手册](../agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)给出实际脚本和命令；在采集前建立可复用资产计划，实际运行结束记录produced/open与复用失效条件。已有源级csv、原生bank、原视频、latent、合成2:4数据按各自历史版本复用，不复制成新独立证据。

## 5. 多Agent协作与研究结论更新规则

- **讨论/发现**：独立数学新界、竞争机制、因果定位和物理执行发现均可进入[discoveries](../method_discussion/discoveries/E0/README.md)。形成精确方法提案才写[method_discussion/QUEUE](../method_discussion/QUEUE.md)，外部Agent的支持/异议按原review文件登记；本表的路线优先级不等于方法被同行认可。
- **源图**：I/C可直接添加数学/算子条件边；F具体行为函数由[G13](G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)与Function–Structure索引管理。运行/质量与图注不能通过文案互相替代。
- **下一步选择**：如果已有保存数据能区分两个机制，先做该对照；如果是数值恒等/包含关系则直接推导；如果主张在线成本则优先测真实kernel；只有主张新的视频质量/功能恢复时才进入独立full-depth free-rollout与视频多轴协议。
- **版本同步**：本组合维护每线“最新可验证现象＋最强竞争解释＋最小判别”；各方法方案只维护自己的方法公式、实验arms和接受边界；文献原式留在文献台账；精确执行许可和是否已运行留在ACTIVE。旧报告当时的OPEN/NEXT是历史快照，不抢占现行研究优先级。
