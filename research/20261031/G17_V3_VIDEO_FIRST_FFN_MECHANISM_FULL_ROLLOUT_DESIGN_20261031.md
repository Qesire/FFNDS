# G17 v3 · 从完整视频行为出发的 FFN 功能机制假说与全程自由生成因果实验

2026-10-31 · **METHOD_SNAPSHOT / VIDEO_FIRST / DESIGN_ONLY / STATIC_REVIEW / INDEPENDENT_REVIEW_REQUIRED / NOT_EXECUTED**

> **2026-10-31 证据链同步扩展**：[全深度Dense/Student张量＋A5/HWG系列同状态张量＋生成代码保存性核对](G17_V3_TENSOR_HWG_CODE_ARCHIVE_AUDIT_20261031.md)。VIDEO、TENSOR、HWG三条主证据并行，互相构成可证伪预测；G17 v2局部代数只是工具。此次核对确认现有主要生成入口在Git，但原版A5完整参数producer缺失，LightX2V/HiF4原运行树需通过上游恢复包及每RUN身份证明，**不得称全部生成代码已完备归档**。

**方法学改向**：以**真实 81 帧视频在动作、文本条件、主体、背景、相机、时间、成像与实际成本上的结果**为被解释现象；先提出能够在视频上区分的竞争机制，再使用FFN局部张量、Q闭包、O/D/P及梯度诊断解释。禁止把低L2/高Gram/低缓存开销视为方法质量通过。本文修订[G17 v2](G17_MOTION_SEMANTICS_TENSOR_CHANNEL_CAUSAL_DIAGNOSIS_20261031.md)的**研究顺序与主科学问题**；G17 v2含条件性代数及历史数据，供按需调用，但不能用其中cache-delta误差公式统摄无缓存或函数重表示方案。

## 0. 确认目标与观测层级

科研对象为Wan2.2 I2V A14B：4步HHLL×40层共160个**实际FFN调用位置**，四步均正常完成，自由生成完整视频。每个source条件 s=(input_image,prompt,seed/initial_noise,checkpoint,numeric_recipe,scheduler,decode) 冻结后，在**同一数值族**比较Dense、neutral及真实候选。使用视频级质量向量
\[
Q(V)=(Q_{\mathrm{event}},Q_{\mathrm{motion,obj}},Q_{\mathrm{camera}},Q_{\mathrm{text}},
Q_{\mathrm{subject}},Q_{\mathrm{background}},Q_{\mathrm{temporal}},Q_{\mathrm{imaging}},\ldots)
\]
及物理成本 C=(FFN operator wall, full generation wall, memory, resident/transfer/pack/cache/selector)。正式推断单位为**image cluster**，seed嵌套在cluster；RAFT、token、latent帧和site均不是独立确认来源。

**科学estimand**：相同source及初始噪声下，真实全程算法M对完整视频行为的配对差 \(\Delta Q_j(M,s)=Q_j(V_M(s))-Q_j(V_D(s))\)。其中动作语义依prompt要求与完整视频逐段验证；Dense本身没有执行所需动作时标记BASELINE_EVENT_FAILURE，不能只凭与Dense运动幅值接近宣布方法成功。对照Dense的配对差用于保真，不把Dense轨迹当唯一真实运动路径。

**证据等级**：已完成VIDEO→具有视频变化的条件介入因果VIDEO→分层运行的receiver/latent原生观测→可执行pre-Q2选择→blind VIDEO与实际SYS。反向操作（先优化local proxy、最后才看视频）不得主导方法晋级。

## 1. 已有真实视频的核心现象与正确推论

1. [UB1 20个开发来源](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：同来源Dense→consumer连续RAFT 6.894→1.152（源均值比，-83.3%）；dynamic-degree binary 6/20→0/20；subject consistency 0.9602→0.9809，imaging 72.629→71.626。**动态显著衰减而静态一致性仍较高**，这是机制发现的出发点，不代表“所有稀疏方法都冻结运动”。
2. [BF16 value10/DCAC10 20来源](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)：Dense RAFT 7.058、value10 3.181、DCAC10 3.101；AC局部代理改善未转为视频质量。是另一numeric recipe与方法，不能和UB1强行建立跨方法的同seed因果差。
3. **最关键的现成视频2×2干预**：[Chipmunk P64 四臂](../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)，来源898与96的Dense/NONE、High-only、Low-only、Both，每source一个seed、全4步×40层、八轴与连续RAFT均已实际完成。898 imaging分别77.431/77.423/75.311/75.998，RAFT 22.726/23.078/21.790/21.595；96 imaging分别72.307/74.219/70.823/71.623，RAFT 8.539/8.326/7.055/7.050。**在两个曝光source上Low-only损失较明显且H×L交互在成像轴异号**：源898 +0.6957、源96 -1.1131（B−H−L+N）。不能据此宣称Low专家独立承担运动，因为expert与HHLL阶段绑定，且96原版dynamic degree=0。
4. [High旧Y复用 vs High P64 vs Dense](../20261030/CHIPMUNK_HIGH_Y_REUSE_QUALITY_RESULTS_20261030.md)：898/96旧Y复用的RAFT相对Dense为0.8507/0.9199；对应High-P64约1.0155/0.9751。898截图显示鸟翼中段重叠轮廓，**视觉描述来自静态接触表而不是本轮重新观看连续视频**。旧Y复用与fresh选择组更新的区别是候选机制，不将其作为普遍结论。
5. [四来源High-P64视频](../20261030/CHIPMUNK_INPUT_VIDEO_QUALITY_RESULTS_20261030.md)中979相对Dense的RAFT≈0.7876、MUSIQ−6.382，而1040相应RAFT≈1.1318、MUSIQ−1.118。不同来源的质量变化不被同一运动幅值标量解释。
6. [SharQ/TEAL全160视频与执行证据](../RESEARCH_NAVIGATION.md)已经存在；应当登记为已完成的独立FFN强基线，核查其真实物理操作与numeric recipe后用于竞争解释，**不重跑已完成范围**。

**现在可以提出的事实**：稀疏/复用FFN可能改变动态幅度、对象边缘、主体一致性和成像；局部输出误差不能直接预言最终视频；阶段与来源交互可见。**尚不能确定**：负责的正常功能方向、是否固定“运动通道”、Q2闭包/后缀的因果份额、跨来源普适性以及任何候选部署质量非劣。

## 2. 中心假说与竞争预测——先在视频上可证伪

### H-F · 动态相关的状态更新（主假说；不预设通道专属）

同一expert第二次去噪调用的部分FFN实际fresh写回影响**主体动作进展、相机/主体相对运动、轮廓时序和事件时间**。省算策略如果冻结/删除这些更新，最终视频可能更静态、产生重影或事件延迟，尽管单帧外观保持较好。

**可区分预测**：(a) 在固定物理K与完整自由生成下，不同HH/LL执行位置的dense restoration对动作与成像的收益出现来源/事件依赖；(b) 对old-Y-only恢复合法fresh更新会改善一部分视频行为，而纯增加低价值计算不会取得同等效果；(c) 在至少部分source中“全视频收益排序”与局部L2/RAFT总幅值排序不一致。**反例**：补充fresh更新后只减少像素差，但实际动作事件/时间/方向不变；或相同计算预算的随机恢复等效。

### H-C · 一般容量/视觉质量解释（强竞争）

损失主要来自整体FFN有效容量减少、局部norm/边缘退化或泛化的图像质量下降；RAFT变化可能是纹理/相机背景及tracking失效的伴随指标，并非动作计算本身。**预测**：等K、等恢复算量下，不同阶段/功能位点的恢复效果近似由总预算或画质恢复解释；控制imaging后，目标事件的条件效应缺少稳定选择性。**反例**：在画质和计算成本相近的两臂中，一臂选择性恢复提示动作，另一臂不能。

### H-P · 学生状态与后缀动力学放大（强竞争）

同输入单FFN差异不大，但经过真实后续40层中的**剩余层**、scheduler和后续步，偏差改变学生输入、gate/Attention状态及最终视频；主损伤可能不由单组直接写回范数确定。**预测**：同一个合法FFN动作，teacher-forced局部与自由学生视频的损失排序不一致；在早/中/晚层的**等层数、等计算预算**受控dense restoration和高/低时段恢复产生不同的receiver→step→最终视频路径。**反例**：后缀传播可由同输入本地写回误差稳定预测，恢复位点顺序几乎不改变视频行为。

### H-H · 历史缓存错配（限定于cache路线）

缓存保留外观的稳定部分，但历史输出与学生当前状态之间的错位使fresh变化无法完全写回；**只对带cache方案适用**。预测：old-Y-only相对相同第一调用/相同数值族的fresh-delta版更易出现动作滞后或轮廓拖影；带cache和不带cache的误差模式不同。**反例**：video退化几乎完全由保留K决定，改变历史缓存语义并不改变视频行为。无缓存静态稀疏和新函数表示是独立竞争路线，不使用 \(h_1-h_0\) 作为它们的统一误差对象。

## 3. 机制对应的科学对象与行为级因果量

设P是已定义、可运行、确实作用于FFN的完整策略；对合法site集合 A 实施“指定site第二次调用restore Dense FFN”操作，未指定site继续用P，在**相同完整生成与原生数值路径**得到 \(V_{P\oplus A}(s)\)。主行为效应：
\[
R_j(A;P,s)=Q_j(V_{P\oplus A}(s))-Q_j(V_P(s)).
\]
对已经完成的高/低两因子四臂，另计算视频级交互
\[
I_j(s)=Q_j(V_{\rm both})-Q_j(V_{\rm high})-Q_j(V_{\rm low})+Q_j(V_{\rm none}).
\]
此值是**整体策略×来源的交互**，不是“找到FFN组非线性因果”；动作只改变FFN，但下游正常Attention/后缀自由响应。每个Q轴单独保留数值、方向、失败/缺失；不能把任意VBench binary或MUSIQ和RAFT混成总分。异号交互是需要解释的source异质性。

辅助分层：在**已有视频表现出选择性差异之后**，于真实配对RUN保存同源step pre/postlatent、选定site FFN preQ2/postQ2/gate、第一receiver以及真实历史缓存状态，比较：
\[
\delta u_{k,l}=u_{P,k,l}-u_{D,k,l},\quad
\delta z_{k+1}=z_{P,k+1}-z_{D,k+1}
\]
并严格注明它们的输入分别属于自由学生/自由Dense。另用同学生输入的 Dense shadow 分离**条件局部函数差**与**上游状态差**，观察首次明显偏离边界；不将全视频的自然差异直接归给某个组。O/D/Q/P作用仅在**视频恢复现象已定义的有限位点**上服务于机制解释，G17 v2的local Gram及 \(W_2^\top\) 是可选证据而非主算法。

## 4. 完整视频实验路线：优先重用，之后最小有判别力的新生成

### V0 · 已完成视频因子审计（READ_ONLY；没有新视频）

复用P64 898/96 四臂和High-Y对照、High-P64四来源、UB1 20来源、BF16 value/DCAC 20来源及SharQ/TEAL视频。每方法严格登记各自numeric recipe、source、seed、控制policy、真实FFN作用范围和已测轴，**绝不跨不同后端/方法做虚假的配对因果差**。对于有原MP4的四臂，逐视频制定分段行为见证：对象出现/丢失，动作起止、方向、幅度、四肢/翼/鳍轮廓，背景/相机运动，曝光/残影、闪烁；视频必须实际逐帧及连续播放检查（当前尚未完成，不能把旧AI接触表冒充）。

V0的决定：
- 若Low-only的画质损失有明显动作/事件对照，H-F进入特定任务验证；
- 若RAFT变化主要来自背景/相机而主体未明显变化，H-C优先；
- 若898/96交互异号可由prompt、场景及动作归因解释，登记每source机制与反例，不强行平均；
- 若旧视频未能区分，明确MISSING，并进入V1新增有区分力的source，不调整错误代理到“看上去拟合”。

### V1 · 视频优先的因果恢复（NEW FULL ROLLOUT；待不同Agent评审批准）

**固定一个已原生运行的数值族和执行策略作为主要测试床**，首选已经有全160真实视频与4臂因子结果的Chipmunk P64；如使用BF16、HiF4/NVFP4或不同权重版本，**分开实验**。新runner支持按source/seed/step/expert/block切换Dense或P64，确保每臂4步×40个FFN正常生成、未被干预位点仍是该臂学生实际state，hook Dense恢复只表示当前site完整计算（没有跨RUN张量移植）；每条完整81帧MP4及最终latent实际落盘。

**实验臂与强对照**（优先分批，非一次全部运行）：

| 类型 | 完整视频arm | 作用与可证伪点 |
|---|---|---|
| 控制 | Dense / Instrumented-neutral / 原P64 Both | 160位点数值身份、observer中性、真实基线 |
| 已有主因子 | High-only / Low-only | 相当于Both策略分别恢复完整Low/High第二调用；已有898/96须reuse，不重复收费 |
| 追加阶段对照 | 同数量、同预算的High/Low替换；若需要重新选择层，先冻结层集合 | 检验阶段恢复是否具有不同最终视频作用 |
| 分层选择 | 早、中、晚各**相同数量**代表层恢复Dense，均覆盖不同深度/两个expert，严格记增加W1/W2调用 | 只有源级完整视频发现阶段作用后才做，避免无限组合 |
| 匹配控制 | 等实际恢复层数、等K/额外MAC或更严格等实测FFN时间的random-site恢复，以及同预算local-energy优先恢复 | 区分“任何恢复计算”与“特定更新功能”；冻结随机数、层数、K和视频seed |
| 历史机制 | 同数值族、同来源、同High范围old-Y-only vs High-P64，必要时加合法No-cache静态K策略 | 分离历史冻结和纯通道选择；只有已接入同原生后端才纳入 |

**首批source**：以898（主体/背景ROI需重新校验）、96（相机/低dynamic）、979（High-P64已观察到画质+运动异常）、1040（不同运动响应）作为**曝光开发对照**；新增方法的blind确认不能使用它们。优先利用已有视频/同RUN全160数据；新增缺失因子才申请新生成，记录未满足的runner和资产条件。每个来源在不同seed下仍是同一image-cluster的嵌套观测。不要选只显示“有益效果”的样本，包含Dense本身缺少动作、强运动、静态/低运动、相机/物体相对运动等场景。

**阶段层集合**：Pilot预先从40层内等量抽样如早{2,4,6,8,10,12}、中{15,17,19,21,23,25}、晚{28,30,32,34,36,38}（每组6层），但先审模型真实编号、每层职责和预计成本；各组的恢复须在完整4步视频执行，不能在单FFN后停止。上述编号是**待评审候选**，不是实验获批的冻结集合。若只恢复6层无法影响行为，结果仅对这一有限剂量有效，不可推断该阶段不重要。

### V2 · 定位视频恢复的第一计算图原因（NATIVE DIAG；紧随V1）

选择真实视频动作质量改善与无改善的**成对反例**，沿FFN输出→gate/residual→下一个receiver→step更新→下一step输入→video逐边捕获。四边界：
- O：当前位点post-W2/pre-gate局部恢复；
- D：固定post-Q2分析能否由某组列贡献产生；
- Q：真正pre-Q2 group skip与Smooth2/Q2受影响尺度闭包，确认物理可执行性；
- P：自由后缀与指定receiver观测对照。
候选组/ROI来自**先于诊断结果冻结的视频事件**；禁止仅按Dense最大flow选主体ROI。group K64与layout是物理候选，不宣称稳定语义神经元。需要重读已存在资产时只读；缺少same-RUN匹配数据明记MISSING，申请针对性新producer。G17 v2的一阶Gram只能作为候选解释：若和finite video行为矛盾，以video为准。

### V3 · 独立来源与原生省算验收（FULL VIDEO + SYS）

通过V1/V2形成机制驱动的**具体物理FFN方法**后冻结支持/时点/数值族/权重/选择器；以独立image-cluster重新采样，保留多seed配对，数量由来源级方差/失败率与预算事前决定（例如pilot计划8–12个未曝光来源×2seed，只是设计规模，不是功效保证）。至少有Dense、neutral、机制候选、同K且同执行形状的强基线与matched random，以及按需原有P64/TEAL/SharQ。正式声称“VBench-I2V标准结果”须依官方套件每image-prompt**五条真实独立采样**；仅开发自选视频时标CUSTOM_INPUT，所适用指标仍需调用真正官方函数，不伪造9维总分。

**视频验收必须完整**：
1. **动作/语义**：冻结prompt动作动词、目标物、起止方向与事件完成标准；主体轨迹/速度/加速度、接触交互/事件顺序。若Dense事件未发生，分层报告基线失败；人工盲评或经过验证的视频语言读出不可被flow替代。
2. **运动结构**：主体相对相机运动、背景/相机全局光流、时间窗、过度运动/衰减双向容差；完整报告RAFT分布而非仅均值。跟踪无效、目标消失、遮挡/物体丢失单列失败或缺失，不把失败帧删除以获得漂亮均值。
3. **其他完整质量**：有效原版八轴I2V评测、合法的camera-motion子集、flicker/temporal、MUSIQ、主体与背景一致性、I2V原图保真、光照/重影、全81帧实际连续播放的盲审及抽样连续截图。指标相互独立、单位与样本数清楚，不能只用official motion_smoothness/subject consistency作为运动成功。
4. **原生成本**：真实W1/W2 skip与K64/tile布局、native packed核、Q1/Q2、cache build/read/free、selector和布局重排；公平冷暖机下FFN和full-system wall、峰值显存、变异，报告其他瓶颈与降算不降时的负结果。

**多目标规则**：先验证主体动作、事件、语义与整套画质在预注册容差内非劣，再报告FFN/system speed-memory Pareto；无单个大加权分数允许掩盖严重主体动作崩溃。来源内seed配对，来源等权、尾部失败与source-cluster不确定性一起报告。任何实验后改变指标/容差/源选择必须另立探索版本。

## 5. 逻辑链、反例与决定实验去向

\[
\boxed{\mathrm{OLD\ FULL\ VIDEO\ FAILURE}}
\rightarrow \boxed{\mathrm{HYPOTHESES}}
\rightarrow \boxed{\mathrm{FULL\ ROLLOUT\ FACTORIAL/RESTORATION}}
\rightarrow \boxed{\mathrm{BEHAVIORAL\ SELECTIVITY}}
\rightarrow \boxed{\mathrm{FIRST\ CAUSAL\ COMPUTE\ BOUNDARY}}
\rightarrow \boxed{\mathrm{NEW\ PHYSICAL\ OPERATOR}}
\rightarrow \boxed{\mathrm{BLIND\ FULL\ VIDEO + SYS}}
\]

| 预注册观察 | 科学决定 |
|---|---|
| 低运动RAFT但对象动作符合prompt，只是camera/background被抑制 | 放弃把RAFT下降直接归为动作崩溃；H-C或相机分离解释优先 |
| old-Y-only退化、High-P64在相同source恢复事件/轮廓，random同成本不足 | 支持“fresh更新必要”的**限定cache机制**；再定位写回边界，不宣称所有FFN组相同 |
| High/Low完整视频效应异质且相反，局部L2不能解释 | 保留阶段×source作用、反例和事件类别；通过更多source检验是否可稳定预测 |
| 等成本random与针对性恢复同样改善动作和画质 | H-C/容量解释更有竞争力；不继续推广特定“运动组” |
| O或D修复视频有效但Q闭包真实skip无效 | 功能可定位但不能据此构造可执行FFN省算 |
| 真正动作/语义修复但native SYS无收益 | 机制研究成果可保留，方法/加速路线不通过 |
| 开发源改善、独立新source失败 | 不宣称普适，报告条件失配并重新审查核心假设 |

**新方法门槛不是“较好local proxy”，而是：在真实视频中有清晰行为效应、有特定机制的竞争解释可证伪、能落实为FFN真实省算，并在独立来源上保持行为质量。**

## 6. 可执行入口、缺口和资产预登记

仓库已有[完整视频实验手册](../agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)与工具 \`tools/ffnds_experiment_plans.py\`、\`tools/ffnds_run_registered_video.py\`、\`tools/ffnds_stage_registered_i2v.py\`、\`tools/ffnds_cluster_metric_report.py\`。手册示例是旧Dense/UB1 variant；**不能假设既有runner已支持P64的任意site/阶段恢复**。新动作必须先检查实际P64全160原生runner、config、hook Dense与neutral identity、四步状态/缓存的生命期与K真实物理对应，再扩展最小必要接口。没有真实模型环境、HF权重/视频资产与ACTIVE授权，本文件本身不生成MP4或真实评分。

**REUSABLE_ASSET_PLAN（方法侧，后续审查须实例化）**：
- \`video_pair\`：原图+prompt+seed+模型及数值recipe、完整Dense/candidate MP4、最终latent、视频帧数量/时间戳、原始视频asset locator；跨数值族禁止称同条件。
- \`behavior_readout\`：事件标注、对象track、camera/background flow、失败帧和观察者盲审、各官方分维结果、RAFT源级时间窗及置信度；标注与读出冻结版本和适用/失效条件。
- \`ffn_action_and_native_trace\`：160位点真实策略开关、mask、保留K、native FFN动作、e5、合法数值族、selected receiver与step latent（只在有实际producer时生成）、time/peak记录；不将未存储的数据写作已捕获。
- \`mechanism_factorial\`：per-source-per-seed四臂实际视频配对、每维interaction及合法stage层介入记录，所有原始向量与source-cluster统计；每项均提供producer/run identity、method revision、coordinates/units/dtype、存储locator、验证状态和禁止复用场景。
- \`runtime_receipt\`：source×arm×run真实出入状态、初始噪声比对、执行/系统成本、失败/中断与重试版本；任何未产物登记NOT_PRODUCED，而非补0。

**执行许可**：本文件是**作者提出的设计修订**，不是独立Agent批准。须针对每一个新增CPU/GPU/模型/视频scope执行[独立预审状态机](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)，含不同真实Agent反例、作者修订、recheck、ACTIVE冻结及RUN/EXIT。当前仅完成repo文本与已存结果审计；没有新视频、GPU或新增行为注释。下一步的第一个可交付物应为**V0现有四臂视频的来源级动作/相机/画质事件表和缺失资产表**，随后才是新runner与V1独立审查，不允许以CPU局部代理替代视频因果实验。
