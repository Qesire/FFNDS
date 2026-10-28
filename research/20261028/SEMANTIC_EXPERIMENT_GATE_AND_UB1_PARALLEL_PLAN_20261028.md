# FFNDS · 稀疏方法优先的语义实验门槛与 UB1 并行机制研究

**2026-10-28 · DESIGN_ONLY / STATIC_ONLY / DEFERRED_SEMANTIC_GPU.**

## 0. 定位、当前优先级与禁止冒领的证据

本文是**一个增量实验设计**，不是新的统一研究目标、现行方法权威正文、GPU 授权或新方法讨论队列。现行顺序仍为：
1. [当前上下文](../agent_protocol/CURRENT_CONTEXT_PACKET.md)和[现行M-A/M-B/M-C方案](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)中用户指定的 **M-B B0 → B1 → 合法 B-H/B-F 学生**；M-A0 6016 维真实组剪枝学生是强对照/可用结构稀疏起点；M-C 维持独立。
2. **UB1 失配机制的离线并行分析**：仅复用已存在的视频、真实原生统计、RUN及SHA；出现缺失资产时登记 locator/producer 与 NOT_MEASURED。下述 U0-U2 是机理研究，不是另设 GPU 主任务。
3. **语义实验先设计后激活**：只有主线出现通过 S1-S4 的稀疏候选（原生少算、完整视频质量可用、真实系统成本有益），才考虑登记新增文本对照视频；当前所有新增生成、作者 ML LM 评分 API、原生干预、量化重拟合与新 GPU 作业均为 **NOT_REGISTERED / NOT_RUN**。每次执行须先检查 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 并冻结确切范围。

原始证据：
- [A0 局部6016恢复](../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)已完成4个(step,block)×两初始化、16训练/8旧诊断来源、真实窄 HiF4 Q2；尚非全部80个 expert-block FFN 与自由视频。
- [UB1 consumer20质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：20个**已暴露开发来源**，fresh Dense/consumer一一同初始噪声配对；成像 72.629→71.626，RAFT连续幅值6.894→1.152，官方二值 Dynamic Degree 6/20→0/20；仅10来源I2V subject与另10来源I2V background，各自统计；人工语义与camera motion未完成。
- [UB1机制审计](../20261027/UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)在旧formal20两个场景中发现犀牛与住宅运动衰减；住宅prompt并不要求运动。当前 consumer20 与旧 formal20不混成一个可独立确认集合。
- [C5合成CUTLASS](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)在两个合成大M形状上的全在线融合收益仅约3.5%、0.7%，没有Wan端到端加速证据。
- [G13 O/D/Q/P](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)与[Calibration V2](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)已定义原生边界、量化差项、独立source数据单位；不要重复发明现有协议。

## 1. 主线：先达到“可用稀疏方法”

### 1.1 稀疏的操作定义

一个方法应把实际 W1 / GELU / HiF4 Q2 / W2 等计算中某部分工作移除（例如剪掉完整64通道组后重新形成更窄W1/W2），并真实执行比原模型更少的算术/访存或更短时间。输出低秩 `y=Φc(x)` 可能构成有价值的**紧凑函数**，但除非编译为具有相应可测省算的算子且存在明确被跳过的计算，不能把它直接写成“动态稀疏”。在线条件选择若在计算完整W1/全部Q2后才获得，属于后验重要性或付费代理，需要计入观察成本。

**可行的S1工程锚点**：M-A0从216组保留94个完整64维组，h=6016；在W1/W2同步真正变窄的条件下，二者主GEMM MAC相对原h=13824减少56.48%。这相当于静态结构组稀疏/裁剪后执行紧凑dense GEMM，**不自称输入自适应动态稀疏，也不自称端到端56.48%加速**。 M-B继续其原定优先级；M-A0作为能运行的同资源竞争器，不要求替代当前M-B任务。

### 1.2 分级 gate：与科学价值对应

| Gate | 预期证据 | 当前身份/入场判据 |
|---|---|---|
| S1 NUMERIC_SPARSE | 非零省算，真实组边界合法，原生QDQ/Q1/W1/GELU/Q2/W2/gate、重载/中性合同，完整native算子校核 | A0局部四site已有；尚需全部适用site的编译/验证 |
| S2 FULL_ROLLOUT | 当前实际4-step high/high/low/low、两套expert的40 blocks都按声明的方法执行；视频完整81帧，声称的工作没有偷偷Dense补算 | 尚未完成该新方法合同 |
| S3 VIDEO_USABLE | 预冻结包含动作、静态、camera、细小目标的独立image来源；完整视频质量、相机与主体运动、纹理/光照/语义动作复核；无严重崩溃，独立失配率/尾部和源级配对效应通过事前质量界 | **视频质量最低门槛；新增语义生成仍需 S4 物理收益** |
| S4 PHYSICAL_VALUE | 实测完整 FFN wall、整端 generation wall、显存、packing、QDQ、observer/kernel成本；匹配对照至少显示真实 FFN wall 正收益且整端不因新增机制退化；若宣称端到端加速需测到实际正收益 | 支撑“有效率收益” |
| S5 INDEPENDENT_CONFIRMATION | 冻结模型和阈值，在真正未暴露来源+独立seed上复验质量/成本 Pareto | 支撑最终投稿结论 |

S3验收界限不要事后按UB1坏结果选定。建议在开发集先冻结：画面/动作关键场景的人工严重故障定义、运动失真要素及允许退化边界、Image/Aesthetic或LPIPS的有向不劣阈值，另记录不同seed Dense自然变异的参考水平。真正确认时用source配对CI+坏例尾部。**有动作要求的提示词**要同时看事件完成和物理轨迹；**静态提示词**不要求保持Dense可能多余的运动。S1-S2任何时候可独立研究，不等待语义指标。

### 1.3 主线实验最短路径（不重新排队）

- **M-B B0/B1 现行优先**：真实block39 post-write u/head/flow producer、neutral、有限响应（与此前合同相同），检查source-joined的J_s、gate+suffix连续/离散差异。并行使用A0作为 matched comparator。严禁把head65连续充分统计当成全40block语义充分统计。
- **A0强结构基线的后续研究对象**：在同样参数/训练预算下比较 Energy94、Random94 及明确冻结的组评分方法；训练对象必须包括所有80个FFN函数（两个expert各40个block）、四步与合法Q2。Teacher状态局部恢复和Student自由轨迹分开报告；优先比较全端质量，再讨论更复杂动态路由。不要将四site teacher-state小误差外推成可用视频。
- **端到端物理成本**：记录源×seed同硬件、精度、scheduler；分开一次性准备、FFN step/block耗时、所有FFN总耗时和完整视频墙时。运行前冻结计算图路径，评测是否在模型生成中执行了完整网络/隐藏的额外恢复。理论压缩比只放在算子理论列。
- **动态方法升级（条件性）**：如果后续方法要称“动态稀疏”，至少比较输入/文本/step依赖的可执行选择与同等MAC的固定组，并证明选择在实际W1等被省算之前可获得；selector、materialization、K/N小tile占用、权重resident与额外QDQ全部算成本。可基于M-B功能发现提候选，但当前不抢占B0/B1。
- **失败停止/转向**：若完整生成崩溃或原生省算消失，先以首个不匹配合法数值边/物理内核定位；将失败限定为该结构及条件，再考虑替代组宽、student分布校准或不同合法执行表示。不引入以ACL分数为目标的额外训练以“救”尚未可用的方法。

## 2. 并行 UB1 失配研究：已有数据优先，不与主线争夺 GPU

### U0 · 全20来源视频复核与文档标注（offline，第一优先）

观察单位：consumer20的**20个 image-prompt来源**，每个来源一条fresh Dense、一条consumer；同来源的81帧和40个光流帧对是时间重复观测，不是独立样本。

现有工具：`tools/ffnds_video_metrics.py`与实际RAFT/官方输出；重用 `UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.json`、source/prompt分类、原始视频SHA；若文件未保存在当前可访问资产中，记录MISSING_ARTIFACT并先做JSON诊断，不假装运行视频算法。

1. **事前构造提示词标签**：ANIMATE_SUBJECT（存在清晰主体动作，如叼鱼、游泳、吃草）；CAMERA_REQUIRED；STATIC_ALLOWED（如住宅）；AMBIGUOUS；对每一条保留原文本、动作谓词、可观测目标、相机指令、物理可行性。标签不依赖生成结果，以免结果选择偏差。可以有多标签，但聚合时簇仍为同一source。
2. **统一观测坐标**：全帧RAFT连续值（已完成）、是否有相机运动、主体运动相对相机轨迹、轮廓/关键点位移、方向一致性、可跟踪性和模糊；事件是否发生及完成、人物/物体是否丢失、状态持续性。最先复核20来源中已有最大运动差与高动作语义者，同时按冻结表覆盖低变化来源；该选择只用于机制发现，所有20来源都保留统计。
3. **评估器隔离**：首次用预定义ROI/跟踪算法及可用的本地计算/人工双盲评分；MLLM语义自动评估参考OSC/TC断言，但应校验准确率/校准，不把其单一分数作真值。额外离线推理也需登记其真实耗时与输入身份，不是GPU生成许可。
4. **输出**：逐来源 `motion_ratio`（Dense>0才定义）、`subject_relative_displacement`、`camera_displacement`、`event_observed`、`terminal_state`、`tracking_confidence`、`blur_delta`、`prompt_class` 和各自NOT_RUN/UNKNOWN状态；保存首尾及状态变化关键帧、时间索引和证据SHA。聚合用source bootstrap描述，弱小类别给出n而非不稳定显著性。

**可区分性**：若全部场景RAFT下降但只有动作提示词存在明显事件丢失，支持“运动幅值变化”和“条件语义失败”是不同函数；若独立ROI轨迹不下降而RAFT下降，优先检查纹理/锐度/估计机制；若相机提示词失效而主体移动尚存，提示相机与主体作用分离；若静态场景变静止，按prompt判断其正负质量，不预设Dense运动更多一定更好。

### U1 · 保存的latent/flow和原生组作用账本（offline、仅在同identity资产实际存在时）

先列每个source×seed×step×expert×block实际保存的是原native `q,h,a,y,e5,u`哪一边，以及poststate/flow文件SHA。按G13区分：(i) 固定post-Q2列的signed直接贡献；(ii) 重新Q2时补集scale/code的连带变化；(iii) 自由后缀的传播和跨四步误差累计。缺完整组张量时先只分析已保存的真实投影与频带signed cross，标记实际可支持的机制，不从标量FFT推断某组为“运动神经元”。

可检验的恒等式（只在相同张量空间、同一**固定线性**时间差算子D下）：

```text
d = D(Y_D); e = Y_S - Y_D
||D(Y_S)||² - ||D(Y_D)||² = 2 <d, D(e)> + ||D(e)||²
```

其结论是“误差与时间变化的signed方向”而不是RAFT的代数分解。必要时先对21 latent视频时间槽构造D，清楚区分去噪step、latent时间和解码81帧。测试poststate对原本已观察的跨来源/跨seed符号反转的解释能力。所有对UB1的新计算因果主张保留OPEN直至同状态合法干预。

### U2 · 冻结竞争假说、使用现有资源获得最便宜判别

| 假说 | 不同预测 | 便宜的先行观测 |
|---|---|---|
| H-MEAS 图像纹理/跟踪偏差 | RAFT下降显著，但人体/动物ROI位置或独立轨迹下降不成比例 | 已有成片 ROI轨迹与sharpness对照 |
| H-CAP 静态组选择过度保留外观 | 动作丰富source坏例与被删组的signed时间差作用相关 | 真实已存Q2/W2/gate数据；按source group，不用事后回归冒充独立因果 |
| H-DATA 校准与目标分布失配 | 同预算更多motion-domain source或来源数会减少动作尾部失败 | 先用CAL24来源统计/稳健性识别潜在差异；新fit与视频需以后注册 |
| H-TRAJ 传播/状态偏移 | 局部FFN输出相近而下游flow/latent时间方向持续偏移 | 已有poststate和whole-flow反事实范围，逐步记录signed作用 |
| H-SEMANTIC 文本指定状态失配 | 事件终态与动作断言失败，即使整体背景保持 | 旧配对视频的prompt-conditioned事件盲评 |

U0/U1能产生对新方法的“必须避免什么、在哪里检测”的信息，但**绝不以UB1失败为理由直接更改当前M-B目标、强制引入语义loss或大量新GPU实验**。

## 3. 语义研究：仅 S3 与 S4 均达成后开始新增生成

### 3.1 主要研究命题与模型对象

固定初始图像 `x` 和初始噪声 `ξ`。在语义上合法的成对文本 `c+`、`c-` 下，比较同一Dense/稀疏模型的文本响应：

```text
R_D(x, ξ) = a(G_D(x,c+,ξ)) - a(G_D(x,c-,ξ))
R_S(x, ξ) = a(G_S(x,c+,ξ)) - a(G_S(x,c-,ξ))
τ = E_source,seed [ R_S - R_D ]
```

其中a为**同一固定定义的有符号连续读出**（例如剔除相机运动后的x向位移、对象间距离变化），不是根据每个prompt重新换符号的成功率。如果是抓取完成、状态持续，独立使用预先定义的二值/分级事件断言而非强求梯度。报告 `R_D` 和 `R_S` 两个原量及 `τ`，避免Dense本身不响应时使用不稳定比值。若稀疏具有真实输入条件mask，记录当前条件下的选择；若是静态稀疏，比较文本条件和输出变化即可，不假设有动态路由。

核心估计量分开：
1. **动作/状态正确率差**：在每条source×prompt×seed配对下，稀疏－Dense，按source等权。
2. **语义对照响应差**：同source×seed的c+与c-差，进一步做稀疏与Dense的差分（上式）。
3. **非语义质量与物理成本**：同冻结模型与推理配置的主体/背景、运动、细节、camera、完整视频和FFN/全网耗时；不合成单个无解释的“语义与速度分”。

### 3.2 输入设计：先保留 I2V，控制首帧可行性与语言变量

**实验簇单位是image identity（包括近重复检测），prompt变体嵌套其内。** 选择物体真实存在、动作由首帧物理状态允许的照片；不要让c+要求不可能的状态或与首帧矛盾。构造两类主要干预：

A. **可观测有符号动作**：同一主体“向左/向右移动”、摄像机“左平移/右平移”；只选择两方向都可行的场景，冻结同一个独立方向读出；语言细节、描述长度、主体和场景尽可能匹配。

B. **状态转移 / no-op**：如“手提起桌上的杯子”与“手接触杯子但不提起”。若基线难以成功，记录其失败；不得只保留Dense成功样例作为总体效果。评测对象为抓取发生、物体离开桌面、最终状态/持续帧数，而不只是RAFT。对于仅具有首帧的Wan I2V，沿用OSCBench/TC-Bench的**语义断言方法**而非直接套用带末帧条件的I2V分数。

建立固定字段 `{source_id,image_sha,prompt_pair_id,prompt_text,predicate,actor,patient,initial_state,target_state,validity_check,readout_id,scene_stratum,requested_camera}`。对难以被模型理解的prompt也留在结果集，用Dense成功率和实际条件敏感度作为两个独立结果。

### 3.3 顺序采样与门槛（建议的实验预算，不是当前授权）

**Stage E0（现在，可不做新模型生成）**：UB1历史视频盲评、测试标注schema、开源OSC/TC评估器与样例标注一致性；可预备小规模语义评估器对抗测试（blur/overlay/静态欺骗），但不接入方法选择。

**Stage E1（S3+S4之后的小规模可辨识性pilot）**：8 个真正新image簇 ×2对照prompt ×1 seed ×2臂（同版Dense/稀疏）=**32个完整视频**。按明确动作/状态变迁与静态/镜头等分层，至少每个可检验类别有代表；目的是检查 I2V 中文本操纵、读出、标注一致性，不能用于声称独立泛化。Dense若两种文本都无响应，记录“模型能力/数据天花板”，调整下一版本输入条件，而非静默丢样本。E1执行前必须按ACTIVE独立登记上限与资产。

**Stage E2（冻结确认建议）**：24 个真正未暴露image簇 ×2 prompt ×2 seed ×2臂 = **192个完整视频**（如source方差很大，再由pilot估计样本量，不把24视为保证统计功效的神奇常数）。按scene/semantic类型和prompt pair源级分层；所有来源都未进入校准、训练、方法择优或E1；每个生成保留完整逐帧收据。正式VBench-I2V的Image Suite、每prompt五个实际独立视频和九维工具属于另一**官方可比性**任务，不以本192视频实验冒充官方leaderboard。

**Stage E3（确有语义效应后，G13定位）**：先用已有hook收集相同x,c,ξ和准确分支的状态/FFN写回；在同native prefix做O→D/Q→P局部有限干预，优先挑选有方向的少量block/step与matched随机组，再覆盖整四步全网络自由rollout。严格区分postQ2直接列与preQ2共享scale的连带效应；比较Sham、随机、同组数、同局部误差组与门控方向；报告原始行为、实体关系、状态readout以及非目标画质是否同时变化。

**Stage E4（方法机制迁移）**：仅当E3建立有辨别力的结构时，在不更改主线现有强基线身份的前提下测试语义相关importance / M-B paired-joint / motion-source calibration，和同W1/W2 MAC、同后训练预算的Energy94、Random94及可用稀疏候选匹配。是否使用MLLM监督/轻量后训练由E3结果决定，所有训练成本和在线observer严格分开。

### 3.4 指标合同（手工+自动，避免指标幻觉）

| 轴 | 必须保留的观测 | 注意 |
|---|---|---|
| Language→action | 目标动作是否出现/方向是否符合；盲审关键时间帧 | prompt与首帧一致，MLLM标注用人工样本校准 |
| OSC terminal/continuity | 终态正确性、状态首次到达时刻、持续范围 | 与运动幅度独立 |
| Temporal relation | 前后事件顺序、对象关系变化、同一实体连续性 | 借鉴TC-Bench assertions |
| Physical motion | camera-compensated subject轨迹、方向、RAFT连续幅值 | 正常静态prompt不因静止被罚 |
| Appearance | VBench I2V分维、画面/纹理/小物体/光照、人工全片 | 保留完整多轴和source-level坏例 |
| C execution | 真正W1/GELU/Q2/W2工作计数、量化、selector/packing、FFN wall/全网wall、VRAM | claim任何加速时必须实测 |

自动评估器本身先进行标注者间一致性和人机相关性复核；检查严重模糊、物体消失、静止帧等反例。OSCBench官方代码提供20帧抽样、逐维提问和相关性分析思路： https://github.com/iLearn-Lab/ACL26-OSCBench 。VBench I2V官方协议参见 https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v 。TC-Bench论文 https://aclanthology.org/2025.findings-acl.241/ 。作者benchmark的T2V/首尾帧I2V身份不能直接变为本LightX2V单首帧 I2V分数。

### 3.5 配对效应、来源聚类和可靠性

- 主要独立单位为image cluster，prompt pair与seed为其内重复，frame/token/block/row是嵌套观测。先算每source内的平均配对差，再对source等权；用source cluster bootstrap构造不确定区间，报告各类别sample n与坏例尾部。显著性/置信区间只在足够独立来源时作确认性解释。
- 单一实验可预先设立主要结局（例如prompt-conditioned event completion difference）和几个共同次结局（state continuity/subject track/cost）；其余作为机制诊断，避免大量事后择优。
- 固定发问、MLLM版本、提示词、温度、抽帧位置、人审盲标、物理mask、帧率/分辨率；标注不确定或模型拒答不能当成0或完成。
- 训练、calibration、历史开发、E1探索、E2确认按image id及近重复隔离。若用新数据改过一次method/threshold，原确认集降级为开发暴露，下一独立确认重新注册。
- 真正的ACL贡献要求语言条件的**差分作用**和其可定位的正常Wan计算关系；模型质量常规分数仅是约束。只有外观/RAFT改善、没有条件响应差异时，清楚保留为视频压缩/动力学研究成果即可。

## 4. 与当前仓库的实施接口

**不新增优先级、方法队列或ACTIVE任务。** 本文件是一次性静态设计，现有 [P0-P4导航](../RESEARCH_NAVIGATION.md)、[M-B下一Agent](../agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)和所有原始RUN保持权威不变。

- source bank/实验注册：`tools/ffnds_experiment_plans.py`（新source先分簇再选prompt、正确标记历史暴露）
- source/row/native身份：`tools/ffnds_native_geometry.py`、`tools/ffnds_validate_calibration_operands.py`
- 登记的视频生成与官方I2V：`tools/ffnds_run_registered_video.py`、`tools/ffnds_stage_registered_i2v.py`（只在S3+S4后并有具体ACTIVE scope时启动）
- 视频解码/真实质量收据：`tools/ffnds_video_metrics.py`、`tools/ffnds_video_metric_gate.py`
- source聚类：`tools/ffnds_cluster_metric_report.py`
- 机制图：G13自然/因果/可执行三级；G14论文→真实算子焦点新增 LIT 语义读出节点时，保留原20来源视频的VIDEO身份与作者文献LIT身份分离。

**最近的实际下一行动**（不变更原主线）：继续 M-B B0 producer/neutral；UB1旁线优先核实已保存20来源视频与prompt ledger并产出事件/相机/ROI离线标签及缺口表。语义实验E1–E4当前全为DEFERRED；没有运行或提交任何新原生实验。
