# UB1 运动幅度衰减：外观—动态分离、计算压缩敏感性与因果机制研究

> 2026-10-27｜AutoResearcher NOTICE→MAP_FLOW→DERIVE→EXPAND→DISCRIMINATE→RECORD；SRC/NATIVE-VIDEO-OBS/ALG/LIT/OPEN分开记录。**本稿是既存数据的重读和研究方案，不是新模型/RAFT/视频/后训练执行。** 当前研究入口：[UB1方案](UB1_STEP_BUDGET_CALIBRATION_AND_VIDEO_GENERALIZATION_RESEARCH_PROPOSAL_20261027.md)、[A5+HWG专题](A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)、[质量协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)。

## 0. 首要修正：外观近似与运动幅度保真是不同研究对象

**已经实测的不是UB1普遍“失去运动能力”，而是两条历史正式20来源里，UB1相对于配对Dense的原版VBench/RAFT连续top-5%光流幅值分别下降61.9%、90.81%；犀牛的持续头/后腿动作在逐帧静态复核中也显著减弱**。原视频共81帧，832×464/16fps；用于RAFT的41帧和40相邻采样对按原官方函数计算。注意逐帧对并非40个独立source，新方法质量阈值不可事后由这两例建立。完整数值与SHA见[原版动态幅值实测](UB1_DYNAMIC_DEGREE_RESULTS.json)、[原测量说明](UB1_DYNAMIC_DEGREE_RESULTS.md)、[逐帧独立审阅](UB1_INDEPENDENT_VISUAL_REVIEW.md)。

| 已暴露formal20配对，seed42 | Dense连续幅值 px | UB1连续幅值 px | UB1/Dense | 相对降低 | UB1相对Dense较小的40对数 | 官方二值Dense/UB1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| source882，犀牛吃草 | 3.3932188034 | 1.2936739003 | 38.13% | 61.87% | **38/40** | 0/0 |
| source80，山坡住宅 | 1.9949876755 | 0.1833207669 | 9.19% | 90.81% | **40/40** | 0/0 |

**窗口重检（每个连续8对、窗口不重叠；来自已存40数值，不是再次执行RAFT）**：
- 犀牛五窗口UB1/Dense＝**25.6%、49.4%、29.8%、32.5%、59.5%**。
- 住宅五窗口＝**27.3%、8.5%、7.4%、9.1%、5.6%**。
- 这表明差异在两场景各自大部分视频时间都存在，**不证明跨scene/seed泛化，也不是“28/40显著性检验”**；同视频frame对在时间上相关，且排序/密度方法的测量噪声未做置信校准。

原函数指标定义：保持原RAFT checkpoint及官方\`compute_dynamic_degree\` 0.1.5，取原832×464图像，视频16fps隔两帧采样，RAFT每对20迭代；每对取绝对光流幅值最高5%像素的**均值**，得到40个pair score。超过10.875 px的pair需≥10才使该视频binary=true。这四段全部0对超过阈值，因此原binary动态分数四段全0。**连续top5统计不是“整段原版Dynamic Degree官方连续质量分数”**，是从原函数中间结果提取的诊断量。原正式20只有Aesthetic/Imaging/Motion Smoothness三项历史VBench聚合，Motion Smoothness Dense0.983923、UB10.983850；未执行全20来源的动态幅值、二值Dynamic Degree或其它全维验收。

**两个source的提示词不同**：\`a large rhino grazing in the grass near a bush\`具动作语义，已逐帧看到Dense头部/后腿姿态改变而UB1相对静止（但UB1并非完全静止）；\`a bunch of houses that are on a hillside\`没有动作/镜头要求，UB1几乎静止**不自动更差**，Dense额外运动也可能是无需的镜头/纹理变化。由此定义强证据“相对Dense运动衰减”；动作语义、主体物理动态、整体能力、跨来源倾向仍OPEN。

## 1. 新的测量学问题：运动减少，还是光流可观测性减少？

**最强竞争解释**不只是“本来静态场景合理”：RAFT输出的2D光流来自视频外观，本身可能随纹理、模糊、编码和遮挡变化。若UB1的动物外形纹理被轻微平滑，在物体真正移动速度相近时，RAFT也可能估计更小的运动。反过来Dense中纹理闪烁/不稳定对应关系可能扩大光流，尤其全帧Top5最高幅值倾向选择非主体运动、镜头运动或局部异常。

因此未来至少报告四个不同对象：
1. **全帧连续光流幅度** \(\bar q=\operatorname{mean}_{t} \operatorname{mean}(\operatorname{top5\%}|\operatorname{RAFT}(V_t,V_{t+2})|)\)：本次确已测，但不等于真实动作；
2. **主体ROI和相机补偿运动**：分割主体、估计与固定全局相机变换/背景静止集合，统计主体轮廓中心/可追踪关键点的位置与光流方向及幅度，按每对visibility/confidence与原生两帧对齐；运动场估计多种算法、人工关键帧校验是质量控制；
3. **动作语义/物理过程**：grazing是否真正出现连续低头/嘴部活动，房屋无需移动；人工完整原16fps播放和来源prompt对照，不能让“接近Dense”自动定义语义正确；
4. **外观/跟踪可观测性**：局部模糊、sharpness、物体轮廓、局部光流正反一致性、遮挡及噪声变化，测试光流估计是否受外观变化而偏差。

**区分反例**：两段完全相同位移的纹理场，一个严重模糊甚至无纹理，光流算法的可观测性与结果可能不同；因此“RAFT小=物体一定移动少”不是一般定理。反之如果独立关键点/分割/轮廓轨迹也一致确认UB1位移缩小，支持**真实运动幅度衰减**而不仅是测量退化。

## 2. 把运动看作函数方向，而非直接命名“运动神经元”

Wan真实路径：同一视频I2V初始条件 → 4个去噪step(\`high/high/low/low\`专家；每step40个Block) → 每block \`norm2/AdaLN/e3e4→Q1→W1→tanh-GELU→Q2→W2→e5 gate残差\`，随后attention/blocks/flow/scheduler → 21个latent视频时间槽 → VAE解码81个RGB帧。**四个去噪step不是视频的81帧，也不是latent视频21时槽**。

UB1仅按step固定整64 hidden组 \((54,108,108,162)/(216)\) ＝25/50/50/75%，同均预算50%主GEMM MAC，再有rank256线性残差；offline C_mean在校准postQ2的 gated W2原组贡献上排序。它**不作当次token/场景输入自适应mask**。按总输出能量\(E\|g\odot W_{2,G}a_G\|^2\)选组，尚未优化某条视频的**跨latent视频时间方向**或视频运动语义。

可以先在对齐的真实张量上用一个冻结的**线性时间差算子** \(D_\tau\)（例如latent时间槽一阶差分；若有运动warp，固定同Dense估计出的warp，并明示插值近似线性）分解：
\[
Y_S = Y_D + e,\quad d_D=D_\tau Y_D,\quad d_e=D_\tau e,
\]
\[
\boxed{\|D_\tau Y_S\|^2-\|D_\tau Y_D\|^2
=2\langle d_D,d_e\rangle+\|d_e\|^2}.
\]
这是真实相同张量边界+固定线性D下的**恒等式**；它不等价于光流RAFT均值的代数分解，也不证明运动“神经元”。出现动态方向抑制要求有足够负的signed交互；**即使每个token FFN output SSE很小，运动方向仍可以被系统性改变**。

一个可构造的条件反例：局部像素/latent输出 \(Y_D(t)=s+ \epsilon v(t)\)、\(Y_S(t)=s+\gamma\epsilon v(t)\)， \(0\leq\gamma<1\)。当静态成分\(\|s\|\)占主导，空间相对误差可很小，而时间变化被缩至\(\gamma\)；这是数学说明**外观保真与动态保真不等价**，**不是测得UB1 \(\gamma=0.381\)**。如果只匹配单帧/全输出SSE，不会保证 \(D_\tau\)方向误差。

该数学形式的成熟邻居是VideoJAM对appearance vs motion监督与Inner-Guidance、Motive的motion-weighted gradient和校准数据精选；严格区分它们已经存在的机制与本仓库的**原生量化FFN下有效计算结构**。无需为发明motion loss而强行追求算法不同。

## 3. 四个可区分的机制而不是单一故事

| 科学假说 | 源码/证据连接 | 若成立的预期 | 最能降级它的观察 |
| --- | --- | --- | --- |
| **H-MEAS：光流可观测性/质量偏差** | UB1可能改变纹理；RAFT依赖局部外观 | UB1全图RAFT小，但高置信ROI物理位置/人工追踪可能仍与Dense相近 | ROI关键点/轮廓位移也持续、明显更小且跨跟踪器一致 |
| **H-CAP：被删组携带少见但动态重要的有符号功能** | 原C_mean静态平均能量选择；RT-Lynx提示当前激活与长期权重容量不同 | 固定支持保留外观mean但跨latent时间差/高运动来源tail差，少数未保留组在动态方向有低能量高投影 | 同输入恢复有预算组/重构消费者后无dynamic方向改善，差异仍在后缀 |
| **H-TRAJ：流场/状态传播致运动抑制** | 完整E0 P/N反事实已见不同source/seed signed变化，**不是UB1本人已验证的因果结果** | 某step对正常Dense运动方向的有限作用被改变，error沿四步生成与attention传播导致终端差异 | 同状态/全层后缀干预后运动仍相似或差异主要是视觉估计偏差 |
| **H-DATA：校准来源/目标偏向稳定外观** | UB1原24来源×256行/位置，不按场景真实动作分层选数据；补偿拟合后Q2 teacher local SSE | motion-rich校准来源或motion-conditioned loss/消费者重构在同预算下改善动态尾部，而其它质量不显著恶化 | 固定样本总量/相同方法换独立运动来源后，motion-rich holdout完全不改善 |

**比较基线同样重要**：静态UB1、同预算C24/UB2、不同校准对象R256/消费者MPC/OBS、原Dense resident、Affine54以及实现明确的RT-Lynx 2:4属于不同合同。静态与输入动态不是学术优劣门槛；若固定结构能够通过成熟校准/小训练恢复运动，直接采用就是有价值进展。

## 4. 真正有辨别力的最小实验次序（先利用现存资产）

**M0 源与度量身份复核（本轮已完成，RECORD）**：原 \`UB1_DYNAMIC_DEGREE_RESULTS.json\` 的四个SHA绑定视频、16fps/41帧/40对、原二值0/0、40对全数据；五个8-pair窗口比+低于Dense次数已经重算，仍只有2个**旧formal20 sources**。source882吃草动作差异比source80更能检验prompt motion。原剩余formal20与disjoint20 Dynamic Degree尚无正式完成记录，禁止在图谱中把40个pairs当40独立视频。

**M1 既有原视频的测量机制判别（条件有worker MP4时；不新生成模型）**：同source/seed SHA解码全81帧；原16fps人看整片并核对动作语义，事先定义subject/background/camera类别。保留官方Dynamic Degree的原binary与continuous诊断身份。选择confidence有效的主体ROI、跟踪轮廓/关键点，分出global camera motion/flow和subject-relative motion；记录trace方向、幅度、时刻、累计位移、人工ROI时刻。另报告sharpness+occlusion指标，排除“blur降低RAFT”的解释。**不能只按Dense含运动且UB1更弱的样例事后筛选形成虚假的泛化统计**；若按Dense运动分层，进入确认集之前预冻结定义且完整报告被筛掉的数量。

**M2 新独立来源的统计验证（需要按ACTIVE冻结/授权；不把旧20新20重称holdout）**：分层选明确物理主体动作、camera motion、静态场景与混合场景（含低纹理/高纹理），保留整source/seed独立性与原输入图/prompt SHA。每条视频至少subject/background/aesthetic/imaging/motion smoothness、原Dynamic Degree binary+continuous（有原RAFT时）、主体动作语义与物理追踪的多模态评价；按source bootstrap与paired effect报告减幅置信区间；对camera-only不能把稳定住宅画面直接判为失败。运动幅度应以Dense参考+目标动作共同解释，不鼓励无限放大。

**M3 精确native FFN方向对照（另经实际模型执行注册）**：用同一个source/seed、相同4-step/40block真实pre/post Q1、Q2、e5 gate，分别在student或dense prefix上对齐，对受控有限FFN组支持重建记录\(\Delta\)的空间/latent视频时间方向。比较同局部SSE、同真实MAC的支持选择，\(\|D_\tau e\|^2\)、signed\(\langle D_\tau Y_D,D_\tau e\rangle\)、真实Flow/latent边界与真实后缀方向；如果只能保存全局状态，不要把具体FFN group叫“运动功能”而需做局部合法插入/末端video受控因果。Q2 64-group共享尺度与真实BF16 path沿源计费。

**M4 已有成熟方法迁移（视M1–M3决定，不预设训练）**：先采用VideoJAM的appearance+motion监督直觉或Motive的motion-weighted calibration / 选择数据，或者OBS/2ndMatch/CTEC块间/跨step目标用于后训练；对照相同源/计算预算的原C_mean、原R256与纯affine/ID消费者。若需要LoRA或rank256重训，登记真实样本来源、steps、GPU/成本；若找到不需训练的稳健静态group改法，也完全成立。终端必须有整40层四步自由视频与实际FFN+generation wall，不能仅以motion loss改善宣布部署成功。

**M5 判定语句（预先注册的可能结果）**：
- 多个真正独立motion-rich来源同时呈主体轨迹/方向可见衰减且appearance仍近似→UB1存在特定测试分布的运动幅值保真短板；若只在旧两个来源→只成立当前历史观察。
- 相机/主体分解显示住宅主要是镜头变静止、或blur使RAFT偏低→修正“运动偏置”作用范围，不应把“更少运动”自动认作质量损失。
- 同预算更好的support/补偿恢复motion方向且整生成有效→发现高价值的动作预算/函数敏感性；若只恢复本地方向不恢复视频→优先再检验后缀。
- 强对手已有方法（VideoJAM/Motive/OBS/CTEC）直接有效→准确记作有价值迁移和验证，不要求形式上独特。

## 5. 原论文对照与新颖性审查（READ-LEVEL精确记录）

| 文献 | 已公开成熟机制 | 对FFNDS的实质可用性和限界 | 阅读深度 |
| --- | --- | --- | --- |
| [VideoJAM, ICML2025](https://proceedings.mlr.press/v267/chefer25a.html) | 将pixel/appearance主导与运动coherence目标区分；训练预测视频和运动、推理Inner-Guidance利用模型自身motion预测 | 跨模型运动监督的强先例；其输入/训练/额外head不同Wan FFN，因此不可声称“在本FFN低成本校准已成功” | 正式PMLR论文摘要/方法总览；未逐页审作者PDF |
| [Motive, ICML2026](https://proceedings.mlr.press/v306/wu26aq.html) · [作者技术介绍](https://research.nvidia.com/labs/sil/projects/MOTIVE/) | AllTracker motion magnitude作区域权重，在latent分辨率施加运动加权loss，再以梯度影响精选motion相关后训练source；报告VBench Motion Smoothness与Dynamic Degree共同改善 | 对“校准/训练数据是否以静态外观为主”提供直接改造原语；是训练数据归因，不是原Wan离线组选择的同一算子 | 正式PMLR摘要＋NVIDIA作者方法页面含loss公式；未逐页校PDF |
| [Dual-Expert Consistency Model, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lv_Dual-Expert_Consistency_Model_for_Efficient_and_High-Quality_Video_Generation_ICCV_2025_paper.html) | 在少步视频蒸馏中分开semantic/motion与detail expert，并加Temporal Coherence及GAN/Feature Matching目标 | 具体说明**少步生成外观与运动的训练目标不同**；其semantic/detail双专家**不是**Wan high/low噪声两expert，切勿把名称当作源码架构等价 | 会议原摘要 |
| [VBench, CVPR2024](https://github.com/Vchitect/VBench) | 区分Motion Smoothness、Dynamic Degree、Background Consistency和Temporal Flickering | 保持原version0.1.5二值与本仓库连续RAFT诊断分开；质量向量不应合并为一维 | 官方源码及FFNDS原程序、checkpoint SHA经既有RUN核验 |
| [RT-Lynx, ICML2026](https://proceedings.mlr.press/v306/cong26a.html) | 激活2:4、Norm+LoRA保留图像DiT质量，与静态weight压缩对比 | 保留全部权重长期容量≠每次都保住视频运动功能；适合作为独立C线物理比较器，未有Wan运动实测 | 见[本仓库一手正文审查](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md) |

**新颖性定位**：运动/外观的分解、motion-weighted loss和训练后动态校准思想已有论文先例；当前值得探索的是**原生Wan可执行FFN分组/数值/预算究竟怎样改变正常时空函数，是否能低代价恢复**。数学时间差算子与signed误差分解是经典工具，不应包装成新颖定理。没有任何“必须和论文不同”的研究准入要求。

## 6. Skill检查与历史证据治理

- **NOTICE**：两个source连续运动值大幅小于Dense、外观基本稳定；38/40与40/40跨窗口趋势加强同一视频内的现象连续性。
- **MAP_FLOW**：查原HiF4 Q1→W1→GELU→Q2→W2→gate、UB1固定支持与R256，真实scheduler四步与21 latent视频槽，原video81帧；绝不混合step和frame。
- **DERIVE**：冻结线性时间差与signed cross恒等式；同时从RAFT/模糊反例提出H-MEAS这一强对手。
- **EXPAND/REFORMULATE**：从“平均幅值组重要性”到“长期静态外观功能 vs 当前运动功能/真实边界”，保留I/F/C三种入口；允许成熟方法直接迁移。
- **DISCRIMINATE**：M1独立主体轨迹先于无限模型训练或试错；M2新来源/seed确认，M3真实后缀/分组机制，M4已有监督或执行动作。
- **RECORD/权限**：文件只附加新分析并更新导航。原UB1 20+20视频、RAFT四视频、E0其它候选原生P/N与所有RUN保持原身份；不从现存数据补造不存在的Dynamic Degree全套分数；所有模型/新CPU/GPU/video执行按[ACTIVE](../ACTIVE_RESEARCH.json)的精确已登记范围决策。

**短结论**：两条已暴露视频的连续RAFT显著衰减是重要的正常生成科学线索，但尚未证明所有真实物理运动都缩小或模型普遍无动态能力。更值得问的是：**原FFN保留外观与运动所需函数方向的预算/校准敏感性如何不同，且能否把这个区分变为确实可用的省算？**
