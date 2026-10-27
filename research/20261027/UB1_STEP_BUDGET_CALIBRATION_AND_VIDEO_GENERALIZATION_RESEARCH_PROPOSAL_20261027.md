# FFNDS · UB1步间计算预算、校准泛化与真实视频功能研究方案

**2026-10-27｜独立科研方案 v1｜状态：OPEN / DISCUSSION_ONLY（不含新模型执行授权）**  
**研究对象**：LightX2V Wan2.2 I2V A14B 四步蒸馏 D-Q/HiF4 FFN；UB1是现有的可靠工程与研究基线，而非默认的最终算法。

> **中心问题**：在少步视频生成中，为什么一个可以真实减少FFN运算、在原20来源接近Dense的按步固定预算模型，会对另一些来源产生集中的画质退化？哪些条件决定有限预算应该被分配到哪些step、block、组或新的计算单元？能否通过充分的校准、成熟方法迁移、合理的函数/轨迹目标或更好的系统实现，获得更稳健的质量—成本折中？

**研究原则**：复现与采用有效成熟方法是完全合理的研究策略。近期核实的TARDIS/LLMFolder、Model Preserving Compression、FLAP、BRECQ、RT-Lynx分别在部分线性、消费者重构、结构可恢复性、块间误差与真实N:M kernel上提供直接参照；DyDiT、DIFS、SlimDiff、OBS-Diff、2ndMatch和CTEC继续覆盖时间预算和跨步目标。它们都能启发UB1的改进，而不要求每种改法刻意不同。共同的逐篇一手方法与Wan适配边界集中在[文献审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)，本方案只保留实验决策。I/F/C并行，成功复现与系统改进也是真实成果。

## 1. 严格定义UB1：其“动态性”、计算与训练身份

Wan的每个去噪step调用40个Transformer block，双expert依次为high/high/low/low，共160个FFN位置。典型token数M=31,668、输入/输出通道D=5,120、隐藏通道H=13,824=216×64。真实计算边：norm2/AdaLN/e3e4→Smooth1/HiF4 Q1→原BF16 W1+b1→tanh-GELU→Smooth2/HiF4 Q2(64→8→4共享尺度)→原BF16 W2+b2→e5门控BF16残差→后续block/flow/scheduler。

UB1原配置：

| Step | Expert | 原64通道组保留数 | hidden宽度 | 原主GEMM工作量比例 |
| --- | --- | ---: | ---: | ---: |
| 0 | high | 54 | 3456 | 25% |
| 1 | high | 108 | 6912 | 50% |
| 2 | low | 108 | 6912 | 50% |
| 3 | low | 162 | 10368 | 75% |

四步均权、每步相同40层/M时，主W1+W2 MAC密度\((54+108+108+162)/(4×216)=0.5\)。rank-256辅助有2个BF16投影，额外线性MAC约原主FFN的\(256/13824=1.85185\%\)，主+辅助线性MAC理论合计约51.85185%（不含QDQ/gate/内存/启动）。原生运行记录已审计真实全部160处形状；不是仅后验mask不省W1的方案，更不是packed FP4 GEMM。

其组支持\(S_{t,l}\)和比例仅随**预知step、block**变化，对当次视频、token、seed的在线输入不变。术语为 **timestep-conditioned static structured FFN pruning with calibrated low-rank correction**：可以从时间维称条件化动态宽度，**不是input/content-adaptive动态稀疏**。

校准：继承C24的24个来源、每位置每来源256行；利用Dense保存post-Q2和原W2/gate，按每个原64组输出能量\(C_j=\mathbb E\|g\odot W_{2,G_j}a_{G_j}\|^2\)排序取组；拟合中心化非负标量gain \(\alpha\)、折叠bias；再以post-Q1输入拟合白化rank256 \(\lambda=1\) 的FFN残差线性支路。只为step0/3各40个位置重新拟合，step1/2沿用冻结C24，模型不进行新的大型端到端蒸馏训练。组选择发生在**离线**，不向线上提供免费post-Q2观测器。

冻结实现与协议：[实际UB1 policy](../../reports/worker_20261020/UB1_policy.json)、[事前注册](../../reports/worker_20261020/step_budget_ub1_registered_protocol.json)、[拟合脚本](../../tools/fit_step_budget.py)、[执行器](../../tools/step_budget_runtime.py)、[数学和查重审查](UB1_METHOD_AND_PRIOR_ART_REVIEW_20261027.md)。

## 2. 已知成功、负例及可靠统计单位

| 已完成比较 | 数值事实 | 科研含义 |
| --- | --- | --- |
| 原正式20来源（10 subject + 10 background） | aesthetic/imaging/motion相对下降0.87385%/1.34475%/0.007365%；原历史2%逐项门槛均通过 | 原开发来源上**质量接近Dense**的有效固定配置 |
| 另20不相交来源 | aesthetic/imaging/motion相对下降1.93372%/**3.16597%**/−0.09419%；imaging未过原2% | 有真实跨来源泛化缺口，不能叫全面无损 |
| 新20逐来源 | 10/20来源Imaging相对下降>2%；高损失样本包括鸟群、雪夜公交与暴风雪车辆 | 尾部质量可能由输入功能异质性驱动；具体因果尚OPEN |
| UB1正式160处FFN计时 | Dense10,177.569ms→UB1 6,582.689ms，**1.546111×**，各三次重复 | 真实FFN效率成功，并非仅理论FLOPs |
| R17一来源×5技术重复全生成 | Dense50.847063s→UB1 47.314734s，**1.075683×**；FFN(含modulation)9.915117→6.468229s | 全生成速度收益远小于FFN边界收益；该时间比不代表20来源普适结果 |
| 相同预算C24、UB1、UB2 | C24四步(50,50,50,50)%、UB1(25,50,50,75)%、UB2(25,25,50,100)%；UB2两批imaging下降2.218%/4.652% | 预算转移不是单调改善，不能由末步局部误差归零推出最终视频更好 |
| OP3学生状态拟合 | 匹配的12来源新/旧状态拟合，原生SSE有微小改善；8/8项均未超过冻结24来源UB1 | 状态相关性存在，但“只更换拟合状态”未能证明解决泛化；24源与12源不构成等样本的因果对照 |
| 旧R12/R14/C1动作空间 | 某些冻结动作族的事后oracle依然差于UB1 | 关闭其限定动作/损失/预算；不否定所有动态方法、其它函数族或训练后新支持 |

**采样单位**：24个校准source，6,144行/位置是高度相关的token样本，不是6,144个独立场景；两批各20个独立来源，三个VBench维度不是60条独立视频。原20反复参与研究开发，新增20在此刻也已被观测，后续调参后的最终确认需要另一批**此前完全隔离**的来源/seed。两组20来源确定的2%门槛结果不同，不能据此就推出总体均值必然>2%的显著性论断。完整视频MP4有历史路径与SHA元数据，但不在公开git文件树。

证据：[联合工程RUN](../../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json)、[原20 CSV](../../reports/worker_20261020/formal20_UB1_paired_sources.csv)、[新20 CSV](../../reports/worker_20261020/disjoint20_UB1_paired_sources.csv)、[补充20审查](../../docs/29_冻结候选的来源分离补充验证.md)、[R17全系统时延](../20261023/R17E0_system_timing_prefix.json)、[UB2真实失败和完整轨迹](../../docs/38_固定预算下恢复末步完整FFN.md)、[OP3比较](../../docs/46_真实状态重拟合与导入延迟修正.md)。

**新增正式研究命题——相对Dense的运动幅度衰减**：[完整独立机制审查](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)、[原版RAFT+SHA](UB1_DYNAMIC_DEGREE_RESULTS.md)、[逐帧复核](UB1_INDEPENDENT_VISUAL_REVIEW.md)。两个已暴露formal20来源UB1/Dense top5%连续光流38.13%/9.19%（衰减61.87%/90.81%）；犀牛38/40对、住宅40/40对比Dense低，五个不重叠8-pair窗口均低。原版二值Dynamic Degree四视频均0；原正式20仅有的Motion Smoothness均值0.983923/0.983850不能揭示这一变化。这是**两视频内的持续测量现象而不是普遍动态功能失去的证据**，且视觉质量/模糊变化影响RAFT追踪可观测性；真实主体位移、动作语义、相机运动与泛化还需独立确认。

## 3. 研究机制：五个能够分别修订的假说

### H-Data：来源覆盖和有效独立样本不足

同样24×256行有不同的场景、seed、相机运动和背景组合。若主要问题是输入覆盖，增加独立来源/让覆盖更均衡，在固定支持规则、alpha/R256/λ下应该改善新来源尾部退化；若仅增加相同行的重复token而收益微小，则更加支持独立来源的作用。

**竞争解释**：即使来源增加，固定whole64支持仍可能删去仅在少数source/seed/step重要的功能；这种跨输入“长期能力”与当前token激活大小不相同。RT-Lynx在图像DiT对权重2:4与输入激活2:4的质量比较提供值得直接转移的对照，但不能把权重一律视为不可稀疏。**判别实验**可固定总行数同时改变独立来源数和每来源行数，并独立比较固定权重/whole64支持与按本次输入稀疏的质量尾部，避免把不同计算合同混写成“同一种动态方法”。

### H-Support：支持的可恢复性、消费者重构与合法量化单位

UB1的C_mean在保存post-Q2量化输出上计算每个64组\(c_g=g\odot W_{2,G_g}a_{G_g}\)的平均输出能量。固定某个删除集合R时，原输出差异的数学式为
\[
\|\sum_{g\in R}c_g\|^2=
\sum_{g\in R}\|c_g\|^2+2\sum_{g<h\in R}\langle c_g,c_h\rangle.
\]
这证明单组能量排序**没有直接优化组间的联合抵消/增强**，但不能由此断言C_mean就是新20成像下降的唯一原因。另一个直接问题是原组的**可恢复性**：某组单独能量大，若它能够由保留组近似且通过重新拟合输出消费者恢复，删除它未必昂贵。

**第一批可直接采用的成熟算法**：
- [Model Preserving Compression 2022](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)：合法量化激活组的ID选择及将重构矩阵折入消费者（在Wan上不是无条件可以直接搬运任意单通道）；
- FLAP 2024：以波动/可恢复性及均值bias恢复替代单纯幅值；
- SlimDiff与OBS-Diff：协方差、二阶/分组序贯重构和时间预算；
- 若这些仍不足，也可把更完整的非线性函数表示作为独立候选，参照并行[Affine54方案](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)。

核心对照应**固定step宽度、原64组单位、独立校准来源、V_S/或R256的参数预算与BF16实际算子**；其次再研究多因素联合改进。E0五组的后验联合最佳不同于独立TopK，只是一项重要线索，不能宣称它已识别全216组UB1的最优支持。

**区分性现象**：若仅改变支持/消费者就改善持出成像，说明原C_mean或旧补偿对目标不充分；若源扩展后改善而方法变化无效，来源覆盖可能主导；若局部重构改善但视频依旧差，则需要后缀/轨迹层证据。以上均允许直接使用已知成熟实现。


### H-Temporal：时间步与层级计算职责依赖真实状态

UB1/UB2已表明四步固定总预算的作用不单调。正常模型的真实sigma经shift5约为[1,0.9375,0.83333,0.625]，理想步进绝对系数为[0.0625,0.104167,0.208333,0.625]；末步系数较大不等于它对应的FFN组更重要。

对相同候选/输入，本研究可以比较固定step预算、block内异质预算、每expert不同保留作用，以及基于下游真实有限映射的时间敏感性。**从其他E0候选观察到的P/N跨seed符号变化**只能作为启发，不能登记为UB1自身已测。基本比较对象为\(\mathcal L_{\rm local}\)、实际nextnorm/flow响应、同前缀下一latent及完整视频行为；可直接复用OBS-Diff的时间加权或CTEC的rollout目标，不需刻意发明另一个权重公式。

### H-Calibration：低秩补偿和数值训练路径限制跨来源泛化

UB1的R256是post-Q1→两次BF16投影，解决部分遗漏输出；它与真正输入条件非线性恢复、训练后重组FFN承担者是不同函数类别。旧拟合使用保存Dense post-Q2子集，推理执行原生缩窄W1/GELU/Q2，需检验是否存在数值路径差异。OP3比旧24源UB1差，不能单独判定rank不足还是校准来源数造成。

竞争方案：**不变结构**先比较同一状态、同源同预算下的native路径；**不同目标**可直接采用现有groupwise sequential calibration、2ndMatch/Jacobian敏感性、CTEC/rollout或轻量LoRA；**不同函数类**则允许rank、非线性组、block参数和训练后结构改变，但要实际计入成本。Q1/Q2/写回需要沿真实源码比对。

### H-Motion：外观保留与运动衰减为何可能并存？

UB1按step静态组选择C_mean、R256恢复局部输出；它没有逐token动态mask切换，所以[A5+HWG](A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)的阈值抖动不是UB1的直接解释。现有两条已暴露视频仍呈相对Dense持续低RAFT动态幅度；[完整假说、数据与最小判别](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)将这一观察升格为正式待检验问题。

- **H-MEASURE**：UB1纹理、模糊、背景/编码差异可能降低RAFT在固定真实位移下的追踪可观测性。先在原81帧、16fps原视频上比较ROI关键点/轮廓位移、相机补偿、遮挡置信和跟踪器一致性；不将top5 RAFT直接命名真实主体速度。
- **H-CAP**：传统平均gated W2能量组选择与rank256残差保留外观，却可能遗漏仅在罕见动作/latent视频时间变化上重要的有符号联合函数。固定计算预算比较同source组选择及MPC/FLAP/OBS成熟消费者重构，必要时做合法同prefix原生因果插入。
- **H-TRAJ**：下游attention、flow、四步去噪和VAE可能改变运动方向。其它E0候选的P/N finite轨迹数据只支持“存在这种可能”，**不是UB1本人运动衰减已定位**。用same-input FFN→nextnorm→整步flow→latent→video逐边检验。
- **H-DATA/SEM**：24独立校准来源×256行每source的覆盖、传统局部输出loss、不同prompt要求都可能影响结果。犀牛grazing有动作语义；住宅静态prompt没有镜头运动要求。固定总行数下分层motion-rich/static/camera来源，报告动作是否“符合prompt”，而不是单纯要求候选运动越大越好。

**新数学对象**：对已固定来源/图边的**线性latent视频时间差分**\(D_\tau\)（21 latent时间槽，不是4 denoising steps或81 RGB frames），若\(Y_S=Y_D+e\)，
\[
\|D_\tau Y_S\|^2-\|D_\tau Y_D\|^2
=2\langle D_\tau Y_D,D_\tau e\rangle+\|D_\tau e\|^2.
\]
这一通用恒等式分出正常运动方向与扰动方向的signed作用，不能代替非线性RAFT/对象运动指标或冒充新颖定理。原FFN group不必有专属“运动语义”；观测的函数方向与物理可省算单位是不同对象。

**论文成熟入口**：VideoJAM(ICML2025)联合外观/运动目标及Inner-Guidance；Motive(ICML2026)利用运动区域加权梯度选择微调来源；Dual-Expert Consistency Model(ICCV2025)将semantic/motion与detail训练分工。它们是可直接迁移或比较的先例；**DCM semantic/detail专家与Wan按噪声等级high/low专家不是同一个计算结构**。未来若恢复动作保真，需同来源相同GPU成本和质量向量、真实整40层四步视频确认，而不先验要求一定采用训练或动态路由。

### H-System：质量改善与真实系统收益是不同的资源坐标

R17中的resident dense FFN含modulation约占完整生成19.50%。若非FFN部分固定，条件Amdahl上界约\(1/(1-0.195)=1.242\)，UB1实际完整生成1.0757×；这个上界**只针对R17的特定resident强基线、硬件和串行成本假设**。真实QDQ、启停、weight residency、额外R256、kernel形状与pack内存成本都可改变比较。并行研究真正packed低位dense或核融合，完全有可能在更强系统基线上验证更好的执行策略；不把输入动态作为必须实现的功能。

### 横跨支持、预算和视频行为：历史A5+HWG提供的独立观察

[历史A5+HWG时空函数审查](A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)表明：逐token动态hidden mask下主体大体结构可维持，但背景照明/纹理可能闪烁，小目标细节可能模糊；其证据身份是旧视频视觉报告，而非已经得到原生mask切换到视频ROI的因果链。UB1按step固定支持，**没有同一类型的逐token开关**，但后续[UB1连续运动补评](UB1_DYNAMIC_DEGREE_RESULTS.md)仍发现两来源动作幅度显著不同；所以视频质量问题同时有**固定支持容量**与**时空协调函数**两条可研究解释。

这为H-Data、H-Support与H-Temporal增加一个共同评价坐标：对同一Dense配对逐视频比较(i)大体结构/真实动作；(ii)小物体/细纹理保留；(iii)相机/运动对齐后的背景亮度与纹理稳定。先用已有[完整视频指标合同](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)能正确计算的proxy；运动ROI、语义小目标/异常因果未执行即记OPEN/NOT_RUN。**不得将A5旧视觉描述直接记作UB1新来源的闪烁频率，也不从低运动误差推出动态路由应优于静态UB1。**

**新判别**：若固定支持始终舍弃低幅值但持续重要的小目标函数，增加mask时间平滑不会解决UB1问题；若一个新输入动态方案发生阈值支持切换，则其mask波动和下游时序作用另需匹配测量。S-SWITCH/H-PERSIST可相互区分，但都允许直接采用已成熟的算法进行验证。


## 4. 成熟方法的匹配使用与论文查重

**详细论文调查只保留一份**：[部分线性FFN、结构重构、跨步纠错与稀疏kernel共用审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)。以下对应UB1的**研究动作**而不是“创新性必须不同”的限制：

| 成熟工作 | UB1研究可直接借鉴 | 公平比较条件 |
| --- | --- | --- |
| **DyDiT / DIFS** | Timestep-wise可训练宽度、少步结构优化与输出rollout | 固定Wan模型/step、记录额外训练、整模型FLOPs与真实wall |
| **SlimDiff / OBS-Diff** | covariance/二阶式真实支持选择、序贯组补偿、时间/层预算 | 原whole64 Q2尺度、同校准来源/组数/R256或消费者预算 |
| **Model Preserving Compression (NeurIPS2022)** | 激活子集ID及下游消费者重构 | 必须检验真实quantized整组表示，别把逐列ID误认原HiF4可直接W1省算 |
| **FLAP (AAAI2024)** | 波动/可恢复性组分数、分层预算、baseline bias compensation | 常量bias方案表达能力低于UB1的R256，但评分/泛化原理可以很有用 |
| **BRECQ (ICLR2021)** | 不仅逐FFN输出，而按可管理的相邻block范围做二阶/分组重构 | 原目标是CNN post-training quantization；对于Wan是合理迁移而非同模型结论 |
| **CTEC / 2ndMatch / TCEC** | 跨步错误纠正、扰动敏感性/有限响应、低成本补偿 | 完整四步原scheduler、同source/seed、teacher仅离线可见 |
| **TARDIS / LLMFolder** | 若原子组缩减能力受限，可探索conditional affine+必要真实非线性等竞争函数类 | TARDIS在线2-bit预测器和原FFN修正付费；LLMFolder已确认EuroSys发表但公式尚未逐节查阅 |
| **RT-Lynx ICML2026** | 测试保持稠密权重完整能力、仅对本次GEMM输入2:4稀疏与norm+LoRA恢复；W1输入和W2输入分别定义 | 与UB1直接删whole64组是不同动作；原tile全满不阻止SpTC 2:4，作者图像DiT/H20数据不等于Wan；线上内核和训练均计费 |
| **TAPE (2026)** | 视频时序预算与内容运动平滑 | 原重点token处理，不能将其指标横向换算UB1的hidden 64组 |
| **VideoJAM (ICML2025) / Motive (ICML2026)** | 外观与运动联合监督、motion-weighted来源选择与动作区域加权 | 是成熟训练/数据方法，不代表已经验证UB1 FFN恢复或无开销部署 |
| **Dual-Expert Consistency Model (ICCV2025)** | 少步蒸馏中semantic/motion与detail分工及Temporal Coherence | 该semantic/detail双专家不等同于Wan high/low噪声expert |

**学习最优先例**与**准确报告可能的新机制**并不冲突。UB1原静态4步宽度不是时间条件计算的首创；Model Preserving Compression意味着“选择部分通道并重构消费者”也非新思想。若直接采用这些做到了更好的Wan视频质量/成本，可以如实作为有价值的复现和适配成果。

## 5. 具体工作包：先选择有分辨力的对照，再决定计算范围

### WP0 · 冻结基线与原始证据复核（当前允许，SRC/RECORD）

整理UB1步骤/参数/组SHA、原校准24的来源身份、原20及新20视频/分数、Q2及BF16执行链、R17计时协议、UB2及OP3冻结负结果。明确完整张量可读/仅有哈希或JSON：**不能依据历史路径存在推断当前能读取视频或hidden**。保证新的研究不会将已用过的正式20重新称为真正全新持出集。产物：协议对照、独立来源/种子账本、算子成本表。

### WP1 · 最小数学与文献对照（当前允许，ALG/LIT/SRC）

分别复核C_mean的有符号交叉缺口、基于同一总预算的四步组数约束、原生Q2与R256的实际成本。检查SlimDiff的协方差结构、OBS-Diff的序贯组校准与DyDiT/DIFS的step训练对象，并尝试构造1–2个**改变研究判断**的明确反例：如两个相同组能量的不同相干结构产生不同真实输出，或UB1/UB2局部末步误差改善并不降低前缀状态误差。数学构造可沿用现有定理，不必须“首次提出”。

**运动专项**：本轮已只读核对原RAFT 40对×4视频、五窗口比以及38/40、40/40对的持续性；未补测其余formal20/另一20，也未获得新主观或物体追踪结论。已暴露视频用作机制发现，新source用于事前冻结motion/static/camera类别的确认。具体M0–M5见[独立机制审查](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)。

### WP2 · 来源级泛化解释与已有数据（当前允许，已有文件只读）

从现有逐来源CSV对每条视频记录正/负指标变化、tail、source组、静态step policy与已存轨迹是否可匹配；不使用评分高低来偷偷选择未来确认样本。来源级统计和token级校准信息分开。比较现有R12/R14/OP3作用范围和自由视频异常，登记“哪个现象有原生来源、哪个只是跨候选假说”。现存资产不足时保留OPEN。

### WP3 · 独立来源校准充分性（**未来另经授权的定点实验**）

建议一个可先缩小的**双因素对照**，非既定必须网格：以固定总6144行/位置，分别选24×256、48×128、96×64个**独立校准source×rows**，完全相同的C_mean/gain/bias/R256/λ、budget和真实原Q1/Q2；另在固定来源数增加行数以估计token采样饱和。条件允许时再以来源类别均衡、不同seed、相机运动/背景分层交叉检验。

**可区分的预测**：若覆盖是关键，固定总行数下增加来源应改善新确认集尾部质量；若容量/目标主导，来源增多后仍显著损失，且不同表示/目标在相同来源数下有收益。训练和质量分割按**source及seed**隔离，不将同视频token随机拆分成“独立验证”。方案先按可获取的独立数据量缩放；24/48/96只是设计示例，不声称目前已有足够新来源。

### WP4 · 同预算组选择与时间预算（未来条件执行）

**两轴分离后再可选联合**：第一轴固定原UB1宽度(54,108,108,162)，对比C_mean、FLAP式波动/可恢复性、MPC-ID消费者折叠、SlimDiff/OBS-Diff二阶组选择，以及各自匹配成本的R256或输出重构；第二轴固定四步原主预算 \(\sum_t k_t=432\) 且每step 40block相同M/D，比较C24、UB1、UB2以及少量事前注册、真正具有不同机制预测的step/block配置。**不做无解释的大型离散预算扫描，也不排斥成熟方法原样迁移。**

直接对“训练后支持”或“在线动态支持”设立独立对照：若采用TARDIS热区与2-bit proxy，先核实原64组Q2尺度、观测器和tile；若采用RT-Lynx的2:4 SpTC，核实其真实activation数据结构和LoRA成本。过去R12/R14/C1事后oracle失败只限制各自旧动作族；不能关闭其它合法动态动作或函数重组。严格区分**后验最优支持的估计**与推理前实际可获得的信息。


**RT-Lynx激活/权重稀疏对照（独立C/I研究，前置只读检查）**：不要把UB1的54/108/108/162原64组缩窄直接与2:4激活稀疏称为“相同50%计算预算”：前者**W1与W2主形状**同步降低，后者在Q1后压缩W1输入可仅减少W1部分乘加；Q2后压缩W2输入仅减少W2乘加且保留原Q2有效尺度，双位置组合才可能理论上令两层乘法各减半。先固定W1/Q1、W2/Q2两种2:4输入合同，对比naive weight2:4、activation2:4、norm、LoRA与敏感位置Dense保留，记录每步每block/seed的相干功能、尾部和真实A100 Sparse Tensor Core路径。[RT-Lynx专题深读](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)。保留原始UB1及其24来源校准身份，这个新候选属于**新动作族**，旧R12/R14 oracle不约束它；未登记新原生RUN、视频或性能。

### WP5 · 轨迹及低成本补偿（**未来可选授权**）

通过OP2/OP3已有资产先检验真实UB1学生状态与旧teacher输出是否确实可配对。若有条件开展新原生观察，在**同一source/seed、同一student或dense prefix**上比较：仅组变化、仅alpha/b2、仅R256、同预算新组选择、跨步损失/2ndMatch/CTEC类成熟训练目标。检验被修改的位置在原生写回、后缀nextnorm、整步flow、下一latent及视频行为上的作用。对于trajectory误差使用实际有限映射\(T_D,T_S\)而不默认HiF4/Jacobian光滑；2ndMatch的敏感性loss可作为成熟好方法直接实现。此阶段旨在区分机制，**小规模局部成功不是视频级成果**。

### WP6 · 原生视频与系统验证（**有实际候选且获授权后**）

全40层×4步、同输入/seed配对、独立从未用于调参的确认来源、真实81帧自由生成；完整逐视频Aesthetic/Imaging/Motion及Dynamic Degree、Subject/Background一致性、Temporal Flicker、必要ROI/相机运动。逐轴报告相对Dense变化、源级tail和组别；测试模型指标版本、正反方向、评价粒度固定。明确原历史≤2%门槛是UB1原formal20内部验收条件，研究发现并不以先过门槛为必要资格；**若对外声称质量保真**，则应先预注册明确可审查的多指标标准并实际通过。

性能以相同硬件/运行次数/分辨率/权重驻留的Dense D-Q、原UB1、相同预算C24/UB2及可获得的成熟论文实现比较；汇报完整FFN、DiT主干及整段生成wall、VRAM、冷/热启动、bank、校准与可选训练代价。检查后验组稀疏是否真的减少W1、Q2/W2、合法M/K/tile执行；强Dense resident与native packed低位Dense可作为C线独立竞争基线。不能把50%主GEMM直接写成50%整网FLOPs或2×整体加速。

**新增motion必要分辨力**：逐视频原版Dynamic Degree binary + **标注为自定义连续诊断**的40-pair top5 RAFT均值/五窗口，主体/相机/背景运动、动作语义、对象轨迹、ROI纹理可追踪性；缺的项目标NOT_RUN。原Motion Smoothness不能替代动态程度，结构稳定不能直接验证运动正确。标准详见[指标合同](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)与[运动研究](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)。

## 6. 决策逻辑与可交付成果

| 已观察的判断分歧 | 最有价值的下一动作 | 对方案的作用 |
| --- | --- | --- |
| 扩大独立来源显著改善尾部质量 | 保留原UB1功能/预算，优先形成稳健校准实践 | 直接有效，无需强行提出新路由 |
| 更多来源没有明显改善，同预算组选择改善 | 研究联合/协方差/OBS或其它成熟组评分与真正算子闭包 | 选择函数目标胜过纯数据规模解释 |
| 组选择效果有限，时序预算或轨迹目标有效 | 采用DyDiT/DIFS/CTEC的时间和训练机制，并衡量新开销 | 从静态UB1演化成可训练step结构 |
| 旧固定与动态动作都不足 | 放宽原动作表示，进入I/C新的函数/量化/kernel或轻训练 | 旧oracle约束不限制新表示 |
| 质量改善但系统加速有限 | 查QDQ、R256、驻留及非FFN瓶颈，可能沿用最强Dense系统方案 | 在用户真正关心的完整wall口径上决策 |
| 出现可证明的函数/传播规律但暂未加速 | 独立保存数学/机制结果与可证伪条件 | 不因没有即时部署就废弃科学成果 |

**方法失败是具体候选在指定来源/成本/质量协议下的事实**，不作为否决所有动态预算、二阶校准或训练后方法的理由。文献借鉴、精确数学分解、复现、方法组合与负结果都能帮助发展研究。

## 7. Skill复审与真实研究边界

- **mechanism-research**：保留真实W1/GELU/HiF4 Q2/W2/e5、时间step、后缀及成本语义；为五个解释各给出不同预测和最便宜证据。
- **research-expansion**：从校准来源/组间耦合/时间预算/训练函数/执行kernel多个入口探索，允许直接用已有最佳方法，不固定算法族。
- **problem-reformulation**：将“按step配额是否最佳”改为“固定物理预算下哪些可支付的信息、函数与真实视频质量相对应”，允许数学或系统独立成果。
- **web-research-discovery**：从DyDiT/DIFS/SlimDiff/OBS-Diff/2ndMatch/CTEC/TCEC/TAPE的原论文机制出发，具体比较训练身份、time/width支持、算子及计时口径，检索帮助借鉴和澄清，不设强制创新差异门槛。
- **FFNDS领域Skill**：I/F/C可平行推进，UB1不是全模型输入动态；NATIVE/SRC/ALG/LIT/OPEN清楚区分，研究与部署主张按各自证据程度判断。

**与并行Affine54方案的关系**：[Affine54/量化一致函数与轨迹研究](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)关注**函数表示**（仿射+必要非线性）和部署数值；本UB1方案以**已有原数值子组支持+rank补偿**研究校准、预算/step和泛化。两者共享原Wan图、真实video/speed标准、论文工具与信息/传播理论，可互相提供对照，却不要求彼此先完成或采用对方函数结构。

**执行范围**：本方案自身保持OPEN/DISCUSSION_ONLY。用户已在当前会话恢复离线I/F/C持续推进，当前状态及逐项冻结授权以[ACTIVE_RESEARCH](../ACTIVE_RESEARCH.json)为准，最多同时四张空卡；原暂停保留历史身份。WP3–6的具体来源、代码、资产和预算仍须分别冻结登记，本文不派发这些实验。已完成一次性交接已删除，稳定入口为[当前上下文](../agent_protocol/CURRENT_CONTEXT_PACKET.md)；原始RUN、CI及质量验收保持各自身份。

## 8. 精简后的参考入口

[共用部分线性FFN文献审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md) · [RT-Lynx激活稀疏、权重能力及2:4原生Wan合同](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md) · 
[UB1源码级方法审查](UB1_METHOD_AND_PRIOR_ART_REVIEW_20261027.md) · [质量指标与FFN速度RUN](../../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json) · [新20逐来源CSV](../../reports/worker_20261020/disjoint20_UB1_paired_sources.csv) · [C24/UB1/UB2固定预算对照](../../docs/38_固定预算下恢复末步完整FFN.md) · [OP3状态重拟合](../../docs/46_真实状态重拟合与导入延迟修正.md) · [此前Skill复审的研究解释](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md) · [领域研究目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。
