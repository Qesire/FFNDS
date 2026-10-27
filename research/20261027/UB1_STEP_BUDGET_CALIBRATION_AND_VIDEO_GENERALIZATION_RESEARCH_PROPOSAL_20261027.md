# FFNDS · UB1步间计算预算、校准泛化与真实视频功能研究方案

**2026-10-27｜独立科研方案 v1｜状态：OPEN / DISCUSSION_ONLY（不含新模型执行授权）**  
**研究对象**：LightX2V Wan2.2 I2V A14B 四步蒸馏 D-Q/HiF4 FFN；UB1是现有的可靠工程与研究基线，而非默认的最终算法。

> **中心问题**：在少步视频生成中，为什么一个可以真实减少FFN运算、在原20来源接近Dense的按步固定预算模型，会对另一些来源产生集中的画质退化？哪些条件决定有限预算应该被分配到哪些step、block、组或新的计算单元？能否通过充分的校准、成熟方法迁移、合理的函数/轨迹目标或更好的系统实现，获得更稳健的质量—成本折中？

**AutoResearcher原则**：从成功和反例提取机制，重新审查实际计算对象、最强已有方法、不同解释及可区别证据；允许直接复现、采用、组合DyDiT、DIFS、SlimDiff、OBS-Diff、2ndMatch、CTEC等方法，**不为了刻意显得不同而改变有效方法**。科学解释、稳健复现、系统改进都可以是重要成果；新的原创性结论由实际发现而定，绝不是研究启动门槛。研究始终保留I（信息与表示）/F（函数与动力学）/C（计算与执行）并行入口。

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

**当前会话的局部跟进**：[Terra逐帧复核](UB1_INDEPENDENT_VISUAL_REVIEW.md)与[原版运动补评](UB1_DYNAMIC_DEGREE_RESULTS.md)实际覆盖两个已暴露formal20来源、四原视频，光流均值约为dense的38.1%/9.19%，而二值dynamic degree四条均0。它们为外观/运动观察提供补充，保留原20/新20聚合以及WP3独立来源设计的完整范围。

## 3. 研究机制：五个能够分别修订的假说

### H-Data：来源覆盖和有效独立样本不足

同样24×256行有不同的场景、seed、相机运动和背景组合。若主要问题是输入覆盖，增加独立来源/让覆盖更均衡，在固定支持规则、alpha/R256/λ下应该改善新来源尾部退化；若仅增加相同行的重复token而收益微小，则更加支持独立来源的作用。

**反向解释**：更多样本也无法让静态whole64支持表达少数关键联合功能；原始训练目标可能与视频行为不一致。**判别实验**应同时改变来源数和每来源行数，控制总行数，不把多个改变混写成“增大校准”。

### H-Support：C_mean独立组能量忽略组间协同

给定真实合法post-Q2输出组贡献\(c_j=g\odot W_{2,G_j}a_{G_j}\)，删去集合R的输出误差包含
\[
\|\sum_{j\in R}c_j\|^2
=\sum_{j\in R}\|c_j\|^2+2\sum_{i<j\in R}\langle c_i,c_j\rangle.
\]
C_mean主要利用对角贡献，未直接最小化有符号交叉项。旧E0五组实验已经出现独立能量排序与联合最优选择不一致，但它属于**后续其它模型作用域的小组候选**，不是UB1全216组最优解证据。

**正面方向**：允许直接尝试成熟的协方差/OBS/SlimDiff二阶组校准，也允许廉价的有限联合修正和原Q2尺度组闭包；比较保留原W1+W2可执行组数与整体视频质量，而不是选择器分数本身。**关键反例**：统计上最优的后验组集合可能在真实部署Q2舍入或未来输入上不稳定。

### H-Temporal：时间步与层级计算职责依赖真实状态

UB1/UB2已表明四步固定总预算的作用不单调。正常模型的真实sigma经shift5约为[1,0.9375,0.83333,0.625]，理想步进绝对系数为[0.0625,0.104167,0.208333,0.625]；末步系数较大不等于它对应的FFN组更重要。

对相同候选/输入，本研究可以比较固定step预算、block内异质预算、每expert不同保留作用，以及基于下游真实有限映射的时间敏感性。**从其他E0候选观察到的P/N跨seed符号变化**只能作为启发，不能登记为UB1自身已测。基本比较对象为\(\mathcal L_{\rm local}\)、实际nextnorm/flow响应、同前缀下一latent及完整视频行为；可直接复用OBS-Diff的时间加权或CTEC的rollout目标，不需刻意发明另一个权重公式。

### H-Calibration：低秩补偿和数值训练路径限制跨来源泛化

UB1的R256是post-Q1→两次BF16投影，解决部分遗漏输出；它与真正输入条件非线性恢复、训练后重组FFN承担者是不同函数类别。旧拟合使用保存Dense post-Q2子集，推理执行原生缩窄W1/GELU/Q2，需检验是否存在数值路径差异。OP3比旧24源UB1差，不能单独判定rank不足还是校准来源数造成。

竞争方案：**不变结构**先比较同一状态、同源同预算下的native路径；**不同目标**可直接采用现有groupwise sequential calibration、2ndMatch/Jacobian敏感性、CTEC/rollout或轻量LoRA；**不同函数类**则允许rank、非线性组、block参数和训练后结构改变，但要实际计入成本。Q1/Q2/写回需要沿真实源码比对。

### H-System：质量改善与真实系统收益是不同的资源坐标

R17中的resident dense FFN含modulation约占完整生成19.50%。若非FFN部分固定，条件Amdahl上界约\(1/(1-0.195)=1.242\)，UB1实际完整生成1.0757×；这个上界**只针对R17的特定resident强基线、硬件和串行成本假设**。真实QDQ、启停、weight residency、额外R256、kernel形状与pack内存成本都可改变比较。并行研究真正packed低位dense或核融合，完全有可能在更强系统基线上验证更好的执行策略；不把输入动态作为必须实现的功能。

## 4. 从原论文吸取成熟方法，允许直接复现与采用

| 最相关工作 | 已公开方法与真正迁移对象 | 与UB1比较时要保持的条件 |
| --- | --- | --- |
| [DyDiT ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html) / [DyDiT++](https://pubmed.ncbi.nlm.nih.gov/41538341/) | Timestep-wise dynamic width、spatial token、少量追加训练 | UB1属固定按步宽度；可直接借鉴其step动态训练，无需人为创造不同规则 |
| [DIFS 2026](https://arxiv.org/abs/2607.06631) | Wan少步结构化稀疏与联合蒸馏、Progressive Training、Output Rollout | 可以作为含训练路线；将训练总成本、同任务视频行为与推理wall分别测，不能拿不同设备1.2×横排 |
| [SlimDiff 2025](https://arxiv.org/abs/2509.21498) | activation-covariance与谱结构、跨步/模块稀疏分配、gradient-free | 可优先复现其covariance group思想，区分约500个样本与UB1的24来源×token行 |
| [OBS-Diff ICLR2026](https://arxiv.org/abs/2510.06751) | 时间步加权Hessian、groupwise sequential pruning和补偿 | 能直接替代C_mean或作为组校准比较；必须对齐原HiF4闭包和组预算 |
| [2ndMatch CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html) | 带\(J^\top J\)扰动响应的后训练 | 比较原值匹配vs方向/敏感性匹配，QDQ数值梯度和参数训练预算单独测 |
| [CTEC AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34039) | 受扰状态下的跨步误差纠正与teacher学生蒸馏 | teacher下一状态只在离线可见，不能当实时UB1免费标签 |
| [TCEC ICML2026](https://proceedings.mlr.press/v306/liu26fv.html) | 有成本的时间误差补偿和拟合 | 数学递推由原Wan scheduler校验，不无条件接受预印本可能有问题的展开式 |
| [TAPE 2026](https://arxiv.org/abs/2605.17837) | 视频token随扩散阶段的预算配置 | 早少算晚多算思想已有，UB1保留的是hidden FFN whole64而非token数 |

查重结论不是“禁止采用已发表算法”，而是厘清技术的训练身份、可见信息和成本。最实用的成果可能是**发现某篇成熟方法在固定Wan四步/原生量化下确实显著改善了UB1**；这种迁移、成功和失败都可形成可复用知识。研究贡献和新颖性在结果充分时再界定。

## 5. 具体工作包：先选择有分辨力的对照，再决定计算范围

### WP0 · 冻结基线与原始证据复核（当前允许，SRC/RECORD）

整理UB1步骤/参数/组SHA、原校准24的来源身份、原20及新20视频/分数、Q2及BF16执行链、R17计时协议、UB2及OP3冻结负结果。明确完整张量可读/仅有哈希或JSON：**不能依据历史路径存在推断当前能读取视频或hidden**。保证新的研究不会将已用过的正式20重新称为真正全新持出集。产物：协议对照、独立来源/种子账本、算子成本表。

### WP1 · 最小数学与文献对照（当前允许，ALG/LIT/SRC）

分别复核C_mean的有符号交叉缺口、基于同一总预算的四步组数约束、原生Q2与R256的实际成本。检查SlimDiff的协方差结构、OBS-Diff的序贯组校准与DyDiT/DIFS的step训练对象，并尝试构造1–2个**改变研究判断**的明确反例：如两个相同组能量的不同相干结构产生不同真实输出，或UB1/UB2局部末步误差改善并不降低前缀状态误差。数学构造可沿用现有定理，不必须“首次提出”。

### WP2 · 来源级泛化解释与已有数据（当前允许，已有文件只读）

从现有逐来源CSV对每条视频记录正/负指标变化、tail、source组、静态step policy与已存轨迹是否可匹配；不使用评分高低来偷偷选择未来确认样本。来源级统计和token级校准信息分开。比较现有R12/R14/OP3作用范围和自由视频异常，登记“哪个现象有原生来源、哪个只是跨候选假说”。现存资产不足时保留OPEN。

### WP3 · 独立来源校准充分性（**未来另经授权的定点实验**）

建议一个可先缩小的**双因素对照**，非既定必须网格：以固定总6144行/位置，分别选24×256、48×128、96×64个**独立校准source×rows**，完全相同的C_mean/gain/bias/R256/λ、budget和真实原Q1/Q2；另在固定来源数增加行数以估计token采样饱和。条件允许时再以来源类别均衡、不同seed、相机运动/背景分层交叉检验。

**可区分的预测**：若覆盖是关键，固定总行数下增加来源应改善新确认集尾部质量；若容量/目标主导，来源增多后仍显著损失，且不同表示/目标在相同来源数下有收益。训练和质量分割按**source及seed**隔离，不将同视频token随机拆分成“独立验证”。方案先按可获取的独立数据量缩放；24/48/96只是设计示例，不声称目前已有足够新来源。

### WP4 · 同预算组选择与时间预算（**未来条件执行**）

两个正交轴：在固定step配额(54,108,108,162)下比较原C_mean、组间联合/二阶SVD/OBS/SlimDiff以及无训练/轻训练组选择；在固定总四步主预算432组×40block下比较C24/UB1/UB2和**预注册的有限step/block再分配**。如引入源/seed依赖动作，先报告原动作空间的离线oracle提升与可能的在线pre-W1信息价格，分别计核算observer、routing、padding。原R12/R14/C1只是旧动作族有限负例，不封锁新的支持或DIFS式训练。

### WP5 · 轨迹及低成本补偿（**未来可选授权**）

通过OP2/OP3已有资产先检验真实UB1学生状态与旧teacher输出是否确实可配对。若有条件开展新原生观察，在**同一source/seed、同一student或dense prefix**上比较：仅组变化、仅alpha/b2、仅R256、同预算新组选择、跨步损失/2ndMatch/CTEC类成熟训练目标。检验被修改的位置在原生写回、后缀nextnorm、整步flow、下一latent及视频行为上的作用。对于trajectory误差使用实际有限映射\(T_D,T_S\)而不默认HiF4/Jacobian光滑；2ndMatch的敏感性loss可作为成熟好方法直接实现。此阶段旨在区分机制，**小规模局部成功不是视频级成果**。

### WP6 · 原生视频与系统验证（**有实际候选且获授权后**）

全40层×4步、同输入/seed配对、独立从未用于调参的确认来源、真实81帧自由生成；完整逐视频Aesthetic/Imaging/Motion及Dynamic Degree、Subject/Background一致性、Temporal Flicker、必要ROI/相机运动。逐轴报告相对Dense变化、源级tail和组别；测试模型指标版本、正反方向、评价粒度固定。明确原历史≤2%门槛是UB1原formal20内部验收条件，研究发现并不以先过门槛为必要资格；**若对外声称质量保真**，则应先预注册明确可审查的多指标标准并实际通过。

性能以相同硬件/运行次数/分辨率/权重驻留的Dense D-Q、原UB1、相同预算C24/UB2及可获得的成熟论文实现比较；汇报完整FFN、DiT主干及整段生成wall、VRAM、冷/热启动、bank、校准与可选训练代价。检查后验组稀疏是否真的减少W1、Q2/W2、合法M/K/tile执行；强Dense resident与native packed低位Dense可作为C线独立竞争基线。不能把50%主GEMM直接写成50%整网FLOPs或2×整体加速。

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

## 7. 使用的Skills、科研协议与方法边界

- **mechanism-research**：保留真实W1/GELU/HiF4 Q2/W2/e5、时间step、后缀及成本语义；为五个解释各给出不同预测和最便宜证据。
- **research-expansion**：从校准来源/组间耦合/时间预算/训练函数/执行kernel多个入口探索，允许直接用已有最佳方法，不固定算法族。
- **problem-reformulation**：将“按step配额是否最佳”改为“固定物理预算下哪些可支付的信息、函数与真实视频质量相对应”，允许数学或系统独立成果。
- **web-research-discovery**：从DyDiT/DIFS/SlimDiff/OBS-Diff/2ndMatch/CTEC/TCEC/TAPE的原论文机制出发，具体比较训练身份、time/width支持、算子及计时口径，检索帮助借鉴和澄清，不设强制创新差异门槛。
- **FFNDS领域Skill**：I/F/C可平行推进，UB1不是全模型输入动态；NATIVE/SRC/ALG/LIT/OPEN清楚区分，研究与部署主张按各自证据程度判断。

**与并行Affine54方案的关系**：[Affine54/量化一致函数与轨迹研究](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)关注**函数表示**（仿射+必要非线性）和部署数值；本UB1方案以**已有原数值子组支持+rank补偿**研究校准、预算/step和泛化。两者共享原Wan图、真实video/speed标准、论文工具与信息/传播理论，可互相提供对照，却不要求彼此先完成或采用对方函数结构。

**执行范围**：本方案自身保持OPEN/DISCUSSION_ONLY。用户已在当前会话恢复离线I/F/C持续推进，当前状态及逐项冻结授权以[ACTIVE_RESEARCH](../ACTIVE_RESEARCH.json)为准，最多同时四张空卡；原暂停保留历史身份。WP3–6的具体来源、代码、资产和预算仍须分别冻结登记，本文不派发这些实验。已完成一次性交接已删除，稳定入口为[当前上下文](../agent_protocol/CURRENT_CONTEXT_PACKET.md)；原始RUN、CI及质量验收保持各自身份。

## 8. 参考入口

[UB1源码级方法审查](UB1_METHOD_AND_PRIOR_ART_REVIEW_20261027.md) · [质量指标与FFN速度RUN](../../reports/worker_20261020/UB1_joint_engineering_acceptance_RUN.json) · [新20逐来源CSV](../../reports/worker_20261020/disjoint20_UB1_paired_sources.csv) · [C24/UB1/UB2固定预算对照](../../docs/38_固定预算下恢复末步完整FFN.md) · [OP3状态重拟合](../../docs/46_真实状态重拟合与导入延迟修正.md) · [此前Skill复审的研究解释](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md) · [领域研究目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)。
