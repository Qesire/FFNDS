# FFNDS · UB1与后续机制研究：Skill复审后的统一科学判断

> 2026-10-27 · 本文**直接修订原交叉推理稿**，依FFNDS的mechanism-research、research-expansion、problem-reformulation、web-research-discovery及AutoResearcher v2.4准则，使用P7「从重复调参中发现固定假设」重审。结果区分 ALG数学、SRC源码、NATIVE已测、LIT论文及OPEN待判别。现存研究目标是I信息/表示、F函数/动力学、C计算/执行三线并行，科学发现与部署算法可分别成立。没有启动新CPU/GPU模型、训练或视频实验；本次只读既有JSON/CSV与原论文。ACTIVE权限、一次性交接、原始RUN及历史门槛不变。

**后续已完成的实测与构造**：[共享输入联合支持](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.md)把第5节的信息种子推进为ALG＋604有限核对，同时给出固定方向精确投影这一强比较器；[原M D/Q](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)将真实量化直接/补集作用分开并严格复现基线。用户要求的[UB1全帧复核](UB1_INDEPENDENT_VISUAL_REVIEW.md)与[四原视频RAFT补评](UB1_DYNAMIC_DEGREE_RESULTS.md)发现外观保留及运动量差异可并存，原二值对这两对均为0。本段记录后续不同范围，原20/新20聚合、旧RUN及文献分析不覆盖改写；当前逐项授权见ACTIVE。

## 1. 此次改变了什么理解？

**中心发现**：原Wan视频FFN内**有意义的可变函数、可提前获得的信息、真实可省的物理计算**属于三个不同的对象。后验25%隐藏单元可以解释相当多的同expert变化量（限定的五block相对SSE 2.44–7.07%），却需要覆盖几乎全部原64组和M16/K64 tile；独立组能量选择还可能破坏联合有符号功能；而比原型更好的局部FFN SSE又未保证自由视频保真。由此打开的不只有动态稀疏，而是**新表示/函数重参数/共享计算/量化代数/时序计算**的宽泛空间。

上一版实质性偏差：
1. 将条件性P/N传播列为「最高科学优先级」，容易让所有方法先走G13。修订为**有原生证据的重点机制问题之一**，不压过I/C独立发现。
2. 过快把P/N交叉变号视为创新机制。**P/N是精确代数分解，交叉项本身是二次范数恒等式**；source425同输入不同seed的变号是条件性观察，若没有固定方向/同范数/第一变化边界及可预测的行为读出，则只是值得研究的现象。
3. 将R12/R14/C1旧动作族低oracle空间变成对动态方法整体不利的结论。实际只约束**具体组交换、补偿、4-token stripe、whole-site等冻结动作/输入/损失/预算**。另一种重参数、信息源或执行粒度即打开不同问题。
4. 曾把计算代数与执行称新增E线。它本来属于并行C线，不应另设新研究准入分类；训练后函数迁移则可从I/F/C任一起步。
5. 把24来源校准的泛化缺口预判为样本量不足。OP3说明状态匹配12源比同量旧状态拟合略有益，却未胜旧24源冻结UB1；它不能识别只增加校准来源、改训练目标及改数值实现中哪个是主因。
6. 过早对\(e^\top J^\top HJe\)传播建模赋新颖性。2026正式论文已直接研究扩散剪枝的Jacobian/finite-time敏感性匹配和量化累积误差，下游几何公式应作为**基础/竞争参照**。

**研究目标仍是**：发现视频生成FFN的信息—函数—计算组织规律，并以合理的真实执行代价维持视频生成行为。不同科学入口均可提出未有即刻加速方案的开放假说；有部署主张再要求来源泛化、完整生成质量、强成本基线。

## 2. 实验结论的精确作用范围

| 锚点 | 实际结果 | 说明什么 / 打开什么 |
| --- | --- | --- |
| UB1 | 四步25/50/50/75% whole64支持，总主MAC50%；原20 aesthetic/imaging/motion下降0.874%/1.345%/0.0074%；完整FFN 1.5461× | 冻结静态调度曾在特定正式集质量/FFN双轴成功；不是输入动态 |
| UB1新20 | imaging下降3.165968%，10/20来源个体下降>2%；同来源子集仍含尾部集中 | 固定校准/支持/目标/执行的**跨来源画质缺口**；原因没有归到单个因素 |
| OP3 | 状态匹配R256在同等12源对照有小幅变化，但8项均未胜旧24源UB1 | 同旧函数类、数据量/状态组合的有限重拟合不足；不否定更大校准或不同训练目标 |
| R12/D0、R14/N0、C1 | 旧R12 oracle仅胜其best fixed 0.2319%且较UB1差2.574%；R14 oracle SSE9747.50>UB1 8183.32；C1全site 72/72仍选UB1 | 已有冻结候选**动作空间的机会有限**，选择器不是当前最有解释力的第一问题；不否定新算子族或其它动态方式 |
| R16 | 四个观测站点local SSE较两固定对照优6.1276%，同来源完整视频 pooled RGB MSE却高8.2257% | 在这个来源/指标/有限动作下，局部预测有用但全程作用不同；非一般动态方法失败 |
| E0有符号五组 | 两/三组独立能量排名与受限组合的有限LN最佳不同28.91%/30.99%；FFN2至BF16写回的非加性比值从1.76–1.93%升至26.64–27.87% | 联合性不仅位于W2代数，也可来自数值边界和后缀度量；这些百分数均为**有限候选组**，不代表全216组最优方案 |
| E0全hidden/tile | 五block变化量后验单元25%相对SSE2.44–7.07%，同通道数按原whole64组32.42–54.67%；原完整M在M16/K64 tile100%活跃 | 先改变共享支持/函数表示/硬件工作单位；后验oracle信号**不是提前可用的决策器** |
| E0四步 | 另一个候选/诊断的10来源12轨迹1920嵌套site，source425同图同prompt跨seed的后三步signed cross反向 | 前缀状态、当前作用和后缀传输具有条件性；**不是UB1轨迹的原生因果定位** |
| Affine54 | 全40层320局部例平均8.54% gated SSE，FFN原生2.182×/完整生成1.120×，六视频有明显画质偏移 | 廉价的混合仿射+真非线性表示值得研究，但当前原型不满足全视频保真 |

原始数据：[UB1技术审查](UB1_METHOD_AND_PRIOR_ART_REVIEW_20261027.md)、[历史发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)、[R12](../20261022/R12_D0_native_results.json)、[R14](../20261022/R14N0_native_results.json)、[R16](../20261023/R16V0_analysis.json)、[E0联合组](../20261025/E0_FOLLOWTHROUGH.md)、[物理隐藏](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)、[Affine54完整视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)。

**校准统计新认识**：基于来源作为单位，原20和新增20 imaging比值均值分别下降1.345%与3.166%。对已测试20个来源的样本验收是确定的，未过2%的是新20。若仅在这批来源间做非参数bootstrap，区间会很宽、且原20参与多次开发，因而不能以这样的探索性区间宣称目标总体的统计显著优劣。后续区分“校准来源数/场景覆盖”和“同函数类表达上限”时，必须固定拟合配方、独立来源、相等总行数或同时明确两者变化。

## 3. 文献核验新增的强先例：支持直接迁移与准确定位

| 原作者/正式论文 | 可证实的核心机制 | 对我们原判断的修改 |
| --- | --- | --- |
| [DyDiT ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html)、[DyDiT++ TPAMI2026](https://pubmed.ncbi.nlm.nih.gov/41538341/) | Timestep-wise Dynamic Width、空间token及更广生成模型 | UB1四步不同宽度是**已有概念的静态实测实例**，不能以此单独主张首创 |
| [Dynamic-in-Few-Step，2026](https://arxiv.org/abs/2607.06631) | Wan少步蒸馏与step-specific结构联训 | 比UB1更接近同任务/模型，训练预算与真实执行/画质要公平比较 |
| [OBS-Diff，ICLR2026](https://openreview.net/pdf?id=eYdPMpW5os) | 时间加权Hessian、OBS结构剪枝、group-wise sequential校准 | 简单固定timestep Hessian不新 |
| [2ndMatch，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html) | 剪枝学生的\(J^\top J\)敏感性与有限时间Lyapunov-inspired匹配 | Jacobian/敏感性/动力学匹配已有直接先例；需要指出我们发现的**额外实数/量化/条件性机制** |
| [Error Propagation and Compensation，ICML2026](https://proceedings.mlr.press/v306/liu26fv.html) | 扩散量化误差传播的累积方程与时间步补偿 | 推导step error propagation/compensation本身已有先例 |
| [Chipmunk，2025](https://arxiv.org/abs/2506.03275) | 前置块统计、列变化量、voxel重排、压缩kernel、cache | 「delta列集中→前置选择→真实kernel」已有方法，关键新差异必须落在不同动作/信息/格式/因果作用 |
| [TinyFusion CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html)、[Dense2MoE ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html) | 后训练恢复性与显式重构专家功能 | 训练后职责转移是已有大类；在原HiF4量化/四步视频下提出具有不同预测的函数构造才有贡献 |
| [QuantSparse，ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) | 视频量化+attention稀疏、二阶重参数及多尺度恢复 | 量化/稀疏系统协同亦有成熟近邻，须区分改变了Attention还是FFN真算子/表示 |

**文献利用结论**：时间异质、Hessian、Jacobian匹配、动态列、低秩补偿与部分线性折叠都有成熟先例。对Wan原生问题，最有价值的工作可以直接复用这些方法，形成有效的系统与科学实证；发现额外新结构时再界定原创性，不把刻意不同当研究门槛。TARDIS/LLMFolder、MPC、FLAP、BRECQ、CTEC和RT-Lynx的证据深度、适用范围、有效计算边界已归入[共用文献审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)。原Wan dual-expert/HiF4/BF16/大M执行、条件性finite传播仍是值得研究的具体对象；论文其它设备的加速数字不能直接组成Wan排行榜。

## 4. 研究对象重构：把「动作、信息、后缀」拆开

对于指定模型、同一source/seed/expert/step/site，定义一个**完整实际执行的近似动作** \(a\in\mathcal A\)，在当前状态\(z\)上的真实FFN作用 \(\hat F_a(z)\)，其代价为\(c_a(z)\)。对于决策前已有信息 \(\mathcal I\)，动态策略为\(\pi(\mathcal I)\)。

**动作空间上界（一个固定度量与数据分布）：**
\[
L_{\mathrm{static}}^*=\min_{a\in\mathcal A}\mathbb E[L(a,z)],\qquad
L_{\mathrm{oracle}}=\mathbb E[\min_{a\in\mathcal A}L(a,z)].
\]
这是对**指定动作集、损失、预算**的事后上界，不能自动转移到新编码、另一层级动作或完整视频行为；上面的R12/R14/C1实际就是其反例。更强的科学问题可以直接选择新的\(\mathcal A\)而不是重新估计\(\pi\)。

**信息价格：** 能在post-W1/GELU/Q2看到的「后验贡献」可能解释自然数据，却不等于在W1前有可部署预测器。Prox的INT4 proxy需付费，Spark通过训练把信息前移，G12只有码值直方图的单行重排界还需昂贵在线查询。[G11作者代码](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[G12数学成本](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)。

**后缀与行为作用：**
\[
P_t=T_D(z_t^S)-T_D(z_t^D),\quad
N_t=T_S(z_t^S)-T_D(z_t^S),\quad
\|\delta z_{t+1}\|^2=\|P_t\|^2+\|N_t\|^2+2\langle P_t,N_t\rangle.
\]
这是**对原生finite map的严格恒等式**而非新动力学定律。其研究价值来自在**固定合法扰动方向、匹配当前/历史状态和后缀路径**时发现可预测的不同作用。例如：若两个seed当前FFN写回方向/能量经匹配后仍相同，cross符号因Dense后缀响应变化而变，则运输条件性更强；若原始写回已显著变化，则早期FFN/输入端差异可能更重要。光有cross变号和local误差计数相同，无法区分这两种机制。[原生P/N](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)、[G13正常函数定位](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)。

## 5. 从原函数结构探索其它省算表示：已有折叠原理与开放的Wan数值约束

此前此节将\(\phi(z)-\phi(-z)=z\)及神经元反向配对突出为“新科学构造”。**Skill与新文献复审后修订**：这是tanh-GELU已知恒等式，原Affine54早已利用\(\phi(z)=z/2+e(z)\)的奇偶分解；把近似非线性折叠成D×D主干、保留必要真实残差也已是TARDIS(2025)的核心方法，其研究团队在EuroSys2026发表LLMFolder。完整激活子集与消费者重构的MPC(ID，NeurIPS2022)亦是强先例。它们都应被作为**可直接采用的技术资产**，不因已有论文而被淘汰。[统一一手论文方法与Wan适配审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)、[当前Affine54研究方案](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)。

**真正仍开放的问题**是：在原Wan的HiF4 Q2共享尺度、BF16舍入、双expert四步状态以及M=31668的GEMM形状里，何种固定/条件的仿射–非线性功能组合有合理的实际成本、能保持完整视频行为？直接沿用TARDIS的热区预测与动态修正、MPC的消费者权重吸收、CTEC与BRECQ的跨步/块目标都合理；无需重新命名。先检验可提前获得的信息、修正的合法原量化组粒度、异常分布占用的实际tile及Q2/BF16数值差别。

**两个数学种子分开保留**：
1. 对称GELU：连续实数层面若两W1行及bias相反、W2列相反，则对应联合函数可折叠为仿射项；但这不等于自然Wan存在足够配对，也不等于Q2量化后精确。这是现有Affine54/恒等分解之内的**条件检查**，不是独立核心原创主张。
2. G12量化码联合信息：原逐W1行signed code histogram重排界单行紧，多行共享相同输入码顺序，联合可行域更受约束；可考察少量与输出相关的投影方向是否存在廉价联合作用证书，先分别做数学证明与实际查询成本核算。即使无法部署也可能形成纯数学结论。

**已测负例边界**：Affine54全160处可运行且局部SSE较好，但六视频仍有画质改变；旧单通道支持后验集中却无法产生whole64/tile空块。它们并不否定所有条件折叠或可训练/重编码计算，只要求在真实物理与视频边界判断。

## 6. 如何从已有实验走向科学判别，而非新的局部优化循环？

| 入口 | 本次修订后的科学问题 | 可直接从旧证据/源码完成的判别 | 作者最强近邻 |
| --- | --- | --- | --- |
| I · 信息 | 原Q1量化码/权重/共享输入中是否有可付费的联合函数信息？ | G12小维度可达域、少量输出方向界、查询次数与metadata字节核算 | LUT-GEMM、CodeGEMM、Prox、G12 |
| F · 函数 | BF16/Q2/后缀哪个最早产生观测的有符号交互？条件性差异是否改变预定视频行为？ | 按source+seed核查同前缀的postQ2/W2/gate/写回/后缀确实保存了什么；复核旧P/N签名与行为差异 | 2ndMatch、OBS-Diff、因果patching、ICML误差传播 |
| C · 执行 | 是否存在比whole64静态子集和原native dense更好的实际函数表示/执行单位？ | 从已存W1/b1/W2组配对及Q2闭包推数学条件；从M/D/H+作者源码做packed kernel/搬运/R256核账 | Chipmunk、SharQ、QuantSparse、Dense2MoE |
| I/F/C · 训练 | 少训练能否改变任务相关函数承担者的分布，形成新紧凑结构？ | 从TinyFusion/DIFS/Dense2MoE真实训练作用、训练预算、输出目标建立匹配候选/不同预测 | TinyFusion、DIFS、DyDiT++ |

这里的「从旧证据复核」是**科学观察可行性**，还不是模型新执行授权、实验成功或先行步骤门槛。对某一路线无法获得完整源时，可先做数学假说和成本构造；研究入口可在三个领域间双向迁移。

## 7. 系统性能的恰当定位：Amdahl是实例假设，不是科研路线封顶

R17同一来源×每臂5次技术重复的resident dense完整生成50.847s、全160 FFN含modulation计时约9.915s，约占19.50%。按**其它运行时费用保持固定、无重叠改变**的Amdahl模型：
\[
S(s)\approx\frac1{(1-p)+p/s},\qquad p\approx0.195,\qquad S(\infty)\approx1.242.
\]
实际UB1 FFN约6.468s、完整生成47.315s（1.07568×）。这为**特定resident环境下仅优化FFN耗时**提供近似范围；若改变QDQ/访存/fusion/attention/VAE/硬件占用、重叠或精度与布局，则p和整个成本结构变化，不能用1.242倍否定其它高效计算可能。[R17原始计时](../20261023/R17E0_system_timing_prefix.json)、[完整Dense权重驻留](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)。

方法层最强比较器是：固定模型/输入/seed/视频规格的resident Dense、原UB1与真正相同硬件packed Dense，真实online observer/metadata/cache/gather/QDQ、VRAM和启动成本全部扣账。原始全模型free-rollout与九维视频/ROI/时序指标才提供部署与质量结论；嵌套site、局部SSE、后验oracle都保留其有价值但不同的科学身份。

## 8. 收束的新判断

**科学前沿**不是先争论哪个mask、rank或时间步更好，而是识别：
\[
(\text{真实正常计算中的函数结构})
\longleftrightarrow
(\text{实际能够获取的条件信息})
\longleftrightarrow
(\text{新的精确/近似计算表示})
\longleftrightarrow
(\text{完整视频中的条件性作用}).
\]

最直接的科学资产有：原生Q2/BF16联合函数、五组与全hidden支持的表示矛盾、source425跨seed有符号作用、旧动态动作族的oracle边界，以及Affine54证明的「可实现但视频未保真」实例。这些资产分别可以产生有价值的数学证据、因果定位、算子改写或训练后新结构，不要求某条路径为另一条开路。

下一份真正改变研究判断的成果，最好是**一个经过原Wan源码与数据验证、能够区分实际计算/视频作用或取得真实收益的构造及其反例**。它完全可以是最成熟先例的成功迁移、复现或合理组合，而不只是更多路线宣言。结合当前暂停权限，优先使用数学推导、论文原文和现存原生操作数复核；新的推理、CPU模型测量、GPU、视频与训练按ACTIVE另行处理。
