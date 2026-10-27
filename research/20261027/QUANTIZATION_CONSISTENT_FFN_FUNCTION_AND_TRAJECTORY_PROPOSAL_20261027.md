# FFNDS · 量化一致的FFN函数表示与跨步响应研究方案

**2026-10-27｜独立方案 v1｜机制与方法并行探索｜研究状态：OPEN（未授权新模型执行）**

## 0. 研究主张与原则

**核心问题：在Wan2.2少步视频生成中，是否能够以更低物理成本实现原FFN的重要函数作用，并在原生HiF4/BF16执行和完整视频轨迹上维持质量？**

现有证据表明：Affine54已将一部分非线性映射折叠成D×D仿射层，在全40层×4step实际运行中具有显著FFN提速；但它的局部拟合优势没能维持完整视频。现存证据还显示旧FFN后验单通道变化可集中，原始合法64组和M16/K64 tile支持却往往几乎全满。由此形成三个相互关联、但可以分别成功的研究对象：**量化一致的函数表示（I/C）、有限状态扰动与视频作用（F）、实际执行与可部署成本（C）**。

**Skill审查的科研原则：学习和复用最好的已知方法，不人为要求每一个构造与论文不同。** 已发表的结构化剪枝、TARDIS线性化/回退、2ndMatch扰动敏感性、CTEC跨步蒸馏、TCEC误差补偿、OBS-Diff组校准，均可以直接作为可行分支和比较器。真正发现了新机制时据证据表述新颖性；只是成功复现、迁移、组合、改进系统时就准确表述工程/实证贡献。**一个好的已知方法优于刻意不同但质量更差的新方法**。文献查重服务于寻找成熟原理、了解假设与成本、设计合理对照，而不是强制差异化的淘汰门槛。

研究科学内容可仅形成定理、函数实测或反例；实际部署主张另需质量和物理性能证据。此提案不以UB1、Affine54、GELU配对、G13或指定loss作为不可替换的路线。I/F/C可独立入场。

## 1. 研究对象：完整原生计算图与已证实的起点

**Wan2.2 I2V A14B D-Q (LightX2V)**：high/high/low/low双expert、每步40 block、四步160个FFN位置；代表性M=31,668，D=5,120，H=13,824=216×64。被研究FFN调用路径为

~~~text
post-attention residual → norm2 / AdaLN / e3,e4 modulation
→ Smooth1 → original HiF4 Q1 → BF16 W1+b1
→ tanh-GELU → Smooth2 → original HiF4 Q2 (64→8→4共同scale)
→ BF16 W2+b2 → e5 gate → BF16 residual
→ attention / later blocks / flow → original scheduler → actual frames
~~~

量化基线实际为 **fake-QDQ + BF16 Linear**；真正packed FP4是可以并行研究的其它算子，并非已在原Wan部署。任何函数分析要固定source、seed、expert、step、block、输入边界与caller的gate；Q2选择必须顾及原完整64组scale闭包。

### Affine54已经实现的表示

对真实preactivation \(a=W_1q_1(x)+b_1\) 和保存的原post-Q2响应 \(q(a)\) 进行逐坐标斜率/截距拟合：
\[
q(a)=\kappa\odot a+\beta+r(a),\qquad r(a):=q(a)-\kappa\odot a-\beta.
\]
连续数学上完整余项是恒等分解；丢弃部分余项才是近似。将全部固定仿射基项离线组合成D×D层、保留54个原64组的真正W1/GELU/Q2并对新读出V作闭式ridge校准后，**实际部署执行形式**为
\[
\widehat F_{\mathrm{A54}}(q_1(x))=
A_{\mathrm{bank}}q_1(x)+b_{\mathrm{bank}}
+V_S\,Q_{2,S}\!\left(\phi(W_{1,S}q_1(x)+b_{1,S})/s_{2,S}\right).
\]
其中A_bank、V_S、b_bank是新BF16参数，读取q2的旧fit与缩窄Q2部署的数值路径存在待核对差异。门控残差由旧caller处理，和运行时bank的前馈输出是两个不同边界。

真实结果：两个持出但已开发暴露来源×160位置=320局部例，A54 gated SSE平均8.54%，Whole94为11.67%；A54在305/320例占优。三臂两来源共六条真实81帧视频均已生成；来源12热态FFN 9.707→4.449秒（2.182×），完整生成50.750→45.313秒（1.120×），但画面出现模糊/运动及细节变化，因此**尚未通过完整视频质量验收**。历史offline校准为4来源×每expert两步×每步32行=256行/函数，另有有限full-width neutral接线测试；它并未证明所有partial bank的原生部署数值已完全对齐。

[GELU反向配对]的恒等式 \(\phi(z)-\phi(-z)=z\) 只是上述仿射+余项框架的一个条件特例。原Wan权重的真实配对密度、Q2量化后的误差及物理收益尚未证明，**不单列为新算法**。

来源：[bank构建](../../tools/build_e0_reconstruction_bank.py)、[实际运行hook](../../tools/e0_reconstructed_runtime.py)、[全层匹配对照](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[六视频原生记录](../20261026/E0_RECONSTRUCTED_VIDEOS.md)、[此前Skill审查](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md)。

**已完成的相关数值控制**：[W2形状对照](E0_W2_SHAPE_CONTROL_RESULTS.md)在八个原Dense保存条件上，M160有约0.292%差异，恢复原M/row位置则全部输出逐字节复现；[原M D/Q](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)严格通过Q2和W2基线，保留直接/补集两种作用。该结果支持WP0–1的精确数值边界研究，尚未核对Affine54全部partial bank或解释其视频失败。本方案的H1–H3与A1–A4继续分别保留OPEN。

## 2. 最重要的三个机制解释及其不同可观察结果

| 假说 | 具体机制 | 区分性预测 | 已有证据状态 |
| --- | --- | --- | --- |
| **H1 数值部署失配** | 使用保存Dense post-Q2拟合的bank在原生缩窄W1/Q2、BF16矩阵与写回时产生额外误差 | same-input同gate比较离线拟合输出、native partial FFN2、post-gate/writeback，第一差异在数值路径出现 | SRC可确认拟合/部署边界不同；误差份额OPEN |
| **H2 函数响应/后缀失配** | 当前值SSE较低，但对真实前缀扰动的有符号有限响应与Dense不同，后续block/step传播改变视频行为 | 同值误差级别的候选，finite signed响应或相同前缀下suffix行为出现不同排序；模长敏感性不足以预测方向 | E0有局部-vs-视频矛盾和其它候选的P/N跨seed证据；A54专属归因OPEN |
| **H3 表示容量/校准失配** | 固定仿射+54 whole64无法表达自然视频所需的联合非线性；也可能原256行校准覆盖不足或正则未适配 | 增加独立来源/改变监督目标/改变合法表示，各自对持出函数响应与视频质量的影响不同 | A54强于全仿射且仍失真；各因素贡献OPEN |

三种假说可以同时成立；研究优先以**最少变更区分因果来源**，不以“必须与论文不同”作为成功标准。能直接复用论文方法改善质量或系统效率，是有效的研究推进。

## 3. 可以直接借用的论文机制与真实迁移范围

| 论文 / 年份 | 优先吸收的成熟方法 | 对Wan的合理应用与须验证的假设 |
| --- | --- | --- |
| [TARDIS 2025](https://arxiv.org/abs/2501.10054) | GELU近线性拆分、预折叠、动态回退 | 比较Affine54的压缩表示，保留真实Q2、gate与Wan大M GEMM执行约束 |
| [2ndMatch CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html) | \(J^\top J\)方向扩张敏感性与随机投影，可作轻量后训练目标 | 先建立与原生finite response的关系；HiF4舍入点的导数及计算开销需匹配；这是**很好的现有基线**，无须刻意改掉 |
| [CTEC AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34039) | 累计错误驱动的跨步Teacher→Student蒸馏 | 在Wan真实四步scheduler与双expert内构造offline target，不把Teacher未来信息当部署可见特征 |
| [Error Propagation/TCEC ICML2026](https://proceedings.mlr.press/v306/liu26fv.html) | 步间误差递推与低价补偿 | 使用实际Wan flow-step算子重新推导；论文不同版本的具体闭式展开需以正式终稿和代码逐项核对 |
| [OBS-Diff ICLR2026](https://arxiv.org/abs/2510.06751) | 时间步加权结构化剪枝、groupwise sequential calibration | 可以作为whole64支持/敏感性选择器，评价目标与配额固定，避免与别人的FLOPs口径混淆 |
| [DyDiT ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html)、[DIFS 2026](https://arxiv.org/abs/2607.06631) | 时间步宽度及少步联合蒸馏的成功经验 | 允许直接借鉴或有限复现，不将静态25/50/50/75宽度包装成首创 |

**重要数学关系**：

A. 2ndMatch的Gram形式重视方向扩张的**模长**。它对许多任务可能已经足够，不应先验贬低。若需区分有符号响应，可令同一条件 \(c\) 下的真实FFN完整写回函数 \(f_D(b;c)\)、候选 \(f_S(b;c)\)，选择来自实际前缀的有限扰动 \(\Delta\)：
\[
L_{\rm value}=\|f_S(b;c)-f_D(b;c)\|^2;
\]
\[
L_{\rm secant}=
\|[f_S(b+\Delta;c)-f_S(b;c)]
-[f_D(b+\Delta;c)-f_D(b;c)]\|^2.
\]
**这是常规有限差分响应匹配，并非声称原创公式**。作为竞争目标与2ndMatch、原值蒸馏同预算比较；只在出现可预测差异后考虑组合。举例 \(J_S=-J_D\) 时二者\(J^\top J\)相同而方向不同，但不意味着2ndMatch在实际模型必然失败。若扰动改变condition \(c\)（如AdaLN/e5），须将其作为**联合条件扰动**另比较，不能偷偷固定正常应变化的条件。

B. CTEC类跨步校准在理想实数步进 \(z_{t+1}=z_t-h_tv_t(z_t)\) 下，若Student与Dense当前状态差为\(\delta_t=z_t^S-z_t^D\)，令下一状态理想一致的**整个模型**输出目标为
\[
v_S^*(z_t^S)=v_D(z_t^D)+\delta_t/h_t.
\]
这是offline Teacher轨迹纠错目标，不是单个FFN直接标签，更不是推理时无代价获得的信息。真实scheduler的乘加/步长/BF16后处理需按源码修正；在四步迭代中，上式往往要求较大的控制幅度，应同时检验目标是否落在受限FFN表示的可表达范围。

C. TCEC的累积误差竞争对象：令\(T_D,T_S\)为完整原生单步有限映射，则
\[
P_t=T_D(z_t^S)-T_D(z_t^D),\quad N_t=T_S(z_t^S)-T_D(z_t^S),
\]
\[
\|\delta z_{t+1}\|^2=\|P_t\|^2+\|N_t\|^2+2\langle P_t,N_t\rangle.
\]
这只是精确代数恒等式。此前E0**其它候选**source425在相同prompt/image、不同seed的后三步cross变号，说明可以进一步检验条件性，但不能冒称A54或UB1已经完成对应实验。[原P/N](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。

## 4. 研究中的函数表示候选（开放且不限定为“新”）

| 备选表示 | 真正改变的工作 | 主要价值与挑战 |
| --- | --- | --- |
| **A0: 历史Affine54** | D×D folded linear+54组原非线性；实际BF16 bank | 现成对照，定位失败 |
| **A1: Affine54原生数值一致化** | 不先改变结构，仅检验/修正Q2、BF16与拟合/部署不一致的部分 | 成本相近但能隔离H1；如效果更好，可直接采用 |
| **A2: 文献支持的函数/损失迁移** | 沿用TARDIS近线性分区、2ndMatch响应匹配、CTEC状态目标或OBS-Diff组敏感性（可单用或组合） | 完全允许采用最佳已有方法；关键是wan实际数值与资源/质量 |
| **A3: 完整64组块耦合残差字典（探索）** | 将逐神经元斜率\(\kappa_j\)推广为每64组 \(K_g\in\mathbb R^{64\times64}\)，用组级耦合表示共享scale响应 | 256行拟合4096参数/组易过拟合；先检查有限统计是否显示额外可预测结构；离线变化未必减少实际GEMM |
| **A4: 其它I/C表示** | 量化码的联合信息界、函数重参数、布局/融合/编码域计算、必要时轻训练 | 不受A0–A3或G13约束；按数学、真实源码、物理代价决定推进 |

组块字典的数学形式为：令 \(z_g\in\mathbb R^{64}\)，原合法后量化函数\(q_g(z_g)\)，可定义
\[
r_g(z_g)=q_g(z_g)-K_gz_g-d_g,\quad
A_K=\sum_g W_{2,g}K_gW_{1,g}.
\]
保留全部余项是恒等分解，去掉部分余项得到近似。A0是K_g对角的近邻；K_g若不能增加有效泛化信息，优先回到A1/A2，**不为了不同而采用更昂贵的块字典**。Q2尺度可能依赖实际group及更细子组，要以源码可证明的闭包建模，不能把64组内坐标错误假定互不相干。GELU反向配对仅可作为其中一个数学特例验证，不设立独立原创目标。

## 5. 工作包与可证伪的里程碑

**WP0｜准确还原与现存资产整理（SRC/ALG，当前允许）**  
从真实bank构造函数和runtime hook列出每一实际W1/GELU/Q2/W2/affine、group/smooth/gate/BF16转换；找到校准四来源、held0/12的相关输入/输出是否保留完整张量，区分“已保存向量”与“只有JSON统计/SHA”。复核A54 full-width neutral范围与partial真正运行范围。产物是一张**原生同输入数字身份与数据可用性表**。这是修正真实误解，不是所有I/C发现必须先做的长期流程。

**WP1｜最小数学/源码判别（ALG/SRC，当前允许）**  
对H1从Q2闭包和cast次序找出可能的最早失配边；对H2给出匹配值误差但响应方向不同的构造及2ndMatch比较；对H3先判断256行是否支持拟合K_g以及对应有效rank，或者优先采用已有成熟算法。以确切约束/反例与真实执行成本决定是否值得新的原生验证。只需一两个真正改变推断的构造，**不设必须“原创不同”的结果指标**。

**WP2｜已存证据上的机制比较（READ-ONLY；受实际资产可读性限制）**  
在完整同source/seed/expert/step/site、相同输入和条件边界下比较离线输出、原生bank输出、原生后缀状态、当前值误差和有限响应。已有E0其它候选P/N只作为证据启发，不能移作A54实测。如果现有文件只有聚合标量，则诚实登记缺失，不凭空造运行结果。新CPU模型测量、大规模HF tensor实验及任何GPU/视频动作按ACTIVE需要新的用户授权。

**WP3｜经授权后才执行的分辨性原生试验（可选）**  
针对WP0–2尚不能区分的H1–H3，冻结来源/seed/站点与最小全40层范围，在同一原生prefix，使用matched dense/A54和一项竞争方案比较：teacher→离线拟合→partial native→写回→后缀有限响应。小试验用于验证机制，不冒称完整视频质量；发现数值失配先修正它，发现后缀敏感再决定是否要训练/换表示。不同训练目标保持同参数与数据预算。

**WP4｜经授权后进行强可比的原生质量与性能验证（条件触发）**  
真实候选需要全部40层×四步、完整自由81帧生成、独立冻结来源、成对seed，以及VBench Aesthetic/Imaging/Motion，补齐Dynamic Degree、Subject/Background、时间闪烁与运动/细节代理；同时报告原生FFN、主干/完整生成wall、显存、setup、bank存储和训练成本。与resident Dense、UB1、Affine54及可获得的成熟先例比较。历史UB1原20的2%门槛只属于其原验收，不自动成为每个数学探索成功的硬要求。

> **决策规则**：当A1足以解释画质缺口，优先完善数值实现；当已有2ndMatch/CTEC迁移有效，直接沿用并研究在Wan的真实效应；当块耦合K_g体现稳定的额外信息且成本可控，才扩展该新表示；当这些条件均不满足，转向不同I/C工作单位或更高容量模型。任何结果都是对具体机制的更新，不是对全部科研方向的否决。

## 6. Skill逐项审查与记录

| AutoResearcher Skill | 已落实的检查 | 应继续保留的开放性 |
| --- | --- | --- |
| **mechanism-research** | 源真实Q1/W1/GELU/Q2/W2/门控，分别提出数值、响应、容量竞争解释与最小判别 | 假说可以先于全部素材取得；纯数学结论不必跑视频 |
| **research-expansion** | 表示、数值算子、传播目标、训练/不训练、硬件布局都有不同入口 | A3只是探索候选，不规定为最终方法 |
| **problem-reformulation** | 将删通道问题改为函数实现及条件成本；反向GELU已归入Affine54特殊情况 | 容许完全采用TARDIS/2ndMatch/CTEC/OBS成熟方案 |
| **web-research-discovery** | 官方原论文与真实先例映射到具体计算步骤、目标和适用范围 | 先例应促进合理借鉴，差异发现后再谈新颖性 |
| **FFNDS领域Skill** | I/F/C并行、实测/推导/假说区分、强Dense和原生质量/成本保留 | G13仅为F线专题；未改ACTIVE、CI或一次性交接 |

**交付主线**：一项可核实的数值失配定位、一项可检验的有限响应规律，或一项真正更廉价的函数表示，三者均是合格进展；能否发表独立新机制依据最终证据，而不是起步时要求刻意不同。

## 7. 证据与后续科研入口

- [已审核UB1和科研空间](UB1_TO_FFNDS_EVIDENCE_DRIVEN_ROUTE_REASSESSMENT_20261027.md)
- [E0全40层匹配局部](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[六视频与系统计时](../20261026/E0_RECONSTRUCTED_VIDEOS.md)
- [E0完整hidden与物理tile](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)、[Q2合法组与物理算子](../20261025/E0_SHARED_TILE_PRE_W1.md)
- [独立G13方法](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[三线目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)
- 一手论文： [TARDIS](https://arxiv.org/abs/2501.10054)、[2ndMatch](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html)、[CTEC](https://ojs.aaai.org/index.php/AAAI/article/view/34039)、[TCEC](https://proceedings.mlr.press/v306/liu26fv.html)、[OBS-Diff](https://arxiv.org/abs/2510.06751)、[DIFS](https://arxiv.org/abs/2607.06631)。

**静态文档完成标准**：本文所有已测数字与原RUN/源码一致；内部链接可访问；理论等式标明成立条件；每个解释存在对照及可修订结论；不把复用成熟方法排除在外，也不预先宣称新方法/新执行授权。