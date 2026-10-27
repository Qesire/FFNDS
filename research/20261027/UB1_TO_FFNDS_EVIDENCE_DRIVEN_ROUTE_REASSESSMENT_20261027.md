# FFNDS · 从UB1到机制科学与低成本计算：论文—实验交叉推理

> **2026-10-27 · 科研判断文档（静态研究）**。本稿从[UB1方法及文献查重](UB1_METHOD_AND_PRIOR_ART_REVIEW_20261027.md)、[E0发现谱系](../EXPERIMENT_FINDINGS_ATLAS.md)、[G7–G13综合](FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)与作者原论文/代码证据出发，对A联合函数、B传播、C条件动态、D后训练四条路线作机制判断，同时恢复独立的E编码/执行研究入口。对数学推论、外部先例、原生实验、待验证设想分别标记 ALG/LIT/NATIVE/OPEN。**本次没有新CPU/GPU/模型/训练/视频实验；不更新ACTIVE、临时一次性交接及历史RUN。**

## 1. 统一研究对象及对照约束

基线：LightX2V Wan2.2 I2V A14B四步D-Q（high/high/low/low双expert×每步40block）；M=31668、D=5120、H=13824=216×64为代表性shape；原FFN路径 \`postQ1→W1→GELU(tanh)→HiF4 Q2(64→8→4共享scale)→W2→e5/BF16 residual→downstream\`。该计算图里post-Q2选列与pre-Q2/前W1省算为不同算子，原HiF4 fake-QDQ + BF16 GEMM与native NVFP4/MXFP4 Tensor Core GEMM不同。

科学成果：可证伪的新信息/函数/动力学/计算结构结论；方法成果：保持**逐维完整视频质量**并降低可复核的真实FFN及总系统成本。这两类成果可独立成立；I/F/C研究没有先后门槛。外部论文的FFN-FLOPs、DiT-FLOPs、全模型-FLOPs、图片FID和视频VBench不得横向直接排数值名次。

## 2. 六个已知事实与不能做的推断

| 身份 | 原生事实 | 足够支持 | 暂时不能推出 |
|---|---|---|---|
| NATIVE UB1 | C24四步全50% → UB1四步25/50/50/75%，同50%主FFN MAC；原20 VBench三项逐项<2% | 确实存在有效的非均匀静态计算预算 | 时间步预算单调、UB1全局最优、输入动态方法 |
| NATIVE UB1 holdout | 新不相交20来源imaging下降3.165968%，原20为1.344750% | 冻结方案存在来源相关的成像质量缺口 | 所有视频普遍>2%，哪组FFN负责异常 |
| NATIVE OP3 | 24源冻结UB1与12源状态匹配R256比较8项，新配方都未胜原24源 | 仅换教师/学生校准状态并延续原函数族不足以打败已有基线 | 更多校准来源无效，所有后训练无效 |
| NATIVE E0 group | 五个既有group、六个来源，二/三组单组能量选择与联合最优不一致28.91%/30.99%；FFN2→BF16写回的非加性比从1.76–1.93%至26.64–27.87% | 有符号协同/真实数值边界重要；需要逐边定位 | 全216组最优subset已知，联合Gram必然产生视频质量 |
| NATIVE E0 physical | 同专家变化量，后验25%单通道相对SSE2.44–7.07%，原whole64后验对照32.42–54.67%；完整M在M16×K64及以上占用100% tile | 后验稀疏与当前硬件合法空tile明显分离 | 任何新编码/重排/训练后函数无法省算 |
| NATIVE E0 flow | 六旧+四新来源、12条轨迹、1920site；source425相同图片/提示、seed不同，后三步P/N cross由全负变全正；局部更小的raw误差占多数 | 真实状态与候选当前作用可放大/抵消且有条件性 | cross属于某个特定FFN组，修复cross即修复视频 |

**关于统计**：按配对来源作为抽样单位、对两批各20来源Imaging的比值-of-means做固定种子非参数bootstrap 30,000次，仅作为探索性不确定度描述：原20相对下降约1.345%，朴素95%区间约[0.065%,2.803%]；新20约3.166%，约[1.441%,5.301%]。这不是预注册置信区间，也不能校正原20多次被开发使用的选择偏差；正式的“新20未过历史2%门槛”是确定性本次来源集合判定，并非已证明总体平均大于2%。新20前3来源贡献该集合净Imaging差额约56.98%；按subject/background分开聚合约2.907%/3.423%。[原CSV](../../reports/worker_20261020/formal20_UB1_paired_sources.csv)、[新CSV](../../reports/worker_20261020/disjoint20_UB1_paired_sources.csv)。

## 3. 从旧动作负例得到的新优先级规则：先行动作价值、再比选择器

### 3.1 可判定的动作族空间与选择价值

设 \(\mathcal A\) 为**确切执行定义**且部署预算相同的一组动作，\(L(a,z)\) 为固定输入和评价边界的实际损失，\(I\) 为决策点前可得的在线信息。先定义：
\[
L_{\mathrm{best-static}}=\min_{a\in\mathcal A}\mathbb E L(a,z),\qquad
L_{\mathrm{oracle}}=\mathbb E\min_{a\in\mathcal A}L(a,z).
\]
\[
\Delta_{\mathcal A}=L_{\mathrm{best-static}}-L_{\mathrm{oracle}}\ge0.
\]
对于实际预测器 \(\pi(I)\)，其机会缺口 \(L_\pi-L_{\mathrm{oracle}}\) 与求I的成本、真实算子成本分别记账。**事后oracle只用于判定当前动作族理论上有多大空间，绝不是可部署方法。**

- [R12旧两组交换](../20261022/R12_D0_native_results.json)：同预算事后oracle比最优固定仅改善0.2319275% flow SSE，且仍比UB1差2.5741%。只能关闭此历史动作空间。
- [R14四token stripe](../20261022/R14N0_native_results.json)、[指标审计](../../docs/78_指标审计与已有实验映射_R15_20261023.md)：当前有限动作族旧oracle SSE=9747.50，UB1=8183.32，动态=10733.59；**即使真实后验oracle也未胜UB1**。当时将超额损失约61.33%归因于动作族限制、38.67%归于selector regret，只是该冻结采样的账本。
- [C1 whole-site](../20261023/C1N0_analysis.json)：6来源×12site，旧UB1/swapA/swapB三个同站动作的72/72 oracle仍选UB1；虽另有逐行的post/nextnorm排序翻转7.10%/7.50%，不能由逐行翻转推断whole-site策略有价值。
- [R16V0动态恢复](../20261023/R16V0_analysis.json)：四个被观测位置的local SSE较最优固定对照改善6.1276%，同来源完整自由视频pooled RGB MSE反而高8.2257%；没有获得独立视频质量验收。

**由此将原路线C修订为条件性选择的两门槛**：(1) 存在既省物理工作又有可比较质量优势的动作族；(2) cheap predecision信息足以区分最佳动作，且observer+selector开销不会吃光收益。无证据支撑第1项时，优先研究新算子结构，而不是增加router复杂度。

### 3.2 真实系统收益的量级约束（ALG+R17 SYS）

R17一次来源每臂五次真实计时：resident dense完整生成50.847063s、含modulation的全160处FFN 9.915117s，FFN时间份额 \(\rho\approx19.50\%\)。UB1完整生成47.314734s、FFN6.468229s，整网1.075683×，FFN约1.533×。观察到的整网秒数减少3.53233s，其中FFN同计时边界减少3.44689s（约97.6%的整网净节省），这是跨不同计时边界的近似核账，不是逐核无重叠的严格加法。

若**其他所有开销完全固定**，Amdahl模型给出
\[
S(s_{\mathrm{FFN}})\approx\frac{1}{(1-\rho)+\rho/s_{\mathrm{FFN}}},\quad
S(\infty)\approx\frac1{1-0.1950}\approx1.24.
\]
即在这一强dense resident实例中，即使理想完全消除FFN耗时，整体生成加速极限也约1.24×。这是**特定模型/硬件/驻留/执行环境的条件上界**，不是对其它硬件/并行融合/编译重排/不同token数的物理定理。今后的C/E线可同时优化权重resident、QDQ和额外非FFN瓶颈，使上界本身随执行策略变化。

[原始计时](../20261023/R17E0_system_timing_prefix.json)、[dense全常驻两卡实测](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)。

## 4. 路线A：从联合TopK改为“真实有符号功能/数值来源→替代计算”

**研究判断：保留，拆为两个独立可成立的方向。**

- **A-science**：在同一个source/seed/expert/q和合法Q2尺度组中，区分W2输出直接有符号合成、Q2尺度改变对未选组的连带影响、BF16残差写回、下游attention/行为读出的非加性。G13的D(fixed post-Q2)/Q(valid pre-Q2)/O(writeback)/P(suffix)提供F线定位工具。非加性本身可能来自二次行为指标，不能自动视为神经元联合因果。[G13方法](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)。
- **A-compute**：研究某种函数是否可由更便宜的非线性/仿射/共享组表示，或以保真条件下的合法原量化算术精确构成。其研究可从恒等式/内核开始，不需要先把函数语义化。

**旧负例**：[全40层匹配重构](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)：Affine54+ridge 320个持出case平均局部8.54%（优于Whole94 11.67%），但[真实三臂六视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)完整源0/12均有明显感知劣化。说明“比线性更好”的有限结果不能满足full-video质量。

**最小判别**：先从已存资产核对真实完整W2 group贡献、pre/postQ2+gate、BF16写回和下游nextnorm是否为**同前缀、同step、同source**；如果只存了标量Gram，先查HF/manifest而不拼造联合向量。数学上提出两类不同的Q2作用预测，并用合法作用边界与已存数值复核。独立来源确认、真实模型干预需后续单独授权。

## 5. 路线B：从固定时间权重变为条件性有限传播科学

**研究判断：当前原生数据最强的科学候选，但需精准承接先例。**

严格有限映射恒等式（已测，不假设QDQ可微）：
\[
P_t=T_D(z_t^S)-T_D(z_t^D),\quad N_t=T_S(z_t^S)-T_D(z_t^S),
\]
\[
\|\delta z_{t+1}\|^2=\|P_t\|^2+\|N_t\|^2+2\langle P_t,N_t\rangle.
\]
实验中source425同来源不同seed使后三步cross符号全部翻转；来源16/step2曾出现候选净作用 \(\|N\|^2+2\langle P,N\rangle=-946.8824\)。它说明当前扰动可以抵消此前传播，而不是一切精度损失只能正向累积。**这不代表整个视频改善或某个具体FFN行有害。**

数学研究层面，可把站点局部扰动\(e_{t,l}\)通过后缀传播的一阶算子\(J_{t,l}\)联系到行为读出H，得到局部二次形式\(e^\top J^\top HJ e\)，不同step交叉项也可能存在。但这一线性化只是候选建模。针对原Wan，需用有限干预/完整suffix验证QDQ和BF16是否使其失效，并识别在真实视频轴上表现的作用。**直接提出timestep-weighted Hessian缺乏新颖性**：OBS-Diff（ICLR2026）已经做了时间权重和误差累积，SlimDiff（2025）采用跨时间步激活协方差和功能权重组，ERTACache/IC-Caching讨论采样积分与缓存误差。[OBS-Diff](https://arxiv.org/abs/2510.06751)、[SlimDiff](https://arxiv.org/abs/2509.21498)、[Increment-Calibrated Caching](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html)。

**关键竞争解释**：
B1) 主要来自前面step已发生的状态偏移，后续Dense也会放大；
B2) 主要来自当前FFN的联合有符号写回；
B3) 两者交互和后缀attention/flow随source/seed条件变动而改变最终行为。
已测P/N支持B3交互的存在，还不足以单独归因全模型质量失配。下一步首选在现存trace里追溯相同base状态的第一可分辨图边；G13的O/P可用于未来经授权的必要反事实，优先检查“输出不同”与“下游放大不同”的可区别预测。

## 6. 路线C：具体可省算单元之前，不预设输入动态路由

**研究判断：保留可选动态性，但直接“更好的TopK/NetVOI”暂缓。**

[完整hidden](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)与[连续token共享组](../20261025/E0_SHARED_TILE_PRE_W1.md)揭示：
- 在原五个block，后验25%逐通道可比较好地重建同expert创新，但同预算whole64成组损失更大；
- 原完整M的M16×K64、M64、M128整块布局对12.5%–50%的后验活跃率仍全部tile满；
- 连续M128共享25%列的同专家创新SSE在中间block10/20/30大约38.99%/42.21%/40.12%，比逐行6.51%/6.72%/7.19%差很多；
- 当下游Q2使用共享尺度时，只在GELU前删除某列并重算Q2会影响补集，未保证原生函数相同。

**要区分的三种动作**：pre-W1合法缩窄（真正省W1+W2）、post-Q2 W2-only sparse（W1已付）、重编码/后训练形成新group和新物理tile（改变函数和部署语义）。每种动作的信息时间和精度合同不同。

**最近论文**：Chipmunk（2025）已经有pre-W1列变化估计、voxel重排、实际压缩kernel与缓存；Prox（2026）INT4代理先付费再做selected投影；Spark（NeurIPS2025）通过训练让router信息前置。新颖性须明确与这些方法的动作、训练状态、QDQ/group和大M执行差别，而不能只把selective TopK改名。[Chipmunk](https://arxiv.org/abs/2506.03275)、[G11源码核对](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)。

**最小判别顺序**：(i)已存合法动作族的事后质量上界是否胜强UB1？(ii)动作组支持是否形成空的原内核计算块、或重排后能实现新kernel？(iii)上游cheap信息能否跨source预测？(iv)observer、routing、packing和cache成本是否吃掉收益？每一步只是可选进入准则，不是永久要求先做函数定位。

## 7. 路线D：后训练是函数重分配实验，不是加大R256

**研究判断：保留为独立机制及系统路径；目前算子/目标认识不足以预设具体训练方案。**

DIFS（2026）联合结构稀疏与少步蒸馏、Output Rollout学习；DyDiT++利用time-specific LoRA；Dense2MoE、Spark显示模型可以通过训练产生新的低成本函数承载结构；TinyFusion（CVPR2025）强调**压缩后恢复能力**而非即时压缩前误差。[DIFS](https://arxiv.org/abs/2607.06631)、[DyDiT++](https://arxiv.org/abs/2504.06803)、[Dense2MoE](https://arxiv.org/abs/2510.09094)、[TinyFusion](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html)。

我们现有的OP3与R7D2/R11T1已表明在旧函数类别内换状态或直接改目标，未自动胜冻结UB1；R8F2训练空间Hessian改善却在验证空间外退化。这些关闭的是具体校准、参数化和目标，不是“轻量训练”这一整类。可研究的不同命题是：**低训练预算能否使原分布式联合功能转移到新的quantizer-consistent紧凑表示，并在独立source完整视频中保持职责？** 先以已有论文训练协议和真实Wan参数/组约束算训练+推理成本；未来有明确授权才训练。

## 8. 新增路线E：不以稀疏为前提的计算代数与物理布局

**研究判断：独立成立，近期直接工程价值可能更高。**

- **强系统基线**：[18视频dense all-resident](../20261025/E0_DENSE_WEIGHT_RESIDENCY.md)：原值不变的完整生成热态75.172/75.372s→50.419/50.789s，1.491/1.484倍；代价常驻52.34GiB、峰77.08GiB。对受内存限制设备不是免费收益。
- **已实现低位基元**：[SM120 NVFP4 primitive](../20261026/E0_SM120_NVFP4_PHYSICAL_PATH.md)五次官方CUDA GEMM和host reference一致，但未集成Wan也未证比BF16更快。
- **编码域数学和执行先例**：[G12](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)单W1行仅有signed code hist的rearrangement上下界是严格最紧；多行共享同一输入排列，独立行界笛卡尔积不等于联合可达域。但所有行所有group生成证书的复杂度估计巨大，不能当廉价加速器。
- **作者源码检查**：[CodeGEMM](../literature/20261027/G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)公开实现对multi-token退回全部dequant+torch matmul，SpQt multi-token也退dense；Wan M≈31668不能照搬GEMV报告。SharQ有sparse GEMM+dense FP4 residual两笔工作。

可独立研究：真正改变多token GEMM与QDQ融合、合法数据布局、有限码本局部乘积或权重常驻切换，不预设全模型存在可删通道。若声称原函数精确等价，须逐步证明原scale/group、舍入、BF16、gate；若近似，则进入与原UB1同等视频质量门槛。

## 9. 收束：一组相互竞争的实质假说，而非新的固定路线列表

| 命题 | 支持/削弱证据 | 与最接近论文的不同预测 | 下一最廉价判别 | 状态 |
| --- | --- | --- | --- | --- |
| **H-F1 条件性传播结构** | source425同输入跨seed cross翻转；R16 local gain而视频MSE变差 | 对相同步序、相近local SSE，传播方向/视频作用随前缀状态改变；超出固定timestep Hessian | 旧trace逐source/seed配对P/N与图边；核实原UB1轨迹对应state是否保存，分清E0候选身份 | NATIVE phenomenon / causal attribution OPEN |
| **H-F2 Q2尺度功能与direct写回分离** | 原Q2组闭包反例；BF16写回非加性增大 | 若尺度调制承担主要作用，valid pre-Q2与fixed post-Q2的补集W2方向及后缀行为有可预测区别 | 原生Q2公式+可访问保存载荷先静态核对；后续合法D/Q/P | ALG / NATIVE limited / OPEN |
| **H-I1 低价联合代码信息** | 单行signed重排界成立，多行共同配对、信息价格昂贵 | 联合可达域明显窄于各行界积且只需可共享摘要；与per-row bounds/LUT-GEMM不同 | 小维度解析反例/已有聚合统计，独立算hist+内存成本，别把数学界等同native结果 | ALG / OPEN |
| **H-C1 数值保持的新计算表示** | 组级后验稀疏无法产生空tile，原native QDQ与packed内核不同 | 新合法布局/码表示对真实大M减少kernel/访存/QDQ，强于resident BF16或packed dense | 只读算子图、既存shape、作者kernel与原生G11代码的成本账本 | SRC/LIT / OPEN |
| **H-T1 训练后职责迁移** | Affine54 nonlinear有益但完整视频失真；DIFS的rollout与训练专家 | 小参数训练后，新函数能在较小物理结构里保持动态视频行为，而非靠同一C_mean恢复 | 明确原/新参数和目标，文献最强对照与成本；新训练另授权 | LIT / OPEN |

**建议的科学优先级**：H-F1/H-F2科学定位最有现成原生证据；H-I1/H-C1作为独立数学与执行路径并行，不必等待G13；H-T1在发现可迁移表示/训练目标后推进。对于方法实施，以动作空间价值、决策时点、真实物理成本、独立视频质量四个维度判定，不对研究设定固定算法或必然训练假设。

## 10. 数据、评价和终端验证的防混淆说明

- UB1原20和新20是来源聚合三维VBench，独立视频数各20；源内81帧、每位置256校准row或1,920 nested FFN site不可作为独立来源数；新20Imaging>2%是原门槛的实测失败。
- 后来E0 P/N、source425和Affine54视频属于**其它候选/诊断**，只是启发UB1潜在机制，不能写成已测UB1本人的P/N视频作用。
- 终端质量至少增加Dynamic Degree、subject/background、flickering及相机/主体运动与ROI稳定指标，与三维历史评价分开登记；原VBench2/九维环境准备失败时不得声称完成。
- 所有候选加速主张应以等硬件/权重驻留、正式配置、真实FFN+全生成时延、VRAM、冷启动/训练摊销对比，和优质静态UB1、resident dense、相邻论文作者实现形成真实Pareto。
- 本文是数学推导、来源审阅、已存实验综合和静态成本判别；未来的原生运行必须依据ACTIVE与用户授权。不会覆盖原始RUN，也不清除下一Agent一次性修正交接。
