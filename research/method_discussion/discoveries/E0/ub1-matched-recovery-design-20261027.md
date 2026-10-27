# UB1 同来源支持与消费者恢复：可辨别的小规模成熟方法比较

2026-10-27 · 独立设计，OPEN；无拟合、模型、GPU或新数据分析。四个预定位置：step 0/3 × block 0/39，分别保留原 54/162 个有序整64组。原 calibration24 以 source-disjoint 12/12 作一对固定互换折，训练每源 256 行共 N=3072；每折所有评分、支持选择和参数拟合只用其训练12源。持出源以源为单位计算误差和方向，不能把 token 或四位置称为独立视频。确认24源资产可读、两折无源/seed泄漏后冻结；既有24源均已用于历史UB1开发，不能称盲测或最终确认。

**最小有辨别力的主实验是四臂同一R256成本/训练数据的 2×2 对照**：原冻结UB1支持与 FLAP式组波动支持，各与原标量消费者及 ridge恢复消费者搭配。每臂重新拟合同一 R256，先拟合主支路，再按同一 λ、rank、规则拟合 residual；固定 source、rows、位点、dtype与全部超参。A 冻结24源UB1另作未匹配工程锚点，不与12源训练当成等数据量的算法归因。可先运行同支持两臂再扩到支持轴，但事前定义全部方法、冻结选择与 budget。

| 臂 | 支持 | 主消费者 | 辅助 |
|---|---|---|---|
| B | 冻结UB1（24源选择） | 标量α+折叠bias，以训练12源重拟合 | 同预算R256重拟合 |
| C | 与B完全相同 | 带原W2先验的ridge+bias | 同预算R256重拟合 |
| D | 训练12源 FLAP式 whole64波动 | 如B | 同预算R256重拟合 |
| E | 与D完全相同 | 如C | 同预算R256重拟合 |

B/C直接比较改变消费者；D/E同理；B/D和C/E比较支持，但**B的原支持已使用24源、D仅使用12源**，因此不是严格的匹配数据支持选择。最简修正是增加 B0/C0：在每折训练12源重新计算原 C_mean 支持及各自同配方拟合。正式方法轴比较 B0/C0 与 D/E；B/C为隔离原UB1固定支持的消费者恢复，原冻结UB1是额外锚点。6个重拟合臂×2折×4位置，非全160或视频候选；算力若受限可先预注册4位置×一折，保留另一折作为事先冻结的互换确认，不以首折结果重新调选择规则。

## 操作、算式和尺度

在固定位置收集原 post-Q1 (X\in\mathbb R^{N\times5120}\)，full post-Q2 (A\in\mathbb R^{N\times13824}\)，同输入 dense W2输出 (Y\in\mathbb R^{N\times5120}\)，原W2 (W\in\mathbb R^{5120\times13824}\)，固定该位置原 gate (g\)。实际历史训练先用dense V2切片，部署是缩窄W1→GELU→Q2；本设计在 native一致性结果出来后决定使用原V2切片或同源真实缩窄Q2作为训练特征，**所有臂统一**，不先假定等价。(A_S\)只取原整64组，step0 (K=3456\)，step3 (K=10368\)。前项不含 b2 时让bias吸收原 b2。

- 原方法组能量：\(C_j=N^{-1}\sum_i\|(A_{i,G_j}W_{:,G_j}^{T})\odot g\|_2^2\)，训练折计算后排序取k。冻结UB1支持另取历史pack。
- FLAP式低成本替代：\(F_j=\sum_{h\in G_j}\operatorname{Var}_{i}(A_{ih})\|g\odot W_{:,h}\|_2^2\)，先在训练源内、再跨12源等权聚合，保持每源权重一致；不估跨channel signed covariance。这是面向原whole64 group/gate的FLAP式适配，不声称原版完整复现。分别记录组重合率，若排名完全相同则方法轴无实际干预。原文在结构化剪枝中结合波动评分与bias恢复：[FLAP](https://ojs.aaai.org/index.php/AAAI/article/view/28960)。
- 标量主支路按旧 `fit_step_budget.py` 中心化、gate加权、非负α与折叠bias训练，保持历史数值语义，输出 BF16。
- 新消费者以 \(Y\approx A_S V^T+b\)，围绕原消费者 \(V_0=W_{:,S}\) 做训练折中心化的ridge。先拟合门控输出坐标 \(T=Y\operatorname{diag}(g)\)、先验 \(U_0=V_0^T\operatorname{diag}(g)\)：\(\min_{U,c}N^{-1}\|A_SU+\mathbf1c^T-T\|_F^2+\lambda\|U-U_0\|_F^2\)。固定gate非零坐标再除gate折回W2，近零gate回退原列并登记阈值；此除法可能放大误差，native BF16转换与gate值必须验证。更稳妥可改拟合未门控输出并用gate作为评测权重，但须先冻结，不能以持出数据选择。bias不惩罚；正则不是数值小技巧，而是未观测方向回归原W2的先验。
- 每臂残差用原 post-Q1 X 和自身训练主支路误差，调用相同中心化、白化R256、λ=1和 BF16 双线性配方。保留 no-R256 的主支路训练/持出误差作为离线读数，不额外跑部署算子；它揭示消费者收益是否被 R256 吸收。若加训练迭代或联合拟合，视为另一个拟合预算，不混入首轮。

## 3072 行的限制与 MPC 意义

中心化使 \(\operatorname{rank}(A_{S,c})\le 3071\)，而K分别为3456和10368，因此最少385和7297个输入特征方向在训练矩阵核空间不可辨识；训练误差小不意味着该解在新源稳健。一般满行秩时，若任意新增64列不提高当前数值秩，未正则化训练重构残差可在3071个独立方向已张成后归零或平坦。step3的162组远大于此秩上限；所有组并非物理冗余证明，也不能解释为后续持出源同样无信息。稳定性需要源级保留集、正则、数值秩和奇异谱，有效样本更少因同源token相关。

[MPC原论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/f8928b073ccbec15d35f2a9d39430bfd-Paper-Conference.pdf) 用 post-activation 列插值 \(A\approx A_ST\)，折入消费者 \(T W^T\)。逐列ID选出的通道未必凑成原始64组，不能直接跳过不整组的W1及保留Q2语义；简单先选单列再补齐64会改变有效预算与算法。若追加 block-MPC，用组约束 pivot/残差更新选整个组，ridge投影且预先固定tie-break，训练折上监控每次block residual和数值秩；不是原算法逐字复现，也不能在训练 rank 已饱和后把任意选组解释为等价泛化。组数162时若逐组训练残差曲线在饱和后平坦，优先加 source-held-out投影评估或正则，避免用伪精确训练排序；原W2输出加权或消费者ridge与未经权重的ID优化目标不同，应分开比较，不混成单一“ID比FLAP好”。

## 可证伪预测、成本和旧结果

- C 相对 B 同支持持出源改善而no-R256主支路改善更明显：消费者有恢复空间；若训练好、持出恶化，支持过拟合/先验不足；若 R256 后差异消失，则原支路已有补偿。
- D/E 对 B0/C0 改善且组重合不全：方差排序在该4位置及折中比 C_mean 更好；未改善或负向则降低这一特定组聚合/校准方式。既然 FLAP 原本含 bias 补偿，同组同消费者对照可隔离评分，但不是论文完整端到端复现。
- 局部门控误差改善却新视频动作/成像不改善，支持目标边界不充分而非判定支持无效。趋势不一应报告每位置、每来源以及回退风险；先冻结一个主响应如两折源级 gated relative SSE，报告原始 SSE分母、等源权重和最差/分位，不以post hoc挑选门槛断言成功。

每次输入前端组数和两主GEMM形状不变；各臂保留相同R256两BF16 GEMM，线性MAC相同，不保证wall一致。新消费者是 \(5120K\)个BF16权重：step0 17,694,720（约33.75 MiB），step3 53,084,160（约101.25 MiB），替代原相同形状的W2片而非额外再加；打包、拟合、加载、显存及alpha/bias、QDQ、gate必须计入。训练ridge的矩阵规模/分解方法、精度、源数据搬运、耗时和峰值内存另计，样本域解只在λ>0时可省K×K内存，但仍有N×N与N×D中间量，预先冻结 solver和正则调参；从多λ中用外层持出折选超参会泄漏，须训练源内嵌套小折或先验固定。最终需原生算子/整160和独立视频才讨论质量速度。

OP3只测试固定支持/gain、12源换学生或teacher状态后的R256，8项未超过24源冻结锚点；本设计改变消费者与组评分，并有训练数据匹配控制，故未被该特例覆盖。原24源UB1和两个12源拟合不是公平训练数量对照；旧C_mean支持本身看过24源的泄漏已由B0/C0更正。无新模型/训练/实验授权，所有运行前需按现行ACTIVE登记代码、资产、预算、停止规则与确认集。

## 独立代码审阅 · 已选定的 ungated ridge

审阅 `tools/ub1_recovery_math.py` SHA256 `b3eebd874828ff130e39259d6f8cc7470302b10580704f6ad07e81d90025be37`，`tools/e0_ub1_matched_recovery.py` SHA256 `9133e435de698b81267158d5acfeac0b16596aafea9f3bcec4fdb5ac9d04d39b`；对应测试 SHA 分别为 `92e8c07865e06c30752892bb382c8d5153eac96b255e4a17439a897c585ce3ca` 和 `4d1dc398d4549987f054b4ecb7ba3b51fc13b2a46b9166144e1d33eda72cd741`。两组小型 CPU 测试共12项通过；不运行真实拟合、GPU或生产分析。**在指定数学、来源分离及计数范围未发现必须修复错误。**

该实现实际采用 ungated Y 的先验中心化 ridge、非惩罚截距及 shared BF16 alpha feature，不使用我上文作为候选的门控逆恢复；因此上文候选公式不是本轮执行公式。`ridge_consumer:118–132` 先以原selected W2和bias生成先验输出，再在中心化特征与残差上解对偶岭回归，折入权重和bias后转BF16，数学一致；gate仍用于支持、标量α、R256训练及评估。其优化的是 ungated目标，不能报告为严格 gated最优。`select_supports:45–53` 的组内激活和gate加权W2的 FP64 Gram二次型保留了组内符号交叉；FLAP 式 pooled variance包含源间均值变化和源内变化。支持、alpha、ridge与R256均由 `training_subset` 返回的12源形成，另外12源仅评分；4位置×2折×4臂，各臂各自R256。历史来源曾经用于UB1开发：该CV为已暴露来源的局部诊断，而非新视频盲测。

计数复核：4组W2 Gram、8组激活Gram、96个 dense FP32投影、192个 scalar FP32投影、16个消费者ridge、32个R256 fit；每臂训练3次BF16线性加12个持出源×3次，共4×2×4×39=1248次BF16前向。fit内部FP64矩阵运算作为fit成本的一部分，并非零算力。

最强边界反例：即使所有臂都完全无泄漏、相同公式，训练前向在M=3072、持出在M=256；BF16 GEMM对形状敏感时，训练/持出差异会混入不同执行形状的数值差，不能单独归因为来源泛化或原生M31668行为。各臂使用相同形状仍可比较这一诊断设置内的差异；需要原M一致性与未来独立视频来验证更强主张。先前16位置的 narrow Q2精确一致只覆盖该固定source/step/block和所检160行，不自动证明本次96源位置的全部行一致。支持评分在训练源同权pool，均值跨源变化可主导FLAP分数，是预先明确的设计选择，不能将其解释成纯源内波动。

### 前向形状修订复审

新的 driver SHA `dfbca76f6396820bb9ab25a0f1603aaf7f0f8101dbc0cb28b837970162209969`、driver tests SHA `e95e183f035298a275b76bd65c94119bf8051a968f89d2232508ddc34d1da5aa`，math SHA 保持 `b3eebd874828ff130e39259d6f8cc7470302b10580704f6ad07e81d90025be37`。CPU 测试两组共14项通过；未运行真实拟合或GPU。

`sourcewise_forward` 按连续256行切分并保持拼接顺序，训练特征和目标均由相同有序train列表拼接；主支路每源1次、R256预测每源2次，再合并用于该源指标。held侧同样每源256行、每源3次。4位置×2折×4臂×(12训练源+12持出源)×3=2304次BF16前向，其他fit/FP32/Gram计数不变。该修订消除上一版本训练M3072与held M256的BF16前向形状混杂；R256拟合本身仍使用合并3072行（这是训练数据集而非前向M）。M256仍不能代表原M31668的GEMM数值或自由视频，本次只评估保存teacher状态；源级数据和曾暴露身份不变。

## CV 完成后的下一项判别：尽早进入完整自由轨迹

**结果身份**：实际 `e0_ub1_matched_recovery_20261027/attempt1/RUN.json` SHA256 `d21216e83a3566499062dd7327e18d5147aa4ad8ceb25d2acf5463fd1f2513e3`，状态 `COMPLETE_EXPOSED_SOURCE_CV_ONLY`；4位置、24个已开发来源、2折、4臂，32次R256/16次ridge，无模型或视频。协调者汇总的消费者ridge平均corrected SSE约降低39%、FLAP scalar恶化约9%、step0/block39 ridge收益弱、ridge后R256附加收益近零，均是该有限源级teacher-state实验的观察，不能直接认作完整轨迹或固定原UB1支持的同幅收益。

**首选：冻结原24源UB1的全部160位支持，构建一个同源24训练的消费者ridge bank，尽快做少量成对完整视频。** 原因：本次可证据支持的是消费者轴，而FLAP未显示先改支持的必要；原支持/宽度/alpha和原FFN计算图不变，可集中看“换W2消费者”是否带来视频功能变化。它是具有明确比较对象的真实 candidate，不需先把四臂乘以160做完整两折CV。已有局部匹配fold是12源、重新选C_mean支持；直接使用24源冻结UB1支持是合理但新建的另一个方法，效益和泛化均待实测。

1. **bank前置对照，按资源可用条件继续**：逐位置绑定原UB1支持、alpha与b2、有效原W2/Q2、24源相同源级行和gate。用同一已冻结ridge prior/λ拟合完整160消费者并按BF16写回，保留原R256与重拟合R256的选择作为不同对象：推荐先**ridge + 原冻结R256**，同时只离线拟合一套对新主支路的R256作为第二诊断/候选，预先说明前者可能与新主支路不匹配。因为局部CV中ridge后R256附加量近零，不能直接在完整运行删去R256；若想省两次GEMM，应另做 `ridge-only`，其错误/费用变化与计算形状需独立实测。固定原支持、alpha与参数身份可避免把支持变化和消费者变化混在一起；产物及预处理账本必须覆盖160位，即使先只生成一条视频。校准资产缺位时如实标MISSING，不用近邻块参数填充。
2. **最低成本的轨迹判别**：先在事前固定的一个已有相同source/seed对照上运行全160 ridge+原R256自由视频，并与原UB1、dense的原生成视频和RUN SHA匹配（原参考字节、模型/conditioning及参数差异均先核验）；如已有dense/UB1可复用则只需新candidate 1条。选运动差明确且完整输出可读的既有场景；它是开发诊断，不能表述为盲验。同步逐step latent/flow和原生全FFN及全生成wall记录，带无修改hook控制和每层真实执行覆盖；不存在必需历史轨迹时如实记录而非生成冒充。若初例无严重崩坏、运行身份正确，直接追加预先固定的背景/成像尾部来源及动作来源至少各一例，包括同源dense和UB1（已有精确配对则复用），并在来源层报告人工/模型观察状态和完整81帧动态/背景/光照。不要把首例正面效果设成可任意改换场景/阈值的试错门槛。
3. **方法归因与确认**：若原R256抵消或损伤ridge，应在同输入/seed、同支持和bank下比较 `ridge-only`、`ridge+原R256`、必要时 `ridge+重拟合R256`，RR移除的效率和数值必须单列。若ridge candidate局部改善但自由视频不改善，下一次先读实际flow/latent/视频差，再针对来源进行小规模轨迹/消费者目标的区分；不要把四位置的局部SSE继续做为单调代理。若视频出现明显改善，预先登记从未用于拟合、调参、方法选择的独立source×seed扩展，以原UB1/dense为配对，事前冻结多轴质量与tail、总成本和收益；两批原20以及本次24源都已暴露，非盲确认。

**竞争机制及相反预测**：若 `ridge+原R256`改善局部但视频差、而 `ridge-only`改善，支持旧R256针对旧主分支的补偿不再匹配；若无R256两侧都差、重拟合R256改善，支持剩余误差对当前输入仍可线性恢复；若三臂相似且轨迹偏离，支持训练状态/监督边界或支持容量未覆盖。自由生成存在非线性轨迹分叉，不能由两视频确定某一层的因果，也不能用其外观判断总体质量。条件预测是决策指引，真实输入状态差、数值转换和源差异会产生竞争解释。

**实际费用与可替代方案**：160位单bank ridge需要160次完整24源拟合与对应 BF16 权重/pack（各位置 5120×64×组数；四步合计单bank约10.55 GiB BF16消费者权重，按40层×每步固定宽度估算，不含bias、元数据或旧权重共驻），加原QDQ、W1/W2、R256、模型资产加载、构建和源观察；算力、峰值显存及产物视实际计量而定。新ridge bank不改原支持的理论主MAC，但新W2参数/加载和真实GEMM性能须计，native HiF4 QDQ是fake-QDQ/BF16 GEMM；不能继承 UB1 的1.546×速度。`ridge-only`可在同组宽下节省每位两个R256 BF16 GEMM，必须重测完整性能与视频质量。选full160×两折×支持×消费者CV可以更好测位置泛化，但拟合/存储大幅增加而仍缺自由视频，不应作为所有视频的先决条件；在小规模视频暴露新分歧后可针对全160按资源分批进行。若不具备全部160拟合资产，先使用已完成4位建在原UB1中作固定位置干预可诊断其有限作用，但不能替代全bank，也不应将正面或负面归因外推全部位置。正式执行须先冻结资产、方法、上限和现行授权，本文仅设计。

## 固定原RR的剩余消费者拟合：独立数学复核

2026-10-27，协调者的新构造：全160处固定历史UB1支持、alpha、b2、R256，24源×256行；在选中真实保存V2上对现有W2加ridge增量，目标为密集Y与原UB1完整主支路+RR之间的差。此构造不同于本轮12源CV的“以Y为目标拟合新消费者后重拟合RR”，不能继承其39%局部收益。以下为静态推导，未拟合或运行。

令 BF16 舍入记 \(\operatorname{fl}\)，\(F=\operatorname{fl}(\alpha A_S)\)，原主支路 \(B_0=\operatorname{fl}(FW_S^T+b_0)\)，原双线性支路 \(R_0=\operatorname{fl}(\operatorname{fl}(XP_1^T+p_1)P_2^T+p_2)\)，原输出 \(C_0=\operatorname{fl}(B_0+R_0)\)，原Y为保存的真实BF16 dense W2输出。采用实数域目标 \(E=\operatorname{float}(Y)-\operatorname{float}(C_0)\)，只拟合 \(F\Delta W^T+\Delta b\approx E\)，对 \(\Delta W\) 以原W2为零增量先验作事前冻结的 ridge fraction .01，bias不惩罚；保存/部署 \(\widehat W=\operatorname{fl}(W_S+\Delta W)\)、\(\widehat b=\operatorname{fl}(b_0+\Delta b)\)。线上严格执行 \(\widehat B=\operatorname{fl}(F\widehat W^T+\widehat b)\)、\(\widehat C=\operatorname{fl}(\widehat B+R_0)\)，之后原gate与BF16残差写回。原R0只计算和相加一次；原b2/alpha也只进入既有主支路一次。先拟合 E 而保留R0，避免直接用Y重拟合整个消费者再叠加原R0造成可预期的双补偿。

若用 \(E_{\rm real}=\operatorname{float}(Y)-[\operatorname{float}(B_0)+\operatorname{float}(R_0)]\)，它对应**未对分支相加结果舍入**的实数对象；\(E-E_{\rm real}=\operatorname{float}(B_0)+\operatorname{float}(R_0)-\operatorname{float}(C_0)\)。两种目标差恰是旧BF16相加舍入残差。实际线上目标是BF16输出，推荐前者，但线性模型仍不能完美模拟新W2/BF16 cast与两次加法的非线性舍入。重要反例：即使实数拟合 \(F\Delta W^T+\Delta b=E\)，增量因并入BF16权重而消失、或新的主线性/最终相加跨舍入区间，\(\widehat C\) 仍可不同于Y。必须基于真实最终 \(\widehat C\) 和 gate 观察训练/来源级误差及视频，而不能报告线性求解残差为部署误差。

**关键控制**：先在同输入同M=256、同缓存、原W2/b2/alpha及原RR的真实 BF16实现中重建 \(B_0,R_0,C_0\)，记录各分量和合成输出的SHA、各源与gate误差；按严格确定性和相同运行环境核对原UB1历史或本地原算子的有限控制。零增量必须经过同样bank打包、BF16 cast、runtime patch、R256及gate路径，逐位复现原UB1相同输入的输出；完整视频/latent对齐还需 same-source/seed、相同runtime/model/采样并验证可重现环境，若已有历史视频不位等先报告实际差异及同环境原UB1对照，不能把不匹配误认candidate效应。候选与neutral共享加载/模型/计算形状，直接比较全160实际覆盖及用户可见视频。固定 \(\lambda\) 和step/block、训练源、输入行与target精度；不凭已观测视频换目标或调参。

在每原生训练源M=256下计算基线需要每位置/来源至少主线性1次及RR两次，即 160×24×3=11520 次BF16 GEMM；基于已存post-Q2/Q1则不新增W1或QDQ。增量ridge的FP64投影/分解、一次或更多真实BF16候选预测、训练/持出存储与原生自由生成均需单独登记实际预算，24源×256行是6144行/位置而非6144独立场景；同源早期视频仅开发诊断。若资源/完整资产确实允许，先做冻结小子集的zero-delta、BF16边界和两位置ridge烟测能在构建160银行前发现明显符号、bias重复、训练/部署不一致或资源失败；它是技术停止检查，不应为了重复局部SSE排名延迟有条件的完整视频。未观测这些问题时不以小subset代替160 bank/自由轨迹。

## 完整视频WP0：历史对照与资产身份

只读核查已写私有 `.runtime/ub1_resume_20261027/UB1_CONSUMER_VIDEO_READINESS.json`，记录两source的精确input、视频、四步latent文件及其原始tensor SHA、当前图像SHA、原runtime contract与双profile资产状态；没有新生成、GPU或评测。formal20 input00犀牛（source882）与input10山坡住宅（source80）均 seed42，negative prompt为空。当前两张图分别SHA `0dca6cbfea5dfc5f61c9f30ddc01bbdad43be2023f80ee19911ae279b35e621a`、`4fc1b8c60af3b129dc1a687cdf29aa06a6242bd6db7ec0d838545ed2f0aeee37`，与两臂各自INPUT的已恢复图像provenance吻合；原记录明确 `historical_image_sha256=null`，历史使用图片的字节身份仍未证明。

历史dense及UB1四条MP4均可读，SHA匹配旧comparison manifest；各臂四个latent文件共16份可读且SHA匹配各自生成INPUT轨迹收据。复现neutral UB1的四latent及视频SHA可验证当前条件下历史UB1中性路径；若没有历史dense原始图片字节证据，应使用当前图像、相同source/seed新跑dense以建立严格配对，不以UB1重现推断dense图片身份。两例均为已观察正式20的开发测试，非盲来源。原contract含四step、81帧、16fps、target 832×480、SDPA、shift5、guide1、CFG关闭，实际视频832×464；各INPUT记录完整原contract而不能以输出分辨率反推target。原prompt、生成设置及所有SHA见私有收据。

当前历史 `generate_formal_suite.py` 和native source绑定共7项有6项哈希匹配；唯一不匹配的 `distill_assets.py` 历史版本在git中可取得，比较显示工作版本的差异为worker身份校验方式调整，不能仅由此声称运行完全一致。指定的两工作profile各12个model-view文件（两expert、VAE、T5、tokenizer等）已在实际worker上只读检查均可读、size吻合且与manifest blob同inode；monolith均可读且size吻合，未重哈希大模型或cache。GPU/python实际生成环境包版本及encoder确定性仍需在冻结时按执行环境检查，并由neutral四latent+视频严格回放检验；CPU系统python包缺失不代表实际生成环境缺失。

## 全160 fixed-RR bank builder独立审阅

审阅 builder SHA256 `10c660abd6bf59750dd7504f1eca78cb84f9bc41d7838cf641fdd126863a35bd`，math SHA256 `cf577f2f6eaa222261e061604077eb126c74d071e9eeca1b413a29d66aac365e`；小型CPU测试10项通过，无GPU或真实拟合。

**必须修复：位置冻结的原始摘要与实际alias缺少等值核对**。builder `107–115` 对 `private_assets` 的文件SHA本身和sidecar/payload/fingerprint自洽作检查，但不比较 draft protocol每位已有的 `main_npz_sha256`、`main_json_sha256`、`rr_npz_sha256`、`rr_json_sha256`、`shard_sha256` 与对应 `pp[pos[key]]`。若资产alias误指另一个有相同basename/同store fingerprint且自己sidecar自洽的chunk，检查可以接受与原冻结摘要不同的内容；对step1/2同一block原W2相同，现有 `load_coeff` 的 main分支没有 metadata step/block 比较，更不能依靠W2唯一性排除错误来源。建议逐pos逐项将实际绑定SHA与对应冻结SHA直接比较，shard同样核对；对两份独立合法但不同chunk进行alias交换的CPU测试应失败。完整 report/identity/basefits/cache链由协调者预检冻结回执另绑定，仍需该逐位置映射。

其余审阅范围：`baseline_for_source:88–92` 按同M=256实现主分支、RR两线性、BF16相加一次；`172–178` 目标为 `Y.float() - BF16(main+RR).float()`，候选为新主分支加原RR一次。中心化双域ridge的增量、自由截距及fold回原权重/bias在math `40–51` 与约定一致；候选训练误差实际经BF16新权重/线性和相加后计算，不能外推held-out/视频。四分片各10 block×4 step×24源，主+RR2+candidate主共3840 BF16线性/part、40 fit/40 packs；总15360线性/160 fit，现有逐操作计数逻辑相符。V1/V2/Y/门控/b2 shape和BF16检查、源的INPUT SHA及其160 scope展开、逐payload SHA/size于加载前核验已实现。whole64组与原W2/b2/RR张量身份另由 `load_coeff` 检查。实际原始report的scope绝对路径需在运行工作机同hf根下解析，现有构造严格拒绝其他根，应在冻结前以真实profile预检；这不是自动证据该路径必然不可用。

`fit_delta_consumer` 零误差保留原W/bias位模式的CPU测试只覆盖数学拟合；全pack的零增量、原生hook真实BF16输出和整视频neutral复现仍由后续运行器另行验证。历史24源全用于训练，没有独立持出；每次训练指标仅反映保存teacher前缀的已暴露源。静态审核不代替上游链冻结、真实GPU执行或完整视频验证。

**针对上条唯一必须修复项的复审**：builder SHA256 `0d8b05171560d14cf205660cb22a6b5825079043116285ccedeff6a055d13010`，测试 SHA256 `2baa128a3c66c6d2fd6a23b34af0ee37f17e518ba4d77d01c5df8cde92ccfa7e`。`validate_position_bindings` 逐位比较main/rr的npz、json及shard五个alias实际绑定SHA与冻结位置SHA，且在对应文件已验证后、运行前被调用；有效错误alias和sidecar交换反例测试已覆盖。两组CPU测试共12项通过。此前指出的alias映射缺口在本版本已解决；完整上游资产链与具体protocol仍由协调者冻结的独立预检回执负责，本次未运行GPU或真实拟合。
