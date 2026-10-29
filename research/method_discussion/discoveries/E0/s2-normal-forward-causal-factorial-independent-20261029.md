# S2 原BF16正常后缀：局部作用、状态传播与τ0的区分性实验

2026-10-29 · `/root/s2_theory_design_independent` · ALG/SRC/OPEN，仅设计，不批准、注册或运行。V2/V3完成与否、位点排序、视频和系统结果均以实际各自RUN/EXIT为准；不从尚未结束的160位置报告挑高响应层。实验固定同checkpoint、原W1/GELU、现有value10/dcac10 rank960因子及原W2/bias，不重新拟合。

## 三种不同预测

- **H-L 单点局部作用主导**：在已形成的学生状态上，后段单点B的原W2恢复，对下游flow/视频或source级相对Dense误差造成稳定改变；在同等干预预算下，相同B的作用对先前A是否执行不太敏感。反例是B在自然学生前缀及不同前缀下响应差异大，或分配给前面A的预算在后续造成更大作用。此机制不要求整个网络只有一个必要点，结论限定被测点、方向和扰动量。
- **H-P 上游状态/闭环传播主导**：对较早A只在该处执行一次D(h_S)恢复，后续正常student自由演化；如果A的作用跨attention、flow、下一去噪步持续，且使后段B的局部恢复效应发生方向/量级改变，则支持前缀依赖及传播。若A基本无跨层/跨步响应、B效应不依赖A，削弱所测A→B闭环机制。A本身是局部W2干预，前缀效应经后续传播产生；不能从P大或factorial交互自动称为已识别的唯一路径。
- **H-0 条件槽特异**：在同step/block、同样选取的52个xy上，τ0对其他时间槽attention后状态及flow的影响，在实际postwrite扰动量级、方向与图像条件敏感对照下，稳定超过τ10及预定随机普通τ；否则削弱所测条件槽特异论。τ0更大可来自更大局部误差、绝对时间位置、边界感受野或幅度，不自动证明图像condition特异作用。

三者可并存；以下是区分在具体位置和source上的相对作用，非互斥分类器。

## 源码与干预点

[SRC] 实际BF16图为 [self-attention → cross-attention残差 → norm2/shift/scale → W1+b1 → tanh GELU → W2+b2 → gate → BF16 residual postwrite](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py:171)，下一block通过self-attention消费该state。四步high/high/low/low每步40层；需每次取对应expert、step及block的原W2与该family已有factor；不把相邻expert当同一函数。[I2V condition构造](../../../../.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py:627)使τ0的4个mask通道为1、其他τ为0；VAE条件及噪声在 [patch embedding之前拼接](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py:107)，时间patch stride 1。当前A14B蒸馏 [step_post](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py:80)更新全latent，不得把另一模型的首槽硬夹持套用在此。

[ALG] 在一个学生当前正常状态、被选行`R`，只更改原读出：`D_R=h_S[R] W2+b2`，`Q_R=(h_S[R] L)R_factor+b2`；其余行保持当前学生`Q`，原g和r及原生先BF16乘再BF16加的时点不改。干预变化是 `U_D−U_Q`，实际逐元素通过native写回测得。此处`h_S`是每个自由干预臂自身运行在目标点形成的状态；既不替换为Dense hidden，也不把teacher视频帧值作为在线信号。D额外GEMM是付费oracle/因果工具，原W1和候选因子已执行，**不构成省算候选**。Dense成对完整视频及其状态只作评价/参照；取未来视频光流作ROI是后验oracle，不能当免费在线输入。

## 选点及采集先决条件

1. 等对应V2/V3真实全160位置、native anchor bridge、source级输入/输出及EXIT齐全；在一个预声明开发集合上按规定函数/step/expert选A、B，不使用验收源结果或视频指标扫选。可用规则例：A为同expert中前半层的稳定非零真实typed ΔU点，B为**同step后续层**的对应位置；step传播问题可另以B选下一step的早层并保留expert身份。各位点均先用固定的跨source中位数与低误差/不同位点对照，无法找到多源可比位置即保留未满足，不强行选最高单个source。上述只是规则模板，须先注册阈值、tie-break和选点/确认分源。
2. 两张已有暴露pilot 898/96可验证hook、相同seed的中性及运行链路；独立性/机制趋势在另冻的≥12个**完成图簇身份、无历史/保留确认重叠核验**的开发图上分析。已有新18静图仅为provisional，不能直接声称12个最终合格；动作/相机/遮挡分类只从真实Dense视频及人工核验取得。冻结外部确认图像、独立seed及source-level分析另立scope。
3. 在每个source，所有arm复用同输入图、prompt、初始seed、4step scheduler、模型/数值recipe和同一factor SHA；不同臂使用重置、独立同seed generator（或锁定相同噪声draws），保留初始/每步latent SHA与实际draw记录。不得在两臂共用被前一臂推进的generator。source是独立单位，52行×层×时间不是额外source。

## 一组最小有向干预：固定时间槽上的2×2

同一个预定τ、相同52个去重xy，选早A及后B；从相同学生起点完整跑4step、其余159/158个site正常使用该family，至少用两源验证链路后扩源。四臂复用baseline student，以A/B开关为 `(0,0),(1,0),(0,1),(1,1)`：

| 臂 | A原W2恢复 | B原W2恢复 | 区别 |
|---|---|---|---|
| S00 | 否 | 否 | 自由学生；与原student正常运行逐byte验证。 |
| S10 | 是 | 否 | 测早期扰动经过后续普通层与调度后的作用。 |
| S01 | 否 | 是 | 测既定自然学生prefix上的单点局部效应。 |
| S11 | 是 | 是 | B由S11自身新鲜h计算D/Q，测早期干预改变B恢复响应。 |

额外no-op hook arm应与S00同seed所有关键latent及输出逐byte一致；有hook但无替换也必须执行相同收集/同步路径。S00不重算可复用已有完整视频仅在同checkpoint/数值recipe/seed且结果中间状态验证完全一致时；如果无相同条件的state记录则新注册生成。S11的B输入与S01的B输入自然不同，这正是对前缀依赖的观测，不应把两者local值强行当同一个向量。

预选读出 `z` 包括B写回后同τ与其余τ下一block self-attn输出/残差、同step最终flow、下一step输入与block0 state、最终latent及视频评价。定义 `ΔB|A0=z01−z00`，`ΔB|A1=z11−z10`，`J=(z11−z10)−(z01−z00)`。向量报告范数、方向和有符号投影；若用相对Dense的非线性误差`E(z,D)`再算factorial差，需要另外说明该统计包含读出非线性。`J`是恢复操作与前缀变化的总体交互，包括各臂B的h/方向变化、非线性与所有后缀，不是把自然L/P/I各项识别为单独因果机制。只有沿A→下一层跨τ→flow→next-step的实测先后响应，才对传播路径提供支持；仅末视频变化不定位该路径。

## τ0及非定向对照（第二小组）

在同一预定step/block、同一52xy和同一family/seed上分别执行S00、B只恢复τ0、B只恢复τ10、B只恢复一个在执行前固定随机τ∈1..20（排除10）四臂；每臂相同被恢复row数、相同位置预算。测每臂真实ΔU范数、方向与变化占比，以及**其他τ**的下一层attention/state响应、flow、nextstep和最终视频；不能用全局ΔU差异归一化后当因果结论。52行是从1508个τ平面空间点选的局部区，阴性只限此ROI。

为检验是否仅由注入能量/时间位置决定，在另一个预冻结小范围加入等后写回范数的随机方向或原恢复向量空间打乱/反号对照，并分别记录实际BF16 ΔU；用缩放插值匹配norm时，它不是完整原W2恢复，且BF16台阶可能无法精确匹配，匹配失败标不可比。若允许同一预算，再加入Dense轨迹上同位置、同原recipe的Q(h_D)−D(h_D)反向压缩扰动，使用Dense自己当前h；该臂用于响应对称性/方向对照，**并非拿Dense hidden修复student**。若τ0优势在equal-ΔU与图像条件替换控制下消失，优先考虑位置/量级而非condition特异机制。反向扰动不等于原受影响的整个自由轨迹，非线性不要求正反一致。

若想同时测τ0和A×B交互，预先指定一个主检验并扩展为`A∈{0,1}`×`B∈{none,τ0,τ10,随机τ}`八臂；这些不是两个相互独立的样本。第一组已测同一个B τ时，重复S00及相同臂可复用，但仅在代码、source、seed、hash和中间状态完全相同的实际receipt下去重。单family的探索最多8个配对student arm，加no-op与若选择的Dense反向控制；完整评估另需配对Dense，已有同条件输出可按真实receipt复用。采用另一family是新的成对研究条件，计算量翻倍而非新独立source。

## 原生小M边界与退出准则

D(h_S)被恢复的52行是M52额外原W2读出，数值可能与真实M31668 GEMM不同；不能把小M四角误差当全M native exact。按每个source×被选expert/block和该臂**自己的h_S**，在另冻结的前向对照执行同W2/bias的full-M输出，比较被选行M52与M31668的Y、gate后、真实typed postwrite差、相对被恢复ΔU及其signed cross；单个τ的52行本身不支持时间频谱，若需频带必须采同xy的21τ并按所用M形状另付费核对。若桥误差与干预差同量级，停止该数值定位或改用当次实际full-M读出，不调整结论。全M dense读出约 `31668×13824×5120≈2.24×10^12 MAC/次`，仅Y BF16约324 MB；M52约 `3.68×10^9 MAC/次`，还须记输入获取、scatter、同步、observer、额外权重和完整视频wall/显存。这些是逻辑预算而非已测资源，也不是动态省算证明。

执行顺序：先实现noop及原生M桥，实际source×seed neutral通过；再pilot一次A/B与τ对照观察下一block、flow、next-step；若hook干扰、seed不等、postwrite不还原、源状态/权重SHA不符、full-M桥无法区分干预，明确失败并保留receipt；最后分source采样扩展与完整自由视频多维质量。输入及中间数据、结果大小、每臂实际GPU/CPU/IO、一次调用的W2次数、额外时延与session_max_GPUs=2都要在独立ACTIVE scope冻结，任何小型因果实验不自动升级为全160位点质量或部署验收。

## 先例与证据边界

已有G14 [TokenFlow C11](../../../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)是输入视频特征对应传播，[OmniCache C13](../../../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)讨论去噪step缓存与采样轨迹误差；二者提醒信息可见时点与后缀传播，但不提供该Wan原W2局部恢复的因果实测。沿用已登记的原文/局部源码阅读深度，不以本文新增深读或修改图。本方案是真实介入构造与可证伪判别，所有效果与资源数值均未运行、未判定。
