# S2：首条件槽影响与多层闭环漂移的可区分干预

2026-10-29 · 独立 `/root/s2_theory_design_independent` · SRC/ALG/OPEN，与真实视频或部署收益分离。未执行模型/GPU/大张量；未改已冻结文件或ACTIVE。数据为私有未完成replay的执行顺序前缀，仅用于确定下一判别；必须等待全部RUN/EXIT和全160位置的正式分析。

## 真实条件图与槽身份

[SRC] 本pilot protocol的 `runtime_overlay` 记 `model_cls=wan2.2_moe_distill`、`task=i2v`、`use_image_encoder=false`。原生路径以 [WanRunner.get_vae_encoder_output](../../../../.runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py:627) 将原图按尺寸插值，拼接第1帧图像与80个零视频帧，经VAE编码；对应二值条件标志原始帧0为1、其余为0，把帧0重复4次并按时间4重排，得到4个mask通道，在21个latent时间槽上τ0四个通道均1，τ1–20均0，再与VAE编码通道拼接。**不能因原视频后80帧为零就断言编码的τ>0都为零**：VAE时序卷积、缓存和偏置允许传播及非零输出。相关作者 [Wan2.2 I2V实现](https://github.com/Wan-Video/Wan2.2/blob/main/wan/image2video.py) 同样明确生成首帧condition、四重排mask并拼接。

[SRC] [pre_infer](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py:92) 在每个去噪步把condition通道与本步噪声latent按通道拼接，再调用原patch embedding；[pre_weights](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/weights/pre_weights.py:10) 的stride为(1,2,2)，其τ0 token承载当前噪声状态、首帧VAE条件和显式mask标志。21×29×52是patch后时空网格；每τ1508 token，52个是本pilot ROI的子集。序列flatten与三轴RoPE保留位置身份。use_image_encoder=false，故本scope不是通过CLIP image cross-attention专用K/V路径输入；text cross-attention仍有。自注意力使各token能读取跨τ其他token，具体权重、方向和效应需测。条件是输入通道，不代表某个后层FFN坐标仍单独承载原图信息。

[SRC] 不能把另一个`model_cls=wan2.2`的conditioned latent prefix规则套到此A14B：此scope用 [Wan22StepDistillScheduler](../../../../.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py:66)，其`prepare`走初始噪声，`step_post`按flow更新，无调用该另一条scheduler的首latent硬覆盖。本段据冻结config和源码路径，不假称逐bit runtime动态采样已证实；调度类、输入mask/latent与gate应由计划中的原生receipt确认。τ0不是必然干净或不可更新的latent。图像condition在每个模型调用重新拼入，而网络内部τ0 state依然受状态与其他token影响。

## 已存部分结果与竞争假说

[OBS-PARTIAL] `.runtime/S2_INCOMPLETE_INTERIM_CENTRAL_AC_20261029.json` 包含95个source×family×step×block标量，执行顺序截断，仅block0–5，不齐全；中心ROI的τ1–20 AC输入不是τ0。用户给出的≈0.077局部、≈0.35 prefix、≈0.05交互是该未完成集合的概要，并非全层/全来源估计；总差含有符号交叉，平方不能相加。该局部L/P/I属于gate后四角状态对照，不是因果归因。具体反例：step0/block0两来源prefix均0；step1/block0 value10 source898为0.0218808，source96为0.0008698，约25.2倍（两者各自归一化，绝对能量需另看）。这是跨步状态已变化的可观测差异，不足以判明其来源或预测最终视频。DCAC10训练AC相对SSE改善而τ0与DC变差，是既有[拟合报告](../../../20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)的不同数据对象；两种学生完整视频均运动偏低也是[VIDEO报告](../../../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)的独立观察。中间差异与视频退化间的桥仍OPEN。

| 假说 | 核心机制及区别性预测 | 反例/控制 |
|---|---|---|
| H0 图像条件槽误差传播 | τ0的FFN压缩扰动进入后续self-attention及step后缀，改变其他τ及后续去噪；等坐标、层/step/row预算与可比有效注入范数下，恢复τ0相对于τ10和随机τ，预期更改善其他τ的state/flow和图像身份及运动指标 | 同注入范数时τ0无特异性、作用仅本槽或不经attention而直接尾部输出，则削弱具体传播论；若位置本身对全局attention更敏感，也不能仅凭τ0>τ10归因于图片condition。 |
| H1 普遍多层闭环状态漂移 | 许多τ、多个block/step的有符号局部扰动经残差、注意力和scheduler组合，state随步改变；source898/96差异及后续累积或由轨迹敏感性而来。多个分散槽/层等预算恢复可与τ0等效或更好，首层step1状态差可由step0整体变化产生 | 若τ0的特异效应在匹配误差方向/作用范数及注入时点后仍稳定大于控制，且短后缀跨τ响应明确，则单纯均匀漂移不足；两种机制可同时存在。 |
| H2 时间位置及decoder/scheduler敏感性 | τ0位于边界、时间位置或后续video decoder感受野不同，即使原图condition不特殊，任何同幅度扰动也可能有不同效果 | 比较匹配norm的随机/对照方向、Dense反向扰动；追踪相同位置的局部update→下一层其他τ→flow→step末，避免将decoder边界效应误标attention条件传播。 |

[ALG] 对同一个student历史`h_S, g_S, r_S`，由原W2产生D(h_S)、候选Q(h_S)，仅在已选`(step,block,τ,xy)`用D替换Q，后写回的实际影响为 `round_BF16(r_S+round_BF16(g_S·D(h_S)))−round_BF16(r_S+round_BF16(g_S·Q(h_S)))`；这只移除该位置/当前学生状态的局部算子差。用D(h_D)代替是替换输入前缀+局部算子的复合干预，不能叫局部恢复。dense老师当前自由轨迹的值、预测的真实full-M W2都属付费oracle；同τ位置相同不是同物体轨迹。因子/原W2、bias、原g/r及native typed rounding均需保持。

## 可实施、有辨识力的候选实验（另行登记）

1. **被动选点与可比性前置**。等实际pilot全收齐，固定source×family×step×block，同步报τ0和τ10的gY与typed-u局部恢复向量、能量和gate、各自small-M/full-M偏差，及残差比、场景位置。首尾/早中层预先分层；以跨来源规则而非单个898挑site；低误差位置配对控制。若τ0或对照注入为0，则该对无法对比方向及归一化响应，应报告不可辨而不是人为调大。
2. **有限one-shot正常后缀干预**。在相同source/seed、同一自由student运行、相同已选step和早中block，在以下三个随机化且固定配对arm上各对同一52个真实xy（或更小、同样数目）的W2执行D(h_S)恢复：τ0、τ10、预先抽取τ1–20的一个τ；其余FFN正常执行，支持另加无干预、hook调用但原Y、原W2输出不变的byte-neutral controls。每个arm改同一数量row和site，但输入h和真实local差不同，实际额外W2、scatter、同步与wall必须测；52行仅占1508个空间点，不能把局部阴性推广整个τ0。记录原生g/r/u、下一block其他τ实际state/attention output、同step最终flow、每步latents及完整视频，以相同现存cache/recipe和额外采集scope执行。τ0与τ10读取的是各自h_S，不能复用同一个Dense特征。
3. **剥离天然误差幅度**。前述自然恢复的注入方向和大小均由候选决定，等row不等实际有效作用。测每臂真实Δu、对照量级分布，并在另注册的机制probe中配对不同τ相同实测postwrite Δu范数和局部方向匹配/随机方向，例如施加按预冻结α缩放的`D(h_S)−Q(h_S)`加法，保留BF16实际写回，另加符号反转、相等范数空间洗牌；缩放不再是完整的原W2替换，务必单列。若无法匹配不宜作位置强弱比较。以自身reference量度向量方向及跨τ响应，不只看总SSE。若只有τ0自然差更大但等幅局部敏感度相近，支持“错得更多”，不是τ0具有特定传播优势。
4. **反向对称、跨步与分散控制**。在Dense normal flow同槽注入Q(h_D)−D(h_D)，匹配预算并注意非线性：Dense→压缩与student→恢复反向响应不必相等；非对称即报告。固定后缀窗口比较step0→step1 block0的h/r/flow，及step0内下一层跨τ响应；可另设总row×site相同但跨多个τ/层分配恢复，比较单槽集中与分散多层。后者改变干预结构，需计相同算子调用次数、rank及开销。最后层因无下一个block attention，对跨τ传播的观测必须到flow/后续step，不能以它单独证明下一block机制。
5. **条件特异性与负对照**。τ0胜过τ10仍不能区分首帧条件与时间边界/更大误差；预冻结同source图像条件敏感比较（改变图像或同图特定可控扰动时同seed/步骤完整配对），只在新、合理可解释的输入分布内进行，保留text与scheduler；不同图之间不把相同位置视为同一语义对象。比较图像条件敏感输出方向与误差方向的响应，若结果只随τ位置而不随condition变化，偏向位置机制。明确任何篡改原mask/condition的实验可能脱离训练分布，不能单独当方法的质量证据。

优先采用单位置即时后缀来确认可测性，再按来源扩充并注册完整视频，固定来源级独立观察、同checkpoint/数值与seed；不能将观察到的响应直接归因为最终运动。质量桥单独比较主体/相机/背景/遮挡、时间与图像语义、连续运动和完整系统成本。真正决定性预测是：**匹配有效注入大小/方向与source后，τ0的跨τ后缀响应是否具有稳定、可被图像条件操纵的特异优势**，而不仅是局部误差更大或视觉上先于其他帧。

## 近先例与新颖性边界

G14已有[TokenFlow C11](../../../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)使用已有视频的跨帧特征对应，讨论的是编辑特征传播，不能作为τ0 W2恢复因果证明；[OmniCache C13](../../../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)以去噪轨迹位置与缓存误差修正提出跨step机制，提醒局部相似不等于最终后果，不能将其缓存误差公式等同本实验的局部L/P/I。两者沿用前次已登记的确切论文/源码阅读深度；本次并未新增论文深读或修改G14。以上控制是机制定位与可证伪设计，不是在线可部署的新加速方法。
