# FFNDS 的附加场景：实时、交互式流式视频生成

2026-10-26 · **研究场景扩展/设计分析；未执行新模型/streaming测试** · 适用范围：一个附加研究目标，不修改 [E0唯一主任务](../20261025/E0_FFN_FIRST_TASK_20261025.md) 的原本四步Wan I2V质量-真实FFN计算目标。

## 1. 先把“实时”定义成可比较的系统性质

原Wan2.2四步、同一整段81帧的bidirectional/full-sequence计算，不会因为一个FFN局部算子提速就自动变成可以连续显示、响应用户输入的生成器。考虑**两种并列场景**，而非一个项目做两个互相替代的指标：

| 场景 | 可检验任务/主指标 | 关联的FFN科学问题 |
| --- | --- | --- |
| **A 固定段离线I2V**（主研究） | 相同prompt/seed，完整生成latency、真实FFN边界、全部模型壁钟、画质相对dense、物体运动与背景闪烁 | 如何用真实可运行且省W1/W2的表示，在整个四步轨迹中维持下游功能 |
| **B 实时交互/流式I2V/V2V**（新增） | TTFF（首可显示帧）、steady-state frame/chunk cadence、p50/p95/p99延迟/抖动、missed deadlines、交互输入到可见响应、持续内存、长流漂移/画质 | 在给定历史与deadline下，哪些FFN函数需要重新计算，哪些能可信复用/压缩，并避免自生成上下文污染 |

**示例目标而非承诺**：对24FPS显示，帧截止间隔约41.67ms；30FPS约33.33ms。若系统按c帧chunk输出，则需报告chunk完成时间、帧交付时间及真正的首帧时间；平均吞吐>=目标FPS是**必要但不充分**条件。缓存预热、首段context、decoder、传输与交互反馈也必须计入SLO。明确模型输入帧、可视输出帧、去噪step、模型latent chunk这四种“帧/步”不是同一计数单位。

对于稳定实时系统，使用定义：
- TTFF：用户新生成请求（或明确分开的首段已准备状态）到**第一张可显示解码帧**的时间；同时记录cold与warm，不能将预计算/权重装载永久置于免费区。
- Inter-frame/chunk latency与deadline miss：用输出显示时间戳和真正负载到达时间测量分布；`effective_FPS=N/T_whole`不得冒充`stream_FPS`。
- 响应延迟：例如prompt/motion/camera指令在t时变更，测量最新条件何时首次影响可见输出以及前后帧连续性；记录旧KV/状态重置策略。
- 实时内存和持续性：KV/rolling-window/FFN cache、resident weights、attention sink、allocator、随时间的增长或重置；在目标GPU/分辨率/帧数明确报告peak与steady state。
- 质量：空间感知/细节/主题运动、背景亮度抖动、运动相干、长期identity/geometry/semantic drift、突然提示切换后的污染；以dense或可靠streaming teacher在同输入/可比历史协议下对照。

## 2. 原生FFNDS能迁移多少：计算算术约束和架构鸿沟

最新[六条完整原生对照](E0_RECONSTRUCTED_VIDEOS.md)：81帧整段，强resident dense热态`T=50.750s, F=9.707s`；affine54近似`T=45.313s, F=4.449s`。affine54 **FFN边界2.182×、整段1.120×，且画质失败**。原始`F`包含FFN入口norm/modulation，不是纯主GEMM。

- 离线81帧/45.313s约1.79帧/秒只表示**整段均摊**，不是每帧按时交付。原一次整段推理在视频保存前已经完成全部去噪，不是producer-consumer streaming。
- 原强dense的非FFN耗时约`R=50.750−9.707=41.043s`。即使完全消除其FFN边界，若沿用同执行结构，81帧仍至少约41s，整段墙钟最多约1.2365×。这并非新流式架构的理论下界。
- 24FPS理想整段预算`81/24≈3.375s`，与现有50.750s的同结构强dense相差约15×；即使理论上消去FFN边界也还差约12×。这是**说明不可只做FFN优化的条件算术**，不是证明所有架构均不可实时。
- 当前全段bidirectional attention依赖整个序列，并不提供一条自然的、按帧输出且只看过去的执行路径。若希望真正在线流式，必须确定**模型是否改用 causal/chunked/rolling执行，历史缓存及prompt变更语义如何定义**。FFN优化是其中一层机制，不能替代架构/调度工作。

**物理成本区分**：这次whole94/affine54有真实减少投影宽度/理论MAC并实际加速，但仍保留完整FFN所有160处，且两方法质量失败。此前full resident dense将单卡热态耗时约75→50s却未减少主FFN MAC、物理显存逼近80GB；实时场景经常需要KV和latent窗口，故**权重常驻与rolling缓存发生显存竞争**，需要以硬件预算形成Pareto比较；不可把两个优化假定为完全可无成本相乘。

## 3. 核心科学价值：从“同step省一个FFN”转到“流式条件状态下保存功能”

若生成chunk `n` 时缓存历史 `H_{n-1}`，有效FFN映射一般取决于 `(x_{n,t},t,e,layer,conditioning,H_{n-1})` 及模型执行上下文，不自动享有Wan四步`high,high,low,low`提供的**同一expert两次合法复用**。滚动窗口状态、视频运动/scene变化、交互提示变更可能影响任何静态保留集或低秩近似。

优先考虑的**竞争机制，不是算法路线表**：
- **M1 条件函数驻留**：自生成历史中的某些场景/外观FFN输出方向对长期一致性更重要；其余局部变化可能适合条件重算。判别：跨chunk固定功能基是否随着运动/scene cut/提示变化失效？与只保留hidden topk相比，在相同真实执行预算下是否减少漂移？
- **M2 信息价值随deadline和场景改变**：某种低成本W1前代理可在简单静态场景复用函数，却在运动/提示切换/边界帧失效。判别：对同成本随机/定期刷新与状态变化驱动刷新比较；计入observer、deadline miss和cache sync费用。
- **M3 有符号误差跨流积累**：局部输出SSE较小不代表`H_n`可靠，也不保证下一段质量；负误差方向可随反复使用形成漂移。判别：保持本次输入相同，比较teacher历史与自生成历史的同算子FFN差、下一chunk响应和长期外观稳定性。源依据：[受扰前缀/后缀/轨迹恒等式](../method_discussion/discoveries/E0/transported-function-and-trajectory-target-20261026.md)。
- **M4 端到端调度/显存支配**：若attention KV/window、VAE decoder、I/O、weight residency或同步决定p99 deadline，FFN省算未必解决SLO。判别：按frame/chunk拆分stream latency/内存，而不是仅报告`T_FFN`和总均值。

## 4. 最强近邻：机制及不可直接借用的成绩

| 原始工作 | 已公开机制/研究问题 | 对FFNDS的有限启发 |
| --- | --- | --- |
| [StreamDiffusionV2（MLSys 2026）](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) | 训练自由的streaming pipeline，逐帧SLO、首帧、rolling KV、SLO-aware batching/多阶段pipeline；4×H100上14B模型报告58.28FPS及<0.5s首帧（作者条件） | 实时主要是**时延分布、切分与调度问题**，不能把其多卡FPS搬到单A100 Wan或归功于FFN |
| [Self Forcing（NeurIPS 2025）](https://proceedings.nips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) | AR rollout条件于自生成历史，rolling KV和训练时视频级监督以减少exposure bias | 研究FFN近似的`teacher-input`与`student-input`迁移，不能仅局部同输入Q2/SSE |
| [CausVid（CVPR 2025）](https://causvid.github.io/) | bidirectional→autoregressive因果结构，少步蒸馏与KV缓存 | 模型形态改变会使FFN激活/step专家分布变化，原Wan校准规则需重新验证 |
| [Rolling Forcing（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html) | 联合多帧progressively noisy滚动去噪、attention sink和长视频误差控制 | 流式长期相干与硬per-frame计算预算产生不同的FFN功能价值 |
| [LongLive（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html) | causal frame-level生成，KV-recache/动态提示、更长历史训练与attention sink | 动态条件切换时，FFN缓存/selector的有效性与失效告警需要明确 |
| [Causal Forcing（ICML 2026）](https://proceedings.mlr.press/v306/zhu26bf.html) | 分析AR蒸馏的双向teacher与因果student架构差距及可辨识性问题 | 如果换成真正流式teacher/student，不能把离线双向模型的功能近似直接当在线student有效先验 |

**创新性空间的精确边界**：KV缓存、autoregressive distillation、rolling窗口、SLO-aware scheduler已有成熟先例。FFNDS可有价值的新研究应是**真实视频FFN功能/算子在流式条件、响应时间与长程质量中的动态省算规律**，并与上述原始系统机制做对照，而非仅把现成KV或cache技术重命名。

## 5. Graph-Agent协作与研究准入

从现有G0–G5的F03/F04/F06/F08/F09、F12/F13/G21/D09、P12/P17/P22读取准确计算流和视频质量失败证据。新增G6是**需求/反事实架构层**而非本repo已实现的streaming runtime：R01离线baseline→R02实时deadline→R03因果缓存合法性→R04 FFN条件复用/失效机制→R05持续时间/视频消费者。由不同Agent分别挑战“硬实时被FFN主导”“原4-step history可跨chunk复用”“低局部SSE确保长期稳定”三个假说；允许任何agent提出更有力的新节点/模型。

**最小可执行研究前置（未获执行授权）**：先确定确切`streaming`模型/代码和真实causal window/rolling state与参考硬件，测逐帧/首帧SLO基线，再研究同预算/同history的FFN省算。没有原生流式执行源码时，只完成结构假说、数学时间账本和文献比较；**不能伪造fps、TTFF、p99或历史缓存观察结果**。

**任务权限**：本文件不改变E0研究阶段，不选定新算子，不将视频失败原型晋级，不自动执行GPU/模型/训练，也不以实时生成取代原始Wan固定段四步I2V目标。运行层面的新硬件、模型、缓存或视频生成范围须有明确授权及冻结协议。
