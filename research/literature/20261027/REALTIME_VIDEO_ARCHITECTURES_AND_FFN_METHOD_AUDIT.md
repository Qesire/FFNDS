# 实时视频生成：一般视频模型如何转为流式计算，以及 FFN 的物理需求（2026-10-27）

> **并行文献研究线 G10**。本研究由用户本轮明确要求恢复；仅恢复真实论文/公开源码/数学与计算图整理。原来暂停的 GPU、CPU 原生模型试验和视频生成继续暂停，权威边界见 `research/ACTIVE_RESEARCH.json`。外部论文或作者系统均是 **LIT**，本项目 Wan2.2-I2V-A14B D-Q HiF4 的 G0–G5 为 **SRC/NATIVE**。来源等级及源码 blob 见 [来源账本](REALTIME_VIDEO_SOURCE_LEDGER.json)。

## 1. 先分清生成“整段很快”与真正“流式”

**Offline clip**：有全部潜在视频token \(X_{1:T}\)，同一次 DiT 推理各步可进行整个序列上的双向自注意力：
\[
X^{(k+1)}_{1:T}=X^{(k)}_{1:T}+\Delta \sigma_k\,V_\theta(X^{(k)}_{1:T},\sigma_k,c).
\]
在没有改写时间依赖时，早期帧依赖未来帧，不能从该图推出实时显示的第一帧。

**Causal/rolling video**：给已提交的历史 \(H_{<n}\)，在时刻 n 只解当前 \(C_n\) 的若干潜在帧，经有限去噪 \(k=1,\ldots,K\) 产生 \(Z_n\)，更新KV并解码输出：
\[
\hat Z_n=\mathrm{Denoise}_{K,\phi}(\epsilon_n,t,c_n\,\mid\,\mathrm{KV}(H_{<n})),\qquad H_{\le n}\leftarrow H_{<n}\oplus\hat Z_n.
\]
此式为跨论文的研究抽象，具体作者可能同时联合去噪数帧（Rolling Forcing），或者输入V2V参考帧（StreamDiffusionV2），不是所有模型均严格一帧一步。

指标必须分开：TTFF (请求→首可显示帧)、steady frame cadence、p50/p95/p99期限命中、prompt/action变化→首次可见反应、长时质量漂移、总GPU显存、每生成帧/每chunk实际耗时、模型GEMM占用和VAE/通信开销。视频完整生成 \(F/T_{\rm clip}\) 的平均FPS不可替代按时交付。高吞吐的多GPU结果也不等同单流端到端延迟。

## 2. 原始教师模型与因果学生：到底是不是“一般视频模型”？

### 路径A：双向视频DiT → 因果自回归学生 → 少步与长时自生成训练

- **L55 CausVid**（[CVPR2025正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html)、[项目](https://causvid.github.io/)、[作者源码](https://github.com/tianweiy/CausVid)）：从预训练双向扩散teacher得到 causal student；先由teacher ODE解做初始化，再 asymmetric distribution matching distillation (DMD) 压到4步，KV caching使每次预测不必重算整段历史。作者公开9.4 FPS、1.3s TTFF为原实验条件下指标。**这是一种对普通视频模型的训练后改造**，而非把原bidirectional的attention flag简单改成causal就能直接达到原质量。
- **L56 Self Forcing**（[NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html)、[源码](https://github.com/guandeh17/Self-Forcing)）：训练时执行有限步 autoregressive rollout，以学生自身已生成的帧/KV为下一帧上下文，再进行视频级分布监督；解决teacher GT-context与部署student-generated context的分布差。这与本项目[Student前缀 vs Dense前缀](../../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)存在**结构相似性**，但本项目是离线两轨全片推理，不是持续流式历史。
- **L57 Rolling Forcing**（[ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html)、[源码](https://github.com/TencentARC/RollingForcing)）：在滚动窗口**联合**去噪多帧（噪声水平递变），保留attention sink；窗口蒸馏从自生成context减轻曝光偏差。不能把该作者的方法画成简单逐帧严格串行生成。
- **L58 LongLive 1.0**（[ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/91a1610c6ed9e02d33f826b46f472b92-Abstract-Conference.html)、[官方项目](https://research.nvidia.com/labs/eai/publication/longlive/)、[源码](https://github.com/NVlabs/LongLive/tree/main/LongLive1.0)）：frame-level AR、fixed rolling window + frame sink、prompt KV recache和长视频训练；实现持续生成与互动条件版本更新。作者称一张H100约20.7FPS，具体模型1.3B/原论文实验协议。
- **L62 MotionStream**（[ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0cece806cd3d1dfad4a893f016ad3d7d-Abstract-Conference.html)）：T2V teacher加运动条件，再通过Self Forcing/DMD生成因果student；rolling sink和window使速度相对稳定。论文单卡最高29FPS，为作者不同模型/实验条件。

### 路径B：在因果模型上做“可持续实时服务”系统

- **L59 StreamDiffusionV2**（[MLSys2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html)、[项目方法全文](https://streamdiffusionv2.github.io/)、[官方源码](https://github.com/chenfengxu714/StreamDiffusionV2)）：不是重新从零训练完整生成模型的方案，而是以 causal DiT（基于CausVid）与 rolling KV、sink、短块 StreamVAE、motion-aware noise、跨 denoise/layer 多卡流水线、SLO scheduler构成streaming v2v/T2V系统。用户请求帧到来时的生存周期是 **capture → encode → noise/DiT → decode → delivery**。该论文4 H100、14B 58.28FPS、首帧<0.5s为系统条件；不是本FFNDS显卡性能。
- **L63 StreamDiT**（[CVPR2026正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Kodaira_StreamDiT_Real-Time_Streaming_Text-to-Video_Generation_CVPR_2026_paper.html)）：多步扩散的buffer与chunk划分/partition-aware distillation，作者16FPS/512p/单GPU。当前为正式摘要一级；尚未核其原源码和全部方法方程。
- **L64 FAST-AR**（[ICML2026](https://proceedings.mlr.press/v306/samuel26a.html)）：TempCache对历史KV做时间压缩，AnnSA/AnnCA检索选择相关self/cross attention键；作者报告长时缓存和attention峰值近似恒定、整网最多5–10倍；其核心动作发生于 **attention** 而非W1/W2。随着attention被优化，FFN占总算量的相对份额反而可能增大。

### 路径C：直接重构时间因果与去噪的计算职责

- **L60 Separable Causal Diffusion / SCD**（[CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html)、[作者源码](https://github.com/xingjian-bai/sparse-causal-diffusion)）：作者的原始实测观察为**早中层跨 denoising-step 特征相似、深层cross-frame attention稀疏**。因此用 once-per-frame causal temporal encoder 获得条件表征，再用多步 frame renderer 做细节。FFN的启发来自时间/去噪次数的**因子分离**，不是预设隐藏通道可删。已直接核查作者 `scd/models/scd_model.py` 中单个 SCDTransformerBlock 顺序：AdaNorm→attention→gate残差→AdaNorm→`FeedForward(...,activation_fn='gelu-approximate')`→gate残差；`SCDAttnProcessor` 有历史K/V拼接与缓存；该作者具体结构/任务与当前Wan2.2 G1不同。公开repo README把预训练权重/标准推理列于注释状态；不得把源码存在解释为已获得大规模Wan可用checkpoint。
- **L61 LongLive-2.0**（[原arXiv论文，含Eq2–6与表3](https://arxiv.org/abs/2605.18739)、[官方代码](https://github.com/NVlabs/LongLive/tree/main/LongLive2.0)）：基于 **Wan2.2-TI2V-5B**（不是当前A14B双expert），训练期把多段clean history/noisy targets用Balanced SP与NVFP4组织；原模型经AR finetune，少步能力可由LoRA 4→2步训练赋予。推理为W4A4 NVFP4 GEMM + NVFP4 KV-cache + multi-shot sink + asynchronous VAE。在作者原始论文 §3.2 `K,V` 按8帧latent chunk缓存，§3.3把chunk VAE解码与下一chunk DiT重叠。官方config `LongLive2.0/configs/nvfp4/inference_nvfp4.yaml` 明确 `num_frame_per_block:8`、`sampling_steps:4`、`kv_quant:true`、`streaming_vae:false`、`async_vae:false`：**这些能力是可选配置，不可把默认示例脚本说成已经并行打开streaming VAE**。官方 `pipeline/causal_diffusion_inference.py` 已读取，确认 `WanDiffusionWrapper(...,is_causal=True)`、滚动KV/shot sink与缓存量化可独立开关。论文表3在**1×GB200 180GB，异步解码另用1GPU**的条件下，5B 2step模型仅DiT吞吐45.7FPS，E2E时间另列（16s视频11.2s）。不能说“单卡45.7FPS包含VAE输出”。

- **工业近期补充**：[Runway 2026-09-10技术陈述](https://runway.com/news/research/towards-instant-video-generation)：自述将普通foundation模型后训练改为因果AR，再做 off-policy→on-policy DMD，音视频decoder也改为可逐段流式；这是厂商公开技术叙述，不是公开可复验源码/客观benchmark，单独在ledger标 INDUSTRY_PRIMARY_STATEMENT。

## 3. FFN如何从普通full-sequence变成真正实时工作负载

**原模型 G0–G5** 的每个FFN为本次 `q = AdaLN(norm2(r2))` 经HiF4 QDQ的固定函数 \(f_{\theta}(q)\)。历史 \(\mathcal H\) 并不是额外决定FFN输出的隐含变量：给定完整 \(q,\theta\) 时 \(f_\theta(q)\) 已确定。历史只可能通过：(i) 改变当前 attention/residual, 从而改变q；(ii) 提供用于预测 \(q\) 或 \(f(q)\) 的便宜侧信息；(iii) 新的训练后模型让函数直接把历史作为输入。

对当前chunk实际 token 数 \(M_c\)，FFN dense主投影 MAC：
\[
C_{\rm FFN,chunk}^{(1\,step)} \simeq 2 L M_c D H,\quad
D=5120,\ H=13824,\ L=40
\]
（当前原Wan双expert一次step只用一套权重）。普通整段需要对全部M行、K步执行；流式模型 \(M_c\) 由当前新chunk决定，原模型的参数D/H/L/K**可能同时改变**，不能将该式当实时模型实测。计时还要加入norm/cond/QDQ/KV/attention/文本/decoding和调度。

**三个硬件状态**：
1. 普通 full M：大矩阵乘更接近算力受限；若计算得到的token×channel稀疏点接近均匀散点，则原布局tile仍被触发（FFNDS [E0原生tile证据](../../20261025/E0_INTRINSIC_FULL_HIDDEN.md)）。
2. 短chunk small M：线性层更可能受重量读取、kernel launch、数据布局/路由开销支配；须用形状与Roofline测量判断，原full-M的稀疏mask收益不能按M线性外推。这是工程**推论**，不是新native profiling。
3. 长stream：KV/window驻留与FFN weight residency竞争显存；量化KV、异步VAE能释放计算/带宽预算，但任意FFN缓存又消耗历史内存。系统最优可能是**低位连续密集GEMM**而不是碎片化稀疏；LongLive2原始NVFP4实现正是值得同成本比较的系统对照。

对延迟要求 \(T_{\rm deadline}=1/\mathrm{FPS}_{target}\)，必要条件是服务系统在该资源、并发下达到给定chunk/帧的 **p99和TTFF**。若稳定流水线中DiT和VAE已充分重叠，steady周期近似由 \(\max(T_{\rm DiT},T_{\rm VAE},T_{\rm transfer})\) 支配；首帧/冷启动仍受串行准备与初段延迟约束。FFN本身更快不会消除其它瓶颈。

## 4. 从论文证据生成三条互相竞争的FFN研究机制，而非规定三种算法

**H1：功能计算的时间层级（SCD启发）**。部分FFN功能可能承担低频语义/身份/scene state更新，其余承担每一步高频纹理细化。数学研究对象是将整个**函数响应**分成时间慢变化 \(\mu_{n}\) 与当前创新 \(u_{n,k}\)，测试其在gate/writeback/下游flow上的独立贡献：
\[
g_{n,k}\odot f(q_{n,k})\stackrel{?}{=}\mu_n+u_{n,k},
\quad \text{compare }\operatorname{Var}_k(\mu_n)\text{ and finite downstream effects.}
\]
该式是**可证伪候选表示**，现有SCD论文只证明原作者模型的架构观察，并未验证Wan FFN功能分离。竞争解释：早层step相似由 attention/归一化共同造成，FFN value变化仍必要；最小区分需将pre-FFN q、post-Q2系数与gate后有符号贡献分别对齐。

**H2：缓存有效性的多级时间尺度（Self Forcing/LongLive）**。一个已经计算出的FFN值 \(f(q_{\mathrm{old}})\) 在自生成 \(q_{\mathrm{new}}\) 下是否可复用，取决于 \(\Delta q\)、expert/condition的变化和下游有限响应，而不只本地SSE。合法reuse还要求缓存来自同层、同权重、相同QDQ与兼容条件；跨high→low的已有full-model FFN缓存通常不满足同函数条件。竞争解释：真实有用的复用主要来自**attention KV**，FFN缓存收益不及访存/失效开销；最小判别为按scene/prompt/rolling历史变化分段，比较固定周期刷新与条件刷新成本，以及下游运动/颜色漂移。

**H3：模型重构胜过原生mask（FFF/SCD/LongLive2共同启发）**。更小的renderer FFN或训练后的低位密集替代函数，可能在短chunk固定SLO下胜过原稠密权重的动态稀疏/补偿。竞争解释：训练后的新架构成功来自全网重分工和分布适应，无法直接转移到现有冻结Wan。最小判别先做数学对齐及真实可见信号表，再在未来明确授权后比较同硬件、相同质量与完整端到端时间的条件算子；当前阶段不运行新模型。

**最重要的新研究问题**：当前 FFNDS 本来观测到局部FFN误差较小与四步状态/视频漂移并存。真实流式模型的自生成历史会再次喂回下一chunk，它将这个反馈链从四步延长到不断滚动的时间轴。因此优化目标应在**多时标、端到端功能保真和严格截止时间**内定义，而不能将“后验局部 SSE 更小”升级为可实时缓存的证据。原FFNDS的 [10源/12seed/1920sites](../../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md) 可以提供未来判别方法的算子/轨迹构想；当前没有对应长期流式实验。

## 5. 来源分级与研究后续

最强可复核实时物理先例：LongLive2原始数学及代码（Wan2.2 TI2V-5B低位内核）、StreamDiffusionV2官方source（Wan2.1 causal V2V/T2V和SLO）、SCD源码（因果模块与FFN可分层结构）。CausVid/Self Forcing/RollingForcing、LongLive/MotionStream用于蒸馏与自生成分布，FAST-AR用于明确attention压缩的强竞争基线。Runway只是近2026年工业方案声明。

继续文献深读要寻找明确讨论 **FFN/MLP** 在 `once-per-frame encoder` 和 `multi-step renderer` 中角色/耗时的原始消融，提取每层/每step分项FLOPs、真实microbatch shapes、命中deadline分布；其次用同一计算图比较低位稠密内核、训练后窄FFN、原生channel稀疏与历史函数缓存。**所有G10节点是文献计算图，不改变暂停的G6真实实验权限。**


## 6. 2026年9–10月的新增流式模型：与Wan/LightX2V更直接的迁移边

这批来源进一步更新了“实时的视频基础模型是怎样来的”：**通常保留强大的离线视频foundation prior，然后通过causal/block-AR蒸馏/条件化/系统优化成为能持续交付的小步模型；另一些把外部world state/音频/动作与视频rendering剥离或同步。**

- **L65 Causal Forcing++（arXiv:2605.15141，v4于2026-10-04）**：[官方HTML全文](https://arxiv.org/html/2605.15141v4)、[源码](https://github.com/thu-ml/Causal-Forcing)。§2–3一手方法明确批判**双向teacher ODE轨迹给因果student直接回归**存在未来帧不可见的结构性错位。采用causal teacher的 **causal consistency distillation (CD)**，仅需相邻时间点的一次online ODE step监督来学习AR-conditional flow map，并以Self-Forcing形式自生成rollout的asymmetric DMD做第三阶段优化。测试Wan2.1-1.3B逐帧1–2步。作者报告framewise 2step初帧延迟较其前版本降低50%、阶段2训练成本约4×节省；这是**学生初始化/训练机制**，不意味着FFN动态mask有50%时延收益。区别于旧四步长chunk：两次去噪都发生在**更短的帧级计算单位**，会改变FFN GEMM 的M、K和偏移反馈频率。
- **L66 EdenLive（2026-10-18发布）**：[开发方技术页](https://www.tripo3d.ai/research/edenlive)、[原始可用代码及部署说明](https://github.com/VAST-AI-Research/EdenLive)。这是最接近当前工具链的**参考条件流式renderer**：Wan2.2-TI2V-5B和Wan2.1-I2V-14B foundation分别转换为 bidirectional/few-step chunk-AR 模式，visual encoder对graybox/normal map/真实参考视频抽取**几何与运动条件**；真实world state由游戏/代码/外部代理维护，模型负责渲染外观。每chunk4 denoise steps、固定历史；upstream simulation→reference encoding→generation可并行。其README确认**LightX2V**辅助低位量化、融合算子、CUDA graphs、缓存和流式LightTAE；作者报告5B 1280×704 **RTX5090约24.7FPS inference throughput**。这个数字是官方工程实验描述，**未单独证明TTFF、含传输的p99或帧级独立因果功能**。README同时记载普通安装测试GPU需80GB，不应混作5090部署最低要求。对FFNDS最大启发不是再删通道，而是**外部结构/运动明确给出后，FFN中的功能与外部renderer需求发生角色变化**：变外观与保运动的条件功能是否存在不同的计算周期？需下游视频与真实图层检验。
- **L67 Helix World（2026-10-12）**：[作者技术说明](https://noiz.ai/lp/blog/meet-helix-world)、[开源代码](https://github.com/NoizAI/HelixWorld)。基于LTX-2.3视听foundation，转换成以相机/键盘动作条件控制的少步因果学生，并同步输出视频与空间音频。厂商称单H800 768×512 24fps，RTF=0.77含音视频解码，但动作指令作用于下一chunk约1.3秒后。由此说明 **稳定24FPS吞吐≠毫秒级交互响应**，如果FFN策略延迟scene/音频同步信息，其系统后果需要跨模态测量。此项为作者公开技术与代码存在性，未独立复验。
- **L68 Astronex-World 1.0（2026-09-17技术报告）**：[arXiv原始报告](https://arxiv.org/abs/2609.20034)。基于Wan2.2-TI2V-5B的bidirectional与block-causal两个模型，跨block KV、按层 action stream（64-d）和camera PRoPE；五阶段迁移包括causal改造、少步蒸馏/混合域恢复与分布匹配。作者称在832×480以24FPS流式生成，并把文字事件放入指定rollout位置。对FFNDS而言，**time-AdaLN和action conditioner是时间变化的前置合法信息**，但“动作低维”不等于只需算少量FFN通道。

这四项在 [G10](../../../docs/98_端到端计算图与证据注解_20261025.md) 增加 **S21–S25**：因果一致性学生初始化、逐帧1–2步、外部scene state/视觉proxy、LightX2V流式renderer、动作/音视频同步。它们的核心共同点是时间依赖变更有**训练机制**和**推理算子**两条证据通道；我们要在当前原Wan中的真实函数角色、可用观测及硬件预算上独立求证。避免只把企业FPS当端到端deadline通过率。

**FFN机理新判别**：若参考给出的layout/motion已知，则在同模型中对保持参考不变而只改appearance conditioning的样本，计算 \( \Delta U^{\rm appearance}\)；反之保appearance不变只改motion参考计算 \(\Delta U^{\rm motion}\)。探索两类变化是否落在不同的 **gate-weighted W2 value subspace**，并向跨chunk video/flow传输。真实Wan是否支持这个分解尚未实证，来自EdenLive“外部状态→渲染函数”的机制启发，而非其作者FFN观察。小成本判别应先找已有条件对照与冻结权重，不能把训练后模型的功能分工当冻结原FFN性质。
