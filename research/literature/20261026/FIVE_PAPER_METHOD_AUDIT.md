# FFNDS · 最强五篇直接近邻的方法级深读（2026-10-26）

**状态：五篇审查的证据深度不同。** 本轮亲读 DiffSparse ICLR 2026 官方19页论文的方法§3.2、实验§4.1和附录A.5.2–A.5.3；亲读 Chipmunk 作者Part I/II技术正文和GitHub实际Python、CUDA源码；RT-Lynx与Dense2MoE核对官方正式摘要/可访问部分PDF文字，第三方alphaXiv自动概述只能列为二级待核证据；Activation Concentration核对arXiv v2官方摘要，正文/模拟未精读。**这不是五篇的完整代码复现，也没有进行新的Wan FFN实验。** 逐项见[机器审查账本](METHOD_AUDIT_LEDGER.json)、[完整33篇索引](PRIMARY_LITERATURE_INDEX.json)。

## 1. DiffSparse：它优化的是跨步、跨层token复用预算，且WAN不是四步

[正式ICLR论文PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5e66b9f710a8e38351da690f3806cba9-Paper-Conference.pdf) §3.2：token selector根据已定义的Attention/缓存等信号给token排序，固定保留率选择重新计算的token；其余token复用历史layer features。各层、各去噪步的保留比例不是手工配置，而是用维度为 (T×L)×|S| 的可学习cost matrix产生。给出候选稀疏率集合S（例如0、0.25、0.5、0.75、1）；离线动态规划在全局稀疏率预算下最小化各位置cost之和。原文方程(6)–(7)的递推可写成：

\[
F(i,r)=\min_{s\in S,\,s\leq r} \bigl(F(i-1,r-s)+C_{i,s}\bigr).
\]

这一步DP用于**训练/优化期**而非每次推理时实时规划（论文正文给出其在相应配置下约30秒的离线求解时间）。通过STE处理离散mask对cost的不可导，最终teacher/student完整生成的**LPIPS**用于训练稀疏配置，而非以FFN中间输出SSE作为唯一学习目标。两阶段训练首先确定有利的full-step位置，随后把该prior融入跨层预算微调，降低对固定full-step调度的依赖。

**决定性适用边界**：官方§4.1清楚列出 FLUX.1-schnell 四步、PixArt-alpha 20步、DiT-XL/2 50步，而视频模型 **Wan2.1-1.3B使用25个flow-matching步骤**。训练使用约1万条conditioning文本，Wan来自WebVid10M captions；按作者记录每个实验约4–10小时、8张AMD MI250 80GB。Wan视频评价是950 prompts、共4750条、VBench 16维。摘要中的“20步PixArt减少54%计算”不能写成“4步Wan视频FFN减少54%”。

**与FFNDS的关系**：这是已经采用最终生成功能代理来学习全程预算的具体先例，但原子动作是token级缓存/跳过层特征，而不是直接找到原Wan W1/GELU/Q2/W2神经元的必要函数；它也包含清楚的训练成本。研究新FFN方法时应避免再发明“LPIPS+DP调节每层保留率”的相近优化器作为主要创新。详见[原文与实验范围](DIFFSPARSE_PRIMARY_READING.md)。

## 2. Chipmunk：原MLP实现中存在真实、计费的掩码预测与缓存结构

[论文](https://arxiv.org/abs/2506.03275)、[作者Part I](https://sandyresearch.github.io/chipmunk-part-I/)、[作者Part II](https://sandyresearch.github.io/chipmunk-part-II/)与[官方代码](https://github.com/sandyresearch/chipmunk)。作者从 **MLP每个post-GELU标量系数乘静态W2 value向量**出发，将更新理解为新旧step的value贡献之差：若是同一权重和同一数值语义的FFN，则一个完整输出可由旧输出缓存加上稀疏系数变化的W2投影近似。**变化稀疏**不等于当前数值很小，也不等于自然token×channel掩码就有硬件局部性。

本次直接检查的代码快照为 Git tree 177fdefc4803e621fd8a02aad491812e9d1ecc94：
- [src/chipmunk/modules/mlp.py](https://github.com/sandyresearch/chipmunk/blob/master/src/chipmunk/modules/mlp.py) blob 1415c033df8af9ec388b315036b450a7d2ea87ee。full step保存postactivation、输出和W1中间结果的token-block均值；sparse step先对block_mean(x)做**缩小token规模的fc1计算**，与上次blockmean_mid比较，调用topk_indices和copy_indices得到索引，再运行稀疏核。这直接证明mask并不是“无需计算即可读到的真实完整hidden”。跳过mask重算的一条源代码条件还涉及 **inference_step>=10**（并带缓存存在、block_mask_cache时序条件），对四步Wan不能直接使用默认调度推论。
- [src/chipmunk/ops/mlp.py](https://github.com/sandyresearch/chipmunk/blob/master/src/chipmunk/ops/mlp.py) blob a42074fd41477cf666ae298a994baa09f68214c7：W1有BF16与特定FP8路径，先稀疏mm1，再使用mm2与scatter-add改写cached output，不包含Wan HiF4 post-Q2精确路径的支持声明。
- [csrc/mlp/csp_mlp_mm1.cu](https://github.com/sandyresearch/chipmunk/blob/master/csrc/mlp/csp_mlp_mm1.cu) blob 7fe07b98462f748af5405e94ce3f430a1f7bf89d：gather、GELU、缓存差值在定制Kernel实现中相互耦合。
- [csrc/mlp/csp_mlp_mm2_and_scatter_add.cu](https://github.com/sandyresearch/chipmunk/blob/master/csrc/mlp/csp_mlp_mm2_and_scatter_add.cu) blob 5048cb5ac007ff0c93f77abaf4e76a7607c5a68a：异步scatter-add/稀疏GEMM与SM分配共同决定真实费用。

作者报告Wan2.1以及HunyuanVideo的大幅速度改善，官方Wan示例与其代表性的**50步**生成配置有关；不应转写成Wan2.2 I2V四步已经有同样的缓存复用率。四步high/high/low/low至少有两个同专家相邻步，跨专家历史不具相同W2身份；量化后Q2 group scale仍可能依赖更广的hidden组。详见[源码审计细节](CHIPMUNK_SOURCE_READING.md)。

## 3. RT-Lynx：同一步半结构化activation N:M，不是跨step缓存

[ICML2026正式论文页](https://proceedings.mlr.press/v306/cong26a.html)和[arXiv 2605.26632](https://arxiv.org/abs/2605.26632)官方摘要确认：权重2:4式半结构化稀疏会伤害DiT容量，激活N:M在测试对象上更鲁棒；论文用激活稀疏、误差补偿与优化CUDA，作者报告linear layers平均最高约1.55倍速度。

**原文尚待核实的细节**：在线norm compensation、LoRA并行补偿与选择保持dense层在[alphaXiv自动论文概述](https://www.alphaxiv.org/abs/2605.26632v2)中有描述，但本轮无法完整打开原ICML PDF及官方实现。这些**只记二级解读，不能当“已审查原算法方程”**。也不能假设它完全training-free（LoRA是否需训练要看正式方法）。即便post-GELU变为N:M可执行，W1已运行，所省计算主要在相应消费者GEMM。与FFNDS已有动态稀疏的同构点是“输入侧选择少计算某些GEMM系数”，不同点是N:M直接约束物理格式，原Q2/HiF4/GEMM压缩与dense权重格式需重新确认。

## 4. Dense2MoE：容量重组是经过训练形成的，不是原Dense FFN天然MoE

[ICCV2025官方论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html)与原PDF检索摘录明确提出：将Dense FFN替换为MoE，按输入选择激活FFN专家，再增加Mixture of Blocks；使用Taylor重要性初始化、多阶段知识蒸馏、负载平衡与group feature loss。官方摘要报告FFN activated parameters降低62.5%（指定方法设置），这一百分比不是Wan2.2模型的参数减少。

注意原论文详细路由公式/超参和消融的完整PDF目前受403影响，部分常见“shared expert + token topk”和训练阶段描述只能从[alphaXiv自动概述](https://www.alphaxiv.org/abs/2510.09094)了解，需要原论文/源码复核。因此现阶段可以断言**这一方向的真实方法需要结构转换和蒸馏**，不能断言原Wan的high/low两个时段主专家已经学成了同样的层内路由规律。这两个expert概念根本不对应。

## 5. Activation Concentration：稀疏散点不一定转化为可执行空列

[arXiv 2606.00567v2](https://arxiv.org/abs/2606.00567)官方摘要报告对七种工作负载、三类架构、四种模态的实验；元素级稀疏相对于硬件列级最高**高估78个百分点**，纯transformer DiT在论文定义的column/cycle模型中对应**12.4%**减少。作者还报告模拟GDDR6加速器memory stalls达到84–89%，motion模型有随阈值变化的精度断崖。

它提供对FFNDS过去token×channel近均匀散点的很强竞争性解释：即使大量单独坐标很小，真正skip的列或tile上仍可能处处被少数token激活。**尚未验证原论文“column”定义、完整阈值策略和模拟源码**；更不能将systolic模拟数字换成Blackwell/A100原生Tensor Core的实测速度。它说明应该首先研究真正可执行原子，再判断哪个数值信息有价值。

## 五种研究对象的对照

| 核心对象 | 哪个信息时点 | 真正减/改什么 | 是否额外学习 | Wan四步最大的辨别问题 |
|---|---|---|---|---|
| RT-Lynx激活N:M | 当前GEMM输入已产生 | sparse GEMM数据/元数据 | 误差补偿训练细节待原文验证 | W1与W2是否都能实省；原Q2及kernel怎样接入 |
| Chipmunk跨步MLP delta | 旧cache + 当前块均值预测 | 历史贡献的稀疏更新GEMM | 无额外模型训练，实时选择和cache有代价 | 高/低主expert身份、仅两步同expert、缓存重用 |
| Dense2MoE条件容量 | 蒸馏后的router决策 | 激活FFN专家与Transformer blocks | **需要多阶段蒸馏** | 原Dense功能是否含可学习条件分工及路由成本 |
| Activation Concentration硬件列 | activation真实分布与阈值 | 物理空列/存储周期（模拟） | 分析而非部署算法 | element散点在实际Wan GEMM/NVKernels中是否成块 |
| DiffSparse学习token复用 | learned cost profile + 在线token score +旧cache | token×layer计算量与缓存 | **需要离线学习** | FLUX4步≠Wan25步，Wan2.2四步迁移未证 |

## 学术新颖性的下一步：先验证文献细节再形成新主张

最强直接邻接已包括后验activation N:M、跨step value-delta、专家路由重组与低成本token预算学习。若提出FFNDS新方法，应清楚指出它与哪些先例在**冗余定义、信息时点、真实计算动作、训练量、跨专家约束和完整视频结果**上不同；只有更换几种SSE、加新FP4格式或改Top-K并不自动形成新机制。

**文献后续重点**：补RT-Lynx与Dense2MoE原始全文及公开代码，精确核对Activation Concentration的列定义/模拟，深读DiffSparse的Wan视频质量表、Chipmunk缓存/打包费用和同专家运行条件，再将它们与真实Wan计算图做条件级比较。本阶段未执行任何新的GPU模型任务。
