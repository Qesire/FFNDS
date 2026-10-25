# FFNDS · 本征 FFN 科学机制与竞争先例地图（2026-10-25 · WARD 扩展）

**用途**：向 FFNDS 提供新的机制、不同数学表示与最强先例；[纲领](E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md)、[首个任务](E0_FFN_FIRST_TASK_20261025.md)。本轮 2026-10-25 使用公开 Web 搜索，分 FFN 函数解释、表征叠加、DiT 激活/动态计算、视频功能质量、实际稀疏执行五族检索；继续从 Chipmunk 理论篇追到 Geva 的 FFN key-value 解释，从 activation-concentration 追到真实 GPU GEMM tile，从背景闪烁问题追到 VBench 时间维度。部分来源读到作者方法正文；其它仅核实原始会议摘要 / 官方项目方法说明，逐项标注。

## I. FFN 究竟在计算什么：基础功能机制

| 文献与本次阅读深度 | 来自原文的结构 | 对 Wan 的可区分研究问题 |
|---|---|---|
| [Geva et al. · Transformer FFN Are Key-Value Memories, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.446/)（官方摘要/论文结构） | W1 输入模式（keys）经非线性产生系数，W2 作为 value vectors 的组合；语言模型中的人类可解释模式。 | Wan FFN 中是否有随时空/conditioning 激活的共同 value basis？单通道功能意义与整体向量组合是否同一件事？ |
| [Geva et al. · FFN Build Predictions by Promoting Concepts, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.3/)（官方摘要） | FFN 增量可分解为多个向量子更新，经 residual 后续作用。 | 把 **输出影响** 与 **某个 hidden 通道值** 分开；功能价值必须追踪消费者。 |
| [Toy Models of Superposition, 2022](https://transformer-circuits.pub/2022/toy_model/index.html)（作者页面/理论摘要） | 特征可超过 neuron 数，在同一表示坐标相互干扰；基与神经元不必单义。 | W2 通道/组幅值分布可能不是最合适的计算作用坐标；如果换表示，何种可计算量与功能更稳定？ |
| [Chipmunk Theory Part II](https://sandyresearch.github.io/chipmunk-part-II/)（作者正文，MLP 加权 value vectors 方程） | `GELU(XW1)W2` 与 attention 都可写成加权 value vectors，扩散步上的更新可以从基础贡献组合研究。 | 优先比较静态 basis 与随条件变化系数，而非把 full output cosine 视为全部解释。 |

**数学落点**：对于固定 gate 的 `d_j=h_j(g⊙W2[:,j])`，计算 `Σ_jd_j` 与有符号 Gram；进一步审查 `W2 diag(φ') W1` 的局部 Jacobian 的方向依赖（数学恒等式，不是 Wan 特有实测低秩）。**最强竞争解释**：看似集中的神经元信号可能来自 superposition / basis choice，真正可减少的作用未必对应当前 channel。

## II. 激活稀疏与有意义的执行单元

| 文献与本次阅读深度 | 原始机制 | 对 FFNDS 下一项区分 |
|---|---|---|
| [Sparsing Law, ICML 2025](https://proceedings.mlr.press/v267/luo25i.html)（PMLR 官方摘要） | 从可用性/任务指标测量 FFN 激活稀疏，考察架构与训练依赖；主要对象是 LLM。 | 近零率和「任务可容忍的稀疏」区别，避免从 LLM 迁移一条经验阈值到 GELU DiT。 |
| [Universal Properties of Activation Sparsity, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec3c79dc0c2b85532cfd1012a4aaa923-Abstract-Conference.html)（官方摘要） | LLM FFN 的广泛激活稀疏性质，包含 diffusion LLM，非视频 DiT。 | 澄清 GELU 非严格零、模型/训练结构与安全稀疏预算的区别。 |
| [RT-Lynx, ICML 2026](https://proceedings.mlr.press/v306/cong26a.html)（PMLR 官方摘要） | 对 DiT activation 做 N:M 而非只稀疏 weights，配合补偿与 CUDA kernels；论文报告 linear 最高平均约 1.55x。 | 如何比较 FFN 激活 N:M 与贡献方向选择？低训练代价是否可以得到新的无补偿机制？ |
| [Activation Concentration, arXiv 2606.00567v2](https://arxiv.org/abs/2606.00567)（原始 arXiv 摘要/日期） | 多种 diffusion 工作负载 element-vs-column 稀疏差距，pure DiT 在相关模拟设置中呈分散，不同模型差别很大；对应 systolic array 抽象。 | 已知散点支持能否在输出方向、真实 tile、重排或时空条件坐标下变成有用的执行局部性？需要实测 Wan 自己的分布。 |

**关键新信息**：加速不由单个稀疏率决定，而由「怎样观察有效贡献」和「可以排除哪个原生执行 tile」共同决定。RT-Lynx 与 Activation Concentration 是紧邻计算竞争；不能用 N:M 或列稀疏命名本身作为创新。

## III. 四步去噪与跨空间的函数创新

| 文献与本次阅读深度 | 已报道机制 | 对 Wan 四步双专家的分歧 |
|---|---|---|
| [Chipmunk, arXiv 2506.03275](https://arxiv.org/abs/2506.03275)、[Theory Part II](https://sandyresearch.github.io/chipmunk-part-II/)（原始论文摘要、作者正文） | 先缓存再动态更新 column-sparse delta，token voxel reorder 和核实现；论文有 WAN2.1 组合结果。 | **同专家仅两步**时能否从第一步获得足以降低第二步 FFN 重算的结构？与跨 expert 不等价。 |
| [AdaCache, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html)（CVF 原始摘要） | 因视频运动变化而选择缓存调度，运动信息影响推理预算。 | 有关运动的条件量能否提前影响 FFN，而非只调度整 block 的缓存？ |
| [Token-wise Feature Caching, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbe024e0517fe12ac3a8d388b19ff9fe-Abstract-Conference.html)（官方摘要） | 不同 token 缓存的质量敏感性明显不同，有些 token 比别的 token 更易损伤质量。 | FFN 单位的低局部误差与高功能影响可能在 token 级放大。 |
| [DiffSparse, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html)（官方摘要） | 通过可微分稀疏配置与两阶段训练适配少步 DiT，含 Wan2.1。 | 把「训练自由/小型后训练」与「目标 FFN 计算粒度」作为差异，而非少步适用性的空泛主张。 |
| [Model Reveals What to Cache, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.html)（CVF 原始摘要；HTML 深入打开受 403） | 识别前景与背景的不同缓存倾向、跨 block/step 复用。 | 旧 A5+HWG 背景光照不稳提示重要性不是简单前景优先；需分清机制与实验场景。 |
| [Hidden Semantic Bottleneck in Conditional Embeddings, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1ec299a5229034141e58aeded0d0b9de-Abstract-Conference.html)（官方摘要） | 在 DiT 条件 embedding 中，方向高度相关但语义分布集中部分维度。 | Wan Block0 residual 共同方向可能来自 conditioning/共同 bias，定位前需对比 attention 与 FFN 的实际来源。 |

**重要分离**：跨步相似本身既可能来自稳定公共方向，也可能来自低变化方向；只有扣除尺度漂移、同/跨 expert 切换、真实在线历史可用性后才对「缓存什么」有信息。

## IV. 终端功能：背景、闪烁与时空扰动

| 资料与阅读深度 | 原始评价/现象 | FFNDS 可提出的判别 |
|---|---|---|
| [VBench, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)（官方论文摘要；PDF 结果文本关键量） | 将背景一致性、temporal flickering、motion smoothness、dynamic degree、subject consistency 分开；原始论文提及静态片段的闪烁检测。 | 历史「主体稳定、背景光照闪动」建议把质量向量和定向时序差分都保存，不让主体一致性平均掉局部异常。 |
| [Blind Video Temporal Consistency, SIGGRAPH 2015](https://doi.org/10.1145/2816795.2818107)（ACM 原始摘要） | 像素/梯度域的时序规则性能约束，独立逐帧处理容易产生闪烁。 | 在视频比较中，可把时间一致性视为独立于单帧误差的评价维度；与 FFN 因果解释仍需具体实验。 |
| [RSTR, ICML 2026](https://proceedings.mlr.press/v306/sun26aw.html)（PMLR 摘要） | 研究 guidance timestep 和 block sensitivity 的时空冗余，含蒸馏/剪枝外的预算分配概念。 | 本 FFN 的影响可能随 block/step 变化；该论文的 CFG 双前向假设需要核对当前冻结 Wan CFG 路径。 |

**决定性机制分歧**：`||δ||` 小、nextnorm 一阶响应小、最后 flow SSE 小、视频静态区域的时间抖动小，是四个**不同的可观察量**。后续控制实验应考虑将动作的局部幅值配对，再比方向、时序和原生数值边界；已完成的两组 fixed deletion 提供有限证据，但不等同于这些匹配干预。

## V. 硬件与表示：数学节省如何变成时间节省

| 原始资料与阅读深度 | 机制 | 对 FFNDS 的约束 |
|---|---|---|
| [NVIDIA Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)（官方 Tiling/Requirements 正文） | Tensor Core GEMM 以 M/N/K tiles 执行；tile 利用率、维度对齐及 K 维 reduction 影响真实吞吐。 | 单个非零/零元素对真实 GEMM 代价的影响取决于整个 tile；必须知道动作在哪个轴减少执行。 |
| [NVIDIA Block-SpMM Guide](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)（官方实现介绍） | 块稀疏与索引/访存布局共同决定 sparse-vs-dense 收益。 | 剩余高密度或随机稀疏不可仅用 FLOP 算加速；将布局纳入 NetVOI。 |
| [RT-Lynx 实际 CUDA](https://proceedings.mlr.press/v306/cong26a.html)（论文摘要，仅核对 CUDA 主张） | 激活 N:M 稀疏须配套匹配的 kernel 才能获得执行收益。 | 将新想法的「数学动作」「物理可执行核」「整视频速度」逐层绑定。 |
| [BlockSmith, TACO 2026](https://doi.org/10.1145/3828547)（ACM 原始方法正文，主要是一般 SpMM/GNN） | GPU sparse rows/block 重排和代价受数据布局影响。 | 作为**可迁移系统类比**思考 tile 重排；本 FFN 并未使用其算法，不能移用 GNN 速度数字。 |

## 本轮搜索记录与下一步检索任务

- **已搜簇**：FFN key-value/Geva、superposition、GELU/activation sparsity、DiT/MLP temporal deltas、few-step token caching、video temporal flicker、Tensor Core tiling/SpMM。
- **已亲读源码级信息**：Chipmunk 作者 Part II 中 MLP 形式及 value-vector 分解，NVIDIA GEMM 计算 tiling 官方说明，FFNDS 固定版 Wan/LightX2V 入口与历史实验绑定；Geva/ICLR/PMLR/ICCV 等大多核对**官方摘要**，尚未逐论文复现全部方法或 ablation。
- **继续深读任务**：选最有辨别力的 F1/F2/F3 机制后，再以原论文特定算法、补充材料、正式源码和前/后向引用展开，而不是每次新增泛泛的文献标题。
- **近邻比较规则**：始终记录对象/时点/数学/动作/训练/原生代价/视频验证七维中的真正差异。表达「已知基本原理，Wan 上的新观察」「竞争的完整方法」「尚未证实的跨模型类比」三种关系。

**综合启示**：最有价值的新增研究空间不只是「大部分 `h_j` 是否小」，而是 **FFN weighted-value 的功能几何、条件创新的时间结构、误差的质量放大、以及可执行 tile 之间能否建立新的机制关系**。这些机制可以引出通道稀疏，也可能引出不同的等价计算表示。
