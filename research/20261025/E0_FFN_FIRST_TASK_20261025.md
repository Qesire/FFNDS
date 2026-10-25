# E0 · 第一轮研究任务：发现 FFN 的条件函数结构

**唯一活跃任务** · [研究纲领](E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) · [文献与机制地图](E0_FFN_INTRINSIC_PRIOR_ART_20261025.md) · [活跃状态](../ACTIVE_RESEARCH.json)

## 首先回答哪一个问题？

> `W1→GELU→W2` 的函数贡献如何随输入、时空位置、去噪步和专家变化？哪种新表示最能解释“某些计算可以减少，而其他小误差仍严重影响视频”？

**第一步是填补 FFN 研究信息缺口，不是继续重算旧 LayerNorm 径向指标。** Agent 可以用已有证据或不同的文学线索开启方法讨论，发现新机制以后再收紧观测范围。

### A · 函数与表示：找到真正有意义的计算单位

先建立**实际 FFN 数据字段地图**：真实 pre-W1 输入、W1 输出 `a`、GELU `h`、Q2 saved operand、decoded W2 sampled contributions、gate、bias 和 residual 的 tensor shape、可用 source/step/block/token 范围。特别说明哪些是整个 hidden，哪些只是32位置×五64组面板。

以 `h_j`、`d_j=h_j(g⊙W2[:,j])`、`Σd_j`、局部 Jacobian 为四种表达，检验**条件集中**与**superposition/方向干扰**两种机制。证明最小反例：`|h|` 排序可以与净功能贡献排序不同；同时验证 **GELU 前的任意旋转并非一般等价变换**。创新空间包括神经元组合、条件敏感子空间、输出方向和坐标重表示，并非只有 channel-group mask。

**交付**：一个具体公式/反例；准确 operand 覆盖地图；当前历史资产中的最小可判别观测或需要的一个精准新字段。

### B · 条件创新：研究 FFN 实际重复了什么

在固定源重建四步双专家时间表与真正可复用的历史（同 expert 的两次，不假设更多历史）；分析绝对 `a,h,y`、幅度、方向、变化 `Δ` 的区别；再比较相同 latent 网格位置与时间/空间邻近位置的结构。

竞争机制例如**共享低维变化**对**表面高 cosine 但创新仍大**、**expert 切换机制**对**同一源 seed 或采样协议产生的共同方向**。与 [Chipmunk Part II](https://sandyresearch.github.io/chipmunk-part-II/)、[AdaCache ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html) 和 [DiffSparse ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html) 区分不同缓存时点、训练和算子对象。

**交付**：一个真正解释时间机制的判别式/反例与可检验预测，不以 cosine 数字直接证明复用可行。

### C · 功能与物理自由度：让视频效果和计算结构同框

用真实 gate→residual→next consumer→scheduler 路径重构已保存的有界 finite effects；同时借助 [VBench CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) 的背景一致性、时间闪烁、运动平滑等区分**终端质量结构**。研究问题是：同样局部误差能量，为何作用方向、空间位置、时间连续性会使感知质量表现不同？

在数学上寻找引发可执行工作的表示自由度。对候选种子标明观测时点（FFN 前 / W1 后 / GELU 后 / W2 后）、实际动作（改变 W1/GELU/W2 的哪块工作）与最近相关 kernel/论文。对原先散点式 token×channel 支持，可比较 column/tile、重排、共享基等**假说**，优先用 [Activation Concentration](https://arxiv.org/abs/2606.00567) 和 [NVIDIA GEMM Tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) 的已知物理约束构造判别。

**交付**：局部量与时间质量不一致的至少一个有内容的机制解释，和一项明确可移除的物理计算候选或能够否定它的实证需求。

## 独立 Agent 的研究交接

任一 Agent 可以在 A/B/C 间自主选择最有解释力的问题，或以新的数学构造拓展问题；另一位 Agent 从不同来源独立审查。统一使用 pinned AutoResearcher v2.4 的 mechanism-research、research-expansion、problem-reformulation、web-research-discovery，按 [DEEP_WEB_RESEARCH.md](../agent_protocol/DEEP_WEB_RESEARCH.md) 主动搜索原始论文、后向/前向引用、反例和作者源码。

**报告建议**：`新认识 → 精确数学与源码 → 竞争解释及不同预测 → 文献新信息 → 下一条判别观察 → 可能的真实 FFN 工作减少`。选择不同观察值应当为了解释同一个科学分歧，而不是机械填满所有栏目。

**历史资料的检索入口**：[源身份](../20261023/C0_coupling_offline_results.json)、[E0_OPERAND 局部统计](E0_RADIAL_PAIRING.md)、[原生四臂结果](E0_FOLLOWTHROUGH.md)、[时序校准](E0_TEMPORAL_CALIBRATION.md)、[完整证据索引](../EXPERIMENT_EVIDENCE_INDEX.md)、[两轮历史数学审查](E0_SKILL_WEB_REAUDIT_20261025.md)。读取它们时保存样本/数值条件；目前没有新的 model/GPU/视频/训练实验授权。

**当前最短可落实的研究成果**：给出 FFN 输入、hidden、输出贡献与下游作用的**实际可观测字段—数学关系—竞争机制**三方对应，明确哪一条现存证据能立即区分解释。由此再提出方法种子，下一轮才选定最有价值的额外采集或 kernel 检查。
