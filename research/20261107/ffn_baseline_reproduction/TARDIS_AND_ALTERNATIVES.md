# TARDIS 与其他 FFN 竞争方法：可忠实复现的范围

**状态**：文献与源码身份审查；只将可核实的算法/代码结果写入强基线，不把模型间数字直接放入G19论文主表。

## 一、TARDIS：G19 的函数折叠最强竞争者

- 原论文：Gansen Hu et al., *Accelerating Large Language Models through Partially Linear Feed-Forward Network*, arXiv:2501.10054，https://arxiv.org/abs/2501.10054。
- FFNDS已审原文：[全FFN函数表示先例](../../literature/20261029/S2_WHOLE_FFN_REPRESENTATION_PRIMARY_REVIEW_20261029.md)，尤其 C17 节。已读内容提到常用区间线性化、在线离群预测和原始 FFN 部分回退。
- **此次未确认官方完整开源GitHub仓库**，因此不是 `AUTHOR_CODE_REPRODUCED`。若以后发现官方实现，先固定完整SHA与作者组织/README/许可证；不要将社区近似实现冒充原作者。

### 原理与 Wan 用的行向量表达

FFN `z=uW1ᵀ+b1, F(u)=φ(z)W2ᵀ+b2`。用真实CAL中的神经元高频输入区间拟合 `φ(z_j)≈a_j z_j+c_j`。离线折叠：
`A=W1ᵀ diag(a) W2ᵀ`；
`b=(a⊙b1+c)W2ᵀ+b2`。
只用 `uA+b` 为 `TARDIS_STATIC_FOLD`，**这是廉价控制组，不是完整 TARDIS**。

完整方案需要提前预测需回退的神经元集合 `S(u)`，对选中神经元去掉静态近似、补原非线性：
`F̂(u)=uA+b+Σ_{j∈S(u)}[φ(z_j)-(a_j z_j+c_j)]W2[:,j]`。
如何选择/读取 W1/W2、原始作者的区间与2-bit轻量预测机制、是否每token独立/块共享，均必须回到原论文明确实现；预测器计算、索引、gather、回退、额外矩阵和kernel成本全部计入。

### 与G19的关键判别

| 对照 | 当前输入使用方式 | 历史 | 必要付费 |
|---|---|---|---|
| G19 | 当前所有token执行 `uA` | 每专家锚点 `C_s` | 80 Dense锚点、160 A GEMM、12.08GiB BF16 C |
| TARDIS_STATIC | 当前所有token执行 `uA+b` | 不使用 | A GEMM，常数 |
| TARDIS_FULL | 当前输入＋在线outlier预测/精确恢复 | 不要求跨步 | predictor＋索引＋真非线性/GEMM＋A |
| TARDIS_ORACLE | 使用当前完整真实z/h选S | 不要求 | **oracle paid**，仅作质量上界，禁止放入真实加速榜 |

候选训练数据应全部来自 G19 CAL 四来源的原生预激活z，且下游用当前gate评估。TARDIS按LLM单token推理报告的速度不能外推到Wan N=31668的BF16大M GEMM；加上预测器后必须测全160端到端和峰值。

### 复现阶段

1. 从原论文逐公式抽取区间拟合、预测器/二值回退与显存数据，形成 `PAPER_FIDELITY` 与差异目录。
2. 同态验证 `TARDIS_STATIC` 与全Dense GELU，测真实区间覆盖、source/层/expert后门控误差、pred与true outlier精确率/召回率。
3. 只有当合法低成本预测器已实现并通过数值单测，才作为 `TARDIS_PAPER_REIMPLEMENTED` 进行A100真实kernel成本与完整视频。
4. 如无作者忠实实现且只完成STATIC，则把STATIC作为G19“无历史”强控制组；保留TARDIS作为方法先例，论文不得声称已经完整复现。

## 二、次级方法：结合已存证据安排

| 方法 | 公开论文/仓库 | 与FFN相关的核心动作 | FFNDS已有状态/下一步 |
|---|---|---|---|
| **RT-Lynx**（ICML 2026） | https://proceedings.mlr.press/v306/cong26a.html | 激活N:M稀疏，误差补偿和专用GEMM | 已做[Wan BF16 2:4启发试验](../../20261030/RTLYNX_WAN_BF16_2TO4_PAIR_RESULTS_20261030.md)：一源门后误差仍0.24–0.27、补偿大M更慢；**非作者完整代码复现**。不是第一批再跑同试验 |
| **TEAL** | https://github.com/FasterDecoding/TEAL | 通过前置/激活支持实现稀疏GEMV | [作者代码独立审查](../../20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)核作者kernel主要单token、Wan大M会dense fallback；[真M采集](../../20261030/TEAL_WAN_FULLM_H_CAPTURE_RESULTS_20261030.md)已做。除非新BF16大M物理核成立，否则仅作历史旁证 |
| **SharQ** | 仓库自己的既有Wan FP4路线 | 结构化N:M/原生FP4与Dense残差 | [全160视频及质量](../../20261030/SHARQ_WAN_ALL160_VIDEO_V2_QUALITY_RESULTS_20261030.md)已测，数值族不是G19 BF16，不能平行声称公平比较 |
| **SlimDiff**（2025） | https://arxiv.org/abs/2509.21498 | FFN功能权重组、激活协方差及免训练压缩 | 需要独立FFN分支/训练成本/模型原始质量核验；暂不取代ICC |
| **ToaST**（作者仓库自称ICML2026） | https://github.com/SHANNonLab-HUFS/ToaSt | TCS选择动态FFN通道，联合attention剪枝 | 官方README仍有paper-title/author/arxiv TODO；主要ViT分类且部分表依赖finetuning，不是Wan免训练FFN原样基线 |
| **FocusDiT**（2026） | https://arxiv.org/abs/2606.02090 | 对token/query选择FFN计算 | 需代码身份、训练/剪枝与任务条件审计，后续关注 |

作者论文“整体模型最高加速”“大M单层核收益”“四步FFN-only生成时间”是三种不同量。严禁把RT-Lynx线性层1.55×、Chipmunk+StepCache的2.67×、TARDIS LLM服务器1.6×或TEAL单token推断速度，直接写成对G19的性能差异。

## 三、决策门槛

首批只需要三个独立可解释的强基线：
1. **ICC-SVD/CD-SVD**：直接比较两级低秩增量与G19一次完整折叠；
2. **已有Chipmunk P64 + 必要匹配补试**：比较稀疏真实增量与G19近似全增量；
3. **TARDIS-STATIC 或忠实FULL**：比较无历史完整线性折叠与动态回退。

若某方法缺作者代码/核/模型适配而只能达到oracle局部误差，标 `SIMPLIFIED_CONTROL`/`NOT_REPRODUCED`；不能用它在总表中冒充已运行方法。任何新的科学执行需独立review、ACTIVE授权及真实收据。
