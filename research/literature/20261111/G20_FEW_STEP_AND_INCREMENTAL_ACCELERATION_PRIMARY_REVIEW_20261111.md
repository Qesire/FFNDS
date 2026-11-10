# G20 · 少步蒸馏、函数增量与跨模块缓存：新增原始论文核读簿

**日期：2026-11-11｜身份：LIT / SOURCE-AUDIT / NO_MODEL_RUN**

此文件补充 [主文献索引](../20261026/PRIMARY_LITERATURE_INDEX.json) 的 L99–L109。早已收录的 **L39 SVD-Cache**、**L52 L²P**、**L20 BWCache**、**L34 DIFS**、**L35 FIS-DiT**、补充研究 **C16 LEMS**、**C17 TARDIS** 直接复用，**不重复赋 ID**。关联 [G20 新研究综合](../../20261111/G20_ICML_FORWARD_REVERSE_SYNTHESIS_AND_EXPERIMENT_DECISIONS_20261111.md)、[G20 既有数学](../../20261110/G20_FFN_FUNCTIONAL_OPERATOR_REDERIVATION_20261110.md)。

## 0. 核读与证据范围

本次已核正式会议页面/原始 arXiv 摘要、已明确的论文机制，以及可获得的官方作者 README/示例命令；并非逐篇完成全文每一公式、全部附录与作者 kernel。每条的「核查范围」限制了后面的推断。原论文作者报告的跨模型、跨设备加速，与本仓 Wan2.2-I2V-A14B-LightX2V-Distill-4step 的 A100 / BF16 / HHLL / 无 CFG / 81 帧 / 160 FFN 站点**不构成同配置排名**。

| ID | 论文与原始来源 | 已有能力及当前推理设置 | 与 G19/G20 的关系与核查范围 |
| --- | --- | --- | --- |
| **L99** | **ICC** — *Accelerating Diffusion Transformer via Increment-Calibrated Caching with Channel-Aware Singular Value Decomposition*, CVPR 2025，[会议原文](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html)，[作者代码](https://github.com/ccccczzy/icc) | 历史输入/输出 + 低秩权重增量校正，通道感知分解；作者主要 DiT 图像与 35-step DDIM | **外层增量代数近邻**。ICC FFN 分两级低秩、保留 GELU；G19 一次折叠完整非线性 FFN 增量。官方摘要 + 既有作者源码专门审查 [移植合同](../../20261107/ffn_baseline_reproduction/ICC_FFN_PORT_SPEC.md) |
| **L100** | **GOC** — *Accelerating Diffusion Transformer via Gradient-Optimized Cache*, ICCV 2025，[会议原文](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Accelerating_Diffusion_Transformer_via_Gradient-Optimized_Cache_ICCV_2025_paper.html)，[作者代码](https://github.com/qiujx0520/GOC_ICCV2025) | 缓存梯度队列、传播变化与拐点调度；论文 ImageNet/DiT | **缓存误差传播先例**，不以终端梯度/历史差分本身宣称首创。官方会议摘要；本次未运行作者代码 |
| **L101** | **RFC** — *Relational Feature Caching for Accelerating Diffusion Transformers*, ICLR 2026，[会议页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4019681ef05e8e347c3067d7f973a364-Abstract-Conference.html) | 当前模块输入预测输出变化**幅值**，历史方向辅助预测；输入关系也控制刷新 | **当前输入驱动预测的近邻**；G19 直接预测向量增量。官方摘要；方法公式/代码深审待补 |
| **L102** | **LearniBridge** — *Learnable Calibration of Feature Caching for Diffusion Models Acceleration*, ICML 2026，[PMLR](https://proceedings.mlr.press/v306/huang26f.html)，[作者代码](https://github.com/Iiiiiiirene/LearniBridge) | 跨 prompt 共享低秩校正，LoRA 学习；作者 Wan2.1-T2V-1.3B 示例 **sample_steps=50、CFG 5、N=5 为缓存参数** | **共享低秩校正先例**。其 4.10× 是作者 50-step Wan 实验报告，非四步蒸馏结果。PMLR 摘要 + 作者 wan/README.md 的实际命令 |
| **L103** | **BudCache** — *Budget-Constrained Step-Level Diffusion Caching*, ICML 2026，[PMLR](https://proceedings.mlr.press/v306/lei26j.html) | 固定推理预算优化最终输出质量；离线模拟退火/爬山步骤策略 + cache-aware scheduler | **终端质量—预算优化先例**。以步级为动作，不等于 FFN/Attention 付费子算子联合机制。PMLR 摘要 |
| **L104** | **DisCa** — *Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching*, CVPR 2026，[原文](https://arxiv.org/abs/2602.05449)，[官方代码](https://github.com/Tencent-Hunyuan/DisCa) | 学习小型 DiT 特征预测器；包含 Restricted MeanFlow / 少步蒸馏联合设计，HunyuanVideo 1.5 I2V 四步等 | **少步蒸馏与缓存兼容性已是顶会明确选题**。其比较包含联合蒸馏方案，不能把各臂当成同一冻结 checkpoint 上的纯缓存消融。arXiv 摘要 + CVPR 作者 README + 已公开附录摘要 |
| **L105** | **AutoTarget** — *Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection*, arXiv 2610.03577 (2026-10-02)，[原文](https://arxiv.org/abs/2610.03577)，[作者代码](https://github.com/wali1024-offical/AutoTarget) | 少量无缓存运行校准目标张量，按模型/分辨率/solver/复用日程选缓存位置；PixArt-LCM、FLUX.1-schnell及蒸馏视频 DiT | **少步选择“缓存哪个计算结果”已有先例**。arXiv 摘要；具体 Euler 等价条件/实验完整设置需正文补核；不宣称已录用 |
| **L106** | **Chorus** — *Beyond Few-Step Inference: Accelerating Video Diffusion Transformer Model Serving with Inter-Request Caching Reuse*, arXiv 2604.04451 (2026-04-06)，[原文](https://arxiv.org/abs/2604.04451)，[项目页](https://mlsysu.github.io/chorus/) | 工业四步蒸馏模型，跨请求相似状态复用；作者报告最多 45% 节时 | **四步单请求内冗余有限时的系统替代方向**；跨请求复用需要业务流量和条件匹配，不等于单请求 FFN 方法。arXiv 摘要/作者项目页；不宣称已录用 |
| **L107** | **RECAP** — *Reciprocal Sparse Attention and Hierarchical Caching for Efficient Video Diffusion*, arXiv 2610.26244 (2026-10-19)，[原文](https://arxiv.org/abs/2610.26244)，[作者代码](https://github.com/lyraxtg/RECAP) | 稀疏 Attention 当前粗估 + 缓存 dense 细节校正，在线漂移反馈及整步预测复用；HunyuanVideo 等 | **Attention 稀疏 + 缓存误差耦合的直接系统近邻**；“简单联合”/“残差校正”已被占据。arXiv 摘要 + 作者 README/operator.py 的代码说明；尚属预印本 |
| **L108** | **SenCache** — *Accelerating Diffusion Model Inference via Sensitivity-Aware Caching*, CVPR 2026，[会议原文](https://openaccess.thecvf.com/content/CVPR2026/html/Haghighi_SenCache_Accelerating_Diffusion_Model_Inference_via_Sensitivity-Aware_Caching_CVPR_2026_paper.html) | 对去噪 latent 与时间扰动的输出敏感度，逐样本控制刷新 | **Jacobians 用于缓存决策已有先例**。官方摘要；跨层伴随传播及实际状态反馈仍需独立判别 |
| **L109** | **SeaCache** — *Spectral-Evolution-Aware Cache for Accelerating Diffusion Models*, CVPR 2026，[会议原文](https://openaccess.thecvf.com/content/CVPR2026/html/Chung_SeaCache_Spectral-Evolution-Aware_Cache_for_Accelerating_Diffusion_Models_CVPR_2026_paper.html) | 区分去噪特征中的内容与噪声频谱变化，用滤波后的变化决定缓存日程 | **纯幅值或固定频谱统计已有对照**；视频运动/终端功能不等于频谱代理。官方摘要 |

### 既有条目，不重新编号

- **L39 / SVD-Cache**（CVPR 2026）：主子空间预测 + 残差复用；包括与蒸馏技术兼容性研究。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)。
- **L52 / L²P**（CVPR 2026）：将 Taylor 等历史预测归为固定线性组合后学习系数；报告含 Qwen-Image-Lightning 少步实验。[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_Beyond_Fixed_Formulas_Data-Driven_Linear_Predictor_for_Efficient_Diffusion_Models_CVPR_2026_paper.html)。
- **L20 / BWCache**（ICLR 2026）：Block-wise feature reuse。
- **L34 / DIFS** 与 **L35 / FIS-DiT**：分别是四步结构训练/帧交错省算的强先例；保留原审计版本。
- **C16 / LEMS**（ICML 2026）：全局传播误差与 rank allocation；**C17 / TARDIS**（2025）：GELU 等激活的线性折叠；都在既有 supplemental 映射中。

## 1. 四种不能混写的比较

1. **原生 35–50 步**：ICC、LearniBridge 的原始模型报告，复用机会较多；作者几倍加速不是四步配置性能。
2. **四步蒸馏，但与缓存一起训练/改 solver**：DisCa；应标明蒸馏 recipe 与缓存训练/测试是否同一 checkpoint。
3. **四步蒸馏、同一次生成内缓存**：FFNDS G19；两个专家各两次，强制 step0/2 建立锚点。
4. **四步模型、跨请求复用**：Chorus；吞吐/请求分布/语义匹配假设不同于单视频延迟。

论文中的 `N=5`、`4.10×`、`50 steps` 是各自算法参数、速度与原始采样步数，必须记录单位。作者最高速度、FFN 核单层速度和同模型 end-to-end wall time 分开。

## 2. 反向论文审查：如何从相似算法形成不同贡献

- ICC 的新对象是基于低秩权重的**增量校正**；GOC 的新对象是缓存梯度变化与轨迹拐点。
- L²P 证明若干历史外推可写作固定系数线性组合，再通过少量数据学习系数；LearniBridge 发现跨 prompt 校正共享子空间。
- BudCache 从阈值触发反转到固定预算的终端质量优化；AutoTarget 从“何时缓存”推进到“缓存哪个张量”。
- DisCa 将少步蒸馏与学习特征预测协同设计；RECAP 将稀疏 Attention 的误差与后续缓存反馈联合处理。

这些论文被接受/发表的事实与其**实际审稿接受原因**不同；以上仅从正式论文提供的科学论证与实验对照归纳。正式比较需核查是否有 NoCache Dense、最邻近已有方法、等预算控制组、组件消融与独立模型/步数，不能以“作者和干净模型比”概括全部实验。

## 3. 最短的后续核读清单（OPEN）

- DisCa：逐表核对四步 checkpoint、Restricted MeanFlow+predictor 的训练/评测身份；找出同 checkpoint 的 predictor-only 对照。
- AutoTarget：阅读 solver 不变性条件、哪一类缓存张量可保持 Euler 轨迹；排除把单纯张量选择包装为全新方法。
- RECAP：读取 `operator.py` / `controller.py` 的付费 coarse/mask/log-mass 路径及缓存有效性；复核其与单请求四步的兼容。
- LearniBridge：核对 `wan/run.py` 及 `wan/README.md` 中 50-step / CFG 的默认命令和训练行为；核对所报告 VBench 的设置。
- ICC-FFN：按旧 [移植合同](../../20261107/ffn_baseline_reproduction/ICC_FFN_PORT_SPEC.md) 完成同四步数值/显存验证（尚未据本簿执行）。

**身份**：此文献簿不产生新的实验数据/确认性性能，也不将用户报告的最新 ICC/Chipmunk 结果当作已由仓库收据独立核验。