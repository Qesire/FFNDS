# FFNDS · 扩展文献的源码细节审查（2026-10-26）

> 此轮优先补足被早期33篇机制表遗漏的直接Wan四步/FFN channel/函数谱研究。不是执行研究模型。不同材料核验深度明确区分。

## NerVE：不是只给出一个“新的频谱loss”

[ICLR2026 NerVE论文](https://arxiv.org/abs/2603.06922)与[作者开源源码](https://github.com/nerve-eigenspectrum/NerVE)，实际检查到Git tree 35a8f957a65b6d925d126712e2deec6a560636bc、源码`nerve/metrics.py` blob 417a7bbc37018f3a51f24449cc5f6b62da36e67c。它对pre-/post-GELU的hidden token观测做中心化协方差，在完整维度的eigvalsh求降序特征值，再定义：

- spectral entropy：归一化特征值的Shannon entropy；
- participation ratio：`(sum λ)^2/(sum λ²)`；
- eigenvalue early enrichment：谱累积曲线偏离均匀谱的积分尺度；
- pre/post Jensen–Shannon divergence：两个各自**按值排序**的谱的分布差异。

这篇研究从FFN激活本身如何重塑表示容量出发，比“低能量误差频带可删除”更接近函数本征问题。**但实际源码只计算特征值，不保留特征向量，因此其SE/PR/EEE/JS不足以追踪一个具体Wan视频输出方向，也不能凭谱变化断言功能因果必要性。** 此外原Wan H=13824的完整协方差为13824²元素，应用到完整token必须评估统计/访存代价；用户并未要求运行这个诊断。

原文实际对象是LLM、GPT2/GELU及多种规范化/优化器配置，与Wan AdaLN/HiF4/Q2/蒸馏不同。存在开放对照：功能维度扩大是否来自真实新信息、激活分布/scale或原Q2 scale树；其作用需通过原生有限函数变化和视频对应区域确认。

## FIS-DiT：原始实现开源状态

[FIS论文](https://arxiv.org/abs/2605.11869)作者在Wan2.2/HunyuanVideo1.5等few-step模型提出frame-interleaved layer computation。实际读取[Tencent/KsanaDiT](https://github.com/Tencent/KsanaDiT)官方README（最近代码快照2026-05），有作者明示“FIS-DiT code正在整理”的声明；所查源码树中未见明确的FIS-DiT模块。该仓库支持Wan2.2推理和缓存、量化等框架功能，不能误将整个kDiT当成FIS公开实现。**论文摘要的2.11–2.41x是作者实验结果，不是本地复现。**

## SVD-Cache：主/残差功能和公开代码现状

[CVPR2026 SVD-Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)强调主特征可以预测，而难预测的低能量残差更适合复用。原论文与作者公开网站链接的[GitHub](https://github.com/BlackMaple1203/SVDCache)当次main tree只有单项空骨架，暂不能称为“官方代码实现已审查”。其研究功能是**跨step缓存/预测**，与NerVE的“FFN非线性使谱维度扩张”不是同一问题；对FFNDS也不等于低频SSE权重优化。

## Mobile Video DiT：直接FFN channel强近邻

[作者项目与arXiv](https://snap-research.github.io/mobile_video_dit/)，[论文 arXiv2507.13343](https://arxiv.org/abs/2507.13343)提出由教师知识蒸馏支撑的block/head/FFN-channel三层敏感性裁减、压缩VAE和四步对抗性蒸馏。**其FFN channel压缩属于物理改变矩阵尺寸的强先例**；不能用整套手机15fps证明FFN裁减单独具有该性能收益。还需原文/官方源码进一步判别FFN神经元mask是否按step变化、裁减通道是静态还是动态及微调开销。

## Dynamic-in-Few-Step：四步Wan联合结构近邻

[arXiv2607.06631](https://arxiv.org/abs/2607.06631)直接把少步蒸馏和按step组织的结构稀疏统一为Mixture-of-Models，采用progressive training及output rollout，在Wan-14B四步报告per-step FLOPs降低24%、额外wall约1.2x。摘要没有细到“哪个原神经元如何跳算”、checkpoint是否当前Wan2.2 I2V或具体高/低主expert identity。下一步应追踪论文方法公式、损失、真实部署engine、每step width/depth和可读仓库；避免假设其全部省算来自FFN。

## 当前最强机制交叉张力

- **少步前提并不唯一限制**：DiffSparse Wan2.1使用25步、Chipmunk在多步Wan2.1利用缓存；但FIS-DiT已在few-step Wan2.2找到与步历史不同的帧级计算自由度。
- **更好的静态重构不说明函数必要性**：NerVE非线性谱和SVD-Cache时序残差提示数值能量既非功能唯一坐标也非可执行替代物。
- **动态稀疏与训练形成结构需要分开**：原生untrained Wan自身的物理稀疏机会不等价于Dynamic-in-Few-Step、Dense2MoE、PARE通过蒸馏与router学习出来的条件计算。
- **FFN-only范围与整块代价不同**：Mobile Video DiT的FFN channel修剪、FIS的frame×layer、PARE的attention width与block routing和RT-Lynx的GEMM input N:M减去的物理原子不同；比较必须有统一模型/视频/成本分母。

文献继续开放检索，新增18个工作是发现新科学空间的线索，不是18项必须实现的算法任务。
