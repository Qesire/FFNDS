# FFNDS · 十八篇新增直接/交叉邻域论文（2026-10-26）

> **新版本51项（原33+新增18）**。本文件记录新增研究的科学对象、最强竞争和原始可核验的来源。所有论文条目的详细出处及实际阅读等级在 [PRIMARY_LITERATURE_INDEX.json](PRIMARY_LITERATURE_INDEX.json)；对五篇首轮深读仍保留 [五篇方法审查](FIVE_PAPER_METHOD_AUDIT.md)。本次新增工作的绝大多数仍停留在官方摘要、论文门户片段、作者项目页；不是18篇全部方法/消融/源码精读。

## 重大查重修正：四步Wan与FFN channel已有很强近邻

[Dynamic-in-Few-Step, arXiv2607.06631](https://arxiv.org/abs/2607.06631)在Wan-14B四步蒸馏上联合动态结构稀疏，训练stage-specific Mixture-of-Models并引入progressive training和output rollout；作者报告额外24% per-step FLOPs与约1.2×追加wall提速。**这是动态结构和少步联合优化的最强新近邻**，而非“只研究单步信息”的边缘例子。摘要没有钉住具体Wan2.2 I2V checkpoint、高低专家identity或每步被剪的FFN坐标；原方法与正式源码仍需查证，不能将报告算量全部归功于FFN。

[FIS-DiT, arXiv2605.11869](https://arxiv.org/abs/2605.11869)直接研究few-step Wan2.2/HunyuanVideo1.5的latent-frame interleaving：它认为跨去噪step的缓存机会因少步而减少，转而在网络深度上交错更新视频帧，使所有帧仍能获得覆盖。此处压缩的是**frame×block计算**，而不是hidden channel，也不要求跨step reuse。作者报告2.11–2.41×端到端提升；Tencent/KsanaDiT官方README仅说明FIS-DiT代码“正在整理”，检查的2026-05快照树未找到FIS实现，不当作已开源可直接复现。

[Taming Diffusion Transformer / Mobile Video DiT, arXiv2507.13343](https://arxiv.org/abs/2507.13343)公开基于知识蒸馏的block/head/**FFN channel**三粒度物理剪枝，结合压缩VAE与adversarial 4-step；最终手机fps属于整个多模块方案，FFN裁减单独的真实收益仍应在原论文消融里拆开。任何“首个FFN结构通道剪枝”的新颖性表述都必须经过这篇方法核对。

[PARE, arXiv2605.27336](https://arxiv.org/abs/2605.27336)已在Wan2.1-14B I2V/T2V研究宽度压缩＋以内容和timestep驱动的**block router**；宽度侧强调motion-aware attention heads而非直接FFN hidden。不能把“动态步长/输入”本身当作 FFNDS 新颖性。

## 来自更基础数学的重大线索：低能量与实际功能不是一回事

[NerVE, ICLR2026](https://arxiv.org/abs/2603.06922)在LLM原始FFN前后测 spectral entropy、participation ratio、eigenvalue early enrichment、Jensen-Shannon divergence，发现非线性会**重新注入原本弱的谱方向方差**。它研究的不是视频加速，但从数学上提示：若把FFN所承载功能预先等同于线性低秩/大能量坐标，就可能忽略经GELU激活改变的函数方向。它不是一个可直接用来判断Wan省算的在线指标；应与Wan实际Q2以及视频因果功能分开研究。

[SVD-Cache, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)发现跨步主子空间平滑、残差虽低能量却较难预测。其EMA预测principal、直接复用residual，与“低频/小能量应当直接舍弃”是完全不同的科学主张。检查BlackMaple1203/SVDCache官方仓库时，main tree只发现空骨架；不能称当前拥有可执行实现。

[SPRINT, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e7b28b1eb786547e1b452a0f38eac877-Abstract-Conference.html)把浅层全token细节与深层稀疏token计算通过残差融合；它主要是DiT训练阶段和ImageNet/SiT而非Wan FFN，但与“强行只保留主体token或只测首末层”形成有用反例。

## 对FFNDS源图的关系必须严格区分

| 研究主题 | 原始强近邻 | 实际改变的计算单元 | 与Wan四步FFN有何不同 |
| --- | --- | --- | --- |
| 结构与去噪联合优化 | Dynamic-in-Few-Step、FastLightGen、BLADE、CineMobile | step-specific子模型 / block / attention / step count | 可以有训练或完全新模型；不等于原FFN中天然通道可删 |
| Frame×layer时空覆盖 | FIS-DiT、SPRINT、Test-time Sparsity | latent帧、token/残差、跨层refresh | 不要求post-GELU通道自然稀疏，可能帮助解释少步与背景稳定性 |
| 物理缩窄FFN | Mobile Video DiT、PPCL、Neodragon | block/head/channel，原模型被结构剪枝并恢复 | 原FFN channel剪枝早已有，不能只以同样对象自称新算法 |
| 内容/步条件预算 | PARE、ProCache | router选择block/token及其缓存 | 不是逐FFN神经元，在线router与训练成本必须计入 |
| 主/残差功能分工 | NerVE、SVD-Cache | 表征谱、主/残差预测与复用 | 功能几何观察不是自动因果必要性；频率SSE不等于机制 |
| 质量与轨迹 | TMD、Adaptive Video Distillation | early backbone / flow head、DMD时序正则 | 强调完整生成监督和时序行为，不直接提供FFN少算 |
| 注意力硬件类比 | Veda、BLADE、Attention Surgery | attention tile-skip / sparse mask / hybrid linear | 注意力的QK结构与GELU/W2函数不同，借鉴结构→算子路径 |

## 待进一步按原文及源码深入而非盲目预选路线

1. **真正最接近的直接工作**：Dynamic-in-Few-Step具体剪枝粒度、每步submodel权重身份、四步蒸馏训练损失/费用和部署方法；Mobile Video DiT的FFN-channel sensitivity评分及是否独立报告FFN剪枝的VBench影响；FIS-DiT的frame交错调度是否作用于Attention、FFN以及残差更新顺序。
2. **关键竞争数学**：NerVE的非线性谱扩散是否说明更多实际功能维度，还是仅由activation尺度/采样组成；SVD-Cache的低能量残差为何重用而不是丢弃，它的短序列条件能否匹配四步Wan。
3. **新颖性与成本**：FIS-DiT等“跳frame计算”需要保留全视频所有帧的残差/attention上下文；PARE的router、PPCL恢复训练和Dynamic-in-Few-Step多阶段student都要在计算/显存/训练预算中计价。
4. **多维终端作用**：运动的相位、背景光照、细节闪烁、camera motion等现象在视频质量上单独衡量；Wan2.1/I2V、Wan2.2双expert、四步FLUX不能混称同一实验协议。
5. **继续扩大搜索**：查询经典非线性功能表征、frame子采样和token局部更新的近邻以及可验证的低训练方案，从引用链寻找真正不同于以上已知动作的解释；文献地图不应该成为新的封闭候选空间。

**审查身份**：所有新数字仅为各论文作者报告；本轮没有新增FFNDS native结果、没有独立测试任何作者模型、没有证明FFNDS新方法已成立。
