# FFNDS · FFN 本征结构相关原始文献地图（2026-10-25）

**遵循**：AutoResearcher v2.4 WARD；2026-10-25 公共 Web Search 多组检索，审读对应论文/官方会议 HTML 的摘要和可访问方法页面。本记录是研究方向查新图，未宣称完成全部 PDF、代码或复现。检索关键词族：DiT MLP/FFN activation sparsity; token/channel/tile concentration; dynamic column-sparse delta video; few-step diffusion learned token sparsity; BF16 independent FFN importance; activation-vs-weight N:M; diffusion FFN downstream function.

| 原始出处 | 已报告的机制与本研究关系 | 下一轮必问的差异 |
|---|---|---|
| [Chipmunk, arXiv 2506.03275 / 作者代码](https://arxiv.org/abs/2506.03275)，[代码](https://github.com/sandyresearch/chipmunk) | 训练自由的 DiT MLP/attention 激活动态差分缓存、column-sparse kernels；Wan2.1 也是报告对象。 | 四步双专家的同专家重复次数与激活增量分布；不用其缓存原语能否得到不同的 FFN 结构？ |
| [RT-Lynx, ICML 2026](https://proceedings.mlr.press/v306/cong26a.html) | DiT 激活 N:M 结构稀疏加速，比权重 N:M 的效果更稳健，使用补偿和实际 kernel。 | 本 Wan GELU+W2 的可执行稀疏程度、N:M tile 几何；本文新意不能只是把 activation sparsity 加到 DiT。 |
| [Activation Concentration, arXiv 2606.00567](https://arxiv.org/abs/2606.00567) | 报告 DiT element-level near-zero 率可大幅高于 column-level 可利用率，不同架构有明显差异；其系统模拟与精确 Wan 不是同一对象。 | FFNDS 真实 token×channel 接近散点式分布是否可通过其他坐标/时间结构改变可利用粒度？ |
| [DiffSparse, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html) | 少步 DiT 的 token caching 与 learned 层稀疏分配，含 Wan2.1；训练代价需单独比较。 | 不以 token 缓存为前提的 FFN 计算重构，与轻量后训练有什么可区别机制？ |
| [Universal Properties of Activation Sparsity, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec3c79dc0c2b85532cfd1012a4aaa923-Abstract-Conference.html) | LLM FFN 激活稀疏性质及 diffusion-based LLMs 的统计研究。 | 不把语言模型激活分布直接移植到 Wan GELU 视频模型；先独立测试条件分布。 |
| [Sparsing Law, ICML 2025](https://proceedings.mlr.press/v267/luo25i.html) | 研究 FFN activation sparsity 的有效测量、架构与训练依赖。 | 形成可与真实下游功能和物理粒度对应的表征，而不只统计零率。 |

## 机制上最重要的相反证据

- **动态稀疏存在空间**：Chipmunk 利用激活变化的集中性；RT-Lynx 利用激活 N:M。因此 FFN 的省算并非一定依赖 FP4 尺度或传统通道静态剪枝。
- **细粒度稀疏不等于可执行**：Activation Concentration 讨论 element/column 落差，提醒把 token×hidden 近零率与实际 tile 可跳过率分开。
- **少步蒸馏会重写可用冗余**：DiffSparse 适配 few-step，Chipmunk 的缓存假设需要在四步、双专家上逐项核对，不能照搬其多步策略。
- **功能敏感性高于原始激活值**：模型输出的时空质量和剩余计算的价值需用实数贡献及有限响应区分，不能用对数值很小的激活作为唯一依据。

## 仍需向文献深处追踪

原论文的方法等式、控制变量、训练与推理成本、官方 kernel 真实访问模式、后向/前向引用；LLM 的通道稀疏机制与视频 DiT 的模型构件究竟同构在哪里。对最新 Wan 相关 FFN 方法持续查重，若发现紧邻方法则直接修改研究机制假说。**不是以文献数量配额替代科学创新**。

此地图只提供 FFN 本征研究的竞争机制线索；量化格式本身的优劣比较移入后续部署与实现专项。
