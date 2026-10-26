# FFNDS · 原论文方法与真实 Wan FFN 计算图的逐边映射（2026-10-27）

本研究以 LightX2V pin 运行图 G0–G5 为基底，保留其 SRC 执行实线。四篇外部论文只建立 G7 的 LIT 参考分支，绝不意味着本 Wan 已实现、可加速或质量验收。固定对象：Wan2.2 I2V 4-step high/high/low/low 双整网专家；M=31668 tokens, D=5120, H=13824, 40 blocks, latent temporal=21, decoded frames=81；真实顺序是 norm2/AdaLN → Smooth1/Q1 → W1 → tanh-GELU → Smooth2 → 64→8→4 HiF4 Q2 → W2 → e5 gate/writeback → subsequent blocks/head/flow/scheduler/video。

## L34：Dynamic-in-Few-Step（K01→K03），step-specific 网络本体

[原始 arXiv 2607.06631](https://arxiv.org/abs/2607.06631)的作者摘要确认，step-specific Mixture-of-Models 是通过少步蒸馏与结构化稀疏联合训练产生的；含 progressive training、output rollout 和专门推理引擎；Wan-14B 四步时作者报告在已蒸馏模型基础上，per-step FLOPs -24%、额外 wall gain 1.2x。上述数字不是单独FFN加速。具体 block/head/FFN channel/group-mask STE、权重 gather 等细节目前主要来自 [alphaXiv二级解读](https://www.alphaxiv.org/abs/2607.06631)，未获得完整独立原方法正文核查，**不得标为原文公式已经审核**。

计算依赖：G02 step与expert → K01 训练所得 step mask/architecture → K02 该step紧凑参数和对应block/head/W1-W2算子 → K03 后续step rollout及生成质量。K02对F04/F09的关联是方法粒度比照，不是原HiF4计算被替换。原Wan的high/low整网expert按两个step切换，不等于论文中新训练的四个step模型。

## L35：FIS-DiT（K04→K06），frame×layer 而不是FFN hidden channel

[原始 arXiv 2605.11869](https://arxiv.org/abs/2605.11869)作者摘要确认少步情况下跨去噪step cache机会有限，于是选择latent frame维度上的交错计算，在多个block中轮换frame子集并覆盖全局帧位置；在Wan2.2与HunyuanVideo1.5给出作者报告2.11–2.41x整体提速。原方法详细mask、scatter与attention上下文尚待完全原文核查。

数学研究抽象（并非论文精确伪代码）：设每block的frame集合 S_l 是全部21个latent frame索引的子集，该层执行的有效 token 行数 M_l 随 S_l 而变。选择 Select_l、Block_l、Scatter_l 时，自注意的全部位置K/V与RoPE以及未选帧残差如何被处理会改变真实依赖，不能将 Select→Block→Scatter 当已证数值等价。映射是 G04–G06 / G17 / G21 → K04 frame选择 → K05 部分block计算（注意力与FFN共享行控制）→ K06 跨层覆盖和残差状态恢复 → G21。**21 latent bins 不等于81帧视频**。这研究的是M行及attention上下文，非H列。

## L37：Mobile Video DiT（K07→K09），真实物理FFN宽度裁剪和独立内存优化

[原始论文 arXiv 2507.13343v2，21页](https://arxiv.org/pdf/2507.13343)，本轮审阅第4–6页 §4.2 Eq.(3)–(6)、§4.4和附录相关说明。作者模型不是 Wan2.2。

Eq.(3)：block跳过公式是 y_b = m_b Block_b(x_b) + (1−m_b)x_b。Eq.(4)：线性输出维二元channel mask，y_l = Linear_l(x_l,W_l,b_l) ⊙ M_l。其训练后的mask可压缩为**真实更窄的稠密模型**，并非每次在线花费完整W1然后删输出。对FFN这一改变需要一致缩窄 W1 输出行及 W2 输入列，中间非线性保留；但本Wan HiF4 Q2的64→8→4 scale树在任意width改变后不可无条件复用旧尺度。

Eq.(5)：trainable affine F_i(student_group; Θ_i) 对齐teacher/student block-group feature。Eq.(6)：L = L_flowmatching + α L_distill，作者设 α=0.01。训练支路不是推理时额外GEMM。§4.2.3：自建2B超网28blocks、32heads、FFN H=8192，最终915M，删8blocks、12heads，FFN width降低25%。敏感性扫描显示FFN dimension比block、head更敏感，因此作者偏向保守裁FFN。该结论**只在论文模型/评价设置成立**。

§4.4是另一个独立硬件分支：在FFN扩展维将权重分tile并融合中间激活，减少移动设备I/O而非必然减少理论MAC；作者报告约10% DiT forward增益。作者终端约15 FPS来自VAE压缩、剪枝、四步蒸馏与tile共同作用，不是FFN单项成果。G7映射为 K07离线mask/训练 → K08 narrow dense W1/GELU/W2 → K09 feature KD 和单独tiled IO，再连接F04/F06/F08R/F09/D09作为比较边。

## L50：NerVE（K10→K12），激活谱是一种离线函数结构观察

[作者论文 arXiv 2603.06922](https://arxiv.org/abs/2603.06922)与[官方源代码 nerve/metrics.py](https://github.com/nerve-eigenspectrum/NerVE/blob/main/nerve/metrics.py)，检查tree 35a8f957a65b6d925d126712e2deec6a560636bc；文件blob 417a7bbc37018f3a51f24449cc5f6b62da36e67c。对同一个采样集合的pre-GELU及post-GELU hidden激活 X∈R^(N×H)，计算样本协方差 Σ=(X−mean X)^T(X−mean X)/(N−1)；对排序eigvals λ_i，令 p_i=λ_i/sum λ，得到谱熵 SE=−sum p_i log p_i 和 PR=(sum λ)^2/sum λ²，此外还有EEE与pre/post的JS divergence。

原始程序明确使用 torch.linalg.eigvalsh，只返回**特征值**，因此它能估计有效维度与分布变化，但不提供具体的feature方向，也不证明某个Wan W2输出与视频对象、背景或光照的因果作用。完整H×H协方差至少O(H²)内存和昂贵分解，不能当推理时免费的mask评分。真正的G7关系是 F04→K10 preact covariance；F06→K11 postact eigvals→K12 SE/PR/EEE/JS，其后Q2/门控/视频是**开放的转移问题**，不能画成现成加速执行边。

## 原论文与真实执行图的统一边界

| G7 branch | 原论文改变的数学对象 | 当前 Wan 对照节点 | 当前证据等级 |
|---|---|---|---|
| K01–K03 | 每step网络参数/结构与生成rollout | G02, G04–G09, F04/F09, D09 | 原摘要，方法细节二级资料待核 |
| K04–K06 | 每block latent frame参与的M行和上下文 | G04–G06/G17/G21，W1/W2行 | 原摘要，详细mask与残差公式待核 |
| K07–K09 | 训练后FFN H列缩窄和device tiled IO | F04/F06/F08R/F08/F09 | 原PDF公式及方法已核，尚未Wan适配 |
| K10–K12 | 非线性前后协方差的特征值分布 | F04/F06，Q2后对照尚未执行 | 作者代码与公式核查，无方向因果 |

后续 Agent 应对每条论文新增明确声明的源码节点、信息可用时间、训练成本、数值量化/物理算子/下游消费者，并将争议保留在OPEN边，不把研究发现倒推成SSE评分方法。其它51项先例可按此模板继续映射，图谱不能停留于这四条支路。
