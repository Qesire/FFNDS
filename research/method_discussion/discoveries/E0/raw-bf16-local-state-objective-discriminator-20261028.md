# 用原 BF16 的有限重放拆开局部误差、状态反馈和目标失配

**OBS/SRC：步长解释了旧轨迹漂移的大部分增长量级。** 原 `Wan22StepDistillScheduler.step_post` 为 `z_next=BF16(z-a_t F_t(z))`，`a=[.0625,.1041666865,.2083333135,.625]`。末步权重为首步十倍；同向恒定 flow 偏差在此调度下会给累计误差首→末十六倍。旧六视频实测累计 norm 为 18.01–21.08 倍，而除以累计步长后仅为 1.126–1.317 倍。不能据此唯一归因于步长：时间、expert、状态与 BF16 rounding 同时变化。

| 旧来源/方案 | `||Δe||/a`，steps0–3 | `(||Δe||/a)^2`，steps0–3（千） |
| --- | --- | --- |
| 0 energy | 467.025, 562.236, 621.200, 685.413 | 218.112, 316.109, 385.889, 469.791 |
| 0 random | 463.379, 549.615, 609.061, 645.354 | 214.721, 302.076, 370.956, 416.482 |
| 12 energy | 557.581, 533.788, 642.944, 679.536 | 310.896, 284.929, 413.376, 461.769 |
| 12 random | 575.349, 553.885, 707.075, 712.287 | 331.026, 306.788, 499.956, 507.353 |

这是 `Δe=-a(F_S(x_S)-F_D(x_D))+(ρ_S-ρ_D)` 的 post-state 标量，`ρ` 合并各自原生 FP32 减/加运算与最终 BF16 cast 余项；不能改名为同输入 flow error。精确平方值、输入 SHA 和零模型/零 tensor-read 工作见 [stdlib 标量审计](../../../20261028/COMPACT6016_STEP_WEIGHT_SCALAR_AUDIT_20261028.json)。B-F 的同 teacher-state final/oracle head 差距仍为约45–98倍；它提示可表示空间与已训预测器之间有明显余量，没有区分特征限制、优化和 gated-Y 目标。

**建议下一固定 scope：原 BF16 low/block39、最后两步、两暴露来源，12 个 full-flow replay，零训练、零视频解码。** 先做可部署的表示控制，去掉新预测器这个自由度：保持原 raw `W1/GELU`，仅用原 raw `W2` 构造一个固定 rank512 输出投影。行形式 `Y=H A+b`，候选 `Yhat=(H B)Phi^T+b`、`B=A Phi`；`Phi` 包含两 step 的原 `diag(g_t)K_t` head 空间，以 raw W2 在正交补中的 weight-PCA 补足512维。`b` 始终保留原 b2，等价于中心取 b2；无需 teacher Y 在线、无需旧 HiF4 scale/QDQ/训练权重。静态 Phi/B 转 BF16 并回读，实际三 GEMM，保持 GELU(tanh)，无新中间量化。它是原 BF16 的表示控制，不是原6016学生的跨格式重标。

先取得四个 Dense neutral flow 的真实 gate/head；若同 step 两来源 conditioning 不一致则停止这个共享 head-span 构造。原位 full-M readout 中记录 q、variance、direct/radial/signed-cross 和 BF16 余项；连续 head numerator 保持只是一条代数预测，BF16 export 后必须实测。此候选故意保持 full13824非线性特征，不扫 width/rank/support；原 prefetch 的 full W2 成本继续付，不据三 GEMM MAC 宣称系统加速。

| 每个来源的实际调用 | 次数 | 用途 |
| --- | ---: | --- |
| `D(x_D2), D(x_D3)` | 2 | 与新 raw BF16 baseline 的 FP32 flow、BF16 post 各自 bitwise neutral；捕获原标签 |
| `S(x_D2), S(x_D3)` | 2 | teacher-state 同输入近似误差，前缀全 Dense |
| `S(x_S3)` | 1 | `x_S3=step_post(x_D2,S(x_D2))`，沿候选自己的上一状态 |
| `D(x_S3)` | 1 | **付费 teacher**，拆掉状态传播与近似误差变化的混杂 |

两来源合计6 Dense +6 candidate full-flow、480 FFN、12 head、60 pre MM；只有6次 low/block39 的 W2 被替换。重放一律使用新 raw high/low checkpoint SHA及相同 scheduler/attention。四次 neutral 全通过后才执行候选；独立 flow 都显式 reset low 的 first buffer，成本492次 block load/12 reset，保留预取。另付2次 T5/2次 VAE encode（336 T5 MM/48 T5 block load）；无 decode/CLIP/noise draw/正式评分。新 baseline 的8个 flow及8个 post 中，本 scope 只读两 late-step 对应输入/标签，不宣称它们已包含 text/VAE conditioning。重新编码原 source image/prompt，原生 `step_pre → runner.model.infer(inputs) → step_post`；需要新有限 tap，不能绕过旧 tap 的160-site完整视频验收条件。

**可区分的结果**：令 `a=S3(x_D)-D3(x_D)`，`p=D3(x_S)-D3(x_D)`，`c=[S3(x_S)-D3(x_S)]-a`，则实际 `S3(x_S)-D3(x_D)=a+p+c`。保存三向量能量、全部 signed cross 和逐时间槽分解；scheduler 层另单列 BF16 rounding 闭合。

- `a` 小、`p/c` 主导且与旧误差同向：支持状态反馈或状态分布匹配作为下一训练对象；下一步应采 `D(x_S)`，而不是只加 teacher-state 行。
- `a` 已大、`p/c` 小：优先同输入表示/重构；本表示控制若接近其投影 reference，而旧可学习图远离 reference，才值得检验窄特征/系数预测问题。跨 raw/HiF4 不能直接归因。
- 连续 q 保持、head/flow 误差主要径向或 native remainder，或者 gated-Y 与 `a_t²·head/flow SSE` 权重/排序明显相反：支持 normalization/目标失配为下一具体 loss 对照。用共同的固定目标能量与 source 等权报告绝对及相对误差；没有两次同图同预算拟合就不称已证明 optimizer 或训练目标因果失败。

**LIT 给出的实际动作**：[TinyFusion](https://arxiv.org/html/2412.01199v1)及[固定作者源码](https://github.com/VainF/TinyFusion/blob/a147de7c33c2bc12872b9913f459119068b81ac8/prune_by_learning.py)共同优化结构与可恢复参数，执行的是任务训练 loss；它不是“再最小化局部校准 SSE”。[SVDtrunc §3.3](https://arxiv.org/html/2609.04646v1#S3.SS3)同时用 RF velocity、全模型 velocity KD、沿各自前缀的 feature KD；其 Eq7 的 student 输入是 `hhat_(i-1)`，不是 teacher 输入。该作者仓库本次仍只见 README/待发布训练代码，不能虚构 loss 实现。[Self Forcing §3.2–3.3](https://arxiv.org/html/2506.08009v1#S3)与[真实 rollout 源码](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/pipeline/self_forcing_training.py)将自身历史带入训练并只在选定 step 保留梯度；这里只迁移“状态分布判别”，不将当前双向四步 Wan 改称 AR。当前 LightX2V 的 `FlowMatchingTrainer.compute_loss_on_sample` 和 DMD `run_back_simulation` 也是两个不同目标/状态机制；不是所持公开 checkpoint 的训练溯源证明。

**决策**：先冻结这12-flow表示/反馈控制；不自动追加训练、全80函数或学习率/宽度 sweep。若要直接评估已训新 raw 学生，应以新的 raw producer/训练资产替换 S，再重新登记同样四角设计。上述提议尚未执行；本次只读源码/原文及 stdlib 标量计算。
