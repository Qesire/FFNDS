# 从局部函数重构改写为受扰轨迹上的功能重构

**新问题：** whole94固定原型已经在真实160处FFN运行，热态FFN边界9.707→4.880秒，但source0视频的40/80帧可见重影和纹理破坏。相同四来源的局部重构改善，不能直接替代完整视频质量。这里需要改变解释的对象：区分原轨迹上的局部近似、受扰输入上的函数误差，以及误差经残差/attention/四步调度后的作用。affine54此记录撰写时仍在固定范围运行，不预先写其结果。

这是源码、原论文、条件数学和已完成RUN标量的探索记录；没有新模型、梯度、拟合或科学张量实验，不授予后续执行。视频帧40/80是影片时间，**不是去噪step2/3**；当前只存了各去噪step的latent SHA，无法据此定位误差在哪一步放大。

## 1. 实际Wan更新及两个不同的校准目标

亲读 `Wan22StepDistillScheduler.step_post`，源文件 `.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py` SHA256 `2d560449e0559aea44d80afb539f9e97bd544e5e26abcc06236e61918d01e1db`。它先将flow转FP32，计算 `z-sigma*flow`，非末步再加 `sigma_next*flow`，最后回到latent dtype。四步high/high/low/low；已接受sigma为 `[1,.9375,.833333313,.625]`，末端sigma取0。

在实数表达中令 `h_k=sigma_k-sigma_(k+1)>0`，更新为

`z_(k+1)=z_k-h_k*v_k(z_k)`。

实际代码的两个FP32乘加不保证与一个融合表达逐位相同；下式以rho保留舍入差。令student状态为 `z_k+delta_k`，其flow为vhat，则

`delta_(k+1)=delta_k-h_k*[vhat_k(z_k+delta_k)-v_k(z_k)]+rho_k`。

这不要求量化网络可微。再定义同一student输入上的算子误差 `e_k(u)=vhat_k(u)-v_k(u)`，可精确拆成

`delta_next=delta-h*[e_k(z+delta)+v_k(z+delta)-v_k(z)]+rho`。

因此有三个不同对象：teacher输入误差 `e_k(z)`、student输入误差 `e_k(z+delta)`、dense对状态改变的响应 `v_k(z+delta)-v_k(z)`。本次保存标签只覆盖第一个对象的局部FFN版本。局部均值更小不能决定另外两项；有限视频结果也不能单独判定是哪项主导。

若只在理想实数下要求下一状态恰好回到teacher，所需训练目标为

`vhat_k(z+delta)=v_k(z)+delta/h_k`。

这是**受扰状态上的离线训练目标**，不是部署时可以免费知道的修正项。h约为 `.0625,.104166687,.208333313,.625`；相应系数约 `16,9.6,4.8,1.6`，但初始delta为0，不能据系数大小推断哪一步最敏感。大delta、teacher不可见、多个输入共享一套参数、重参数化容量和舍入都能使该目标不可实现。这里没有启动训练。

## 2. 用原生混合路径代替未经验证的Jacobian假设

对固定输入、seed与确定性原生程序，记第j个功能单元为Phi_j，替代为Phihat_j。单元之间可包括原attention、norm、gate/residual、scheduler和最后decode。设G_j使用前j个替代单元、其余全部原始单元，则

`Fhat(x)-F(x)=sum_j [G_j(x)-G_(j-1)(x)]`。

其中每个差都是：在同一个**student前缀**上，原生dense后缀分别消费该处的真实替代输出与原输出。这是纯组合的望远镜恒等式，无需把BF16写回或QDQ视作光滑函数。它改变了诊断的单位：应问一次变化经真实后缀造成什么，而不只是当前位置损失多少欧氏能量。

只有在另行验证的连续近似/有限扰动线性区间，才可进一步写 `Delta_final ≈ sum_j T_j*b_j`，其中b_j应是实际门控和BF16写回后的扰动，T_j是相应后缀的传递近似。目标于是出现 `sum_(i,j) b_i^T T_i^T T_j b_j`，包含方向、跨层交叉项和相消；原来的独立局部ridge并未优化这些量。真实离散网络的普通导数可能在量化格内为0，不能把它直接当作有效有限传递。

完整枚举所有混合路径很昂贵，**本恒等式不授权161条生成**。最小后续观测可只选少量预先固定边界：同时比较teacher前缀和student前缀上的同一算子、以及一次原生dense后缀延续。它能区分输入偏移与下游放大；不必先扫组数或阈值。具体预算仍需单独登记与授权。

## 3. 三个竞争机制及不同预言

| 机制 | 不同预言 | 最小有价值的区别 |
| --- | --- | --- |
| 覆盖/输入偏移主导 | 同一固定原型在student输入上的局部误差显著高于teacher输入 | 同输入配对原FFN与原型，分别在两种前缀观测；不能用旧teacher标签代替 |
| 下游方向与跨层相干主导 | 局部误差相似，但实际gate/写回后某方向经过dense后缀造成更大flow变化，或多个误差相干叠加 | 同前缀有限干预＋少量固定后缀边界；正交欧氏分量不保证功能独立 |
| 轨迹校准目标不足 | 即便student flow接近teacher flow，已有delta仍被保留；受扰目标而非单纯更低局部SSE才对齐下一状态 | 比较对teacher flow的误差与对下一teacher状态的误差；受扰轨迹/target只允许离线取teacher |

空间共同函数、step/expert条件表示、原值向量重组、低成本线性控制、与训练后的功能补偿仍是开放切面；这三种解释不是排他的完整分类，也不证明继续使用当前affine/组支持一定成功。

## 4. 实际检索、原始依据与搜索改变

2026-10-26检索簇为“block reconstruction / cross-layer dependency / diffusion error propagation / calibration input mismatch / cross-timestep correction”。沿论文到作者代码、再到相近的另一条原始扩散工作；没有将搜索摘要当成实现验证。

- **BRECQ**：[原论文v2 §3.1–3.2、式6–8与Algorithm1](https://arxiv.org/pdf/2102.05426v2)提出跨层依赖与重构粒度的偏差/泛化权衡；不能据此断言整个网络拟合总优于局部。亲读作者 [block_recon.py，固定commit e455d62e](https://github.com/yhhhli/BRECQ/blob/e455d62e93c70351961f8991c913b59435bd165f/quant/block_recon.py)，文件SHA `5b00499e83ce5b90dfb325ffcbbf26b612c03906006eb4e8dd9abb5a762ce471`：默认mse，另有fisher_diag/fisher_full、asym输入及Adam更新；不能把默认实现称为总使用精确Hessian。本文只借鉴目标/粒度问题，未复现其CNN量化训练。
- **CTEC**：[AAAI2025正式PDF，式10/13/14、Algorithm1及Distillation节](https://ojs.aaai.org/index.php/AAAI/article/view/34039/36194)明确把前步状态差加入后步目标，用LoRA学习。上述Wan理想目标属于同类思想的调度器特化，**不是新的跨步纠错发明**。原文20/100步图像条件不能视为四步双专家视频验证；未找到并核验完整作者实现，也未借用其结果。
- **TCEC**：[v1 §3.1–3.3](https://arxiv.org/html/2508.12094v1)另走通道输出比例拟合与在线累积修正，并采用忽略Jacobian、时间局部等近似。本文不默认这些近似对Wan成立。其印出式7到式8的逆矩阵/索引转换需要独立核查，当前不照抄其累积公式或“只需相邻两步”结论，不泛化否定该论文的实验。

搜索改变了下一动作：whole94画面失败后，优先区分目标、输入分布和真实传递，而不是仅为当前支持增加几组。它也给出反例：扩大到整网蒸馏会增加数据、参数估计与过拟合成本，并非自动解决。真正值得推进的是**有实际省算的算子，在能解释自由视频的训练/诊断目标下是否还成立**，同时保留本次已实现的原生FFN收益与强dense参照。

图锚点为 `F03/F04/F06/F08/F09→F12/F13/G21→D09/P12`，新对象是受扰前缀和功能后缀。此处恒等式是ALG；某机制是否解释当前画面是OPEN；没有新算子选定或研究阶段升级。

## 独立审查后的补充

[另一作者的源码/原PDF推导](independent-trajectory-error-and-correction-20261026.md)确认了受扰目标的正号，并给出 `||delta_next||²−||delta||²=h²||d||²−2h<delta,d>`，其中d是student-state与teacher-state之间的完整flow差（理想实数更新）。这个有符号条件比仅测当前flow范数更能区分修复与累积。审查还明确区分：首去噪步外层delta为0，并不排除前几层替换导致后层输入离轨；固定scheduler latent为FP32，网络内部BF16/QDQ误差属于不同边界。

其原PDF页核验也确认TCEC印式8并非HTML误读：在t=T处已经与声明初值矛盾；按式7正确展开不含inverse。此反例只约束所见递推展开，不否定作者实测或其可独立比较的通道比例拟合组件。新增paired状态、后缀干预或训练仍没有从这一数学审查获得执行授权。

**部署数值这一前置竞争解释仍OPEN。** 全40层8.54%来自连续保存数据诊断；160pack的SHA/shape/finite与四处full-width neutral并未逐层验证partial bank与连续函数一致，旧partial对照只覆盖0/30。后续最少paired观测应先量化真正部署bank的same-input误差，再谈分布偏移/功能传递/轨迹目标；否则可能把部署数值差异误归因成新的科学机制。本次结果不唯一识别原因。
