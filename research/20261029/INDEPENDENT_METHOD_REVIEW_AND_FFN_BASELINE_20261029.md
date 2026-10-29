# FFNDS · 独立路线评议、实验前复审与免大训练 FFN 正向基线（2026-10-29）

**身份：独立研究评议 + 新实验设计请求；不是已实施的基线、Agent 同意票、GPU 授权或实验结果。** 审计时 main HEAD = \`0fc3671ce47c4c425f456d754161dbb0c1a2b8d3\`。依据 AutoResearcher 的 CR / DF / PRD / MEAR / WARD、FFNDS 的 experiment-evidence skill、真实原始 BF16 160-site 全深度状态与视频质量收据；并于本日检索原论文及公开作者仓库。禁止把上述设计直接当作已复现的方法。

## 一、研究现状及独立审查判断

科学主线已经从简单阈值、末层 SSE 转向完整模型函数结构、时空机制和真实视频行为，这符合成功论文的「观测→可推广结构→新计算动作→独立质量/成本验证」范式。但 **采到的结构 ≠ 证明新颖 ≠ 有可用方法**：

1. 完成原 BF16 全160位置观测、两曝光源 Dense/value10/dcac10 的 960 native packets 与 640 个 L/P/I 四角记录、10源 21τ 统计、12张新开发图 Dense、7来源连续截图独立诊断，是真实成果（原记录和 hash 不改）。
2. value10/dcac10 全20源视频连续 RAFT 相对同 BF16 Dense 仅约 0.451/0.439；shared/step-r960 约 0.490/0.477；局部 AC/SSE 优化无法替代运动验收。历史 UB1 HiF4 数值家族不同，不混入原 BF16 的 Dense 配对。
3. L/P/I 在 gY AC 的明显抵消可由近似正交投影性质解释，不能直接认定发现新的视频运动规律。在理想线性投影 Q(h)=(hW)Π+b、D(h)=hW+b 中，L=−h_DW(I−Π), P=(h_S−h_D)W, I=−P(I−Π)，故 <P,I>=−||I||²，<L,P+I>=0；实际 gate/BF16/下游使该关系非严格。真正需要查明的是被过滤的方向对消费者、运动与条件语义的有限因果影响。
4. 时间固定 xy 的弱 lag1、高 rank 只刻画既定观察轴，不等于运动对齐后的有效表示或后续行为敏感性维度。因此「时间有统计结构」目前属于研究资产，不是第五方法，也不是可直接压缩定律。
5. S1 跨步共享/r960：可信比较器但目前未保质量；S2 时空行为：最强发现入口；S3 提前物理合法组动作：可能形成真实 W1/W2 省算；S4 固定随机草图变 rank：无偏性经典、W1 仍支付，必须超越等预算静态及已有动态 rank。M-B/G13 仅为功能诊断与竞争解释，不再串行地控制所有路。
6. 方法讨论队列里的 S1/S2/S3/S4 是 research seeds，不得凭文档描述自动标记 INDEPENDENT_PEER_SUPPORTED / NATIVE / VIDEO / ACCEPTED。

**新的跨机制可证伪问题**：FFN 中是否存在局部能量小、但对最终主体动作/相机/语义响应大的方向或物理函数组？在执行 W1 前哪些低成本信息能够识别这些方向/组？这是由行为敏感性激发的新研究假设，而非本轮已发现的定律。

写为 f_l(x)=b_2+sum_g GELU(x W1_g+b1_g) W2_g，局部动作误差 δu 在适用线性化下的下游风险 R≈δu^T J_>^T M J_> δu。这个 Jacobian 风险泛式已有近例（例如 2ndMatch）；原创性必须来自新作用结构、合法提前信息与实际低成本动作，而不是重命名 Jacobian。若 J_> 的方向性不可稳定预测，应如实收束该假说并转别的表示/算子。

### 相似论文及边界（原始资料，2026-10-29）

| 论文/作者路径 | 真正省什么、何时决策 | 原复现条件 | FFNDS 中的身份 |
| --- | --- | --- | --- |
| Chipmunk, https://arxiv.org/abs/2506.03275 ; https://github.com/sandyresearch/chipmunk/tree/main/examples/wan ; https://github.com/sandyresearch/chipmunk/blob/main/src/chipmunk/modules/mlp.py | 缓存上一步 MLP/attention 值并稀疏更新增量，带硬件列稀疏 kernel；原始视频实验主要多步（50 steps） | 作者提供 Wan2.1 适配与源码，原论文 Wan2.1 2.67× 含 step caching，是作者系统/模型配置而不是 FFN 单项增益 | **首要视频 DiT、FFN 机制真实复现基线**，且首选分离 MLP-only 与联合 acceleration |
| TEAL, ICLR2025, https://arxiv.org/abs/2408.14690 ; https://github.com/FasterDecoding/TEAL | 对 MLP 输入/隐藏作幅值门限，可结合稀疏核；无需继续预训练 | 原版 Llama SwiGLU、FP16 sparse kernel；Wan 原始 BF16/GELU 未直接兼容 | **首要训练自由 FFN activation sparsity 基线**；需明确 input-mask/W1 与 postGELU-mask/W2 的不同收益 |
| Prox, https://arxiv.org/abs/2607.27591 | 低价代理预测中间 channel 排名，再精确计算选中通道，力求真正省前后 projection | 原始针对 SwiGLU 的三线性支路；Wan GELU 为两线性支路；截至本次审查未核实作者可直接运行的官方发布代码 | **算法/信息时点的最近先例**，待源码和 GELU 映射评议后再考虑迁移 |
| FIS-DiT, https://arxiv.org/abs/2605.11869 ; https://github.com/Tencent/KsanaDiT | few-step 下跨层 frame-interleaved skip/refresh，整体 block/operation 级稀疏 | 原论文包含 Wan2.2，但截至本次检查作者项目 README 仍把 FIS-DiT inference code release 列为 in progress | **few-step 视频正向机制的强比较器**；源码释放前只允许「按论文重建」，不声称官方忠实复现 |
| SVD-Cache / DiffSparse / 2ndMatch 等 | 分别为子空间时间职责、可学习 token 分配、Jacobian 响应保持 | 训练预算/对象/可用源码各不相同 | 机制级邻近工作与消融，非免训练 FFN 直接复制品 |

**避免错误对比**：作者所报 50-step 的总系统倍速、LLM 解码倍速、Wan2.2 全 block FIS 的倍速，均不能当成 Wan2.2 四步 FFN-only 的预期实测。实验需记录训练量=0或仅少量统计校准；适配修改和底层 kernel 改写与作者方法的版本关系分开记录。

## 二、从可信的优秀方法中建立两个可比较对象

### B1 — Chipmunk MLP-only 作者机制（最高优先级）

先固定作者代码 commit、许可证、Wan2.1 原生/测试 harness、precision、硬件与 steps；对比作者 vanilla Dense 和其 MLP-only 增量策略，复核原论文「缓存/稀疏选择/实际 ColumnSparseGEMM」是真工作而非伪掩码。**第一阶段原作者语境复现**可作为最强可验证成功参考，不要求用 FFNDS 旧模型冒充同一结果。

**第二阶段独立 Wan2.2 四步迁移**：明确 high/high/low/low 是两个 expert，每 expert 仅出现两次；禁止跨 expert 错复用缓存，t0 首次执行要建立缓存，t1 同 expert 是有限复用窗口，切换 expert 必须分开建/失效。按实际模型时序核对 step 顺序及模块缓存 key=(expert,block,shape,source,seed,step)。对照：
- Dense 同 backend/模型/seed（必须保持 Attention dense）；
- dense-equivalent cached full-update, 浮点/舍入等价性；
- MLP-only Chipmunk 正式作者门限与稀疏更新；
- 可选作者含 Attention/step-cache 的完整组合，**另列为系统级正向控制，不能计入 FFN-only 的质量或速度收益**。
- 设置 1–2 个低风险保守稀疏率仅用于诊断/适配，不复用作者 50-step 比例当既定目标。

应预先回答的关键证伪问题：两步 expert 缓存增量分布是否仍集中？如果同专家两次 FFN 状态差异大，Chipmunk 在四步 Wan2.2 上可能只有「成功论文的高质量系统基线」，而不会成为可加速的 FFN-only 配方。这是对 FFNDS 的有价值结构结论。

### B2 — TEAL MLP FFN 幅值门限（强基础校准基线）

在 Wan FFN = W1 → GELU(tanh) → W2 上首先区分两种候选，不能把它们的计算节省混写：
- **B2a preW1 input magnitude mask**：x→mask(x)→W1，配合可验证的 sparse GEMM 省相应 W1 输入乘加/读存；对 W2 保持 dense，记录稀疏 kernel/pack + full FFN wall。若只是密集 GEMM 的 zero-masking，身份只能叫质量诊断。
- **B2b postGELU h magnitude mask**：h→mask(h)→W2；已完整支付 W1 和 GELU，只可能在真实稀疏 W2 kernel 下省 W2。对照 postGELU random (等支持比例)、静态层级阈值、source-held 校准 threshold；无需大型训练。
- 先使用已有 Dense 原生 h 数据离线绘制 **全160 site** sparsity–gated output error、不同 source/step/block/latent 时间/空间分层和支持重合；若能用现有保存 tensor 评价就不申请 GPU。
- 对局部质量有希望者，再建立真正 B2a/B2b 稀疏 kernel + 全160真实自由视频。原 TEAL 是 FP16 LLM，Wan BF16 的数值/性能差异必须记录；不能把未编译的 BF16 kernel 结果称作作者性能复现。

**成功基线的双层定义**：
- LIT-REPRODUCED：在原作者模型/数值/步骤和原生代码上重现其质量、算子及系统结果（或有解释的偏差），是「论文成功复现」；
- WAN-POSITIVE：在 FFNDS 实际四步 Wan2.2 同 Dense 配对下，完整160 site 自由生成的主体动作/相机/语义/时序与质量达预先冻结要求，且真实 FFN/System 成本形成可报告 Pareto 优势。
同一方法可在第一层成功、第二层失败。这不是矛盾，是本课题需要的对照与科学发现。

### B3 — 若 B1/B2 在 few-step 不适用，重新审查 FIS

FIS-DiT 的 few-step frame 计算冗余是已发表技术报告中更贴近当前 Wan2.2 的可行正向方向，然而它是整 block/frame skip，**不是纯 FFN 稀疏**。可以在源码可得后建立「整体 Wan2.2 正向基线」与「FFN-only 受控改造」，两者不可互相冒充，也不能把它当作我们新方法。先核验实际 release code / paper 实验细节，必要时联系作者或按文献重建并如实命名。

## 三、实验执行前必须再次独立评议（所有新的 E0/CPU/NATIVE/VIDEO/SYS 作用范围）

**正式机制合同：** [EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)。此处不因旧 method review 通过、旧 scope 有 RUN，便自动允许下一个 scope。尤其重要的是要**重新审查实验本身**，不只是复述「方法值得研究」。审查必须在 commit/code/ckpt/cohort/budget 等实际 scope 冻结之前重新完成，在提交调度前再验证没有关键改变。

实验设计需能回答具体科研问题。设两个会产生**相反可测预测**的机制解释，指出什么输出会支持/推翻、出现中间情况如何修改下一动作。至少涵盖：

- **论文忠实度与最近先例**：作者 commit/公式/正常运行的原始实现，WAN 改动点，训练和校准范围；真正新的主张与作者方法关系。
- **计算对象/合法性**：Wan high/low expert 路由、40blocks×4step、gate/e5、GELU/bias、BF16/HiF4/NVFP4；代码关闭、候选 mask/补算信息是否在线可得、cached/quantized 数据何时生产、free student state、native sparse kernel。
- **对照/等价性**：同模型/精度/seed/initial noise/调度、真实无操作 neutral、random、保守静态强基线、作者原实现；作者论文结果不混同。
- **来源与指标**：独立 image-cluster、训练/调参曝光和新 heldout、主体/相机/背景/动作/时序与 RAFT、source-level 统计，不能从 source×step×layer×token 伪增 n。
- **资源/效果**：actual RUN/SHA、可调用 GPU、适用 session_max_GPUs（旧适用 scope 两卡）、原生 kernel / observer / packing / cache / prefetch / offload、FFN与全系统时间，公平冷暖机。
- **检验能力**：首个实验最小可辨识版本、预期成功/失败各自改变哪个机制判断；如果只有「调参后找到能看的视频」则先修正科学设计。

**独立性**：主 Agent 草拟并自审执行设计，另一真实独立 Agent（非作者同一上下文）审其反例、代码/论文忠实度与实验性可判别；不同 Agent 的结果原样保存、异议由主 Agent 逐条处理，最终 reviewer re-check。禁止预填赞同票；尚未调度就标 REVIEW_REQUESTED，尚未投票不能标 APPROVED。没有真实可用独立 Agent 接口时，保持等待评议，科学草案和无实验性的文献阅读可以继续。外部人手/算力一旦满足，另行完成实际审查与范围登记。

本轮仅完成独立评议与仓库协议登记：**没有执行 B1/B2 的任何源代码、模型/新视频，未申请/冻结新 GPU 范围，未声称它们是 Wan 上已经好的稀疏方法。**

## 四、执行优先级和交付

1. P0：独立 reviewer 先审 B1/B2 的两个完全不同实验目的、作者版本与迁移合法性；尤其检查四步 expert 的缓存时间窗口和 B2 的 W1/W2 成本。
2. P0：基于已有全160 Dense packets 计算 TEAL 门限 baseline 局部 Pareto 与来源级错误分层（新 CPU 实验也须执行前审查）；同时做 Chipmunk 作者源码/模型适配静态差异表。
3. P1：在真实原生 GPU 条件、reviewer 合格结论和 ACTIVE 有限范围登记后，分别进行作者条件的 reproducibility 与 Wan2.2 少步迁移 neutral/pilot；保留第一位失败、所有新版本和实际成本。
4. P1：一条成功的全160视频质量—成本曲线建立后，将「原论文方法在哪些视频动作/step/block失败」与 S2 学到的行为方向、S3提前合法组信息交叉研究；不是仅继续增加局部 SSE 曲线。

期望产物：论文-原码-算子映射、真实复现版本 SHA、基线原生 runner、独立复审记录、Dense/candidate 视频与指标、FFN/全机成本、逐层/step/source 归因和至少一条据此提出的新结构假说。
