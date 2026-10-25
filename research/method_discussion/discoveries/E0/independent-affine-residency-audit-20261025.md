# 独立审查：affine 的有效性尚不识别其机制；dense residency 是另一条已测执行自由度

**日期 / 角色：** 2026-10-25，独立讨论审查 Agent。没有授予 method endorsement、候选选择或阶段晋级。

**新增认识：** 已读源码支持两个有限而实质的结论：另一来源固定 affine 加整组残差，在保存面板上优于其投影 MAC 匹配的 group-only 对照；保留完全相同的 decoded 权重可以减少重复搬运，并在已测视频上保持原 latent/video hashes。但是，前者尚不能区分“单元处于可折叠的局部热区”和“GELU 自带线性分量、偶函数分量经过 W2 叠加后较小”。后者减少的是分配/搬运，未减少 FFN GEMM。两者应成为相互约束的比较对象。

## 范围和独立性

已读根 `AGENTS.md`、`research/ACTIVE_RESEARCH.json.current_stage`（`E0_OPEN_DISCOVERY`）、讨论区 `AGENTS.md`、第一任务及 discovery 模板；使用 FFNDS domain skill、mechanism-research、pinned WARD v2.4 / harness。上游绑定为 `8b88a52d894568dc303db727decfbf13b5fabf3b`。

本次只读源码、协议、已有 RUN 标量和原始网页；JSON 的均值/计数仅为现存标量的描述性汇总。没有导入 torch、运行模型/数值实验、访问远端 worker、调度任务或操作 Git。只新增本文件，没有改动共享状态。下文 HF 引用均相对 `ffnds-derived`，不摘录私有资产路径、节点名、主机映射或硬件 UUID。

亲自读取并检查了以下七个协议中 `bound_sources` / `driver_sources_sha256` 和 `native_source_sha256` 所列本地文件的 SHA256；此次读取时全部匹配：`e0_all_block_tangent_saved`、`e0_fold_group_residual`、`e0_absolute_affine_fold`、`e0_fold_microbenchmark`、`e0_dense_weight_residency`、`e0_dense_residency_crossover`、`e0_all_weight_residency`（均为 `config/*_20261025.json`）。这绑定的是本次审查对象，不替代原运行的资产验证。

## 1. 重建实际数值路径和可以精确成立的式子

**OFFICIAL_LOCAL_SOURCE。** 原适配器 `.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py:652–692` 的每次 linear 是 Smooth scale 除法、回到输入 dtype、native activation QDQ、读取 decoded 权重并转为相同 device/dtype/contiguous，最后 `F.linear`。LightX2V `transformer_infer.py:366–424` 在 norm/modulation 后调用第一投影、`gelu(approximate="tanh")`、第二投影；gate 与 residual 相加在其后的 `post_process`。所以这里的数学 `x` 是实际 post-Q1 值，不是未经量化的 norm 输入。

采用列向量记号，令 `a=W1 x+b1`，原生 Q2 系数为 `q`，gate 为对角矩阵 `G`，则 decoded 的实数重建为

\[
y=G(W_2q+b_2),\quad q\approx S_2^{-1}g(a).
\]

选任何固定对角斜率 `K` 与截距 `c`，**定义** `r=q-Ka-c`，即可得到

\[
y=G(Ax+d)+\sum_{j=1}^{216}G W_{2,j}r_j,\quad
A=W_2KW_1,\quad d=W_2(Kb_1+c)+b_2.
\]

这在 decoded 实数图上是恒等式，无需假设 GELU 可线性化。只修正组集合 `T` 时，误差恰为剩余有符号向量之和 `−Σ_{j∉T}G W2,j rj`。因此，“affine 加全部组残差还原”通过，并不独立证明剩余项稀疏，也不证明选组分数最优。

`run_e0_absolute_affine_fold.py` 实际先重建全部 preactivation，再用此恒等式作诊断；它没有合成或部署 A。若未来形成固定 A，基项可在 W1 前应用；现有诊断取得全部 a/q 的成本不能被当成已实现的可省计算。按原始 64 通道组修正有合理的 Q2 依赖单位，但还需保留原生数值顺序和 GEMM shape 改变带来的舍入检验。

**MATH_CONDITIONAL。** 若只计理想投影 MAC，`MD²/(2MDH)=D/(2H)=40/216`；修正 k 组的绝对函数方案为 `(40+k)/216`，故 affine+27 对 group-only67、affine+54 对 group-only94 是精确 MAC 匹配。时间增量方案重算旧/今两份所选 W1 和一份差值 W2，成本是 `40/216+1.5ρ`；linear+27 与 group-only54 相差 `0.0023148` 个原 FFN 投影预算。它们均不包含矩阵形成、QDQ、索引/packing、缓存、gate 或真实 latency。

## 2. 跨来源校准：未发现标签混入；泛化范围仍有限

**SOURCE_CHECK。** `run_e0_fold_group_residual.py` 中，每个来源自己的 `cal_score` 使用自己的 `old['full_M_old_derivative_mean']` 计算自身 residual；到评价循环才通过 `train=cases[other]` 读取此分数。held 来源的线性项也使用 `train_old` 的 κ。`raw_score` 和 scalar-plus-vector affine 控制亦只取 `train`。因此没有出现“用 held κ 在训练源上挑组”的隐蔽交叉依赖。

`run_e0_absolute_affine_fold.py` 在每个来源自己的 uniform32 上拟合单元 slope/offset，自己的残差产生自己的分数；评价时整个模型与固定支持取 `cases[other]`。GELU_OLS 还用**训练源**的实际 Q2 均值重定截距，这不是纯 GELU 无量化拟合，但不含 held 标签。40/160 个 RUN case 的 `source_index != calibration_source` 记录与源码一致。

held 的当前 Q1、gate 是合法的当前输入/调制状态；诊断中的当前 q2 只应作为 exact correction 标签。`posthoc` 分支明确读取 held uniform32 标签，不能叫前置 selector，也不是独立测试：它在 uniform32 上选组，再同时在 uniform32 和 center128 报误差。

这里只有 source0/source12 互换；二者均已暴露。32 行按来源排除，不等于 fresh-source/seed 验证。当前源码没有发现实现泄漏，不支持把已暴露双来源结果升级为独立部署泛化。

## 3. 保存结果说明什么，以及不说明什么

**ARCHIVED_SCALARS。** 以下均是各 case 相对当前 gated 输出 SSE 的算术均值，非 pooled SSE、误差范数百分比或视频质量指标。

| 保存研究 / 面板 | affine / linear only | 加27固定组 | 匹配的 group-only | 其它固定对照 |
| --- | ---: | ---: | ---: | ---: |
| temporal，160 cases，center128 | 25.7569% | 20.2410% | 54组：33.5771% | other-source affine：30.6409% |
| temporal，160 cases，uniform32 | 18.5410% | 14.9803% | 54组：25.1732% | other-source affine：26.3067% |
| absolute Q2_OLS，40 cases，center128 | 15.1532% | 12.4870% | 67组：43.6362% | constant mean：38.6023% |
| absolute Q2_OLS，40 cases，uniform32 | 13.8084% | 11.2697% | 67组：43.1280% | constant mean：41.4674% |

absolute+54 固定组的 center128/uniform32 均值为 10.1242%/9.0646%，匹配 group-only94 为 29.4527%/29.1446%。Q2_OLS+27 与 GELU_OLS+27 的 center128 为 12.4870%/12.5270%，该汇总下差异很小；不能据此断言量化已无关。

这支持所测 affine 基项比同预算直接丢组更能保留函数，不支持“已可替换模型”。全40层 tangent 的“full-M”只指旧 κ 来自全部31668行；误差仍只在 center128+uniform32 上测量。其 case 级 full-M slope 相对当前输出 SSE 为 0.0045495–1.8963231。Block39 center128 四个 source×expert case 均值约1.0929%，不能取代所有层结论。旧 `E0_SHARED_TANGENT.md` 的384行五层表是另一个观测范围，应分别保留。

all-block 的连续重建相对 native 增量 SSE 最大 `2.2242e−4`，W1 差值重建 SSE 最大 `1.0325e−4`。这有助于解释大误差的量级，仍不是逐位 native 等价。

原始相对引用：`e0_all_block_tangent_20261025/{saved_gpu_attempt1,fold_group_gpu_attempt1,absolute_fold_gpu_attempt1}/RUN.json`。

## 4. 独立强竞争机制：GELU 的偶分量与 W2 干扰

**PRIMARY_PAPER + MATH_CONDITIONAL。** 我直接读了 Hendrycks–Gimpel [GELU v5 §2](https://arxiv.org/html/1606.08415v5) 的 `xΦ(x)` 与 tanh 公式，以及 [PyTorch GELU 定义](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.activation.GELU.html)。对本项目的 tanh 版本，令 `u(a)=sqrt(2/π)(a+0.044715a³)`，则

\[
g(a)=\tfrac12a+e(a),\quad e(a)=\tfrac12a\tanh u(a),\quad e(-a)=e(a).
\]

所以 `g(a)−g(−a)=a` 精确成立于实数公式，不需要 a 很小。若一个通道在校准样本中符号对称、均值零，带截距 OLS 的斜率满足

\[
\frac{\operatorname{Cov}(a,g(a))}{\operatorname{Var}(a)}
=\tfrac12+\frac{E[a e(a)]}{E[a^2]}=\tfrac12.
\]

于是一个宽分布、强非线性的单元也可以有稳定的1/2回归斜率。更直接地，从 `a_old=−t` 到 `a_new=t` 的有限更新，割线恒为1/2；当 t 较大时旧点导数接近0，旧逐token tangent 可以很差，而平衡的共享斜率仍可正确。这个反例解释了“共享导数优于更详细逐token导数”的一种机制，未假设 Wan 实际已经符号对称。

另取两单元 `a2=−a1`、值向量 `v2=−v1`，则 `v1 g(a1)+v2 g(a2)=v1 a1` 为严格线性，而两个单元都在做非线性计算。故输出可折叠也可能来自有符号组合。量化/平滑尺度、偏置和原生舍入会破坏这个简例的精确性；它作为连续 FFN 的有效反例，不能宣称原生 HiF4 恒等。

**可区分预测。** 局部热区解释要求单元输入局限在低曲率范围；偶分量/抵消解释允许宽分布，但预测 `G W2 S2⁻¹ Δe(a)` 比单元 `Δe(a)` 小。一个强而简单的未来控制是固定 `Kjj=1/(2s2,j)`，只用另一来源 uniform32 拟合实际 Q2 截距；与逐通道 OLS 的同样40/216基项成本比较。它没有斜率估计噪声，也无需旧 full-M 导数。当前结果尚未包含此控制，本审查没有运行它。

**另一条必要反例。** 组能量排序不优化有符号净误差。设三个组 residual value 分别是 `10v,−10v,v`。总 residual 是 v；只纠正能量最小的第三组，剩余项完全抵消，而纠正最大的第一组会留下 `−9v`。因此 `posthoc` top-group-energy 也不是子集误差的 oracle 下界；增加纠正组不保证净 SSE 单调下降。真正的目标包含组 Gram 的交叉项。现有 RUN 中 source0/step1/block0 的27/54组例子也有 posthoc54 比 source-fixed54 更差，但反例本身已经足以否定一般最优性。

## 5. Dense residency：数值控制成立于已测轨迹，收益来自物理执行

**SOURCE_CHECK。** 原 `_weight` 在 block key 切换时 clear tensor references、调用 `empty_cache()`，然后从同一 `record['weight_qdq']` 执行 `.to(device,dtype,non_blocking=False).contiguous()`。`block_no_flush` 保留相同 key、转移与 cast；`ffn_resident_no_flush` / `all_resident_no_flush` 以 `(expert,module_name,device,dtype)` 保存该 tensor。`module_name` 包括 block 路径，当前不可变 cache 下没有跨层重用错误。Smooth/QDQ、bias、F.linear 和 gate/residual 均未被这些控制更改。若未来同 key 下更换 record/权重，缓存键需重新审查；当前冻结场景没有此动作。

每个已完成 phase 都要求：1600 linear、1600 activation QDQ、零 cache misses；四个 scheduler latent hashes 和最终视频 SHA 与历史 dense 相同，仅允许搬运计数减少。全部18个 phase 的 RUN 为 PASS，并记录 `EXACT_HISTORICAL_LATENTS_AND_VIDEO`。我另行逐项比较了这18份 JSON 中的 trajectory/video SHA 与 `research/20261025/E0_NATIVE_OBSERVATIONS.json` 对应来源，全部相等；三个研究的 RUN protocol SHA 也与当前相应冻结协议相符。这比局部 SSE 更直接地支持这些已测轨迹的数值保持；不是不同硬件/shape 上的普遍逐位定理。

FFN resident 首次视频减少160次转移、后两次各减少320次；all resident 首次减少800次、后两次均零权重转移。保留量分别为22,649,241,600和56,203,673,600 bytes。all-resident 已记录物理使用峰值78,932MiB；这是 sampled peak，不是连续显存上界。

**ARCHIVED_TIMING。** 下表保留 source0 cold、source0 repeat、source12 的顺序，单位秒，计时包括视频保存及共同的轨迹观测。

| RUN 模式 | cold0 | repeat0 | source12 |
| --- | ---: | ---: | ---: |
| initial original | 80.174 | 75.297 | 85.364 |
| initial block_no_flush | 65.346 | 60.126 | 60.359 |
| crossover original | 84.323 | 75.172 | 75.372 |
| crossover block_no_flush | 62.080 | 60.011 | 60.383 |
| FFN resident | 60.200 | 57.388 | 57.446 |
| all resident | 57.068 | 50.419 | 50.789 |

本地卡索引关系与协议的 reciprocal crossover 一致；所有 RUN 的 sampled isolation 检查通过。以此支持“已测配置下，去除显式 flush 的方向在反向配对中仍存在”是合理的。本次读取的 RUN 标量并不提供稳定的硬件/主机身份全链条，不能独立把本地索引相同扩大成跨主机物理身份认证；此处不披露或复制私有映射。all-resident 与 FFN-resident 也没有各自在两张卡上的完整 crossover，不宜把二者差值全归因为策略。

**计时边界：** `FFNStageTimer` 包住整个 `infer_ffn`，包括 attention 输出加到 residual、norm/modulate、Q1/Q2 和两次投影；gate 与之后 residual 不在此边界。CUDA events 经过生成后的 synchronize 才读取，基本异步计时顺序正确；但此边界并非纯 GEMM 时间。`weight_fetch_wall_seconds` 是主机侧 `_weight` 调用耗时，可能含同步/allocator 等待，不能从它直接推得 PCIe 带宽或与 event 时间相减分解纯计算。cold phase 含库/流水线冷启动；每种策略只有一次三视频序列，不能给总体置信区间或声称已得到生产环境固定倍速。

读到的 [PyTorch allocator / async 文档](https://docs.pytorch.org/docs/main/notes/cuda.html) 区分 unused cached blocks 与 live tensors，并要求同步或 event 计时；[NVIDIA CUDA Event API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EVENT.html) 也明确 event 是流上的经过时间。这支持当前机制解释，而非把 latency 下降归为少算了 FFN。DeepSpeed 原始 [ZeRO-Inference §How it works / Alternative approach](https://www.deepspeed.ai/2022/09/09/zero-inference.html) 将 GPU compute 与 PCIe fetch 分开，并讨论在显存中保留权重；它是相同的驻留/搬运基础原理，但其吞吐导向超大 LLM 情境并不预测本项目四步视频的收益数值。

原始相对引用：`e0_dense_residency_20261025/*_attempt1/RUN.json`、`e0_dense_crossover_20261025/*_attempt1/RUN.json`、`e0_all_weight_residency_20261025/all_resident_no_flush_attempt1/RUN.json`。

## 6. 当前最有价值的下一项判别

已有 isolated benchmark 不能把 A 的应用时间当作完整方法速度。在其首个 case（source0/step1/block30），已保存的旧导数计算约54.4ms、BF16矩阵形成约4.07ms、应用约8.55ms，对照 native after-Q1 FFN 约53.4ms。若前者每个旧调用专门重算、同expert仅有一次后续复用，则这份实现的总增量成本已经吃掉省下的投影工作。其 full-M timing 由128行重复到31668行，未包含自然 full-M 分布、完整生成、权重流动或所有缓存管理；该结果约束当前实现，不能排除融合统计或离线固定斜率。相对引用：`e0_all_block_tangent_20261025/fold_benchmark_attempt1/RUN.json`。

因此下一项数学/保存数据判别应直接比较“固定半斜率＋训练源均值截距”“现有逐通道 OLS”“旧full-M导数”，同时分开报告偶分量的 hidden 能量与 W2 后净能量，再用 exact-dense residency 配置作为真实计时比较基线。若半斜率已接近 OLS，优先解释偶分量/抵消和组 Gram；若差异大且对应稳定低曲率区，再支持局部 affine 热区。此处仅提出判别，不增加任何执行授权。

## 实际 Web 覆盖与尚未完成的核查

2026-10-25 使用 `web.run`，先查三族：GELU 原始函数/对称性；weight offload/residency；CUDA allocator/event timing。原始读取包括 GELU v5 §2，并沿其链接打开作者 `hendrycks/GELUs` 仓库首页；后者只作来源追踪，未声称检查其训练脚本。读取 PyTorch GELU 定义、CUDA semantics 的 Memory management / Asynchronous execution 段、NVIDIA Event API 和 DeepSpeed 2022官方说明。旧报告中的 TARDIS、TaylorSeer、Increment-Calibrated Caching 仅通过本地报告获知，本次没有把它们算作独立原论文复核，也不据此宣布新颖性。

本次搜索改变了解释：从“affine 拟合为何有效”转为“内置奇/偶分解与 W2 signed combination 是否已解释大部分效果”；从“FFN 边界耗时”转为“算术与分配/搬运应有不同的因果账本”。尚未完成 fresh-source 统计、native folded 替换、完整质量验证或部署比较；它们均不是本审查的工作或权限。
