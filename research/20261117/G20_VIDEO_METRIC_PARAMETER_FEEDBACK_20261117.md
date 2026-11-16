# G20：用完整视频指标校准折叠 FFN 参数

日期：2026-11-17。类型：ALG / SRC / LIT + 非模型软件验证。真实生成质量 VIDEO、CUDA 原生兼容性 NATIVE 与公平物理成本 SYS 均未由本次提交建立。

实现入口：[video_metric_feedback](../../tools/video_metric_feedback/README.md)。本说明补充“如何从已有局部算子继续得到可检验的视频改进”，不替代 G20 结果入口、参数资产清单或 ACTIVE 执行范围。

## 1. 判断与具体路线

**视频指标值得参与参数选择，首轮应将它用于已有可折叠方向的低维校准。** 局部 FFN 拟合给出便宜、结构明确的参数方向；完整视频评分决定这些方向在真实学生轨迹上的有效强度。这样可以直接检验局部改善是否传到生成结果，也把单轮生成预算控制在可核算的范围。

本次选择两个系数，分别共享于 high、low 各三个已有 L64 kit 站点。用固定 CAL 图像、prompt、seed 和初始 noise 的完整视频进行坐标有限差分，构造带运动/主体约束的局部候选，再实际生成并评分该候选。接受与否由真实分数和 guard 决定。

这里存在三个不同命题：

1. **目标更直接：** 视频评分能观测局部 FFN MSE 未显式表达的下游影响。这是目标定义与计算图层面的结论。
2. **当前方向有可用信号：** 两个 L64 系数附近能产生超过量化平台和评价波动的分数差。这需要真实探针数据。
3. **更新有独立收益：** CAL 选出的系数能在未参与选择的 image-cluster 上改善生成，并保持动作、身份与实际系统成本。这需要独立视频确认。

本次实现第一个命题对应的算法与接口，提供检验后两个命题的可执行路径。小矩阵和解析分数的单元测试不能证明第二或第三个命题。

## 2. 为什么局部 FFN 误差不能决定最终最优系数

用行 token 约定，在一个 layer/expert 的 hit 站点写作

\[
\widehat Y(X;A)=Y_a+(X-X_a)A,
\quad X,X_a\in\mathbb R^{T\times d},\quad A\in\mathbb R^{d\times d}.
\]

`T` 是实际视频 latent token 数，`d=5120` 是通道轴。anchor 的输入、输出来自同一次完整 rollout。gated residual 与后续层/去噪步继续传播该预测。原 binding 的 pre-gate 输出和 host gate 位置保持不变。[R1]

局部拟合通常控制某个状态分布上的

\[
\mathbb E\|\widehat Y-Y\|_F^2.
\]

它对输出通道采用均匀欧氏权重。最终视频目标还依赖各方向经过后续层、去噪调度和解码后的影响，以及学生自己访问到的状态。即使局部误差减小，影响主体、纹理或运动的少数方向仍可能变差。

为说明这种区别，暂时考虑光滑、未量化的模型：

\[
z_{t+1}=\Phi_t(z_t;A(\theta)),\qquad V=\mathcal D(z_4),
\]

\[
u_{t+1}
=D_z\Phi_t\,u_t+D_A\Phi_t\,\partial_\theta A,
\qquad u_t=\partial_\theta z_t,
\]

\[
\partial_\theta q(V)
=\langle \nabla_Vq,\,D\mathcal D\,u_4\rangle.
\]

若一组方向的局部误差近似受下游 pullback 度量 `JᵀWJ` 加权，那么最佳系数通常与在原输出空间做无权 MSE 的系数不同。这给出了重新校准的机制动机；并不推出当前两个方向一定有用。实际 BF16 导出和官方评分还包含离散操作，以上链式公式用于解释依赖，不作为官方实现可直接反传的承诺。

完整正负扰动会连同后续轨迹、anchor 值和最终视频一起改变，因此包含这些路径的有限幅度响应。若只在 hit 时换矩阵却复用旧候选的 anchor，则测量的并非上式中的完整生成函数。本实现每个候选重新生成，每个样本重建 anchor cache。

## 3. 参数空间：从已有低秩方向到两个标量

已有线性 kit 提供

\[
P\in\mathbb R^{d\times64},\quad
\gamma\in\mathbb R_{>0}^{64},\quad
C\in\mathbb R^{64\times32},\quad
Q\in\mathbb R^{d\times32}.
\]

\[
\Delta A_L=(P\operatorname{diag}(\gamma^{-1}))CQ^\mathsf T.
\]

指标表达为

\[
(\Delta A_L)_{ab}
=\sum_{j=1}^{64}\sum_{k=1}^{32}
P_{aj}\gamma_j^{-1}C_{jk}Q_{bk}.
\]

`a` 是输入通道，`b` 是输出通道，`j,k` 是低维特征轴。因此 `rank(ΔA_L)≤32`；L64 不能解释为一个 rank-64 输出修正。

设 `g(s)∈{high,low}` 表示站点所属 expert：

\[
A_s(\theta)=
\operatorname{BF16}\!\left(\operatorname{FP32}
(A_{\mathrm{q1},s}+\theta_{g(s)}\Delta A_{L,s})\right).
\]

两个数控制 high/4、19、34 与 low/4、19、34；其余 74 站点来自完整 base manifest。这个参数化估计的是已有方向的有效强度，不能恢复方向空间之外缺失的非线性或时空信息。

端点以保存的位模式为准：`theta=0` 使用 `A_Q1_BF16`，`theta=1` 使用 `A_L_BF16`；其他系数走 CPU FP64 重建及 FP32→BF16。实现记录公式重建与保存端点的位差。若两种生成路径存在极小数值差异，不能把保存端点视为任意后端重建的逐位证据。

**对照身份必须准确。** 六个 kit 的 CURRENT_Q1 与历史部署 Q1r2 并非同一矩阵。`theta=0` 是“六站点 kit CURRENT_Q1 + 其余站点 base”，`theta=1` 是“六站点 kit L64 + 其余站点 base”。要比较历史 Q1r2，应单独使用并标明历史完整 manifest；它不自动等于本参数族的零点。[R2]

折叠后仍执行原有 dense `delta_x @ A`。该方案没有为每个 token 新增低秩支路，也没有改变主 GEMM 维度与 anchor slab 形状。离线多次视频生成和评分是新增成本；实际在线速度是否维持，需要相同执行路径的原生对照。

## 4. 视频指标的使用方式决定优化会得到什么

VBench 将视频质量拆分为多个维度，并为各维度设计对应的提示与评价；不能把任意来源上的一个分项提升当作总体质量提升。[L1] 本实现使用固定 cohort 的逐视频原始输出，保留适用维度和 source/image-cluster 身份。

| 指标性质 | 作为更新信号的限制 | 本次处理 |
| --- | --- | --- |
| 帧间主体/背景一致性 | 减少运动可能提高帧间相似度 | 与连续运动及 Dense 配对变化约束共同使用 |
| I2V 主体/背景 | 包含输入图关联，也包含帧间相似项；不能独自排除静止偏好 | 可作为目标，并保留运动约束 |
| motion smoothness | 插帧重建一致性不等同于正确动作幅度 | 报告分项，不单独最大化它 |
| temporal flickering | 官方实现明确面向静态场景检查 | 不作为通用运动视频的默认奖励 |
| dynamic degree | 每段视频最终返回 bool，局部扰动常不改变分类 | 保留原官方结果，另用连续运动诊断 |
| camera motion | 离散类别/是否匹配指令 | 只在有显式 camera 元数据的来源检查 |
| imaging quality | 官方总分已除以 100，但逐视频 MUSIQ 仍是原始量级 | 导入逐视频值时除以 100，原值另存 |

这些处理来自固定 VBench 源码核对。[L2–L7] 它们说明“可被优化的分数”和“想要的视频行为”之间需要约束；不是说这些官方指标本身无效。

首版配置示例的目标包括 I2V 主体、背景与成像质量；约束包括每个 image-cluster 的主体/背景退化上限，以及对固定 Dense 视频的 `flow_l2`、`temporal_delta_rmse` 退化上限。后两项来自既有 FFNDS 诊断工具，是本地代理，不标为官方 VBench。[R3]

Farneback 流场误差比逐视频 bool 更连续，但仍会受纹理、运动估计和视频时间对齐影响。它不等于真实物体运动，也不是唯一验收标准。运动幅值比在 Dense 近静态时尤其不稳定。某个来源不适用的指标应在看候选结果前通过 eligibility 明确排除；不能优化后再选择有利的指标或来源。

## 5. 目标、统计单位与更新公式

令 `q_m(i,s,r;θ)` 为图像簇 `i` 的 source `s`、seed/sample `r` 的第 `m` 项分数。先在同一 source 内平均样本，再对同一图像的 source 平均，最后对该指标适用的图像簇等权平均，得到 `q̄_m(θ)`。每项指标的适用簇可以不同，但 eligibility 在所有候选间固定。

定义 `a_m=1` 表示越大越好，`a_m=-1` 表示越小越好。`w_m≥0` 和尺度 `s_m>0` 事先确定：

\[
\ell(\theta)
=-\sum_m\frac{w_ma_m}{s_m}
\left[\overline q_m(\theta)-\overline q_m(\theta^{\rm ref})\right],
\]

\[
f(\theta)=\ell(\theta)
+\frac{\lambda}{2}\|\theta-\theta^{\rm ref}\|_2^2.
\]

`theta_ref` 是本次搜索的初始参数，不是 Dense 参数。Dense 视频用于配对运动/视觉诊断；guard 的退化限则相对固定的初始候选诊断值。这两种“参照”不能混淆。

对越大越好的指标，`max_drop=ε` 表示 `q(θ)≥q(θ_ref)−ε`；对越小越好的指标，表示 `q(θ)≤q(θ_ref)+ε`。也可冻结绝对上下界。若初始点已经违反绝对 guard，优化器返回 `initial_guard_violation`，不花预算搜索一个未定义的可行起点。

在内部点用同一固定 cohort 得到

\[
\hat g_j=\frac{f(\theta+h_je_j)-f(\theta-h_je_j)}{2h_j},
\quad
\hat h_j=\frac{f(\theta+h_je_j)-2f(\theta)+f(\theta-h_je_j)}{h_j^2}.
\]

临近边界时实现使用实际不等距三点公式，只有一侧可行时用单侧差分。以 `H_jj=max(ĥ_j,0)+damping` 构造正的对角模型，同时由探针估计 guard margin 的一阶变化：

\[
\min_d\ \hat g^\mathsf T d+\tfrac12d^\mathsf THd,
\]

\[
\text{s.t.}\quad
\|d\|_\infty\le\Delta,\quad
\theta_{\min}\le\theta+d\le\theta_{\max},\quad
c(\theta)+\widehat J_cd\ge0.
\]

代码把该对角二次问题白化为欧氏投影，用 Dykstra 投影求 box 与 halfspaces 的交；未达到停止与可行性条件时不输出未经认证的候选。它是局部提案器，不构成非光滑视频目标全局收敛的保证。

对唯一候选生成完整视频后，必须同时满足：

- 实际视频损失改善严格超过 `min_improvement`；先验项下降不能单独导致接受。
- 所有真实 guard margin 非负；线性化预测通过不够。
- `f(θ)−f(θ+d)>0`，并达到设定的实际/预测下降比。
- 实际导出的 BF16 矩阵内容与中心不同。

没有通过时保留原参数。代码支持有限多轮和拒绝后的信赖半径缩小，示例只运行一轮，不自动追加试验。

## 6. 固定噪声、量化平台和评测预算

正负探针共享图像、prompt、seed、sample、初始 latent 和评分身份。对两个带随机性的分数 `Q+`、`Q−`，

\[
\operatorname{Var}(Q_+-Q_-)
=\operatorname{Var}(Q_+)+\operatorname{Var}(Q_-)
-2\operatorname{Cov}(Q_+,Q_-).
\]

共同随机数在协方差为正时降低差分方差；不是普遍保证。共同随机数用于仿真优化有既有研究基础。[L8] 本实现逐样本核对实际 noise SHA，但相同初始 noise 不保证 GPU kernel 或评分网络完全确定；环境/权重身份和必要的重复性判断仍属于真实运行证据。

BF16 映射有平台。参数数值不同，矩阵字节可能相同。实现把两类身份分开：

- **执行内容指纹**：完整有序矩阵 payload SHA、形状和约定，用于识别实际相同算子与评分缓存。
- **来源指纹**：系数、kit 和 base 身份，用于审计一个矩阵如何产生。

相同执行内容复用已有分数，并拒绝将同一算子的评分波动当作参数更新收益。trace 区分“矩阵未变”和“矩阵已变但指标平坦”。若全在量化平台上，本轮只能说明当前扰动不足；若矩阵改变而指标仍平坦，则可能是该方向影响小、指标不敏感或离散分类尚未跨阈值，不能直接断言 FFN 修正无用。

`p` 个参数首轮最坏为 `1+2p+1` 组完整 cohort，后续每轮 `2p+1` 组。两参数、N 个 source-seed-sample 组合即最多 6N 段候选视频。六参数需要 14N。固定 Dense 参照资产、参数导出和官方/本地评分另计；评分维度多时，生成次数不变但评分成本增加。

外部评估启动次数在调用前写入持久日志，失败也占次数，顺序重启不会恢复预算。完整有效结果才能进入缓存；改配方或 cohort 不能沿用原目录。一个搜索目录限一个 writer，不支持并行重入。

## 7. 如何判断这一更新是否值得继续

| 竞争解释 | 最有区分力的观察 | 对下一步的含义 |
| --- | --- | --- |
| 局部拟合方向有用，但强度未按视频结果校准 | 固定噪声的方向探针有稳定变化；约束内候选在独立图像上继续改善 | 保留低维视频校准，评估是否需要站点级系数 |
| 一致性提升来自运动减弱 | 一致性升高但 Dense 配对运动/帧增量或动作人审退化 | 拒绝候选；调整目标约束，不能称总体质量改善 |
| 分数变化主要来自量化平台/评分波动 | 相同执行指纹却产生不同分数，或探针未改变矩阵 | 先解决重复性/扰动尺度；不扩大参数维度 |
| 两个方向覆盖不足 | 矩阵确实改变，跨指标响应仍弱或只能做互相冲突的取舍 | 考察其他有机制依据的方向；不是继续盲目调两个标量 |
| CAL 选择过拟合 | CAL 改善而独立 image-cluster 不改善 | 当前系数无可迁移证据；停止以确认集反馈继续调参 |
| 分数改善没有部署价值 | 可接受质量只在改变执行路径或成本上升时出现 | 以同拓扑、同运行环境重做 SYS 对照，重新判断性价比 |

最小有辨识力的真实流程是：先冻结两参数、指标方向/尺度/guard、CAL cohort 与最多六组预算；产生中心、四探针和一个候选；冻结选中的系数，再将初始版本与更新版本用于独立 image-cluster 的配对视频确认。报告全部预先选择的指标和每个 cluster 的变化，不只报告组合目标。人审重点看主体身份、动作是否完成、运动量和相机指令。

独立确认的统计单位是 image-cluster。多个 seed 先在来源内合并，不能当成独立样本。可以对独立图像簇的配对差异报告置信区间或 bootstrap。本包目前输出各 image-cluster 的组合损失贡献和候选相对中心的组合损失下降；逐指标逐簇确认报告、置信区间和 bootstrap 尚未实现。CAL 上的搜索验收不是独立显著性检验。

若收益只在 CAL 上成立，本轮仍能给出“方向响应、约束瓶颈、平台范围”等有用结果；但部署选择继续保持原参数。若独立改善成立，下一项问题才是是否增加到六站点系数、或者增加新的低秩/非线性方向。SPSA、Bayesian optimization、全参数梯度训练和奖励学习不在本次实现内。

## 8. 实现边界与本次验证

新增代码覆盖完整链路中可复用的模块：kit 导出 → 请求身份 → 已构建 pipeline 的 native rollout → 官方评分 worker/既有本地诊断 → cluster 评分 → 参数提案和验收。外部 oracle 负责按已注册范围调用这些模块，并提供真实 checkpoint、LoRA、环境、资产和诊断成本收据。历史冻结 runner 没有被改造成无范围的新入口。

软件测试覆盖 BF16 舍入/端点/朝向、完整 manifest、实际内容指纹、官方输出尺度和 camera 结构、cluster 权重与固定参照 guard、有限差分/约束/预算、argv 子进程/缓存/失败重启、native cache 生命周期/异常恢复及评分 manifest 绑定。生成和评分的重依赖采用显式 synthetic fake；所有这些结果只证明对应软件语义。

复现命令：

```bash
python -m unittest discover -s tools/video_metric_feedback/tests -v
python -m tools.video_metric_feedback check --config tools/video_metric_feedback/examples/config.example.json
```

当前环境没有 PyTorch、Wan 模型和原生 GPU 运行资产，因此没有报告新的真实视频指标、选中的生产系数或加速倍数。真实 native 路径与官方 API 已按源文件对接，第一次模型运行仍需验证环境兼容性、完整解码及成本收据。示例的数值阈值是配置演示，不是由实际视频数据估计的推荐值。

## 参考依据

- **R1：** 原生 [binding](../../reports/20261107/g19/source/q1_r2/binding.py)、[direct](../../reports/20261107/g19/source/q1_r2/direct.py)、[resident](../../reports/20261107/g19/source/q1_r2/resident.py)、[runner](../../reports/20261107/g19/source/q1_r2/runner_panel.py)。
- **R2：** [G20 既有结果与资产入口](../../reports/20261114/g20_two_track_core/README.md)、[历史完整 Q1 matrix manifest](../../reports/20261107/g19/Q1_matrix_manifest.json)；本次 kit 数学字段与端点检查见 [matrices.py](../../tools/video_metric_feedback/matrices.py)。
- **R3：** [FFNDS 本地视频诊断](../../tools/ffnds_video_metrics.py)与[数据、校准和评价协议](../agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)。
- **L1：** Huang et al., [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982), CVPR 2024。
- **L2：** 固定 VBench 源码 [imaging_quality.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/imaging_quality.py)。
- **L3：** 固定 VBench 源码 [dynamic_degree.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/dynamic_degree.py)。
- **L4：** 固定 VBench 源码 [i2v_subject.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/i2v_subject.py)、[i2v_background.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/i2v_background.py)。
- **L5：** 固定 VBench 源码 [motion_smoothness.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/motion_smoothness.py)、[temporal_flickering.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/temporal_flickering.py)。
- **L6：** 固定 VBench 源码 [subject_consistency.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/subject_consistency.py)。
- **L7：** 固定 VBench 源码 [camera_motion.py](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/camera_motion.py)。
- **L8：** Kleinman, Spall, Naiman, [Simulation-Based Optimization with Stochastic Approximation Using Common Random Numbers](https://pubsonline.informs.org/doi/10.1287/mnsc.45.11.1570), Management Science 45(11), 1999。
