# G20 · 终端可观测 FFN 算子的判别实验设计

**2026-11-10 · DESIGN_ONLY / REVIEW_REQUESTED / UNVERIFIED · NO_NEW_RUN**

[理论重推导](G20_FFN_FUNCTIONAL_OPERATOR_REDERIVATION_20261110.md) · [G19 历史结果与定义](../20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [G19 可复用资产](../../reports/20261107/g19/README.md) · [实验前独立审查](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)

本设计将“终端输出是否是更合适的拟合目标”“是否必须保留路径约束”“时间差分能否改善真正的视频动态”“稳定功能秩是否明显小于 5120”“拟合顺序对收敛的影响”组织为可独立判别的科学问题。**尚未提交 CPU/GPU 新作业；需要当前实验独立评议、修订复核及 ACTIVE 冻结后才能执行。**

## 0. 方法快照与统一比较前提

**研究对象：** Wan2.2-I2V-A14B-LightX2V-Distill-4step；HHLL、无 CFG、BF16、A100 80GB、原生 torch_sdpa、832×464×81 帧；高低专家各40层，合计160个 FFN 站点。Step0/2 为原生锚点，Step1/3 为近似命中。逐专家逐层 A 固定，FFN 前有 norm2/scale/shift，FFN 后当前 gate 及 BF16 舍入边界继续保留。真实 online cache 生命周期与专家覆盖按已归档 fused_post 实现。

**来源身份：** 当前四个 CAL 开发来源 chip10_02/04/07/09（seed42）及七个已暴露开发视频来源 raft/bicycle/guitar/tennis/spoon/eagle/canoe_static_00（seed43/44）已有使用历史。新确认性 cohort 在观察候选视频结果前冻结，按独立 image/prompt source 分配，不把种子、帧、token、block 当独立源。相同原图、prompt、初始噪声、裁剪、scheduler、precision、backend 是配对比较前提。需验证初始 latent 字节身份而非仅检查 seed。

**证据语义：** ALG=定理/计算推导；SRC=代码实现；STATIC_ONLY=只读或单元测试；NATIVE=原生模型推理运行；VIDEO=真实完整学生视频；SYS=公平真实速度/显存对照。每个实验保留 RUN/EXIT、原始文件和失败；本计划没有新的 NATIVE/VIDEO/SYS 证据。

**默认对照（设计可能按问题裁剪但须说明）：** Dense D0、G0/Q0、Q1-r1、Q1-r2、Q2-r1/r2；新的全局方法逐组件消融；Rank0、普通 SVD 与输入/功能加权低秩；ICC-FFN 和既有 Chipmunk 资产在 FFN-only、全160、相同生成条件下比较。当前首项缺失的匹配新复现是 ICC，已有 Chipmunk Wan 实测先资产复用。比较版本、模型和数值 recipe 必须逐臂匹配。

## E0 · 数学与实现合同预检（STATIC_ONLY，可先做）

**目的：** 在投入 GPU 之前使新观测、矩阵方向与历史融合路径同义。

1. 验证行向量 $U\in\mathbb R^{N\times5120}$、$W_1\in\mathbb R^{13824\times5120}$、$W_2\in\mathbb R^{5120\times13824}$、$A\in\mathbb R^{5120\times5120}$、gate 广播、时空 token 索引、expert/step 缓存有效性。
2. 对小尺寸 GELU 网络数值检查：
   $F(u_t)-F(u_s)\simeq\Delta u\,\bar J_F$；
   $(F(u_t)-F(u_s))-\Delta u A=R_A(u_t)-R_A(u_s)$；
   $\Delta u A$ 与 $\sum_j\alpha_j(\Delta u w_j)v_j^\top$ 等价。
3. 对单样本秩一精确插值的反例和 $I_d$ 的最佳秩-$r$ 误差界提供解析/小矩阵测试。
4. 针对单个站点，实施 FFN 误差的 pre-gate/post-gate 两种注入并核对一次 BF16 gate 乘法舍入及缓存写入格式。实现层的首次实际不一致位置应由模块 oracle 定位。
5. 对时间算子明确：去噪步索引 $t$ 与视频 latent 时间位置 $\tau$ 不同；81 RGB 帧与 latent 21 时间位置的对应、时间 padding、首尾边界和 camera/motion 参考坐标须经原生 shape 验证。

**完成条件：** 数学与源码张量对应具有同一输入输出定义；独立检查历史 D0/G0 neutral 仍成立。E0 只是静态/最小单测，不称新视频验证。

## E1 · 误差传播、线性化和跨层耦合（最高机制优先级）

**竞争解释：**
- H1：r1→r2 的局部改善落在最终输出较不敏感的方向；终端传播增益能解释质量排序。
- H2：主要差别来自学生自由轨迹的输入漂移；在同一状态下注入时效应较小。
- H3：多个站点经下游传播形成显著抵消/增强；简单单层相加预测不足。
- H4：一阶 Jacobian 失配显著，高阶反馈和锚点依赖主导误差。

**最小原生对照：**
1. 固定同一 Dense/学生真实状态、专家和输入，先在前/中/后代表层进行单站点 FFN 注入。使用原生 Y 与 Q1-r1/r2 的实际后 gate 增量差；保留相同其它层和 RNG。
2. 记录三种观测：(a) 该站点的门后误差；(b) 当前去噪预测 $P_t$；(c) 完整 0→3 学生 rollout 末端 latent $Z_4$。将同状态直接干预与自由学生整个轨迹分别登记。
3. 用冻结参考下的 JVP 预测末端变化，并与真正的后缀前向及多剂量 $\eta\in\{0,\eta_1,\eta_2\}$ 干预比较；$\eta$ 的具体值以 BF16 可分辨且未触发数值不稳定的预检查范围冻结。记录相对预测残差 $\|\delta y_{\rm actual}-\delta y_{\rm JVP}\|/(\|\delta y_{\rm actual}\|+\epsilon)$ 与 $\|\delta y_{\rm actual}\|$。
4. 取两层或两个专家对应的受控联合干预，计算：
   $I_{ij}=\delta y_{ij}-\delta y_i-\delta y_j$。
   所有 $\delta y$ 使用同一初始参考状态；同时做前向和反向替换，区分干预顺序和冻结输入条件。
5. 在学生自由轨迹重采锚点，比较同输入函数误差、候选轨迹漂移、step1→step2→step3 跨步反馈的贡献。高低专家分开统计并保留专家切换时的缓存清理证明。

**主要统计单位：** source 为独立单位，同源 seed 为配对重复；层、token、时间位置均为嵌套测量。开发诊断优先选 Canoe（方向变化明显）、Guitar（相对 Dense 较低）、Bicycle（相对稳定），新增源用于后续确认。

**决定性解释：** 若 JVP 对单站点扰动具有稳定预测力，可将其用于 E2 的固定传播代理；若随扰动幅度显著失配，转向限制更新步长及频繁学生重放；若交叉项大，E2 必须包含跨层联合项；若单站点同输入误差小而自由路径差大，优先建学生状态校准。

**需保存：** 每臂输入张量身份、锚点状态、pre/post-gate 扰动、D0/候选当前预测、四步末端 latent、JVP/VJP 输出、层/step/专家和原始 token 坐标、运行/内存日志。全 Jacobian 无需显式保存。

## E2 · 终端目标与路径/时间约束的独立消融

固定 E1 判断的有效传播域、相同选定层、相同初始化、相同迭代/GPU 预算：

| Arm | 局部 $L_{\rm local}$ | 终端 $L_{\rm out}$ | 轨迹 $L_{\rm path}$ | 时间差分 $L_{\rm temporal}$ |
| --- | --- | --- | --- | --- |
| Q1-r2 | 原定义 | — | — | — |
| Global | 可选最小先验 | ✓ | — | — |
| Global+Path | ✓ | ✓ | ✓ | — |
| Global+Path+Time | ✓ | ✓ | ✓ | ✓ |

**首先保持完整 A，不改变 online GEMM/缓存/kernels**，以便识别损失变化的效果。优化 Q1 系数 $\alpha$（受限外积基底）与全自由 A 的比较应另设表示消融；原 Q2 的单秩一离线修正可保留对照但不是全自由最优。

对真实自由生成进行四步重放，按相同输入时刻观察候选和 Dense 的去噪预测，同时保存各自自由轨迹的状态。时间差分作用于候选与 Dense 的**误差**，在同一个 latent 视频时间坐标上计算。若采用 $x_0$ 估计，先以实际 Wan scheduler 公式定义转换，再比较原始预测与 $x_0$ 两种监督位置，避免把未绑定的转换当成已实现。

**拟合顺序作为独立因素：** 同一目标比较 forward 0→39、backward 39→0、交替和联合；令计算预算、线性化更新频率、初始化相同。诊断跨层 Gram/Hessian 的强耦合块，再尝试按耦合分组优化。后向梯度传播不等同于后向参数拟合；四步去噪真实状态永远按 0→3 顺序重放。

**主要结果：** 新来源终端预测 MSE、末端 latent error、视频时间差分误差、时间高频残差、视频八轴及运动/画质指标。不以 CAL loss 的单独下降认定视频改进。

**切换条件：** 如果 Global 在自由轨迹发生输入漂移，优先加入/增强 Path 和 local 约束；若 Time 对高频误差有益却未改善运动方向，保留其闪烁意义并研究 motion transport，避免把视觉帧差解释为位移方向。

## E3 · 终端功能有效秩与执行模式

**竞争解释：**
- R1：真实输入协方差集中，使输入加权的固定 A 有较快奇异值衰减。
- R2：下游仅对部分写入方向敏感，使功能加权谱衰减更快。
- R3：权重/数据源显著异质，固定低秩仅在训练来源有效。
- R4：高保真需要较高维度，但少数层可采取 Rank0 或原生回退。

**比较：** rank-0（旧 FFN 输出复用）、rank-256/512/1024、完整 A、原生 FFN；普通 SVD、输入加权 SVD、可分离功能加权 SVD、相同 rank 下的终端直接拟合。对于一般非可分离 Hessian，使用矩阵自由梯度/JVP，不把加权 SVD 作为真实全局最优保证。

**跨来源稳定性：** 对每层报告奇异值累计功能能量与源外预测损失，分别统计 high/low、层深和步骤；固定拟合数据源后在来源外进行相同因果干预，切勿用同一批源同时定 rank 和报告最终泛化。

**计算合同：** 完整 $A$ 用一次 $N\times5120$ 乘 $5120\times5120$；rank-r 用两级 $N\times5120$→$N\times r$→$N\times5120$；rank-0 命中返回原生锚点的 pre-gate FFN 输出并乘当前 gate；原生回退执行 W1/GELU/W2。所有模式在锚点构造缓存时同步复核可用信息与是否重复计算 A。执行时间包括投影、缓存读写、offload/prefetch、kernel launch 与合并开销；前期理论 MAC 仅用于设计，不替代 SYS。

## E4 · 完整视频、前沿对照与真实系统 Pareto

**最小臂：** D0，G0/Q0，Q1-r1，Q1-r2，E2 选定 Global+Path/Time，E3 选定低秩/层级模式；按经批准的时间和资源补 ICC-FFN，Chipmunk 优先复用已保存的真实视频与 producer，再决定新的同源对照。

**质量：** 原版完整 8 轴（subject/background consistency、motion smoothness、dynamic degree、aesthetic、MUSIQ/image quality、I2V subject/background），逐来源等权；RAFT 同位置光流方向、模长、$\beta$ 与正交残差；CoTracker 固定三方共同可见点集的位移/轨迹；camera_motion、semantic foreground/background、temporal flicker 独立指标及视觉完整 81 帧核验。缺失观测标记 NOT_MEASURED，不把自定义 8 轴称为官方完整 VBench 榜单。

**计时：** 同节点相同精度和后端、预热、交替 ABBA/BAAB 或等效随机平衡顺序，记录原生 FFN 事件时间、整体 wall time、峰值 allocated/reserved、模型加载与离线拟合额外开销；同 GPU 测量期取消 probe，视频可复核 latent/RGB/MP4 的 neutral 身份。不要从理论 MAC 推出实际 wall 加速。

**判定：** 在读确认性 cohort 的结果前冻结质量非劣阈值、主要运动指标、速度优效或资源预算、独立来源数、seed、缺失数据处理、聚合及 PASS/FAIL/INCONCLUSIVE 规则；开发样本与正式确认分别报告。实际门槛需结合部署目标和原始测量变异确定，本设计不预填未经讨论的数值阈值。

## 5. 独立预审与真实资产清单

**正式方法快照字段：** METHOD_ID、源码 commit/blob SHA、模型及 distill checkpoint、scheduler、BF16/GELU、设备及 backend、路径 identity、A/gate/cache tensor shape、输入时空坐标、calibration/source cohort/seed、arm、目标与 λ、求解器、rank/mode、最小数值 oracle、输出文件哈希、预计 GPU/VRAM/time、系统指标及允许副作用。

**须经实际独立评议的风险：** Jacobian 线性化能否预测真实后缀；实际 rollout 是否可微且省算模式是否正确；时间网格与视频帧坐标对齐；Rank0 与当前 gate 语义；合适的新来源是否独立；单站点替换与多站点联合是否以相同参考状态定义；终端收益是否仅来自大容量过拟合；A100 上低秩小 GEMM 是否真的省时。

**计划资产按 producer 登记：** 已有 Q1/Q2 BF16 矩阵 manifest、G19 CAL 张量、raw MP4、RAFT 40 帧对场、CoTracker 观测、seed/noise identity、原始运行记录；新资产优先保存 terminal deltas、线性化/非加性指标、source-wise rank 谱、时空坐标、自由学生预测和加速记录。旧大文件仍在其已登记原位置；本设计不复制也不重建已完成 run。

**状态：** E0–E4 均为计划。E1 是最有判别力的首项新模型实验，其前提是主 Agent 自审、真正独立 reviewer 实质质疑、修改及 reviewer recheck、ACTIVE 有限范围冻结。本文归档没有触发任何执行。
