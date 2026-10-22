> **FFNDS research-library snapshot · V4 multi-scale exploratory action designs.** Global budgets and spatial tile×hidden actions are unvalidated constructs, not selected main algorithms; R12 E0 post-hoc output splice is NOT native tile routing or an attainable bound.
>
> Evidence policy: cited prior FFNDS results retain their original provenance; suggested operators and training schedules have **not** been executed by virtue of this import. Do not treat archived "next" instructions or "not yet merged" source-state language as current decisions. Read [the authoritative exploration entry](EXPLORATION_INDEX.md), [docs/66](../../../docs/66_R12等预算动作的操作数与条件粒度审计.md), and [CURRENT_DECISION](../../CURRENT_DECISION.json) first.

---

# FFNDS V4：全生成过程的后训练动态 FFN——可行算法路线与支持实验

**审阅稿｜2026-10-22｜未合并 FFNDS｜没有新增 Wan GPU 结果**

> **研究目标**：在**冻结已完成四步蒸馏的 Wan2.2 I2V D-Q 主干**上，通过轻量后训练和原生执行优化，按输入、空间区域、网络层及去噪阶段分配 FFN 的**两侧主 GEMM**计算，在真实自由生成视频上实现质量保持与实际加速。机制用于生成算法和判别方法，不作为并列的论文目标。
>
> 原正式工程合同：全 160 位置主 FFN GEMM MAC ≤ dense D-Q 的 50%；完整 FFN ≥1.5×；**真正新来源的自由学生生成**中 aesthetic / imaging / motion 各自相对 D-Q 下降≤2%。此外记录视频端到端时间、逐视频失败率、时序稳定性和与最强等预算静态方案的 Pareto 差异。

**本版替换 V3 的“相邻 block30/31 是第一主试验”的局部优先级**。R12 的全量操作数审查、原始 V1/V2 发现和负结果都保留；本文件先独立审阅，不修改 CURRENT_DECISION/模型作业准入。

## 0. AutoResearcher v2.3：由新证据选择新的计算自由度

使用 `mechanism-research` 的 NOTICE→MAP_FLOW→DERIVE→EXPAND→DISCRIMINATE→BUILD→RECORD，及 `research-expansion` 的“改变动作/信息/表示/因果时域，而不是单纯调旧配方”；`DEDUCTION_FIRST` 先推等预算与不可提前获知条件；`GOAL_ALIGNMENT` 以**视频质量—真实时延**为唯一最终指向。先大胆构造、随即给出相反结果与可辨实验。实验成本投入只向两个方法候选收敛。

### 0.1 关键新增观察：整源与空间的条件价值差异

[FFNDS docs/66 R12 E0](https://github.com/Qesire/FFNDS/blob/main/docs/66_R12等预算动作的操作数与条件粒度审计.md)独立复核四个已开发来源上 R7 的两种**超预算完整 FFN 恢复**：相对最佳固定恢复 B，整源事后 A/B 选择只额外降低 **0.171022%** flow SSE，事后逐 latent frame 输出拼接为 **0.314848%**，事后逐输出 token 拼接为 **5.089427%**。

这三个数字不是三种新 FFN kernel 的精度；后二者利用真实 D-Q 标签拼接*已经计算完成*的 flow，**不是实际混合 FFN 动作，也不是其数值上界**。不过它们给出有实验价值的预测：完整序列级同宽策略可能掩盖不同区域对 FFN 计算的异质需求。原 R7 的“3/4 局部/下游排序相反”同样不能代替这个条件价值量。

### 0.2 从失败中明确构造空间

- UB1：主 MAC=50%，原正式20的完整 FFN 1.546111×，三轴通过；来源分离20的 imaging **−3.165968%**。**单纯静态改变 step 宽度已能省算，但不足以跨来源保障质量**。
- UB2：末步全 FFN，源内末步同状态误差归零，实际补充20 imaging **−4.651736%**；全四步反馈是必要研究对象。
- R4E1：block20 同层、16 候选组、q≤2、匹配读出，原生筛选失败；不是对**跨层、跨 step、真实空间 tile、不同表示**的全局排除。
- R7D1/R11T1：误差有符号传播影响动作排序，局部损失与流目标不一致；R11 的固定 20 步方案虽在开发 flow 取得改善但没过门槛。下一轮用**真正全轨迹动作干预和质量反馈**，不再把改善某一局部目标当论文终点。
- R12 E0 的 gate 在已检查的同层/step 跨来源恒定，AdaLN-only 不能当作内容路由；内容信号来自已执行 Attention 和 residual/modulated input 等**W1 前**可见状态。实际 LightX2V 的 time 输入采用 sample-shift 后约 [1000,937.5,833.333,625]，不能把配置选择标签 [1000,750,500,250] 当实际模型 timestep。

## 1. 重新定义操作对象：不仅是两层，而是 4×40×时空区域

固定 D-Q 每步 40 个 block，四步合计**160 个 FFN 位置**，hidden 宽 13824=216×64，输入/输出 5120。FFN 所处的实际计算上下文：

`attention 完成并写回 residual → norm2/AdaLN 调制 → Smooth1/Q1 → W1 → GELU(tanh) → Smooth2/Q2 → W2 → b2/gate/residual → 后续 FFN/attention/step → 解码视频`。

每个位置按**真实 token 分组**（原形状的 latent-frame×空间 patch）拆成若干规则 tile；每个 tile 使用由 packed 64-channel groups 定义的宽度。基础动作先考虑 `k∈{54,108,162,216}`，真实裁剪发生在 W1/W2 两侧，保留每个 tile 的完整 5120 维输出与原 gate 写回；不偷用 post-W1 的完整 hidden 当免费 mask。若量化 QDQ 对拆分后的行几何有变化，必须进行原生全 M 的重新验证。

定义位置 `i=(s,l)`，其 tile `q` 占 `m_{i,q}` 行，共 `M_i` 行，所选 hidden 组数为 `k_{i,q}`，则**主 GEMM 的可核对计算份额**为

\[
\rho_{\mathrm{MAC}} = \frac{\sum_{s,l,q}m_{s,l,q}k_{s,l,q}}{216\sum_{s,l}M_{s,l}}.\tag{1}
\]

在共享 d=5120、主 W1/W2 两侧成本相等的情况下式 (1) 准确核算主 MAC；如果步骤实际 M 不同，用本式的实际逐位置行数计算，不能简单对百分比取均值。补偿、norm、QDQ、特征、pack、layout、GPU launch 等**不包括在式 (1)**，要在完整 FFN 时延中独立计费。

等预算下不只可以做 `block30宽↑ / block31宽↓`，还可以：

- **跨层/跨 step 转移**：把一个位置的 54-group 增量转移至更有生成收益的另一位置。
- **相同 block 内的空间交换**：两个等行数 tile 用 `108/216` 代替两处 `162/162`，保持该位置原主 MAC 不变，但真实 W1/W2 只在相应行与宽度上执行。
- **两层×多 tile 的联合转移**：不再只能选某个 block 的全图固定宽度，跨下游传播实际核对补偿与误差方向。

**注意空间动作的独立性界限**：FFN 对每个 token 的两层线性本身是 row-separable，而后续 Attention、norm、条件和四步去噪不是。因此“按输出 token 拼接 flow”不能替代“在 FFN 执行时按真实输入 tile 分组计算，随后传播到最终视频”。

## 2. 两个可竞争的主算法架构

### 路线 A（低系统风险）：全 160 位置的动态宽度预算管理 GBA-FFN

**新动作**：全序列 token 在每个 FFN 位置共享 `k_i`，但 `k_i` 按当前输入、已完成 block 和剩余计算预算变化；允许跨 block 和跨四步重新分配，不要求单个位置更快就提升完整生成质量。

**简单而严格的预算表示**：以全部 160 位置 25%（54组）作为非零计算底座，再分配 54组 increments。若每个位置的 M 相同，UB1 四步 `[54,108,108,162]`×40 层对应**恰好 160 个增量**，总 MAC=50%。设 `a_i∈{0,1,2,3}`，则 `k_i=54(1+a_i)` 且 `Σa_i≤160`。这是构造规则，不意味着达到任何精度。M 不同时替换为加权式 (1)。

**在线决策**：每个 step 开始可以利用输入图像、文本、前一 step latent 的廉价摘要分配 step 额度；进入 40 层后按 `post-attention residual/u` 和**剩余额度**顺序选择每层宽度。为未来层保留至少 54组底座。未执行的后续 block 隐藏量和未来官方分数绝不作为在线信息。预测器共享少量参数，用 step、layer、预算及当前廉价 sketch 编码；不需要 160 个独立大 router。

**一项最有信息量的测试**：固定相同整步入口，比较 UB1、全模型最佳（独立来源选择的）**静态**分配、各来源后验离线预算计划，以及用实时可见状态决定的在线顺序策略。重点测自由四步视频，不能以 best-UB1 作为唯一固定对照。先对代表性的早/中/晚 block 进行预算转移，随后扩到全40层×4步；不是永远停在 block30/31。

**最强竞争解释**：不同来源最终都偏好同一份 schedule；动态策略的收益只来自找到更强的静态层敏感性。如果数据支持这个解释，A 形成优秀静态控制/离线预算器，不再强加 router。

**优点**：重用已有 group packing，所有 M 同形状，每个 FFN 的原生两 GEMM 简单；容易满足低额外开销与时延核对。**短板**：R12 整源 A/B oracle 增益仅0.171%，这种全 token 动作可能保留不了充分的内容条件差异。此数来自另一动作族，不能视为 A 的正式上界。

### 路线 B（科研潜力最高）：全网络时空 tile × FFN 宽度的预算分配 STB-FFN

**新动作**：每个 FFN 的真实时空 token 划成**规整的 latent-frame×空间 tile**。不同 tile 使用独立 packed hidden 宽度，但都输出完整 5120 维并接受原生 gate/writeback；这些预算可在多个 step/block 之间转移。每一笔 FFN 计算都形成“在哪个 token 区域、在哪个 block/step、执行多少 hidden 非线性组”的联合动作。

**先做两个极简非零算子控制**，而不是直接全 token 动态 mask：

1. **同 block 空间半区对照**：选择某一 step3 层，把恰好各占半数 token 的左右空间 tile 从 `162/162` 改为 `108/216` 与 `216/108`；其他 159 位置保持 UB1。这个设计所有 tile 都计算至少 50% hidden，非跳过 FFN，在式 (1) 下严格等 MAC。相比 R12 的跨两个 block 宽度交换，改变的是**信息与动作粒度**。
2. **多层交替但非简单帧跳过**：在一个覆盖早/中/晚深度的短窗口内，用相同整体预算，让低宽度区与高宽度区按照当前输入的廉价统计形成不同**空间布局**；先和周期性/棋盘式/随机固定布局比较，再让全40层参加分配，检验相邻层/Attention 对传播的影响。约束每个 tile 都有非零 FFN 与数值正确的完整输出，不直接将 FIS-DiT 的 frame skip 当新方法。

**选择器的可行输入**：当前 block `u` 的 tile-wise RMS、有限维线性投影、`post-attention` 残差变化及已有 step/block 条件。只在当前 tile 的 W1 前读一次这些量。可用上一 block 缩略信息，但不可使用待路由的 FFN hidden 后验值；前置 feature extraction 必须包含在完整时延中。使用多个相邻 patch 形成一个可映射成矩阵切片的 tile，避免 old token×channel 近均匀散点的无效稀疏形态。

**控制变量**：相同 token 分区、相同主 MAC、同样的可训练补偿机会；和全 M 同宽的 UB1、最优静态空间模式、按来源选择的固定图案、逐视频后验 oracle、FIS-DiT 帧交错/跳算原理基线、真实动态方法分别比较。要记录 region 间调制、Attention 混合和真实四步反馈，而不是只取局部 SSE。

**最强竞争解释**：R12 逐 token 拼接的5.089%完全来自**后验混合输出的自由度**，实际 FFN tile 混合后经后续 Attention 仍只有微小净收益；或细粒度 GEMM 的额外 kernel/访存成本吃掉所有理想省算。这是 B 最先要判别的两个问题。

**创新空间与先例分工**：FIS-DiT 已在 Wan2.2 使用 frame-interleaved sparsity；E-DiT 有 latent 相关 MLP 宽度；PARE 有视频 content/step block routing。B 不能主张首次按帧或按输入减计算。真正有意义的差异是**完整的冻结少步 FFN 内、可执行的空间 tile×hidden 双侧 GEMM 选择，联合跨层预算与自由学生质量监督**是否形成独立质量—加速优势。这仍待原生实验和对照。

**两条路线共享的回退**：若原有 top-C nested groups 动作的 oracle 几乎没有价值，另行研究输出有符号效应/实际量化成本定义的 packed 功能组；这是**更换动作表示**，不是扩大 R4E1 的 q/pool 重试。先全支持数值验证，记录 R9A1 仅7/16压缩通过及 R8F2 的验证空间外误差，不挪用它们的失败作为新方法结果。

## 3. 共同的后训练：全轨迹条件动作价值，而非单层 C-score

本研究的训练不对整个 Wan 做新一轮大型蒸馏。冻结所有 D-Q/主干权重；首先只训练非常小的预算/动作预测器，在必要情况下为几种动作匹配拟合轻量补偿。这里训练的是**同一算法的决策与恢复**，不是新造一个视频生成主干。

### 3.1 从局部 reward 换成完整四步生成影响

将全部在线历史记为 `h_t`（输入、已执行模块状态、剩余额度等），动作 `a_t` 为当前 FFN 宽度/空间分区；定义最终自由视频的研究代理损失 `L_roll`（同 seed 的终态差异、解码时空纹理/运动稳定性等），及已知成本 `c(a)`。需要估计的是**当前动作加上随后真实策略执行的差值**：

\[
\Delta Q_t(h_t,a)=\mathbb E\{L_{\mathrm{roll}}(\mathcal G_{\pi\,[a_t=a]})-L_{\mathrm{roll}}(\mathcal G_{\pi})\mid h_t\},\tag{2}
\]

或者直接使用保留预算状态的动作值 `Q^π(h_t,a,b_t)`。这比 R11 在单 block39 对一个 teacher-forced flow 调参数更大尺度：**从干预点继续执行全部剩余 block、四步和生成视频**。在固定相同初始图文/seed 时，离线原生 fork/rollout 可以产生可比较的动作标签，不意味着在线会进行双分支推理。

**相互作用**不可忽略：对两个不同位置或 tile 的交换，测 `I_{a,b}=L(a,b)-L(a)-L(b)+L(0)`；交互强时不能把所有单层收益相加后假装得到全局最优。先在少数分层位置评估，再用粗窗口策略近似，不必穷举巨大的 160×tile 动作组合。

### 3.2 后训练流程：从已知轨迹到自身轨迹

- **第一轮（离线，开发来源）**：固定 D-Q/UB1 的相同初始随机种子，选分层、同预算的动作交换，分别真实向后 rollout，生成候选动作差值。先用完整 flow/终态 latent 的数值信号训练极小预测器，保存与它对应的真实视频，检查排序一致性。
- **第二轮（on-policy，小规模）**：在路由器自己生成的四步状态上重新收集有针对性的分歧样本，更新小预测器；评估每次 rollout 的来源数量/视频/峰值显存与训练时间。区分训练输入暴露、路由选择来源与真正最终独立质量来源。
- **第三轮（可选恢复）**：若动态动作确有优势但成像仍劣于门槛，固定已有主 GEMM 路由，比较小型修正头的“局部教师目标 vs 同预算全轨迹目标”及其部署开销；不默认增加 rank、FFN 主宽度或完整反传所有40层。

**损失选择**：全轨迹 flow/最终 latent、解码后视频的成对细节与时序稳定性是训练/候选选择的代理；最终三轴官方分数必须通过自由学生视频独立测得。避免把同 seed 像素一致性过强的 loss 当生成质量本身；额外报告内容/多样性与时序运动，而不是优化一幅画面接近 dense 就算成功。视频指标在实验反馈中尽早出现。

### 3.3 两层在线决策与预算因果性

跨 step 的**预算先验**可由输入图像/文字和已完成 step 信息给出，block/tile 的动作只能用到当时已经执行的算子输出；如果后续 layer 的信息未知，控制器必须保留最小 54组费用，并在剩余预算 `b_t` 内执行。对于全160严格≤50%目标，每次真实视频都记录 token 数加权式 (1)，而不是依赖平均密度。可先维持 UB1 各步固定额度，仅在40层×区域内动态；只有这一简单全网络版本成立后，开放 step 间额外可转移的额度。

## 4. 支持实验：先全网络科学判别，再增大算法与系统工作

**不要先跑一个 block 再自动推断全部160位置有效。** 单点测试只用于核对原生 kernel；科学实验至少覆盖不同 step、深度与时空区域，并且尽早执行完整四步生成。

| 编号 | 实验范围/算法对比 | 核心观测和相反结果 | 交付证据及下一步 |
|---|---|---|---|
| **E0 源码与成本先验** | 四步×40层的实际 FFN 次序，Attention/AdaLN 可用信号，当前 64-group pack/QDQ 与全步 MAC；读 R12 E0 | 哪些 tile 行可无损拆分、哪些参数/信息在算前存在；gate-only 是否退化常量 | 完整执行 DAG、行列/tile 布局及 BF16 bitwise identity；无需训练或新视频 |
| **E1 全局层/步预算试验（A）** | 四步早/中/晚 block 的等额54组互换，冻结 UB1 对照；跨来源/跨 step 和真实四步 rollout | 同 budget 是否存在不同来源首选**不同层/step**的真实收益？最佳固定全网络配置是否已经吸收主要价值？ | 来源×动作×最终 flow/解码视频矩阵；静态 vs 后验 vs 可见状态；A 的候选优先级 |
| **E2 原生空间 tile 试验（B）** | 同一 block 两个等行数空间半区 108/216 vs 216/108 vs 162/162；跨 3 个深度/多个 step | native tile 混合是否带来比最佳固定位置更大的有符号收益？是否只在无法部署的逐输出拼接里有收益？ | 同输入 native 完整 FFN 两侧 GEMM +后续真实 flow；原生全 M/identity；不用拼接历史 flow 代替 |
| **E3 联合跨层×空间条件价值** | 2–4 个相距不同层的真实 tile 动作，交换同等 MAC；小窗口、跨 step 各有控制 | 有利 tile 动作在多个 block 协同后是否保留优势，还是经 Attention 抵消？非加性作用多大？ | 完整 rollout、两两交互 `I`、source-wise 与 tile-wise regret、较强 static 空间模式 |
| **E4 小后训练与强固定控制** | 共享 tiny route head，线性/MLP，step-only、固定分配、on-policy refinement；必要时同成本轻修正 | 提前可见信息能恢复多少离线动作价值？已训练 router 是否在**自己状态**保持收益？ | 真实可预测值、regret、Router FLOPs/bytes/时延、训练预算与稳定性曲线 |
| **E5 全执行成本和融合** | UB1 与 A/B 最优候选在目标 GPU 上：不同 packed 宽度、动态 tile、norm/QDQ/GELU/gate 与补偿 | 成本是否由小矩阵、layout、launch/weights吞没？融合是否给出额外净收益？ | 两主 GEMM 真算、全 FFN>=1.5×、单视频端到端时间、峰值显存/持久权重，三重复稳态测量 |
| **E6 视频质量决定性试验** | 同一冻结候选全160位置×四步自由学生；已开发视频的早期机制比较，随后真正未见来源一次冻结评测 | 三轴独立质量是否达标？背景光照/小物体稳定性是否更好？是否在强 static 同成本条件下仍增益？ | 官方每轴指标、来源分层、运动/背景诊断、失败视频分析，最终 Pareto 图 |

**执行顺序**：E0→A 的 E1 与 B 的 E2 可分支并行→基于真实“条件收益×硬件可达性”选择主要算子→E3→E4/E5 并行迭代→E6。需要真实新 GPU 工作时独立冻结协议与资格，保留当前注册失败，不将本审阅稿当作自动启动指令。

## 5. 最小反事实设计和正确的负结果解释

### 5.1 等成本动作至少需要四臂比较

对每个实验层/区域，保留：

1. **dense D-Q**：与同输入/同调度的一致模型参考。
2. **UB1 和最优固定新动作**：不可只跟随意弱静态对照比较。
3. **等预算的实际原生行动 A/B 或 tile pattern**：两个主 GEMM 都真实减少，必须验证新动作对下游的改变。
4. **离线后验 oracle / 在线 pre-W1 策略**：拆开动作可达收益与信息价值。后验拼接仅是诊断，不是可部署 oracle。

对一组**完整真实生成**动作矩阵 `L_{i,a}`（同一来源分组、同一评价尺度），

\[
V_{\mathrm{cond}}=\min_a\mathbb E_iL_{i,a}-\mathbb E_i\min_a L_{i,a}\ge0.
\]

它只评价当前有限动作集合。候选动作改变全程自由状态时，其 loss 必须来自相同 seed 的完整执行，不能把不同视频的单步局部 `L` 按行拼成看似有效的策略。在线多次顺序决策需要用预算状态和已知历史训练 `Q^π`，不能借后验分数。

### 5.2 三种结果对应三种建设性推进

- **E1全序列层预算收益小，E2真实tile收益大**：主路线转向空间 tile×hidden 执行，保留跨step预算作为较强 static/策略上下文；最直接支持“整源平均掩盖条件差异”的新结构猜想。
- **E1跨层收益大但E2细粒度耗时过高**：用全 M 宽度动作完成可部署论文主路线；用 E2 的空间特征作为路由的 cheap summary，而非强行开发昂贵 token gather。
- **E1/E2 原生值改善明显，真实视频无质量增益**：把训练目标升级到 rollout 后，保留原动作，先建立与 motion/illumination 缺陷相关的监督；不是立即回到扩大 mask 搜索。
- **两者动作收益都小**：先改具体 FFN 可执行表达（例如 quant-compatible 功能组或共享低成本基底+非线性增量），以完全相同的全生成质量与成本指标重新评估；不在同动作集上堆更大路由器。

这是决策价值链，而非围绕未知机制无限增加旁支。

## 6. 算子实现：让全模型动作可部署

**预先证明主 GEMM 真实变小。** 两侧线性从 `[M,5120]×[5120,k]` 与 `[M,k]×[k,5120]` 执行；空间动作按每 tile 的真实行数 `m` 切分，每个 tile 都写回完整 5120 输出。与现有 dense D-Q/UB1 路径逐项追踪 Smooth1/Q1、W1、GELU(tanh)、Smooth2/Q2、W2/b2、gate/residual、原R256及真实 bf16 casts，禁止把零填充的大 dense GEMM 当真实减算。

- **GEMM 布局**：首选半区/连续多帧或稳定空间条带，成批成块运行；优化选择器读数与 W1 入口共用同一内存访问，规避 old token×channel 均匀散点导致的 launch/gather 开销。
- **权重部署**：复用低宽度/高宽度预打包支持、限制同一时刻活跃宽度数量，不默认复制全160位置的全部四个 packed 版本；记录GPU常驻峰值和 bank 访问/切换。
- **融合先例**：FlashInfer 已提供 DiT residual/norm/modulation/gate 融合，FIS-DiT已有frame interleave，RT-Lynx已有激活半结构稀疏 kernel。本工作必须证明的系统增量是在**当前 Wan+HiF4 数值链**中的真实 tile×packed FFN 分支和轻量统计共取，不能把常见算子融合命名为理论贡献。
- **质量与数值**：拆行前后改变矩阵乘法 kernel 和 QDQ 调用边界，先检查恒等/等宽/排列逆变换与 native D-Q/UB1 的允许误差；跨行的量化尺度或随机行为若存在，需固定正确实现而不是继续沿用错误“等价”。

旧同条件计时：D-Q **10177.568613ms**，UB1 **6582.688544ms**，其全 FFN 1.546111×。1.5× 历史时间门槛约 **6785.045742ms**，名义动态选择额外预算约 **202.357198ms**；这是同一历史条件下的差值，**不是新系统或其他GPU的预报**。减少额外 kernel/权重切换成本是方法构造的一部分，不能放在全文最后才验证。

## 7. 最相近学术工作：与之比较什么，而不是改名字

| 最近先例 | 已有贡献，FFNDS不得重复主张 | 我们必须给出的超越证据 |
|---|---|---|
| [FIS-DiT (2026)](https://arxiv.org/abs/2605.11869) | 少步视频中 frame-interleaved 计算、跨层刷新，含 Wan2.2；不能称首次空间/帧稀疏 | 不同的**FFN hidden×tile 正规 packed 双 GEMM**、内容条件下的计算价值、完整 native HiF4、最强等额 FIS/stagger 固定对照及质量/时延收益 |
| [E-DiT (ICML 2026)](https://proceedings.mlr.press/v306/wang26dd.html) | latent-based block skip 与 MLP width router/caching | 在冻结四步视频生成的 FFN 内，严格成本守恒的**跨位置/空间条件预算**及整轨迹后训练监督，优于宽度/跳层静态或训练式路线 |
| [PARE (2026)](https://arxiv.org/abs/2605.27336) | Wan2.1 视频 content/time block routing、渐进蒸馏 | 不是“按输入动态执行层”，而是冻结 D-Q 的**实际 FFN 双侧 GEMM +空间组预算**、不用重新联合蒸馏的真实质量/速度 |
| [Dynamic-in-Few-Step (2026)](https://arxiv.org/abs/2607.06631) | 少步蒸馏联合 step-specific 模型结构计算与输出 rollout | 后蒸馏、轻量学习的在线输入/空间动作，不进行大型重新蒸馏；公平比原始四步同 FLOPs/质量/时延 |
| [TRACK (2026)](https://arxiv.org/abs/2609.30096) | 用校准的轨迹分歧做大小模型按 step 的固定切换；无需训练 | 本研究的 FFN 位置×空间实时预算，而非离线固定 step 模型选择。轨迹 disagreement 要作为重要静态基线，而不是被误称新发明 |
| [RT-Lynx (ICML 2026)](https://proceedings.mlr.press/v306/cong26a.html) | 2:4 activation、误差补偿、优化 kernel；其线性层速度与完整 FFN不同 | 与 semi-structured activation 稀疏进行原生同硬件对照；我们的速度必须是完整 FFN/视频级 |
| [TAPE (2026)](https://arxiv.org/abs/2605.17837) | 时间一致性与 mask 平滑、背景 flicker 是既有观察 | 用实际 FFN 行为和生成质量证明动态值/稳定性差异，不能以“观察到闪烁”本身作为创新 |

以上是直接竞争风险，不是对“没人做过所有组合”的保证；论文新颖性由实际**方法构造 + 可证伪预测 + 强基线结果**决定。

## 8. 提议的研究选择与资源顺序

**优先科学候选：B——时空 tile×FFN group 的全网络动作**；**优先稳定工程对照与最快可发测试：A——整 M、跨160位置动态预算**。两者共用式(1)、原生动作标签、闭环训练和视频评价。只做一条小后训练方法，先别造几十个预测器。

不需要在任何一个局部 block 获得“完美”后才验证网络：真实实验应**以至少一个不同 step/早中晚深度的分层探针开始、用真实四步延续评估、再推广到全部160位置并考虑多个作用域之间的交互**。全模型能否真正超过当前 UB1 imaging 降幅以及最强静态时延才是主要问题。

最先投入的三个任务：

1. **E0**：复用 R12 operand/source 审计与 native pack 工具，增加 `latent [T,H,W]→规则 tile→Q1/W1/Q2/W2→原5120输出` 的数值完整性及 budget ledger；不要重做已有 gate/step-only 内容性检查。
2. **E1+E2 受控原生前向**：在四步不同 step/深度的分层作用域并行比较两类动作，真实全后续流及少量开发自由视频；按 source hash 保留 A/B/UB1、最佳静态、相同费用。
3. **E3/E5**：若 tile native 有明显额外条件价值且 kernel 开销可控，就把 B 作为主要算法、A 作为等预算对照，增加**小共享 actor + full rollout 质量监督**，最后完整全160部署。

**停止将单个 block 的局部 teacher SSE、逐输出拼接理论收益、合成 GEMM 倍数或已暴露20来源的质量通过写成全模型可发布结论。** 把这些实测/推导用于选择下一次有价值的干预，不把它们当作研究范围的终点。

## 9. 附录：可直接运行的预算算术控制

附属脚本 `ffnds_global_budget_sanity.py` 使用 **CPU-only、无模型运行**方式验证 UB1、跨 step 全位置预算交换、相同 FFN 的两个等行数空间 tile 用108/216组替换162/162组三个配置均保持 **50% 主 MAC**。这是式(1)的精确分数算术验算，不是 native HiF4/GPU/质量证明。后续 native runner 应用真实 M、实际分组及原生 shape 日志重新作预算校验。

## 10. 当前证据与源码索引

- 最新决定与 R12 E0：[`research/CURRENT_DECISION.json`](https://github.com/Qesire/FFNDS/blob/main/research/CURRENT_DECISION.json)；[`docs/66`](https://github.com/Qesire/FFNDS/blob/main/docs/66_R12等预算动作的操作数与条件粒度审计.md)；[`research/20261022/HANDOFF_R12_E0.md`](https://github.com/Qesire/FFNDS/blob/main/research/20261022/HANDOFF_R12_E0.md)；原始 [`R12_E0_flow_partitions.json`](https://github.com/Qesire/FFNDS/blob/main/research/20261022/R12_E0_flow_partitions.json)。
- 历史目标/视频失败和方法差异：[`docs/28`](https://github.com/Qesire/FFNDS/blob/main/docs/28_固定步间计算预算对照.md)、[`docs/29`](https://github.com/Qesire/FFNDS/blob/main/docs/29_冻结候选的来源分离补充验证.md)、[`docs/38`](https://github.com/Qesire/FFNDS/blob/main/docs/38_固定预算下恢复末步完整FFN.md)、[`docs/53`](https://github.com/Qesire/FFNDS/blob/main/docs/53_R4补偿感知支持交换的原生筛选.md)、[`docs/57`](https://github.com/Qesire/FFNDS/blob/main/docs/57_固定双层恢复的原生传播诊断.md)、[`docs/64`](https://github.com/Qesire/FFNDS/blob/main/docs/64_相同部署预算下的局部与真实flow目标对照.md)。
- AutoResearcher v2.3：[`mechanism-research`](https://github.com/Qesire/AutoResearcher/blob/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818/.agents/skills/mechanism-research/SKILL.md)、[`research-expansion`](https://github.com/Qesire/AutoResearcher/blob/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818/.agents/skills/research-expansion/SKILL.md)、`core/DEDUCTION_FIRST.md`、`core/GOAL_ALIGNMENT.md`。

**证据分级**：R12 原始flow/空间偏好、UB1/UB2质量与时延是既有实测；式(1)和 CPU算术控制是数学/程序验证；A/B 的动态动作、整轨迹后训练和融合加速是新候选，**还没有新模型/视频/最终品质结果**。