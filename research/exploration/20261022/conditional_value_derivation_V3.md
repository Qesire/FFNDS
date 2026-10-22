> **FFNDS research-library snapshot · V3 candidate and conditional-value derivation.** Two-action equations are conditional theorems; pair-specific experimental order was subsequently superseded by docs/66 and expanded open exploration. This is a seed, not a mandatory experimental queue.
>
> Evidence policy: cited prior FFNDS results retain their original provenance; suggested operators and training schedules have **not** been executed by virtue of this import. Do not treat archived "next" instructions or "not yet merged" source-state language as current decisions. Read [the authoritative exploration entry](EXPLORATION_INDEX.md), [docs/66](../../../docs/66_R12等预算动作的操作数与条件粒度审计.md), and [CURRENT_DECISION](../../CURRENT_DECISION.json) first.

---

# FFNDS V3：收敛到一种可部署的后训练动态 FFN 稀疏方法

**审阅稿｜2026-10-22｜研究任务：收敛而不是增加研究机制｜未合并仓库｜未启动新 Wan 模型实验**

> **一句话研究目标**：在冻结四步蒸馏 Wan2.2 I2V 模型的基础上，设计能真正减少两侧 FFN GEMM 的输入条件计算方法，只使用昂贵主 GEMM 前已经存在的信息，通过有限的后训练和原生执行优化，在真实自由生成视频上同时获得可信的质量保持与时延下降。
>
> 本文覆盖并**替换 V2 第 7–10 章的下一步研究优先级**，保留 V1/V2 的全历史事实、65 份文档的证据索引和独立的学术先例审查。原 V2 的 12 个机制节点从“十二条并列主线”改为“支持一个方法的备查工具箱”。

## 0. 按 AutoResearcher v2.3 的收敛判断

使用 `mechanism-research` 的 NOTICE→MAP_FLOW→DERIVE→EXPAND→DISCRIMINATE→BUILD→RECORD；用 `research-expansion` 将旧失败转化为新的**动作或信息构造**，但不将每个解释都扩充为一项拟发表贡献。使用 `GOAL_ALIGNMENT` 区分：目标是**自由生成视频的质量—速度 Pareto 收益**，局部输出/同状态 flow 是研究显微镜，数学洞见是找到节省计算的依据，而不是论文终点。

**此次只选择一个算法问题**：在相同计算预算下，针对每个输入，怎样选择下一段 FFN 应当执行的结构化计算动作，使生成损伤尽量小？其最小候选叫作 **Pre-W1 Conditional FFN Action Allocation**（临时代号 CAF）。

它不是已经证实的独创算法。PARE 在 Wan2.1 已做内容/时步条件 block routing；E-DiT 在 DiT 中做按样本 block skip/MLP width；Dynamic-in-Few-Step 已把 FFN 分组剪枝与少步蒸馏联合优化。我们的贡献不能只是“动态宽度/内容路由”，而应争取：**已完成蒸馏的 HiF4 FFN，等预算下游反事实收益作为监督目标，单点提前信息足够预测，经过自由视频与完整 native FFN 时延核验**。

### 决定性的历史信息（只选与此算法有关的四项）

1. **UB1**：四步固定 54/108/108/162 个 64-channel groups，总主 GEMM MAC 50%，原正式20完整 FFN 1.546111×且三项质量通过，但补充20 imaging 下降 3.165968%。说明节省工作量已经真实可实现，但该固定方法没有跨来源质量保证。
2. **AR2/HP/RB/R4**：后验 C-score 的局部选择可能明显优于随机/静态，但前置代理、四 bank 和局部近邻支持交换没有可靠变成可部署收益。问题不应再写成“把 C-score 预测得更好”。
3. **R7D1/R11T1**：固定原生干预观察到局部误差与下游收益的排序分歧；R11T1 的 flow 目标开发端相比 UB1 改善约 1.491%，局部目标开发端略变差；R11T1 固定门槛未过。指向**选择动作的目标比再增大局部恢复头更加重要**。
4. **UB2/BO1**：单纯固定后移预算、或在已经生成的 UB1/UB2 两视频间整条选择，没能解决补充 imaging；最后一步同状态误差为零也没挽救真实学生轨迹。于是，视频质量必须在研究早期进入反馈环，而不能被 teacher-forced SSE 代替。

其余 OP、NF、R8/R9、OD1、跨步缓存、时空频谱、结构变换等保留为与相应失败诊断对应的证据库，不作为本轮额外并列实验立项。

## 1. 把“真实质量 + 实际加速”写成唯一优化问题

设冻结 dense D-Q 四步生成器为 \(\mathcal G_D\)，候选为 \(\mathcal G_\pi\)，\(\pi\) 控制每个 FFN 真实执行动作，质量评分有 aesthetic、imaging、motion 三轴 \(Q_j\)。我们希望在真实输出样本分布下研究：

\[
\min_{\pi,\;\text{small post-train parameters}}\;\mathbb E[T_{\mathrm{video}}(\mathcal G_\pi)]
\quad\text{s.t.}\quad
\mathbb E[Q_j(\mathcal G_\pi)]\ge 0.98\,\mathbb E[Q_j(\mathcal G_D)]\quad (j=1,2,3).
\]

当前 FFNDS 额外固定工程验收：**完整 160 个 FFN 位置相对原生 dense 的主 GEMM MAC ≤ 50%，完整 FFN 时间加速 ≥1.5×**；另报告视频端到端实测延迟。这里的质量约束是由未见来源统计实测验证的标准，不是声称对所有提示词做形式保证。最佳等预算**静态**方法与 Dense D-Q 都是不可缺的对照。

三种数字不得互相冒充：local native FFN SSE、同状态 downstream flow SSE、自由学生四步输出视频质量。前两项是用于辨别干预的代理，最后一项才是实际目标。

## 2. 重建唯一需要的计算链（MAP_FLOW）

当前研究对象不是泛用 Transformer 的抽象 FFN，而是固定源版本的 Wan2.2 I2V 四步 D-Q，\(d=5120, h=13824=216\times64\)，40 个 block、4 个 scheduler step [1000,750,500,250]，expert 序列 high/high/low/low。模型路径和精度以 `docs/08`、`docs/55`、`docs/63–65` 及真实 `infer_ffn`/HiF4 adapter 为准。

在一个 block 入口，已完成注意力写回的状态记为 \(r\)；当前 AdaLN/时间调制给出 \(\beta,\gamma,g\)，则非量化的概念式为：

\[
 u=(1+\gamma)\odot\mathrm{Norm}(r)+\beta,\qquad
 y=W_2\,\mathrm{GELU}(W_1 u+b_1)+b_2,\qquad
 r^+=r+g\odot y.
\]

**实际实现**还包含 Smooth、激活 HiF4 QDQ（第一/第二位置）、BF16 cast、实际缓存有效权重、GELU(tanh) 和原生 bias/gate 写回顺序。\(g\in\mathbb R^{5120}\)，hidden 分组存在于 \(\mathbb R^{13824}\)，**gate 不是 hidden mask**。同 step/block/time 输入下 AdaLN 单独通常不含视频内容差异，内容来自 \(r\)、注意力输出和调制后 \(u\)。因此在本方法中：

- **决策信息**：当前 block 的 \(u\) 或 \(r\) 的极低成本摘要；AdaLN/time/block identity 是条件背景或风险加权，不被当作一条独立的动态算法。
- **执行动作**：提前选定、已经 pack 好的稀疏 hidden 组；同时影响 W1/W2 的实际矩阵形状。初试优先全 token 序列共享一种动作，从而避免极散的 token×channel gather。
- **跨 block**：计算预算作用的区域和下游流传播的比较单位，而不是再增加一个独立的 routing 大模型。
- **融合**：把特征读取、native QDQ/pack/补偿费用纳入一个完整执行时间账本。现有 DiT fused norm/gate 算子是技术基线，不主张首次发明融合。

## 3. 从 FFN 数学推导“应当预测什么”（DERIVE）

### 3.1 为什么单独按 activation/C-score 排序不足

令 hidden 输出为 \(h=\mathrm{GELU}(W_1u+b_1)\)，被某个动作忽略的 group 集合为 \(S\)。在同一精度的实数辅助表示中，其未补偿输出差为

\[
\Delta y_S= W_{2,:,S}h_S,\quad
\|g\odot\Delta y_S\|_2^2
=\sum_{i,j\in S}\langle g\odot v_i,\,g\odot v_j\rangle,
\qquad v_i=W_{2,:,i}h_i.
\]

第二式含 **非对角交叉项**；把各组的 \(\|v_i\|^2\) 当作独立贡献，会遗漏它们的方向抵消和强化。即使两种支持有相近 activation/C-score，它们仍可能在原生下游产生不同结果。此式是帮助构造动作的实数公式，并不是绕过 QDQ/BF16 的部署等价证明。

### 3.2 用同状态原生反事实，而不是再发明“重要性分数”

对同一输入状态 \(x\)，令 \(F_D(x)\) 为 D-Q 的完整同一步 flow，\(F_0(x)\) 为参考稀疏路径，\(F_a(x)\) 为动作 \(a\) 对应的真实完整后续 flow。定义

\[
 e=F_0-F_D,\qquad d_a=F_a-F_0.
\]

则动作的**精确 flow SSE 改善**为

\[
 G_a(x)=\|e\|^2-\|e+d_a\|^2=-2\langle e,d_a\rangle-\|d_a\|^2.
\]

这是对测得相同形状 flow 向量的恒等式，允许原生 QDQ 非线性真实存在。它说明应辨别的是“动作与已有下游误差是否方向有利”，而不是单层重建大小。**但有符号 flow 误差仍只是候选监督标签，不自动等于视频感知质量增益。**

### 3.3 新的核心推论：动态价值 = 异质性 × 可提前获知性

先只考虑**两个同成本动作** A/B，其在同一批输入上的损失为 \(L_A(x),L_B(x)\)（首先取同状态原生 flow SSE），令 \(D(x)=L_B(x)-L_A(x)\)。

不使用输入信息，最佳固定动作风险为

\[
 R_{\rm fixed}=\min\{\mathbb E L_A,\mathbb E L_B\}
=\tfrac12[\mathbb E(L_A+L_B)-|\mathbb E D|].
\]

知道每次动作结果的事后 oracle 有

\[
 R_{\rm oracle}=\mathbb E\min\{L_A,L_B\}
=\tfrac12[\mathbb E(L_A+L_B)-\mathbb E|D|].
\]

因此**动作集合自身的条件价值**有严格公式

\[
 \boxed{V_{\rm oracle}=\frac{\mathbb E|D|-|\mathbb E D|}{2}.}
\]

真正的推理时间只有决策前特征 \(Z=\phi(x)\)。理论上只使用此信息的 Bayes 策略是：当 \(\mathbb E[D\mid Z]>0\) 选 A，否则选 B。它相对最佳固定动作的**可用信息价值**为

\[
 \boxed{V_Z=\frac{\mathbb E|\mathbb E[D\mid Z]|-|\mathbb E D|}{2},\quad 0\le V_Z\le V_{\rm oracle}.}
\]

公式前提是每个样本两种动作具有相同**被比较成本**，损失定义和样本权重一致。证明仅使用 \(\min(a,b)=(a+b-|a-b|)/2\) 与条件期望（Jensen 不等式给出上界）。为了验证公式而构造的数值微例：若两个输入的动作损失分别为 A=(1,4)、B=(4,1)，最佳固定损失=2.5，事后 oracle=1，故条件价值=1.5；如果提前特征完全无法区分这两个输入，则可获信息价值为零，若能无误区分则为1.5。这是数学例子，不是 Wan 新数据。

**这是本轮最关键的演绎**：

- \(V_{\rm oracle}\approx0\)：不是 router 太弱，而是这两个动作本身几乎没有动态价值；换真实可执行**动作集合**，不要扩大网络。
- \(V_{\rm oracle}\) 大而 \(V_Z\approx0\)：动作差异存在，但当前前置信号不足；改变低成本**信息/观察时刻**，不必马上创造新残差模型。
- \(V_Z\) 大但实际策略未兑现：属于监督标签、拟合/泛化或实际 GPU runtime 问题。这里才值得小型后训练。

对于 UB1 加 A/B 三臂，继续直接用 \(\min_a\mathbb E L_a-\mathbb E\min_aL_a\)；上面闭式两动作公式先用来明确机制并降低首轮实验复杂度。

### 3.4 为什么原生 flow 不等于最终视频质量

四步去噪可写为 \(z_{s+1}=\Psi_s(z_s,F_s(z_s))\)。一个在 step \(s\) 引入的 flow 修正，可能沿后续状态演化改变后面每一步输入。一阶示意为

\[
 \delta z_4\approx\sum_{s=0}^3 J_{\Psi_{s+1:3}}\,\partial_F\Psi_s\,\delta F_s,
\]

并可能存在跨步干预交互。真实 Wan/HiF4 的强非线性和离散 QDQ 要用自由学生轨迹验证，而不是把这个公式当精确量化计算。这同时解释历史 UB2：单一末步同输入误差归零不能恢复已经偏移的前序状态。**因此本方法用 flow 标签寻找候选，但在形成可部署策略之前必须进行真实自由视频成对评价。**

## 4. 最小算法构造（EXPAND，但只保留一个）

**CAF 原型：提前状态摘要 → 两个等预算 packed-FFN 动作 → 选择其下游收益更好的执行配置。**

首个真实判别作用域沿用 R7 已建立诊断基础的 **step3/block30–31**，保留全部其他 FFN 为冻结 UB1。用同一种 64 通道分组排序和同一个原生值域，在两个 block 间选择三种完整形状：

| 臂 | block30 | block31 | 层对总组数 | 用途 |
|---|---:|---:|---:|---|
| UB1 | 162 | 162 | 324 | 旧静态控制 |
| A | 216 | 108 | 324 | 增加前一层预算 |
| B | 108 | 216 | 324 | 增加后一层预算 |

这三种动作只是**科学显微镜**，不是论文最终方法已经固定为三选一。其差异与 R4E1 同层 q≤2 交换、R7D1 完整教师恢复、UB1/UB2 整视频二选一都不同。主 GEMM 的等总组数相同，补偿/打包/访存/时间并不由此自动相同。对每种动作都要匹配离线恢复训练机会和上线实际费用。

先在真实原生层对测 A/B 反事实与最佳固定控制，检查 \(V_{\rm oracle}\) 是否存在**源/内容相关**的符号反转。再在干预入口使用已有 \(r_{30}\) 或 \(u_{30}\) 的少量统计 \(Z\) 预测 \(D\)，可把 \(g\)、\(\gamma\)、block/step 当作已知调制或归一化条件。因为路由在 block30 W1 之前决定，不能使用 block31 尚未执行的 hidden 或教师标签。

**极简后训练**：先用最小线性或小 MLP 回归 \(D(x)\) 或成对偏好，目标为降低实际 action regret，而不是重建 216 维 C-score。只有动作和提前信息都有价值时才做此步。必要时对 A/B 各配有限补偿；补偿共享/独立必须公平比较。

原生加速：权重支持提前 pack，选择整个序列共享的规则宽度，真实执行两侧 GEMM；计时包括计算前 signal extraction、QDQ/布局、bank 选择、补偿与 gate。与现成 fused DiT norm/gate 和未融合 HiF4 版本同时比较。融合是这个方法的**执行细节/系统消融**，不是第二个必须成立的理论假说。

## 5. 只安排三项会改变决策的实验（DISCRIMINATE）

### E0：真实性与前置信息检查（源码 / CPU / 已存数据）

**核心问题**：预期输入 \(Z\) 是否真能在算 W1 前廉价读取，并且不是 timestep 常量？

- 对实际 `infer_ffn`、time_embedding/AdaLN、HiF4 QDQ、W1/W2、Attention 写回和 R256 回放建立一张**单一具体 block30–31 子图**，给出时序、精度、shape 与可测位置；不再绘制全领域机制地图。
- 对同 step/block/expert、不同真实视频验证 AdaLN 值是否相同；核对 \(u\) 或 \(r\) 的轻量 sketch 具有内容变化；将 feature source 的获取时刻定位到 block30 W1 前。
- 盘点 OP2/TF2/R7 中能复用的整步入口、gate、teacher flow、native sparse 权重与实际时延，区分仅 256 采样行和完整 M；现有 OP2 不能凭 256 行直接代替完整层对的下游 flow。

**产物**：一份真实子图与 operand-coverage 表；最小的 A/B 同状态回放器所缺的具体操作数，及每个候选特征的成本。E0 不消耗新视频，只为下一步准备。

### E1：同预算原生动作价值（最关键实验）

**核心问题**：A/B 是两种真正不同的内容条件选择，还是单纯一种静态宽度更好？

- 使用相同 D-Q/UB1 步入口、同一份图文/seed、原生 quant/cast/gate 顺序；对 A、B、UB1 分别执行到相同步末 flow。block31 必须接受 block30 经动作改变后的真实状态。
- 保存 per-source \(L_A,L_B,L_{UB1}\)，以及有符号 \(D\)、动作收益方向、best-fixed/oracle gap。单纯统计 C-score 不作为标签。
- 把 A/B 的动作成本与后训练机会匹配；测随机重复/同动作重复来确定噪声底。需要跨多个**开发**来源判断异质性，旧 formal20/supplementary20 均已暴露，不能充当最后一次独立验收。
- 对真实 \(Z\) 先做无需复杂训练的分箱/线性预测检查 \(V_Z\)，并以 source-wise 划分测 held-out regret；**明确区分训练拟合分布、观察决策分布和独立最终质量分布**。

**结果如何改变算法**：

- A/B 几乎无条件收益 → 不训练 router；只改变一次可执行动作定义（如固定计算预算下不同 hidden-group pack），保留一样的监督/系统主线。
- 条件收益足够但 \(Z\) 无预测信息 → 将 \(Z\) 换为更有内容的当前 `post-attention residual`/低成本投影，而不是扩大旧 C-score 网络。
- 条件 oracle 与提前预测都成立 → 进入 E2，直接实现最小路由器。训练是否改善要报告与 best fixed 的真实 regret，不以 test accuracy 代替。

### E2：尽早用真实视频裁判方法，而非等机制都研究完

**核心问题**：E1 的真实 flow 改善是否改善自由四步学生视频的 imaging/时序质量，并产生真实加速？

- 在**已经作为开发来源**的少量输入上，以固定同一完整算子分别生成 D-Q、UB1、best-fixed(A/B)、后训练策略的自由学生四步视频，不在步与步之间恢复 D-Q。按来源成对保存图像/seed/输出，记录每轴指标与背景闪烁等可解释诊断。
- 运行完成160 FFN 的原生计时及视频端到端计时，单独列出选择器、QDQ、权重选择/布局、两个主 GEMM、修正、gate；**报告原实际部署选择的每个分支频率与费用**。
- 使用自由视频结果检验 flow 标签在不同样本上是否仍然能预测质量变化；若排序不一致，研究一次最小的 quality-aware 标签修正（比如对已观察到的闪烁敏感时空分量调整监督），但不开展新的机制课题竞赛。
- 固定完整方法和所有参数后，用真正新的来源进行首次官方三轴评价；原合同的三轴各≤2%下降、全局主MAC≤50%、完整 FFN≥1.5×三者由**同一部署候选**同时成立。只有这时讨论论文验收。

## 6. 算子融合：唯一服务于 E2 的工程回路

历史原生数值后端是 HiF4 数值 QDQ + BF16 `F.linear`，**不是**已经采用硬件 FP4 GEMM。历史 D-Q 完整 FFN 10177.568613ms，UB1 6582.688544ms；维持 ≥1.5×，在这一组历史计时口径下最大总时延约 6785.045742ms，因此旧 UB1 时间点估计的剩余开销约 **202.357198ms（约 UB1 的 3.074%）**。

因此预测器要轻，不能为了减少一点矩阵算术引入等量 kernel launch、全 hidden materialization 或繁重 gather/scatter。预打包少量动作宽度，尽量在现有 Norm/AdaLN 读取、QDQ 调用、W1 入口融合 signal sketch；与现有 fused 实现做原生数值和时间对照。每个融合候选首先保证固定输入与旧原生语义一致（对非等价浮点重排明确给出数值差距），再讨论节省时间。历史点估计不是新 kernel 的稳态测量，也不构成速度先验保证。

这里改变的是相同 CAF 算法的执行体：**不另立“融合机制论文”或独立新 route**。

## 7. 与已知学术论文精确错开的位置

| 已知先例 | 已经研究的能力 | CAF 若成功必须额外证明 |
|---|---|---|
| PARE（2026；arXiv:2605.27336） | 视频 timestep/content block skip + 结构剪枝/恢复 | 不把 block routing 当新意；聚焦冻结四步模型的 FFN 内等预算动作的 **signed downstream margin 可预测性** |
| E-DiT（ICML 2026） | 样本级跳 block、改变 MLP width、feature cache | 宽度路由不是首次；需在**真实原生 HiF4 视频 FFN**证明质量意义的条件动作标签、额外收益与硬件执行 |
| Dynamic-in-Few-Step（2026；arXiv:2607.06631） | 联合四步蒸馏 + FFN/head/block 结构稀疏 + rollout | 冻结已经蒸馏的主干，小后训练，动态**当前输入**动作价值，而非重新联合蒸馏形成 step-static 模型族 |
| Prox / Fast Forward / WIDE / RT-Lynx | Pre-W1 proxy、动态组/宽度、补偿/packing/kernel | 不是 Proxy-only、组重排或融合本身新；从**自由视频质量约束和原生 downstream signed action labels**获得独立可验证方法差异 |

**写作时的关键区别**：方程 \(V_Z\)、有符号 flow SSE 恒等式本身是基础数学，不是新发明；论文方法需要一个确实有效的动作定义、低开销的行动前估计器，以及自由视频/加速结果，最好能在更多层对或模型精度设置复现。

## 8. 收敛的执行顺序和可编辑研究状态

| 顺序 | 当前决策需要什么 | 产物 | 下一选择 |
|---|---|---|---|
| 1 | AdaLN/time 内容性与特征的实时可得性 | E0 计算图 + operand 覆盖 + sketch 开销 | 形成最小可执行 A/B 方案 |
| 2 | 等预算动作有无可学习的条件价值 | E1 原生 \(L_{i,a}\) + \(V_{\rm oracle}\) + \(V_Z\) | 是否需要训练 router，或仅换一次动作空间 |
| 3 | 真实生成视频质量是否受益 | E2 少量开发视频 + 学习策略 + runtime/profile | 冻结可信候选或修正质量监督 |
| 4 | 严格部署与外推验收 | 完整 FFN + 新来源官方三轴 + 端到端视频 | 形成论文核心实验 |

研究记录采用一个 \`primary_route\`（CAF）及其状态 \`EXPLORATORY → GROUNDED → NATIVE_ACTION_VERIFIED → VIDEO_PILOT → INDEPENDENT_EVAL\`；各阶段的失败配方依旧独立关闭，其观察反馈给下一次**不同构造**。实验部分只有 E0/E1/E2 三项，旧 V2 的 12 节点全部不再是需要逐项启动的待办表。

**首个 agent 实际任务**：从当前 `AGENTS.md` 和 pinned skill 出发，完成 **E0 的真实 block30–31 子图、可用状态与张量/权重覆盖清单**，然后写出并自检三臂 native replay 的最小工程协议。该任务不需要新 GPU 训练；没有新增 Wan 视频、native A/B 或融合速度测量的完成证据。对 E1/E2 另行登记同一冻结研究合同和算子/来源身份，再执行有成本的实验。

## 9. 追溯入口（历史完整、优先级收敛）

- 历史探索与证据地图：`FFNDS_全历史机制探索网_审阅稿_20261022.md`；`FFNDS_全历史探索网_V2_调制跨层融合与质量驱动_20261022.md`，但本 V3 替换其中的研究项目优先级。
- 文献一致性审查：`FFNDS_V2_学术查重与机制一致性审查_20261022.md` 与 `FFNDS_V2_论文查重_12节点判定表_20261022.csv`。
- FFNDS：[`OBJECTIVE_CONTRACT`](https://github.com/Qesire/FFNDS/blob/main/research/OBJECTIVE_CONTRACT.json)、[`CURRENT_DECISION`](https://github.com/Qesire/FFNDS/blob/main/research/CURRENT_DECISION.json)、[`docs/28`](https://github.com/Qesire/FFNDS/blob/main/docs/28_固定步间计算预算对照.md)、[`docs/29`](https://github.com/Qesire/FFNDS/blob/main/docs/29_冻结候选的来源分离补充验证.md)、[`docs/57`](https://github.com/Qesire/FFNDS/blob/main/docs/57_固定双层恢复的原生传播诊断.md)、[`docs/64`](https://github.com/Qesire/FFNDS/blob/main/docs/64_相同部署预算下的局部与真实flow目标对照.md)、[`docs/65`](https://github.com/Qesire/FFNDS/blob/main/docs/65_研究目标层级与R11解释修正_20261022.md)。
- AutoResearcher：[`mechanism-research`](https://github.com/Qesire/AutoResearcher/blob/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818/.agents/skills/mechanism-research/SKILL.md)、[`research-expansion`](https://github.com/Qesire/AutoResearcher/blob/d24f18091e0ee1f75e5f7ce7fddb7c2f16d6b818/.agents/skills/research-expansion/SKILL.md)、`core/DEDUCTION_FIRST.md`、`core/GOAL_ALIGNMENT.md`。
- 竞争先例：PARE `https://arxiv.org/abs/2605.27336`；E-DiT `https://proceedings.mlr.press/v306/wang26dd.html`；Dynamic-in-Few-Step `https://arxiv.org/abs/2607.06631`。

---

**本版本的研究选择**：我们现在不需要更多并列的新机制。先证明或推翻一个能够产生**可预测、可部署的 FFN 动态价值**的最小算法；接着用真实自由生成视频与完整 FFN 计时裁决它。已经发现的其他结构关系继续保存在历史证据库，等待某个具体实验结果真正需要它们时再调用。