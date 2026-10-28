# FFNDS · M-B四轮交锋与全局审视：统一结论、证据边界与下一判别

> 2026-10-28｜独立科研综合（DISCUSSION_SYNTHESIS / DECISION_GUIDANCE）  
> 方法：AutoResearcher / FFNDS 的 mechanism-research（MAP_FLOW→DERIVE→DISCRIMINATE）、problem-reformulation（数学对象重构）、research-expansion（并行解空间）和 ffnds-paper-discovery（原生计算与同条件强对照）。  
> 证据等级：ALG 数学；SRC 仓库源码审查；REPO-NATIVE/VIDEO/SYS 引用原有 RUN；LIT 一手论文先例；OPEN 待检验。**本轮没有执行新模型、GPU、学生训练、原生 finite、视频或 SYS。**  
> 研究身份：保存独立讨论后的综合判断，**不是**现行 M-B v2 METHOD_SNAPSHOT 替代物，不自动获得 method_discussion/QUEUE 同行认可，不更改 ACTIVE_RESEARCH.json 的冻结任务和授权。当前任务导航仍以 RESEARCH_NAVIGATION.md、三方法整合 v2、B0/B1 交接为准。

## 0. 一页结论（先于辩论）

**研究北极星：在不损害视频实际生成行为的前提下，找到减少真实 FFN 物理计算的可执行方法，并从真实证据中发现信息、函数、动力学和计算组织的新规律。**

四轮讨论真正完成了三件事：

1. **澄清可忽略误差**：在满足所有后续 Pre-LN/Cross-LN 分支条件的连续模型中，逐 token 通道公共平移是跨层的精确规范自由度；可导出无需在线额外算子的 quotient loss。此定理的运行时适用条件尚需核实。
2. **纠正测量对象**：A0 已报的 source-row mean fraction 是对 256 采样行求均值，不是对 5120 通道求均值；不能作为规范误差比例。行×通道四正交分解与共享 W2 bias 校正可成为最便宜的下一判别。已存 scalar JSON 不足以推回四个向量分量。
3. **揭示研究排序问题**：四轮数学结论都不能单独推出新的 GPU 免算；最接近工程基线的是已运行四位置原生 Q2 适配的 A0 6016。M-B 同 source 配对敏感、四步可观测性及 B-F 都保留独立科研潜力，但现在没有相同成本的新学生 VIDEO/SYS 胜出结果。

**建议的资源组织**：
- **工程可用性主轴**：A0 扩展源/层/共享 step/完整 Student rollout；优先形成至少一种真实低成本、完整生成且保留动态的视频 FFN。属于战略建议，**不擅自替换**当前显式指定的 M-B B0 优先任务。
- **当前指定 M-B 工作**：依原 B0/B1 交接取得真实 post-write u、r、gate、head/flow 的中性身份与有限响应；以 fixed readout、marginal、paired、shuffled 和同在线成本 student 验证非平凡增量。不要重新启动广泛第一轮口头审议。
- **低成本同步诊断**：先用匹配 teacher/student 原生操作数计算 row/channel 四分量与 source-level bias 相干；按结果决定是否训练 B-QUOTIENT 或折叠 bias；这些是 baseline/机制资产，非旗舰创新。
- **并行高风险探索**：M-C 编码计算、I/C 的量化和算子路线、F 的时空传播维持独立入口；不预设只有 M-A/B/C 三路。

**研究决策**：M-B = CONDITIONAL_CONTINUE（科学价值保留；部署潜力有待同成本验证）。A0 = 最强本地紧凑学生 baseline，尚未完成 80 函数、Student 全轨迹和真实整端速度。B-QUOTIENT/bias = 可直接竞争的训练/校正 arm；B-OBS/B-CROSSSITE = 科学诊断，不应默认升格为计算方法。

## 1. 四轮贡献如何整合，而非留下四套互相竞争的结论

| 来源 | 可保留的非重复成果 | 本次统一解释 |
|---|---|---|
| [第一轮：独立方法评议](INDEPENDENT_REVIEW.md) | 同态敏感度配对、末层 LN/head Jacobian 的固定 readout＋状态秩一修正、65 中间统计与 direct64 比较、B-F 物理成本 | 强化 B-NATURAL/B-PREDICTABLE/B-COMPUTABLE 三个不同命题；数学可表示 ≠ 在线学生可实现 |
| [第二轮：前层/多步评议](EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md) | 全后缀连续 Pre-LN gauge、单状态 Jacobian 零空间维数、Self-/Cross-Attention 局部结构、scheduler 有限时域终端风险、hybrid telescoping | Gauge 是条件全局函数对称；98.75% 是单状态维数界，不是可删通道比例；跨步 H 需任务读出 Q |
| [第三轮：对第二轮回应](RESPONSE_TO_SECOND_AGENT_20261028.md) | B-QUOTIENT 与共享可忽略误差的必要区别；80 来源共享核反例；拆开两种 JOINT；两位置四臂 finite interaction | 优先低成本基线；状态配对与跨 site 有符号交叉不得同名误用 |
| [第四轮：回应评议](FOURTH_ROUND_REVIEW_OF_QUOTIENT_RESPONSE_20261028.md) | A0 行均值指标源码纠错、行×通道四分量、同 expert 双 step 共享 folded bias 最小二乘、gate 中心化次序 | 更准确的下一动作是先测真实分量/向量相干性，再决定 quotient 或 bias 的训练/校准投资 |
| **全局审视（本综合）** | 每项理论如何与可学习 FFN、合法 64-group Q2、真实 kernel、四步动态连接；重新分离短期工程价值与长期科研价值 | 结束递归口头争论，以明确的可复用证据、差异预测和可部署对象继续 |

以上是**内容上的批判综合**，不是将多 Agent 讨论改写成原作者的正式支持票。

## 2. 正确的原生计算对象：输入、数值、空间与时间

当前冻结研究基线：LightX2V Wan2.2 I2V A14B four-step high/high/low/low，两个 expert、每个 40 blocks；共 **80 个 expert/block 参数函数、160 次按 step/block 的 FFN 调用**。典型 M=31668 latent patch tokens，D=5120，hidden H=13824；81 个输出 RGB 帧、21 latent 时间槽与四个去噪 step 属于不同统计轴。

单次真实 FFN 主路径：

1. post-Q1 的合法输入 x：[M,5120]；
2. W1：5120→13824，GELU(tanh)，Smooth2 和 HiF4 Q2(group64→8→4)，W2：13824→5120；
3. pre-gate y：[M,5120]，调用者 e5 gate D_g、残差 r，post-write u=r+D_g y；
4. 后续 Pre-LN self/cross/FFN 或最后 LN＋时间调制＋64D head，再经 flow/scheduler/后续 step 到视频行为。

**在线合法信息**：当前 x、调用者已有 r/g/t/conditioning、离线冻结的模型参数与已付费历史状态；teacher 全 y、完整后缀 J、未来 step 结果仅为离线校准 oracle，不能计入“免费选择信号”。现有 HiF4 实现是 fake-QDQ＋BF16 GEMM；packed FP4、BF16 BF16 和实数代理分别验收。

**独立样本**：image/source cluster；每 source 256 sampled rows、不同 step、不同 site 是嵌套测量，不变成 256 或 160 个独立来源。

## 3. 已证数学：它究竟允许我们忽略哪一类误差？

### 3.1 逐 token 通道平移：条件精确等价

定义通道常量投影 P_C=11ᵀ/D、中心化 R_C=I−P_C。若后续所有读取原残差的非残差分支先经过 token-wise LN（包括 cross norm3）、gate/time/context 固定且没有未归一化直读，则

$$
B_l(X+b\mathbf1_D^\top)=B_l(X)+b\mathbf1_D^\top,\qquad
H_l(X+b\mathbf1_D^\top)=H_l(X).
$$

对 pre-gate 学生错误 e=\hat y−y，固定当前 gate 后，若每个 token 满足

$$
R_CD_g e=0,
$$

则 post-write 差异是通道常量平移，当前连续后缀最终 flow 精确不变。多个 Block 若依次在对应 Dense 的规范输入上满足该条件，可沿残差归纳得到同一 step 的连续 flow 不变。这是强**充分条件**，不是 BF16 bitwise 身份，也不是某个 64-group 自然可删除的结论。实际 LightX2V runtime 的 norm3、cross_attn_norm、adapter/before_proj 以及 BF16 有限响应须用冻结代码/数值验证。

**最便宜可执行化**：普通 A0 学生原算子保持不变，只将训练监督从 ||D_g e||² 改为 ||R_CD_g e||²。在线无新增运算，可能提高同成本质量，尚未观察到收益。注意中心化与 gate 通常不交换：必须先得到 post-gate E，再沿通道轴中心化，而非先中心化 pre-gate e。

### 3.2 大单点 Jacobian 零空间：可证明的上界，而不是压缩率

当前 flow 输出维数为 64M，故 J_s:R^(5120M)→R^(64M) 至少有 5056M 个单点一阶零方向（输入维度的 98.75%）。但在 80 个不同 source 取一般位置的 64 维 rowspace 并使其共同覆盖 5119 维中心化空间时，所有 J_s 的共享零空间可仅剩每 token 通道常量方向。

**可用于研究的对象**是合法共享学生函数 e_φ(x_s) 与 downstream 作用的组合，且受到核算的 OnlineCost(φ) 约束，而不是所有随 source 自由旋转的 oracle nullspace。

### 3.3 末层读出、65 统计与 fixed＋rank1

固定末层 time modulation 与 gate，令 z=R_C(r+D_gy)、σ²=||z||²/D+ε、T=W_head diag(1+scale)：

$$
F(u)=Tz/\sigma+c,\quad
J_y\delta y=\sigma^{-1}TR_CD_g\delta y
-\frac{(Tz)(z^\top D_g\delta y)}{D\sigma^3}.
$$

这是 **固定读出＋状态依赖至多 rank-1 归一化项**。连续 head 可由 q=Tz（64D）和 v=||z||²/D（1D）重建，但 65 个中间充分统计**不意味着压缩复杂度低于直接预测 64D head**。必须比较 direct64、q/v、残差增量 Δq/Δv、同预算 A0；block39 仅覆盖四步中的四次 FFN 调用，不能承担整网加速主张。

### 3.4 两种 JOINT 与真正的 site 干涉

- **B-JOINT-STATE**：同 source 的 G_s 与 C_s 配对，E_joint=E_s||J_sD_g(I−P)(y_s−μ)||²，对比边际均值分解；既有 ALG 反例能反转排序，但实际 Wan 配对增益仍 OPEN。
- **B-CROSSSITE**：多个 site 的误差在 terminal 空间可能有正负内积；它与上面的 source 配对是不同数学对象。E0 两源 Whole-Flow 已有先前 state 偏差传播与当前候选作用的有符号交叉，不是两个不同 FFN site 的实测。

真正要区分“非线性交互”与“仅向量相消”，固定两 site A/B 并定义四臂 Y00/Y10/Y01/Y11：

$$
I_{AB}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
$$

即使 I_AB=0，两个独立有限效应的内积也可为负。这一 factorial finite hybrid 是较单向 telescoping 更有信息的对照，后续需按 ACTIVE 冻结。

## 4. 第四轮的关键源码修正：两种中心化不能混用

对同一 source/step/block 的真实配对 **post-gate 连续误差** E∈R^(R×D)，R 为旧 pilot 保存的 256 行，D=5120。定义

$$
P_R=\mathbf1_R\mathbf1_R^\top/R,\quad R_R=I_R-P_R,\quad
P_C=\mathbf1_D\mathbf1_D^\top/D,\quad R_C=I_D-P_C.
$$

正交分解：

$$
\boxed{E=P_REP_C+P_RER_C+R_REP_C+R_RER_C.}
$$

四项的 Frobenius SSE 精确加总。它们含义依次是：共同均值的规范方向；跨行常量且通道可观测的偏置；随行变化但规范不可观测的漂移；随行/通道共同变化的函数残差。为保持相同写回 gate 语义，E 必须先由 (student_pre−teacher_pre)×g 构造再做两轴投影。

**A0 真实源码**：[compact6016_adapter.py](../../../tools/compact6016_adapter.py) 的 error_metrics 先计算 error64=pre_error.double()，随后 mean_error=error64.mean(0,keepdim=True)。这是 **跨行 R 求均值、生成 1×D 通道向量**，对应 η_row=||P_R E||²/||E||²，而非 η_gauge=||E P_C||²/||E||²。

A0 energy94 的原生 pilot 在四个 (step,block) 的 relative gated SSE 为 **2.349%、1.061%、2.546%、2.045%**，所报告的行均值误差份额分别 **3.393%、22.257%、13.396%、16.823%**。这是 train16/已曝光 heldout8 的 teacher-state 局部、来源等权比例；**这些数字不能推出 gauge 份额、四分量误差，也不能当作 free-video 改善**。

**统计可取得性**：已有 report/JSON 中的 scalar SSE、mean_SSE 不是足以恢复四分量的向量。先寻找合法、同坐标配对保存的 E、g 和 Student 输出；不存在时考虑注册最低成本的重放 A0 exported weight packs 与真实 saved x/y/g，而不是重跑 Dense 视频。

## 5. 零额外在线算子对照：bias 与 quotient 的正确比较

### 5.1 Folded W2 bias：先用可辨认的来源相干检验

A0 当前 rank32 LoRA 仅训练 W1/W2 低秩因子，W2 原 bias 固定。若 pre-gate 残差 e_s 在不同 source、同 expert 的两个共享 step 中存在稳定 channel-wise 偏置，则固定每 expert/block 的 δb 可折入既有 W2 bias，**部署没有额外 GEMM**。

对冻结来源/step 权重 w_s,t，连续 gated-L2 最优坐标为

$$
\delta b_c^*
=-\frac{\sum_{s,t,p}w_{s,t}\,g_{s,t,c}^2e_{s,t,p,c}}
{\sum_{s,t,p}w_{s,t}\,g_{s,t,c}^2}.
$$

权重须与原 source-normalized loss 相容；零分母坐标单列。先报告每来源行均值向量之间的方向相干和 heldout 改善，再决定 folded BF16 的原生复核。**同 expert 两 step 共享同一套 block 权重**；单 step 最优 bias 不能冒充可同时部署在两个 step 上的权重。此项是成熟 bias correction baseline，不单独宣称创新。

### 5.2 B-QUOTIENT：先测可被忽略的份额，再考虑训练

对同一份 E，有精确 loss 分解：

$$
\|E\|_F^2=\|ER_C\|_F^2+\|EP_C\|_F^2,\qquad
\eta_{\mathrm{gauge}}=\|EP_C\|_F^2/\|E\|_F^2.
$$

各向同性误差的 1/D≈0.0195% 仅为量纲零假设，不是 Wan 观测值。η_gauge 不小，也不能直接推断重新优化后的收益大小；η_gauge 很小则说明此特定旧 Student 的**当前误差**较少落在规范方向，应据此降低只调整 loss 的试验优先级。

有区别性的训练 arms：A0 普通 gated loss / A0 quotient loss / A0 folded shared bias（可选与 quotient 正交组合），同窄 Q2、LoRA rank32、真实两 step 权重共享、source 级预算及独立验证。要同时记录普通 gated SSE、quotient SSE、真正 BF16 writeback/head/flow、Student free-run。

### 5.3 为什么这些 baseline 重要而不是主方法

偏置修正与 LN 平移不变性均有文献先例；本轮的新信息来自其**在真实 Wan gate、Q2、shared expert 和 downstream 轨迹中的定量作用**。若只改善 teacher-state 局部 loss，还不是原创的动态 FFN 算子，也未证明省去任何新的 W1/Q2/W2 工作。它们的价值是使 A0 和 M-B 的强对照更公平，释放研究资金给更难的机制和真实算子。

## 6. 当前证据的全局矩阵：事实、缺口、如何改变决策

| 路线 / 提议 | 已有证据 | 尚缺的决定性证据 | 使用方式 |
|---|---|---|---|
| **M-A0 width6016** | 四站点、16 train/旧 heldout8、rank32×128更新、native narrowed Q2；理论主GEMM为原 FFN 的43.5185% | 其余 site、shared expert 双 step、Student 自己的四步轨迹、独立来源视频动态、wall/VRAM | 工程强基线、最接近可运行低成本函数；不把局部恢复推广成整网完成 |
| **旧 UB1 consumer20** | 已曝光20来源：imaging 72.629→71.626（约−1.38%），RAFT 连续6.894→1.152（−83.28%），动态程度6/20→0/20 | 真正运动 vs RAFT 纹理观测器、不同版本和 A0 的因果连接 | 明确「外观保真、运动衰减」的功能动机；不是 M-B 已失败 |
| **M-B native B0/B1** | 条件 ALG 与末层源码数学、联合/边际反例 | 相同状态 post-write u/head/flow 中性、BF16 finite、真实 paired vs marginal/fixed/shuffled 来源优势 | 维持当前已指定首要方法 B0/B1；决定下游统计是否有真实价值 |
| **B-QUOTIENT / bias** | 源码与 ALG，可构造 matched 同成本对照 | E 的四分量、来源相干、共享步 bias、再训练与 native rollout | 支持/质疑训练归纳偏置，属强 baseline |
| **B-FACTORED** | MAC 公式：h6016/r512 为 36,503,552 MAC/token（A0 的59.26%）；折叠形态没有额外在线 MAC 节省 | 量化舍入、额外 GEMM、临时张量、真实 latency、全模型视频 | 真正具备改变运算结构潜力；需硬件数据 |
| **M-C** | 输入压缩信息界、输出秩界、LUT 资源预算 | 真实编码碰撞、模型可训练性、大 M memory/kernel、视频 | 保留并行高风险创新，不把它受制于 B |
| **Activation 2:4 C5** | 合成 A100 同次 online M×K×N 中位数与 Dense 比约+3.5%/+0.7% 速度余量 | 真实 Wan 端到端 QDQ/FFN/视频、公平交错 SYS 统计 | 提醒稀疏/低MAC 不天然加速；改变方向应从物理瓶颈出发 |

**物理成本关键对照**：Dense FFN W1+W2≈141,557,760 MAC/token；A0=61,603,840（43.52%）。这是结构算术预算而非系统测量；将 loss、bias、head-oracle 变得更精确，并不会自动缩减此 MAC。甚至只加速 block39 也仅覆盖四步 160 次 FFN 调用中的 4 次（若各站点算量近似等额，为主 FFN 部分约2.5%，尚不包含注意力/VAE）。

## 7. 下一Agent工作包：能交付什么，何时改变决策

**W0 — 当前低成本离线诊断（I/F，parallel，优先作为评议后的最快行动）**

- 输入：实际 saved x、teacher pre-gate y、gate、A0 exported student packs/相同坐标。缺完整 E 时先查缓存，最小合法 replay；不要从 scalar report 硬反推向量。
- 实现：对每个 source×step×block，先构造 E=(ŷ−y)⊙g；对 R=256 行与 D=5120 通道分别投影；核对四项 SSE closure；计算 η_row、η_gauge、η_bias_observable 和行/通道共同剩余；来源等权，给 p50/p90、跨源行均值通道向量相干；同时计算 actual BF16 writeback 分开核验。
- 输出：带实际 asset/protocol SHA、source exposure、每来源记录和 OPEN/MEASURED 身份的四分量 JSON/简报；**当前未执行**。
- 判别：gauge 占比高且 heldout 方向稳定→B-QUOTIENT 进入同预算训练；稳定行均值可观测偏差高→shared folded bias 对照；若两者小→直接回到复杂函数/后缀/轨迹研究。

**W1 — 当前正式指定的 M-B B0/B1（F/I，保留原任务路由）**

- 原生 post-write full-M u、r/g、真实 final LN/head/flow + neutral；固定 source/step/状态，区分 FP32 math/BF16 finite；
- fixed head 与 rank1、PCA、marginal、paired、source-shuffled 同 rank/source heldout，且绝不借 teacher J/y 作为在线特征；
- 用来源配对排序确实不同且学生在相同成本可学来判断进入 B-H/B-F 的资源强度。完整 B0/B1 以原交接与 ACTIVE 冻结后执行；**当前仍 OPEN**。

**W2 — 工程主轴 A0 完整可用性（C/F，战略优先建议，不改 ACTIVE）**

- 逐层/两 expert、共享两 step 做 full coverage；以相同来源身份区分 teacher-state、Student-state、双 step、末层输入扩散；
- 同初始噪声自由生成，对主体/背景/动作幅度/ROI/相机/时序语义、imaging、真正独立来源和所有保留指标做聚合，避免更静止视频因某项一致性指标获得奖励；
- 输出全 FFN MAC/QDQ/launch/权重驻留/每 step latency/整端 wall/显存，配合 matched Dense、A0 与新增方法比较。只在实际出现质量/成本收益时提出部署 Pareto 主张。

**W3 — 机制与创新分叉（I/F/C，按辨别结果投入）**

- quotient/bias 有益：作为 A0 强控制和适配资产，并继续探索跨步保持；不以 LN/bias 命名为主要创新；
- M-B paired 超出 fixed/quotient 且同成本学生能兑现：推进可扩展早层下游作用（包括跨 token attention、terminal risk）并形成真正 B-F 算法候选；
- 单点作用足够、但 full-video 漂移明显：研究 scheduler/attention/post-FFN signed propagation，做两站点四臂判别，按源/seed 复现；
- B-F 的 factored kernel 真正节省 wall：将表达与执行联结；否则调整 O-D 的布局/融合或探索 M-C/新算子。

**排序原则**：并行工作包不建立新的门槛链。W0 最便宜且补足第四轮欠缺的向量观测；W1 是现行 M-B 指定任务；W2 为近期工程可用性战略优先建议；W3 按观察调整，不自动扩大 GPU。项目可以同时有一个成熟基线、一个机制前沿和一个新算子探索。

## 8. 不再重复的辩论与论文主张范围

**已关闭的代数问题**：连续 LN 平移恒等式、Jacobian 输出秩上界、固定 head Jacobian 和 65 连续中间统计、四正交误差分解、固定 student 的加权 LS bias、finite hybrid 望远镜等式。这些是 ALG，不需要反复举新的文字投票来证明。

**仍 OPEN 的真正科学争议**：当前 runtime 是否符合 gauge 条件；真正 gauge 误差占比；来源条件 $G_s,C_s$ 配对优势；从合法 pre-W1 输入对这些坐标的低成本可预测性；跨 site finite 互作是否有复现性；真实 BF16/Q2 下的函数和视频行为；FFNDS 新结构与成熟方法相对的增量；全系统质量–速度 Pareto。

**最近先例的正确位置**：LayerNorm 的投影几何见 Brody–Alon–Yahav Findings ACL 2023；bias correction 见 Nagel et al. ICCV 2019；CoBRAS 的 state/adjoint 平衡投影、2ndMatch 的 Jacobian response 匹配、TinyFusion/相关 compact DiT 的恢复训练均是强竞争机制。论文级贡献应识别 **Wan 门控和真实 Q2、共享多步结构、非平凡函数可预测性、视频动作保持、物理计算** 的增量，而不将已知 LN 或 bias identity 重新包装成新发现。相关原始论文/源码 URL 与适用条件保留在四轮文档及原 source ledger 中。

**科学防偏移规则（本综合建议）**：新评议需要改变一个实际决策——新的精确不变量、首次可得信息、允许的学生结构、不同的实测预测或成本结构；若仅重复已证 ALG，可归入单一来源注解，而不再新增一轮独立权威文件。机制研究本身具有独立价值，不因未立即具备部署加速而失效；部署主张则必须有真正低成本和完整视频证据。

## 9. 原始资料、Skill、版本维护

- [首份 M-B 评议](INDEPENDENT_REVIEW.md)
- [前39层与四步后缀扩展](EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md)
- [第三轮对话回应](RESPONSE_TO_SECOND_AGENT_20261028.md)
- [第四轮源码/误差评议](FOURTH_ROUND_REVIEW_OF_QUOTIENT_RESPONSE_20261028.md)
- [现行权威 M-B v2 方法快照](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)
- [M-A/M-B/M-C 统一方案 v2](../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)
- [实验和原生 SRC：A0 pilot](../../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)、[adapter](../../../tools/compact6016_adapter.py)、[UB1 consumer20](../../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)、[E0 whole-flow](../../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[C5 稀疏 kernel](../../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)
- [FFNDS 领域 Skill](../../../.agents/skills/ffnds-paper-discovery/SKILL.md) · [mechanism-research](../../../.agents/skills/mechanism-research/SKILL.md) · [problem-reformulation](../../../.agents/skills/problem-reformulation/SKILL.md) · [research-expansion](../../../.agents/skills/research-expansion/SKILL.md)
- [研究导航](../../RESEARCH_NAVIGATION.md) · [ACTIVE 执行状态](../../ACTIVE_RESEARCH.json) · [讨论规则](../../method_discussion/AGENTS.md)

**维护边界**：本文为截至本轮源状态的单份综合判断。未来原生结果改变路线时，应由对应 RUN/report 更新事实源，再有目的地修订当前整合方法、图谱和导航；保留四位评论者的原始观点与提交历史。此文本写入本身只改变文档，不产生新的实验结果、运行许可或同行背书。
