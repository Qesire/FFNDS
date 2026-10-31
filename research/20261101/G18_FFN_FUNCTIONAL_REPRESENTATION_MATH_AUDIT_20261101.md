# G18 · 从 Attention 重表示到 Wan FFN 函数表示：严格复审、数学边界与视频可证伪方案

2026-11-01 · **AUTHOR_SELF_AUDIT / DERIVATION_REVISED / DESIGN_ONLY / SYNTHETIC_ALGEBRA_FIXTURE / NO_NEW_NATIVE_OR_VIDEO_RUN / INDEPENDENT_EXPERIMENT_REVIEW_REQUIRED**

> 本文专门复审此前对话提出的 GELU 奇偶分解、函数群 Taylor 聚合、FFN 局部 Jacobian 与轨迹切空间方案。定位为新的 **FFN 函数表示候选空间**，不取代 [G17 v3 的完整视频优先机制方案](../20261031/G17_V3_VIDEO_FIRST_FFN_MECHANISM_FULL_ROLLOUT_DESIGN_20261031.md)；与 [Dense/Student 全深度、A5/HWG 联动审计](../20261031/G17_V3_TENSOR_HWG_CODE_ARCHIVE_AUDIT_20261031.md)并行。没有改 ACTIVE、没有 GPU/原生 Wan 数值、完整视频或独立 Agent 评审。本篇的“复审”是作者自审，不是独立审批。

## 0. 科学问题及证据边界

**目标**：寻找能保留 Wan2.2 I2V 完整视频动作/事件/语义/时间质量，同时减少 *真实* FFN W1/GELU/W2 成本的低复杂度函数表示。原先的神经元坐标是参数化方式；是否也是最优执行基，是待验证的科学问题，而非既有结论。重要的是原函数如何被表示、低成本需要哪些充分统计量、误差如何影响视频，而不是预先追求稀疏率、补算或缓存。

两个 Attention 启发：[HyperMLP, ICML 2026](https://proceedings.mlr.press/v306/lu26aq.html) 将自回归 Attention 头从 context-instantiated dynamic MLP 角度重述；[MiTA, NeurIPS 2026 论文作者实现](https://github.com/QishuaiWen/MiTA) 从可压缩/可路由的动态 fast weights 设计新 Attention。类比仅提供**研究步骤**“等价视角→设计自由度→新算子→任务验证”；不能从 Attention 的 K/V 动态槽位推出固定权重 FFN 也能等价 top-k。已知 FFN 字典记忆视角见 [Geva et al. EMNLP 2021](https://aclanthology.org/2021.emnlp-main.446/)。

**数学层级**：ALG_EXACT（实数代数）→ LOCAL_CONDITIONAL（局部且须满足假设）→ STRUCTURE_HYPOTHESIS（需原生资产支持）→ VIDEO_CAUSAL（真实完整自由生成）→ NATIVE_SYSTEM（计入物理执行与成本）。不可跨级宣称。

## 1. Wan FFN 计算边界与统一记号

采用行向量：`u∈R^{1×d}` 为已经完成 norm2 和 e3/e4 调制的 **当前实际输入**；`W1∈R^{d×m}`、`b1∈R^m`、`W2∈R^{m×d}`、`b2∈R^d`。定义

\[
a=uW_1+b_1,\qquad h=\phi(a),\qquad y=hW_2+b_2,\qquad x'=x+(y\odot e_5).
\]

如果某实际 checkpoint 的线性层省略偏置，则对应偏置取零；数值后端的 Q1/Q2、量化、smooth、pack 闭包必须在真实实现核对。历史研究的典型 `d=5120,m=13824`、HHLL 4 step × 40 layer 共 160 个实际 FFN site，不能把 expert 历史对与这些 site 混为独立来源。具体维度与算子布局仍须针对运行版本复核。

令 `w_j=W1[:,j]`、`v_j=W2[j,:]`，原函数是

\[
f(u)=b_2+\sum_{j=1}^m \phi(u w_j+b_{1j})v_j.
\]

这是**向量值 ridge-function 字典展开**。已有键值记忆和基函数观点；作为解释有用，但本身不是新发明。门控 `e5` 必须在当前同状态比较中于输出之后作用一次，避免 G16 曾发现的重复 gate 问题；学生自由传播时 `u,e5` 都可能改变。

## 2. GELU 奇偶分解：准确的 ALG_EXACT 与修正

精确 GELU `φ(x)=xΦ(x)` 满足 `Φ(-x)=1-Φ(x)`，所以 `φ(x)-φ(-x)=x`。PyTorch `approximate='tanh'` 的 `φ_t(x)=x[1+tanh(c(x+0.044715x³))]/2`，其中 `c=√(2/π)`，内层为奇函数，也严格满足该恒等式（指此近似函数的实数表达，非“它与精确 GELU 完全一致”）。

令 `ψ(x)=φ(x)-x/2`，则 `ψ(-x)=ψ(x)`（偶函数），严格得到

\[
f(u)=uA+c_0+\sum_j\psi(uw_j+b_{1j})v_j,\qquad
A=\tfrac12W_1W_2,\quad c_0=\tfrac12b_1W_2+b_2.
\]

**重要修正**：`ψ(x)` 在 `|x|→∞` 时近似 `|x|/2`，因此“偶函数部分”并非数值上很小的 residual，也不能据此预言它易于截断。把每个 GELU 先改成 `uA` 再完整保留所有 `ψ`，只会额外增加一条 `d×d` 计算路径，不能宣称省算。

另一条严格的**反向 preactivation 配对**恒等式，若两个原子满足 `a_k(u)=-a_j(u)` 对所有输入，即 `w_k=-w_j,b_k=-b_j`，则

\[
\phi(a_j)v_j+\phi(-a_j)v_k
=\frac{a_j}{2}(v_j-v_k)+\psi(a_j)(v_j+v_k).
\]

近似反向配对可研究，但在 Wan 中是否存在这样的足量权重结构完全 OPEN；若不存在，不得强制配对制造伪加速。

## 3. 函数群：精确合并与有余项的一阶表示

将神经元划分为 `G` 个互斥组。若组内 `w_j=\bar w_g,b_{1j}=\bar b_g` 对所有输入严格相等，则

\[
\sum_{j∈g}\phi(uw_j+b_{1j})v_j
=\phi(u\bar w_g+\bar b_g)\,C_g,\qquad C_g=\sum_{j∈g}v_j.
\]

这是严格等价的**同函数原子聚合**。仅 `w_j` 余弦相似、高相同输入响应、或 `W2` 共线都不充分。特别地，组内正负 `v_j` 的抵消属于联合输出；不能只根据单独神经元能量去定义组价值。

一般情形写

\[
w_j=\bar w_g+\epsilon_j,\quad b_{1j}=\bar b_g+\eta_j,\quad z_g=u\bar w_g+\bar b_g,\quad\delta_j=u\epsilon_j+\eta_j.
\]

Taylor 定理：

\[
\phi(z_g+\delta_j)=\phi(z_g)+\phi'(z_g)\delta_j+R_j,\quad
|R_j|\le\tfrac12 M_{2,g}(u)|\delta_j|^2,
\]

其中 `M_{2,g}(u)` 是连接 `z_g` 与全部 `z_g+\delta_j` 区间内 `|φ''|` 的上界；若统一全局二阶导数上界存在也可使用。于是

\[
f_g(u)=\phi(z_g)C_g+\phi'(z_g)(uM_g+c_g)+E_g(u),
\]

\[
M_g=\sum_{j∈g}\epsilon_jv_j\in\mathbb R^{d×d},\qquad
c_g=\sum_{j∈g}\eta_jv_j,\qquad
E_g=\sum_{j∈g}R_jv_j,
\]

\[
\|E_g(u)\|_2\le \tfrac12M_{2,g}(u)\sum_{j∈g}|\delta_j|^2\|v_j\|_2.
\]

**核心限制 1**：上界按范数求和较松，丢失了 `v_j` 的有符号抵消；需同时记录实际 `E_g`、交叉项、组间协方差和尾部，不能只用此上界做排序。

**核心限制 2**：`M_g` 一般是稠密 `d×d` 矩阵，甚至各组接近满秩。写出一阶聚合式 **不构成压缩证明**。只有存在合法、静态可存储或低价可选的 `M_g≈L_gR_g`（rank `r_g`）且总成本有余量时，这才是一个候选省算算子。低秩误差需与 Taylor 余项分开：

\[
\|\Delta f_g(u)\|_2
\le \|E_g(u)\|_2+
|\phi'(z_g)|\,\|u(M_g-L_gR_g)\|_2 .
\]

门控后，同状态输出误差是 `(Δf_g)\odot e5`，不应默认其范数等于 pre-gate 的误差。

**核心限制 3**：分组中心从校准输入拟合得到时，低校准误差不证明来源外、不同 expert、不同 step 的小 `δ_j`；尤其不保证学生自由状态持续落在该区域。

**核心限制 4**：多组误差的和 `∑_g E_g` 可以有抵消/增强；局部误差与完整视频质量不存在已证明的单调关系。方法候选可以“完全不补算”，不需要把原 Dense 恢复作为目标。

## 4. 真实成本不等式：上一版遗漏的主约束

先忽略偏置、非线性标量代价和 memory/launch；原 FFN 的 W1+W2 理论 MAC/token 约 `C_dense=2dm`。

**方案 B0（直接对原 GELU 函数群做零阶/一阶聚合）**：`G` 个中心投影及输出，`R=Σ_g r_g` 个一阶低秩矩投影及读出，理想主要 MAC/token 约

\[
C_{B0}\simeq 2dG+2dR,\quad
\text{仅在 }G+R<m\text{ 时满足理想 MAC 优势。}
\]

**方案 A+B（先折叠奇部，再对偶函数 `ψ` 的字典聚合）**：主 MAC/token 约

\[
C_{A+B}\simeq d^2+2dG+2dR,\quad
\text{仅在 }G+R<m-\tfrac d2\text{ 时满足理想 MAC 优势。}
\]

这里 A+B 聚合中心应作用于 `ψ`，其导数与二阶余项也必须换成 `ψ′,ψ″`；不能直接用对原 `φ` 的 B0 近似再加 `uA`，否则**双计线性部分**。

以研究记录 `d=5120,m=13824` 为例，A+B 的理想必要条件为 `G+R<11264`；B0 为 `G+R<13824`。这些只是忽略访存/预处理/核效率的粗 MAC **必要条件**，绝非实测速度保证。朴素存储每个 `M_g` 需要 `Gd²` 权重；低秩存储约 `2dR` 权重，折叠线性骨架另需 `d²` 权重。所有 W1/W2、组中心、参数缓存、rank factor、转置/切分、门控、reshape、量化 Q1/Q2 和 kernel launch 须单独计算真实成本。

最低合理对照：Dense、同 MAC/形状的紧凑学生、已有 Chipmunk/TEAL/SharQ、单纯神经元静态聚类、零阶聚合、B0 一阶/低秩、A+B，所有方案把准备/校准成本和在线成本分开登记。

## 5. Jacobian / 局部动力学：从可观察方向到执行边界

在当前 FFN 调制输入 `u` 不变、只对 `u` 求导时：

\[
J_f(u)=W_1\operatorname{diag}(\phi'(uW_1+b_1))W_2,\qquad
f(u+\Delta u)-f(u)=\Delta u\,J_f(u)+O(\|\Delta u\|^2).
\]

该公式在行向量约定下正确。**它不是整个 Wan Block 对 `x` 的 Jacobian**：若要求完整 block 对 `x` 的导数，还须链入 norm2、e3/e4、post-gate e5、residual，进而是 Attention、后缀、scheduler 与后续步。固定门控的局部 pre-gate 方向不能冒充视频事件因果方向。

若真实输入扰动近似位于 `r` 维切空间：`Δu≈zP_T^T`，可做观察性投影 `Δf≈zP_T^TJ_f`；然而即使 `r≪d`，显式构造 `P_T^TJ_f` 仍可能需要完整 `m` 激活或大矩阵，切空间还可能随着来源/step/层变化。**低维轨迹 ⇒ 低在线计算** 没有被证明。此路线现阶段定位为 G17 VIDEO/TENSOR/HWG 机制解释和下一阶段可证伪结构假说，而非独立加速算法。

## 6. 构造强反例，而非局部优化代理

| 反例/竞争解释 | 对数学表示的实际挑战 | 能改变判断的证据 |
|---|---|---|
| 组内 W1 相似但 φ' 横跨变化区域 | 同向权重不足以控制 `δ` 和二阶误差 | 跨 source/step/High-Low 的实际 `φ,φ',δ,E` 分布 |
| `v_1=-v_2`、输入原子近似相同 | 原子级能量很高但联合输出很小 | signed cross、输出 Gram 和候选组实际效应 |
| `M_g` 很高秩 | Taylor 折叠无真实物理优势 | rank—误差—MAC Pareto、原生 kernel |
| 同状态函数小误差但学生自由输入漂移 | teacher 校准不能验证自由视频 | Dense shadow、真实学生全160位置和完整4step视频 |
| RAFT 变小但主体动作仍正确 | 相机/背景、成像或追踪失败可混淆运动标量 | 主体相对相机运动、事件、整视频盲审与失败样本 |
| BF16 vs HiF4/Q1-Q2 并列混测 | 数值闭包和量化时点改变算子 | 同一 numeric family 独立复算，不跨家族做配对因果 |

[G17 v3](../20261031/G17_V3_VIDEO_FIRST_FFN_MECHANISM_FULL_ROLLOUT_DESIGN_20261031.md) 的 UB1 动态衰减、Chipmunk High/Low 差异与旧 Y 复用不作为“必有运动通道”的证明；应检验函数群方案能否解释已有动作/事件、伪影与来源异质性，并给出与一般容量缩小/学生后缀传播相反的预测。

## 7. 最小有决策价值的研究程序

1. **R0 / 现有来源只读审计**：优先核对既有 898/979/1040 HiF4 matched fullM、96 缺口，以及真实 BF16 Chipmunk 全160来源张量。按原 RUN 保存 source/step/expert/layer/token/ROI/numeric_family/teacher_or_student 的身份；校准来源与验证来源隔离，原 A5 producer 缺失维持 MISSING。
2. **R1 / 数学表示结构检验**：分别于 preactivation、post-GELU、pre-gate Y、post-gate 写回计算原子响应距离、signed 联合输出、`δ` 分布、余项、`M_g` 谱、低秩逼近的真实 FLOPs/储存。另做正负近似配对的存在性检查；若稀少及时关闭。
3. **R2 / 原生完整函数强对照**：仅对 R1 显示可压缩余量的若干预注册组，构造无缓存/无 Dense 补算的 B0 与 A+B，可静态导出的独立 High/Low 函数群，和 matched compute 的紧凑学生/HWG/P64 强对照。按真实 Q1/Q2、e5 和 native 执行图核查组/低秩 readout 成本。
4. **R3 / 完整视频行为干预**：经独立实验复审、ACTIVE 冻结和身份复核后，在同一数值族以 4step×40 FFN 做 free-rollout；优先从已存在的视频失败建立动作/事件假说，再新增有判别力的来源。主指标为完整 81 帧动作事件、语义、主体/背景、时间、相机、成像、原图保真、视频可视审查与实际系统成本，来源/image-cluster 为统计单位。
5. **R4 / 决策门**：若无低秩谱或余量→停止函数群加速路线，仅保留数学解释；若局部良好而完整视频失败→重新审视所保留的功能与状态传播；若视频良好而 SYS 无收益→仅机制论文线索；若跨来源真实视频与原生速度/显存同时占优，再展开最近先例差异和论文主张。

**下一次研究实验前必须遵守** [独立预审](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)：真实不同 Agent 审核方案和反例、作者逐项修订、reviewer recheck、ACTIVE 冻结。本文没有授权新增 GPU/CPU 科学实验。单纯小矩阵代数 fixture 属于静态软件检查，不代表真实 Wan 质量。

## 8. 新颖性初审与来源

- [HyperMLP — ICML 2026](https://proceedings.mlr.press/v306/lu26aq.html)；[MiTA — NeurIPS 2026 作者实现与论文链接](https://github.com/QishuaiWen/MiTA)：提供“重解释→新计算算子”的方法论，未提供 Wan FFN 压缩保证。
- [Geva et al. 2021 — FFN as Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/)：键值字典解释不是本文新颖性。
- [NeuronMerge, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.471/)：研究神经元功能分组用于模型合并，不等于已直接实现本文同函数群 Taylor 聚合；它构成相近先例和必读文献。
- [Merging Feed-Forward Sublayers for Compressed Transformers, 2025](https://arxiv.org/abs/2501.06126)：侧重跨 FFN 子层合并与激活相似性，说明“合并”不是新概念。
- [Neuron merging via inverse-activation regression, 2026-10 preprint](https://arxiv.org/abs/2610.02559)：展示非线性神经元合并已经有多种拟合方案；限定原论文 sigmoid 网路，不能照搬其结果为 Wan GELU 的有效性证明。
- [PyTorch GELU 规范](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)：核对 exact/tanh 两数值定义；实际 Wan 路径以 run 版本为准。

**当前新颖性判定：OPEN / NEEDS_CLOSEST_WORK_COMPARISON。** 数学分解、Taylor 一阶、神经元聚类和低秩分解皆非独立新颖性；只有在 **Wan FFN 可证实的新函数结构 + 新的低成本算子 + 实际视频行为选择性 + 独立来源视频质量与 SYS Pareto** 同时成立后，才可能提出具体论文贡献。

## 9. 本轮自审与静态 fixture 收据

检查项：GELU exact/tanh 奇偶恒等式、线性折叠、反向对偶表示、完全相同原子的精确合并、Taylor 一阶聚合式、导数、低秩代价、不允许神经元任意旋转、post-gate 与数值族边界。配套 [小矩阵数学 fixture](../../tests/test_ffn_functional_identities_20261101.py) 使用 NumPy float64 固定 seed 20261101，不用真实 Wan/权重/视频：实数等价式在容差内通过，缩小 perturbation 使 Taylor 截断误差明显下降。局部随机测试不能替代高维、BF16、FP4、物理 kernel 的验证。

**作者自审决定：REVISE_AND_RETAIN_AS_REPRESENTATION_HYPOTHESIS。** 目前只有严格代数、条件性余项和理想 MAC 门槛；方法尚未形成被检验的物理候选，也没有完整视频质量或速度的新证据。后续由不同 Agent 独立评议 *拟执行的具体实验*，而不是把这篇作者自审当作批准。
