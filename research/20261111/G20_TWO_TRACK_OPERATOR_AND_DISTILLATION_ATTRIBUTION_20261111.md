> **2026-11-13 接续：** 当前未来顺序、配额和判断以[统一 VBench 路线](../20261113/G20_VBENCH_RESTART_V4_20261113.md)、[判断台账](../20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)与[资产树](../20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)为准。保留双主线，B归因不以A候选阳性为前置；CAL11/DEV11先做表示与密度，CAL22按问题触发，O40单列有限预算。R1/R1b已完成，数值修复未消除旧DEV失败，失败机制仍待判别。下文保留11/11理论与文献身份；v3的大型配额、七臂全展开和旧Next不再调度未来工作。

# G20 双主线：新算子与 Wan／蒸馏归因的并行研究合同

**2026-11-11 · DESIGN_ONLY / LIT+ALG+ARCHIVED_OBS · REVIEW_REQUIRED · NO_NEW_RUN**

[G20 正反向综合](G20_ICML_FORWARD_REVERSE_SYNTHESIS_AND_EXPERIMENT_DECISIONS_20261111.md) · [G19 方法及实测](../20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) · [ICC 移植合同](../20261107/ffn_baseline_reproduction/ICC_FFN_PORT_SPEC.md) · [Chipmunk 历史实测](../20261107/ffn_baseline_reproduction/CHIPMUNK_REUSE_AND_COMPARISON.md) · [新文献 L99–L109](../literature/20261111/G20_FEW_STEP_AND_INCREMENTAL_ACCELERATION_PRIMARY_REVIEW_20261111.md)

## 0. 决策与不可混合的指标分母

**两条 P0 并行路线**：
- **A / 新算子**：从 Q1-r2 的实测成功与失败出发，设计能够控制重要状态相关非线性增量的廉价算子；在 FFN 以外优先拆解 Attention 聚合的真实增量与成本，最终要求相对强独立组合出现额外质量—wall-time 收益。
- **B / 方法差异归因**：在合法相同输入、gate、数值和成本定义下定位 Q1-r2 与实际 ICC/Chipmunk 的质量差异，分离 **Wan 架构 / 原始与蒸馏权重 / 采样间隔与状态分布 / high-low 专家切换 / 移植与 kernel 系统成本**。

**论文指标的三个分母**：
\[
S_{\mathrm{additional}}=\frac{T_{\mathrm{same\ distilled\ checkpoint,\ uncached}}}{T_{\mathrm{same\ distilled\ checkpoint,\ candidate}}},\qquad
S_{\mathrm{overall}}=\frac{T_{\mathrm{original\ native\ steps}}}{T_{\mathrm{distilled,\ candidate}}}.
\]
前者是本文新增的少步算法收益；后者包含蒸馏、CFG、采样、硬件和可能的其它组合收益，应逐组件拆分。另以**同样端到端实测预算**比较最近邻及 G19+强 Attention 方法的直接组合；主论文需要 Dense、最近邻、直接组合、完整消融，不仅一条相对 Dense 的速度比。

**外部证据锚点**：
- ICC 原论文核心是 DiT 35-step DDIM 的增量校正，不是已验证四步 Wan FFN-only。
- Chipmunk 作者 Wan 默认 YAML 将 MLP 关闭、Attention/step caching 打开；本仓 FFN-only P64 是特定 Wan 移植，不可拿作者整系统倍数充当 MLP-only 性能。
- LearniBridge 官方 Wan2.1-T2V-1.3B 为 \`sample_steps=50\`，其 4.10x 并非四步蒸馏的额外算法收益。
- DisCa (CVPR 2026) 已证明少步蒸馏与缓存适配是顶会题目；其部分表中各臂可能包含不同蒸馏/联合训练 recipe，不能无条件当作同一冻结四步权重的纯缓存消融。
- L²P (CVPR 2026) 在 Qwen-Image-Lightning 中也测少步，因此「首次在蒸馏模型加速」不成立。

## B · 解释为什么 Q1-r2、ICC、Chipmunk 在四步 Wan 上分化

### B0. 现有实测与新结果先完成身份对齐

- USER_REPORTED：用户已实现 ICC/Chipmunk 并报告 Q1-r2 优于它们。**不改写成仓库已验证的同源同配置优效**。截至先前归档，Chipmunk P64 有旧两暴露来源、full160 VIDEO/RAFT/ABBA；ICC 官方移植规格仍为 DESIGN_ONLY / NOT_RUN。先定位用户较新的 run/代码/manifest，再按实际状态更新证据；无文件就记 OPEN_EVIDENCE。
- 逐 arm 明确：模型高低专家 checkpoint SHA、样本 \`image/prompt/seed/noise\` SHA、实际 scheduler timestep、同样 FFN \`U\`（norm2+scale/shift 后、W1 前）、BF16 GELU tanh、gate、舍入与缓存寿命、原生 attention backend、offload、P64 掩码与 ICC rank/CA-CD-SVD、实际 FFN 漏重算位点。
- 保持作者忠实算法与 FFNDS 优化适配的版本差异：Chipmunk 的 proxy / mask / packed / 31668 token 尾块 52 行；ICC 保留 W1→中间 GELU→W2 的两级低秩增量与原作者缓存的真实费用；G19 单残差和 \`uA\`。所有 oracle 只作可表达性上界，不放主加速表。

### B1. 首先把三种相互独立的「冗余」分开

给定同专家锚点 \`s\` 与命中 \`t\` 的真实 FFN 张量，统一：\(\Delta U=U_t-U_s,\ \Delta H=H_t-H_s,\ \Delta Y=F(U_t)-F(U_s)\)。令 \(\mathcal G_t[E]=G_t\odot E\)。

\[
R_m(\mu)=\frac{\mathbb E_\mu\|\mathcal G_t[\widehat{\Delta Y}_m-\Delta Y]\|_F^2}
{\mathbb E_\mu\|\mathcal G_t[\Delta Y]\|_F^2+\epsilon}.
\]
同时报告绝对门控误差和 \(\|\mathcal G_t[\widehat Y-F(U_t)]\|^2/\|\mathcal G_t[F(U_t)]\|^2\)，防止 low expert 中小的 \(\Delta Y\) 分母造成误判。

**机制 1 · 稀疏更新性（Chipmunk）**：同付费预算的合法 K 子集与后验最优 K-subset 分列；理想误差来自 \(\Delta H_{\bar S}W_{2,\bar S}^\top\) 经当前 gate 读出。后验最优 \(\min_{|S|\le K}\) 在相同 output metric 下测，并检查 channel signed cross；前沿 mask 的 proxy / 选列 / gather / GEMM 时间全部计价。

**机制 2 · 低秩表达性（ICC）**：以真实 W1、GELU、W2 两级增量执行，扫 rank 和作者 SVD/CD-SVD；分离 first-linear 增量错位、\(\phi\) 曲率与低秩 W2 读出差异，缓存 4态/2态及分层 offload 的执行成本单列。不能将 ICC 直接折叠成 G19 全 A 后宣称复现。

**机制 3 · 完整函数增量可预测性（G19）**：\(\widehat{\Delta Y}_{G19}=\Delta U\,A\)。Q0/Q1-r1/Q1-r2 与全自由 ridge A、同成本普通 SVD/输入加权 SVD 比较。与当前 gate 一起查看误差方向，而不只看隐藏 MSE。

真实 \(\Delta H\) 可以稠密而 \(\Delta Y\) 容易线性预测；低 rank 不能自动由单来源小样本插值得出。这三种冗余互不蕴涵，结果需在 heldout source 与学生自由轨迹成立。

### B2. 「Wan 还是蒸馏」必须分解为至少三个因素

令 \(\mathcal E_m(W,\mu,G)\) 是给定 **FFN 权重 W**、**输入对分布 \(\mu\)**、**门控度量 G** 和匹配算法 m 的来源外误差。对同一种 Wan2.2 架构，可做**函数回放 2×2 交叉诊断**：

| | 原权重 W_O | 蒸馏权重 W_D |
| --- | --- | --- |
| 原模型实际轨迹 U_O | \(\mathcal E_m(W_O,\mu_O,G_{\rm ref})\) | \(\mathcal E_m(W_D,\mu_O,G_{\rm ref})\) |
| 蒸馏模型实际轨迹 U_D | \(\mathcal E_m(W_O,\mu_D,G_{\rm ref})\) | \(\mathcal E_m(W_D,\mu_D,G_{\rm ref})\) |

- **同一 U 比权重**：在真实 pre-W1 界面截取 U，分别喂两个同维度 expert 的 FFN；为这一反事实比较固定同一 G_ref（可分别用两组原 gate 复查）。允许为每个 W **使用相同来源的独立 CAL 重新拟合**，不能给 Q1 特权或混合 weight-identity。此项是离线函数诊断，交叉权重输出不能称其完整学生真实轨迹。
- **同一 W 比状态分布**：比较 \(\mu_O\) 与 \(\mu_D\) 的位移幅值、方向、GELU 中点、割线斜率离散、实际 timestep 与专家分组；在固定分箱中匹配 \(\|\Delta U\|\)、actual diffusion timestep / time gap，避免把所有大步长效应归于权重蒸馏。
- **同一 W 比相邻间隔**：从原始 40-step 轨迹上取近邻调用对与长跨度配对作为离线诊断；不要把原权重强行四步生成的失败质量当正常部署 baseline。蒸馏四步的 timestep 需读取真实 scheduler（不能拿 nominal step list），high/low 只在本专家内配对。
- **架构因素**：仅在第二个非 Wan、同时有原始与少步蒸馏对应模型且可忠实移植时检验；仅有同架构 Wan2.2 原/蒸馏 2×2 无法判定「Wan 独有」，只能判在 Wan 内权重和状态效应。非 Wan 的 FFN 函数/激活/门控形态必须作新的方法绑定；两模型比较不等于统计上证明某一架构因果。

候选假设及预定不同预测：
- H_W：相同 \(\mu,G\) 下不同权重改变 Chipmunk/ICC 与 G19 的相对风险，源外可重复 → **权重函数差异**，未必纯粹由蒸馏单独因果造成。
- H_Δ：同一权重下状态配对间隔增大，稀疏性/低秩性退化而线性可预测性较稳 → **采样输入分布机制**。
- H_I：相同理想后验表达误差，但真实方法退化来自 mask proxy、低 rank、cache、BF16 或 runtime → **实现/预算机制**。
- H_T：同态局部排序与完整视频相反，受终端传播和锚点反馈控制 → **功能传播机制**。
- H_A：只有 Wan 的某个 expert / head / layer 发生特异模式，第二架构没有 → **Wan 条件性**；需跨架构匹配，避免未经验证的普适结论。

### B3. 最短可辨识证据链

**S0（静态）**：矩阵/模型/实验身份/原作者代码差异、缓存与 rank/mask 预算对齐；复用既有 Chipmunk P64、G19 Q1 视频和真实 \(\Delta U,\Delta H,\Delta Y\) 资产。ICC 新成果优先定位凭证。
**S1（新实验需独立评议）**：少量匹配状态上重放原生 FFN 和三种方法，完成 B1 的输出误差分解 + B2 2×2 反事实；先预注册对照量和分组。
**S2（新实验需独立评议）**：相同冻结图文源、seed 与模型 checkpoint 的完整学生视频，分 high-only/low-only/both，质量与效率分别证明。
**S3**：再用不同架构/蒸馏 recipe 和新独立 source 检验外推；若没有第二架构，将「Wan 问题还是蒸馏问题」保持部分未识别。

统计单元是独立 image/prompt 来源；seed 为嵌套配对重复，frame/token/layer 非独立样本。先在旧DEV发现方向，独立确认前冻结阈值与算法。

## A · G20 新算子设计（用 B 的结论选择表示）

### A1. 延续已证明的计算基础

G19 对一行 pre-W1 输入 \(u\) 使用
\[
F(u)=\phi(uW_1^\top+b_1^\top)W_2^\top+b_2^\top,\qquad
\widehat F(u_t)=F(u_s)+(u_t-u_s)A.
\]
\(A=W_1^\top\operatorname{diag}(\alpha)W_2^\top\)，真实有限增量为 \(\Delta u\,\overline J_F\)。损失为 \(\Delta u(A-\overline J_F)\)。保留高维全 A 作为强参考：全 A 的降秩和曲率低维是两项不同假设。

### A2. 状态相关的低成本残差函数（非线性核心候选）

\[
B_\theta(u)=uA+g_\theta(uP)Q^\top,\qquad
\widehat F(u_t)=F(u_s)+B_\theta(u_t)-B_\theta(u_s).
\]
参数共享、锚点精确、端点差反对称；昂贵函数仍在合法 anchor 运行，命中计算 \(B\) 的两个端点投影，可在 anchor 缓存必要 \(u_sP\)；不引入按 prompt 的 d×d 矩阵库。用额外参数 \(r,k\)、真实 GEMM 及缓存成本计价；若低秩非线性核在源外不优于 full A、且无法抵消新增运行时间，则放弃在线分支。

**判别对象**：保持 \(\delta=u_t-u_s\) 时移动中点 \(m=(u_s+u_t)/2\)，测
\[
D_m\big(F(m+\delta/2)-F(m-\delta/2)\big)[a]
=a(J_F(u_t)-J_F(u_s)).
\]
G19 固定 \(\delta A\) 对中点导数为零；非零实际中点响应是有潜力的遗漏信息。测 Hessian 的 \(D^2F[a,\delta]\) 收缩及当前门控、W2读出、终端功能敏感度；完整学生 BF16 有限幅干预为最终 oracle。优先测试小型 \(g\) 与原生回退对照，不能仅凭函数形式宣称创新。

### A3. 信息下界决定是否买在线统计

合法廉价信息 \(Z\) 满足首次可用时点和费用 \(C(Z)\)；平方误差理论下界
\[
R(Z)=\mathbb E\,{\rm tr}\,\operatorname{Cov}(Y\mid Z),\quad
R(Z)-R(Z,S)=\mathbb E\|\mathbb E[Y\mid Z,S]-\mathbb E[Y\mid Z]\|^2.
\]
这是已有条件期望正交理论；研究意义来自受成本限制的可得 \(Z\) 和真实终端质量。量化符号、幅值、gate、浅层读出/时间调制只在能提前廉价观察时用；不可先完整运行 W1/GELU 再称免费代理。

### A4. 跨 FFN-only 天花板：先注意力聚合，再考虑 Block

先保留 QKV/RoPE / 输出线性、FFN、norm、sampler 等的原生边界，分析注意力真实增量
\[
\Delta O=\overline P\,\Delta V+\Delta P\,\overline V,\qquad
O=PV.
\]
分析读写耦合、低成本可观测 \(Z\) 和残差方向的增长；**RECAP 已包含当前稀疏 Attention + 缓存 coarse/detail 校正 + 反馈**，不能以直接拼接首创。若要说模型级新算子，必须在独立视频上优于：
1. 4-step distilled Dense；
2. G19 Q1-r2；
3. 强 Attention 加速单独；
4. **G19 + 强 Attention 加速的直接组合**；
5. 新 A 方法/与 Attention 联合（同 wall-time / 画质预算）。

若 5 只在局部 MAC 改善而全系统无加速，保留为机制研究，不做系统加速主张。

### A5. 先后条件与停止条件

- B1 显示固定 A 的主要源外剩余误差由读出/curvature 驱动 → 测 A2；若源外曲率有效谱平坦/状态条件弱，保留 full A 与少量原生回退。
- B1/B2 显示某些 ICC/Chipmunk 方法在低 rank 或 mask 付费处失败 → 优先展示 G19 的优势来源，避免把失败解释成蒸馏必然性质。
- G20 E1 的 JVP/有限幅后缀未能预测实际视频变化 → 先做真实学生重放/信赖域，不求解超大冻结终端 Hessian。
- Attention 新算子不优于直接组合 → 不宣称跨模块新机制，保留 G19 FFN 结果与机制证据。
- 新源、GPU 或完整视频实验都需独立研究评议、recheck、ACTIVE 冻结 scope、原生 RUN/EXIT 与质量验收；本文**未批准/未启动**任何此类作业。

## 两条线的最终科学交付

**路线 B 的成果**不是一张简单的名次表，而是说明三种计算冗余在少步采样、蒸馏权重和高低专家状态中如何分化，并形成对既有 ICC/Chipmunk/G19 局部误差和视频结果的有区分力解释。

**路线 A 的成果**必须是有在线成本优势的新算子与扩展计算单元：以有限输入状态函数增量为数学对象，在真实视频质量约束下跨过 FFN-only 的上限。

**现状**：只登记新研究合同，不改旧方法/旧数据、不覆盖 E0–E4 和 ACTIVE，不把尚无收据的优效说成已核验事实。
