# G20 R1b 数值复核 · 实际执行结果（2026-11-12）

**证据身份：NATIVE/OBS（A100 真实 BF16 原生 FFN 与回放统计）；无 VIDEO/SYS 主张。** 本文报告对已发布 R1 六站点昂贵拟合的**纯数值复核**（R1b）实际执行结果：用认证精确的稳定 F64 解重解**同一注册目标**，并逐字重算原 FP32 解做 SHA 身份门，以判别 R1 发表的 DEV5 退化究竟是 FP32 数值伪影（H_artifact）还是 ~500 行严重欠定 CAL 的真实 ridge 过拟合（H_overfit）。0 次新生成/视频/基准；DEV5 仍为**已暴露开发来源外，非盲测**。完整机器可读证据、冻结源码与运行收据见 [reports/20261112/g20_r1b_recheck](../../reports/20261112/g20_r1b_recheck/README.md)。

## 1. 执行身份与治理

- **Scope**：`G20_R1B_NUMERICAL_RECHECK_20261112`，SHA256 `e03d1748ef988181666a5350b40053d0eb73f0605fadf22a70c00aba8a783561`；基础为已发布 R1（commit `0b37a7ac`，scope `4bbf8496…`，job 156476 EXIT 0）及用户递交的独立审查（8 项发现，作者全部静态确认）。
- **治理链**：作者自审 → 真实独立 reviewer（`qoder_independent_reviewer_g20_r1b_20261112`）首轮 **REQUEST_CHANGES**（1 阻塞：selfcheck 病态测试用 FP32 生成 X，其数值秩 12 ≠ 数学秩 6，`D_numerical_rank<=6` 断言会在任何站点前误触发 abort）→ 作者按 reviewer 首选方案 (a) 修订（F64 构造测试 X，原 FP32 路径喂 FP32 舍入副本）+ 恢复 checkpoint SHA 守卫 + 将 omega 基对齐 R1 精确 Gram 判据 → reviewer **独立 Node F64 复算**（8 种子 D 秩均 6，probe 1.5–3.9e-9，cross 1.7–3.6e-8，oosFull ~1e-14，原 FP32 泄漏 0.49–0.70）→ recheck **APPROVE/APPROVE** → 阶段级 ACTIVE 冻结（1 job/1 GPU，会话上限 2）→ 计算节点 preflight + 发射前逐文件 SHA 复核 → 单次提交。旧指纹 `c7ffb052…` 未复用。记录见 [evidence/INDEPENDENT_REVIEW.json](../../reports/20261112/g20_r1b_recheck/evidence/INDEPENDENT_REVIEW.json)、[AUTHOR_REVIEW_RESPONSES.json](../../reports/20261112/g20_r1b_recheck/evidence/AUTHOR_REVIEW_RESPONSES.json)。
- **实际运行**：Slurm job **157314**，anode16（P107-A100，A100-SXM4-80GB×1，4 CPU，64 GiB），2026-11-12 13:58:38 启动，**EXIT.json exit_code 0**、SHELL_EXIT 0，耗时 **2097.9 s**，峰值 RSS 1.59 GB，stderr 空。[RUN.json](../../reports/20261112/g20_r1b_recheck/evidence/RUN.json) 记录 scontrol 实时 RUNNING 分配核验；[EXIT.json](../../reports/20261112/g20_r1b_recheck/evidence/EXIT.json) 绑定 SUMMARY SHA `ef35bcd695578cee0343165daacb02b03e026969d6d80856395f55eef40dea9a`。
- **自检（运行内，GPU，阻塞修复后全部通过）**：病态测试 ρ=1.065e-7∈(1e-8,1e-6)；稳定解 vs 原始 F64 逐列参考 probe **3.89e-9**≤1e-7；SVD vs F64 eigh cross **5.49e-9**≤1e-7；**stable_D_rank=6**（修复生效）；全样本空间 oos **1.32e-14**≤1e-12；原 FP32 路径泄漏 **0.9797**>1e-4（判别力保留，O(1)）；R1 连续性（良态/秩亏 eigen-vs-dense、ICC 零因子/分解恒等、reuse-Y、gate_fro2、零矩阵回放）全部通过。发射前 reviewer 已用纯 Node F64 独立复算同一电池（8 种子），运行内 emitted dict 与之一致。

## 2. 判别规则（预登记，执行前冻结）

- **H_artifact** 成立当且仅当：稳定 main 解 DEV5 配对变化在 ≥4/6 站点回到 Q1 的 1% 以内（或更好），**且**原 D 在 R1 判据 CAL 子空间外能量 ≥10%（rel Fro）于同站点。
- **H_overfit** 成立当且仅当：认证稳定 main 解仍在 ≥4/6 站点使 DEV5 math-lift MSE 相对 Q1 退化 >1%（证书通过），且原 D 子空间外能量相对其总量小（即子空间内部分单独负责）。
- 否则 **mixed**：逐站点报数与份额，不做二元裁决。"substantial"=|source 等权配对 mean_rel_change|>1%（仅报告阈值，不选择任何东西）；两 λ 两解算器全报；身份门失配则 warn+continue 并视身份比较为可疑。

## 3. 身份门与证书（全部通过）

- **身份门**：六站点重算 BF16(D_orig_main) SHA **全部等于**冻结 R1 SHA（`all_identity_sha_match=true`）；逐源 SSE 最大相对差 **0.0**（逐位复现）；重算 D 数值秩与 R1 发表值**逐一相等**（main 4239/4515/4389/4711/4732/4683，robust 2834/4091/3958/3769/4405/4296）。原 FP32 路径被忠实重算，R1 发表数字可复现。
- **稳定解证书**：D 数值秩**恰为 CAL 行数 517（high）/523（low）**，两 λ 皆然——精确满足 exact bound rank(D)≤rank(X_CAL)，R1 发表的 4239–4732 确为违反；F64 正规方程相对残差 **8e-15–2.5e-14**（机器精度）；SVD vs F64 eigh cross **6e-12–1.7e-11**≤1e-7；D 在 R1 判据 CAL 子空间外能量 **~7e-14**（构造上在行空间内）；全样本空间 oos ≤1e-10 运行时强制通过。
- **omega 可比性**：`omega_stable` 与 R1 参考 omega **逐源逐位一致**（基判据已对齐 R1 精确 Gram 判据 mu=sv²>1e-6·max(mu)，且六站点 gram_numerical_rank==N_CAL，两基重合）；范围 0.363–0.689 不变。

## 4. 核心裁决：**H_overfit 成立（6/6 站点），H_artifact 被否定**

DEV5 相对 Q1 的配对 MSE 变化（math lift；mean of per-source relative changes）：

| 站点 | 原 FP32 main | 稳定 F64 main | 稳定 F64 robust(λ=1e-2) | ICC r512 | 原 D 子空间外能量 | 稳定 D 秩 |
| --- | --- | --- | --- | --- | --- | --- |
| high/4 | +28.05% | **+27.22%** | +23.87% | +1.71% | 0.86% | 517 |
| high/19 | +15.60% | **+15.56%** | +14.91% | +26.97% | 0.42% | 517 |
| high/34 | +15.08% | **+15.08%** | +14.67% | +41.72% | 0.53% | 517 |
| low/4 | +28.75% | **+28.45%** | +26.97% | +8.18% | 0.70% | 523 |
| low/19 | +19.28% | **+19.30%** | +18.89% | +71.73% | 0.75% | 523 |
| low/34 | +17.05% | **+17.03%** | +16.80% | +75.33% | 0.78% | 523 |

- **认证精确解仍在全部 6 站点退化 +15%–28%**（>1%），证书全过 → H_overfit 预登记判据满足（≥4/6）。
- **原 D 子空间外能量仅 0.42%–0.86%**（远低于 H_artifact 要求的 ≥10%）→ H_artifact 的能量判据不满足；且稳定 main 未在任何站点回到 Q1 的 1% 以内 → H_artifact 双重否定。
- **FP32 伪影解释的退化份额 ≤0.83 个百分点**（orig−stable_main 差：0.83/0.04/−0.00/0.31/−0.02/0.01 pp），即 ≤3% 的 +15–29% 效应；其余 ≥97% 是 CAL 行空间内的真实过拟合。
- **CAL4 样本内**：原与稳定解配对变化均 ≈ **−99.98%…−99.9996%**（MSE 降至 1e-6–1e-8）——样本内近乎精确 + 来源外一致恶化，是过拟合的经典 signature。
- **robust λ=1e-2 不挽救**：稳定 robust 仍退化 +14.7%–27.0%，仅边际改善；过拟合非正则化刀锋伪影。
- **math vs BF16 replay 分离**：稳定 main 的 replay 与 math 配对差 ≤0.001 相对（如 high/4 math +27.22% / replay +27.13%）——退化在 FP32 数学 lift 与部署 BF16 回放**同时出现**，非亚舍入噪声拟合。
- **D 能量守恒佐证**：稳定与原 D_fro2 相对差 ~7e-5–4e-5，与子空间外能量 0.4–0.9%（Fro 范数比）平方一致——FP32 噪声以**大量微小奇异值**抬高数值秩（4239–4732），但携带能量 <1%，故秩违反是真伪影，却**不是** DEV 退化的主因。

## 5. ICC 两残差 F64 分解（R1 缺失的归因）

DEV5 平均份额（total_f64 = stage1 + stage2 + cross，恒等残差 ≤1.9e-16）：

| 站点 | stage1 传播 | stage2 原生 h | cross |
| --- | --- | --- | --- |
| high/4 | 12.3% | **80.2%** | +7.5% |
| high/19 | 28.8% | **71.4%** | −0.2% |
| high/34 | 38.7% | **64.3%** | −3.0% |
| low/4 | 15.7% | **82.5%** | +1.8% |
| low/19 | 24.7% | **74.7%** | +0.6% |
| low/34 | 27.8% | **72.8%** | −0.6% |

冻结 svd_r512 配置的 ICC 门后误差**主要来自 stage-2（原生 hidden 处的第二残差，64%–82%）**，而非传播的 stage-1；cross 项小（−3%…+7.5%）。BF16 pred 舍入相对量 9e-6–9e-5；零增量控制 max-abs 0.016–0.125（BF16 级，anchor/hit 舍入不可结合），零因子自检精确。此为**该冻结配置**的描述性归因，不裁决其他秩/形态，不构成新 ICC 主张。

## 6. 这些数据说明什么

**（1）R1 的定性结论在更正证书后存活。** 自由 ridge 臂 A=A_Q1+D 在来源外一致恶化，失败方向是**真实过拟合**：~500 行严重欠定 CAL 对 5120×5120 D，即便精确求解，CAL 行空间内的修正仍拟合 CAL 特有残差结构并部分转移到 DEV5（omega 0.36–0.69 表明 DEV 增量 36%–69% 在 CAL 行空间外，fit 只能作用于子空间内部分）。这不是 FP32 噪声伪影，也不是 Q1 基底限制的证据。

**（2）秩违反 anomaly 被 resolves 为数值伪影，但非 DEV 退化主因。** R1 发表 D 秩 4239–4732 违反 exact bound ≤517/523 确由 FP32 Gram/RHS 形成的零空间噪声（~1e-7 相对）被 g_j²/ρ~1e6–1e7 放大所致；稳定解秩恰为 517/523。然而该噪声仅携带 D 能量的 0.4–0.9%，解释 ≤0.83 pp 退化。**"秩违反"与"DEV 退化"是两个可分离的事实**：前者是伪影，后者是真实过拟合。R1 §5(1) 的覆盖/过拟合解释由此获得精确解支撑。

**（3）判别动作不变。** 覆盖不足解释未被排除（竞争解释：真实完整线性算子存在但 CAL4 无法识别）；判别仍是扩大 CAL 来源/行数后重拟合，观察 omega 是否下降、DEV 配对差是否转负。在覆盖扩大前，把来源外失败归因于 Q1 基底限制或真实状态依赖，都超出本批证据。R2 盲/共享捕获与 R3 full-M 写回仍推迟。

**（4）reviewer 的科学反例保留在案。** chip10 同质队列下 CAL4/DEV5 可能共享低维主导 DeltaU 行子空间；子空间内过拟合可转移并降低小 M 配对门后 MSE，而真实 full-M 算子未必优于 Q1。本批 omega 高（0.36–0.69）表明 DEV 增量大幅在 CAL 行空间外，任何变化主要是子空间插值；结论须保持**开发筛查归因**，绝不升级为"目标泛化"或"基底限制被证明"。

## 7. R1 文档版本化更正（本批后执行）

依 scope 预登记，R1 结果文档三处错误已作版本化更正（冻结 R1 发布快照不改）：seed 43→42（两处）；Q1 数学 lift vs BF16 回放"全部 80 站点 ~1e-4"过度陈述→实测逐源中位 0.0250%、最大 1.1103%，站点组 source 等权 MSE 最大差 0.6655%（cheap_low_16 DEV5）；ICC high/4 "abs −9.4e-6 在 epsilon_num=1e-6 尺度无收益"→3/5 DEV 源改善、2/5 恶化，source 等权 MSE 变化 −0.0849% vs 逐源相对均值 +1.7064%，epsilon_num 是相对数值容差不可与绝对逐元素 MSE 变化比较。详见 R1 结果文档顶部更正注记。

## 8. 边界（主张级）

- small-M（每源 108–146 匹配行）门后代理，**不含 full-M 残差写回**；不能外推为全层/全 token 效果。
- DEV5 已暴露开发来源，非盲确认；CAL4 已用于原 Q1 拟合；单 seed 42；6 昂贵站点为预登记分层样本。
- ~500 行对 5120×5120 D 严重欠定；rank≤517/523 是界不是识别；omega 界定可解释 DEV 份额；1% 内结果偏好伪影解释但**不证明泛化**（本批结果为 H_overfit，无此歧义）。
- 无视频、无质量、无速度主张；无景观重跑、无触发、无新生成/基准；chipmunk 臂仍 MISSING_ASSET（R2）。
- 24 张 BF16 矩阵（各 52,428,928 字节，~1.2 GiB）保留集群原位置，未上传；SHA256 见 [NPY_SHA256.txt](../../reports/20261112/g20_r1b_recheck/evidence/NPY_SHA256.txt)。

## 9. 证据索引

- [SUMMARY.json](../../reports/20261112/g20_r1b_recheck/output/SUMMARY.json)（SHA 与 EXIT 绑定）· [output/](../../reports/20261112/g20_r1b_recheck/output) 6 站点逐源收据（含 omega_stable/omega_r1_reference、ICC 分解、10 臂 SSE、身份比较、矩阵 SHA）
- [scope.json](../../reports/20261112/g20_r1b_recheck/evidence/scope.json) · [独立审查](../../reports/20261112/g20_r1b_recheck/evidence/INDEPENDENT_REVIEW.json) · [作者回应](../../reports/20261112/g20_r1b_recheck/evidence/AUTHOR_REVIEW_RESPONSES.json) · [自审](../../reports/20261112/g20_r1b_recheck/evidence/ROOT_SELF_AUDIT.txt) · [ACTIVE](../../reports/20261112/g20_r1b_recheck/evidence/ACTIVE_RESEARCH.json)
- [RUN](../../reports/20261112/g20_r1b_recheck/evidence/RUN.json)/[EXIT](../../reports/20261112/g20_r1b_recheck/evidence/EXIT.json)/[SHELL_EXIT](../../reports/20261112/g20_r1b_recheck/evidence/SHELL_EXIT.json) · [DISPATCH_INTENT](../../reports/20261112/g20_r1b_recheck/evidence/DISPATCH_INTENT.json)/[SUBMITTED](../../reports/20261112/g20_r1b_recheck/evidence/SUBMITTED.json) · [NPY_SHA256](../../reports/20261112/g20_r1b_recheck/evidence/NPY_SHA256.txt)

本次发布不启动新作业；后续范围（R2 盲/共享捕获、R3 full-M 写回）按路线另行独立复审与 ACTIVE 冻结。