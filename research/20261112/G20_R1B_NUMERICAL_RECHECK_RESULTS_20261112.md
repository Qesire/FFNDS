> **2026-11-13 解释修订：** 更正相对Frobenius范数/平方能量术语，收回≥97%的机制因果份额及以omega直接解释退化的推断。既有数值、RUN/EXIT、冻结scope与reports原件不改；当前判断见[台账](../20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)。[修订前正文](https://github.com/Qesire/FFNDS/blob/84630899eb041ddd6d84a759a76b04710e974ed7/research/20261112/G20_R1B_NUMERICAL_RECHECK_RESULTS_20261112.md)保留可追溯版本。

# G20 R1b 数值复核 · 实际执行结果（2026-11-12）

**证据身份：NATIVE/OBS（A100 真实 BF16 原生 FFN 与回放统计）；无 VIDEO/SYS 主张。** 本文报告对已发布 R1 六站点昂贵拟合的**纯数值复核**（R1b）实际执行结果：用认证精确的稳定 F64 解重解**同一注册目标**，并逐字重算原 FP32 解做 SHA 身份门，以检验 R1 发表的 DEV5 退化能否由稳定求解消除，并按当时冻结的 H_artifact/H_overfit 操作性规则报告结果。该规则能区分数值修复是否消除负结果，不能进一步识别来源覆盖、估计器结构或状态信息的因果份额。0 次新生成/视频/基准；DEV5 仍为**已暴露开发来源外，非盲测**。完整机器可读证据、冻结源码与运行收据见 [reports/20261112/g20_r1b_recheck](../../reports/20261112/g20_r1b_recheck/README.md)。

## 1. 执行身份与治理

- **Scope**：`G20_R1B_NUMERICAL_RECHECK_20261112`，SHA256 `e03d1748ef988181666a5350b40053d0eb73f0605fadf22a70c00aba8a783561`；基础为已发布 R1（commit `0b37a7ac`，scope `4bbf8496…`，job 156476 EXIT 0）及用户递交的独立审查（8 项发现，作者全部静态确认）。
- **治理链**：作者自审 → 真实独立 reviewer（`qoder_independent_reviewer_g20_r1b_20261112`）首轮 **REQUEST_CHANGES**（1 阻塞：selfcheck 病态测试用 FP32 生成 X，其数值秩 12 ≠ 数学秩 6，`D_numerical_rank<=6` 断言会在任何站点前误触发 abort）→ 作者按 reviewer 首选方案 (a) 修订（F64 构造测试 X，原 FP32 路径喂 FP32 舍入副本）+ 恢复 checkpoint SHA 守卫 + 将 omega 基对齐 R1 精确 Gram 判据 → reviewer **独立 Node F64 复算**（8 种子 D 秩均 6，probe 1.5–3.9e-9，cross 1.7–3.6e-8，oosFull ~1e-14，原 FP32 泄漏 0.49–0.70）→ recheck **APPROVE/APPROVE** → 阶段级 ACTIVE 冻结（1 job/1 GPU，会话上限 2）→ 计算节点 preflight + 发射前逐文件 SHA 复核 → 单次提交。旧指纹 `c7ffb052…` 未复用。记录见 [evidence/INDEPENDENT_REVIEW.json](../../reports/20261112/g20_r1b_recheck/evidence/INDEPENDENT_REVIEW.json)、[AUTHOR_REVIEW_RESPONSES.json](../../reports/20261112/g20_r1b_recheck/evidence/AUTHOR_REVIEW_RESPONSES.json)。
- **实际运行**：Slurm job **157314**，anode16（P107-A100，A100-SXM4-80GB×1，4 CPU，64 GiB），2026-11-12 13:58:38 启动，**EXIT.json exit_code 0**、SHELL_EXIT 0，耗时 **2097.9 s**，峰值 RSS 1.59 GB，stderr 空。[RUN.json](../../reports/20261112/g20_r1b_recheck/evidence/RUN.json) 记录 scontrol 实时 RUNNING 分配核验；[EXIT.json](../../reports/20261112/g20_r1b_recheck/evidence/EXIT.json) 绑定 SUMMARY SHA `ef35bcd695578cee0343165daacb02b03e026969d6d80856395f55eef40dea9a`。
- **自检（运行内，GPU，阻塞修复后全部通过）**：病态测试 ρ=1.065e-7∈(1e-8,1e-6)；稳定解 vs 原始 F64 逐列参考 probe **3.89e-9**≤1e-7；SVD vs F64 eigh cross **5.49e-9**≤1e-7；**stable_D_rank=6**（修复生效）；全样本空间 oos **1.32e-14**≤1e-12；原 FP32 路径泄漏 **0.9797**>1e-4（判别力保留，O(1)）；R1 连续性（良态/秩亏 eigen-vs-dense、ICC 零因子/分解恒等、reuse-Y、gate_fro2、零矩阵回放）全部通过。发射前 reviewer 已用纯 Node F64 独立复算同一电池（8 种子），运行内 emitted dict 与之一致。

## 2. 判别规则（预登记，执行前冻结；历史标签不扩成机制因果结论）

以下保留原scope的阈值与裁决身份。原文把 rel Fro 称为“能量”，实际字段是相对 Frobenius 范数，本版只更正术语；冻结源码、规则文件与数值均不改。H_overfit 在本实验内的操作性含义是“稳定解仍存在CAL拟合/DEV恶化”，不意味着已识别其具体机制。

- **H_artifact** 成立当且仅当：稳定 main 解 DEV5 配对变化在 ≥4/6 站点回到 Q1 的 1% 以内（或更好），**且**原 D 在 R1 判据 CAL 子空间外相对 Frobenius 范数 ≥10%（rel Fro）于同站点。
- **H_overfit** 成立当且仅当：认证稳定 main 解仍在 ≥4/6 站点使 DEV5 math-lift MSE 相对 Q1 退化 >1%（证书通过），且原 D 子空间外相对 Frobenius 范数相对其总量小（依原scope的操作性标签判定；并非进一步的机制分解）。
- 否则 **mixed**：逐站点报数与份额，不做二元裁决。"substantial"=|source 等权配对 mean_rel_change|>1%（仅报告阈值，不选择任何东西）；“两λ两解算器全报”为原计划表述。实际输出：原FP32 main及stable main/robust有CAL/DEV风险评分；原FP32 robust仅有解/秩/子空间诊断，没有其风险表。身份门失配则按原scope warn+continue并视比较为可疑；本批身份门全部通过。

## 3. 身份门与证书（全部通过）

- **身份门**：六站点重算 BF16(D_orig_main) SHA **全部等于**冻结 R1 SHA（`all_identity_sha_match=true`）；逐源 SSE 最大相对差 **0.0**（逐位复现）；重算 D 数值秩与 R1 发表值**逐一相等**（main 4239/4515/4389/4711/4732/4683，robust 2834/4091/3958/3769/4405/4296）。原 FP32 路径被忠实重算，R1 发表数字可复现。
- **稳定解证书**：D 数值秩**恰为 CAL 行数 517（high）/523（low）**，两 λ 皆然——精确满足 exact bound rank(D)≤rank(X_CAL)，R1 发表的 4239–4732 确为违反；F64 正规方程相对残差 **8e-15–2.5e-14**（机器精度）；SVD vs F64 eigh cross **6e-12–1.7e-11**≤1e-7；D 在 R1 判据 CAL 子空间外相对 Frobenius 范数 **~7e-14**（构造上在行空间内）；全样本空间 oos ≤1e-10 运行时强制通过。
- **omega 可比性**：`omega_stable` 与 R1 参考在浮点精度内接近，**不是逐位一致**：六站点×DEV5共30项，0/30逐位相等，最大绝对差3.4488828404288796e-8（high/4、chip10_01）。基判据对齐R1的mu=sv²>1e-6·max(mu)，六站点gram_numerical_rank==N_CAL；稳定omega范围0.3631738998428839–0.6889323017288771。

## 4. 核心裁决：**冻结 H_overfit 判据满足（6/6），稳定解未消除开发负结果**

DEV5 相对 Q1 的配对 MSE 变化（math lift；mean of per-source relative changes）：

| 站点 | 原 FP32 main | 稳定 F64 main | 稳定 F64 robust(λ=1e-2) | ICC r512 | 原 D 子空间外相对 Frobenius 范数 | 稳定 D 秩 |
| --- | --- | --- | --- | --- | --- | --- |
| high/4 | +28.05% | **+27.22%** | +23.87% | +1.71% | 0.86% | 517 |
| high/19 | +15.60% | **+15.56%** | +14.91% | +26.97% | 0.42% | 517 |
| high/34 | +15.08% | **+15.08%** | +14.67% | +41.72% | 0.53% | 517 |
| low/4 | +28.75% | **+28.45%** | +26.97% | +8.18% | 0.70% | 523 |
| low/19 | +19.28% | **+19.30%** | +18.89% | +71.73% | 0.75% | 523 |
| low/34 | +17.05% | **+17.03%** | +16.80% | +75.33% | 0.78% | 523 |

- **认证精确解仍在全部 6 站点退化 +15%–28%**（>1%），证书全过 → H_overfit 预登记判据满足（≥4/6）。
- **原 D 子空间外相对 Frobenius 范数仅 0.42%–0.86%**（远低于 H_artifact 要求的 ≥10%）→ H_artifact 的该预登记范数判据不满足；且稳定 main 未在任何站点回到 Q1 的 1% 以内 → H_artifact 双重否定。
- **原解到稳定解的最大净变化约0.8273个百分点**（orig−stable_main：0.8273/0.0364/−0.0044/0.3050/−0.0197/0.0127 pp）。这说明此次数值修复未消除主要DEV退化；删除旧文“其余≥97%是某机制的因果份额”的推断，方向、幅度、来源、正则结构和状态仍待区分。
- **CAL4 样本内**：原与稳定解配对变化均 ≈ **−99.98%…−99.9996%**（MSE 降至 1e-6–1e-8）——样本内近乎精确 + 来源外一致恶化，是过拟合的经典 signature。
- **robust λ=1e-2 不挽救**：稳定 robust 仍退化 +14.7%–27.0%，仅边际改善；过拟合非正则化刀锋伪影。
- **math vs BF16 replay 分离**：稳定 main 的 replay 与 math 配对差 ≤0.001 相对（如 high/4 math +27.22% / replay +27.13%）——退化在 FP32 数学 lift 与部署 BF16 回放**同时出现**，非亚舍入噪声拟合。
- **范数与能量分开**：原D行空间外相对Frobenius范数为0.4236%–0.8627%；相应平方能量比例为0.001794%–0.007442%。秩伪影与DEV失败分别由秩证书和稳定风险表支持；不能把范数比例当能量或把参数分量直接当风险因果份额。

## 5. ICC 两残差 F64 分解（R1 缺失的归因）

DEV5 平均份额（total_f64 = stage1 + stage2 + cross，相对恒等残差最大3.082303275473884e-16（DEV30项，low/19、chip10_05；CAL+DEV54项的最大值相同））：

| 站点 | stage1 传播 | stage2 原生 h | cross |
| --- | --- | --- | --- |
| high/4 | 12.3% | **80.2%** | +7.5% |
| high/19 | 28.8% | **71.4%** | −0.2% |
| high/34 | 38.7% | **64.3%** | −3.0% |
| low/4 | 15.7% | **82.5%** | +1.8% |
| low/19 | 24.7% | **74.7%** | +0.6% |
| low/34 | 27.8% | **72.8%** | −0.6% |

冻结 svd_r512 配置的 ICC 门后误差**主要来自 stage-2（原生 hidden 处的第二残差，64%–82%）**，而非传播的 stage-1；cross 项小（−3%…+7.5%）。字段`bf16_pred_rounding_rel`在DEV30项的范围为1.7090858331851085e-6–6.259122802036202e-4，定义为`abs(icc_gated_sse−total_f64_gated_sse)/total_f64_gated_sse`，是SSE变化比例，不是参数范数或prediction L2比例；零增量控制 max-abs 0.016–0.125（BF16 级，anchor/hit 舍入不可结合），零因子自检精确。此为**该冻结配置**的描述性归因，不裁决其他秩/形态，不构成新 ICC 主张。

## 6. 这些数据说明什么（2026-11-13 解释修订）

**当前估计器的开发负结果被稳定求解保留。** 旧CAL4约500行、DEV5、六站点与已测正则下，CAL近插值、DEV恶化确实存在；当前证据不足以区分修正方向错误、幅度过大、来源广度、行密度、受限结构与端点状态信息。FP32数值伪影和BF16舍入没有解释掉这次负结果，也不由此证明全部线性方法失败。

**omega 说明不可改变的方向。** 对精确各向同性ridge，`D=P_CAL D`，所以`X_DEV,perp D=0`。行空间外的DEV增量沿用Q1；被修正部分与原误差的关系才决定新增风险。原文以高omega直接解释退化的强度需收回，omega也不能作为检验非线性的前置门槛。

**下一项判别先用实际BF16预测。** 固定同输入、实际consumer完成gate/cast后的FFN贡献z，令`e=z_Dense−z_Q1`、`c=z_stable−z_Q1`，以F64累加方向项`a=<e,c>`与幅度项`b=||c||²`，区分非正方向和有利方向过冲；G乘pre-gate差另列数学诊断，输出线性混合公式不代替重新量化参数的实际验收。新VBench先做CAL11的表示与256/512密度对照，来源广度仍会改变判断时才扩CAL22。详细推进/停止规则见[当前路线](../20261113/G20_VBENCH_RESTART_V4_20261113.md)与[判断台账](../20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)。

**ICC分量提示诊断，不保证修复。** 当前stage2占比不能当可挽回比例。需要native/surrogate两级四格；native-h修复在low19/34仍高于Q1只限制这个修复，误差抵消仍可能使其它修改不同。high4来源等权绝对MSE与逐源相对均值方向不同，必须保留各自聚合身份。六站点数据不识别high对终端视频的因果重要性。

**旧reviewer反例与开发身份保留。** chip10来源和稀疏配对面板不足以外推full-M、全80或自由轨迹；更大row count或满秩也不保证健康谱与来源泛化。新的权重/状态归因线独立推进，不以新算子成功为前置。

## 7. R1 文档版本化更正（本批后执行）

依 scope 预登记，R1 结果文档三处错误已作版本化更正（冻结 R1 发布快照不改）：seed 43→42（两处）；Q1 数学 lift vs BF16 回放"全部 80 站点 ~1e-4"过度陈述→实测逐源中位 0.0250%、最大 1.1103%，站点组 source 等权 MSE 最大差 0.6655%（cheap_low_16 DEV5）；ICC high/4 "abs −9.4e-6 在 epsilon_num=1e-6 尺度无收益"→3/5 DEV 源改善、2/5 恶化，source 等权 MSE 变化 −0.0849% vs 逐源相对均值 +1.7064%，旧更正注记又将epsilon_num误称相对容差，现一并改正：冻结R1 runner的触发条件是`ridge_mse < q1_mse - epsilon_num`，所以它是绝对MSE差阈值，且用于该触发器而非ICC显著性检验。ICC high4应保持绝对MSE与逐源相对聚合各自身份。见[冻结R1 runner](../../reports/20261112/g20_r1_screen/source/runner_screen.py)及[R1 living更正](G20_R1_SAME_INPUT_SCREEN_RESULTS_20261112.md)。

## 8. 边界（主张级）

- small-M（每源 108–146 匹配行）门后代理，**不含 full-M 残差写回**；不能外推为全层/全 token 效果。
- DEV5 已暴露开发来源，非盲确认；CAL4 已用于原 Q1 拟合；单 seed 42；6 昂贵站点为预登记分层样本。
- ~500 行对 5120×5120 D 严重欠定；rank≤517/523 是界不是识别；omega描述CAL行空间外方向比例，不识别风险机制；冻结1%判据不构成来源泛化或机制因果证明。
- 无视频、无质量、无速度主张；无景观重跑、无触发、无新生成/基准；chipmunk 臂仍 MISSING_ASSET（R2）。
- 24 张 BF16 矩阵（各 52,428,928 字节，~1.2 GiB）保留集群原位置，未上传；SHA256 见 [NPY_SHA256.txt](../../reports/20261112/g20_r1b_recheck/evidence/NPY_SHA256.txt)。

## 9. 证据索引

- [SUMMARY.json](../../reports/20261112/g20_r1b_recheck/output/SUMMARY.json)（SHA 与 EXIT 绑定）· [output/](../../reports/20261112/g20_r1b_recheck/output) 6 站点逐源收据（含 omega_stable/omega_r1_reference、ICC 分解、10 臂 SSE、身份比较、矩阵 SHA）
- [scope.json](../../reports/20261112/g20_r1b_recheck/evidence/scope.json) · [独立审查](../../reports/20261112/g20_r1b_recheck/evidence/INDEPENDENT_REVIEW.json) · [作者回应](../../reports/20261112/g20_r1b_recheck/evidence/AUTHOR_REVIEW_RESPONSES.json) · [自审](../../reports/20261112/g20_r1b_recheck/evidence/ROOT_SELF_AUDIT.txt) · [ACTIVE](../../reports/20261112/g20_r1b_recheck/evidence/ACTIVE_RESEARCH.json)
- [RUN](../../reports/20261112/g20_r1b_recheck/evidence/RUN.json)/[EXIT](../../reports/20261112/g20_r1b_recheck/evidence/EXIT.json)/[SHELL_EXIT](../../reports/20261112/g20_r1b_recheck/evidence/SHELL_EXIT.json) · [DISPATCH_INTENT](../../reports/20261112/g20_r1b_recheck/evidence/DISPATCH_INTENT.json)/[SUBMITTED](../../reports/20261112/g20_r1b_recheck/evidence/SUBMITTED.json) · [NPY_SHA256](../../reports/20261112/g20_r1b_recheck/evidence/NPY_SHA256.txt)

本次解释修订不启动新作业；后续范围按[统一VBench路线](../20261113/G20_VBENCH_RESTART_V4_20261113.md)准备，实际计算另经具体独立复审与ACTIVE冻结。
