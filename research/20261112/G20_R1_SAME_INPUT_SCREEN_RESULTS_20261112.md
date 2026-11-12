# G20 R1 同输入筛查 · 实际执行结果（2026-11-12）

**证据身份：NATIVE/OBS（A100 真实 BF16 原生 FFN 与回放统计）；无 VIDEO/SYS 主张。** 本文报告 [下一轮实验路线](../20261111/G20_NEXT_EXPERIMENT_ROUTE_AND_DECISION_TESTS_20261111.md) 首批 R0/R1 的实际执行结果。R0 为只读资产核查；R1 在既有 CAL4/DEV5 捕获上完成 80 站点廉价景观与 6 站点昂贵拟合，0 次新生成。完整机器可读证据、冻结源码与运行收据见 [reports/20261112/g20_r1_screen](../../reports/20261112/g20_r1_screen/README.md)。

## 1. 执行身份与治理

- **Scope**：`G20_R1_SAME_INPUT_SCREEN_20261111`，SHA256 `4bbf8496484f80506bf59565e2b9c1556e19a908de0c8736643126724cf41625`；设计依据 main@`ab41d7389f949150c74131e03c8c2b2106e720ac`。
- **治理链**：作者自审 → 真实独立 reviewer（`qoder_independent_reviewer_g20_r1_20261111`，与作者不同 agent）首轮 REVISE（发现 ridge 正规方程残差诊断的列缩放阻塞性 bug）→ 作者逐项修订 → recheck APPROVE/APPROVE → 阶段级 ACTIVE 冻结（1 job/1 GPU，会话上限 2）→ 发射前远端逐文件 SHA 复核 → 单次提交。记录见 [evidence/INDEPENDENT_REVIEW.json](../../reports/20261112/g20_r1_screen/evidence/INDEPENDENT_REVIEW.json)、[AUTHOR_REVIEW_RESPONSES.json](../../reports/20261112/g20_r1_screen/evidence/AUTHOR_REVIEW_RESPONSES.json) 与 [experiment_reviews](../experiment_reviews/G20_R1_SAME_INPUT_SCREEN_20261111/v1.json)。
- **实际运行**：Slurm job **156476**，anode16（P107-A100，A100-SXM4-80GB×1，4 CPU，64 GiB），2026-11-12 00:32:19 启动，**EXIT.json exit_code 0**、SHELL_EXIT 0，耗时 1059.6 s，峰值 RSS 1.60 GB，stderr 空。[RUN.json](../../reports/20261112/g20_r1_screen/evidence/RUN.json) 记录 scontrol 实时 RUNNING 分配核验；[EXIT.json](../../reports/20261112/g20_r1_screen/evidence/EXIT.json) 绑定 SUMMARY SHA `46ba72860d19fa453bdda5d2eaedad0ad90ddbbb6a598484694d52f69962dd5d`。
- **自检（运行内，GPU）**：共享 Gram 精确解 vs 逐列 F64 稠密参考相对 L2 `2.47e-7`；秩亏构造案例 `2.88e-6`（Gram 数值秩 2）；ICC 零因子零增量精确；reuse-Y 恒等；零矩阵回放 anchor 精确。全部通过后才进入正式计算。

## 2. 数据与臂

CAL4 = chip10_02/04/07/09（拟合），DEV5 = chip10_01/03/05/06/08（已暴露开发来源，**非盲测**）；seed 43，四步 HHLL，d=5120，grid [21,29,52]。每站点按 source 等权、当前门后 SSE/MSE；数学 lift（FP32）与 BF16 cache/gate 回放分报。匹配行数 high 每源 126/137/129/125（CAL 合计 517），low 146/129/132/116（合计 523）。

三臂：**Q1_R2_DEPLOYED_BF16**（G19 部署 A=W1ᵀdiag(α)W2ᵀ）、**FREE_RIDGE_A_Q1_PLUS_D**（A_Q1+D，共享 Gram 一次 eigh 精确解，λ_main=1e-3，λ_robust=1e-2 仅诊断）、**ICC_SVD_R512_TWO_RESIDUAL**（一个冻结 SVD r512 配置，两级低秩+中间 GELU，两残差形态）。**Chipmunk 作者块均值臂 = MISSING_ASSET**：需要 full-M（31668 行）anchor/hit U 的真实体素块均值/mask/有效尾行，现有仅 2048 行均匀面板；按合同不以散点代理打分，推迟到 R2 共享捕获。B_L/B_N 非线性臂本批未运行（预登记条件触发）。

## 3. 80 站点廉价景观（Q1 与 reuse-Y）

| 量（source 等权门后 MSE） | high CAL4 | high DEV5 | low CAL4 | low DEV5 |
| --- | --- | --- | --- | --- |
| Q1 math mean | 0.02824 | 0.03788 | 0.005139 | 0.005596 |
| Q1 math median | 0.03276 | 0.04259 | 0.005207 | 0.005363 |
| Q1 BF16 replay mean | 0.02824 | 0.03788 | 0.005139 | 0.005595 |
| reuse-Y（零修正）mean | 0.13440 | 0.14876 | 0.02075 | 0.02065 |

- **Q1 的数学 lift 与 BF16 回放在全部 80 站点一致到 ~1e-4 相对量级**：部署缓存路径数值忠实，剩余误差是表示性的，不是数值路径故障。
- low expert 绝对门后误差约为 high 的 1/5，且来源外退化更小（DEV/CAL mean 比 low 1.09 vs high 1.34）。**high expert 是主要误差来源**，也是校准覆盖扩展的优先对象。
- Q1 相对 reuse-Y 在 high 消除约 79%（CAL）/75%（DEV）门后 MSE，在 low 消除约 75%/73%；增量归一化残差（increment_normalized）high 约 0.21–0.41、low 约 0.34–0.52，即 Q1 仍留下可观测的增量能量。

## 4. 六站点昂贵拟合：来源外配对差

DEV5 相对 Q1 的 source 等权配对 MSE 变化（math lift；replay 差异 <1e-3 相对）：

| 站点 | FREE_RIDGE mean_rel | ICC r512 mean_rel | omega_dev 范围 | Gram 数值秩 |
| --- | --- | --- | --- | --- |
| high/4 | **+28.0%** | +1.7%（abs −9.4e-6） | 0.363–0.436 | 517 |
| high/19 | **+15.6%** | +27.0% | 0.638–0.689 | 517 |
| high/34 | **+15.1%** | +41.7% | 0.592–0.631 | 517 |
| low/4 | **+28.8%** | +8.2% | 0.464–0.497 | 523 |
| low/19 | **+19.3%** | +71.7% | 0.601–0.685 | 523 |
| low/34 | **+17.0%** | +75.3% | 0.648–0.682 | 523 |

CAL4（样本内）：FREE_RIDGE 六站点配对变化均 ≈ **−99.99%**（MSE 降至 1e-6–1e-8）；ICC 反而 **+29.5%…+99.6%**。

ridge 诊断（λ_main=1e-3）：正规方程相对残差 3.2e-6–9.5e-5；D 数值秩 4239–4732（λ_robust=1e-2 时降至 2834–4405）；Gram 数值秩恰为 CAL 行数 517/523，最小特征值 0 —— **5120×5120 的自由修正由 ~500 行识别，严重欠定**。omega_dev=‖X_test(I−Π_train)‖²/‖X_test‖² 直接测得 DEV 增量能量的 36%–69% 落在 CAL 行空间之外。

## 5. 这些数据说明什么

**（1）当前校准覆盖不足以识别自由线性修正，失败方向是过拟合。** FREE_RIDGE 样本内近乎精确、来源外一致恶化 +15%–29%，且 omega_dev 高：这是覆盖不足的定量 signature，而不是"线性基底必然不够"的证据。竞争解释——真实完整线性算子存在但 CAL4 无法识别它（独立 reviewer 的反例 1）——**未被排除**；判别动作是扩大 CAL 来源/行数后重拟合，观察 omega 是否下降、DEV 配对差是否转负。在覆盖扩大之前，把来源外失败归因于 Q1 基底限制或归因于真实状态依赖，都超出本批证据。

**（2）冻结 ICC r512 两残差配置在自然端点不优于 Q1。** 六站点 CAL/DEV 全部劣于 Q1（DEV +1.7%…+75.3%；high/4 的 abs 变化 −9.4e-6 在 epsilon_num=1e-6 尺度上无实际收益）。这是**该冻结配置**的有界结论：两级低秩+GELU 残差在 r512 下没有比部署对角-α 线性映射捕获更多增量结构；不裁决 ICC 的其他秩/形态。数值身份边界：两残差形态在零增量时输出有 0.016–0.125 的 BF16 级 max-abs 偏差（anchor/hit 舍入不可结合），零因子自检精确；部署前需单独处理中性。

**（3）Q1 仍是当前参照。** 数值路径忠实、来源外稳定，本批没有产生可推进的候选。

**（4）预登记补充站点触发未 firing**：high 40 层 median(reuse−Q1)=0.0721>0 且无 high 昂贵站点 ridge DEV 优于 Q1，high/0 未运行，符合冻结规则。

## 6. 决策与最小下一步

按路线的停止逻辑，本批**不触发 R3 候选干预**（无胜出候选），Q1 保持参考臂。信息价值排序：

1. **校准覆盖判别（最便宜、最直接）**：新增独立 CAL 来源/行（复用 R2 共享捕获），重跑自由 ridge，预登记判据为 omega_dev 下降且 DEV 配对差转负 → 支持覆盖解释；覆盖充分后仍无收益 → 转向状态依赖解释，运行同 bank 的 B_L/B_N 非线性臂（含 delta-only 控制），不新开视频。
2. **R2 共享动态 pilot**：一次捕获同时补 Chipmunk full-M proxy 所需字段（full_M_U_anchor/hit、体素块均值、author top-k mask、有效尾行）与端点 payload；Attention 组合背景（D_A/G_A）按路线前移。
3. **high expert 优先**：景观显示 high 的绝对误差与来源外退化均更大，覆盖扩展与非线性检验先在 high 站点判别。

## 7. 边界（主张级）

- small-M（每源 108–146 匹配行）门后代理，**不含 full-M 残差写回**；不能外推为全层/全 token 效果。
- DEV5 为已暴露开发来源，非盲确认；单 seed 43；6 昂贵站点是预登记分层样本，不作全 80 站点均匀性主张。
- 无视频、无质量、无速度主张；ICC/Chipmunk 的全视频比较另需同 cohort 臂。
- 6 张 ridgeD BF16 矩阵（各 52,428,928 字节）保留在集群原位置，未上传；SHA256 见 [NPY_SHA256.txt](../../reports/20261112/g20_r1_screen/evidence/NPY_SHA256.txt)。历史收据不保证中间数组永久存在。

## 8. 证据索引

- [SUMMARY.json](../../reports/20261112/g20_r1_screen/SUMMARY.json)（87 个输出 JSON 的汇总，SHA 与 EXIT 绑定）· [output/](../../reports/20261112/g20_r1_screen/output) 80 廉价 + 6 昂贵逐站点收据
- [R0 资产清单](../../reports/20261112/g20_r1_screen/R0_ASSET_INVENTORY.md)（可复用/缺字段表、gate 逐位核验、交集计数）
- [scope.json](../../reports/20261112/g20_r1_screen/evidence/scope.json) · [独立审查](../../reports/20261112/g20_r1_screen/evidence/INDEPENDENT_REVIEW.json) · [作者回应](../../reports/20261112/g20_r1_screen/evidence/AUTHOR_REVIEW_RESPONSES.json) · [自审](../../reports/20261112/g20_r1_screen/evidence/ROOT_SELF_AUDIT.txt)
- [RUN](../../reports/20261112/g20_r1_screen/evidence/RUN.json)/[EXIT](../../reports/20261112/g20_r1_screen/evidence/EXIT.json)/[SHELL_EXIT](../../reports/20261112/g20_r1_screen/evidence/SHELL_EXIT.json) · [receipt_index.json](../../reports/20261112/g20_r1_screen/receipt_index.json) · [SOURCE_MANIFEST.json](../../reports/20261112/g20_r1_screen/SOURCE_MANIFEST.json) · [PUBLICATION_MANIFEST.json](../../reports/20261112/g20_r1_screen/PUBLICATION_MANIFEST.json)

本次发布不启动新作业；后续范围按路线另行独立复审与 ACTIVE 冻结。