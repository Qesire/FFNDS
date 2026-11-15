# G20 两个核心方向：结果索引与勘误

本索引保留 11 月 14 日 A/BW 的数值复核，并更新到 `0d4d860a847202e9e0b3c4265da7fd64e87eff21` 的输入采集结果。**当前：A 已完成；BW Q1/ICC 已完成；P64 输入已齐；132 个 BW Chip 风险单元仍待计算；BWS 尚无已发布运行结果。** 各实验冻结目录内的源码、输出、审查与收据保持原身份；当前状态与资源勘误由本索引及现有资产/判断台账维护。

A 与 BW 的已完成比较均使用现有 VBench 数据，统计单位为 source；先对三个层等权，再对 high/low 等权，最后对 11 个 DEV source 等权。DEV 未参与本次参数拟合，但此前已被观察，属于开发集。结果限于六个站点及固定 512 行局部回放，视频质量与实际系统时延均未测。

## 2026-11-15：输入已齐，接下来直接计算 Chip 分数

正式采集 job **164928** 已 EXIT 0。遍历[输入 manifest](B_CHIP_STATS/v1/output/B_CHIP_P64_INPUTS.json)得到 **11 来源 × 6 站点 × 2 端点 = 132 份 payload**；每份含 64×5120 BF16 的真实 pre-W1 输入，合计 **8448 行、781,265,364 bytes**。按冻结判据比较新旧 PANEL512 输入与 gate 的 **264 项检查全部通过**，11/11 来源通过；输入可与既有 BW 状态配对。[SUMMARY](B_CHIP_STATS/v1/output/SUMMARY.json) · [EXIT](B_CHIP_STATS/v1/evidence/EXIT.json) · [集群产物验证记录](B_CHIP_STATS/v1/evidence/verify_post_chip.log)。大 tensor 保留在集群，本次公开证据核查没有重新下载其实体。

**这次新增的是可用输入，还没有新增 Chip 风险结果。** `66 unblocked` 是 66 个来源×站点输入对，可供两权重形成 132 个风险单元。新的采集 runner 没有计算 selector/packed-pregate 风险；旧 BW runner 仍只运行 Q1/ICC。接手时复用现有 `chip_predict` 和 `primary_risk`，补接现成固定64输入，只计算缺少的 132 个 Chip 单元，再与既有 Q1/ICC 表计算 d/θ。不要直接重跑旧 BW runner，也不重新拟合 Q1、分解 ICC 或采 D4。[旧 BW 消费函数](BW/v3/source/b_common_replay.py) · [旧 BW runner](BW/v3/source/b_weight_state_runner.py)。这是原 BW 比较的剩余部分，不增加研究方向。

本次采集有两项应如实记账的偏差：

- **实际可见消耗为 12 条 D4：正式 11 条，加 smoke 164771 的 1 条。** 自审明确说明 smoke 使用一条完整 D4；正式 job 又捕获全部 DEV11，包含相同的 vbi_0056，未复用 smoke 临时产物。这比原输入补齐总上限 11 多一条。正式 scope 的 11 条计数不能代表全部消耗。[自审 §7](B_CHIP_STATS/v1/evidence/ROOT_SELF_AUDIT.txt) · [smoke 日志](B_CHIP_STATS/v1/evidence/cluster_evidence.txt) · [正式逐源日志](B_CHIP_STATS/v1/evidence/slurm-164928.out)。
- **正式执行了 11 次 VAE 解码和 11 个 MP4 写出，smoke 另有 1 次。** 正式日志的 VAE 解码合计 38.622604 秒。`new_video_decode=0` 不能用于 latent→frames 生成解码；可保留的零计数是未事后读取 MP4 做帧解码/质量评分。解码在输入捕获后发生，不推翻已采 U。后续既定纯状态捕获应省去末端 VAE/视频写出；该修正并入原消费者，不另开验证轨迹。[正式日志](B_CHIP_STATS/v1/evidence/slurm-164928.err) · [生成调用](B_CHIP_STATS/v1/source/b_chip_stats_runner.py)。

输入已满足消费需求，以上偏差只更正记录，不触发重新捕获。当前正式作业累计 **5 次，3 成功、2 失败**；smoke 和验证作业另记。新发布清单 38 项有 37 项字节匹配，`runtime_guard.py` 仍发生已知 CRLF→LF 转换；执行/仓库摘要映射及本次进度核查见[现有判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)的 `progress_check_20261115`。

剩余工作只有 **BW 的 Chip 评分**与既定 **BWS**。BW Chip 消费现有输入，新增完整轨迹为 0；BWS 保持 CAL3/DEV3、最多 6 条 O40，首次用于验证的完整轨迹也计入这六条并保存复用。BWS Q1/ICC 不等待 Chip 评分。完成后统一解释三个实际方法的权重与状态差异，不再派生输入复审、独立 smoke 或参数网格。

## A：Q1 残差修正有用，当前最佳为可折叠线性修正

A 保持已发布的 Q1 参数，用 CAL11 拟合三个固定残差修正，再在 DEV11 读出。下表从 [原始逐源、逐站点 READOUT](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_READOUT.json) 重算；math 与 BF16 各有 528 个单元，分别排名。

| 方法 | DEV math gated MSE | 相对 Q1 改善 | DEV BF16 gated MSE | 相对 Q1 改善 |
|---|---:|---:|---:|---:|
| CURRENT_Q1 | 0.014899471534 | — | 0.014898749938 | — |
| Q1 + LINEAR64_Q32（L） | **0.014292594482** | **4.0731%** | **0.014292358009** | **4.0701%** |
| Q1 + DELTA_N32（D） | 0.014378064351 | 3.4995% | 0.014377692845 | 3.4973% |
| Q1 + ENDPOINT_N32（E） | 0.014352600253 | 3.6704% | 0.014353198519 | 3.6617% |

三个修正臂在两种精度下均为 **11/11 source 改善**，留一源均值均保持正方向。L 相对 Q1 的 math 留一源改善范围为 `5.87566e-4～6.29778e-4`。L 对 D 为 11/11 source 胜出，对 E 为 10/11；唯一 E 略优于 L 的来源是 `vbi_0800`。L、E 均在 66/66 个 DEV source×site 优于 Q1，D 为 65/66。留一源结果表明均值方向不由某一个来源决定，不等同于新的独立验证。

**端点条件只有弱局部优势。** E 对 D 为 9/11 source 胜出，math 平均改善 `2.54641e-5`，BF16 为 `2.44943e-5`；math 的 high/low 改善分别为 `5.06853e-5`、`2.42916e-7`，优势几乎全部来自 high，low04 方向相反。E 没有胜过当前最佳线性修正，因此本结果不支持把非线性设为主推荐或追加非线性搜索。

L 的修正为 `(P64/gamma_L) C_L Q32ᵀ`：输入基 64 维、输出基 32 维，**修正矩阵秩至多 32**，C 与 D/E 同为 2048 个拟合系数。该修正可预先并入 Q1 的 A，沿用一次 dense matmul 和一个 5120 维残差 cache，无新增运行期非线性支路。此处说明算子结构，实际时延未测。

[源码](A_Q1_RESIDUAL/v1/source/a_q1_residual_repr.py)与[运行器](A_Q1_RESIDUAL/v1/source/a_q1_residual_runner.py)符合固定 Q1 祖先、CAL-only P/Q、pre-gate 残差 Q32、中心/尺度、每臂 RMS 和独立 BF16 回放的约定。[CHECKS](A_Q1_RESIDUAL/v1/output/CHECKS.json) 中 132/132 个 Q1 祖先风险完全一致，18/18 个 CAL 正则目标下降，C=0 检查通过。

新 A 相比旧 Q0 构造同时改变了基底、Q 残差坐标、RMS 缩放及端点中心/尺度。它说明新构造在当前局部 DEV 上有效，不能把转正单独归因于其中一个改动；旧构造的负结果仍按其原范围保留。两张精度表保留相同收益方向和量级，也不能据此宣称逐输出数值等价。

## BW：共同 μ_D 上，权重切换造成的 gap 变化很小

BW 对原始权重 W_O 和蒸馏权重 W_D 使用相同 D4 状态 **μ_D**、相同 PANEL512 和 G_ref；每个权重重算自己的 anchor/target，并各自拟合 Q1。主风险是实际 BF16 pre-gate prediction 与 own-W target 的误差，升 F64 后乘共同 gate 计算 MSE。BF16 gate 舍入辅表另列，不与主表混合。

依据 [BW_CELL](BW/v3/output/BW_CELL.csv)、[BW_SOURCE](BW/v3/output/BW_SOURCE.csv) 与 [BW_AGGREGATE](BW/v3/output/BW_AGGREGATE.json)：

| 权重 | Q1 主风险 | ICC 主风险 | d(W) = R_ICC − R_Q1 |
|---|---:|---:|---:|
| W_O | 0.014902516232 | 0.021427942884 | +0.006525426652 |
| W_D | 0.014899847213 | 0.021438562810 | +0.006538715597 |

`theta = d(W_D) − d(W_O) = +1.32889445e-5`，约为 W_O 方法 gap 的 **0.204%**。11/11 source 的 theta 为正，11 个留一源均值也为正，范围为 `1.23801e-5～1.40511e-5`。但 expert 分解的方向不同：**high 为 +2.7958613e-5，low 为 −1.3807240e-6**。

能够支持的结论是：**在共同 μ_D、这两套权重及固定方法配置下，两个权重条件的 Q1 平均局部风险均低于 ICC，权重切换对方法 gap 的平均改变量很小，且 high/low 存在方向差异。** ICC 在 124/132 个 source×site×weight 配对单元中风险较高，仍有 8 个局部例外。这不建立“权重独立”或“权重没有影响”的结论，更不代替尚未完成的 BWS 状态比较。A 的固定历史 Q1 与 BW 按权重重新拟合的 Q1 属于不同实验，两张表分别解释。

方法成本尚未匹配。按 [BW_COST](BW/v3/output/BW_COST.csv)，Q1 的 analytic MAC/token 为 **26,214,400**，ICC 为 **19,398,656**；对应 cache bytes/token 为 10,240 与 37,888。ICC 的算术量更低、cache 更大，Q1 的风险更低；这些是当前配置的风险与成本比较，不能写成 Q1 在相同预算下全面胜出。表中整次运行耗时不代表方法本身的原生时延。

## 已完成产物与既定余项

| 项目 | 当前状态与消费入口 |
|---|---|
| A_Q1_RESIDUAL | `COMPLETE`；[SUMMARY](A_Q1_RESIDUAL/v1/output/SUMMARY.json)、[参数清单](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_PARAMS.json)、[源码清单](A_Q1_RESIDUAL/v1/source/SOURCE_MANIFEST.json) |
| BW_Q1ICC | `COMPLETE`；264/264 个有效 DEV 单元；[SUMMARY](BW/v3/output/SUMMARY.json)、[参数清单](BW/v3/output/PARAMS_MANIFEST.json)、[源码清单](BW/v3/source/SOURCE_MANIFEST.json) |
| B_CHIP_P64_INPUTS | 已完成；[固定64输入清单](B_CHIP_STATS/v1/output/B_CHIP_P64_INPUTS.json)，132 份 payload、11/11 同状态通过 |
| BW Chip GLOBAL_P64 | 输入已齐，132 个风险单元待计算；复用 Q1/ICC 结果，只补 Chip 评分 |
| BWS | 尚未执行；共同 μ_D 的 BW 不能回答原生状态变化问题 |

下一步按[既定执行规格](../../../research/20261114/G20_TWO_CORE_EXECUTION_SPEC_20261114.json)与[交接手册](../../../research/20261114/G20_TWO_CORE_EXECUTION_HANDOFF_20261114.md)消费已完成输入：只补 132 个 BW Chip 分数，并推进固定 BWS（最多 6 条 O40，包含任何验证用完整轨迹）。已完成 A、BW Q1/ICC 与 P64 采集直接复用；BWS 不由 theta 的符号决定是否执行。本次维护不授权或调度新作业，也不增加图文、CAL_B、all80 或视频评分。

## 对 11 月 14 日 A/BW 冻结摘要的勘误

以下更正供后续引用；原冻结档案不原地改写。

1. **A 输出预算。** 参数实测 `3,173,954,223 bytes`，约 2.956 GiB，超过原 SPEC 明确的 `2,147,483,648 bytes` 总上限。实现额外保存了六份可从 A_q1_f32 精确重建的 F64 A_q1，共 `1,258,291,200 bytes`。2 GiB 不能事后改称每文件上限或仅为估算。保留原产物及 SHA，下游紧凑导出可省去这些副本；此问题不推翻评分，也不要求 A 重跑。原[审查](A_Q1_RESIDUAL/v1/evidence/INDEPENDENT_REVIEW.json)与[答复](A_Q1_RESIDUAL/v1/evidence/AUTHOR_RESPONSE.json)对该项的表述据此修正。
2. **BW 失败原因与正式作业计数。** v1 作业 `164189` 实际因 BF16 tensor 的 `.numpy()` 哈希路径失败；v2 作业 `164225` 实际因 `aggregate_bw` 方法键错误失败。参数未保存是预审发现，不能写成 v1 作业的退出原因。v3 `164312` 与 A `164072` 成功；加上前述两次失败，A/BW 阶段共 **4 次正式 dispatch（A 1次、BW 3次）：2 成功、2 失败**。另有已记录的 GPU smoke 作业 `163922`、`164074`、`164207`，v3 smoke 未列 job ID；A smoke 为 CPU。这些 smoke 不计入上述4次正式 dispatch，总运行资源尚未完整量化。原[BW 自审](BW/v3/evidence/ROOT_SELF_AUDIT.txt)、[独立审查](BW/v3/evidence/INDEPENDENT_REVIEW.json)与[答复](BW/v3/evidence/AUTHOR_RESPONSE.json)保留供追溯。
3. **源码执行身份与归档身份。** 8 个源码文本在仓库归档时发生 CRLF→LF，执行字节 SHA 与仓库归档 SHA 应分别记录，不能声称两者逐字节相同。具体映射见[既有 decision ledger](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json) 的 `execution_updates_20261114.two_core_20261114.postpublication_check.source_publication_identity`，并对照两项实验的 source 清单。此归档差异不表示运行时计算被更改。
4. **结论强度与 BF16 导出验证。** BW 原摘要中的“权重基本无影响/weight-independence”按上述共同 μ_D 的小平均改变量解释。A 的 BF16 分数来自重载参数重建的执行路径，显式导出相等检查只覆盖 A_Q1/A_L；不能称全部导出 BF16 张量都已直接端到端消费验证，也不能把两表均值接近写成逐输出精度等价。

