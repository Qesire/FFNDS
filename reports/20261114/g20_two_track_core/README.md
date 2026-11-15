# G20 两个核心方向：结果索引与勘误

**当前判断：A 的固定残差比较已完成，保留可折叠线性修正 L；BW 的三方法比较已完整，Chip 局部误差最低，但主路径计算量是 Q1 的 2 倍；BWS 已登记运行成功，原始四格结果尚未发布，方向 B 的状态解释仍待完成。** 本次核查基于提交 `1d5075286e54a2e141893bfac55f3c7238989f65`。更新已有索引、[资产与队列](../../../research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)和[判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)，新增计算作业为 0。

| 阶段 | 当前证据与剩余工作 |
| --- | --- |
| A_Q1_RESIDUAL | 已公开完整结果；L 的 DEV 风险较其 Q1 基线改善约 4.07%，当前无需追加候选或扩层 |
| BW | 已公开全部 396 个 DEV 单元：Q1/ICC 264 个原值复用，加 132 个实际 GLOBAL_P64 Chip 单元；评分已完成 |
| BWS | `REPORTED_EXIT0_PUBLICATION_PENDING`：记录称 job 166225 EXIT 0、216 个 DEV 单元；当前缺的是原始产物归档和解释 |

A/BW 均使用现有 VBench 数据。先对三个层等权，再对 high/low 等权，最后对 11 个 DEV source 等权；独立统计单位为 source，n=11。DEV 未参与本次拟合，但此前已被观察，属于开发集。结论限于六个站点、固定 512 行局部回放；视频质量与实际方法时延未测。BWS 为单独的 CAL3/DEV3 设计，不能与 BW 的 CAL11 表拼接。

## BW 已完成：Chip 的误差优势与计算成本同时成立

新评分 job **166479**（V2）已 EXIT 0，补齐 **132 个 Chip 单元**。评分使用用户实际 `CHIPMUNK_GLOBAL_P64`，按 own-W 的固定 64 行 hidden-change RMS 稳定排序，选择 K=5120，执行选定 W1/W2 的残差路径。新旧 264 对配对单元的 U0/U1、行号、G_ref、own anchor/target 等 16 项公共身份字段全部一致；264 个 Q1/ICC 风险值逐项原样复用，没有重新拟合、分解或重放。该次评分 **0 新 D4/O40 轨迹、0 新图文、0 视频解码**。[运行摘要](BW_CHIP_SCORE/v2/output/SUMMARY.json) · [RUN](BW_CHIP_SCORE/v2/evidence/RUN.json) · [EXIT](BW_CHIP_SCORE/v2/evidence/EXIT.json) · [实际运行器](BW_CHIP_SCORE/v2/source/bw_chip_score_runner.py)。

### 三方法在共同 D4 状态上的风险

BW 对原始权重 W_O 和蒸馏权重 W_D 使用相同 D4 状态 μ_D、PANEL512 和 G_ref；每种权重重算自己的 BF16 anchor/target，并按 CAL11 规则各自拟合 Q1。主风险以 BF16 门前 prediction 与 own-W target 的误差升至 F64，再乘共同 gate 求 MSE；BF16 gate 舍入结果另表。

| 权重 | Chip 主风险 | Q1 主风险 | ICC 主风险 | Chip 相对 Q1 风险下降 |
| --- | ---: | ---: | ---: | ---: |
| W_O | **0.011515711132** | 0.014902516232 | 0.021427942884 | **22.7264%** |
| W_D | **0.011497516244** | 0.014899847213 | 0.021438562810 | **22.8347%** |

数据：[Chip 逐单元](BW_CHIP_SCORE/v2/output/BW_CHIP_CELL.csv)、[Chip 逐源](BW_CHIP_SCORE/v2/output/BW_CHIP_SOURCE.csv)、[Chip 聚合](BW_CHIP_SCORE/v2/output/BW_CHIP_AGGREGATE.json)，以及原 [Q1/ICC 逐单元](BW/v3/output/BW_CELL.csv)和[聚合](BW/v3/output/BW_AGGREGATE.json)。百分比由上述来源等权平均风险之比计算。

**两个权重下均有 11/11 source 的平均排序为 Chip < Q1 < ICC。** Chip 在 116/132 个 source×site×weight 单元胜过 Q1；16 个局部例外集中于 low/19 和 low/34。Chip 在全部 132 个单元胜过 ICC；完整三者排序在 108/132 个单元成立。因此，当前局部结果不能继续把 ICC 与 Chip 合写为“均劣于 Q1”。

### 权重切换对平均方法差的影响很小，但有 expert 异质性

定义 d_m(W)=R_m(W,μ_D)−R_Q1(W,μ_D)，θ_m=d_m(W_D)−d_m(W_O)。

| 方法 | d(W_O) | d(W_D) | θ | θ 绝对值 / 原 gap 绝对值 |
| --- | ---: | ---: | ---: | ---: |
| ICC | +0.006525426652 | +0.006538715597 | +1.32889445e−5 | 0.2036% |
| Chip | −0.003386805100 | −0.003402330969 | −1.55258689e−5 | 0.4584% |

ICC 的逐源 θ 为 11/11 正，11 个留一源均值为 `+1.23801e−5～+1.40511e−5`。Chip 的逐源 θ 为 9 负、2 正，11 个留一源均值为 `−1.95447e−5～−1.16032e−5`。留一源不是新增独立样本。Expert 分解：ICC high/low 的 θ 分别为 `+2.7958613e−5 / −1.3807240e−6`，Chip 为 `−3.24111379e−5 / +1.35940007e−6`。[BW_SOURCE](BW/v3/output/BW_SOURCE.csv) · [BW_CHIP_SOURCE](BW_CHIP_SCORE/v2/output/BW_CHIP_SOURCE.csv)。

这支持**共同 μ_D、固定方法配置和各权重 CAL11 拟合下，权重切换对相对 Q1 的平均误差差距影响很小**。它不建立一般的权重独立性，也不说明 mask 相同：Chip 的 66/66 个 source×site 在两权重下选择索引 SHA 均不同。不同 SHA 只说明索引不完全相同，不量化差异大小。

### 成本决定如何理解“更好”

| 方法 | 主路径 analytic MAC/token | 相对 Q1 | cache bytes/token |
| --- | ---: | ---: | ---: |
| ICC r512 | 19,398,656 | 0.74× | 37,888 |
| Q1 | 26,214,400 | 1× | 10,240 |
| Chip K5120 | **52,428,800** | **2×** | 37,888 |

按 [BW_COST](BW/v3/output/BW_COST.csv) 的公式与实际 [Chip packed-pregate 实现](BW_CHIP_SCORE/v2/source/chipmunk_wan_packed_pregate.py)，Chip 两个选定分支各为 5120×5120，主体 MAC 合计为 Q1 的 2 倍、ICC 的约 2.70 倍。这里还没有计入 64 行 proxy、排序、权重 packing 与 cache clone；整次评分的约 345.8 秒不是任何单一方法的原生时延。旧 BW_COST 中 Chip 的 blocked 标记只描述当时评分状态，成本公式沿用。

当前三个配置展示了不同的误差与计算量取舍；并未建立等计算成本、真实速度或视频质量的优胜。A 的 L 可折叠进 Q1、没有新增在线 GEMM，仍有研究价值。**A 的约 4.07% 与这里 Chip 的约 22.8% 使用不同 Q1 基线及目标生成口径，不能直接排成同一候选胜负表。**

### 实现与归档核验

当前评分采用已声明的 `OWNW_LOCAL512_P64ANCHOR64_V1`，不改称与历史 full-M GEMM 逐位等价。P64 manifest 包含 **132 个独立 anchor/hit payload**，同一批文件在两权重下共加载、校验 **264 次**；不能写成 264 个独立 payload 或样本。

本次新档案的 PUBLICATION manifest **35/35**、SOURCE manifest **14/14**、scope 代码 **13/13**及 B_BINDINGS hash 均与仓库实际字节匹配；RUN/EXIT 与输出身份相符。旧批次 CRLF→LF 的归档勘误仍保留其历史身份。[发布清单](BW_CHIP_SCORE/v2/PUBLICATION_MANIFEST.json) · [源码清单](BW_CHIP_SCORE/v2/source/SOURCE_MANIFEST.json) · [冻结 scope](BW_CHIP_SCORE/v2/evidence/scope.json)。

仓库记录 V1 job **166419** 又因 BF16 tensor 直接 `.numpy()` 散列而失败，原失败日志未在本批公开；V2 恢复了 BW/v3 已验证的 uint8-view 实现，未改变预测、目标或风险定义。此问题的修复应直接复用已验证函数，并在新输出路径做一次真实 dtype 的紧凑检查；不增加 pilot、复审层级或科学重跑。[记录](BW_CHIP_SCORE/v2/README.md) · [自审](BW_CHIP_SCORE/v2/evidence/ROOT_SELF_AUDIT.txt)。

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

## BWS：已登记计算完成，剩下已有四格结果的归档和解释

当前证据来自[既有判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)的 `execution_updates_20261114.two_core_20261114.BWS`。该记录报告 job **166225 EXIT 0**、CAL3/DEV3、两权重×两状态×三方法，共 **216 个 DEV 单元**（144 Q1/ICC + 72 Chip），报告平均交互为 ICC `+9.680e−6`、Chip `−2.674e−5`。但本次提交没有发布 BWS 的 scope、RUN、EXIT、参数 manifest 与逐单元/逐源四格表，因此这些数字目前属于报告值，尚未像 BW 一样独立复算。状态使用 `REPORTED_EXIT0_PUBLICATION_PENDING`。

仓库还记载 “Publication and direction-B closure deferred per user”。这是现有发布暂缓记录，继续保留；它不意味着再申请研究目标许可，也不将已完成的实验重新加入执行队列。

**不能只凭小交互值把方向 B 宣布为“权重和状态都无影响”。** 令 d_m(W,μ)=R_m(W,μ)−R_Q1(W,μ)，现有交互定义为：

$$
I_m=[d_m(W_D,\mu_D)-d_m(W_O,\mu_D)]-[d_m(W_D,\mu_O)-d_m(W_O,\mu_O)].
$$

即使 I_m 很小，两个权重下仍可共同存在较大的状态主效应 `d_m(W,μ_O)−d_m(W,μ_D)`。归档后应从**同一 BWS 混合 CAL3 拟合**读取四格绝对风险、每格方法差、各权重的状态主效应、交互及逐源/expert 差异。不得用 BW 的 CAL11 θ 与 BWS 的 CAL3 交互拼接反推缺失格，也不把原生 O40/D4 的复合差异单独归因于步数、CFG 或某一个状态因素。

剩余动作都消费现有产物：整理已存在的 scope/RUN/EXIT、参数和输入 manifest、216-cell 表及逐源/聚合表，按既定发布安排归档后完成上述解释。无需再 capture、refit、replay、独立 smoke、扩图文、CAL_B、参数网格或全层视频；记录缺失先定位原产物，不把“未公开”当作“未运行”。

记录中的 O40 消耗为 **6 条正式 + 2 条 smoke = 8 次执行**，覆盖 6 个不同来源；两个 smoke 来源 `vbi_1005/vbi_1013` 被重采，未复用，超过原 6 次总上限。此处依据进度记录，原始日志尚待归档核验。纠正方式是保留真实消耗和结果，在今后 producer 中落实复用；不通过再跑 BWS 修复已发生的偏差。

## 已完成资产、正式作业与当前顺序

| 资产 | 当前消费入口 |
| --- | --- |
| A 参数与结果 | [参数清单](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_PARAMS.json)、[READOUT](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_READOUT.json)、[EXIT](A_Q1_RESIDUAL/v1/evidence/EXIT.json)；完整复用 |
| BW Q1 参数、ICC 因子与 264 个分数 | [参数清单](BW/v3/output/PARAMS_MANIFEST.json)、[BW_CELL](BW/v3/output/BW_CELL.csv)、[EXIT](BW/v3/evidence/EXIT.json)；完整复用 |
| P64 固定输入 | [132 份 payload 清单](B_CHIP_STATS/v1/output/B_CHIP_P64_INPUTS.json)、[EXIT](B_CHIP_STATS/v1/evidence/EXIT.json)；已被 Chip 评分消费 |
| BW Chip 132 个分数 | [BW_CHIP_CELL](BW_CHIP_SCORE/v2/output/BW_CHIP_CELL.csv)、[SUMMARY](BW_CHIP_SCORE/v2/output/SUMMARY.json)、[EXIT](BW_CHIP_SCORE/v2/evidence/EXIT.json)；固定 selector，无新增 CAL fit 资产 |
| BWS 状态、参数与 216 个分数 | 既有 ledger 报告已产生；待已有原始文件归档，不请求重新生产 |

本阶段已登记的正式 job 至少 **8 次：报告成功 5、失败 3**。其中 **4 个成功作业有公开 EXIT**（A 164072、BW 164312、P64 164928、Chip 166479），另一个成功是尚未公开收据的 BWS 166225；三次失败登记为 BW 164189/164225 与 Chip 166419。该计数是已登记下界，排除 smoke/verification，不断言没有其他未公开的 BWS 尝试。详细证据层级保存在现有 ledger 的 `progress_check_chip_score_20261115`。

**当前顺序只有 BWS 已有产物归档 → 四格解释 → 两方向总结。** A、BW 与输入生产均不再排队。此前的预算与具体 scope 保持原身份；本次检查新增作业为 0，不新增执行授权或审批层级。

## 历史：P64 输入采集及资源勘误

以下保存 `0d4d860` 输入采集发布时的证据和计数；后续 Chip 已完成，当前状态以上文为准。

正式采集 job **164928** 已 EXIT 0。遍历[输入 manifest](B_CHIP_STATS/v1/output/B_CHIP_P64_INPUTS.json)得到 **11 来源 × 6 站点 × 2 端点 = 132 份 payload**；每份含 64×5120 BF16 的真实 pre-W1 输入，合计 **8448 行、781,265,364 bytes**。按冻结判据比较新旧 PANEL512 输入与 gate 的 **264 项检查全部通过**，11/11 来源通过；输入可与既有 BW 状态配对。[SUMMARY](B_CHIP_STATS/v1/output/SUMMARY.json) · [EXIT](B_CHIP_STATS/v1/evidence/EXIT.json) · [集群产物验证记录](B_CHIP_STATS/v1/evidence/verify_post_chip.log)。大 tensor 保留在集群，本次公开证据核查没有重新下载其实体。

**在该次输入采集发布时，Chip 风险尚未计算；此缺项已由后来的 BW_CHIP_SCORE/v2 补齐。** `66 unblocked` 是 66 个来源×站点输入对，可供两权重形成 132 个风险单元。新的采集 runner 没有计算 selector/packed-pregate 风险；旧 BW runner 仍只运行 Q1/ICC。后续评分已复用这些输入，完成状态以上文当前索引为准。

本次采集有两项应如实记账的偏差：

- **实际可见消耗为 12 条 D4：正式 11 条，加 smoke 164771 的 1 条。** 自审明确说明 smoke 使用一条完整 D4；正式 job 又捕获全部 DEV11，包含相同的 vbi_0056，未复用 smoke 临时产物。这比原输入补齐总上限 11 多一条。正式 scope 的 11 条计数不能代表全部消耗。[自审 §7](B_CHIP_STATS/v1/evidence/ROOT_SELF_AUDIT.txt) · [smoke 日志](B_CHIP_STATS/v1/evidence/cluster_evidence.txt) · [正式逐源日志](B_CHIP_STATS/v1/evidence/slurm-164928.out)。
- **正式执行了 11 次 VAE 解码和 11 个 MP4 写出，smoke 另有 1 次。** 正式日志的 VAE 解码合计 38.622604 秒。`new_video_decode=0` 不能用于 latent→frames 生成解码；可保留的零计数是未事后读取 MP4 做帧解码/质量评分。解码在输入捕获后发生，不推翻已采 U。后续既定纯状态捕获应省去末端 VAE/视频写出；该修正并入原消费者，不另开验证轨迹。[正式日志](B_CHIP_STATS/v1/evidence/slurm-164928.err) · [生成调用](B_CHIP_STATS/v1/source/b_chip_stats_runner.py)。

输入已满足消费需求，以上偏差只更正记录，不触发重新捕获。截至该次输入采集发布，正式作业累计为 **5 次，3 成功、2 失败**（历史快照）；smoke 和验证作业另记。新发布清单 38 项有 37 项字节匹配，`runtime_guard.py` 仍发生已知 CRLF→LF 转换；执行/仓库摘要映射及本次进度核查见[现有判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)的 `progress_check_20261115`。

## 对 11 月 14 日 A/BW 冻结摘要的勘误

以下更正供后续引用；原冻结档案不原地改写。

1. **A 输出预算。** 参数实测 `3,173,954,223 bytes`，约 2.956 GiB，超过原 SPEC 明确的 `2,147,483,648 bytes` 总上限。实现额外保存了六份可从 A_q1_f32 精确重建的 F64 A_q1，共 `1,258,291,200 bytes`。2 GiB 不能事后改称每文件上限或仅为估算。保留原产物及 SHA，下游紧凑导出可省去这些副本；此问题不推翻评分，也不要求 A 重跑。原[审查](A_Q1_RESIDUAL/v1/evidence/INDEPENDENT_REVIEW.json)与[答复](A_Q1_RESIDUAL/v1/evidence/AUTHOR_RESPONSE.json)对该项的表述据此修正。
2. **BW 失败原因与正式作业计数。** v1 作业 `164189` 实际因 BF16 tensor 的 `.numpy()` 哈希路径失败；v2 作业 `164225` 实际因 `aggregate_bw` 方法键错误失败。参数未保存是预审发现，不能写成 v1 作业的退出原因。v3 `164312` 与 A `164072` 成功；加上前述两次失败，A/BW 阶段共 **4 次正式 dispatch（A 1次、BW 3次）：2 成功、2 失败**。另有已记录的 GPU smoke 作业 `163922`、`164074`、`164207`，v3 smoke 未列 job ID；A smoke 为 CPU。这些 smoke 不计入上述4次正式 dispatch，总运行资源尚未完整量化。原[BW 自审](BW/v3/evidence/ROOT_SELF_AUDIT.txt)、[独立审查](BW/v3/evidence/INDEPENDENT_REVIEW.json)与[答复](BW/v3/evidence/AUTHOR_RESPONSE.json)保留供追溯。
3. **源码执行身份与归档身份。** 8 个源码文本在仓库归档时发生 CRLF→LF，执行字节 SHA 与仓库归档 SHA 应分别记录，不能声称两者逐字节相同。具体映射见[既有 decision ledger](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json) 的 `execution_updates_20261114.two_core_20261114.postpublication_check.source_publication_identity`，并对照两项实验的 source 清单。此归档差异不表示运行时计算被更改。
4. **结论强度与 BF16 导出验证。** BW 原摘要中的“权重基本无影响/weight-independence”按上述共同 μ_D 的小平均改变量解释。A 的 BF16 分数来自重载参数重建的执行路径，显式导出相等检查只覆盖 A_Q1/A_L；不能称全部导出 BF16 张量都已直接端到端消费验证，也不能把两表均值接近写成逐输出精度等价。

