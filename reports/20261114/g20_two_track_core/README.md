# G20 两个核心方向：结果索引与勘误

**2026-11-16 最新任务：[完整复现与受控重做](../../../research/20261107/ffn_baseline_reproduction/STRICT_REPRODUCTION_AND_REDO_20261116.md)。** 已完成的A/BW/BWS仍按原范围闭合；本次用户新增要求是先补齐作者参考与最小Wan端口，不能以核心公式相符或旧变体视频代替完整复现。ICC四状态/SVD-CA-CD、Chip官方模块/selector/Triton与全部状态分支是新参考；双残差与Global-P64单独消融。已核官方Wan2.1只接attention、A100有官方Triton，发现具体计数与几何接口问题；新增模型运行0。当前已进一步补固定代码引述并收缩预算：主CAL11、三家族局部验收与选择、五臂DEV11×seed42为55视频槽位；全部D4尝试最多99，旧897提案撤回。当前任务是参考实现及运行准备，以下“当前待办为空”只指此前三个固定比较。

**2026-11-16 原文与作者代码核对：[ICC/Chipmunk的Wan适配、FFN结构与蒸馏](../../../research/20261116/ICC_CHIPMUNK_WAN_FIDELITY_AND_FFN_STRUCTURE_20261116.md)。** ICC四状态核心忠实，实际双残差是实数等价、BF16及成本不同的缓存适配；Global-P64为明确变体，作者式Block128的历史producer与编译selector尚未独立完整绑定。Chipmunk论文视频主表只启用attention，不能用其整系统速度代表Wan FFN基线。新报告同时区分必要适配、固定预算选择、局部风险、历史视频质量和逻辑缓存，并确认官方DMD2推理标签及checkpoint SHA。新增模型运行0；现有A/BW/BWS完成状态不变。

**2026-11-15 集中科学分析：[方法差异与函数结构](../../../research/20261115/G20_METHOD_DIFFERENCES_AND_FUNCTION_STRUCTURE_20261115.md)。** 合并旧完整视频、两种 Chipmunk 主实现及分组消融、ICC 两阶段误差与 A/BW/BWS，重新取回七源十臂原始讨论包的逐源质量汇总。完整周期成本勘误：Global-P64 与 Q1 的主 MAC 比为 0.685690/0.685185，近似相同；原先 2× 仅指命中步。新增模型运行为 0。

**当前判断：三个固定比较全部完成并发布——A 的固定残差比较已完成，保留可折叠线性修正 L；BW 的三方法比较已完整，Chip 局部误差最低；其命中步主 MAC 是 Q1 的 2 倍，完整锚点—命中周期主 MAC 与 Q1 接近；BWS 已完成并发布（job 166225 EXIT 0，216 个 DEV 单元），方向 B 的四格解释已完成：权重切换影响很小，复合原生状态大幅改变误差尺度与 ICC 的相对差距。** 本次复核基于提交 `20b3ffd48eb638d01daa19a0053097756212fdbd`，从已发布的 BWS 216 行原始表独立重算四格、状态差和交互；A/BW 沿用已核验结果。更新已有索引、[资产与队列](../../../research/20261113/G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)和[判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)，新增计算作业为 0。各实验冻结目录内的源码、输出、审查与收据保持原身份；当前状态与资源勘误由本索引及现有资产/判断台账维护。

| 阶段 | 当前证据与剩余工作 |
| --- | --- |
| A_Q1_RESIDUAL | 已公开完整结果；L 的 DEV 风险较其 Q1 基线改善约 4.07%，当前无需追加候选或扩层 |
| BW | 已公开全部 396 个 DEV 单元：Q1/ICC 264 个原值复用，加 132 个实际 GLOBAL_P64 Chip 单元；评分已完成 |
| BWS | `COMPLETE`：job 166225 EXIT 0、216 个 DEV 单元（144 Q1/ICC + 72 Chip）；原始产物已发布并独立复算，权重＋状态统一解释见下节 |

A/BW 均使用现有 VBench 数据。先对三个层等权，再对 high/low 等权，最后对 11 个 DEV source 等权；独立统计单位为 source，n=11。DEV 未参与本次拟合，但此前已被观察，属于开发集。结论限于六个站点、固定 512 行局部回放；视频质量与实际方法时延未测。BWS 为单独的 CAL3/DEV3 设计，不能与 BW 的 CAL11 表拼接。

## BW 已完成：Global-P64 局部风险更低，完整周期主 MAC 与 Q1 接近

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

| 方法 | 命中步 analytic MAC/token | 相对 Q1 命中步 | cache bytes/token |
| --- | ---: | ---: | ---: |
| ICC r512 | 19,398,656 | 0.74× | 37,888 |
| Q1 | 26,214,400 | 1× | 10,240 |
| Chip K5120 | **52,428,800** | **2×** | 37,888 |

按 [BW_COST](BW/v3/output/BW_COST.csv) 的公式与实际 [Chip packed-pregate 实现](BW_CHIP_SCORE/v2/source/chipmunk_wan_packed_pregate.py)，Chip 两个选定分支各为 5120×5120，命中步主体 MAC 合计为 Q1 的 2 倍、ICC 的约 2.70 倍。这一行不包括锚点成本，也还没有计入 64 行 proxy、排序、权重 packing 与 cache clone；整次评分的约 345.8 秒不是任何单一方法的原生时延。旧 BW_COST 中 Chip 的 blocked 标记只描述当时评分状态，成本公式沿用。

**完整周期需要补上 Q1 的锚点投影。** Q1 在锚点计算 `C0=F(u0)-u0@A`，因此一个 anchor+hit 的主成本为 `2dm+2d²`；Global-P64 的锚点直接缓存 native h0/y0，K=d=5120 时同一对主成本也是 `2dm+2d²`，另加当前64行代理。对 N=31668 的 HHLL，每专家一个 anchor+hit：

| 实际路径 | 锚点主 MAC/token | 命中主 MAC/token | 完整周期 / Dense |
| --- | ---: | ---: | ---: |
| Q1 单残差 | 141,557,760 + 26,214,400 | 26,214,400 | **0.685185** |
| Global-P64 K5120 | 141,557,760 | 52,428,800 + proxy | **0.685690** |
| ICC r512 两残差 | 141,557,760 + 19,398,656 | 19,398,656 | **0.637037** |

这纠正了此前用命中成本解释整体预算的写法。Global-P64 的局部回放风险较低；按完整 N=31668 的锚点—命中周期计算，其主 MAC 与 Q1 接近。512行回放及其独立P64端点重算不是完整部署成本的计时证明；缓存、代理、packing、GEMM 形状和真实时延仍不同，当前局部回放也未建立视频质量优胜。上述完整成本与[实际 G19 缓存路径](../../../research/20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md)及[历史完整视频账本](../../20261115/g20_concentrated_analysis/LEGACY_VIDEO_EXTERNAL_SUMMARY.json)一致。A 的 L 可折叠进 Q1、没有新增在线 GEMM，仍有研究价值。**A 的约 4.07% 与这里 Chip 的约 22.8% 使用不同 Q1 基线及目标生成口径，不能直接排成同一候选胜负表。**

### 实现与归档核验

当前评分采用已声明的 `OWNW_LOCAL512_P64ANCHOR64_V1`，不改称与历史 full-M GEMM 逐位等价。P64 manifest 包含 **132 个独立 anchor/hit payload**，同一批文件在两权重下共加载、校验 **264 次**；不能写成 264 个独立 payload 或样本。

本次新档案的 PUBLICATION manifest **35/35**、SOURCE manifest **14/14**、scope 代码 **13/13**及 B_BINDINGS hash 均与仓库实际字节匹配；RUN/EXIT 与输出身份相符。旧批次 CRLF→LF 的归档勘误仍保留其历史身份。[发布清单](BW_CHIP_SCORE/v2/PUBLICATION_MANIFEST.json) · [源码清单](BW_CHIP_SCORE/v2/source/SOURCE_MANIFEST.json) · [冻结 scope](BW_CHIP_SCORE/v2/evidence/scope.json)。

仓库记录 V1 job **166419** 又因 BF16 tensor 直接 `.numpy()` 散列而失败，原失败日志未在本批公开；V2 恢复了 BW/v3 已验证的 uint8-view 实现，未改变预测、目标或风险定义。此问题的修复应直接复用已验证函数，并在新输出路径做一次真实 dtype 的紧凑检查；不增加 pilot、复审层级或科学重跑。[记录](BW_CHIP_SCORE/v2/README.md) · [自审](BW_CHIP_SCORE/v2/evidence/ROOT_SELF_AUDIT.txt)。

## BWS 已复核完成：原生状态改变方法差距，权重切换影响较小

BWS job **166225 EXIT 0** 已公开 **216 个 DEV 单元**（144 Q1/ICC + 72 Chip）。CAL3 为 vbi_1005/0895/0648，DEV3 为 vbi_1013/0886/0800；每套权重在每个站点上用 CAL3 混合 D256+O256 拟合一次，保存并重载的参数跨两个状态固定。**独立统计单位为 3 个来源，216 是嵌套测量单元数。**

这里的 mu_D/mu_O 分别来自 D4/O40 原生轨迹，包含步间隔、CFG 和轨迹等条件差异。两状态均使用同源、同站点的 **D4 hit gate 作为共同风险度量**；每套权重重算自己的 anchor/target。所测量的是固定参考 gate 下的局部 FFN 误差，完整视频质量对应自由生成轨迹及其自身门控。BWS 的混合 CAL3 拟合与 BW 的 CAL11 不同，两表的 D4 数值保留各自身份。

### 四格结果与状态差距

主风险为 BF16 门前误差提升至 F64 后乘共同 gate 的 MSE，按层、expert、来源依次等权；百分比为来源等权平均风险之比。

| 权重 | 原生状态 | Q1 风险 | ICC 风险 | Chip 风险 | ICC 相对 Q1 超额 | Chip 相对 Q1 改善 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W_O | D4 / mu_D | 0.015418117967 | 0.020118909918 | 0.010994489087 | +30.49% | 28.69% |
| W_O | O40 / mu_O | 0.007277827396 | 0.008344406327 | 0.005087690929 | +14.66% | 30.09% |
| W_D | D4 / mu_D | 0.015414242606 | 0.020127472867 | 0.010962878305 | +30.58% | 28.88% |
| W_D | O40 / mu_O | 0.007278507932 | 0.008347844793 | 0.005087379218 | +14.69% | 30.10% |

依据：[原始单元](BWS/v1/output/BWS_CELL.csv)、[原始聚合](BWS/v1/output/BWS_AGGREGATE.json)。固定 W_O，从 D4 换到 O40 后，Q1/ICC/Chip 的风险分别下降 **52.80% / 58.52% / 53.73%**；W_D 下方向与尺度接近。**ICC 的总体相对劣势从约 30.5% 缩小到 14.7%；Chip 的总体相对优势保持在约 29%–30%。** 排序保持，方法间差距的大小仍随复合状态变化。

令 `d_m(W,mu)=R_m(W,mu)−R_Q1(W,mu)`，状态差取 `S_m(W)=d_m(W,mu_D)−d_m(W,mu_O)`：

| 方法 | 权重 | d(mu_D) | d(mu_O) | S：D4 减 O40 |
| --- | --- | ---: | ---: | ---: |
| ICC | W_O | +0.004700791951 | +0.001066578931 | **+0.003634213020** |
| ICC | W_D | +0.004713230262 | +0.001069336862 | **+0.003643893400** |
| Chip | W_O | −0.004423628880 | −0.002190136468 | **−0.002233492413** |
| Chip | W_D | −0.004451364301 | −0.002191128713 | **−0.002260235587** |

**状态差是本次方向 B 的实测发现。** 当前两个原生状态条件之间的方法差距变化，远大于在同一状态下换这两套 FFN 权重的变化。步间隔、CFG 和轨迹各自的贡献仍未单独识别；当前结果已完成既定复合状态比较，这些未识别分量不自动形成新增采集任务。

### 权重效应与交互的尺度

`theta_m(mu)=d_m(W_D,mu)−d_m(W_O,mu)`；`I_m=theta_m(mu_D)−theta_m(mu_O)`。

| 方法 | theta(mu_D) | theta(mu_O) | I：权重×状态交互 |
| --- | ---: | ---: | ---: |
| ICC | +1.243831081e−5 | +2.757930623e−6 | **+9.680380188e−6** |
| Chip | −2.773542058e−5 | −9.922458606e−7 | **−2.674317472e−5** |

交互绝对值分别约为两权重平均状态差绝对值的 **0.27% / 1.19%**。在这三个 DEV 来源中，平均交互相对已观察的方法差和状态差较小；这是效应量比较，未进行带预设等价界的总体等价检验。ICC 的逐来源 I 均正、Chip 均负；high/low 的交互方向则相反。[逐源表](BWS/v1/output/BWS_SOURCE.csv)

### 专家分解与局部例外

下表来自同一原始单元表，取 W_O；W_D 方向一致。

| Expert | ICC 相对 Q1 超额：D4→O40 | Chip 相对 Q1 改善：D4→O40 |
| --- | ---: | ---: |
| high | **26.97%→13.44%** | 29.97%→30.78% |
| low | 53.66%→53.69% | **20.28%→8.03%** |

ICC 总体相对差距缩小主要出现在 high；low 的相对超额接近不变。Chip 总体优势接近，但 low 的优势从约 20% 缩小到 8%。虽然汇总时 high/low 风险等权，先合并风险再求比值会使绝对风险较高的 high 对总体百分比影响更大。

**全部 12 个来源×权重×状态均值均为 Chip < Q1 < ICC。** 细分到 72 个来源×站点×权重×状态配对，Chip 优于 Q1 为 **70/72**，ICC 劣于 Q1 为 **56/72**，完整排序为 **54/72**；来源均值结论保留这些局部例外。

### 实现、归档与记录解释

从 216 行重算与发布聚合的最大差为 **1.74e−18**，与逐源表的最大差为 **3.47e−18**。36 个来源×站点×状态组内，三方法、两权重的 U0/U1、panel 与 gate 身份一致；own anchor/target 在两权重间均不同。12 组 Q1 参数跨状态和 DEV 来源固定，12/12 拟合收敛。公开发布清单 **48/48**、源码清单 **17/17**、scope 源码 **15/15** 均与归档字节一致。[源码](BWS/v1/source/bws_runner.py) · [检查](BWS/v1/output/CHECKS.json) · [RUN](BWS/v1/evidence/RUN.json) · [EXIT](BWS/v1/evidence/EXIT.json) · [发布清单](BWS/v1/PUBLICATION_MANIFEST.json)

参数12份约1.606 GiB与其余集群资产由清单绑定；本次没有重新下载大张量。下面三项为原记录的解释与勘误，保留冻结源码、CSV及README原文：

- **端点标签：** CSV 的 `endpoint_steps` 沿用 D4 site schedule，54 行 mu_O/low 标为 2→3；实际 O40 low 输入来自 **15→16**（timestep 892→882），high 为 **0→1**（999→994）。实际消费按 sidecar 的 anchor/hit 角色选择。该标签已在原审查与作者答复中登记，解释 O40 时间关系时使用真实 sidecar。[原答复](BWS/v1/evidence/AUTHOR_RESPONSE.json) · [端点记录](BWS/v1/output/O40_CAPTURE_vbi_1013.json)
- **P64 内容：** 72 份 O40 payload 都含 PANEL512；其中 **36 份 DEV payload**另含真实固定64输入，36份CAL不需要P64。冻结README将72份统称含P64的表述据此限定。[捕获源码](BWS/v1/source/b_capture_o40.py) · [资产清单](BWS/v1/output/CLUSTER_ASSETS_MANIFEST.json)
- **视频解码：** 正式日志证明 **6 次生成阶段 VAE 解码及6个MP4写出**；零计数指事后未读取MP4做帧解码和质量评分。公开日志直接证明正式6条O40；另2条重复smoke由记账与评议支持，未单独公开原始收据。总计8次、超过原6次上限的偏差保留。[正式日志](BWS/v1/evidence/slurm-166225.err) · [自审](BWS/v1/evidence/ROOT_SELF_AUDIT.txt)

## 两方向的本轮结论

**A 已得到可用的局部改进候选：** 固定 Q1 上的线性残差修正 L 在 DEV11 改善约4.07%，可折叠进原矩阵。已测两种非线性修正均未超过 L，本轮固定算子比较完成。

**B 已把两类原因分开：** BW 的共同 D4 状态与 BWS 的四格比较都表明，当前两套 FFN 权重之间的切换对平均方法差影响较小；BWS 则显示原生状态组合大幅改变误差尺度，尤其改变 ICC 相对 Q1 的总体差距。Chip 在两个状态下保持较低的来源均值风险；其命中步主体 MAC 是 Q1 的2倍，ICC 的命中步约为Q1的74%。按完整 N=31668 的锚点—命中周期，Global-P64/Q1的主MAC比Dense分别约0.685690/0.685185，ICC双残差为0.637037。局部误差、完整周期预算与缓存代价应分别比较。

这轮结果支持将“ICC 与 Chip 在蒸馏设置中一起失效”改为分别解释。ICC 的相对差距对本次原生状态条件更敏感；Chip 的总体相对优势接近，low 专家仍有明显变化。BW 的CAL11与BWS的混合CAL3独立解释，A的4.07%也不与两者的Chip百分比组成同一胜负表。当前范围是六站点局部回放与已观察DEV来源；视频质量和真实方法时延保留各自尚未测量的状态。

**三个固定比较及本轮总结均已完成。** 现有输入、参数与原始风险表直接复用；当前没有未完成的采集、拟合、评分或收尾报告任务。

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

## BWS 交互的有界解释（发布前登记的历史说明）

BWS 最初以 `REPORTED_EXIT0_PUBLICATION_PENDING` 登记于[既有判断台账](../../../research/20261113/G20_VBENCH_DECISION_LEDGER_20261113.json)的 `execution_updates_20261114.two_core_20261114.BWS`（job **166225 EXIT 0**、CAL3/DEV3、两权重×两状态×三方法、216 个 DEV 单元）。本次提交已发布其 scope/RUN/EXIT、参数与输入 manifest、216-cell 及逐源/聚合表并独立复算（见上节“BWS 完成”与[发布清单](BWS/v1/PUBLICATION_MANIFEST.json)）；原“Publication and direction-B closure deferred per user”暂缓记录就此解除，不再将已完成实验重新加入执行队列，也不意味着再申请研究目标许可。

**不能只凭小交互值把方向 B 宣布为“权重和状态都无影响”。** 令 d_m(W,μ)=R_m(W,μ)−R_Q1(W,μ)，现有交互定义为：

$$
I_m=[d_m(W_D,\mu_D)-d_m(W_O,\mu_D)]-[d_m(W_D,\mu_O)-d_m(W_O,\mu_O)].
$$

即使 I_m 很小，两个权重下仍可共同存在较大的状态主效应 `d_m(W,μ_O)−d_m(W,μ_D)`（BWS 实测 mu_O 绝对风险约为 mu_D 一半）。四格绝对风险、每格方法差、各权重状态主效应、交互及逐源/expert 差异均须从**同一 BWS 混合 CAL3 拟合**读取；不得用 BW 的 CAL11 θ 与 BWS 的 CAL3 交互拼接反推缺失格，也不把原生 O40/D4 的复合差异单独归因于步数、CFG 或某一个状态因素（纯分量 UNIDENTIFIED）。

O40 消耗 **6 条正式 + 2 条 smoke = 8 次执行**，覆盖 6 个不同来源；两个 smoke 来源 `vbi_1005/vbi_1013` 被重采、未复用，超过原 6 次总上限 2 次。该偏差已随本次发布归档核验（[slurm-166225.err](BWS/v1/evidence/slurm-166225.err)、[verify_post_bws.log](BWS/v1/evidence/verify_post_bws.log)）；纠正方式是保留真实消耗与结果、在今后 producer 中落实 reuse-first，不通过再跑 BWS 修复已发生的偏差。

## 已完成资产、正式作业与当前顺序

| 资产 | 当前消费入口 |
| --- | --- |
| A 参数与结果 | [参数清单](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_PARAMS.json)、[READOUT](A_Q1_RESIDUAL/v1/output/A_Q1_RESIDUAL_READOUT.json)、[EXIT](A_Q1_RESIDUAL/v1/evidence/EXIT.json)；完整复用 |
| BW Q1 参数、ICC 因子与 264 个分数 | [参数清单](BW/v3/output/PARAMS_MANIFEST.json)、[BW_CELL](BW/v3/output/BW_CELL.csv)、[EXIT](BW/v3/evidence/EXIT.json)；完整复用 |
| P64 固定输入 | [132 份 payload 清单](B_CHIP_STATS/v1/output/B_CHIP_P64_INPUTS.json)、[EXIT](B_CHIP_STATS/v1/evidence/EXIT.json)；已被 Chip 评分消费 |
| BW Chip 132 个分数 | [BW_CHIP_CELL](BW_CHIP_SCORE/v2/output/BW_CHIP_CELL.csv)、[SUMMARY](BW_CHIP_SCORE/v2/output/SUMMARY.json)、[EXIT](BW_CHIP_SCORE/v2/evidence/EXIT.json)；固定 selector，无新增 CAL fit 资产 |
| BWS 状态、参数与 216 个分数 | `COMPLETE`（job 166225 EXIT 0）；[BWS_CELL](BWS/v1/output/BWS_CELL.csv)、[SUMMARY](BWS/v1/output/SUMMARY.json)、[参数清单](BWS/v1/output/PARAMS_MANIFEST.json)、[集群产物清单](BWS/v1/output/CLUSTER_ASSETS_MANIFEST.json)、[EXIT](BWS/v1/evidence/EXIT.json)、[源码清单](BWS/v1/source/SOURCE_MANIFEST.json)；已发布并独立复算 |

本阶段已登记的正式 job 至少 **8 次：成功 5、失败 3**。**5 个成功作业均有公开 EXIT**（A 164072、BW 164312、P64 164928、Chip 166479、BWS 166225）；三次失败登记为 BW 164189/164225 与 Chip 166419。该计数是已登记下界，排除 smoke/verification，不断言没有其他未公开的尝试。详细证据层级保存在现有 ledger 的 `progress_check_chip_score_20261115` 与 BWS 发布记录。

**历史已闭合范围：三个固定比较（A_Q1_RESIDUAL、BW、BWS）及[交接手册](../../../research/20261114/G20_TWO_CORE_EXECUTION_HANDOFF_20261114.md) §7 的两方向总结均已完成，该批范围无剩余待办。** 不再派生输入复审、独立 smoke 或参数网格，也不重新提交要求批准两个方向的泛化计划。A、BW、P64 采集、BW Chip 评分与 BWS 直接复用；BWS 不由 theta 的符号决定是否执行。此前的预算与具体 scope 保持原身份；本次维护新增作业为 0，不新增执行授权或审批层级，也不增加图文、CAL_B、all80 或视频评分。

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

