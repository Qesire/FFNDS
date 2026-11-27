# 第 39 层联合分叉进度

## 最新复核：完整S128对照、四臂消融与四步终点

[独立复算与研究判断](analysis_20261127/paired_value_transport/result_review/NEW_RESULTS_REVIEW.md)核对学校204479/204487/204740：现有centered在固定路由、固定P64、等权PFRR对照中取得低噪恢复优势；新增完整value矩阵在固定配方下未形成稳定增量。已独立复算216个Gram、780个归档数值及四来源完整轨迹指标；对角与配对的前三步一致，最终latent/Dense为1/4来源改善、RGB为0/4。保留centered工作参照，原片可访问链接与完整SparsePR系统比较仍待补齐。本次复核新增GPU运行为0。

## 学校实测：配对 value 的端到端复核

学校 A100 作业 204740 生成三条开发来源及一条留出来源的完整候选配对，核对最后一步 flow、最终 latent 与 81 帧解码视频。局部消融与端到端结果一起解释，视频及指标指纹见[端到端比较](analysis_20261127/paired_value_transport/experiments/E2E_REPORT.md)。

## 学校实测：SparsePR v2 恢复对照与 value 配对消融

[完整结果](analysis_20261127/paired_value_transport/experiments/PAIRED_VALUE_REPORT.md) · [学校收据与复算](analysis_20261127/paired_value_transport/experiments/REPRODUCE_EXPERIMENTS.md)。三个开发来源的 12 个 low 站上，固定 P64 的全通道 PFRR 相对历史 OUT-D 的 head 和 gated point 误差能量比为 4.8837 和 3.9859。同位置配对在固定轨迹上优于打乱配对，C 相对对角 B 的恢复后 attention 和同输入 Dense 块出口误差能量比分别为 1.2154 和 1.0841。局部结果作为机制诊断，候选终点与实际成片另见完整轨迹复核。

## 最新分析：配对 value 通道运输与 SparsePR v2 比较信心

[机制、证据与最小对照](analysis_20261127/paired_value_transport/FRONTIER_MECHANISM_AND_CONFIDENCE.md)从真实旧指标复算得到：wide8192 低噪历史 OUT-D 对受限 CAL32 当前恢复的 gated point 几何平方误差比为 0.3835，三个开发来源的十二站均改善；该对照尚非 SparsePR v2 完整通道恢复。提出用少量同位置 anchor/current value 行识别每头通道矩阵，再更新历史响应，P64 恢复剩余创新；给出精确类、三项误差分解和真实非线性后继判据。附 CPU 参考、旧数据复算、近邻原始来源核查及真实 Wan 尺寸成本。新增 GPU 执行为 0；当前默认运行方法保持。

## 上一轮分析：与 SparsePR 的核心重合和历史信息区别

[逐算子核查与等价性推导](analysis_20261127/sparsepr_overlap/SPARSEPR_CORE_OVERLAP.md)明确：当前high OUT-ABS在实数下属于SparsePR当前sparse输出仿射恢复族的受限输入特征变体；low则在全位置运输底座和跨步字典两处引入历史。推导了同一恢复器下的等价条件及实际attention的区分构造，并核对SparsePR v2自身也具有条件value仿射等变性。附v1/v2原文、五文件固定源码身份与CPU核验。新增GPU执行为0。

## 上一轮分析：与 Chipmunk 等方法的机制和成本比较

[方法得失与复算](analysis_20261126/method_comparison/METHOD_COMPARISON_AND_TRADEOFFS.md)核对 Chipmunk、ICC、SparsePR、MC-Sparse 的原文与固定源码，并重算11开发来源的495条历史指标。当前wide8192、历史ICC r1408和Chip Author Safe的完整FFN周期主算量均约为原生的69%，预算分别投向完整线性项与输出恢复、低秩增量、动态选列。centered与静态历史残差的精确差式和条件value尺度运输已在CPU核验。当前完整C对同四步Dense的1.156×请求加速、历史FFN质量表及作者多步系统倍率保留各自范围；完整当前版本与Chipmunk attention的同批质量/时延比较待执行。报告另列attention-only简化的约7.0个百分点主算量代价与30.2GiB FFN历史槽预算。本轮新增GPU推理为0。

## 上一轮分析：理论上限、实测加速与方法核心

[完整成本与合并指标](analysis_20261126/speed_and_core/SPEED_LIMITS_AND_CORE.md)按真实执行图复算：已列主要矩阵算量约省28%；固定两锚点与现有模块边界的乐观算量参照约省37.02%。完整C相对同四步蒸馏Dense已测1.227× GPU、1.156×预热后完整请求加速；centered另有小幅增量。两条centered开发视频已获用户“基本没问题”的反馈。方法收束为结构预测与P64创新恢复，attention可等价重排为先P拟合、后一次分块输出。蒸馏叠加可以报告总系统加速，并单列同蒸馏骨干上的增量；原模型分母由匹配三臂对照确定。

## 上一轮分析：当前候选的状态表示、联合恢复与完整成本

[job 201500 的整体分析与独立复算](analysis_20261126/mass_centered_review/MASS_CENTERED_RESULT_ANALYSIS.md)核验 112 个真实 Gram 和 183 个非视频汇总数值。八处 attention、七处同输入 Dense 块出口改善；恢复及 BF16 写回后，H 上新旧底座更新保留 2.37%–19.49% 的能量。完整 low 周期四对均更快，来源等权比值 0.98549；整片墙钟 0.99397 保留运行顺序影响。实际峰值分配降低 1.2099 GiB，收益包含当前编译融合实现，历史槽容量相同。用户对两条候选开发成片的总体反馈为「基本没问题」。[人审记录](analysis_20261126/joint_action_review/mass_centered_runtime/MASS_CENTERED_HUMAN_REVIEW_20261126.json)

下一重点是当前固定候选的完整成片评价与独立来源配对确认。

## 最新：low OUT-D 的均值与尺度运输

学校 A100 作业 `201500` 已完成两来源固定轨迹和四组同卡成片配对。八个受测站点的拟合后注意力误差均下降，同块 Dense 出口七处改善。low 复用周期均值减少 1.45%，整片均值减少 0.60%（两开发来源的顺序平衡读数）。[报告、分层结果与运行收据](analysis_20261126/joint_action_review/mass_centered_runtime/MASS_CENTERED_REPORT.md)

## 上一轮分析：运输方向、联合读出与低噪缓存构造

[两来源联合作用的算子分析](analysis_20261126/joint_action_review/JOINT_ACTION_OPERATOR_ANALYSIS.md)独立复算 job 201276 的 256 个 Gram，归档汇总 1,924 个数值字段一致。尺度项与剩余创新的交叉作用解释层间差异；完整块读出及 Dense 背景明确了实际恢复目标。进一步从冻结 C 的非零遗漏质量推导[均值与尺度运输的低噪缓存](analysis_20261126/joint_action_review/MASS_CENTERED_LOW_CACHE.md)，当时完成代数和合成核验；候选随后已完成上方 job 201500 的真实运行。附真实数据复算、语义审计与任务方向可辨识判据。

## 最新：两来源的逐阶段联合作用

学校 A100 作业 `201276` 已复现两条固定轨迹，记录 16 个站点的逐阶段 Gram、同块 Dense 背景和路由／形状／尺度分解。第 0 层的收益在两来源和两专家复现，0392 的低噪第 39 层排序翻转定位到 self gate。[分析与复算](analysis_20261126/joint_action/JOINT_ACTION_REPORT.md)

## 最新：真实运输结果、联合读出与Dense直接对照（2026-11-26）

[更新算子skill后的整体分析](analysis_20261126/UPDATED_OPERATOR_ANALYSIS.md)复算了job198062与199260/199277：冻结C完整请求较Dense省13.52%、四步GPU省18.52%；标准化运输在两专家第0层的同特征拟合能量下降42.10%/43.67%；low39出现attention能量略增而真实wide8192块出口能量下降的同站点排序变化。

明确区分层等权几何平均、总能量和后继读出，并给出有限方向证书、目标可观测条件及一次hit复用的完整成本。附[运输复算](analysis_20261126/TRANSPORT_RESULT_AUDIT.md)、[Dense/C复算](analysis_20261126/DENSE_BENCHMARK_AUDIT.md)与[联合数学推导](analysis_20261126/JOINT_READOUT_DERIVATION.md)。本次分析无新增模型执行；两条新运输视频仍保留待人审状态。


## 最新：冻结 C 直对四步 Dense

学校 A100 已完成六源两轮同卡配对。冻结 C 的完整请求时间减少 13.52%，四步 GPU 时间减少 18.52%。VBench-I2V 的七项六源均分差绝对值均小于 0.31 个百分点。[计时、评分与收据](c_dense_baseline/C_DENSE_BENCHMARK.md)

## 最新：Wan 固定轨迹和两源成片已完成

学校 A100 作业 `198062` 核验了 0392 的原生固定轨迹、八个注意力站点、两处真实 wide8192 块尾部，并生成 0392 与 1013 的完整视频。人审和与冻结 C 的配对计时待完成。[实测与收据](method_transport/WAN_RUNTIME_RESULT_198062.md) · [实现及验收](method_transport/WAN_RUNTIME_IMPLEMENTATION.md)

## 最新：新方法原型与标准化缺陷运输（2026-11-25）

已形成具有完整输入、状态、教师与恢复规则的方法原型，并执行CPU数学参考。新的状态保存标准化attention缺陷，使用当前V的通道尺度还原，P64恢复路由与标准化value形状的剩余变化。

[方法说明与判断](method_transport/METHOD.md) · [参考实现](method_transport/reference_attention.py) · [执行结果](method_transport/reference_attention_verified.json) · [机制查新](method_transport/MECHANISM_NOVELTY.md)。

已证明并数值核验anchor/hit两时刻独立正仿射value等变性；源码推导共用P对后续wide8192拟合的保护。真实投影资产与合成激活分别记录。新Wan/GPU执行数为0；冻结C保留已有六来源成片反馈作为工作参照。

## 上一阶段：六源成片确认后的新推导（2026-11-25）

用户反馈：“6源基本没有差别。”[原话与比较范围](C_CONFIRM_HUMAN_REVIEW_20261125.json)已登记。冻结 C 现在具有六来源的视觉基本相当反馈与四步 GPU 时间减少13.48%的证据；完整请求统计仍保留保存长尾。

[本轮新推导与判断](NEW_DERIVATIONS_AFTER_SIX_SOURCE_REVIEW.md)给出一个完整候选：当前两组K/V正权摘要形成底座，保存完整逐位置缺陷，P64恢复有限创新。它同时明确未来读出如何决定充分状态，以及完整块联合缺陷的教师和shadow成本。

本轮新增真实计算包括历史四臂Gram的正交交互分解及CAL矩阵几何；符号与合成验证、实际尺寸预算分别标明。新增模型/GPU执行数为0。当前优先发展上述信息构造的论证，已有六源人审不再列为待办。

## 上一阶段：回到核心方法与完整证据（2026-11-25）

[本轮核心分析](CORE_METHOD_REVIEW.md)重新区分完整状态、廉价当前作用和稀疏真实观测：FFN 与 low OUT-D 可按有限创新组织；high OUT-ABS 是当前 P64 拟合的共享通道校正。附实际 CAL 矩阵的 QR/SVD、符号核验与真实模型尺寸下的代价计算。

**冻结 C 的六个新来源均减少四步 GPU 时间，C/A 为 0.865241；完整请求 C/A 为 1.031752，保留保存路径长尾。** A/C 峰值为 47.5671/65.4123 GiB，12 条完整视频已生成。该阶段曾待六对人审，现已收到上方记录的用户反馈。[六来源回执复算](CONFIRM_EVIDENCE_ANALYSIS.md)记录完整成本边界与统计单位。

[首步阶段换基分析](PHASE0_EVIDENCE_ANALYSIS.md)确认坐标已有实质变化，H448 误差与运动仅小幅变化；0392 人审确认仍无动作，按约关闭分支。当时优先完成六对视频的人审，现已进入上方的新推导。阶段 CAL、P、rank、λ 的进一步扫描不列为下一执行任务。C 继续作为工作参照。

## 历史阶段：C0 首步替换后的联合状态分析（2026-11-25）

C0 相对 C 的整片配对时间下降 9.246%，但本轮归档人审确认 0392 的 C0 运动失败，1013 视觉变化较小。**C 继续作为工作候选，优先进入独立来源的完整成片确认。** [本轮整体分析](C_SCHEDULE_REVIEW_AND_NEXT.md)解释首步 attention 如何共同构建 latent、FFN 锚点和路由，并给出固定预算、固定 λ 的一次阶段坐标检验。

[原始回执独立复算](C_SCHEDULE_NUMERIC_REVIEW.md)通过 172 项检查；[复算程序](recompute_schedule_evidence.py)、[机器可读结果](schedule_evidence_recomputed.json)、[读出公式符号核验](outabs_scale_identity.json)及[阶段 CAL 资产审查](PHASE0_CAL_ASSET_AUDIT.md)随附。本次分析新增模型/GPU执行数为 0。以下按原阶段保留历史记录。

## 上一轮：三臂成片人审与整体推进

用户本轮反馈：“结果已同步，三个方法的视频对比，肉眼几乎没有差别”。结合正式配对计时，C（high OUT-ABS / low OUT-D＋wide8192）相对A整片时间下降12.01%，四步GPU下降15.42%，成为下一工作候选。[人审后的整体判断](ONLINE_REVIEW_AND_NEXT.md)记录机制归属、完整成本、原始收据复算及只改变high step0的后续比较；[人审原话](HUMAN_REVIEW_20261124.json)保留实际反馈范围。

## 全层执行前的整体判断：进入结构化 attention 的完整闭环

[整体方法分析](GLOBAL_METHOD_SYNTHESIS.md)依据固定P64新实测，把推进对象放回现有high OUT-ABS/low FIN-C0＋wide8192的全层自身轨迹。已完成真实收据及奇异谱复算、在线统计的合成代数核验和完整周期静态算术。下一比较为原生attention＋wide、结构化候选及同专家输出坐标参考；另给出attention独立锚点时序的后续构造。

[在线接口与P覆盖等价](ONLINE_STRUCTURED_ATTENTION.md) · [数值辨析](NUMERIC_GEOMETRY_ANALYSIS.md) · [复算数据](observability_recomputed.json) · [成本算术](extension_cost_arithmetic.json) · [统计归约核验](online_statistics_algebra.json)。本次分析新增真实模型/GPU执行数为0，未将局部回放或静态MAC计为候选成片、实际速度。

## D176 后的机制判断

[回到最初问题：FFNDS 怎样扩展到注意力与组合计算](RETURN_TO_ORIGINAL_QUESTION.md)根据两源真实收据完成独立 CPU 复算。新的判断区分完整字典、固定 P64 当前读出的可达空间、按头/模块教师约束，并给出低噪声下原独立路径在当前 D176 字典之外的作用下界。下一构造保留有限变化与模块结构，用完整块/flow 评价共同作用。

[复算数据](joint39_round3_recomputed.json) · [复算程序](recompute_joint39_round3.py) · [实现语义](D176_IMPLEMENTATION_SEMANTICS.md) · [既有资产的 P 可达空间诊断](diagnose_joint39_observability.py)。统计复算与明确标记的[合成代数核验](observability_algebra_check.json)已执行。[学校 CPU 实测](D176_OBSERVABILITY_REPORT.md)随后补齐了八站点的 P 可达空间下限、奇异谱和三项误差分解，新增模型/GPU 执行数为 0。

最新实验是[两源完整报告](JOINT39_COMPLETE_REPORT.md)及[FFN 核与头间对比四组对照](D176_REPORT.md)。以下保留此前恢复和0392单来源阶段的原始记录。

---

> **恢复后已验收两源。** 0392 和 1013 的四条背景轨迹、八个末层站点均已核验。CPU 剩余误差和头间对比核对照已完成。[两源报告与收据](JOINT39_COMPLETE_REPORT.md)。下文保留恢复前的 0392 单来源记录。

按[联合计算图的更新顺序](../JOINT_COMPUTATION_ANALYSIS.md)执行。学校 A100 已核验 DEV0392 的 Dense 和 BF16 wide8192 两个原生背景、两个专家，共四个末层站点。作业 193423 已启动，学校 SSH 会话故障期间未能取回 DEV1013 收据。下表为单来源结果。

第 39 层固定每条背景轨迹自身的入块状态与 FFN anchor。对原生／候选 self-attention × 原生／wide8192 FFN 做四臂分叉，经过实际 cross、AdaLN、FFN 门控和 5120→64 head。wide FFN 系数先用原生注意力的 P64 拟合，再固定于四臂，同时单列候选注意力输入上的重拟合。两个带观测的基础渲染均复现了各自封存的四步轨迹和解码 RGB。

联合读出从高噪声的 SP／低噪声的 ANCH 注意力底座与 FFN 折叠底座开始。特征为常数项、31 个注意力坐标的跨头均值与 16 个便宜 FFN 坐标。它只用 P64 拟合一次 64 维 flow patch，λ=0.1。独立读出沿用已冻结的高噪声 OUT-ABS、低噪声 FIN-C0 与 wide FFN。两臂都对 P64 精确覆盖，在 H448 上评分。

## 已核验的 DEV0392

| FFN 背景 | 专家 | 联合 / 独立点误差 | H448 同特征表示下限 / 独立点误差 | H448 同正则 / 独立点误差 | 重新拟合 FFN 后的点误差变化 |
|---|---|---:|---:|---:|---:|
| Dense | high | 1.574× | 0.840× | 0.860× | -0.255% |
| Dense | low | 4.037× | 2.006× | 2.029× | -0.145% |
| wide8192 | high | 1.618× | 0.827× | 0.857× | -0.274% |
| wide8192 | low | 13.659× | 7.550× | 7.674× | -0.068% |

点误差是 H448 的 64 维 flow patch 平方误差之和。使用 H448 全标签的两列仅诊断固定 48 列特征的表达能力和 P64 估计差距。BF16 wide8192 的高噪声站点达到 0.827× 的表示下限，但 P64 联合估计为 1.618×。低噪声站点的表示下限仍为 7.550×，说明当前特征族在这个站点存在表示限制。重拟合项使用未覆盖的近似 attention P64 输入，属于单列的条件反事实；它与固定系数的交互项分别保存在收据中。

归档的两个来源、两个背景、两个专家共有八个 P64 检查。只查询 64 行、保留全部 31,668 个 keys 的原生 SDPA，相对全行原生注意力对应值的最大 L2 差异为 0.079%。主四臂标签来自原生全行轨迹的对应 P64/H448 行。0392 的 512 行局部头部回放与全行运行的对应值最大相对 L2 差异为 0.287%。

## 收据与执行代码

- [0392 四臂结果和有符号 Gram](RESULT_0392.json) · [分叉向量校验与 H448 表示诊断](VECTOR_AUDIT_0392.json)。
- [全键 P64 核验](P64_AUDIT.json) · [作业与资产指纹](STATUS.json)。
- [分叉执行](attention_joint39.py) · [P64 核验](attention_joint39_probe.py) · [独立向量验收](audit_attention_joint39_assets.py)。

目前是教师强制下的局部回放。候选自身轨迹、整片质量和加速仍待验证。

## 执行后的机制分析

[本轮机制与推进判断](INTERPRETATION_0392.md)对真实四臂统计做了独立CPU复算：原生FFN对attention扰动的反向响应、head的方向敏感性、联合交互Gram，以及当前48列表示的误差预算。下一步保留已有结构恢复，优先用既有向量比较不同底座的剩余联合缺陷，并检验保留head对比与FFN核信息的共享读出。

[机器可读复算](joint39_recomputed.json)、[复算程序](recompute_joint39.py)、[实现语义审查](IMPLEMENTATION_REVIEW.md)与[算子推导](JOINT_REPRESENTATION_DERIVATION.md)随附。[CPU剩余缺陷重放脚本](replay_joint39_residual.py)已准备，需要学校封存向量；本地只检查了语法，未将该脚本的预期结果计为实测。

[学校 40 层在线比较的启动与验证](ONLINE_START.md)。

[四条完整候选、逐步状态与视频配对](ONLINE_QUALITY_REPORT.md)已验收。

[同卡配对周期、显存与组件时延](ONLINE_TIMING_REPORT.md)已核验 18 次正式运行。

C0 只改变原 C 的高噪声第 0 步 attention。学校 A100 已完成两源成片与十二次同卡计时。 [C0 配对结果与收据](C_SCHEDULE_REPORT.md)

## 冻结 C 的独立确认与首步阶段适配

[C 的六来源配对](C_CONFIRM_REPORT.md)已完成 12 条完整视频和同卡 warm 计时。整片时间比 C/A 为 1.0318×，四步 GPU 时间比 C/A 为 0.8652×，整片统计保留保存阶段的长尾。逐组盲审待完成。A 和 C 都使用 BF16 wide8192 FFN。

[C0-phase0CAL 的 0392 首片](PHASE0_CAL_FIRST_0392.md)已生成。首步换基后的光流代理为 0.0389 px，旧 C0 为 0.0385 px，冻结 C 为 0.4418 px。人工复核确认 0392 仍然没有动作，按方案结束这条首步分支。
