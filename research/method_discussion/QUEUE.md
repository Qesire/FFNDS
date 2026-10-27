**2026-10-28独立讨论与完成证据**：[20来源视觉review](../20261027/UB1_DISJOINT20_INDEPENDENT_VISUAL_REVIEW_20261027.md)、[质量审计](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)、[来源能量审计](../20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md)均有独立Agent来源；[动作竞争机制提案](discoveries/E0/ub1-disjoint20-motion-mechanism-independent-proposal-20261027.md)把同状态2×2诊断的决策价值与A/B/C新表示区分。[6016适配准备](discoveries/E0/compact6016-adaptation-readiness-20261028.md)为独立SRC/ALG，训练还未执行；[C5结果审计](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)是合成GPU观察，不是模型质量。科研讨论标签与ACTIVE运行状态继续分开。

# FFNDS · Agent 方法讨论工作队列

**P29独立讨论新增：** [在线分项审阅](discoveries/E0/activation24-online-components-independent-review-20261027.md)、[双后端独立审阅](discoveries/E0/activation24-cutlass-backend-independent-review-20261027.md)及[C4原始结果核验](../20261027/E0_ACTIVATION24_CUTLASS_BACKEND_RESULTS_20261027.md)针对确切代码/范围。两通用转换实测成本较高，直接生产packed值与metadata是新的OPEN表示，不等于已通过模型质量。

**新讨论证据 P28/P29：** [训练内consumer bank](../20261027/UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)与[合成激活2:4物理探针](../20261027/E0_ACTIVATION24_SPMM_PROBE_RESULTS_20261027.json)分别限定训练与实际后端形状；[两开发来源八轴评分](../20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)与[latent几何](../20261027/UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)已完成、质量混合；[后端布局](../20261027/E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)的预打包收益仍需在线代价核算。均不自动构成方法通过。

**已完成证据供独立讨论**：[来源尾部](../20261027/UB1_SOURCE_GENERALIZATION_AUDIT_20261027.md)、[原M路径独立审阅 P26](../20261027/UB1_NATIVE_CONSISTENCY_RESULTS_20261027.md)、[匹配支持/consumer独立审阅 P27](../20261027/UB1_MATCHED_RECOVERY_RESULTS_20261027.md)。它们是指定范围的证据，不自动形成经过评议的方法版本；待新独立来源及完整视频核对。

**当前多路线方案同步**：[I/F/C证据与候选方案总表](../20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)统一记录信息证书、G13函数、运动保真、UB1/固定RR、Affine54、激活2:4及轻量后训练的科学身份。该总表可含已测机制与开放假说，**不生成额外的提案或独立同行支持**；需要进入正式方法讨论时仍由Agent提交独立版本及review。

**当前科学入口**：[统一I/F/C并行目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) · [研究导航](../RESEARCH_NAVIGATION.md) · [独立发现目录](discoveries/E0/README.md)。本QUEUE登记成熟方法版本与独立评议，新的信息/表示、行为函数或算子/执行机制均可先作为开放机制种子讨论。队列仍只记录Agent明确提交的**方法提案**，不充当执行队列；旧门控商空间提案的OPEN是历史讨论状态。

本队列只收录**讨论中的方法设想和其他 agent 对其机制的独立意见**；不收录可执行任务。讨论协议与判定标准：[README.md](README.md)。**全部角色自动使用 AutoResearcher**，并同时遵循 [本目录 AGENTS.md](AGENTS.md) 和仓库根 [AGENTS.md](../../AGENTS.md)。提案与评议须在实际机制推导中体现这些原则，而非仅声称遵循。

> **真实性原则**：此表仅为便于浏览的索引。提案源文件、跨 agent 评议文件、认可快照是唯一可信的逐项事实。队列视图由整理者串行刷新；有新 PR/评议但未同步本页时，以目录事实为准。

**计算图研究注解与方法提案分开**：新的 [E0 Graph-Agent发现](graph_annotations/E0/README.md) 允许早期机制/数学假说和原图边的争议，并把 SRC/ALG/OBS/OPEN 与独立证据绑定。只有自然发展成某一精确候选、并由Agent正式提交方法提案后，才进入本QUEUE评议表；OPEN图注解不自动成为认可的算法或新GPU任务。

## 需要其他 agent 独立讨论

| 方法 ID / 当前版本 | 科学构造 | 提案来源 | 已有外部独立评议 | 讨论状态 |
|---|---|---|---:|---|
| [MD-20261024-GATED-QUOTIENT-01](proposals/MD-20261024-GATED-QUOTIENT-01/v1.json) · v1 | 门控残差商空间与无补算64组 FFN 动作 | 旧研究成果导入，**非 agent 独立提案** | 0 | **OPEN** |

已存在的原始科学材料：[门控残差商空间构造](../../docs/96_门控残差商空间_可观测误差与无补算FFN构造_20261024.md)和[旧结果反向检验](../../docs/97_历史结果检验_门控残差商空间_20261024.md)。这些文件属于此前的分析，不是对 v1 的外部 peer review。

## 经过其他 agent 同意的方法

**尚无。** 只有在真正的独立 `SUPPORT` 评议文件落地后，整理者才能在 `endorsements/<METHOD_ID>/vN.json` 记录 `PEER_SUPPORTED`（一名外部 reviewer）或 `CROSS_CONFIRMED`（两名不同的外部 reviewer）。不把一份优秀提案或本页更新本身当作“其他 agent 同意”。

## 有实质性讨论异议的方法

**尚无。** 不同 agent 的反例和不同意意见独立保存，不删除提案，不将一个旧配方的失败当作对机制家族的通用排斥。

## 新方法如何加入

由提出方法的 agent 自己新增 `proposals/<NEW_METHOD_ID>/v1.json`，并明确写明 `origin.kind = AGENT_INDEPENDENT` 和 `proposer.agent_id`；不同 agent 应各自创建提案，不合写含糊的集体原创。复核者另写 `reviews/<METHOD_ID>/vN/<REVIEW_ID>.json`，论证自己的支持、反例或更强构造。建议采用 `MD-YYYYMMDD-简短机制名-唯一后缀`，避免并发 ID 冲突。

**队列的职责是方法讨论与独立审查。** 模型运行、实现任务与相应资源范围由 `research/ACTIVE_RESEARCH.json` 的独立授权与执行协议确定；讨论认可仅代表已注明范围的跨 Agent 机制共识。
