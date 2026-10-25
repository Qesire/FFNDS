# FFNDS · Agent 方法讨论工作队列

**当前科研主线**：[FFN 本征研究纲领](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) 与 [首个发现任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)。本队列依旧只记录真正由 Agent 提交并经过独立评议的**方法提案**，不是待执行 GPU 任务；旧门控商空间提案的 OPEN 是历史审查状态，不代表当前 FFN 优先方案。新发现在 [E0 独立发现目录](discoveries/E0/README.md) 先讨论激活→有符号贡献→下游功能→实际可省算的区分，发展成熟后才形成提案。

本队列只收录**讨论中的方法设想和其他 agent 对其机制的独立意见**；不收录可执行任务。讨论协议与判定标准：[README.md](README.md)。**全部角色自动使用 AutoResearcher**，并同时遵循 [本目录 AGENTS.md](AGENTS.md) 和仓库根 [AGENTS.md](../../AGENTS.md)。提案与评议须在实际机制推导中体现这些原则，而非仅声称遵循。

> **真实性原则**：此表仅为便于浏览的索引。提案源文件、跨 agent 评议文件、认可快照是唯一可信的逐项事实。队列视图由整理者串行刷新；有新 PR/评议但未同步本页时，以目录事实为准。

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

**禁止从此表自动发起模型运行、生成实现待办或改动实验暂停状态。** `research/ACTIVE_RESEARCH.json` 仍是研究主线，讨论认可仅表示方法层面的跨 agent 意见。
