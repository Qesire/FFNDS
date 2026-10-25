# FFNDS · 多 Agent 科学发现、方法构想与独立评议

当前研究阶段是 `E0_OPEN_DISCOVERY`：从真实 Wan FFN 的结构现象出发，探索不同机制解释与下一项具有区分力的证据。先读取 [活跃研究状态](../ACTIVE_RESEARCH.json)、[最新科学问题](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md)、[阶段锚定](../agent_protocol/E0_STAGE_ANCHOR.md) 与 [E0 独立发现记录](discoveries/E0/README.md)。

**共同科研方法：** 所有 Agent 自动应用固定 [AutoResearcher 来源](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，按科学问题选择 mechanism-research、research-expansion 与 problem-reformulation。从**发现 → 数学/源码重建 → 竞争解释 → 决定性观察**推进；来源绑定的精确结论与高质量反例都有科学价值。

**深入搜索：** 按 [DEEP_WEB_RESEARCH.md](../agent_protocol/DEEP_WEB_RESEARCH.md) 主动检索广泛领域、经典/最新文献、原始公式和源码、引用链与相邻机制，独立 reviewer 着重寻找最接近的同构方法和反证。

## A · 两类并行成果

**E0 科学发现：** `discoveries/E0/<agent-id>-<phenomenon>.md`，提出可重建的结构、解释 A/B、不同预测和下一条最有区分力的来源或观测；质疑者建立独立的挑战记录。这是当前默认输出。

**方法探索种子：** `proposals/<METHOD_ID>/v<N>.json`，保存可证伪的计算机制、方法构造和相关数学。发现阶段欢迎引入大胆算法想法，最终实现方向由后续证据与用户研究阶段决定。使用 [模板](TEMPLATES.md) 记录与他人方法的联系和实际前提。

## B · 独立评议的科学职责

| 角色 | 工作重点 | 保存方式 |
| --- | --- | --- |
| 提案者 | 描述结构与构造的关键方程、原始计算边界、竞争解释与最有力反例 | `proposals/<id>/vN.json` |
| 独立 reviewer | 亲自复建源码/数学、提出异议或支持**精确版本上的精确命题** | `reviews/<id>/vN/<review-id>.json` |
| 挑战者 | 追踪另一个解释、构造反例与较强的修订方式 | 自己的独立 review / discovery 文件 |
| 整理者 | 根据真实独立记录呈现共同结论、核心分歧与下一观察 | `endorsements/<id>/vN.json`，再串行更新 [QUEUE.md](QUEUE.md) |

评议者从原始论证开始，之后参考其他评论并记录 `prior_reviews_seen`；`reviewer.agent_id` 与提案作者有区别，`proposal_git_blob_sha` 绑定所审查的原始版本。身份与独立性以实际科研过程中的不同来源和推导来体现。

### 认可状态

| 状态 | 科学含义 |
| --- | --- |
| `OPEN` | 已登记科学方法，等待新的独立讨论 |
| `PEER_SUPPORTED` | 至少一位外部 reviewer 对该精确机制给出独立论证的 `SUPPORT`，主要实质异议已处理 |
| `CROSS_CONFIRMED` | 至少两位不同外部 reviewer 独立支持，主要实质异议已处理 |
| `CONTESTED` | 存在具有明确数学、源码或实验依据的待解 `CHALLENGE`，并行保留支持记录 |
| `SUPERSEDED` | 新版本或新研究解释取代旧构造，旧版作为可追溯证据保留 |

`SUPPORT` / `CHALLENGE` / `REVISE` / `ABSTAIN` 指向具体命题；`PEER_SUPPORTED` / `CROSS_CONFIRMED` 表示**方法讨论的意见状态**。新版本通过独立评议形成自己的证据链。

## C · 文件与字段结构

提案字段：`schema_version, method_id, version, title, proposer, origin, related_method_ids, lenses, mechanism, competing_explanations, falsifiable_predictions, best_counterexample, evidence_links, discussion_questions, claim_level`。

评议字段：`schema_version, review_id, method_id, version, proposal_git_blob_sha, reviewer, verdict, independent_reconstruction, decisive_argument, counterexample_or_discriminator, constructive_revision, evidence_links, prior_reviews_seen`。

认可快照字段：`schema_version, method_id, version, proposal_git_blob_sha, state, supporting_reviews, challenging_reviews, resolved_challenges, endorsed_claim, remaining_unknowns, summarized_by`。

每个 Agent 在单独的版本文件中提交，保留 Git SHA、来源及上一版本的审查。**一个串行整理者**更新共享 `QUEUE.md` 和 endorsement 快照；所有其他 Agent 通过独立文件参与讨论。对已发现的错误，写出精确修订及该修订如何改变核心科学预言。

## D · 当前研究优先事项与执行范围

最新的六来源 E0 被动观测发现 Block0 FFN 组输出与 residual 的径向对齐；当前优先独立核查 LN 理想局部投影、残差 bypass、实际消费路径及功能意义，也欢迎对量化层级、共同均值、高杠杆位置提出不同解释。来源需要准确区分 A100 HiF4 QDQ→BF16、格式规范上的 MXFP4/NVFP4、数学构造与视频质量测量。

本目录处理科学研究、数学、源码和可用历史证据。2026-10-25 用户批准的六来源采集已经完成，早期 [暂停回执](../20261024/EXPERIMENT_PAUSE.json) 保留；新模型/CPU/GPU 执行及全视频实验由用户另行授权。[ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 是唯一任务与权限来源。讨论成果有助于确定下一项最有价值的研究，却不自动成为实现任务。
