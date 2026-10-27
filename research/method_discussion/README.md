# FFNDS · 多Agent独立机制发现与方法评议

当前 **E0_OPEN_DISCOVERY** 允许 **I/F/C并行探索**：可从信息/表示、生成函数/动力学或物理计算/算子任何一个研究入口提出机制并由其它Agent独立评议。[统一目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) · [领域Skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md) · [ACTIVE](../ACTIVE_RESEARCH.json)。[G13](../20261027/G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md)与[O/D/Q/P](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)用于F线定位，Function–Structure登记只覆盖相应对象。

**方法**：发现真实现象、信息自由度或算子数学 → 形成可证伪机制 → 精确相关源码/原论文 → 竞争解释与不同预测 → 最便宜的判别证据。[原始文献检索](../agent_protocol/DEEP_WEB_RESEARCH.md)支持跨学科和相反证据，AutoResearcher机制研究/扩展/问题重构鼓励开放数学空间。

## 独立讨论的两种科研成果

**发现记录（信息/函数/算子均可）**：\`discoveries/E0/<agent-id>-<phenomenon>.md\`，记录行为/无标签FFN响应、条件、源码/数学、候选计算结构、竞争性解释和最有信息价值的观察。[独立发现说明](discoveries/E0/README.md)提供示例及历史结果。

**方法提案**：\`proposals/<METHOD_ID>/v<N>.json\`，用[模板](TEMPLATES.md)登记具体构造、作用在什么真实计算边、预计的功能价值与不同预测。提出新表示、功能协作/重组、时空或量化信息机制均可以先作为早期探索种子；真正形成明确方案与版本时进入方法队列。

## 独立角色与证据身份

| 角色 | 科研职责 | 落点 |
| --- | --- | --- |
| 发现者/提案者 | 构造新机制、数学/原始源码依据、精确上下文、不同预测与下一判别 | \`discoveries/E0/\` 或 \`proposals/<id>/vN.json\` |
| 独立评议者 | 独立复建源码、原论文和公式，对精确命题形成支持、挑战或建设性修改 | \`reviews/<id>/vN/<review-id>.json\` |
| 挑战者 | 构造替代解释、条件反例或新的表示，指出实验设计如何区分 | 自己的discoveries/reviews |
| 串行整理者 | 保留独立审阅版本，汇总具体共识/争议及其对应原生图节点 | \`endorsements/<id>/vN.json\`、[QUEUE](QUEUE.md) |

\`PEER_SUPPORTED\` 表示一位外部独立reviewer对精确版本有具体支持，\`CROSS_CONFIRMED\` 表示两位不同外部reviewer形成交叉支持，\`CONTESTED\` 记录有证据支撑的实质性分歧。它们评价科学讨论证据；已运行数值、自由视频与部署的结论有各自的真实RUN/证据身份。

提案字段：\`schema_version, method_id, version, title, proposer, origin, related_method_ids, lenses, mechanism, competing_explanations, falsifiable_predictions, best_counterexample, evidence_links, discussion_questions, claim_level\`。

评议字段：\`schema_version, review_id, method_id, version, proposal_git_blob_sha, reviewer, verdict, independent_reconstruction, decisive_argument, counterexample_or_discriminator, constructive_revision, evidence_links, prior_reviews_seen\`。

认可快照字段：\`schema_version, method_id, version, proposal_git_blob_sha, state, supporting_reviews, challenging_reviews, resolved_challenges, endorsed_claim, remaining_unknowns, summarized_by\`。

**版本规则**：每位Agent保有自己的发现/评议文件，原始Git blob SHA、真正独立论证、提案身份和不同观点持续可追溯；串行整理者更新共享表。新错误通过有界勘误说明，不改变旧实验数据及原始版本记录。

## 并行研究如何在计算图中连接

F线从自然视频与signed响应定位FUNC-OBS，再沿真实W1/GELU/HiF4-Q2/W2、attention/残差和四步状态研究正常承担者；I/C线则可以直接由量化信息、计算代数、共享结构、训练后重组、cache或物理tile形成独立成果。任何成果都可反向解释另两条线。对应[G0–G13图](../../docs/98_端到端计算图与证据注解_20261025.md)及[Graph-Agent协议](../agent_protocol/GRAPH_AGENT_COLLABORATION.md)；来自外部作者的机制保持LIT，数学恒等式保持ALG，有效原生观察和功能因果各按证据等级保存。

**当前研究范围**：独立科学讨论、原论文/源码、数学、已有数据和计算图工作。**新的CPU/GPU/模型/视频运行暂停**；旧E0/paired-native固定实验均已完成或关闭，后续许可看[ACTIVE](../ACTIVE_RESEARCH.json)。所有新视频质量按[全深度指标与画质协议](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)独立记录。
