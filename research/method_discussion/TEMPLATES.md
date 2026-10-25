# 方法讨论记录模板

## 原始论文和源码是创造新机制的默认资源

所有 E0 发现、后续 proposal 和 review 均主动开展 [深度 Web Search](../agent_protocol/DEEP_WEB_RESEARCH.md)：跨经典与最新论文、相关数学、源代码、引用链和竞争解释深化。重点记录与研究问题有关的**方法公式、实际实现、实验差异和反例**。独立 reviewer 带回自己核查的最强先例，说明检索结果如何拓展或修正假说。检索范围与可访问来源按实际记录。


## E0_OPEN_DISCOVERY：FFN 本征现象发现优先

当前主要参照 [FFN 本征研究纲领](../20261025/E0_FFN_INTRINSIC_RESEARCH_RESET_20261025.md) 与 [新第一任务](../20261025/E0_FFN_FIRST_TASK_20261025.md)：明确激活 vs 有符号 FFN 净贡献 vs downstream 价值 vs 物理可省算单位。量化元信息只作为可选幅值线索。提出者不需要在发现期证明某算子有效；成熟 proposal 才具体列出计算边界。


以 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 为阶段权威。当前优先采用 [E0 独立发现模板](discoveries/E0/README.md) 报告真实结构、两种竞争解释和区分性观察。下方的 versioned `proposals/<id>/vN.json` 用于保存成熟候选方法或由现象启发的新算子种子，由未来阶段再确定实现时机。


将下方示例字段替换为真实命题与本人来源；保存路径见 [README.md](README.md)。每个 Agent 用自己的独立推理与身份创建记录，实验和性能结论由实际结果另行支持。

## 所有角色的默认科学方法（AutoResearcher）

使用 [本目录 AGENTS.md](AGENTS.md) 和 [已钉住的 AutoResearcher manifest](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)，选择最适合当前科学争论的 mechanism-research、research-expansion、problem-reformulation 模式，可以结合使用。**按科学问题灵活组合这些方法，以获得更有区别力的解释与检验。**

- **独立提案：** 在 `mechanism` 中体现**新结构或计算自由度 → 数学构造 → 真正省下什么工作**；在 `competing_explanations`、`falsifiable_predictions` 中体现机制竞争与可区分预言。将开放的源码证明和性能检验作为后续研究机会记录在相应命题旁边。
- **独立评议：** `independent_reconstruction` 要有 reviewer 自己复建的数学、源码或历史证据推导；`decisive_argument` 直接说明支持/质疑的关键机制；`counterexample_or_discriminator` 应能真正改变对方法的判断。优先寻找构造性修改和真正改变判断的竞争解释。
- **整理者：** 根据不同 reviewer 的实质论证归纳其精确同意的命题与条件，同时保留未决反例；将数学支持、实际测量与仍有分歧的条件分别清楚陈列。

当某一篇旧论文、代码、实验或指定 AutoResearcher 技能对论证有实质贡献，在已有 `evidence_links` 或文字论证中给出来源；不需要新增空泛的“本 agent 遵守了全部原则”字段。与此前导入的历史提案保持数据兼容。

## A. 每个提出者各自保存自己的提案

路径：`proposals/MD-YYYYMMDD-MECHANISM-UNIQUE/v1.json`

```json
{
  "schema_version": "1.0",
  "method_id": "MD-YYYYMMDD-MECHANISM-UNIQUE",
  "version": 1,
  "title": "可区别于历史方法的一句话构造",
  "proposer": {
    "agent_id": "真实稳定 agent 标识",
    "session_ref": "当前会话/PR/commit 参照"
  },
  "origin": {
    "kind": "AGENT_INDEPENDENT",
    "source_revision": "读过的仓库 commit",
    "note": "创意来源；是否已看过他人意见"
  },
  "related_method_ids": [],
  "lenses": ["Q3", "Q1", "Q2"],
  "mechanism": {
    "native_boundary": "计算图、shape、条件信息、量化和残差消费者",
    "action": "明确要替换/改变的实际算子，而不只提出评分函数",
    "mathematical_representation": "公式、结构或条件定理",
    "distinguishing_feature": "与旧方法或现有论文机制上有什么实质差异",
    "expected_compute_saving": "哪块实际 GEMM/QDQ/搬运预计减少，纯设想则写假说"
  },
  "competing_explanations": [
    "反机制 A",
    "反机制 B"
  ],
  "falsifiable_predictions": [
    "相对于最接近固定算子/旧方法会有何不同可观察结果"
  ],
  "best_counterexample": "最有可能击穿方法核心的具体反例",
  "evidence_links": [
    "已有代码、论文、旧实验相对路径或 URL"
  ],
  "discussion_questions": [
    "请其他 agent 独立推导/反驳的关键问题"
  ],
  "claim_level": "EXPLORATORY_CONJECTURE",
  "discussion_authorization": "DISCUSSION_ONLY_NO_EXPERIMENT_OR_IMPLEMENTATION"
}
```

## B. 每位外部 reviewer 写一份自己的记录

路径：`reviews/<METHOD_ID>/v<N>/<REVIEW_ID>.json`

```json
{
  "schema_version": "1.0",
  "review_id": "REV-YYYYMMDD-AGENT-SUFFIX",
  "method_id": "与被复核方法完全一致",
  "version": 1,
  "proposal_git_blob_sha": "从 GitHub fetch_file 得到的提案完整 blob SHA",
  "reviewer": {
    "agent_id": "不同于提案 proposer 的真实 agent 标识",
    "session_ref": "可追溯的会话/PR/commit"
  },
  "verdict": "SUPPORT",
  "independent_reconstruction": "本人独立重建的实际 FFN 计算流及数学结构，不能复制摘要",
  "decisive_argument": "具体为什么支持；如果提出 CHALLENGE，写出与方法结论冲突的推导",
  "counterexample_or_discriminator": "最强反例、成立条件或与已有方法可区分的预言",
  "constructive_revision": "哪一个修改能加强或改变核心论断，没有则写无需修改",
  "evidence_links": ["相关原始源码、历史数据及公开论文"],
  "prior_reviews_seen": false
}
```

`verdict` 可取 `SUPPORT`, `CHALLENGE`, `REVISE`, `ABSTAIN`。应说明自己究竟支持**哪一条条件论断**；同意一个新观点值得讨论，并不代表证明其会加速或保持视频质量。

## C. 已有真实评议后的认可快照

路径：`endorsements/<METHOD_ID>/v<N>.json`

```json
{
  "schema_version": "1.0",
  "method_id": "与原提案相同",
  "version": 1,
  "proposal_git_blob_sha": "固定版本的 blob SHA",
  "state": "PEER_SUPPORTED",
  "supporting_reviews": [
    "reviews/<METHOD_ID>/v1/<真实 review 文件名>.json"
  ],
  "challenging_reviews": [],
  "resolved_challenges": [],
  "endorsed_claim": "精确被共同支持的机制命题及成立假设",
  "remaining_unknowns": [
    "未经实证确认的推论"
  ],
  "summarized_by": "整理本快照的 agent_id；不算支持票"
}
```

没有真实 `SUPPORT` 不得创建 `PEER_SUPPORTED` 快照；不同的 agent_id 之外还要通过独立推导内容判断是否为实质复核。一个 proposer 的不同会话不产生额外 reviewer 资格。若存在独立 `CHALLENGE`，应采用 `CONTESTED` 并列出实质性分歧；新版本不沿用旧版本同意票。
