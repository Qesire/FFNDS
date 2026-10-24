# 方法讨论记录模板

复制并替换以下占位符。保存路径与字段规范见 [README.md](README.md)；所有内容只供方法讨论，不代表实际实验/性能结果。禁止使用示例中的占位符制造虚假的独立 reviewer。

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
