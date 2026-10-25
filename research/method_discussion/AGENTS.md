# FFNDS 方法讨论 Agent · AutoResearcher 默认原则

**Current independent E0 investigation:** Start with [latest E0 mechanism guidance](../20261025/E0_NEXT_MECHANISM_GUIDANCE_20261025.md) and the exact saved source records. Agent A can independently derive three pairing-sum invariants of the four-arm distance null across sources/subspaces; Agent B can challenge Block0 token pairing versus shared direction, auditing archived raw fields first; Agent C can independently compare temporal common-mode and centered variation with published SVD-Cache/MagCache mechanisms. This is a suggested division of **independent scientific questions**, and creative alternatives remain welcome. Reviewers bind exact blobs and report counterexamples; shared states are updated serially.


**Search-enhanced AutoResearcher v2.4:** Every researcher and independent reviewer inherits the [pinned web-research-discovery Skill](../../.agents/skills/web-research-discovery/SKILL.md), the generic [WARD protocol](../../core/WEB_RESEARCH_DISCOVERY.md) and [FFNDS search specializations](../agent_protocol/DEEP_WEB_RESEARCH.md). Verify closest mechanism-level literature and original source against the exact commit in [source manifest](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json), then derive new independent explanations or strong counterexamples.


**阶段：`E0_OPEN_DISCOVERY`。** 先从 [唯一活跃状态](../ACTIVE_RESEARCH.json) 读取任务与权限，结合 [E0 问题及改进目标](../20261025/E0_GAPS_REMEDIATION_REQUIREMENTS.md)、[阶段锚点](../agent_protocol/E0_STAGE_ANCHOR.md) 选择一个有科学价值的开放问题。

## 所有角色共享的研究动作

1. **发现：** 从真实 Wan 源码、数学张力、历史负结果、外部论文或最新六来源观测发现异常结构，形成最强解释。
2. **扩展：** 探索不同数学表示、信息时机、量化状态、FFN 有符号组合、残差几何、蒸馏/时空和潜在算子自由度。新的算子猜想可直接作为 `idea seed`。
3. **重建：** 对实现相关主张，核对真正的 Q1→W1→GELU→Q2→W2、gate/residual、dtype、step/expert 和下游消费路径。先处理**能改变结论**的源码部分。
4. **区分：** 给出另一种有解释力的机制、不同预测与最经济的源码/数学/保存数据对照。条件定理与有效反例也是完整科研结果。
5. **交付：** 具体新认识、证据级别、竞争结果和一项能改变下一步判断的观察。研究质量由实质论证决定。

按问题使用固定 [AutoResearcher 来源](../agent_protocol/ACTIVE_SOURCE_MANIFEST.json)、[研究 harness](../../core/AGENT_HARNESS.md)、[mechanism-research](../../.agents/skills/mechanism-research/SKILL.md)、[research-expansion](../../.agents/skills/research-expansion/SKILL.md)、[problem-reformulation](../../.agents/skills/problem-reformulation/SKILL.md) 和 [FFNDS domain skill](../../.agents/skills/ffnds-paper-discovery/SKILL.md)。

## 充分且独立的 Web Search

每位发现者、挑战者、reviewer、整理者都主动开展 [深度 Web 文献/源码检索](../agent_protocol/DEEP_WEB_RESEARCH.md)：多领域经典与最新工作、后向/前向引用、官方公式及代码、最强重复机制和反例。记录自己实际检索的范围与来源，尽量形成与其他 Agent **信息来源不同**的证据。E0 的文献发现服务于机制探索，方法构想继续作为开放种子。

## 当前最重要的独立复核

六来源 E0 保存操作数呈现 Block 0 FFN 输出—残差径向对齐。独立从原始计算图、LN `epsilon` 导数、pre/post/leave-one-group-out 和五个固定组的聚合口径复建，再挑战其可能的下游功能解释。并行可研究 HiF4 QDQ 与 MXFP4/NVFP4 的信息时机、时空共同均值或 Source 8 高杠杆行。引用时保留 **6 来源、3 主体/3 背景/0 相机** 的真实采样构成。

## 协作与评议

- **发现者/提案者：** 在 [E0 自有发现文件](discoveries/E0/README.md) 或 `proposals/<method-id>/vN.json` 中保存自己发现的结构、数学/源码依据、竞争解释及下一项观察；早期 operator idea 可保持种子状态。
- **独立挑战者：** 按自己的源码/公式推导或新文献反例审查精确版本。独立的 `reviews/<method-id>/vN/<review-id>.json` 明确写出证实、挑战或需要修订的**具体命题**。
- **整理者：** 依据不同作者的真实 review 整理 `SUPPORT` / `CHALLENGE` / `REVISE` / `ABSTAIN`。一位外部 reviewer 的独立 SUPPORT 可支持 `PEER_SUPPORTED`；两位不同 reviewer 的支持可支持 `CROSS_CONFIRMED`。未解决的实质异议保持 `CONTESTED`。历史提案、评议及来源版本均保持可追溯。
- **共享状态：** 串行协调者更新 [QUEUE.md](QUEUE.md) 与 endorsement 汇总；阶段与实验权限以唯一 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 为准。

## 工作范围

本讨论空间开展科学构想、数学推导、原始源码/文献查找与已保存数据分析。2026-10-25 用户单独授权的六来源原生 E0 采集已经完成；新的 CPU/GPU/model 作业需用户明确授权。[历史暂停回执](../20261024/EXPERIMENT_PAUSE.json) 继续保存。

外部 Agent 启动器应同时提供 [根 AGENTS](../../AGENTS.md) 与本目录指令，使新成员进入相同的科学阶段和协作规则。
