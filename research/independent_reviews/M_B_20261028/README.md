# M-B 独立评议｜2026-10-28

本目录独立保存对 FFNDS **M-B（gate/suffix-sensitive coordinates）v2** 的批判性数学与科研潜力评议，区别于仓库权威方法快照、正式 Agent 方法投票、实验登记及 ACTIVE 运行授权。

## 评议正文

**[INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)** — 包含原生计算流、配对敏感度等式、末层 LN/head Jacobian 分解、65 统计量的真实意义、残差增量学生构造、强基线、现存实验边界、最近先例、最小判别实验及正向分流建议。

## 结论

**CONDITIONAL_CONTINUE。** M-B 具有研究价值，特别是状态配对的下游敏感度；然而 **真实可部署增益尚待验证**。主要待判别问题是：在相同在线计算成本和训练条件下，M-B 是否能优于普通紧凑 FFN（已有 A0 6016 局部恢复基线）并保留完整视频行为。

- 保留 B0 原生 neutral 捕获与 B1 来源配对响应研究；
- E1 增加 **fixed readout vs fixed+rank1 vs marginal vs paired vs shuffled** 判别；
- E2 增加 **direct64、q+v、Δq、Δq+Δv、A0** 同在线成本对照；
- 只有真实局部增益与可执行性成立后再进入完整学生四步自由视频；
- M-A/M-C 与信息/执行线继续并行。

## 身份边界

本目录是 **独立意见（ALG/SRC-REVIEW/REPO-EVIDENCE）**，不是正式 PEER_SUPPORTED、NATIVE/VIDEO/SYS 通过证明。创建文档未执行模型/GPU/视频，也未修改旧方法提案、讨论队列或实验状态。

追溯现行对象：[M-B v2](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) · [整合方案 v2](../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) · [研究导航](../../RESEARCH_NAVIGATION.md)。
