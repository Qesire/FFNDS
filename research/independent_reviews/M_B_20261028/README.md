# M-B 独立评议与全局综合｜2026-10-28

**推荐先读：[四轮综合与全局判断](SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md)。** 这是对多 Agent 交锋后的单一阅读入口，归并数学共识、真实源码纠错、已有数据边界、竞争解释、在线物理成本与 W0–W3 后续判别。完整原始论证按下表保留，不再将每轮评论当成新的并列权威方法快照。

## 阅读路径与原文身份

| 顺序 | 保留文档 | 独有贡献与边界 |
|---|---|---|
| **0 / 先读** | [SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md](SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md) | **独立综合 / 决策建议**：条件 ALG、真实 SRC/原有 NATIVE/VIDEO/SYS、开放实验、跨路线排序与 W0–W3 工作包 |
| 1 | [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) | M-B v2 独立评估：同 source 敏感度配对、末层 fixed+rank1、65 充分统计、direct64 与 A0 成本基线 |
| 2 | [EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md](EARLY_LAYER_SUFFIX_AND_MULTISTEP_REVIEW_20261028.md) | 全后缀 Pre-LN gauge 条件定理、98.75% 单状态 Jacobian 核维度、Attention/四步传播、终端 signed interaction |
| 3 | [RESPONSE_TO_SECOND_AGENT_20261028.md](RESPONSE_TO_SECOND_AGENT_20261028.md) | 对第二轮独立回应：B-QUOTIENT、共享核反例、B-JOINT-STATE 与 B-CROSSSITE 区分、二站点四臂反事实 |
| 4 | [FOURTH_ROUND_REVIEW_OF_QUOTIENT_RESPONSE_20261028.md](FOURTH_ROUND_REVIEW_OF_QUOTIENT_RESPONSE_20261028.md) | A0 **行平均≠通道平均**的 SRC 修正、四正交分量、同 expert 两 step 的共享 folded bias、已有论文先例 |

## 综合结论与后续产物

**CONDITIONAL_CONTINUE**：M-B 数学/机制研究价值成立，尚需证明同成本、合法前置信息、原生数值、完整视频和真实系统的算法增量。A0 6016 的四位置 native-Q2 局部恢复是已有的强学生比较器，尚未得到全 80 函数自由生成的质量/速度结论。旧 UB1 consumer20 的相貌较稳定而 RAFT 运动大幅下降，是需解释的历史视频现象，不能替代新的 Student 成功或失败证据。

- **W0 · 低成本证据补缺**：按同源真实 teacher/student、gate 与 256 sampled rows 计算 row×channel 四分量；source-row mean fraction 不能作为 gauge 份额；检验共享 bias 方向相干、折叠修正与 quotient 的增量
- **W1 · 现行 M-B B0/B1**：原生 post-write u/head/flow neutral/finite、fixed vs paired/marginal/shuffled，在仓库原任务交接与 ACTIVE 约束下执行
- **W2 · 工程战略建议**：扩展 A0 的深度、共享步、Student rollout 和全视频动态/真实 kernel 成本验证；这是综合意见，**未在此修改**当前 B0 优先级
- **W3 · 独立探索**：B-F 真正 factored 算子、B-OBS/跨 site 有符号后缀、M-C 编码求值以及 I/F/C 其他机制，视区分性数据推进

## 角色与方法 Skill

本目录保持 **独立评议** 身份，不是正式 proposal endorsement、ACTIVE 执行权限、NATIVE/VIDEO/SYS 新实测或现行整合方案替代。以 [FFNDS 领域 Skill](../../../.agents/skills/ffnds-paper-discovery/SKILL.md)、[mechanism-research](../../../.agents/skills/mechanism-research/SKILL.md)、[problem-reformulation](../../../.agents/skills/problem-reformulation/SKILL.md) 与 [research-expansion](../../../.agents/skills/research-expansion/SKILL.md) 的真实计算、竞争解释、最小判别和构造性描述组织研究。

正式任务依据：[研究导航](../../RESEARCH_NAVIGATION.md) · [三路线整合 v2](../../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md) · [M-B 方法快照 v2](../../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) · [B0/B1 交接](../../agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md) · [ACTIVE](../../ACTIVE_RESEARCH.json)。
