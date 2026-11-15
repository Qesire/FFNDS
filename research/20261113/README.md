# 2026-11-13 · G20 VBench 路线与判断

**当前进度（2026-11-15）：A、BW Q1/ICC 与 P64 输入已完成；BW 的 132 个 Chip 风险分数与 BWS 待完成。** [结果索引](../../reports/20261114/g20_two_track_core/README.md)保留已有数值结论，并记录新输入的同状态证据、额外 D4 smoke 和 VAE 解码的实际消耗。

[路线](G20_VBENCH_RESTART_V4_20261113.md)、[资产](G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)与[判断](G20_VBENCH_DECISION_LEDGER_20261113.json)区分输入产出和风险评分。下一步按 rev5 [手册](../20261114/G20_TWO_CORE_EXECUTION_HANDOFF_20261114.md)与[规格](../20261114/G20_TWO_CORE_EXECUTION_SPEC_20261114.json)，只消费现成输入补 BW Chip 分数（0 新轨迹），并完成 BWS（最多 6 条 O40，包括验证轨迹）。不再采 D4或重做 A/Q1/ICC，不增加图文、CAL_B、网格或独立 smoke。本次核查新增作业为 0。

## rev4 目标恢复记录（历史）

**当前：2026-11-14，v4 第 4 次修订。两条主线继续，具体新计算尚待独立预审。**

A 设计具有质量—成本价值的新算子；B 解释 Q1r2 与实际 ICC/Chipmunk 的表现差异，分离 Wan 内权重、状态与采样间隔等因素。V4 关闭已测构造的继续盲调，未关闭这两个研究问题。

- [当前路线](G20_VBENCH_RESTART_V4_20261113.md)：两项核心实验、各自判别对象、复用依赖与推进条件。
- [判断台账](G20_VBENCH_DECISION_LEDGER_20261113.json)：已测负结果与两条主线尚未回答的问题分别记录。
- [资产与队列](G20_VBENCH_ASSETS_AND_QUEUE_V4_20261113.json)：并行准备 A_Q1_RESIDUAL 和 BW_PREPARE；计划任务与可运行 scope 分开。
- [33 条官方图文与角色](../../config/g20_vbench_restart_cohort_v1_20261113.json)：CAL_A11、CAL_B 预留 11、DEV11 保留；本轮不新增图文、不采 CAL_B。
- [V4 正式结果](../../reports/20261114/g20_v4_repr_density/README.md)与[本次设计复核](../experiment_reviews/G20_TWO_TRACK_RESTORATION_20261114/REVIEW.json)。

已有 bank 支持 A 及 B 的 Q1/ICC 核心回放。忠实 Chipmunk 若缺匹配 full-M 统计，先复用、定位或精确恢复；确需补采时仅限既有 DEV11，最多 11 条 D4，不解码或评分新视频，不自动补 CAL11。B-WS 最多 6 条 O40 用于原生复合状态比较：同维合法 FFN 接口、同 expert 原生端点、CAL/DEV 绑定及固定参数、参考 gate 和聚合是前提。noise/gap 重叠只限制纯间隔、纯状态或 CFG 成分归因；缺重叠时保留复合风险与交互，对相应子主张标记 UNIDENTIFIED。采集、回放和拟合各自需要具体有限 scope；本次没有新增运行授权或 dispatch。

不恢复独立 floor、无问题驱动的参数网格、失败候选全层视频或旧实验重跑。历史计划、审阅、冻结 scope、实际结果及失败记录保留原身份。
