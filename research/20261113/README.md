# 2026-11-13 · G20 VBench 路线与判断

**当前：2026-11-14，rev5 执行细则；2 个研究方向、3 个固定比较阶段（A、BW、BWS）。** 从[执行手册](../20261114/G20_TWO_CORE_EXECUTION_HANDOFF_20261114.md)与[机器规格](../20261114/G20_TWO_CORE_EXECUTION_SPEC_20261114.json)接手。A 固定四臂，不以最小收益阻塞启动或完成；BW 后复用或补齐 BWS 四格，不依赖 A 或 BW 结果正负。B 固定比较已实现的 `CHIPMUNK_GLOBAL_P64`，沿用真实 proxy/selected_action；缺 P64 固定 64 行输入/统计（按完整 token 索引定位）只阻塞 Chip cell，不要求 full-M 块均值或作者新系统复现。

常规一轮自审、独立审查、集中答复和定向 recheck；保留预启动身份核对，数值/重载检查合并正式 job。BW_PREPARE 与缺项触发的 B_CHIP_STATS 属于准备和资产生产，不增研究方向。0 新图文、0 CAL_B；Chip 只补固定 64 行缺口，最多既有 DEV11 的 11 条 D4；BWS 最多 6 条 O40。已有研究目标无须重问许可；本次没有新增运行授权或 dispatch。下方 restoration review 仅对应 rev4 目标恢复，rev5 细则的实际检查记录见机器规格。

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
