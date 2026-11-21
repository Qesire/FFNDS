# FFNDS 研究导航

从[当前接手说明](agent_protocol/CURRENT_CONTEXT_PACKET.md)开始。默认方法是固定 64 个探针的 `fast512` 逐位置锚点联合读出。下表给出实际计算、独立评价和下一轮对照各自所依赖的材料。11 月 21 日的原始证据使用固定提交链接，切换分支后仍指向同一份实验。[当前方法原文](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md)

| 要回答的问题 | 先读什么 |
|---|---|
| 每层看见什么，哪些量可以离线或融合？ | [算子、锚点与计算图](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/保留锚点方法总结.md) · [真实读出实现](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchored_scale/exact_port.py) |
| 怎样复核来源、画幅和 24 源速度？ | [校准来源](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/provenance.json) · [24 源清单](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/COHORT.json) · [配对计时与功能等价性](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时与等价性.md) |
| 运动方向、局部纹理和人眼结论如何对应？ | [原批方向与运动统计](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/实验结果.md) · [VBench 完整评测](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/vbench/VBench完整评测.md) · [原批视频快照](https://github.com/Qesire/FFNDS/releases/tag/ffnds-results-20261121) |
| 64、96、128 的局部结果能否迁移到视频？ | [第 19 层逐源预算统计](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_capacity/budget_summary.json) · [固定 128 逐片运动和区域指标](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/budget128/逐片方向与区域实测.md) · [轮换布局结果](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/逐片方向与区域实测.md) |
| 怎样核查新片、缓冲复用与缓存编码？ | [本轮六份审片文件的指纹](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/审片文件指纹.csv) · [A100 缓冲计时](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_perf/A100_RESULT.json) · [固定 64/128 的整片配对](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/budget128/配对计时.json) · [无损编码核验](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_memory/GPU.json) |
| 为什么保留某些反例？ | [去掉逐位置状态的三片盲审](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_zero_fused_DEV3/人工判读.md) · [0951 左摇跨种子人审补记](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_0951_seeds/跨种子人工复核.md) · [0951 同景分步、选层对照](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_0951_improvement/0951改进实测.md) |

## 按问题查历史研究

这些条目保留其当时的模型、输入与评测口径。历史目录中的“下一步”和设备配额对应原任务。它们对解释旧结果仍有用，当下工作的取舍见[当前接手说明](agent_protocol/CURRENT_CONTEXT_PACKET.md)。

| 问题 | 历史记录 |
|---|---|
| G19 的 Q1r2 如何折叠和计时？ | [G19 推导与实测](20261107/G19_METHOD_DERIVATION_AND_RESULTS_20261107.md) |
| G20 中 Q1r2、ICC、Chipmunk 怎样复现比较？ | [严格复现结果](../reports/20261117/ffnds_strict_reproduction/README.md) · [G20 两条研究线](../reports/20261114/g20_two_track_core/README.md) |
| 早期少数层的局部改善能否迁移？ | [六站点状态条件 FFN](../reports/20261118/state_conditioned_ffn_operator/README.md) · [全 80 站点状态调制割线](../reports/20261119/state_coupled_secant_ffn_operator/README.md) |
| 更早的研究目标、队列和审批上下文来自哪里？ | [历史入口的归档映射](../archive/20261121/handoff/README.md) · [历史实验与证据索引](EXPERIMENT_EVIDENCE_INDEX.md) · [逐 scope 状态记录](ACTIVE_RESEARCH.json) |
