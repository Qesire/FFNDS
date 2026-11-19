# 全 80 位点状态调制割线的 DEV3 结果

六种方法在三个开发来源各生成一条完整视频，合计 18 条、72 次 NFE。每条使用严格 Wan2.2 四步 HHLL BF16、seed 42 和同源 Dense 初噪。下文的人眼观察来自一位评审者，包含首批和续批的同源对照。三份完整独立票据待收齐。[冻结实验方案](../../../research/20261119/STATE_COUPLED_SECANT_EXPERIMENT_PLAN.md) · [执行回执](output/EXECUTION.json) · [生成清单](output/GENERATION_MANIFEST.json)

## 连续对照与人眼观察

在 `vbi_0392`，同宽线性 48 是评审者此前看到最稳定的一条。续批的两场 16-0392 中 A、B 均可接受，其余已看的 0392 对照画质整体较差。运动削弱在续批仍然存在。每个同步视频均包含 Dense、A、B，采用 81 帧和 16 fps。

| 0392 中表现较好的方法 | 可复审视频 | 观察 |
|---|---|---|
| 同宽线性 48 | [R01 首包 05-B](media/r01_first_05_vbi_0392.mp4) | 此前最稳定 |
| 配对关系加权 48 | [R01 续包 16-A](media/r01_continuation_16_vbi_0392.mp4)、[R03 续包 16-B](media/r03_continuation_16_vbi_0392.mp4) | 两个对照均可接受，使用同一条原片 |
| 配对点值 48 | [R01 续包 16-B](media/r01_continuation_16_vbi_0392.mp4) | 画面可接受 |
| 固定 64 行双线性 48 | [R03 续包 16-A](media/r03_continuation_16_vbi_0392.mp4) | 画面可接受 |

`vbi_1013` 上，评审者在 [R01 首包 06-A](media/r01_first_06_vbi_1013.mp4) 选择历史 R32。对 `vbi_0800` 的反馈为观感接近。破碎等级、Dense 重影保留情况和具体时段仍待记录。[定性观察与场次对应](output/HUMAN_REVIEW.json)

## 运动和质量指标

`vbi_0392` 的六种新方法全画幅 RAFT 位移均值均低于 Dense，范围为 1.5143–1.6687 px/2 帧，相当于 Dense 的 48.5%–53.5%。该量描述固定像素网格上间隔两帧的平均位移。主体动作、交叠与重影的保留由连续对照判读。

| 0392 | RAFT 位移均值（px/2 帧） | 生成作业 |
|---|---:|---|
| Dense | 3.1205 | 历史参照 |
| R32 | 1.6232 | 历史参照 |
| 固定 64 行双线性 48 | 1.5158 | `175748` |
| 同宽线性 48 | 1.5784 | `175749` |
| 宽原生 32 | 1.6219 | `175987` |
| 锚点上下文 48 | 1.6687 | `175988` |
| 配对点值 48 | 1.5143 | `176234` |
| 配对关系加权 48 | 1.5661 | `176235` |

截至 2026 年 11 月 19 日 16:57（北京时间），三批评分分别完成 9、2、5 项，合计 16/27 项。三批的连续 RAFT 均已完成，归档保存每条 5 个时间窗。其余 11 项待运行。镜头指令维度标为未测。[首批逐源评分](output/FIRST_QUALITY_PAIRED.csv) · [首批时窗](output/FIRST_RAFT_WINDOWS.json) · [首批评分回执](output/FIRST_QUALITY_LINEAGE.json) · [后续已完成评分](output/FOLLOWON_COMPLETED_QUALITY.csv) · [后续时窗](output/FOLLOWON_RAFT_WINDOWS.json) · [后续评分回执](output/FOLLOWON_QUALITY_LINEAGE.json)

## 原生执行与同输入诊断

六条生成臂共执行 1,440 次 64 行 FFN 观测，计 92,160 个 token-FFN。配对点值与配对关系加权共用同一组有序 64 行、留出行及双线性 CAL，关系系数分别为 0 和 0.5。两臂的观测矩阵平均特征值均为 1。R32 中性验收另计一条轨迹和 4 次 NFE，初噪、四步 flow 与 latent、最终 latent 及解码像素均与历史 R32 一致。[执行与中性验收](output/EXECUTION.json)

六位点 P512 回放在同输入上记录了 1,584 项查询和 405,504 条有向边。264 次离线 64 行 FFN 对应 16,896 个 token-FFN，新增 NFE 为 0。三源开发视频的结论适用于这些配对输入。独立来源确认和清洁部署计时尚待执行。[P512 逐源逐折汇总](output/P512_LOCAL_SUMMARY.json) · [发布文件 SHA256](SHA256SUMS)
