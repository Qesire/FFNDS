# 同数据value10/DCAC10：20来源40新完整视频及真实成本

两个原生范围均actual exit0，各20新候选81帧、80flow、3200 FFN，合计40完整视频/6400 FFN。原20 Dense视频、状态和CORE复用，不重生成。每新视频FP32 raw/BF16 initial两项噪声均与实际Dense逐字节相等（80次）；原始failed-V1/CPUrepair血缘保留。

| 来源等权单遍实测 | Dense（复用） | value10 | dcac10 |
| --- | ---: | ---: | ---: |
| 生成wall秒（含观察） | 46.115981 | 43.726157 | 43.502860 |
| 全160 FFN CUDA边界秒 | 6.497038 | 4.361996 | 4.368184 |
| 四次infer wall秒 | 35.784175 | 33.554362 | 33.408845 |
| CORE下采样Farneback幅值px | 0.622705 | 0.309250 | 0.309492 |

实测FFN边界约1.489/1.487倍，生成wall约1.055/1.060倍；不同历史attempt的Dense复用、未预热重复且含CPU/GPU观察，故不作为公平SYS验收。按该单遍wall作条件Amdahl分析，FFN只占约14.09%；若其余耗时固定，即使FFN理想为零，总倍率也仅约1.164。这说明当前offload执行配方的整机瓶颈也需要单独研究，不能从W2 MAC减少推导大的video加速。

同数据、同rank960、每层同两GEMM，仅拟合目标不同；原W1/GELU/b2/gate、GPU原buffers和CPU masters保留。两候选各shared80，payload2.91GB，完整160位置自由运行，无shadow Dense、在线selector或缓存。

CORE幅值比例value10/DCAC10为0.49662/0.49701；正式RAFT方向见[八轴结果](RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)，不能据CORE宣布运动恢复。逐源、真实协议/RUN/EXIT、全部账本与成本：[机器结果](RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_NATIVE_RESULTS_20261029.json)。两个source shard外部耗时1130.442/1143.172秒，源为历史暴露开发20图，与新10拟合image SHA不交叠。
