# FFNDS 实验发现与适用范围

默认方法是保留逐位置锚点、每次命中观测 64 个固定位置的 [`wide8192`](../tools/ffnds_fourstep/README.md)。三个开发来源共六对，整片平均较 Dense 省 5.04%。逐片、逐层、逐次计时的原始依据见[证据索引](EXPERIMENT_EVIDENCE_INDEX.md)。当前工作入口见[研究导航](RESEARCH_NAVIGATION.md)和[交接上下文](agent_protocol/CURRENT_CONTEXT_PACKET.md)。

| 问题 | 观察与含义 |
|---|---|
| 为什么保留逐位置锚点？ | 每个位置可以从自身的原生写入估计下一步的增量。三片盲审中，去除逐位置状态的版本均明显模糊，保留版维持了清晰度。两版有多项差异，单个缓存张量的贡献仍需独立干预。[盲审与原片纹理](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_zero_fused_DEV3/人工判读.md) |
| 实际提速来自哪个比较？ | 24 源中，部署版的四步预测、潜变量和解码 RGB 均复现原锚点版。相对 Dense 的热运行墙钟均值为 40.059→39.470 秒，21/24 源更快。21 个扩展源的 2.394× 来自与带核查的旧锚点实现配对。[计时口径与等价性](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时与等价性.md) |
| 更大探针预算意味着什么？ | 64→96、96→128 两次预算增加均使 11 源×两专家第 19 层的留出点边误差、门控邻边角差和差模正交余能下降。两条 64/128 成片上，1013 手部灰度变化误差下降而位移差上升；固定 128 仍未观测到画面右手指。当前默认保留固定 64。[预算、区域覆盖及成片](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md) |
| 扩大跨层观测范围能恢复细小动作吗？ | 轮换邻点使 80 次调用累计触及 2,472 个坐标，1013 三臂与 Dense 共同可追踪的双手位移差却从固定布局的 3.199 增到 3.464 像素。跨层覆盖只能说明某层读过该处，恢复仍取决于当前层的条件信息。[轮换布局逐片测量](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/逐片方向与区域实测.md) |
| 哪些工程改动有实测依据？ | 末端缓冲复用在 A100 全长单站得到逐位相同的输出，命中 29.521→28.818 ms。无损锚槽布局按容量折算节省 6.69 GiB，RTX 4060 Laptop 的写槽加命中耗时增加。整片等价、墙钟和峰值需分别核验。[A100 单站](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_perf/A100_RESULT.json) · [无损布局](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_memory/GPU.json) |

**人审边界。** 已观看的 11 月 21 日原批视频的运动幅度获审片人认可。0951 左摇的四个新增种子均被观察到此前报告的主体不稳定，审片者在其他已看视频中未见同类问题，该条件作为已知例外暂缓专项排查。[VBench 与观察范围](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/vbench/VBench完整评测.md) · [跨种子人审](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_0951_seeds/跨种子人工复核.md)

如需追溯信息、函数、实际执行三条研究线，包括 Q1r2、局部误差与闭环生成背离、量化和历史原生基元，参见[旧版完整发现谱系](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/research/EXPERIMENT_FINDINGS_ATLAS.md)和[归档清单](../archive/20261121/handoff/README.md)。重用早期发现时，应同时带上当时的模型、来源、精度与实测层级。
