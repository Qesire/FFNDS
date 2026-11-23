# FFNDS · Wan2.2 图生视频的前馈网络加速

FFNDS 研究如何减少 Wan2.2-I2V-A14B 在相邻去噪步重复执行的前馈网络（FFN）计算，同时保住主体动作、轮廓和纹理。**当前用于成片对照的基线是逐位置保留锚点、固定 64 个探针的 `wide8192`。** [默认四步入口与复用方法](tools/ffnds_fourstep/README.md) 新接手的 agent 请先读[当前接手说明](research/agent_protocol/CURRENT_CONTEXT_PACKET.md)。自动化工具从[机器可读交接记录](research/CURRENT_HANDOFF.json)取得方法、实现和固定证据版本。

## 当前方法及其代价

四步推理依次经过高、高、低、低噪声专家。同一专家第一次调用时，40 层都为每个时空位置保存原生 FFN 锚点。第二次调用时，每个位置从自身锚点估计本步输出，再用 64 个位置的真实 FFN 响应校正全场。联合拟合同时约束探针上的变化和相邻位置之间的变化；探针位置直接写回原生响应。两个专家分时复用锚点槽。命中步按最多 8,192 行执行 GEMM，锚点直写，最终读出原址累加。[算子与实现](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/保留锚点方法总结.md)

| 已验证的比较 | 结果 | 解释范围 |
|---|---|---|
| BF16 与合并写入的 INT8 锚点缓存，四步 wide8192，同卡十片 | BF16 → INT8：FFN 含门控后处理 4.83165 → 4.81399 秒，10/10 加快；整片 37.90009 → 37.93549 秒，3/10 加快 | 默认使用 BF16。省显存变体 `wide8192_all_int8_joined` 的单独运行峰值为 32.77 GiB，BF16 为 47.57 GiB。十片输出与原 INT8 缓存逐字节一致。[同卡计时与画质证据](reports/20261123/wide8192-cache/README.md) |
| 单张 A100，3 个开发来源，seed 42，832×464、81 帧 | Dense 平均 41.532 秒，`wide8192` 平均 39.440 秒，耗时下降 5.04% | 正、反序共六对全部更快，每片节省时间的中位数为 1.437 秒。[逐次计时](reports/20261122/fourstep-wide8192-185220/timing.json) |
| 历史 `fast512`，单张 A100，24 个来源，seed 42，832×464、81 帧 | Dense 平均 40.059 秒，`fast512` 平均 39.470 秒，21/24 个来源更快，约 1.015× | 热整片墙钟，每源每臂正式计时一次。优化前后的**锚点实现**在 24/24 源的预测、潜变量和解码 RGB 逐字节一致。[计时与等价性](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时与等价性.md) |
| 历史 `fast512` 的显存 | 锚点槽 30.20 GiB，候选运行峰值 47.57 GiB | 两个数字分别表示锚点本身与运行整体；缩减层数会同时改变方法。[当前方法](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md) |
| 增至 128 个探针；或复用末端缓冲 | 增加观测改善了第 19 层的留出统计；两个开发视频的动作和纹理有得有失。缓冲复用在 A100 单站从 29.521 降到 28.818 ms | 这项单站测量隔离了缓冲复用的耗时。上行 `wide8192` 的整片计时包含所有改动。[预算与实现证据](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md) |

质量评价使用联合误差的方向和空间结构、运动补偿后的纹理、配对成片指标和同步审片。原批已审视频的运动幅度获得人眼认可；0951 加左摇指令后，主体不稳定在多个种子复现，已记录并暂缓专项排查。Dense 也会生成与提示不符的画面。视频的最终视觉判断由用户给出。[质量与反例索引](research/EXPERIMENT_EVIDENCE_INDEX.md)

## 按任务进入

| 需要做什么 | 入口 |
|---|---|
| 接续研究、定位脚本与原表、核对复现条件 | [接手说明](research/agent_protocol/CURRENT_CONTEXT_PACKET.md) · [研究导航](research/RESEARCH_NAVIGATION.md) |
| 分辨已验证的结论、反例和待测解释 | [实验发现](research/EXPERIMENT_FINDINGS_ATLAS.md) · [实验证据](research/EXPERIMENT_EVIDENCE_INDEX.md) |
| 修改代码、登记方法、核查环境 | [仓库工作说明](AGENTS.md) · [方法讨论](research/method_discussion/README.md) · [环境与技术文档](docs/README.md) |
| 复查旧结论、旧任务、旧设备或旧版首页 | [归档索引](ARCHIVE.md) |

11 月 21 日的生成实现与实验记录随 [研究 PR #61](https://github.com/Qesire/FFNDS/pull/61) 发布。以上实验链接固定到已核对的提交。新检出可以直接核对记录；重新生成同一批视频还需要对应的模型和蒸馏权重、输入图、离线校准包以及兼容的运行环境。[固定版本的生成入口](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchored_scale/generate.py)
