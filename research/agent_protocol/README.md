# 研究与复算协议

[研究导航](../RESEARCH_NAVIGATION.md)提供当前入口。[交接上下文](CURRENT_CONTEXT_PACKET.md)列出基线和候选，[机器可读交接](../CURRENT_HANDOFF.json)提供脚本使用的入口。实验范围、成片与计时均应追溯到相应版本的记录。

| 需要回答的问题 | 可复用的资料 |
|---|---|
| 校准、探针和评估是否独立？怎样组织同源配对？ | [采样、校准与评估](DATA_SAMPLING_CALIBRATION_EVALUATION.md)。记录条件图来源簇、提示词、种子及初始噪声。分别统计独立来源数、同图变体和重复运行。 |
| 某项论文机制在 Wan 里改变了什么？ | [论文与计算阶段映射](PAPER_METHOD_FOCUS_GRAPH.json)、[计算图绑定](GRAPH_AGENT_BINDINGS.json)、[原始资料索引](ACTIVE_SOURCE_MANIFEST.json)。沿真实算子边标出信号可用时点、张量形状、写入和消费者。 |
| 哪部分视频质量已评价？ | [视频与全深度评估协议](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)。方向指标需带活跃区覆盖，运动补偿后的细节指标需带共同可跟踪区域，审片结论关联原片与具体变体。 |
| 怎样复查锚点读出的速度与数值表现？ | [原版 24 源计时与等价性](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_scale24/计时与等价性.md)、[A100 单站缓冲复用](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/work/anchor_next_perf/A100_RESULT.json)、[两源固定预算配对时间](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/budget128/配对计时.json)。分别读取同配置的输入、逐步哈希、设备、预热与计时窗。 |
| 需要追溯量化或原生计算图？ | [权重与激活操作数时序](WEIGHT_ACTIVATION_QUANTIZATION_TIMING_CONTRACT_20261028.json)、[HiF4 格式记录](HIF4_SIGNED_FORMAT_CONTRACT_20261028.json)、[上游资料同步](UPSTREAM_SYNC.md)。格式和设备结论以各自实验环境为范围。 |

过去的硬件登记、B0/B1 任务表、执行合同和原始调度条件可从[旧版协议索引](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/research/agent_protocol/README.md)查阅。需要核对旧入口身份时使用[交接归档](../../archive/20261121/handoff/README.md)。
