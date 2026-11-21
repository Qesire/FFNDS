---
name: ffnds-paper-discovery
description: Map FFN research from primary papers and code to testable claims about information, output changes and physical execution in Wan video generation.
---

# FFNDS 论文与方法研究

从 [研究导航](../../../research/RESEARCH_NAVIGATION.md) 确定当前比较对象。研究结果、已审视频和仍待验证的改动见 [当前上下文](../../../research/agent_protocol/CURRENT_CONTEXT_PACKET.md)。检索、推导与实验分别对应不同的问题。

## 将文献映射到真实计算

1. 从原文和作者实现定位方法实际读取的输入、使用的历史状态、变化的算子和下游消费者。记录模型、精度、张量形状、信息可用时刻及对照臂。可从 [一手文献检索方法](../../skills/web-research-discovery/SKILL.md)、[论文与阶段映射](../../../research/agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json) 和 [图中算子绑定](../../../research/agent_protocol/GRAPH_AGENT_BINDINGS.json) 找到原始材料。
2. 写出在 Wan 目标位置具体发生的计算。对折叠、稀疏和缓存，分别追踪被消去的矩阵乘、仍需运行的非线性、动态选择或校正开销、常驻和临时显存。例如权重量化比例可以离线确定，当步激活的尺度取决于输入；查询某个标签之前的选择器只能使用当时可得的信息。[执行图与证据](../../../docs/98_端到端计算图与证据注解_20261025.md) · [量化操作数时序](../../../research/20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)
3. 为机制写出可以区分它与另一种解释的观察。以锚点读出为例，扩大探针预算和改变探针分布分别对应“更多当步信息”和“同预算更好的位置覆盖”。在相同留出位置比较方向、有符号投影和正交误差，再看同源同种子视频的局部动作、纹理以及热运行延迟。[锚点方法的实测对照](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md)
4. 区分原论文所用的硬件和系统组合、数学等价、真实 BF16 面板输出、完整生成和人审结论。算子加速须说明实际测量边界；研究质量须交代独立来源、校准与评估的重叠以及细小区域能被可靠跟踪的比例。[校准与评估资料](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md) · [完整生成的历史评估口径](../../../research/agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)

可用 `python tools/graph_agent_bridge.py paper L74` 查特定论文的实现和适用条件，用 `python tools/graph_agent_bridge.py focus T06` 查同一计算位置的候选。图谱中的论文条目提供来源线索，Wan 的原生效果取决于对应模型、执行和结果记录。

此前的 SharQ、TEAL、Chipmunk 等优先级和设备名单属于各自阶段的记录。按当时路径追溯原评议、条件或实验登记，可从[旧版技能原文](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/.agents/skills/ffnds-paper-discovery/SKILL.md)或[交接归档](../../../archive/20261121/handoff/README.md)进入。
