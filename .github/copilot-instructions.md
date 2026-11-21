# FFNDS 仓库协作

先读根目录 [AGENTS.md](../AGENTS.md) 和 [研究导航](../research/RESEARCH_NAVIGATION.md)。继续已有任务时，[交接上下文](../research/agent_protocol/CURRENT_CONTEXT_PACKET.md)说明当前方法、已知例外和候选所处的验证阶段，[机器可读交接](../research/CURRENT_HANDOFF.json)提供对应入口。

默认比较对象是 Wan2.2 四步图生视频的固定 64 探针逐位置锚点版。记录实现身份时写清专家、层、去噪步、门控前后、探针数、分块行数及精度。解释误差时，分别看位置写入、本帧相邻位置的相对变化，以及完整视频的运动方向、纹理与轮廓。未决的视觉质量判断留给审片人。[方法与当前证据](https://github.com/Qesire/FFNDS/blob/2c4aa753761194994ee25ea8caba76a29445912d/reports/20261121/anchored_video_experiments/outputs/FFNDS_anchor_next/当前方法.md)

复用已有输入、原片、运行回执和官方指标。把每个计时写清为单层组件、去噪步或含解码与写出的整片墙钟，并说明设备、预热、重复次数及峰值显存口径。涉及原论文时引用论文或作者代码，将提出的机制与对应实测分开。[论文与方法研究](../.agents/skills/ffnds-paper-discovery/SKILL.md) · [实验取证协议](../research/agent_protocol/README.md)

旧研究阶段的设备、任务优先级和审批记录可从[历史原文](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/.github/copilot-instructions.md)查阅。归档范围及原始版本见[交接归档](../archive/20261121/handoff/README.md)。
