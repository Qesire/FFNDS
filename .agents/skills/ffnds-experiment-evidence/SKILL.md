---
name: ffnds-experiment-evidence
description: Trace FFNDS calibration, model-state measurements, paired video generation, quality and latency claims to their original evidence.
---

# FFNDS 实验与成片证据

从[接手说明](../../../research/agent_protocol/CURRENT_CONTEXT_PACKET.md)和[实验证据索引](../../../research/EXPERIMENT_EVIDENCE_INDEX.md)确定要复算的版本、对照臂及其统计单位。针对一个新候选，先写清打算从测量中区分的两种结果。

- **来源与条件。** 分开登记离线校准、已反复观察的开发视频和独立评估。记录图像、完整提示（含镜头指令）、种子、初始噪声、专家调度和实现版本。同图的多个镜头指令与种子用于同源配对。来源独立性按原始条件图核查。[采样与评估](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md)
- **算子与方向。** 在同一专家、层和去噪步追踪实际可见的信息。对跨步增量及相邻位置的差执行留出检验，同时检查门控后的有符号投影、角度、正交余能与共同／差分误差。标出关键区域是否真正被探针直接观测、跟踪覆盖是否一致。局部 MSE 可作补充。[研究导航](../../../research/RESEARCH_NAVIGATION.md)
- **成片与人审。** 对同源同种子的生成记录核对纹理、细小动作和轮廓稳定性。原片链接和逐来源量化表一起交付。同源同种子比较优先报告末步 flow 与最终 latent 的相对 L2、解码视频的 RGB RMSE。涉及动作时报告帧间差分或光流等时间指标，画质与语义评估采用相应版本可计算的 VBench 分项。各项列出参照、单位、指标方向和配对差值，保留能区分细微差异的精度。标明样本数、测量缺口和人审范围。使用原生 Dense 作参照时也要核查它与提示的符合程度。保留失败样例，真实视频的最终判断交给用户。[当前正负结果](../../../research/EXPERIMENT_FINDINGS_ATLAS.md)
- **运行与成本。** 记录设备、模型精度、实际计算的探针行数、驻留和临时分配、预热、计时窗和重复次数。功能等价的改动先核对预测、潜变量和 RGB；分别报告组件和完整生成的计时。保留原始状态、结果、配对键和文件指纹以便复算。

历史比较使用过不同模型、部署方案和指标覆盖。追溯 10 月的完整质量轴、资产核验和相关协议时，使用[当时的技能原文](https://github.com/Qesire/FFNDS/blob/22d60b645e070ed528b3b435ec0cbe4b26467108/.agents/skills/ffnds-experiment-evidence/SKILL.md)和[归档清单](../../../archive/20261121/handoff/README.md)。
