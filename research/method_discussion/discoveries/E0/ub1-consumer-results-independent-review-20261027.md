# UB1 consumer 六视频官方函数：独立结果核对

**已测观察（VIDEO）**：两个已暴露开发来源上，consumer 的连续 RAFT 配对幅值均处于 neutral UB1 与 dense 之间；原版二值 Dynamic Degree 对六条视频均为 False。成像相对 neutral 在两来源提高，但 aesthetic 在犀牛来源下降、住宅来源上升，方向并不统一。这些是两来源各一 seed 的自定义配对，不是总体质量通过。

独立读取[公开官方函数结果](../../../20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.json)及其原 RUN，实际 RUN SHA256 `b19d868babe3033e8e3593d264b8c79d1cd92b7eefdd166e3eda56534be29d85`、协议 SHA256 `6dce8f177eab984f501bcae55de8982de3da2ebbb02a1072146ce202a906df7e` 一致。原 RUN 中8个完成维度的每个逐视频结果与公开机器表按 source/arm/维度比对无差异；6视频解码检查、240次 RAFT forward及4800次迭代、75次GPU监测采样无违规与报告一致。其他维度的预扣额度不等于内部实测forward数。此审阅未重新调用指标或查看原始模型权重。

犀牛 dense/neutral/candidate 的40对平均 RAFT 幅值为3.3932/1.2937/2.1562，住宅为1.9950/0.1833/0.4459；官方阈值规则下六条均 False。连续图像光流不直接证明主体运动质量或动作提示遵循。犀牛 imaging raw 为75.357/74.579/75.336，aesthetic为0.61721/0.60682/0.59225；住宅 imaging为78.191/76.493/77.434，aesthetic为0.68495/0.70900/0.69599。数值取舍与完整帧的[独立静态视觉观察](../../../20261027/UB1_CONSUMER_INDEPENDENT_VISUAL_REVIEW.md)相互补充，不能替代人工原速播放或新来源确认。

原始完成维度为 imaging、aesthetic、motion smoothness、dynamic degree、subject/background consistency、I2V subject/background；camera motion 为正式九维之一且 **NOT_RUN**。Temporal flickering 是额外维度，静态资格未建立，亦 **NOT_RUN**。两来源每臂一条真正生成视频，不是官方每条件五独立采样或完整 benchmark；质量与人类验收未通过。候选[保存latent几何](../../../20261027/UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)仅描述两条轨迹的向量关系，不能证明画质因果。
