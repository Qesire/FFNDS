# 同预算value10/DCAC10：完整视频八轴结果未达到要求

真实40条新候选的八项原版官方函数全部完成并actual exit0；20条Dense评分直接复用同片、同保护实现和环境的旧成功收据。来源等权连续RAFT幅值为Dense的 **0.450623 / 0.439311**。训练图像上AC相对误差改善7.73%，没有转成这二十来源自由视频的运动/画质收益。DCAC10对value10的八轴平均差均负，连续RAFT均值也降低约2.51%。这是已测试目标/秩/来源的有界负结果，不推翻S2或所有时间表示。

| 原版指标（来源等权） | Dense，复用 | value10 | dcac10 |
| --- | ---: | ---: | ---: |
| imaging_quality | 73.720461 | 71.319122 | 70.365716 |
| aesthetic_quality | 0.625203 | 0.569999 | 0.562486 |
| motion_smoothness | 0.983232 | 0.979375 | 0.978567 |
| dynamic_degree | 0.350000 | 0.100000 | 0.050000 |
| subject_consistency | 0.963478 | 0.967038 | 0.963803 |
| background_consistency | 0.967457 | 0.969523 | 0.965603 |
| i2v_subject | 0.983554 | 0.964693 | 0.957712 |
| i2v_background | 0.991705 | 0.968857 | 0.965535 |
| 连续RAFT top5% px/2frame pair | 7.058015 | 3.180503 | 3.100664 |

Dynamic Degree为7/20、2/20、1/20。静态一致性单项接近不能抵消主体运动与I2V条件保真下降。新的280个video-axis结果、1600次RAFTforward/32000iterations已实际扣账；140个Dense轴结果和20条Dense RAFT序列复用，不构造新的官方aggregate。旧shared/step结果不混入新两候选。逐源尾部、scene分层、精确函数/资产及RUN/EXIT在[机器报告](RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.json)，重放工具只读原始标量，重新汇总与实际RUN逐字节数值一致。

**科研判断**：同输入训练目标改善与完整video目标存在缺口。两种主要竞争为（1）新10源校准的函数方向不能泛化到未拟合的运动区域；（2）局部压缩误差进入student前缀后被attention/后缀放大。去τ0的Dense创新还显示弱相邻相关、高时间秩，支持继续采真正空间邻域而非仅调整时间权重。下一有限采集复用这轮已生成视频/十状态作为被动observer的byte-neutral对照，增加真实teacher/student同坐标邻域。

**已经做的前置**：独立CPU ROI scope actual exit0，在既有Dense官方RAFT最强subject898与camera96的完整81帧上，各用80次相邻帧Farneback得到5x5原生邻域及2个远距低流控制。它选择高流区域，既不保证语义主体，也不提供真实flow对应。原始鸟图可见主体居中而最大流ROI落在右下植物背景，因此下一采集会另外包含预先固定的中央5x5，区分背景平移与中央内容；不会把纯max-flow ROI称为主体定位。无新candidate选择、训练或质量阈值扫描。

该结果为20历史暴露开发image clusters×单seed42的自定义配对证据，正式五独立样本、动作/相机语义、额外flickering、人类全视频与公平预热SYS尚未测；当前方案质量未通过。独立Agent服务额度耗尽，本轮记录为协调者审阅而非独立支持票。
