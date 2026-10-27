# 固定R256 consumer：六条原生完整视频与中性复现

**VIDEO/NATIVE**：原支持54/108/108/162、alpha和R256保持固定，仅替换所选W2/bias后，两条开发来源均能完成40层×4步、81帧自由视频。新鲜Dense与neutral原UB1在两来源上都逐字节复现历史四个FP32 latent及完整MP4；candidate在该中性门槛通过后才执行。此结果把训练内恢复推进到真实生成，但两来源的质量仍有取舍。

来源为seed42的犀牛（formal20 index0、source882）和山坡住宅（index10、source80）。两图SHA均不在原24校准图中，但这些已分析的例子属于DEVELOPMENT；四步、160位置和81帧是嵌套观测，不增加独立来源数。每来源分别新生成Dense、neutral、candidate三条视频，共六条，没有复制五个文件名充当独立采样。

| 来源 | Dense全160 FFN秒 | Neutral秒 | Candidate秒 | Farneback候选/参考逐对幅值比：neutral→candidate |
| --- | ---: | ---: | ---: | ---: |
| 犀牛 | 9.7444 | 6.1382 | 6.1544 | 0.375→0.770 |
| 住宅 | 9.5885 | 6.0625 | 6.0837 | 0.429→0.589 |

Candidate与neutral使用相同主分支形状、R256次数及QDQ边界，当前全FFN时间相近。生成wall还包含四步观察、保存、哈希和视频输出，且固定Dense先行只有一次技术测量，存在冷启动/自动调优混杂；本轮不能据68.998→48.478秒或74.252→49.474秒声称可信端到端速度比。历史R17热身计时保留其自身对象和1.0757倍结果。

全81帧core指标六条均通过合同检查。运动幅值增加同时伴随paired RGB RMSE上升（犀牛0.12074→0.12522，住宅0.15792→0.16715），paired temporal RMSE也上升；轨迹差和感知功能需分开评价。[八轴官方函数结果](UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)进一步显示成像改善与部分运动恢复，同时犀牛美学下降；八轴属于两来源自定义子集，camera motion未运行，静态闪烁资格未建立。独立Agent核对全部81帧接触表后记录[可见结构与运动差异](UB1_CONSUMER_INDEPENDENT_VISUAL_REVIEW.md)，完整人工播放验收仍待完成。

本次原生生成合计24次完整flow、1920次主FFN线性、1280次RR线性、7680次非FFN线性、9600次QDQ，无新拟合或训练。两张A100上的独立任务均实际退出0，423次资源采样无违规，HF新增约321MB。模型参数和大结果留在HF，公开[机器结果](UB1_FIXED_RR_CONSUMER_VIDEO_RESULTS_20261027.json)绑定原RUN、LAUNCH_EXIT、各视频、GENERATION、四步latent与core报告SHA；协议SHA为`ba4a4f8f03a3cdd8c2c796d054f0ff1847e8eab61a6d3e11b1ac477b3710f8a5`。

**下一项判别**：冻结当前bank及所有超参，在已暴露的旧disjoint20上新跑成对Dense/candidate，按最新数据协议保存并严格核对真实初始噪声、逐来源收据和完整160处执行。比较来源级成像尾部、连续运动及其它质量轴，区分两个例子的局部改善与更广来源覆盖；该开发集扩展仍不替代新确认集或正式五采样VBench。
