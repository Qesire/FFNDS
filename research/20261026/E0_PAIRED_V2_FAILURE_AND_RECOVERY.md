# E0 · 双轨DD视频完全中性，后处理路径失败，逐点记录未保存

本轮已真实完成来源0、seed42的两条四步dense轨迹及两条81帧视频。**teacher/student两MP4的SHA都为`23f5063d3565f672ae82244ddaa7d18591ee84f2484ac8b5a0ab21dea75b1ee4`，与原dense视频逐字节相同。** 单张空闲A100，2秒物理采样峰值73475MiB（71.75GiB），同卡隔离采样通过。这证明本次DD最终视频未改变；不代表候选视频质量、部署速度或完整逐点证据通过。

[冻结结果](E0_PAIRED_V2_FAILURE_AND_RECOVERY.json)绑定失败RUN、STUDY、两条视频及失败逐帧报告。每条视频完整解码81帧，保留运动/细节等取证；报告明确缺失完整pair/setup/FFN计时，不能通过完整成本门禁，官方及人工质量仍NOT_RUN。

失败原因是实现错误：旧结果的`HF_relative`相对`HF_HOME/ffnds-derived`，入口误拼为`HF_HOME/HF_relative`。异常发生在两次解码、原生计数断言之后，历史指纹核对及TRACE写出之前。两轨逐block相等断言和四步执行已通过对应控制流，但**160处详细标量、四步latent哈希及精确计时未落盘，无法恢复**；保存trace覆盖必须记为0，不能从视频相同补造历史latent或逐点观测。原失败回执与v2协议保留原字节，余下两组DC停止，未重试。

修复统一历史路径，并在CPU预检中读取实际历史receipt、唯一source/seed及视频SHA；生成后先写TRACE/OBSERVATION，再进入后续解码、计数和历史检查。Python异常时保存已记录几何和完整完成数，标识尚未完成strata/vector的尾site；硬终止仍可能丢失未写内存。controller、observer、residency及trace数学组件与v2相同。8项driver软件测试通过，增加了真实目录层级、越界路径及partial保存的回归检查；总计28项双轨/严格层软件测试。[独立v6源码审阅](../method_discussion/discoveries/E0/independent-paired-driver-path-persistence-audit-20261026_v6.md)限于这些路径/留存修复，不声称运行测试或模型。

另备[v3固定范围](../../config/e0_paired_full_depth_20261026_v3.json)，SHA`3027bc17588efdcdab32aacc5084cca6705b58d76f75529be43edeeda1b7bcc5`：仅补来源0/12的两组DC，4条完整四步视频、320处配对观测，单张空闲A100。复用旧DD两视频只作为最终视频中性证据；新teacher/student仍各需匹配原四步latent及视频指纹。无拟合、训练、分解或官方评分，新增≤0.5GiB，任一失败即停。**v3当前未授权、未启动。**

解释本轮/后续记录还须保持三个限定（已独立只读核对）：

- 32行native–continuous与full-M operator SSE的覆盖、分母及参考函数不同，不能直接比较来判定数值效应主导。连续诊断也包含移除Q2等预期执行差，不能自动视作实现bug。
- DD实际两轨均dense，仍计算candidate shadow。四臂operator SSE可非零；中性看实际q1/block/latent及视频身份。
- 当前普通I2V中，两轨每step的embed0被强制相等，同一block的gate只由modulation+embed0产生。故同step/block gate相同；writeback prefix差来自residual基底、dense FFN输入响应与有限精度乘加，不能将gate变化当作本配置的观测解释。

这些事实限定P23/P24的证据范围；两节点仍OPEN。没有新的候选选择、因果归因、正式质量接受或stage提升。
