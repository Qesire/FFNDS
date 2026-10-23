# C2N0：启动资源检查失败，未开始前向

当前状态为`STOPPED_C2_FIXED_SCOPE_NO_RETRY`。视频、独立FFN、额外norm、selector、训练、分解、官方评分均为0；算法质量与速度未测。没有完整视频，因此本轮不存在人工审阅对象。

北京时间23:14:34取得一张无计算占用的卡；23:17:31监督进程结束，间隔177.356秒。原程序完成CPU资产检查和GPU参数加载后，首次隔离采样发现一个外部计算进程而退出，实际exit1。参数加载56222842880字节、辅助张量851066880字节；失败GPU准备成本已保留。见[原生结果](C2N0_native_results.json)、[扣账](C2N0_budget.json)、[结束摘要](C2N0_execution_summary.json)、[原导出收据](C2N0_completed_resources.json)和[故障解释](C2N0_startup_failure_analysis.json)。

启动器的文件锁是合作式约定；较长CPU准备先于本模型CUDA占用与隔离监控，存在其他程序再次看到空卡的窗口。原导出器的通用PENDING_VIDEO_REVIEW标签不代表实际有视频可审；原始收据保留，故障解释已更正其含义。

## 科学证据与执行范围

原CPU参考在6个暴露来源上，25%采样行捕获下一norm误差从27.8340%到28.8126%；15/72位置下降，其中12处在step2/3。它仍是保存数据诊断，不能替代未产生的C2视频。见[CPU结果](C2_scale_energy_CPU_results.json)和[最新重分析](HANDOFF_PRD_SUCCESSOR.md)。

原独立授权范围为一张空闲GPU、6条完整视频、2次额外norm、3次selector，零独立FFN/训练/分解/官方评估；HF新增结果≤0.5GiB，磁盘保留80GiB，分配峰值≤79GiB。原注册已关闭，`automatic_retry: false`；零前向不产生自动重试权。C1和原R16/R17均不重开。

## 保留的执行前记录

原拟执行旧calibration0 UB1身份视频1条，以及source6/seed42下dense D-Q、UB1、fixed_stride、原动态和尺度动态5条；候选作用全160FFN，每处7917行dense+23751行UB1，保留原支持、alpha、偏置、R256、QDQ与门控，学生不重置。

执行前发现native norm统计可为BF16，已按原FP32评分定义显式转换后平方；[未执行v0注册](C2N0_registered_protocol.json)、[实际使用v1注册](C2N0_registered_protocol_v1.json)、[精度CPU证据](C2N0_prelaunch_CPU_precision.json)和[修正记录](C2N0_prelaunch_precision_fix.json)均保留。原始已执行科学代码不改。

## 已备妥的工程修复

[C2N1固定草案](C2N1_reserved_smoke_DRAFT.json)采用UUID绑定、同模型PID提前建立CUDA占用，并在CPU加载期间监控；六视频生成循环与原版AST相同，原评分/动作/数据不变。[CPU前检](C2N1_startup_CPU_preflight.json)与[合成媒体夹具](C2N1_media_CPU_fixture.json)通过，新GPU执行数0。当前资格为空，等待单独的重试授权。详情见[新交接](HANDOFF_C2N1.md)。
