# C2N2 历史运行终止记录（2026-10-24）

**该时间点的历史状态**：2026-10-24用户暂停当时的全部实验；[原始暂停回执](../20261024/EXPERIMENT_PAUSE.json)与七节点清查记录对应这一时点，C2N2监督器及CPU处理均已退出，证据资产予以保留。后续独立用户授权与当前运行状态以 [ACTIVE_RESEARCH.json](../ACTIVE_RESEARCH.json) 和真实RUN为准。

本轮[独立授权](C2N2_authorization.json)及[正式协议](C2N2_registered_protocol.json)已实际执行一次。专用模型环境修复有效越过HiFloat4导入故障，依赖视图与同进程CUDA UUID占用通过；但另一个进程在首次资源采样约65.962秒后进入选中GPU。199次隔离采样中168次有外来计算。Agent复核后只向本次原生PID发出SIGINT，finally处理保留账本并关闭资格，实际退出码−2；未触碰其他作业。

中断时处于LightX2V构建流程的T5 CPU权重读取，完整生成、独立FFN、额外norm/selector、训练、分解及官方评分全部为0。没有配对视频，算法质量与速度均未测，也不需要人工视频审阅。[原生记录](C2N2_native_results.json)、[预算](C2N2_budget.json)、[资源](C2N2_completed_resources.json)、[故障解释](C2N2_isolation_failure_analysis.json)和[63份实际源码归档](C2N2_executed_source_manifest.json)完整保留。

工程证据表明，416MiB的可见CUDA上下文和项目flock不能阻止其他非合作进程之后占卡。这没有检验C2算法优劣；本轮独立单次范围已关闭，资格为空，无自动重试。此前C2N0/C2N1及R16/R17历史记录和PRD科学结果均保持。
