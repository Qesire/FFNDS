# C2N0：固定范围已授权，等待空闲 GPU

实际完成 C1 后，继续用保存数据比较了原 R16 门控补偿能量的 CPU 参考与一个解析修正：`score = gated_R256_energy * current_norm2_rstd²`。同样选84/336采样行，捕获下一norm误差从27.8340%到28.8126%，六来源均提高；事后采样oracle为36.7028%。见[实际参考结果](C2_scale_energy_CPU_results.json)、[先于评分暴露固定的计划](C2_scale_energy_CPU_plan.json)。没有新增原生FFN或GPU调用；72次CPU第二R256线性重建不称作原GPU评分逐位复现。

额外验证12位置各24份旧校准gate完全逐位一致。源代码中gate由固定时间嵌入与block调制形成，独立于图像/文本内容；C1没有保存gate，因此CPU参考仍依赖该源码条件及相同调度/网格，且CPU BF16线性与归约可不同于GPU。原生候选会直接使用当次真实gate，不复用校准gate。

比523维读出便宜的这个候选已选为**待完整视频判别的备选**，不是已验证方法。选择原因是它复用现有两个物理分支、无需训练/跨层坐标外推，在线只暴露原norm2统计并多做FP32平方与相乘；仍保留原能量代理未必等于真实误差这一缺点。不是为了继续优化无收益的支持交换。状态登记在[候选记录](../candidate_rollout/registry/C2N0_norm_scaled_energy.json)。

## 可直接审阅的固定原生范围

[草案](C2N0_native_smoke_DRAFT.json)已绑定实现与资产SHA，一张空闲GPU，最多6条完整视频、2次额外归一化、3次选择器检查，零独立FFN、训练、分解、官方评估。新增HF输出≤0.5GiB，磁盘保留80GiB、峰值≤79GiB；失败扣账，不自动重试或扩展。

1. 旧calibration0一条UB1视频：全部160处改为读取norm统计，但分派不变，要求原初始噪声与四步latent逐位一致。
2. 原R17 source6 `i2v_subject:91` 同条件五条完整视频：dense D-Q、UB1、旧实测较强fixed_stride、原R16动态、新尺度修正动态。它是按已保存任务元数据选出的**第一个从未生成且不在C1中的来源**，不是看过新候选后挑出的画面。
3. 候选四步全160处真实生效，固定每处7917行dense+23751行UB1；保留原支持、alpha、偏置、R256、量化与门控。学生轨迹不重置，保存媒体、四步哈希、全部路由掩码及单条真实wall时间；单次时间不冒充重复测速。
4. 完成后解码并完整审看候选/dense视频，覆盖主体、运动、背景小物体与稳定区域光照；人工完整播放结论之前不自动判为smoke通过。

程序：[执行入口](../../tools/run_c2_norm_scaled_smoke.py)、[统计/评分挂钩](../../tools/norm_scaled_row_dispatch.py)、[有限预算/授权检查](../../tools/c2_contract.py)。合成测试覆盖真实调用顺序、基线值不变、异常恢复和缺少授权时的执行拒绝；它们不是原生数值控制，后者在上述固定范围内。

用户已明确回复“授权”，见[直接授权](C2N0_authorization.json)和[独立正式注册v1](C2N0_registered_protocol_v1.json)。原草案及固定范围不变，算法、六条视频及所有资产SHA均再次核对。模型资格仅开放此固定C2范围，C1与旧R17不重开。

在任何模型/依赖执行前，CPU合成检查确认原生norm统计可为BF16（rstd=3.359375；直接BF16平方11.3125，转FP32后平方11.285400390625）。为实现原草案已明确的FP32评分，暂停了自己的CPU排队器并显式增加统计量float转换，测试现在会拒绝BF16直接平方。零GPU/原模型调用，六条/2norm/3selector预算、数据与动作均未变；[未执行的v0注册](C2N0_registered_protocol.json)逐字保留，v1重新绑定修正后的两份源码。见[精度证据](C2N0_prelaunch_CPU_precision.json)和[启动前修正记录](C2N0_prelaunch_precision_fix.json)。这不是模型重试。

执行预检发现两台已授权服务器的六张卡均有计算进程。已启动一个脱离SSH的CPU-only排队进程：每15秒检查当前执行服务器，先取得GPU排他锁、再核查无计算占用，随后exec唯一一次模型任务。不会与已有GPU进程共卡，也不会在模型失败后自动重试。排队器为[worker_wait_idle_exec.sh](../../scripts/worker_wait_idle_exec.sh)，已纳入正式注册的源码绑定。私有队列/后台进程心跳保存在忽略的运行目录。

此排队快照尚无C2模型输出。优先完成已授权的配对完整自由视频；等待期间仅核对执行和媒体审阅程序，不继续扫描评分参数。实际运行完成或停止后，按真实扣账和媒体覆盖更新本文件。

## 最新主线合并

已合入`9778e8e`的[完整残差动作价值研究](HANDOFF_PRD_SUCCESSOR.md)。C2仍承担原六视频判别；原R16/R17只作历史匹配对照。保留新重分析中15/72位置下降、其中12处在后两步的异质性；不删位置、不改当前评分或预算。

## 本次固定尝试的实际结束状态

`STOPPED_C2_FIXED_SCOPE_NO_RETRY`。实际扣账：`{"full_generations": 0, "official_evaluator_calls": 0, "standalone_FFN_forwards": 0, "standalone_normalization_calls": 0, "standalone_selector_calls": 0, "training_steps": 0, "weight_factorizations": 0}`。

固定原生尝试已停止，无自动重试。尚无完整媒体审阅结果。见[实际执行](research/20261023/C2N0_execution_summary.json)和[当前交接](research/20261023/HANDOFF_C2N0.md)。

模型固定范围已经闭合，资格为空。完整配对预览和全部帧接触表留在HF缓存；人工完整播放结论仍待确认，没有自动质量通过。
