# C2N1：启动修复已备妥，未获独立重试授权

原C2N0在GPU准备阶段检测到同卡外部计算而停止，视频、FFN、额外归一化、选择器均为0次。算法质量/速度未测；[原始结果](C2N0_native_results.json)、[故障解释](C2N0_startup_failure_analysis.json)和[技术修复](../../docs/89_C2启动竞争与同进程占用修复_20261023.md)可直接核对。

修复只改变资源取得顺序：NVML UUID绑定、同一模型PID提前建立CUDA占用、资产加载前确认NVML可见、准备期间持续隔离监控。新增CUDA上下文的实际显存和准备时间会计入报告。六视频生成循环与原版AST相同，原物理动作、评分、补偿、全160位置和数据均不变；不是新的算法创新或旧路由扫描。

[固定草案](C2N1_reserved_smoke_DRAFT.json)：一张空闲GPU；旧calibration0 UB1身份对照1条，source6/seed42的dense、UB1、fixed_stride、原动态、新尺度动态5条；另2次norm、3次selector，零独立FFN/训练/新分解/官方评分。HF新增结果≤0.5GiB，磁盘保留80GiB，分配峰值≤79GiB；单次尝试，失败保留扣账，无自动重试。输出使用新目录，保留原失败记录。

[CPU前检](C2N1_startup_CPU_preflight.json)和[486合成帧处理夹具](C2N1_media_CPU_fixture.json)通过，新的GPU执行数为0。GPU上的早期占用、数值控制和真实视频仍未验证。原C2N0注册不重开，当前模型资格为空；草案、候选记录和这些CPU检查均不授予重试权限。

入口：[新执行器](../../tools/run_c2_reserved_smoke.py)、[同进程占用](../../tools/startup_gpu_reservation.py)、[新授权与预算检查](../../tools/c2_reserved_contract.py)、[UUID排队器](../../scripts/worker_wait_uuid_idle_exec.sh)。完成后用[CPU媒体处理与导出](../../tools/process_c2_reserved_results.py)和[离线结果检查](../../tools/check_c2_reserved_results.py)，再核对完整视频与真实单次耗时。无视频时不要求人工审阅；有完整配对后才请求全程播放结论。

正式运行前仍需新的用户授权、正式协议及代码/资产SHA绑定。新实例固定范围与原科学假设相同，原C2的未使用额度不能自动用于它。R16/R17保持退役，C1交换动作保持关闭；后续科学主线仍见[残差完整动作价值交接](HANDOFF_PRD_SUCCESSOR.md)。
