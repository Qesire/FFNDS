# 作者回应独立初审 V1 — 待 exact recheck

1. 新增每 part 独立 `..._VERIFY` CPU scope，独立 ACTIVE/reviewer/spec、原 worker profile、1 CPU、0 GPU、1800s（supervisor 1920s），fresh verifier RUN；读46个artifact逐项持久预扣，全部完成仍须真实 verifier EXIT0 后 `inspect-verification` 才给 closed。GPU parent 原 RUN 与真实 EXIT0 先核；无授权不能触发大读取。状态仍待独立复审，不代表登记。
2. CPU preflight 与 GPU 启动分别在 RUN 内预扣 512MiB 完整SHA读取预算；现有 source、ELF、模型元数据、图像、镜像包摘要入口纳入计数，出现失败时保留已完成字节与当前文件 UNKNOWN。小配置/receipt读取另列 UNKNOWN；不重新对原 checkpoint 全文 SHA。两阶段预算独立，全部旧 source/作者资产身份仍固定。
3. 文件 guard 进入移入 GPU 受保护的 try；失败经 except 写 FAILED，再 finally 停 monitor、关闭账本，仅成功进入时退出 guard。输出 soft budget 及失败账保留。
4. Python调用链已核实：原 infer_with_blocks_offload 同一线程顺序调用 prefetch_weights、infer_block、swap，manager 的 load_state_dict 同步回调；唯一 executor.submit 属 lazy_load=true 分支，冻结 runtime_overlay.lazy_load=false。CUDA 两流可重叠，但不构成这些 Python 回调并行，源码在 closure；审稿人同意以此证据解决问题。额外后台 UsageMonitor 仅读目录并不修改 WrittenBytes/journal。

新方案仍按四视频每源、原 prefetch、第二BF16阶段建160份缓存、候选同缓存；warm次序与观测成本限定原wall的解释。24项合成测试通过，未执行模型、GPU或新资产读取。
