# 作者自审：source911 配对 Wan FFN 单算子草案

状态 DRAFT_REVIEW_REQUESTED，CPU 配对资产尚须实际 EXIT0 和 inspect-closed；草案不授权运行。准确协议和 GPU job 待父资产闭合后生成，再由不同作者的评议者审确切版本。该路径仅两个 expert、block20、暴露开发源 911、各 32 行；31668 行由这 32 行循环重复，只用于有条件吞吐诊断。

原 BF16 master 和共享 group32 分别取每个 expert 的原 W1/W2，另一个 group16 使用对应 NVFP4 checkpoint 的固定激活 scale、packed 权重、scale、alpha、bias。四臂各按 W1+b1→tanh GELU→W2+b2→相同真实 gate 执行；每臂的 Q2 从该臂当前 GELU 输出生成。组 32 同一个权重 payload 的 Dense 与 SharQ 是主要近格式对照。SharQ 每层实际付动态稀疏/残差量化、sparse GEMM 和 full-K dense residual GEMM；一个 FFN 两次 sparse 加两次 residual，静态 W 量化另计。输出比较 BF16 每个中间量与原捕获 a/h/Y/gate 的 M32 中性门限 0.03；其它量相对相同 M 的重算 BF16，量化质量误差不设通过阈值。M32 每个量化臂各层另以同编码解码 FP32 线性参考验证，relative L2 上限 0.03；SharQ debug 选择逻辑仍与原作者共享，具体反例与额外成本见 AUTHOR_RESPONSE_V2。非同一 checkpoint 的 group16 差异不能解释为纯格式差。

每 expert×M×arm 做 1 correctness、2 warmup、5 measured，另外 2 warmup、5 measured 仅用整体 CUDA event；共 240 次 FFN。每次整臂预扣实际 Python 调用上限，逐入口 invoked/returned，完成后持久化；失败时可出现预扣与调用不同、CUDA 异步完成未知。两组分别测逐操作 CUDA event 与仅整体 CUDA event，均记录带调用账本的 wall；event 插桩及顺序固定可能影响速度，不是公平系统性能。setup 的静态量化、bundle hash 与行重复另计 wall；H2D 按预扣/同步完成字节，额外 observer 中的张量归约是 16 次比较调用，内部 GPU kernel 数未知。CPU 单次闭合后由 GPU 范围预扣并重哈希 bundle；源 packet/full master 来自 CPU producer 的原实际证明，非合作写者竞争不能完全排除。

真实 worker/job、每卡 UUID 锁、原作者模块与编译 ELF、LightX2V 已有软件身份及 Python/package metadata 均须冻结；环境身份只覆盖 metadata/RECORD 和明确列出的源码/二进制，并非所有 Python wheel 字节的全面证明。GPU0 一张，4 CPU，3600 s，额外 supervisor 3720 s，输出 512 MiB 软监控，主机 RSS 16 GiB、CUDA reserved 72 GiB；内存为采样或操作边界检查，并非硬隔离。离线、无重试、原始错误账保留。无完整视频质量、真实性能通过或系统速度结论。

请求独立审查特别针对：原作者共享 weight scale 与 sparse/residual API，输入 shape/layout 和 LightX2V scale；M32 捕获比较的数值门限；预扣/异步失败语义；16 个 correctness 比较及 event 插桩成本；各路径的 setup、H2D 和 GPU/CPU 资源预算；真实 CPU 上游及作业绑定。请给可能推翻实验的实际反例。真实父资产和该版本修订前，任何 APPROVE 都不能授权最终 scope。
