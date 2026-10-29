# SharQ 同原 BF16 master 的 Wan 全160 FFN 路径：工程设计，待评审

状态：STATIC_IMPLEMENTATION_REVIEW_REQUESTED。当前没有本范围的注册、模型/GPU运行或视频。问题是：同一个原生权重组上，作者 sparse backbone + full-K dense residual 的单 FFN 精度改善是否延续到四步自由生成，其付费量化、权重处理、原始预取和模型总延迟是多少？两个可推翻的预测：A，局部误差改善在两来源视频的动作和外观也有可见改善；B，非线性和四步累积抵消、放大或改变误差顺序。每个 source 是独立观察单位；两来源都是已暴露开发源，seed42 是一条独立生成而不是泛化确认。

已闭合的前提是独立 SharQ V2 单算子 RUN SHA `5921024ab75bd6cc3d12d8ec5811aea40c90a96c2a3368fe6a47300603b76adc`、actual EXIT0。其 M31668 是 32 真实行重复的条件形状诊断：high/low 的整体 FFN CUDA 时间，BF16 为 31.3663/31.3978 ms、同 W Dense32 为 20.7780/20.7668 ms、SharQ 为 28.9571/28.9559 ms；门后相对 L2 为 Dense32 0.0847214/0.0850583、SharQ 0.0647624/0.0658341。这为下游视频提出有区别的预测，不证明真实 fullM 状态质量或公平端到端速度。

## 输入和真实位置

原 master 为 `lightx2v/Wan2.2-Distill-Models` revision `52699089e1dac804cbedd2400d0d2aea969577de` 的四步 I2V 两个 BF16 checkpoint：high SHA `39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa`，low SHA `a3b8680affe2ab2e2121210a83ffa7d351f77da18191e57bb359d7839beba8a2`，各 28,577,915,880 bytes，header SHA 相同 `f40c5e3943554fa03cf2b873620e0d3293fcbfa89825d7c52b291ef6533cc67c`。此前闭合 CPU 证据给出实际全文 SHA 与身份；本次目标 worker 只读 stat 检查两文件均存在、大小相符，且与此前证明的设备/inode/时间身份相符；本次没有重新读大权重。完整 40-block header 的 `ffn.0.weight=[13824,5120]` 与 `ffn.2.weight=[5120,13824]` 被 `tools/e0_original_bf16_videos.py:raw_header_check` 逐 block 严格检查；两 expert 合计 80 block、160 静态权重矩阵，4 步分别 high、高、low、low，共 160 FFN site、320 线性调用/视频。实际 loader 的 BFloat16 及无量化/LoRA guard、fixed header/full SHA 父 proof、同一 worker 适用性仍须每个新 scope 冻结并在启动时重新核实；stat 不能排除非合作写者。

源 `LightX2V` 中 `infer_ffn`：modulate→原 `ffn_0.apply`→tanh GELU→原 `ffn_2.apply`；后续 `post_process` 原 gate 和 residual。`tools/e0_sharq_wan_all160_runtime.py` 在两个 expert 的两个原 CUDA block buffer 上只覆盖这八个实际 MM 对象；既不替换 infer_ffn、GELU、gate/residual、attention、scheduler，也不改变 token/frame/step cache。BF16 模式走原 `MMWeight.apply`。FP4 的 W1 输出经原 GELU 后直接成为该臂 Q2 输入；b1/b2 是当前原始 block buffer 的 BF16 bias。调用进入 `OriginalGuard` 的 active tap，保留全160 coverage；任何缺失、重复、错误 expert 或模式外调用硬失败。

## 两路量化和付费权重

选 `tools/e0_sharq_wan_ffn_ops.py` 已验证作者 API：对当前激活 Q1/Q2 动态计算 global scale，再 Dense32 一次量化+一次 Dense GEMM，或 SharQ 一次 sparse/residual 准备+一个 sparse GEMM+一个全 K、全 N dense residual GEMM；两路各加原 bias。分解是 `X = dequant(Q_sparse(mask(X))) + R`，输出由 `Q_sparse(X) W_shared` 和 `Q_dense(R) W_shared` 两乘积相加，残差 GEMM 仍为完整 K，不得把其中一支速度称作整 FFN 速度。相同的原 BF16 master、一次作者 `quantize_weight_shared_nvfp4` 产生的同一个 packed payload 和两种 scale 同时用于 Dense32/SharQ。Dense16 来源是不同 checkpoint，本范围不作为 same-W 对照。

选首个 Dense32 访问每个 expert/block/linear 时，从**已经被原双缓冲 loader 搬到 GPU 的原 BF16 W**取得 K×N view，实付 transpose/contiguous、作者静态 weight quantization，同步完成后将 q、两份 scale 和 global scale 留 GPU；两个 step、随后 SharQ 复用同 key。全 scope 若同时完整进入高低 expert 是 160 次真实物理静态打包；packed payload 本身为 5,662,310,400 bytes，另付 scale、CUDA allocator/对齐、动态 A/residual、编译/temporary 和 buffer 成本；缓存软上限 16 GiB、总 GPU/host/wall 预算须根据真实 worker 及独立审议确定，绝不假设缓存可容纳。由于保持原始 loader，160 次静态打包并不取消后续任何原 BF16 权重 H2D。原 162 block-buffer 预取每视频，如 source/target metadata 确认为 CPU→GPU，则 W1 包含 bias 请求 162×141,585,408 bytes，W2 含 bias 为 162×141,568,000 bytes，合计 45,870,852,096 bytes；这些是实际源/目标张量元数据定义的请求字节，不是总线流量。`OriginalFFNPrefetchAudit` 必须在实际每次调用同路径检查、计数；一旦发现非CPU源、不同形状或预取次数，真实失败而非报这些预测值。

双缓冲依赖的源码依据：`wan/infer/offload/transformer_infer.py:infer_with_blocks_offload` 先调用 `init_first_buffer`，随后每 block 把下一块预取到 `cuda_buffers[1]`，在 `compute_stream` 对 `cuda_buffers[0]` 调用 `infer_block`，最后 `swap_blocks`；`common/offload/manager.py` 的初始化在返回前同步 init stream，swap 在交换前同时同步 load 与 compute streams。因此当前 compute buffer 在调用 W1/W2 时已经就绪，且当前 block 的计算返回并完成之前不会被下一轮预取写入；本 runtime 的 W transpose 和作者 weight quant 在当前 compute stream 发起，首次打包后额外同步。此论证依赖绑定的具体源码和原无 lazy-load block-offload 分支，目标 scope 须在运行时重新核对，不将静态推理写成实测 stream 证据。

同一进程先两次 BF16，再 Dense32，再 SharQ：首次 Dense32 的生成墙钟包含静态打包，SharQ复用已打包权重。不能把该顺序墙钟当公平每臂独立冷启动；应分别保存每视频原始 wall、全部 160 W transpose/quant wall、缓存成本、阶段/FFN CUDA 时间、实际原预取字节、GPU/host峰值、观察开销，并另给明示分摊规则的配对条件对照。若要比较完全独立的各臂冷启动，另立 scope 并分别实际支付两个打包，不能回填推算。

## 最小后续注册范围与接口

建议按 1040 和 979 各一独立 part、每项一 GPU/4 CPU，在同一 worker 各跑同 source/seed 两次 BF16 neutral + 一次 Dense32 + 一次 SharQ，共 8 条真实完整 81帧视频。每 part 两 BF16 的完整十个 latent/flow/noise 状态及视频 byte neutral；其他两臂真实同 source/seed 原始和初始 noise byte 配对。每条 160 site/320 linear，两个候选共 640 dense32 GEMM、640 sparse GEMM、640 full-K residual GEMM，分两次物理 scope 各 160 静态 W 打包；baseline 不产生静态 W 或激活 FP4。模型/视频和采集的每一次都计账。任何一个 neutral、source、output、模型或 site 证据未闭合则对应 scope 失败；完整视频只是发展观测，后续独立审阅质量指标并与时间成本分开，不提前声称质量验收。

当前已写可导入真实 runtime、原加载观察器和可嵌入的四视频 phase driver，以及无模型/GPU合成测试。driver 接原 `NativeTap`，在 tap 安装后装 `OriginalFFNPrefetchAudit` 并逆序恢复；用仅供旧 `LightTap.transfer_geometry` 判断的 dense proxy 明确观察原 W2 未跳过。driver 调用真实 `pipe.generate` 的前提是由另一个持有 ACTIVE/隔离和持久化证据的 owning producer 传入授权、状态保存及视频审计回调，它自身没有授权功能。当前尚未实现可授权的完整 owning producer、CPU preflight、private durable job、真实 source/weight/import closure、scope 全量预算与 `verify-closed`；不得调用模型。旧 `native_phase` 固定 `numeric_path=BF16_ORIGINAL`、QDQ=0 和旧模式命名，因此不能直接复用其输出身份冒充 FP4；需要新 producer 正确记录而非改其冻结文件。先作者自审、真实不同作者独立反例、修订/recheck，再按资源安排 ACTIVE/启动。最终 source/binary、原 checkpoints、local HF assets 和两项 worker 的 UUID/host 信息按既有 private spec 绑定；任何节点名与 UUID 不进入公开材料。

`tools/e0_sharq_wan_all160_video_evidence.py` 提供后续 owner 可复用的纯记录判定，检查 4 phase、neutral、原预取、160 weight cache、真实物理 GEMM 次数与外部实际 EXIT0；它不读取或绑定任何文件，返回明确的 `REQUIRES_ARTIFACT_BINDINGS`，不能独立称闭合。完整 verifier 仍须证明视频/状态文件、producer RUN/EXIT、协议、所有输入及 supervisor 的真实字节身份并受单独验证预算约束。


2026-10-30 owner 修订：首条 BF16 不安装该 FFN hook，第二条 BF16 从当前原 buffer 构建全部160份静态 packed W，原 BF16 forward 继续执行。仅 neutral 10 状态及视频字节通过后，Dense32 和 SharQ 才依次消费已有缓存；缺缓存硬失败。pack 当前 compute stream 提交并同步，原 double-buffer prefetch 仍付费。各臂固定顺序及不同热状态使生成 wall 属诊断，不支持公平系统速度结论；setup wall 单列，不能简单从 wall 相减。此处是待审方案，不是实际实验结果。
