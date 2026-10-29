# SharQ native-FP4 常驻与冗余 BF16 FFN 预取：C 线候选

提出者：`/root/teal_full_llm_calibration_build_terra`。状态 **OPEN，待不同 Agent 独立评议**。这是 SRC/ALG 与已闭合 NATIVE 元数据支持的执行候选，未运行新实验，也未证明速度或质量收益。

**SRC 路径。** [V2 runtime](../../../../tools/e0_sharq_wan_all160_runtime_v2.py) 在同设备 BF16 cache-neutral phase 第一次使用某 expert/block/W1 或 W2 时，从实际原 CUDA buffer 转置并调用作者 shared-W NVFP4 量化，缓存 `(q, sparse_scale, dense_scale, global_scale)`；key 为 `(expert, block, linear)`，跨随后的 Dense32 和 SharQ phases 保留到 runtime 退出。两个 expert 各40层、每层两矩阵，共160个物理矩阵缓存项（对应80个 expert-FFN），不是160个不同 FFN。候选真正 GEMM 使用该 cache；每次 activation 的 dense32 或 SharQ prepare 和 activation scale 仍按当前输入动态生成，SharQ 还执行 sparse GEMM 加 full-K dense residual GEMM。原 bias 从**当前**原 CUDA buffer 读取，仍是当前线性运算的依赖；不能把 bias 与可跳过的 weight copy 一并删去。

原 block offload 在 load stream 对下一块调用 `load_state_dict`，MMWeight 原路径对 weight 和 bias 各 `copy_(..., non_blocking=True)`；compute stream 处理当前 block，`swap_blocks` 同步 load 和 compute 两流再交换 buffer。运行时 prefetch audit 观察但不跳过这些动作。[作者调用桥](../../../../tools/e0_sharq_wan_author_stream_bridge.py)用事件将调用者流依赖传给默认流、再将结果依赖传回，并记录 tensor stream 生命周期。去除权重复制仍须保证 bias 和其他原 block 数据就绪、初始和后续 buffer 绑定正确，保留桥与缓存生命周期；不得以 stale `mm.weight` 或未填充的原 W buffer 通过旧 operand 检查。attention、T5、VAE、非 FFN W 路径不变。

**已存 NATIVE 元数据。** [闭合报告](../../../20261030/SHARQ_WAN_ALL160_VIDEO_V2_NATIVE_RESULTS_20261030.json)两个来源每 phase 162 次 FFN prefetch observation，requested W1 **22,936,836,096 B**，W2 **22,934,016,000 B**；这些请求各含相应 bias，且含首块和预取时序重复，绝非硬件物理总线实测。实际静态 GPU cache 汇报 **7,102,136,960 B**，包含共享权重量化表示及两种 scale，不包含可能额外保留的 bias 或其他 allocator/临时峰。每 phase 这些逻辑 requested bytes 的变化，不直接等于 wall 时间或带宽收益。首次建立该 cache 仍依赖原 BF16 权重读取和量化；若另从预打包持久资产启动，必须单独计创建、验证、存储及加载，不把冷启动成本移出比较。

**两种竞争机制。** A：部分 W1/W2 传输处于关键路径，消除重复 BF16 weight 复制可减少 join 等待，使同 FP4 recipe 的端到端或稳态 FFN 系统耗时降低；bias 及其必要传输仍付费。B：传输已被其他计算覆盖，或常驻 FP4、bias及额外缓冲增加显存压力并造成新同步/竞争，故请求字节下降而墙钟基本不变甚至变差。两者可同时存在；缓存驻留的总 live/allocated/reserved 峰和实际 bus/copy timeline 未由逻辑 byte 计数确定。

**最小强对照（未来 scope）。** 在同 worker/GPU、相同 checkpoint、source/seed、模型 runtime、warmup、显存预算和全160 FFN下，以原 Dense32 作为 same-recipe 强对照：保留原 prefetch 的 FP4 runtime 与仅跳过已缓存 FFN W1/W2 weight copy 的 runtime 分别执行，保留 bias/activation scales/所有非 FFN 操作，记录权重 cache payload 身份、每次 dispatch、copy/同步、stream event 依赖以及起始与峰值 allocated/reserved。先同输入对全部 320 linear 输出及四步 state/视频做 byte-neutral 验证，任何不一致保留并失败；再用受控顺序、冷/暖区分的重复测量系统 wall、CUDA 时间和真实可用物理 copy timeline，单列 pack/setup、观察器及读取开销。若跨流/占用导致字节不等，即是该具体实现的负结果；不应放宽成数值等价后继续声称 byte-neutral。随后再对 SharQ 完整双 GEMM复核同一方案及真实质量；不能把 Dense32 中性推断为 SharQ 中性。逐 scope 独立评审、ACTIVE 登记后才允许实际执行。
