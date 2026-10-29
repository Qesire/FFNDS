# P64 的局部省算与 offload 关键路径：可区分的 C 线机制

**提出者**：`/root/sm120_full_quality_build_terra`；由 root 提出诊断问题，本 Agent 独立重建源码和推导。**状态：OPEN，等待不同 Agent 独立评议。** 以下是已闭合运行的 NATIVE/SRC 与条件化 ALG，不是新实验或公平系统加速结论。

## 已观察的结构

两开发源、单 seed 的 [V2 报告](../../../20261030/CHIPMUNK_GLOBAL_P64_V2_ALL160_VIDEO_RESULTS_20261030.json)：原 Dense 的 FFN boundary CUDA events 分别 6.28894/6.60719 s，P64 5.67968/5.92437 s，来源等权均值约 **6.448→5.802 s**。原 Dense generation wall 分别 46.64634/47.73397 s，候选 44.85063/46.19073 s，均值约 **47.190→45.521 s**。这是每源每 phase 一次、固定原 Dense→hook Dense→cache Dense→candidate 次序，含观察器且候选控制操作不同，差值没有随机重复与顺序反转，不能视作无偏因果收益。视觉与质量另见相应报告，质量损失约束必须和成本联合判断。

每 source 每 phase 保留 **162 次**原预取观察，逻辑 requested H2D **113,851,754,496 B**，其中原 W2 请求 **22,934,016,000 B**，四 phase 相同；这不是物理总线字节或传输时长。候选每源需80次proxy W1、80次selected W1/W2和两次同mask观察器，完整 cache 创建、clone、读写和同步均真实付费。每 expert 40 层 h+preGateY BF16 payload 47,993,487,360 B = 44.697 GiB；V2 两源累计 allocator peak reserved 69,327,650,816 B = 64.57 GiB，监控物理峰约 66,858–66,859 MiB；reserved 含 live 和未用保留，不能据它倒推出各项峰时内存或设备空闲容量。

## 源码边界和两个竞争机制

[SRC] [NativeTap](../../../../tools/e0_original_bf16_videos.py) 在每次原 `infer_ffn` 前后于当前 compute stream 记录 CUDA event；[V2 tap](../../../../tools/chipmunk_wan_global_p64_tap_v2.py) 的内部动作经 `charged` 做设备同步，P64 proxy、gather、clone、选核与同mask控制发生在被包围的原 FFN 路径。专家最后block的 empty_cache 清理在 `_old` 返回后、外层 wrapper 中执行，**位于该 FFN 事件结束之后**，不能把全部清理成本算入 5.802 s；host ledger/同步、其他流和视频编码的墙钟也不能由事件值直接分解。 [driver](../../../../tools/chipmunk_wan_global_p64_driver_v2.py) 的 wall 从 `pipe.generate` 前开始，经过 GPU同步、tap.finish、视频检查后结束；它包含该范围观察器，不含之前 pipeline 构建。须按源保留 event/wall 原义，不能把两者相减当做精确“其他开销”。

[SRC] 原 [offload loop](../../../../.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py) 在 load stream 预取下一块、compute stream 运行当前块；[manager](../../../../.runtime/code/LightX2V/lightx2v/common/offload/manager.py) 在换 buffer 前同步两流。阶段可能重叠，设备级同步及每块 join 又会暴露尚未完成的部分。原预取保留及 logical 字节不变本身不证明链路饱和，也不证明事件与 H2D 无重叠。

**A（行动开销抵消）**：消去部分 dense GEMM 后，proxy、选取、W gather、缓存拷贝、同步及两次控制占用显著资源；预测在相同状态、同范围的可加总计量中，dense GEMM 减少量大于真实关键路径净改善，尤其 cache/packing 或同步负担可见。当前只有全FFN包围事件，没有无重叠、逐项关键路径剖面；不能量化每项占比，也不能由 logical bytes 当成带宽实测。

**B（管线关键路径受限）**：即便 FFN compute 减少，可能被权重加载、attention 或其他路径覆盖/主导；固定输入下预测减少 FFN 时间但不改变 load 或其他关键路径时，端到端变化明显小于局部节省。A 与 B 可同时成立；一次单序事件/墙钟均值本身不能判别哪个主导。

[ALG] 若每块计算段为 C、可并发 load 为 L，join 后理想块时延 `max(C,L)`；减小 FFN 段 ΔC 的显性收益是 `max(C,L)−max(C−ΔC,L)`，介于 0 和 ΔC 之间。其他依赖、带宽争用、host延迟与同步会改变这个简化式；不同 phase 的事件边界、observer 和 pipeline 状态也使现有 Δevent 不能当 ΔC。一个反例是 C=10,L=9，减 C 两单位却仅节省一单位；另一个是 C=10,L=1，同样减两单位可显露全部两单位。

**条件化 Amdahl 算术**：若且仅若 FFN event 6.448 s 是 47.190 s 同一端到端关键路径上可独立、可完全消除的非重叠墙钟，且其余时间完全不变，则份额 13.66%，极限 `47.190/(47.190−6.448)≈1.158`。实际 event 非此已证墙钟份额、还有跨流并发及 phase/observer 差异，故 **1.158 不是无条件系统上界**；同样本约 1.669 s 墙钟差甚至大于 0.646 s 事件差，可由未控顺序、噪声及观察器/重叠差异产生，不能反证物理定律或证明机制。改变运行时、传输或并行可改变非FFN部分，进一步打破固定其余项假设。

## 可能改变 Pareto 的动作与最低区分证据

**共同工作集协同动作（OPEN）**：把现有每 expert 的巨型 h+Y cache 作为 resident 预算的机会成本考虑，而非只调 P64 selector。探索在合法信息和质量约束下减少 cache 生命周期/体积，换取可复用权重驻留或分块预取改变；同时计入代价：重算、缓存读写、H2D、额外同步、实际新路径产生的输出误差及真实自由视频。当前没有证明 resident 方案在同 GPU 能装下，也没有实现或质量结果；该动作不是免费删除必要状态。A100 master 实际为 offload，不能将文件总大小视为当前 GPU 权重峰；不要用 80 GiB 标称减 64.57 GiB 推断严格剩余预算。

**最有辨识力的下一证据**：在另行独立评审与注册的同机同模型 paired 有限 native scope 中，固定同输入、原 Dense及 P64、同 phase 冷暖/顺序反转，记录逐 block 两流 GPU timeline：compute FFN、下一块 H2D、每次 join 实际等待与同步、attention及其他必要阶段，并统计 cache live/allocated/reserved、物理传输和 observer 成本；同时用同 runtime 同显存预算的 Dense 对照，另设可行的缩短 cache live-set/权重驻留动作，并实付其额外计算与通信。该单一有界对照若大部分时间为 compute 且 cache/selection占净收益，支持 A；若 load/join 在 FFN缩短后暴露且受控 resident 动作降低端到端而不新增过大代价，支持 B 与工作集机制；若内存无法并驻或质量/额外成本抵消，则该 resident 动作失败。profile 的扰动和工作集峰须实测；不能通过该诊断宣称完整视频质量或可部署速度。此处为未来设计，不授权运行。
