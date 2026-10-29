# RT-Lynx 启发的 Wan BF16 2:4 全FFN · 作者自审

科学问题：在相同 source911、block20、high step1 / low step2 的真实 V1 与 gate 及原 BF16 W1/W2/bias 上，2:4 的输出损失由能量衰减主导，还是由被删 W2 方向主导？SA-Native 与行范数恢复是可判别的成对动作。每连续4个 post-GELU BF16 值按幅值稳定取2，tie取较低索引；行补偿为 FP32 `sqrt(sum(h²)/(sum(top2(h)²)+1e-8))`，先乘选中值再 BF16 舍入并 pack。论文作者代码、归约轴、tie/dtype 未证实；这是论文启发的 Wan GELU 无训练移植，不是作者忠实复现。非有限输入拒绝；全零行缩放为0。无 LoRA、无缓存或非FFN跳算。

原 BF16、Top2、Top2+行范数三臂皆实际调用 W1+b1、GELU tanh、W2+b2 和原 gate；后两臂使用 C5 的原稳定 Top2+CUTLASS 压缩 metadata 与 A100 实际 sparse GEMM。范数臂先 materialize mask，再计算范数与缩放，随后再选择/pack，所有动作均计时，不声称融合或免费。2专家×2形状×3臂×(1 correctness+1 warmup+3 measured)=60次完整FFN，其中W1 60、Dense W2 20、稀疏 W2 40、mask预处理20；M32为实际32采样行，M31668以原32行循环重复、仅用于同形成本诊断，不代表真实full-M分布或视频质量。每次 CUDA event 含 W1、GELU、全在线稀疏准备/内核、bias和gate；correctness冷启动含编译、warmup及各3次采样分开列出。固定顺序和observer、同卡其他负载和短采样不能证明公平系统速度。

每种稀疏case另有同mask原dense W2数值参照和与Torch CUTLASS converter packed/meta逐字节比较，排查真实压缩/metadata和稀疏W2错误；同mask相对L2上限0.02。原BF16 M32相对已采 a/h/Y/gate的中性上限0.03。差距不足时保留方向竞争；完整自由视频、全160、正式质量及训练均未执行。真实 bundle由先前 CPU scope EXIT0 证明，730509120 B 单次全文 hash 预扣；后续 mmap 实际物理读 UNKNOWN；H2D为逻辑字节、同步后完成。已闭合C5真实BF16合成布局/算子证据仅支撑可实现性，不给本次真实质量结论。

预扣总上限见冻结协议；每个FFN整体及所有额外参照先持久预扣，Python调用返回与实际GPU完成通过同步后区分，硬中断在途UNKNOWN。C5绑定源码和库本次限额2GiB全文SHA后运行，并核实际已加载 CUTLASS/Triton 动态库。仅单 A100、CPU4，采样资源监控、硬wall与输出限额、独立review/ACTIVE/worker UUID flock、durable实际EXIT和失败不重试。以下还有需独立质疑的工程风险：norm参考与实际动作共享mask选择实现、BF16缩放引发重选边界、backend allocator/非有限、原`to_dense`或converter支持及内存峰值；必须真实复审和必要修订后才注册。静态合成单测不证明任何GPU结果。
