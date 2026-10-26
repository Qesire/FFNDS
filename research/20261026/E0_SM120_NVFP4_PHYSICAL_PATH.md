# 原生 packed FP4 执行路径：Blackwell 数值基元已建立

**新的物理结构是：固定版本 CUTLASS 的 SM120 block-scaled E2M1 矩乘已在空闲 Blackwell 上实际执行，两种小型输入均与官方 CPU 参考 BF16 输出逐元素相等。** 它提供另一种可执行数值表示，补充旧 HiF4 解码后 BF16 矩乘的比较器。竞争解释是实际收益会被在线量化、scale 布局、短矩阵或格式转换误差抵消；下一项有价值的判别应使用真实保存输入，分别计在线量化、已打包矩乘和 BF16 比较成本。

## DISCOVERY：格式与实际物理运算相连

上游 [CUTLASS 79a 固定源码](https://github.com/NVIDIA/cutlass/blob/d55f6beeebb6df501a250dc82827db97660f06e0/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu) 为 SM120 配置 packed E2M1 A/B、分块 scale、FP32 累加与 BF16 输出。当前研究只建立这个执行基元；它保持 dense MAC 数量，尚不构成本征 FFN 少算结果。

[NVIDIA 格式说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) 描述了每16值共享的 E4M3 scale 和全张量 FP32 scale。[FlashInfer 接口](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html) 也将 packed operand 与 scale 布局显式作为输入。因此输入量化和布局转换是下一比较的实际工作，不能从这个已经打包的例子中推断其成本或精度。若从旧 HiF4 解码权重继续量化，必须将它标为额外数值变化。

## EVIDENCE：两个确实执行的有限基元

| M×N×K | alpha/beta | GPU GEMM 次数 | 官方 host reference |
| --- | --- | ---: | --- |
| 128×128×128 | 1/0 | 1 | Passed，逐元素相等且两侧范数非零 |
| 128×256×512 | 0.5/0.5 | 4 | Passed，逐元素相等且两侧范数非零 |

原样使用上游静态 seed=0，以及 operand seed 2021–2025。这些随机 packed 数值/scale 不代表 BF16→NVFP4 量化后的实际 Wan 分布。共5次 GPU GEMM、69,206,016 MAC；两次 CPU 参考共18,874,368 MAC。两个子进程完成，编排 wall 2.099秒；其中包含 CPU reference 与轮询，微型例子的计时没有匹配 BF16 比较或在线量化，故不推导速度优势。

资源轮询没有发现其他计算 PID，但短进程使第一个子进程没有捕获到自身 PID，第二个只捕获一次；采样物理显存最大18MiB不代表真实 context/allocator 峰值。精确峰值及完整时间连续隔离仍未测量。源码、binary、RUN 和日志 SHA 及真实边界都在[机器可读结果](E0_SM120_NVFP4_PHYSICAL_PATH.json)。

## EXECUTION：一次固定范围完成，失败构建独立保留

临时持续授权下登记[固定 primitive 协议](../../config/e0_sm120_fp4_primitive_20261026.json)，一张空闲卡、两次子进程、五次 GPU GEMM；0 Wan FFN、模型、视频、训练或重试。六项 CPU 软件测试覆盖 wrong-architecture exit0、明确数值 disposition、外来 PID、超时及启动失败的终止账本；[独立源码审阅](../method_discussion/discoveries/E0/independent-sm120-packed-fp4-primitive-source-review-20261026.md) 核对源版本、编译回执、binary 和预算。

首次编译因关闭 CUTLASS tools 后缺少 utility include 失败，零 GPU 调用；其原始失败 RUN/log 不变。第二个独立编译协议只启用官方 tools 以取得 include 接口，库/测试/profiler 仍关闭，228.248秒完成，复用既有 CUDA12.8.93，未升级环境。编译和 primitive 均不使用模型权重。

图锚 F04/F08/F09/P14 的实际 Wan 分支仍是历史原生路径；本结果是独立 `NATIVE-ACT` 基元，通向这些节点的格式替换边为 conditional。其价值是让后续 FFN 功能表示能与真实低位硬件比较；完整四步视频功能、实际 FFN 算量降低和端到端速度仍需各自直接测量。
