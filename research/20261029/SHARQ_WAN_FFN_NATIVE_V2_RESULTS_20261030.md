# SharQ → Wan FFN：原生算子 V2 实测

2026-10-30 · **NATIVE / 单来源算子，非视频或系统验收。** 实际 EXIT0；16 个组合、240 次 FFN，source911 seed42、block20 的 high step1 和 low step2。M32 是各状态真实采集的 32 行；M31668 将这 32 行循环重复，仅用于形状成本诊断。

Dense32 与 SharQ 共用同一原 BF16 master 的作者 group32 packed W；Dense16 来自另一 NVFP4 checkpoint。每个线性层 SharQ 执行 sparse 主路和 full-K dense 残差，各臂 W2 输入来自自身 W1 和 GELU。

| Expert | arm | overall-only CUDA ms 均值 [5次最小, 最大] | gate 后相对 L2 |
|---|---|---:|---:|
| high | original_bf16 | 31.36630 [31.33658, 31.38691] | 0.000000 |
| high | dense_group16 | 11.12090 [11.10029, 11.13744] | 0.083815 |
| high | dense_shared32 | 20.77805 [20.74893, 20.82278] | 0.084721 |
| high | sharq | 28.95711 [28.94874, 28.96099] | 0.064762 |
| low | original_bf16 | 31.39783 [31.39136, 31.40544] | 0.000000 |
| low | dense_group16 | 11.11473 [11.10243, 11.12346] | 0.083862 |
| low | dense_shared32 | 20.76679 [20.75450, 20.78954] | 0.085058 |
| low | sharq | 28.95592 [28.94448, 28.97146] | 0.065834 |

上述延迟为完整 FFN 的 overall-event-only 组，每个数字为 5 次真实测量，范围仅是最小/最大值，不是置信区间；仍包含调用和记账观察成本。逐操作 event 组是另一套测量，不与此组相加。

high: SharQ 相对同 W Dense32 的 gate 后相对 L2 下降 23.56%，延迟为其 1.394 倍；相对 BF16 的完整 FFN 速度为 1.083×。
low: SharQ 相对同 W Dense32 的 gate 后相对 L2 下降 22.60%，延迟为其 1.394 倍；相对 BF16 的完整 FFN 速度为 1.084×。

M32 同编码参考的 12 个 W1/W2 检查均低于预定 0.03；作者 debug 与 selector 共用代码，不能独立证明 selector 算法。实际 BF16 相对捕获边界的相对 L2：

| Expert | pre-GELU | h | Y | gate×Y |
|---|---:|---:|---:|---:|
| high | 0.0000000 | 0.0000000 | 0.0032976 | 0.0030984 |
| low | 0.0000000 | 0.0000000 | 0.0032985 | 0.0028348 |

所有 16 组合的五次原始整体 CUDA 样本、M32 同编码参考与比较指标保存在同名 JSON。先前 V1 实际 EXIT1，在 dense32 参考 scale 长度检查失败前消费 45 次 FFN；其失败和成本独立保留，不纳入 V2 成功分母。

V2 RUN SHA256 `5921024ab75bd6cc3d12d8ec5811aea40c90a96c2a3368fe6a47300603b76adc`；EXIT SHA256 `3dbbd686a089af676ea0418b14b8a3ab4693d1553f85a566e7ceafe569a8988b`。此结果不证明完整视频质量、全160位点效果或系统加速。
