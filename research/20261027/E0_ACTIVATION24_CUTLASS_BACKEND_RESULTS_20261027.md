# C4：相同合成输入上的 CUTLASS 与 cuSPARSELt 激活 2:4

两种通用在线转换路径在这两个 Wan 形状上都比同一次运行的 dense 慢，尽管预打包后的稀疏乘法较快。结果提示转换成本是当前实现的关键约束，并不构成所有压缩表示、专用转换器或融合 kernel 的速度下界。

| M × K × N（M 补齐至31680） | dense | CUTLASS 预打包 | cuSPARSELt 预打包 | CUTLASS 压缩 | cuSPARSELt 压缩 | CUTLASS online | cuSPARSELt online |
|---|---:|---:|---:|---:|---:|---:|---:|
| 31668 × 5120 × 13824 | 17.122 | 15.123 | 14.740 | 10.358 | 8.429 | 26.080 | 22.482 |
| 31668 × 13824 × 5120 | 16.686 | 14.459 | 14.800 | 27.292 | 22.397 | 43.074 | 36.764 |

单位为 CUDA event 毫秒，均为本次运行 2 次 warmup 后 5 次测量中位数；预打包不含当次选择和压缩，online 包含它们及功能输出连续化。首次 Triton 调用含编译，分别 2.3445 和 0.2689 秒，单独记录。cuSPARSELt 固定 alg1/split1/mode1，未调用搜索；搜索成本未知。

两个 shape 的 Triton 选择结果与当次 stable top2 参考补齐结果 BF16 字节相同。CUTLASS 解码对 masked 输入的 `torch.equal` 数值相等；这一检查不建立 signed zero 的位模式相等。CUTLASS 相对同一 masked dense 的 relative L2 分别为 5.51e-5、1.201e-4，cuSPARSELt 为 6.07e-5、1.224e-4；两后端最终输出形状正确且连续。RUN 未记录具体 stride，不能据此声称额外 stride 细节。实验为随机合成 BF16，未运行模型、视频或质量评估。

**独立核对**：RUN SHA256 `e79dae0174c71dda110043ddf4ba639bffd3db2791ac3a8e5a0de82741bacf8a`；冻结协议 `3903306e68db2dd70584cb0d2941f5eca65fc8b7472a93c0da7512edc3580e07`；launcher 退出收据 `c5725b385ed011512c580da1041d4b618e80ac195d7440432d00d697248afabf`，退出0。19个实际执行源码快照逐一与协议SHA匹配，包含工具 `7972ff6dbaf09cf5e8bb09a3a803a7ae33ed11dd5816d62fdcfde589ffca9f7a`；其余绑定不都复制为源码快照。RUN报告实际映射并绑定 libtorch_cuda、libcuSPARSELt 和 Triton，五次资源采样无外部进程违规，采样峰值 16082 MiB；采样之间不构成连续隔离证明。实际计数与冻结限额一致：4随机、2 stable、30 Triton、各30压缩、2重构、16 dense、各30 sparse，搜索/模型/拟合/视频均0。计算段12.931秒，外层15.641秒。

完整数值及库身份见 [结果 JSON](E0_ACTIVATION24_CUTLASS_BACKEND_RESULTS_20261027.json)；原始 RUN 和协议保留原身份。
