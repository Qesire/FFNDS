# C5：直接生产 CUTLASS packed 表示

在两个合成 Wan 形状上，融合选择、值打包与 metadata swizzle 大幅降低了通用 converter 路径的开销；同次运行的完整 fused online 中位数略低于 dense。预打包 SpMM 已接近 dense 的时间，说明接下来的关键分歧是稀疏乘法、布局与输出成本能否进一步降低，单独继续调整 selector 的空间有限。

| M × K × N | dense | select + pack | 预打包 SpMM | 原 converter online | fused online |
|---|---:|---:|---:|---:|---:|
| 31668 × 5120 × 13824 | 16.247 | 0.643 | 15.117 | 25.872 | 15.684 |
| 31668 × 13824 × 5120 | 15.982 | 1.537 | 14.479 | 42.819 | 15.867 |

单位为 CUDA event 毫秒，均为同一次运行每项2 warmup后5次测量的中位数。完整 online 包含选择、打包、wrapper、SpMM、裁切及输出连续化；预打包不含选择与打包。fused 相对 dense 的中位数余量约3.5%和0.7%；各路径按顺序测量而非交错，5次样本不足以建立稳健系统加速。两个单独 select+pack 样本序列分别含1.179和5.010毫秒离群值，完整序列见 [结果 JSON](E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.json)。首次 GPU 小形状调度含编译分别7.429和1.818秒；完整形状首次调用分别1.710和1.297秒，均不计入稳态表。

实际 GPU 的两个边界形状和两个完整形状，packed 与 metadata 均逐字节等于该次原 Torch converter。完整形状的 materialized mask 与 stable top2 参考也逐字节相等；CUTLASS 相对同 mask dense 的 relative L2 为5.51e-5和1.201e-4，输出形状正确且连续。该验证包含补齐到31680行；数值比较针对 masked 合成输入，不证明稀疏化保持模型质量。

**独立核对**：RUN SHA256 `a2d8f79fa8bee0a9023b5705c5d145c1c18dabd86169efc04b69866273dac125`，冻结协议 `3237d21b5fd5778edb99cfff9991849cc06d1a4245a7cd79b8e0b6d24629b7ac`，launcher退出收据 `5667c51e6c907a26a142c61ff167fc1362f546bd060701969720dc5b782d0f05`，退出0。31项协议绑定中23份可保存源码快照逐一匹配；二进制及非源码资产不计作快照。预扣数、完成数和协议限额完全一致：4次 GPU packed/meta 字节检查、2次 mask 字节检查、32次 fused pack、32次原 converter、44次 sparse GEMM、16次 dense GEMM；搜索、模型、拟合、视频与官方评分均0。实际映射 libtorch_cuda 和 libtriton 与绑定路径一致；A100采样无违规、采样峰值12802 MiB，采样不证明间隙内持续隔离。计算含setup 90.52秒，外层94.99秒。

这是固定有限BF16合成矩阵上的格式和执行观察；没有运行真实 Wan FFN、模型或视频，尚无端到端收益或质量结论。
