# 固定 step 范数校正：原始 BF16 的原生来源迁移

source0拟合的两个κ，直接用于source12的完整40-block flow；八次flow已实际exit0。四个Dense／bare neutral的flow与post均逐字节匹配历史，通过后才运行校正。没有重新拟合、训练或分解。

| step | 未修正head/flow relative SSE | 固定κ后 | 降幅 |
|---|---:|---:|---:|
| 2 | 0.001223571 | 0.000077417 | 93.67% |
| 3 | 0.000396499 | 0.000069585 | 82.45% |

自身状态末步总flow relative SSE：0.000660873 → 0.000200976，降低 69.59%。与保存行代理不同，这是实际原生full-M读出及付费Dense counterfactual得到的结果。

真实动作是原BF16 LayerNorm返回后，以冻结FP32(1/κ)乘法及BF16 copyback校正归一化分量，然后继续原modulation、shift和headbias；没有缩放完整带常数的head。κ₂=1.039908414775213、κ₃=1.020973593727097来自source0，source12的本次输出不参与选择系数。

三次非单位gain的真实成本单独登记；κ=1的五次调用经过同一非别名检查且不做新算术。raw-u sigma未改，读出的有效分母按raw_sigma/已部署FP32alpha计算，新算子舍入留在native remainder。

修正后的step3四角：局部=0.000069585，Dense状态传播=0.000161674，近似随状态改变=0.000018930；完整JSON保留三项signedcross、21时间槽及FP32/BF16更新余项。状态传播项现在大于局部误差，下一证据应向跨深度表示与原始BF16校准推进，而非继续调末层两个标量。

实际320FFN、635原FFNlinears、5×两因子GEMM、328原checkpointbufferload、3gain、0fit／QR／eigh；909份执行源码及8份capture独立SHA复核，monitor0违规。wall 177.322s /outer 192.429s，峰值reserved 20.850GiB。

这支持当前固定末层投影的可控归一化偏差机制：一份拟合来源向一份已暴露开发来源迁移。全80函数、完整视频质量和真实系统加速仍需其各自实验；原W2预取与新增gain开销均付费。

协议 `44ba9f0cd16da06bcb92ae904322fbc36b6930fa7c13315ea82c429991d8fcaa`；RUN `783e22952f70ab5a8cb0563587f098a674833113b16d31ec65d0acf8ccd72b2c`；actual exit `117740de4db5a57a179e92279c84308cc8c90bb1ae9dbdf170030c75940f92cc`。

[完整机器结果](ORIGINAL_BF16_FIXED_NORM_TRANSFER_RESULTS_20261028.json) · [独立源码审阅](../method_discussion/discoveries/E0/original-bf16-fixed-norm-transfer-independent-review-20261028.md)
