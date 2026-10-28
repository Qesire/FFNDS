# 原始 BF16 保存行：范数偏差与跨来源固定校正

actual CPU exit0：读取8份实际DD/SD capture，四对same-state行，8次sigma、4次标量拟合和8次评估；零模型／GPU。每source×step为256个嵌套行，两个来源均已暴露。

方差差严格分为 2〈center(uD),center(Δu)〉/D + ||center(Δu)||²/D。四位置平均variance下降由负交叉项主导，不能只看被删方向的正能量；真实BF16写回和连续gΔY之间的余项及signed cross均保留。

| step | 拟合来源 → 评估来源 | kappa | 原ratio MSE | 校正ratio MSE | 降幅 |
|---|---|---:|---:|---:|---:|
| 2 | 0 → 12 | 1.039908415 | 0.0012376102 | 8.8186898e-05 | 92.87% |
| 2 | 12 → 0 | 1.034234412 | 0.0016735053 | 0.00010325535 | 93.83% |
| 3 | 0 → 12 | 1.020973594 | 0.00039812499 | 6.2871772e-05 | 84.21% |
| 3 | 12 → 0 | 1.018386578 | 0.00051367668 | 7.6343093e-05 | 85.14% |

定义tau=sigma_D/sigma_S。因此真实LN输出应乘1/kappa，原modulation shift和head bias保留；直接缩放整个带常数的head并不等价。source0拟合出的step2/3系数为1.039908414775213、1.020973593727097；反向fold仅用于开发诊断，不选择更好的来源。

跨两个已暴露来源的ratio proxy改善约84–94%，支持先检验固定step偏差这一强竞争机制。row未按head能量加权，此结果不是full-M native head SSE、视频质量或未暴露泛化。下一项是source12上的有限原生校正与own-state检验，不先训练新的状态预测器。

实际 wall 3.448s，outer 3.591s；4项合成控制经协调者独立运行exit0。

协议 `c4c0b464ef9cd67b6227b9b0a9a28bd59705678820e411d92e6d3179e75ee49f`；RUN `6e265ed29bb6a018e8fe2312259d29a9f12f6e26d38d7c6040870422ffe43a40`；actual exit `e058c632c0347cbee261e2a067e8667b781edb4ab2138a94ddca383ab4c31391`。

[完整机器结果](ORIGINAL_BF16_SAVED_NORM_ROWS_RESULTS_20261028.json)
