# Original BF16 low39：输出投影与跨状态判别实测

12 次完整 40-block flow 已实际 exit0。保留全部 13824 个原始 W1/GELU 特征，仅把低噪 expert 的 block39 W2 改为一个固定 rank512 输出表示。两个已暴露开发来源、seed42、末两步；四个 Dense flow/post 全字节 neutral 均通过。

同状态下，64 维线性读出贡献已经很小，而 LayerNorm 的方差变化仍主导 head 误差。这揭示了紧凑表示需要额外维持的范数自由度；当前结果尚未说明该自由度能以足够低成本预测。

| 来源 | step | gated Y relative SSE | head relative SSE | 线性项 | 径向项 | native remainder |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0.02930796 | 0.00167034 | 3.95e-07 | 0.00166289 | 7.1e-06 |
| 0 | 3 | 0.03399528 | 0.00053165 | 4.18e-07 | 0.00052450 | 6.99e-06 |
| 12 | 2 | 0.02768268 | 0.00122357 | 3.38e-07 | 0.00121619 | 6.84e-06 |
| 12 | 3 | 0.03065170 | 0.00039650 | 3.87e-07 | 0.00038973 | 6.86e-06 |

这些项共享各自 Dense head 能量分母；完整 JSON 保留 direct×radial 及连续项×native remainder 的有符号交叉项，不能把各正能量直接当作互斥占比。native remainder 包含整个离散程序差异，不是独立测出的舍入下限。

| 来源 | step3 总 flow relative SSE | 同状态局部 | Dense 状态传播 | 近似的状态变化 | 三交叉项之和 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00090733 | 0.00053165 | 0.00053010 | 0.00002699 | -0.00018141 |
| 12 | 0.00066087 | 0.00039650 | 0.00034043 | 0.00002479 | -0.00010085 |

四角定义：a=S(xD)−D(xD)，p=D(xS)−D(xD)，c=[S(xS)−D(xS)]−a；总误差=a+p+c。两来源都是局部项与 Dense 状态传播项量级相近、交叉项合计负，21 个时间槽逐项闭合；不是“近似函数随状态变化”一项独占总误差。FP32 scheduler 算术与 BF16 cast 分开记录，末步更新恒等式残差平方均为0。

真实范围：480 FFN、954 次原 FFN 线性、6 次系数＋6 次输出 GEMM、492 次原 checkpoint block-buffer load、两次 QR＋一次 eigh；20 次高层 FP64 统计矩阵乘法，不声称枚举库内 BLAS。实际 wall 267.733s，外层 283.057s，峰值 reserved 21.455GiB。新增结果 351882766B，906 份执行源码与12份capture已独立校验 SHA；monitor 0 违规。

下一判别先利用已保存、同坐标的 u/r/Y 行，检查 sigma_D/sigma_S 在 token、step 和来源间的变化，比较固定时刻校正与依赖状态的范数补全是否可预测。新的原生干预另行冻结。

本轮是原始 BF16、完整 W1、单末层控制。旧 HiF4/6016 学生的训练瓶颈、全80函数、完整视频质量和系统速度保持各自证据身份；全部原始 W2 预取成本仍付费。

协议 SHA `fcd50986dc650e80abc0a1a665fca973b92af940fa6b34bf6615e1c78f322d4f`；RUN `768a5218da2ac17914ffd7e6a3e1623adf848be679a71a941634596c2d1c0088`；actual exit `49e536c4725c4f525ffaece08d5daf23ac0d1781def107c1df8c9325613ae35b`。
[完整机器结果](ORIGINAL_BF16_LOW39_FOUR_CORNER_RESULTS_20261028.json) · [精确源码独立审阅](../method_discussion/discoveries/E0/original-bf16-low39-four-corner-independent-review-20261028.md)
