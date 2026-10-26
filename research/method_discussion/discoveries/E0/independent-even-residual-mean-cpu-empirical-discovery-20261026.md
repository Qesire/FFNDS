# 投影后的边际矩与输入相关性是不同的研究对象

**DISCOVERY：Wan dense FFN 的保存 panel 出现了一个可分离的结构：真实 W1 投影矩能够较好描述经过 signed W2/gate 的 GELU 均值，但仅使用输入通道的均值和各自方差时，投影方差与输出均值明显退化。** 一维边际 Gaussian 闭合与高维输入独立性属于两个命题。这个区分为研究必要公共功能提供了一个坐标，也把在线观测成本显式暴露出来。

竞争解释是：post-W1 Gaussian 的优势来自已经支付的投影信息，剩余误差可能由非高斯尾部、有限 panel、求积误差或 W2/gate 方向选择造成；它尚未构成低成本方法。最小后续判别是保存 z 上的经验 erf 均值与精确 Gaussian erf 公式，以及固定求积的误差核查。下面先记录本次已完成证据。

## EVIDENCE：精确对象与结果

协议只覆盖来源0/12、seed42/54、high-noise block10 step1，原中心 contiguous128 与原 uniform32 四个 dense panel。完整数值是 [逐字节 RUN 副本](../../../20261026/E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.json)；便于比较的表格是 [公共结果说明](../../../20261026/E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.md)。

设 `q=q1_post`、`a=z`、`s=Smooth2`，所有均值/方差均取该 panel 的经验总体矩。三条表示为：

1. **post-W1 矩闭合：** 以保存的 `mu_a,var_a` 代入 Gaussian GELU 期望，再除以 s。
2. **pre-W1 对角闭合：** `mu_hat=W1 mu_q+b1`，`var_hat=(W1∘W1)var_q`，再用 Gaussian erf-GELU 公式。
3. **经验连续参照：** `mean(g_tanh(saved_z)/s)`，经过原始 decoded BF16 W2 的 FP64 signed readout。

erf-GELU 的恒等式是：若 `X~N(mu,v)`，令 `k=mu/sqrt(1+v)`，则

`E[X Phi(X)] = mu Phi(k) + v/sqrt(1+v) phi(k)`。

tanh-GELU 的 Gaussian 比较器采用源内固定32点 Gauss-Hermite；这是有误差的数值估计。真实 tanh-GELU 满足 `g(a)=a/2+r_even(a)`。对称输入可有平均导数1/2，同时均值仍随 even 部分的尺度/尾部改变；“平均 slope”不是“输出均值”的充分统计量。

在同一实际 native FFN 均值目标、b2 加一次、当前 gate 加权的比较下，post-W1 tanh Gaussian 相对均值 SSE 为 `0.7715/0.4533/0.1999/0.1793%`；pre-W1 对角 erf Gaussian 为 `14.2942/8.0769/6.0889/5.0391%`。次序固定为 `0/128,0/32,12/128,12/32`。前者对应约4.2–8.8%相对 L2 均值误差，准确度是否可用于某个方法尚无接受阈值。

这一差异并非仅以 erf/tanh 的名字推断：在相同 post-W1 真实矩上，两个 Gaussian readout 的 gated 差异能量为 `4.43e-5–1.40e-4`；Gaussian-tanh 与经验连续 readout 的差异能量则为 `0.520–1.882`。求积与公式的误差没有被单独认证，故只保留这个实际 comparator 差异，未宣称 Gaussian 模型已精确成立。

## EVIDENCE：一个关于信息的结构性缺口

pre-W1 均值预测的相对 L2 误差仅 `0.0196–0.0411%`，与线性均值传播吻合。对角方差预测的总量却只达到保存 z 总方差的 `52.855–56.619%`。

实数线性运算给出

`Var(w_j^T q) = Σ_i w_ji² Var(q_i) + 2 Σ_(i<k) w_ji w_jk Cov(q_i,q_k)`。

这里重要的不是给每个通道再调一个独立 Gaussian 参数，而是联合输入协方差经过每个真实 W1 行的有符号投影。一个严格的信息反例是：`q=(G,G)` 与 `q=(G,-G)` 的各通道均值/方差完全相同，但对 `w=(1,1)`，投影方差分别为4和0。任意只读取这些对角统计的规则都无法在此两例上同时恢复真实方差。这个反例只界定该观测集合的信息能力，不排除读取其他信息的 FFN 表示。

当前保存值来自原生 BF16 W1，预测来自 FP64 decoded-weight 矩传播；该舍入/累加对方差的影响未独立有界。很小的均值误差也不能自动给出方差误差上界。因此实测强烈支持研究相关性/投影方差缺口，而未把全部43–47%的方差差额因果登记为协方差贡献。对角预测的最大方差也明显小于保存最大值，提示后续需要看方向/异质性，而不只是总尺度；本次没有额外调比例或选方向。

## EVIDENCE：signed readout 与原生数值桥

输出均值对象是 `W2 mean(q2_post)+b2`；FP64 解码与实际保存 FFN 输出均值的 gated 相对 SSE 为 `1.41e-8–1.07e-7`，保留了原生 W2/bias 累加/舍入桥。gate 在此只是 FP64 通道权重，未重演实际 BF16 gate 乘法与残差写回。

连续 tanh 与保存 GELU/Smooth2 的 gated 均值差异能量为 `4.68e-6–2.12e-5`，post-Q2 与保存 GELU/Smooth2 的合并差异为 `0.0309–0.0607`。后者在每个 panel 都小于 Gaussian-tanh 与经验连续 readout 的差异。这削弱“这四个 dense panel 的 Gaussian 均值差距主要由原生数值桥引起”的解释；旧候选在其他输入/深度/步上的失败成因仍是不同问题。

uniform32 的八个共同字段与旧 passive archive 在同一 source ID、seed、采样 seed、row IDs 上逐字节一致，故可使用真实旧 pre-Q2：Q2-only 的 gated 差异能量为 `0.0607675/0.0501859`，Smooth2/cast 桥为 `1.59e-5/1.24e-5`。contiguous128 未保存 pre-Q2，保持合并桥语义。不同桥的误差存在交叉项，以上数值不是可相加的原因占比。

b2-only 留下 `92.90–93.86%` gated 相对均值 SSE，表明数据相关的 FFN 均值方向值得作为独立对象解释。它未证明该方向在完整视频中“必要”，也没有把终端 latent 约26%的 channel-constant 误差归因于这里的 FFN 均值。GELU 的 odd/even 分解允许净均值为正或负；DC 也没有被当作亮度。

## DISCOVERY：保留的竞争预测

- **边际矩与联合信息分离。** 若后续取得更适合 W1 投影的联合摘要，pre-W1 预测可接近 post-W1 参照；需要同时计入观测、W1 向量积和权重访问。
- **非高斯有限样本结构。** 正确的一阶/二阶矩仍可丢失尾部信息；同均值同方差的稀有正负脉冲与 Gaussian 有不同 GELU 均值。本次未测经验 erf 均值或高阶闭合，故此解释仍开放。
- **signed W2/gate 过滤。** Gaussian 相对误差从 raw 到当前 gate 变小，不意味着所有系数都预测准确，也不意味着 gate 无害；必要性在后续路径中才定义。
- **均值之外的功能。** 误差 `[+u,-u]` 的均值为零，逐 token SSE 却非零；空间组织、attention 的非双随机运输和后续非线性还可再产生公共分量。这使 panel 均值成功与完整候选失败可以共存。

最便宜尚未完成的判别是同一保存 z 上的系数级经验 erf 均值与精确 Gaussian erf 对照，并核查固定 quadrature；它无需重跑 W1 或模型。若这一步支持边际闭合，再构造多个**竞争的**可获得联合摘要，不预选某个低秩协方差或均值修复算子。本次固定协议已经结束，以上均是未执行的新研究范围。

## 图与计算含义

图锚点为 F03（post-Q1、before-W1）、F04（W1）、F06（GELU）、F08R/F08（原生 Q2 scale tree/post-Q2）、F09（W2）和 F12（gate/residual）。矩传播边为 conditional 数学表达，保存桥为 posthoc 观察；`panel mean -> full-M DC -> terminal latent -> full-video function` 各边仍 OPEN。

本分析没有移除原始 FFN 的工作。post-W1 矩已经支付 W1；pre-W1 统计仍需读取 q1，并通过两个完整 W1 向量积和完整 W2 均值 readout 才得到本次对象。把它用于实际生成时，需要证明这些费用能换来被避免的行/列/tile 工作，以及剩余逐 token 功能和四步视频仍成立。现有 native HiF4 的共享量化尺度也没有被 Gaussian 标量模型替代。

## EXECUTION / 回执闭合与作者范围

root 对精确协议登记后运行一次；工具作者参与本次回执分析，因此这是独立解释和身份核查，并非另一作者的数值重现。只读终态 RUN，未重新读取原始张量计算其他实测指标。核对完成：

- RUN 与协议 SHA、输入绑定完全一致；四个 panel 顺序、row IDs、seed42/54、block10 step1 相符。
- 128次矩阵乘法全部完成；扣账/完成 MAC 同为 `2,406,481,920=34DH`，包括四 panel 的8个 W1 向量积和26个 W2 readout。
- 3.6591129787s，峰值 RSS `1,200,381,952 bytes`，Torch intra/inter-op 与 BLAS/OpenMP 环境均为1。
- 0 GPU/model/native-QDQ/W1逐行重跑/训练/拟合/权重分解/新 bank/新视频；输出116,710 bytes，失败重试0。

精确哈希：

| 对象 | SHA256 |
|---|---|
| 协议 `independent-even-residual-mean-cpu-plan-20261026_v1.json` | `8eee207d5e0b248270aab3235a50a61c4e8b3ce0a45bbadff40c51453764f01f` |
| 工具 `e0_even_residual_mean_cpu_discriminator.py` | `8dc51961a353f957acaf063cb3421097f43fe44df3150df77fc62c2bcf2a9f1f` |
| 12项合成测试 | `12b551a8de12782c1f48634ffe44e4f4371bc6350003bf51513b6ed0648e5df4` |
| 原始/公开精确 RUN | `541881867992b382bee8ae77f975d9d2e79aa6ec8c1a33fc6f9750f979fdf07d` |

数学/近邻来源沿用已读的 [GELU 原论文](https://arxiv.org/html/1606.08415v5)、[Gaussian GELU 矩推导](https://arxiv.org/html/2601.22307v1)、[PBP 的对角 Gaussian 传播](https://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf) 与 [已知的 analytic bias correction](https://arxiv.org/pdf/1906.04721)。本次机制增量是这个真实 FFN 保存对象上的**边际闭合—联合信息缺口—signed readout**分离，未宣称 Gaussian 期望或均值校正公式本身为新方法。
