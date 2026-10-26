# E0：投影后边际矩与 pre-W1 协方差信息的分离

**DISCOVERY — 在原生 dense FFN 的四个保存 panel 上，真实 W1 输出的均值/方差构成有用的 GELU 均值表示；仅保留 pre-W1 各输入通道方差时，这个表示明显退化。** 这把“投影后边际分布可近似”与“输入通道可视为独立”分开：二者是不同的假设。signed W2 和当前 gate 进一步改变均值误差的大小。

本次只描述来源0/12、seed42/54、high-noise block10 step1 的原始 dense 运算，各取原中心 contiguous128 与 uniform32。四个 panel 属于两个已暴露来源；均值是这些行的均值，未测 full-M DC、候选修复或新视频质量。

## EVIDENCE：同一原生均值目标

下面各列均为 **当前 gate 加权后的均值向量相对 SSE（百分数）**，分母是同一 panel 的实际保存 native FFN 均值能量；预测均加入一次已知 b2。SSE 比例与相对 L2 误差不同，例如 0.77% SSE 约为8.8%相对 L2。

| 来源 / panel | 真实 z 矩 → Gaussian tanh-GELU（固定 GH32） | q1 对角方差 → Gaussian erf-GELU | 保存 z → 连续 tanh-GELU / Smooth2 | 保存 post-Q2 → FP64 W2 + b2 |
|---|---:|---:|---:|---:|
| 0 / contiguous128 | 0.771540% | 14.294183% | 0.0161035% | 0.00000275% |
| 0 / uniform32 | 0.453336% | 8.076856% | 0.0168163% | 0.00001072% |
| 12 / contiguous128 | 0.199907% | 6.088939% | 0.00846563% | 0.00000141% |
| 12 / uniform32 | 0.179350% | 5.039150% | 0.0137612% | 0.00000363% |

真实 z 矩的 Gaussian tanh 预测比对角方差预测更接近该目标；这个观察支持研究矩信息，而没有设立部署准确度阈值。后者采用 erf 闭式；在同一真实 z 矩上，erf 与 tanh-GH32 两个 Gaussian readout 的 gated 差异能量只有 `4.43e-5–1.40e-4`，明显小于其与经验连续均值的差异能量 `0.520–1.882`。这对“主要差别只是换了 GELU 公式”的解释构成局部反证；固定 GH32 的误差尚未独立有界。

| 来源 / panel | 对角预测总 z 方差 / 保存总 z 方差 | pre-W1 预测 z 均值相对 L2 误差 |
|---|---:|---:|
| 0 / contiguous128 | 53.1355% | 0.02043% |
| 0 / uniform32 | 52.8550% | 0.04110% |
| 12 / contiguous128 | 54.8736% | 0.01959% |
| 12 / uniform32 | 56.6195% | 0.03789% |

线性均值预测接近保存值，而对角预测保留约53–57%的总投影方差。实际方差含有 `2 Σ(i<k) w_i w_k Cov(q_i,q_k)`；因此带符号的输入相关性是值得解释的对象。原生 BF16 W1 与 FP64 矩传播的方差差异未被独立有界，表中差额不能全部登记为“协方差贡献”。

## EVIDENCE：数值转换桥与 bias

连续 tanh 与保存 BF16 GELU/Smooth2 的 gated 均值差异能量为 `4.68e-6–2.12e-5`；保存 post-Q2 与保存 GELU/Smooth2 的合并差异能量为 `0.0309–0.0607`。后者小于四个 panel 的 Gaussian-tanh 与经验连续均值差异能量。这个特定 dense 均值诊断没有呈现“原生数值桥主导 Gaussian 均值差距”的模式；它不解释旧候选的全视频失败原因。不同桥的 SSE 含有有符号交叉项，不能相加作原因百分比。

128 行采集没有 pre-Q2，故其差异保留 FP32 Smooth2 除法、BF16 cast 和 Q2 的合并语义。两个 uniform32 与旧 passive archive 的 q1、z、GELU、post-Q2、输出、b1、b2、gate 均逐字节一致，可以借用真实 pre-Q2：两例 Q2-only gated 差异能量分别为 `0.0607675 / 0.0501859`，Smooth2/cast 桥仅 `1.59e-5 / 1.24e-5`。

已知 b2-only 在四个 panel 上仍留下 `92.90–93.86%` gated 相对均值 SSE；因此实际均值方向不是简单地重放输出 bias 就能解释。b1 保留在 GELU 自变量内，没有被错误视作可从非线性外加回的输出常数。

## 竞争预测与最便宜的下一步

当前证据最支持继续区分 **投影后边际矩** 与 **取得投影方差所需的联合输入信息**。强竞争解释包括：非高斯尾部/有限 panel 的相关采样、固定 GH32 误差、以及 W2/gate 对均值误差方向的过滤。真实 z 矩已经支付过 W1，属于 posthoc 信息；它的效果没有证明 pre-W1 的便宜可得性。

最小的后续判别可以只复用同一保存 z，比较经验 erf-GELU 均值与精确 Gaussian erf 闭式，并单独核查固定 quadrature 的敏感性，先在系数均值层区分非高斯闭合误差与激活/求积差异。这是新的拟议 saved-data 范围，本次没有执行。若联合矩信息随后仍有价值，再比较带符号的低成本协方差摘要与实际读取、权重访问和矩阵乘法成本。

均值保真可作为完整 FFN 功能的一个坐标；逐 token、空间/时间创新、后来 attention/residual/四步 flow，以及自由视频仍需要各自的观测。交替正负误差可以均值为零而逐 token 全错，因此本结果没有建立 full-M 常量模式的保护方法或视频修复。

## EXECUTION 与精确回执

注册的一次固定 CPU 分析完成：**2,406,481,920 MAC，128 次矩阵乘法全部返回，3.6591s，峰值 RSS 1,200,381,952 bytes（约1.118 GiB），单线程；模型/GPU/native QDQ/W1逐行重跑/拟合/权重分解/新视频均为0。** 两个 W1 均值/方差向量积每 panel 的成本已计入；元素运算、归约和读取另有成本。该时间描述一次 CPU 诊断，不是部署加速结果。

- [精确 RUN 副本](E0_EVEN_RESIDUAL_MEAN_CPU_RESULTS.json)：116,710 bytes，SHA256 `541881867992b382bee8ae77f975d9d2e79aa6ec8c1a33fc6f9750f979fdf07d`，与 HF attempt1 终态文件逐字节一致。
- [冻结协议](../method_discussion/discoveries/E0/independent-even-residual-mean-cpu-plan-20261026_v1.json)：SHA256 `8eee207d5e0b248270aab3235a50a61c4e8b3ce0a45bbadff40c51453764f01f`，绑定13项源码/来源和确切保存资产。
- [工具](../../tools/e0_even_residual_mean_cpu_discriminator.py)：SHA256 `8dc51961a353f957acaf063cb3421097f43fe44df3150df77fc62c2bcf2a9f1f`；[12项合成软件测试](../../tests/test_e0_even_residual_mean_cpu_discriminator.py)：SHA256 `12b551a8de12782c1f48634ffe44e4f4371bc6350003bf51513b6ed0648e5df4`。

本报告核对终态协议 SHA、两来源/四 panel 顺序、row IDs、seed、输入绑定、128次完成数和预算闭合；没有另跑原始张量重算。工具作者也参与本报告分析，独立原生复现不在本次证据范围内。`quality_accepted=false`，`full_M_DC_measured=false`。
