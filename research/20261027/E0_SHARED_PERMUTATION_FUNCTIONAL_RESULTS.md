# 共享输入的功能表示：联合重排界与更强的精确投影对照

**新结构（ALG）**：同一个输入排列被全部 W1 行共享，能够保留单行区间丢掉的有符号联合关系。对一个固定线性功能方向，先合并权重即可得到精确的联合支持；当实际有序输入已经可读时，同一个合并方向还给出了更强的**直接精确计算**对照。

本轮从最新 [I-P1](PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md) 出发，完成证明、原论文/作者代码比较、604 个有限排列的软件核对及[独立审阅](../method_discussion/discoveries/E0/independent-shared-permutation-functional-bound-review-20261027.md)。本研究不使用真实 Wan 张量；原生 QDQ、模型及视频均为零。

## DISCOVERY：共享的是排列，研究目标是函数

给定有序权重 `W`、bias `b` 和输入 multiset `x`，离散联合域为

\[
\mathcal Z=\{b+WPx:P\text{ 为合法共同排列}\},\qquad
\operatorname{conv}\mathcal Z=b+W\operatorname{Perm}(x).
\]

`W Perm(x)` 是精确凸包，通常不等于离散域。固定有符号方向 `c` 后，令 `p=Wᵀc`，则

\[
\max_{z\in\mathcal Z}c^Tz=c^Tb+\sum_i p_{(i)}x_{(i)},
\qquad
\min_{z\in\mathcal Z}c^Tz=c^Tb+\sum_i p_{(i)}x_{(n+1-i)}.
\]

合法独立尺度交集组分别计算后相加。正尺度、码字/位置约束及已读取的配对均属于输入信息域。自然输入并非被假定在这些排列上随机分布。

**信息区别**：逐行 weight histogram 无法恢复跨行的坐标对齐。两组权重每行都含 `{−1,+1}`，仍可分别给出恒零功能与范围 `[−4,4]`。形成 `Wᵀc` 需要有序权重或足够的新增联合元数据；这项信息与预处理要计价。

## EVIDENCE：有限核对及非线性反例

[冻结协议](../../config/e0_shared_permutation_discriminator_20261027_v1.json) · [原始结果副本](E0_SHARED_PERMUTATION_FUNCTIONAL_RESULTS.json) · [检查工具](../../tools/e0_shared_permutation_discriminator.py)。

| 固定整数构造 | 输入排列数 | 逐行外包络的函数范围 | 共享输入的精确范围 |
|---|---:|---:|---:|
| 相同行、反号系数 | 2 | `[−4,4]` | `[0,0]` |
| 相反行、同号系数 | 2 | `[−4,4]` | `[0,0]` |
| 三行有符号组合 | 24 | `[−35,31]` | `[−23,23]` |
| 两个独立尺度交集组 | 576 | `[−62,67]` | `[−31,40]` |

所有整数极值与穷举一致，且逐个核对 `cᵀWx=(Wᵀc)ᵀx`。真实使用的连续 tanh-GELU 数学表达则给出两种区别：

- `c=−1`、预激活可取 `{−2,−1,1}` 时，线性排序选择 `−2`，实际 `−GELU_tanh` 的最大值在 `−1`。负尾非单调改变极值。
- 反向预激活 `z,−z`、同号系数时，线性功能为零，而 `GELU(z)+GELU(−z)` 保留偶残差。例中两个离散输出都约为 `1.909195`；凸化输入的中点给出 `0`，显示非线性前后凸化顺序的差异。

这些浮点值是 stdlib 标量计算；它们不建立 BF16/HiF4 等价。独立审阅还给出反对称 GELU 配对的精确例外、一般非线性支持的组合困难，以及固定量化尺度仍会因舍入破坏配对恒等式的反例。

## 最强竞争机制与物理含义

固定 `c` 时可以离线计算 `p=Wᵀc`。实际有序 Q1 输入可读后，`pᵀx` 只需 `O(D)` 即得到这个线性功能的精确值。**联合证书首先要和这条精确投影比较**。相对完整 `O(HD)` W1 显得便宜，并不足以证明证书值得运行。

输入直方图已存在、方向可以反复复用时，有序 `p` 的 prefix sums 可将有限码本支持查询变成少量计数查询；同时要核算当前 Q1 编码/统计、prefix 读取、存储和多 token 摊销。若 `c`、门控或非线性包络斜率随当前输入变化，`Wᵀc` 的在线形成又可能恢复 `O(HD)` 成本。真实 gate 是否具有所需可复用性，本研究保留为待核问题。

最接近的成熟方法是 [CROWN §3 的符号相关仿射传播](https://papers.neurips.cc/paper_files/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf)与 [DeepPoly 的关系型抽象域](https://ggndpsngh.github.io/files/DeepPoly.pdf)。共享排列支持可以作为这种传播的输入域 oracle；它不是新发明的通用认证框架。独立审阅实际核对的 auto_LiRPA `BoundGelu` 使用 erf GELU，迁移到 Wan tanh 近似和 HiF4 整组闭包还需相应有效包络。经典重排来源包括 [Vince 1990](https://people.clas.ufl.edu/avince/files/Rearrangement.pdf)；没有数学优先权或算法速度宣称。

**区分预测与下一动作**：若少量固定功能方向及其非线性/Q2残差约束可以复用，证书可能用少于精确求值的信息帮助决策；若必须逐 token 重组方向或读取同等字节，精确投影/原生 dense 更强。下一项有价值的证据是先核实方向、门控和原生尺度何时固定，再比较这两种实际信息/字节成本。完整 FFN 与视频功能依然需要后续独立证据。

## EXECUTION

用户 2026-10-27 恢复指令下登记，单 CPU、604 个整数排列，HF 原始 `RUN` SHA `0bc3ccb775a706520c1cf4d41a3826ea734a736622fcc81a258de1db11acdc71`。四项软件测试通过。原始失败/暂停和既有模型范围保持原样；本轮贡献属于数学与有限软件佐证。
