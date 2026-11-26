# 当前四步方法究竟节省多少算量

日期：2026-11-26。使用 operator-tensor-analysis 的 C01 成本模型、C02 完整复用周期及收缩执行规则，沿实际 Wan / wide8192 / C 的代码计算。本文执行了整数／有理数 CPU 算术，没有新增模型运行或 GPU 计时。

## 1. 可以直接使用的结论

以**相同四步蒸馏模型的原生 attention + 原生 FFN**为分母，当前 BF16 wide8192 + 冻结 C 的主要矩阵运算预计减少约 **28%**，对应约 **1.39× 的算术量倍率**。纳入显式探针、特征与读出矩阵乘后，账本为 **27.81%–27.90%** 的节省、**1.385–1.387×** 的算术量倍率。两个端点来自实际 selected block 尾块长度，而非假定密度精确等于 1/8。

当前 2 个原生锚点 + 2 个命中、只改 FFN 和 self QK/AV 核的调度中，假设所有替代和附加缓存工作都免费，主矩阵算量的乐观结构上限是 **37.02% 节省、1.58774×**。它界定这个固定作用域的算量空间，不构成延迟承诺，也不限制改变计算范围或锚点调度后的其他方法。

新的 centered low 缓存保持原 C 的主要矩阵调用与支持不变，新增的是 O(ND) 的统计、编码和解码。故其理论主 GEMM 节省与原 C 相同；本轮真实时间略降，须作为新表示及编译融合实现的共同结果解释。

## 2. 对象、实际形状和执行周期

每片执行四次 denoiser，专家序列 high/high/low/low；每个专家 40 层。当前四步蒸馏合同中 `enable_cfg=false`、`sample_guide_scale=1.0`，执行收据确认 160 个 attention 和 160 个 FFN 位点。每种模块各 80 个 anchor、80 个 hit；FFN P64 教师在每个 hit 仍执行原生完整宽度。

| 符号 | 当前数值 | 含义 |
|---|---:|---|
| N | 31,668 | 21×29×52 视频 token |
| D | 5,120 | 残差宽度 |
| M | 13,824 | 原生 FFN 隐宽 |
| K | 2,560 | FFN 选中隐藏通道 |
| P | 64 | 原生教师 query/token 数 |
| G | 16 | FFN 共享增益模态 |
| H、d_h | 40、128 | attention heads、每头宽度 |
| T | 512 | 本账本采用的归档文本上下文长度 |
| attention block | 128 | 路由 query/key block |
| selected blocks | 31 | 每个 query block 选中 key blocks |
| wide8192 | 8,192 行 | 命中执行 tile；不改变上述隐藏宽度 |

每个 layer 的四次 self core 调用为：原生、selected + P 教师、原生 + selected 建缓存、selected + P 教师。high anchor 只建路由，low anchor 还要执行 selected 输出和质量计算。

FFN 原生 anchor 只执行一次完整 W1/GELU/W2，然后记录 u0、y0、z0k；当前 `fourstep_anchor.py` 可以让 W2 直接写入 y0 槽。它不额外执行一次完整 dense A0 投影来建锚点。A0 在 hit 执行。

## 3. 原生四步的主矩阵账本

一次乘加记 1 MAC，FLOP=2×MAC。以下按一个 layer、一个 denoiser 调用计，先只覆盖主要矩阵乘：

| 分项 | MAC 公式 | TFLOP | 占本表合计 |
|---|---|---:|---:|
| self QK + AV | 2N²D | 20.53862 | 51.54% |
| self Q/K/V/O 投影 | 4ND² | 6.64126 | 16.66% |
| FFN 两次投影 | 2NDM | 8.96570 | 22.50% |
| cross Q/O 投影 | 2ND² | 3.32063 | 8.33% |
| cross K/V 投影 | 2TD² | 0.05369 | 0.13% |
| cross QK + AV | 2NTD | 0.33206 | 0.83% |
| **合计** |  | **39.85196** | **100%** |

四步、40 层合计约 **6.37631 PFLOP** 的本表主矩阵乘。归一化、GELU、softmax、RoPE、门控、布局、内存、权重卸载及 block 外的 VAE、编码和保存不在这个数值内。

## 4. FFN：当前命中仍包含完整 D×D 作用

真实 `fourstep_hit.py` 的主要 hit 账本是

\[
F_{hit}^{major}=ND^2+2NDK.
\]

第一项来自 ΔU A0，后两项是 selected W1/W2。当前 K=D/2，故两组主项恰好各占一半：各 1.660315 TFLOP，合计 **3.320630 TFLOP**。

相对原生 FFN 8.965702 TFLOP，主要 hit 比为 10/27≈37.04%。但每个 anchor 只供一个 hit，周期比为

\[
r_F^{major}=\frac{2NDM+(ND^2+2NDK)}{2(2NDM)}
=\frac{37}{54}\approx0.685185.
\]

因此这一周期的主要 FFN 算量节省为 **31.48%**，算术倍率 **1.45946×**。

继续计入代码中的 N 行 16 模态映射、N×K×P kernel、80 列全输出读出、单独 P64 operands、原生 P64 教师及显式小矩阵乘，hit 约 **3.38962 TFLOP**，周期比 **0.689033**，即节省 **31.10%**。Cholesky、三角求解及逐元素操作仍另计。

这也给出简化的尺度判断：P64 和小求解不是主要算术体量。即便理想化删除当前 ΔU A0 一项，对整个四步主矩阵分母也只对应约 **2.08 个百分点**；selected 两次投影合计也是约 2.08 个百分点。保留目标质量的前提下，真正的大幅空间变化需要触及作用范围或原生锚点调度。

## 5. Attention：选中约 1/8 的 keys 不等于整周期只剩 1/8

N=31,668、block=128，共 248 个 blocks，末块只有 52 行。每个 query block 选 31 个 key blocks；若包含末块，实际 keys 为 3,892，否则为 3,968。因此

\[
\alpha\in[3892/31668,3968/31668]
=[0.1229001,0.1253000],\qquad
\beta=P/N=0.00202097.
\]

四步共有 2 次完整 core、3 次 selected core、2 次 P64 全键教师，所以

\[
r_A=\frac{2+3\alpha+2\beta}{4}
=\frac12+\frac34\alpha+\frac12\beta
\in[0.593186,0.594985].
\]

对应 **40.50%–40.68% 的 self QK/AV 核算量节省**。当前 QKV/O 投影保持完整，连这些投影一起计，self-attention 主矩阵周期比为 **0.69259–0.69395**，即该模块主矩阵量约省 **30.61%–30.74%**。

额外账本还包括：两次 pooled QK 路由；三次 selected 输出的 128→64→31 特征投影；两次全 N×32→128 恢复；P64 的正规 Gram 和系数乘法。low 原生锚点直接返回 full LSE，当前实现未额外无条件执行一遍完整 QK 以取得它。

极少数数值失配行通过直接遗漏质量重新计算 QK。job 201500 两来源对应 8,776 和 16,485 个 query/head 行，新增 QK MAC 仅约为整片上述原生主矩阵分母的 **0.00112% / 0.00210%**。这说明该分支的 MAC 很小；实际小批调用、同步、归约和索引的时间仍需按执行成本理解。

## 6. 合并账本和指定范围内的结构上限

令 A 为原生 self QK/AV 核，F 为原生 FFN，R 为其余保留的主矩阵乘，D_0=A+F+R。四步统一账本是

\[
r_{all}=\frac{R+r_AA+r_FF}{D_0}.
\]

| 账本 | 剩余比例 | 节省 | 算术倍率 |
|---|---:|---:|---:|
| 主矩阵、保留当前selected及FFN底座 | 0.71951–0.72044 | 27.96%–28.05% | 1.3880–1.3898× |
| 加入列明的特征／教师／恢复 GEMM | 0.72100–0.72193 | 27.81%–27.90% | 1.3852–1.3870× |
| 两个 hit 的 self core/FFN 及附加状态工作全部免费 | 0.629826 | 37.02% | 1.58774× |

第三行是把所有命中替代费用、selected 建缓存等附加费用一并设零的乐观参照：

\[
r_{floor}=\frac{4R+2(A+F)}{4D_0}
=\frac12+\frac12\frac{R}{D_0}=0.629826.
\]

当前列明账本距离这个结构参照只余约 **9.12–9.21 个原生主算量百分点**，相当于当前账本的约 **12.65%–12.76%**。这些数值用于判断算术空间，不是已经可取得的进一步节省。

若扩大作用域到整个 hit block 并让两个完整 hit 都免费，另一个更宽松的二锚点参照才是 50% 算量、2×。它改变了被替换的计算范围，属于另一个方法问题。更换锚点数、跨专家处理或减少 NFE 也会改变参照，不应把 1.58774× 当作未来所有算法的上限。

## 7. 为什么约 28% 算量不直接等于约 28% 时间

本文计算的是有用的矩阵乘量。当前 hit 同时含 BF16 Tensor Core 路径、FP32 恢复与 FP64 小求解，吞吐不同；另有全位置状态读写、归约、selected 稀疏调度、kernel 调用、offload 及固定 pipeline 开销。

FFN 常驻 anchor 容量为 **30.20096 GiB**；low attention 完整 BF16 历史字段另为 **12.08038 GiB**，特征及 rho 还需约 3.34414 GB，状态的读写并非免费。centered 改变历史表示、归约和融合，而没有减少这些主矩阵调用。

对实际时间，合适的 Amdahl 形式是

\[
\frac{T_{new}}{T_{old}}
=1-f_{time}+f_{time}r_{time}+\varepsilon_{time},
\]

其中 f_time、r_time 必须来自一致计时范围，不能直接用上表 FLOP 占比替代。即便去噪加速，完整请求仍含未改变的编码、VAE 和保存费用。当前约 28% 的静态预算与六来源实测约 18.52% GPU 时间、13.52% 完整请求节省应分别报告。

## 附录：锚点复用次数如何改变理论空间

每个专家一个原生 anchor，服务 K 次 hit，FFN 比为

\[
r_F(K)=\frac{F+KH+C_{cache}}{(K+1)F}.
\]

若高低专家各自采用这一周期，沿用当前 selected 路由、低专家一次额外 selected 建锚和每 hit 的 P 教师，则 attention 核比为

\[
r_A(K)=\frac{1+(K+\tfrac12)\alpha+K\beta}{K+1}.
\]

当前 HHLL 对应 K=1。增加 K 改变调用次数与锚点陈旧程度，不能在保持当前四次调用的同时增加摊销次数；质量与状态充分性也需重新验证。蒸馏减少 NFE 的同时会压缩这种复用机会，这正是合并加速指标需要绑定真实中间基线的原因之一。

## 来源与可复算资产

- 主实现：[fourstep_hit.py](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/tools/ffnds_fourstep/fourstep_hit.py)、[fourstep_backend.py](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/tools/ffnds_fourstep/fourstep_backend.py)、[fourstep_anchor.py](https://github.com/Qesire/FFNDS/blob/13ffb127e71a5d4d1519ecb0df9c7b4d03efe601/tools/ffnds_fourstep/fourstep_anchor.py)。前两份本地源码与固定 main 字节一致；anchor 直接读取同一提交。
- attention：[冻结 C 源码](https://github.com/Qesire/FFNDS/blob/ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63/research/20261124/attention_wide8192/results/joint39/attention_online_structured.py)、[centered 实现](https://github.com/Qesire/FFNDS/blob/ec8a39b8a6c8c6a6cb25badfe11e8b6a38b09f63/research/20261124/attention_wide8192/results/joint39/analysis_20261126/joint_action_review/mass_centered_runtime/attention_mass_centered_low.py)。
- 维度来自固定 Wan A14B 源码、真实运行断言及已归档四步合同；N=31,668 和 BF16／P64／HHLL 另由当前收据确认。512 文本 keys 是本静态账本采用的归档配置。
- [recompute_theory_budget.py](recompute_theory_budget.py) 使用 Python 标准库的整数和 Fraction 实际执行；[theory_budget_verified.json](theory_budget_verified.json) 给出全部分项、边界、内存与条件性摊销例子。

运行：`python recompute_theory_budget.py`。脚本不需要模型权重或 GPU。
