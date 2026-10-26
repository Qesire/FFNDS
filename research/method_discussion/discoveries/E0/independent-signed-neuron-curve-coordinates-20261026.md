# DISCOVERY · 将多个神经元重写为一条向量值响应曲线

2026-10-26，独立 mechanism lane B；工作基线 `5a907228d92edcef01d750c3d10dcaf52e92b6e3`。

**研究对象：一簇共享输入投影的神经元，可以合成为保留 W2 有符号几何的向量值一维非线性曲线。** 这把“留下哪些原通道”改写为“哪些投影坐标足以生成必要的 FFN 输出”。曲线可保留弯曲、偏置和相消，无须先把每个神经元线性化。最强竞争解释是：Wan 的函数方向并不聚簇，或者 Q2 共享 scale 使低维曲线变成多变量函数；此时曲线所需坐标/阶数会吃掉收益。最便宜的判别是对已存完整 hidden 小面板做**指定小簇的有符号响应残差**检查，并同时保留原量化闭包，而非先拟合全网新权重。

**执行边界：** 本轮只读源码、JSON、原始论文与作者实现，作条件推导；没有模型/GPU、saved-tensor 数值实验、拟合、分解或新视频。后文的 saved-data 检查是待单独批准的最小研究动作，质量/速度均未验收。

## DISCOVERY：从单个代表神经元到响应曲线

取一行真实 post-Q1 输入 `q∈R^D`，`D=5120`、hidden `H=13824`。记实际 BF16 decoded W1 的行向量为 `w_i`，W2 的列向量为 `v_i∈R^D`，Smooth2 为正数 `s_i`，`φ` 为 Wan 的 tanh-GELU。先定义实数连续参照

\[
F_0(q)=b_2+\sum_{i=1}^{H}\bar v_i\,\phi(w_i^Tq+b_i),\qquad \bar v_i=v_i/s_i.
\tag{1}
\]

这个参照保留原 decoded 权重与 Smooth2；原生 Q2/中间 BF16 的差额在后面单列。`q` 已经是实际 Q1 结果，求它仍支付原 Smooth1/Q1 开销。

若簇 `C_c` 的行满足 `w_i=α_i u_c`，则定义

\[
t_c=u_c^Tq,\qquad
H_c(t)=\sum_{i\in C_c}\bar v_i\phi(\alpha_i t+b_i),\qquad
F_0(q)=b_2+\sum_c H_c(t_c).
\tag{2}
\]

**ALG，条件精确：** 只要这些行严格共享方向，式(2)是直接代入恒等式。`α_i` 可带符号，bias 可各不相同；它要求共同投影，不要求共同激活、共同输出向量或 Gaussian 输入。GELU 非齐次，所以 `H_c` 一般并非“一个 GELU × 合并 W2 向量”。把它当成可压缩的**向量值曲线**，比强迫选出一个代表神经元多一个自由度。

令各 `H_c` 在明确区间 `I_c` 上由 degree `p_c` 的共享标量基 `T_k` 表示，例如缩放后的 Chebyshev 多项式：

\[
H_c(t)\approx B_{c0}+\sum_{k=1}^{p_c}B_{ck}T_k(t),\qquad B_{ck}\in\mathbb R^D.
\tag{3}
\]

常数 `Σ_c B_c0` 可并入 bias。在线图是 `Q Uᵀ → 逐坐标 T_k → Φ Bᵀ+b`，全部为静态稠密小 GEMM 加逐元素运算；不必生成原宽度 H 的 hidden。系数应近似整个 `H_c`，直接保留 `Σ_i v_i γ_ik/s_i` 的符号与方向，而非用 `Σ_i||v_i||` 或激活幅度代替它。式(3)是近似；当前没有 Wan 的误差或可压缩簇数测量。

这是一种已知 ridge/projection-pursuit 与 feature-bundling 原语在 Wan 上的条件构造。新增研究问题是**有符号曲线复杂度与实际 Q2 闭包能否同时保持低维**，不是宣称发明了 ridge functions、kernel fusion 或多项式替代。

## 条件、误差几何与一个尖锐反例

真实小簇可写为 `w_i=α_i u_c+e_i`。令 `r_i(q)=φ(α_i t_c+b_i+e_iᵀq)−φ(α_i t_c+b_i)`，`V̄_c` 的列是 `v̄_i`，当前 gate 为对角阵 `G`。则连续参照的簇误差满足

\[
\|G(F_{0,c}(q)-H_c(t_c))\|^2
=r(q)^T\bar V_c^TG^2\bar V_c\,r(q).
\tag{4}
\]

这是精确有符号度量，含 readout 的交叉项。若相关区间内 `|φ′|≤L_i`，一个保守充分界为

\[
\|G(F_{0,c}-H_c)\|
\le \sum_{i\in C_c}\|G\bar v_i\|L_i|e_i^Tq|.
\tag{5}
\]

多项式的曲线误差另加 `||G(H_c−Ĥ_c)||`。式(5)仅为充分界；数据协方差方向与式(4)的 signed cancellation 可以使它很松。研究应先比较功能残差，再决定是否需要 weight-space 或 activation-space 聚类。

**反例：几乎相同的 W1 行，可能只通过差分表达功能。** 取 `u⊥r`，两行 `w_±=u±εr`，相同零 bias/相同 Smooth2=1，readout 为 `v_+=v`、`v_−=−v`。W1 余弦随 `ε→0` 趋近1。普通 centroid merge 给零，因为两个 readout 相加为零；原函数却为

\[
v[\phi(t+\varepsilon s)-\phi(t-\varepsilon s)]
=2\varepsilon s\phi'(t)v+O(\varepsilon^3s^3),\quad
t=u^Tq,\ s=r^Tq.
\tag{6}
\]

在 `t=0`，由已知 tanh-GELU 的 `φ(x)−φ(−x)=x`，输出恰为 `εs v`。合并丢掉这对神经元的全部相对功能，即使权重无限接近。这不是新的 GELU 恒等式，而是它对 neuron merging 的反例：**小 weight/activation 距离与重要的 signed differential mode 可以共存。** 若此模式在 Wan 出现，应保留共享的差分坐标 `s`，或把该簇留给原计算；二者都要付费。每对都需要独立 `s` 时，W1 不再省算。

另一竞争情形是严格共线但 bias/α 形成许多不同拐点，且 W2 符号使曲线反复弯曲；这时方向压缩成立，低阶曲线压缩仍可能失败。当前构造的价值取决于两个实测量：`需要多少投影`与`每个投影需要多少非线性基`。

## EVIDENCE：真实 Wan 图与 native quantizer 决定哪些等式可移用

已读 `WanTransformerInfer.infer_ffn`：先 `x.add_(attn_out)`，再 norm2/modulation，`ffn_0.apply → F.gelu(approximate='tanh') → ffn_2.apply`；`post_process` 用原 gate 乘输出再写入 residual。实际 adapter 两次 apply 都先做 FP32 Smooth 除法、cast 回 caller dtype，再调用 native HiF4 QDQ，最后用 decoded weight 做 BF16 GEMM。故真实函数应记作

`F_native(q)=BF16Linear[Q2(BF16GELU(BF16Linear(q,W1,b1))/S2),W2,b2]`，

其中写法省略每个已核对的 cast 细节，仅表达先后顺序。式(1)没有把 `q2_post` 当成 GELU，也没有声称跨 Q2 的代数交换成立。

源码链 `wan_i2v.native_hif4_activation_qdq → hif4_backend.hif4_qdq_tensor → quant_dequant_float(QType('hifx4').dim(-1))` 保留组依赖。读取的 Python 参考 `hifx.py` 把最后轴展开成 `(-1,8,2,4)`，max/scale 跨 4/8/64 层次共享；`QTensor.py` 根据形状补齐并选择 CUDA 实现或 Python 路径。这里核对的是源依赖与参考公式，没有反汇编 native `.so` 或执行量化器。

这给出两个明确选择：

- **保留原 Q2 语义的条件路线。** 若某个完整 Q2 闭包中所有实际 pre-Q2 值都仅依赖一个 `t_c`，其整组量化输出经过 W2 仍可定义为一条 `H_c^Q(t_c)`。它通常带量化跳变，近似基要覆盖这些跳变；BF16/GEMM shape 的位级一致性还需原生验证。若同一个共享-scale 组同时依赖 `t_c,t_d`，原生响应通常是多变量的，独立曲线相加就失去该条件精确性。
- **更换计算表示的近似路线。** 保留原 Q1，直接使用式(3)替代原 W1/GELU/Q2/W2；相对原 native 的误差需拆为曲线/方向误差及 `W2[Q2(φ/S2)−φ/S2]`、dtype/GEMM 差额。组64不是这条新表示的强制工作单元，但原 Q2 的功能差额应实际计入；小多项式误差不自动意味着这项也小。

因此，最有价值的源码判别已经得到：**任意 neuron cluster 与原 quantizer closure 是两个不同划分。** “同一标量控制若干神经元”的连续结构，需要与“哪些其它坐标会改变它的 Q2 scale”一起检查。它不是把现有 group64 支持再改个阈值。

## 实际 W1/W2 工作与物理布局

设参与压缩的 `H_a` 个原神经元被重写成 `C` 条曲线，非恒定基总数 `R=Σ_c p_c`；其余 `H_e=H−H_a` 原神经元保留。若保留原 Q2，`H_e` 还必须包括重建其 scale 所需的完整依赖闭包；闭包扩大时相应减少 `H_a`，而非免费补出被省略的 pre-Q2 值。在此条件下，投影 MAC 的账为

\[
\mathrm{MAC}_{old}=2MDH,\qquad
\mathrm{MAC}_{new}=MD(2H_e+C+R)+O(MR).
\tag{7}
\]

其中 `M=31668`，第一个新增 GEMM 真正只计算 `C` 个 `u_cᵀq`，第二个只读出 `R` 个基；bias 另计 `O(MD)`。若每簇8个神经元且全用 cubic，则该部分理想投影比为 `(1+3)/(2×8)=25%`。这是形状算术示例，不是 Wan 可达到的压缩率或时间预测。

原 Q1、gate/residual、未压缩通道的 GELU/Q2、basis 构造、读取 B、可能的 packing 与新增权重驻留都在实际成本里。仅把完整 hidden 投影到少量输出方向仍支付全部 W1；式(7)的收益要求把宽 W1 本身换成 `U`。若增加每簇切向坐标，W1 行数与基数都必须随之增加。

对多区间 spline，不能免费假设“每行选中的 W2”仍是一条稠密 GEMM：计算所有 interval 的 one-hot 特征会使读出宽度增大；按 interval 分桶则增加路由、行重排和小 GEMM。当前先讨论一个冻结区间上的静态多项式图，另把越界率作为判别量，避免把条件 fallback 的成本藏起来。

作者 EUGen 实现给了一个直接启示：`GeluApproxKernel.forward` 仍显式计算 `x0=inputs @ weights.T`；低 random-feature 数本身不保证原投影消失。其 `bundle_sequential_block` 直接识别 `NNK_Relu` 与 Linear 并矩阵相乘。对新曲线应检查真正执行的两条小 GEMM，不能只报 feature 数或照搬 generic GELU wrapper。[原实现 eugen.py](https://github.com/arijitthegame/EUGen/blob/e40677d3ed3ad67c25928db6d053956d2bd2ffd0/src/eugen.py)、[bundle.py](https://github.com/arijitthegame/EUGen/blob/e40677d3ed3ad67c25928db6d053956d2bd2ffd0/src/bundle.py)

## 竞争预测与最便宜的 saved-data 判别

| 机制 | 区分性预测 | 小而有信息的检查 |
| --- | --- | --- |
| 响应曲线可压缩 | 少量已知投影能保留多个神经元合成后的弯曲响应，基预算低于省下的 W1/W2 宽度 | 固定一个小簇/一层，先用保存的 `z/gelu` 作 signed 曲线残差，再对不同来源和同专家两个 step 看同一曲线是否仍有效 |
| 普通近邻合并足够 | 一个代表 GELU 的误差已接近整条向量曲线 | 同一投影预算下比较 representative-GELU、vector curve 与保持差分模式的二坐标版本；配对 readout 符号应预测谁需要第二坐标 |
| 低维只来自 teacher 面板 | teacher 有效，student 保存行偏离曲线或越界显著 | 只对已有 v3 两轨8行 `q1` 求预先固定坐标并报告覆盖/越界；这是小样本否证，不认证 full-M |
| 原 Q2 closure 主导 | 连续曲线好，但 native `q2_post` 加权的曲线残差大，并随共享 scale 或组内其它坐标改变 | 在同一旧48行、同一 W2/gate，分列 GELU/S2 与真实 q2_post 的 signed 残差；保持完整量化依赖组 |
| W1 方向分散/曲线高复杂度 | 候选簇普遍需要多个差分坐标或许多基，`C+R≥2H_a` | 先做预指定少量簇的可行性否证，达到此界即可说明该小簇在本预算下没有投影收益 |

现有原始被动数据有六来源、五层、四步的 `[48,13824]` `z/gelu/q2_pre/q2_post` 与 gate、`[48,5120]` q1/FFN/residual，完整 W1/W2 在旧校验 shard；这里没有读取其张量或计算聚类。文档依据为 `E0_INTRINSIC_FULL_HIDDEN.md` 的字段表。最小下一动作可固定一种来源/一步/一个小簇，利用现成字段先检验式(4)与 Q2 差额，再决定是否值得分层扩展；不需要重采模型。

最新两来源320-site结果已读：局部 raw student-operator SSE 在318处变小、两处相等，而状态差增长。这为新表示提供一项积极目标：观察 response/difference mode 时同时保留幅度和方向。该结果没有测量 neuron clusters，也没有证明曲线表示有效；将来 full-video 桥应比较实际 signed gate/writeback 与 `v_D(z_S)` 响应，来源间覆盖及正式视频质量仍独立检验。局部曲线拟合优越只决定是否继续研究该表示。

## 实际检索、最近机制与来源边界

2026-10-26 使用 web search/open/find、作者 GitHub raw/API；搜索簇依次为 neuron merging/GELU/相似 readout、TARDIS 原始题名/作者实现、polynomial/RaLiMP/SNNK/EUGen、ridge/projection-pursuit。外部信息改变了构造：从“合并近邻神经元”变为“保留 signed differential mode 的向量曲线”，再由 Q2 源码增加闭包条件。二级索引仅用于定位，下列数学比较依据原文/原代码。

| 原始来源与实际阅读位置 | 对本构造的关系与启发 |
| --- | --- |
| [Kim et al., Neuron Merging, 2020](https://arxiv.org/pdf/2010.13160)，§3.1 Theorem1、§3.3/Algorithms1–2；[作者 decompose.py](https://github.com/friendshipkim/neuron-merging/blob/d25cf3e39ecf376a1ffc8b3cbf326a38b6db19cf/decompose.py) | 最近的 neuron 参数合并。其跨 ReLU 缩放要求非负且每列至多一个非零；源码以余弦选代表并用范数比补偿 readout。共享投影是相同原语，GELU 响应曲线及 signed 差分保留是此处另提的单位；没有移用其 ReLU 精确性。 |
| [Hu et al., TARDIS v1, 2025](https://arxiv.org/html/2501.10054v1)，§5.1–5.4、§6、§7.7 | 原文已有每神经元单区间 affine、D×D folding、低精度 W1 predictor 与越界纠错。这里选择少量投影和非线性曲线基，不按原神经元做 affine+fix；任何未来越界纠错都属于已知近邻机制。检索作者名/题名与原文，没有找到并核验公开作者 runtime 代码；§6只是作者实现说明。 |
| [Belrose–Rigg, Closed-form polynomials v1, 2025](https://arxiv.org/html/2502.01032v1)，§3.1–3.4、Appendix B；[polyapprox/ols.py](https://github.com/EleutherAI/polyapprox/blob/430b3e2d3ee8a70492e7b4e1f1637a08868a3420/polyapprox/ols.py) 与 gelu.py | 已知 Gaussian/Gaussian-mixture 下的函数多项式近似。实际 quadratic 分支枚举 input pairs，且代码限制零均值/单位协方差；GELU 代码用 erf/CDF 版本。本构造只在少量 learned/原权重投影上取一维基，无此 Gaussian 假设；也不把 MNIST 结果移到 Wan。 |
| [SNNK v1, 2023 / ICLR2024](https://arxiv.org/html/2310.13225v1)，§3 Eq2–3、§3.3、§4.3.1；[EUGen v1, 2026](https://arxiv.org/html/2601.22563v1)，§3 Eq1–2、复杂度段及作者 eugen.py/bundle.py | kernel feature map 与后一 Linear bundling 已有。共享“薄 nonlinear features→合并 readout”的原理；当前区别是投影由具体 neuron 群的方向/有符号功能决定，依赖有待检验的 Wan cluster 假设。EUGen code 的完整线性支路还提醒不能把理论 feature width 直接当 W1 省算。 |
| [RaLiMP v1, 2026-10-09](https://arxiv.org/html/2610.13568v1)，§2.1–2.3、§2.5、§3.2/D.6 的训练与 distillation说明 | 强竞争者：rational/random-feature narrow bottleneck 可绕开单纯低阶多项式的范围误差。它不要求 neuron 共线，却有特征数/方差与适应训练代价。本文件没有核验其作者完整 runtime，也没有把其模型结果当零训练 Wan 证据。 |
| [Kuniya–Ohkubo, sigmoid neuron merging v1, 2026-10-01](https://arxiv.org/html/2610.02559v1)，§III.2–III.3 Eq17–22 | inverse-activation regression 把加权平均 sigmoid 响应拟合成代表 neuron，readout 求和。它是有意义的竞争构造；tanh-GELU 在负半轴非单射且 Q2 多对一，所以全局 logit 式反函数不直接适用。向量曲线可避开逆函数，并保留相反 readout 所需的差分响应。 |
| [Friedman–Stuetzle, PPR, 1981](https://doi.org/10.1080/01621459.1981.10477729)，原始 abstract；[Hokanson–Constantine v2, 2017](https://arxiv.org/html/1702.05859v2)，§1 Eq1–2 与 related-work | ridge/polynomial reduced coordinates 是经典统计/近似原语。2017工作显式优化投影子空间和多项式；此处没有运行其优化器或宣称该表示新颖。1981扫描 PDF 的公式未成功从文本提取，未冒称亲读完整算法。 |

## 源绑定、图锚点与交付状态

| 本地实际文件 | SHA-256 |
| --- | --- |
| `Wan infer/transformer_infer.py` | `5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225` |
| `HiFloat4/ptq/adapters/lightx2v_wan_distill.py` | `076482829b8ba542c109b63982b7501f131f8035f40a6f1807c8ef080ea00da3` |
| `HiFloat4/ptq/adapters/wan_i2v.py` | `d5a9be8ddb777fe360e66ba8e4ccde033b4a2969d2a0b853c7aefa0f45c44293` |
| `Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py` | `8d7589f6b65acb8e96ce14cf18adbce44f729110536725a825da47f521befbb6` |
| `HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py` | `6a8d1f1b8ea49ce099439cff251309b4d9b3eeeab6a7ee0a83bdcf85b24704f7` |
| `research/20261026/E0_PAIRED_FULL_DEPTH_RESULTS.json` | `1f1087d2d7fa111933fe2ce308ecbf4a9b7bd19f205d821973411a29115e16b2` |

作者源码 SHA256：neuron-merging `decompose.py`=`c4bc0dac56c380ae7791473d4a8f5bc0c2225c5b7bde1b378bbc5ad6e22ea85f`；polyapprox `ols.py`=`ea0b99f8f453e0c2dce87f034ddda8797fec44b5018600551cdbced72ea58320`；EUGen `eugen.py`=`18912f2bab11ac6ff14b890ce5b58d1f2043b8362833c5e3f8e805759e5f3637`、`bundle.py`=`a049f58923041381bd1ecf20575d75f4c4bb3c7c96ef6fabaf8d3edabcc23d9f`。远端内容通过固定 commit raw URL 读取与摘要，不执行外部代码。

源路径根分别为 `.runtime/code/LightX2V/lightx2v/models/networks/wan/`、`.runtime/code/HiF4_Sp/` 和 `.runtime/code/` 下上述 Wan2.2 仓库。图锚点 `G18→F04/F06→F08R/F08→F09→F12/F13`；`P23/P24` 是已测配对证据，给新表示的 rollout 评价提供参照。式(2)/(4)为条件 ALG；曲线低复杂度与新执行图是 HYPOTHESIS/conditional；与最终视频功能的边仍 OPEN。只新增此独立文件，不改共享 ACTIVE、graph、atlas 或冻结资产。
