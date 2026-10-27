# G11 · 量化存储的符号与尺度能否预测FFN的可省计算（2026-10-27）

**对象：** 用户提出的“从量化存储直接获知粗幅值/符号，寻找可预测的计算结构”。研究程序是先验证信息存在性、获取时点和价格，再验证函数结构与物理省算。G11是ALG/SPEC/LIT/OPEN图，绝不表示新增Wan原生结果或选定候选算子。当前CPU/GPU/模型/视频实验继续受 [ACTIVE](../../ACTIVE_RESEARCH.json) 暂停约束。参考：[精确来源账本](G11_QUANTIZATION_SOURCE_LEDGER.json)、[主计算图](../../../docs/98_端到端计算图与证据注解_20261025.md)。

## 1. 信息是真的，但“计算前”有几种时点

有限、规范的低位格式将非零符号和幅值码本明确储存在bit码字。对于常用E2M1 FP4，重建幅值表为0、0.5、1、1.5、2、3、4、6。标准MXFP4为32元素共用一个E8M0幂二scale；NVFP4是16元素E4M3局部scale与FP32全tensor尺度组合。权重码和尺度可以离线常驻，而在线激活Q1码/尺度必须从当前输入做过归一化/量化后才能取得。Q2码/尺度只能在真实W1及GELU之后取得（若另造预测器，代价另算）。NVFP4权重可有2D block scale变体，模型checkpoint与运行recipe决定实际存储，不能互换。零码字的代数符号是0而非正负；正常scale非负，异常值和量化舍入另界定。

格式原文：[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)、[NVIDIA Transformer Engine NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)、[TensorRT-LLM视频量化实现说明](https://nvidia.github.io/TensorRT-LLM/1.4.0rc1/blogs/tech_blog/blog28_Accelerating_Video_Generation_with_GEMM_Quantization_Attention_Quantization_and_Skip_Softmax_Attention_in_TensorRT-LLM.html)。后者明确指出即使static量化checkpoint，NVFP4局部激活scale仍运行时动态生成。

| 信息 | 何时可用 | 额外代价和可省算边界 |
| --- | --- | --- |
| 离线weight码/scale/直方图 | before current block | 可预计算，但多一份weight metadata会耗显存与访存 |
| 当前q以及Q1后的sign/code/scale | before W1 | 要支付norm2/AdaLN/在线Q1；可研究省W1+GELU+Q2+W2 |
| weight×activation对齐符号异或、粗分桶正负贡献 | before W1，但需真正逐对获取或预计算可用摘要 | 对齐pair可能需要接近O(MDH)信息访问，必须与原生FP4 Tensor Core GEMM比较 |
| W1 preactivation、GELU | after W1 | 后验观测仅可能帮助省Q2/W2，不能称省了已经算过的W1 |
| 完整postQ2/HiF4 scale、gate×W2有符号贡献 | after W1/GELU/Q2/W2 | 可做离线监督标签；线上重算决策器/函数必须单独付费 |

## 2. 理论上可识别哪些符号，以及绝对无法“天然知道”哪些符号

编码值x_i=s_x(i)c_x(i)、w_ji=s_w(j,i)c_w(j,i)，故完整量化实数 W1 预激活是：

\[
z_j=b_{1,j}+\sum_i s_x(i)s_w(j,i)c_x(i)c_w(j,i).
\]

如果真正知道**相同坐标对齐**的两个非零码字符号，就知道每一项乘积正负。但点积净符号依赖正负贡献的数量、幅值和相消。仅从正负码字**边缘直方图**不能确定内积：固定w=(+1,-1)，输入x=(+1,-1)和x=(-1,+1)有相同sign/magnitude histogram，而内积分别为+2和-2。完整逐pair读取能解决关联缺失问题，却需要付访存、规约和位运算费，常见FP4低位GEMM已由Tensor Core专门加速。

**确定性证书**：如果已通过合法且成本受控的信息路径取得每个pair的乘积符号p_i，及幅值区间a_i≤|w_ji x_i|≤b_i，那么

\[
L_j=b_{1,j}+\sum_{p_i=+1}a_i-\sum_{p_i=-1}b_i,\quad
U_j=b_{1,j}+\sum_{p_i=+1}b_i-\sum_{p_i=-1}a_i,\quad z_j\in[L_j,U_j].
\]

L_j>0或U_j<0时，可认证**重建量化实数点积**的符号；要认证实际BF16数值须把舍入/scale依赖纳入区间。粗block-scale-only提供的上下界往往太宽，是否有足够“margin”可认证属于OPEN。负侧远离0的tanh-GELU有接近0的贡献，但它不是ReLU精确零；以 \(\Gamma_j=\sup_{z\in[L_j,U_j]}|\operatorname{GELU}_{tanh}(z)|\) 形成保守系数预算。

**有符号下游功能**：仅当原Q2逐组scale作用已合法计算/有可靠覆盖界时，才能构造postQ2系数h_j的范围，并评估被省列的gate/W2写回影响。例如对每个输出坐标r：

\[
|\Delta u_r|\le\sum_{j\in S_{\rm omit}}|g_rW_{2,rj}|\sup |h_j|.
\]

这种简单界抛弃了有符号cross-column抵消，可能很宽；更精确的联合方向估计仍须在同一次q、相同group-scale下成立，并考虑后续block/step/视频。**能判定GELU幅值**与**能判定FFN最终功能可以省略**是不同命题。

**统计预测路径**：另行探索cheap I=(已用的量化scale/sign/coarse codes、step、expert、时间局部性等)对后验结构的条件预测力。分别衡量 (a) W1负侧/小GELU，(b) postQ2门控W2的有符号、可相消、下游敏感方向，(c) token/tile跨时空共享的合法跳块。对同一输入比较scale-only、sign-only、scale+sign histogram、scale+aligned-sign、coarse-code、low-bit proxy；以fullpostW1/postQ2为**离线上界标签**。除了ranking overlap，必须报告FN漏检、source/seed/step分层、signed transported loss、实际kernel时间及元数据价格。增益须超过仅由q幅值/step/固定layer等简单统计构成的基线。

## 3. 原Wan HiF4的关键特例：对全group的依赖不是格式公共特性

FFNDS的原生路径为 norm2和FFN AdaLN → Smooth1/Q1 → W1[13824×5120] → tanh-GELU → Smooth2/HiF4 Q2 → W2[5120×13824] → e5 gate → BF16残差。HiF4原生是QDQ数值路径，Q2有64→8→4的group共享尺度。已保存的 [原Python参考反例](../../20261025/E0_HIF4_COLUMN_DEPENDENCY_COUNTEREXAMPLE.json) 仅把一个输入从7改为14，组内其余输入保持1.25，结果**64个输出都变化**；未变输入的index8输出从1.25变为1.5。因此即使W2选取若干列，该组的精确Q2结果仍可能依赖未选列的GELU，不能把“postquant单列带符号”当W1前已知。

原先 [full-M支撑与shared tile](../../20261025/E0_SHARED_TILE_PRE_W1.md) 已经观测：后验每行选择12.5/25/50%的postQ2活跃坐标仍使M16及更大的原生K64计算块占用100%。要研究的新机制是**可提早预测、可合规聚合、可兑现真实跳过的函数结构**，不是再次假设单元素稀疏率等于Tensor Core空tile率。SM120 CUTLASS packed NVFP4 primitive的源码审计存在，但没有对应当前A14B原生效果证明：[来源](../../method_discussion/discoveries/E0/independent-sm120-packed-fp4-primitive-source-review-20261026.md)。

## 4. 五篇直接竞争文献 L69–L73（阅读等级严格分开）

- [L69 Prox 2026](https://arxiv.org/pdf/2607.27591) 原始PDF §3–§4.3与图3–4已检查：SwiGLU两分支用输入稀疏与INT4 proxy得到 \(\tilde s=\tilde u\,\mathrm{SiLU}(\tilde h)\) 用于选择，随后原始full-precision三投影仅算入选通道。作者Eq.成本 \(C=2\alpha(1-s_1)+3(1-s_2)\)，取\(\alpha=1/3\)（其LLM GEMV实测近似），作者图3 exact/proxy交集约82%。它已占据“低比特proxy产生中间mask→入选通道精确重算”的强prior-art，不得作为FFNDS全新原理。与本研究的区分是**利用已存在的编码/严格margin与有符号Wan功能**，并验证是否省过proxy成本及group闭包。
- [L70 Spark Transformer NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/220d26c7fb478c2a397bd3fef0fe1d6f-Abstract-Conference.html) 官方摘要层：联合训练的低成本FFN/attention激活预测与线性复杂度statistical-topk，作者FFN active 8%及真实decode加速。不能把阈值统计/有训练router当从存储bit天然得到。
- [L71 SpQt](https://arxiv.org/pdf/2511.04477)原文§2–3与图1–3已核：groupwise quant布局与动态列跳算发生冲突，作者用zigzag低比特权重与特殊GEMV在Apple silicon small-batch LLM部署达1.55×。对FFNDS是quant layout/真实kernel强比较器，非A14B FFN大M验证。
- [L72 SQ-DM](https://arxiv.org/abs/2501.15448)作者摘要级：4bit diffusion激活稀疏、step-aware检测与定制稀疏加速器；作者6.91×是定制硬件指标，不可等同原生GPU Wan。
- [L73 CodeGEMM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html)正式摘要级：codebook centroid与activation的Psumbook、码索引查表重构部分内积。这提供**编码驱动计算表示重构**的竞争解释；LLM weight-only低位codebook并非HiF4 W4A4已经可复用的实现。

## 5. 最小区分次序（只设计，不启动模型或CPU数值）

1. 固定不同来源、seed、step、expert，核对已有q、quant codes、scale、postW1/gelu/Q2和gate是否真的保存；未保存则先记录未来观测接口需求，不能推断它们可即时访问。
2. 把scale-only、sign histogram、aligned-sign和码本离散幅值逐层累加的**认证覆盖率vs信息开销**画Pareto，特别看多少preact区间不跨0及GELU幅值预算；一旦读全部aligned code算出接近原点积的费用，则与原生低位dense kernel直接竞争。
3. 在相同信息成本下，预测有符号postQ2/W2/gate函数，检验同专家high/high、low/low与跨source/seed泛化；比较posterior oracle的失效例，隔离真实W2交叉项相消/叠加。
4. 原HiF4合法group闭包+实际M/K64 tile占用；反过来研究是否存在不同的共享功能组、编码域查表或训练后低成本函数表示，避免把已有topk僵化成路线。
5. 未来如另获模型/GPU授权，最后依次检验全40层four-step自由生成质量、背景/照明/运动、原生wall；实时衍生线额外TTFF/p99/chunk M和低位packed GEMM对照。

**科研结论身份**：量化码/scale确实提供初等符号和幅值信息（SPEC）；直方图无法唯一确定signed GEMM dot（ALG+反例）；HiF4 64组数值闭包与full-M空间支持（OBS）已经存在；“存在充分可预测且净物理省算的本征视频FFN函数结构”仍为OPEN。此处坚守 Q1信息存在、Q2信息取得成本、Q3真实省FFN工作三问。


## 6. 2026-10-27 一手论文与源码复核后的机制更新

已完成 [L69–L73 五篇原文方程/实证/源码审查](G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)，并新增 [L74 SharQ：FP4 + N:M 两路真实GEMM、Wan2.2 T2V 原视频数据及NVFP4源码](G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md)。摘要级旧阅读等级已升级。核心调整：Prox的INT4代理**真实花费两次GEMV**；Spark的统计预测来自**联合训练的模型**；SpQt多token prefill用dense，CodeGEMM的多token路径解完整量化权重再matmul；SQ-DM经ReLU替换/微调制造严格零；SharQ在线计算4:8稀疏主值并始终计算**dense残差**。这六个方向不能与“FP4符号与尺度可在W1前免费判定可省净函数”混为同一个主张。所有研究结果的模型/硬件不同，继续以功能净值、真实访存、scale-group闭包、native GPU kernel、完整视频和时间SLO为区别依据。原冻结实验仍暂停。


## 7. G12 信息判定收紧：直方图提供更强的数学依据（2026-10-27）

[完整推导及L75–L79一手对照](G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)、[来源等级账本](G12_SOURCE_LEDGER.json)。G11的scale-only/配对sign-only分类可增加**两个operand分别已知有符号码字直方图、未知same-coordinate alignment**的中间信息层：经典重排不等式按signed码升序给出所有排列中最紧 `[L,U]`。逐步观察部分真实pair可收紧，至全配对时变成普通内积信息读取。这个数学结论不意味着原HiF4在线激活码已免费可用、Wan 5120维自动出现有用符号裕度、或者有合法empty W1/W2 tile。先测信息可得性/价与局部认证，再进入Q2整个64组和下游有符号函数，匹配真实packed FP4 dense kernel。
