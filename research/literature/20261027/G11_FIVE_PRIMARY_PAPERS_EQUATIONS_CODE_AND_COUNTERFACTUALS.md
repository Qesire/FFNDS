# G11 · 五篇原始论文的方程、真实执行路径与竞争机制复审

2026-10-27 · **PRIMARY_PAPER / AUTHOR_SOURCE_READ / ALG / OPEN**。研究按 AutoResearcher WARD v2.4 进行：从已存在的FP4符号和粗幅值出发，核查是否能预测Wan FFN的*有符号必要计算*，以及是否真的省下 W1/GELU/HiF4 Q2/W2。只读论文和公开源码，**没有新CPU/GPU、模型、训练或视频执行**。原研究 [G11方法](G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md)、[原计算图](../../../docs/98_端到端计算图与证据注解_20261025.md)、[source ledger](G11_QUANTIZATION_SOURCE_LEDGER.json)一并保留。

## I. 五篇研究处于三个不同计算层次

| 来源 | 时间上已知的信息 | 产生的选择或运算 | 主要价格/机制边界 |
|---|---|---|---|
| **L69 Prox** | 当前SwiGLU输入及另存INT4代理权重 | 两次低位上/门proxy选择通道，再计算原三投影选定通道 | 额外两次proxy GEMV；依赖中间状态排序；原SwiGLU≠Wan GELU/HiF4 |
| **L70 Spark** | 共同训练所得低维K1路由输入 | K1前r维先判TopK；K2及V只算k列 | 修改了网络与训练；近Gaussian阈值假设；不是原checkpoint天然的sign mask |
| **L71 SpQt** | 当前输入稀疏threshold及离线重排过的Q4_K | zigzag layout + sparse-index collection + balanced GEMV | Apple Metal单token；prefill多token退回dense |
| **L72 SQ-DM** | 经激活函数改造产生的ReLU严格零 + 时间step | 稠密/稀疏PE分工和每步通道类型更新 | 微调EDM U-Net并PTQ、SToNNE自定义模拟硬件；非Wan冻结FFN |
| **L73 CodeGEMM** | weight additive vector-code indices，当前输入 | centroid×输入block预计算Psumbook，再按code lookup/accumulate | 原公开实现仅B×T=1走Psum GEMV；B×T>1完整解量化+torch matmul |

### L69 Prox · 计算前代理的信息价值已被明确研究

**一手**：[arXiv:2607.27591v1](https://arxiv.org/pdf/2607.27591) §3 Fig3、§4.1–4.3 Eq/成本、§5 Table3 A1–A3及Appendix A。原SwiGLU为
\[
u=W_{\rm up}x,\quad h=W_{\rm gate}x,\quad s=u\odot\operatorname{SiLU}(h),\quad y=W_{\rm down}s.
\]
**Stage1** 先以校准固定阈值仅保留幅值大的输入x（输入稀疏度s1），使用INT4低位proxy的Wup/Wgate获取近似u、h，形成中间 \(\tilde s=\tilde u\odot\mathrm{SiLU}(\tilde h)\) 并选出通道mask。**Stage2** 用原precision对被选中的up、gate、down通道做真实投影，代理值只做mask决策。准确说，Stage2仅“被选坐标重算原精度”，其余贡献仍被省略。

作者把原三投影每个看成1单位，两个proxy投影各有相对成本α，明确得到
\[
C_{\rm Prox}=2\alpha(1-s_1)+3(1-s_2),\quad
e=s_2-\tfrac{2\alpha(1-s_1)}3 .
\]
作者用多GPU上INT4/FP16 GEMV latency测得的范围统一取α≈1/3（而不是仅按bit数取1/4），按目标effective sparsity e分配两阶段预算。Fig3同一层70%稀疏的精确/代理top重叠约82.77%，512token平均82.04%。在NVIDIA A6000单batch decode，60–70%原作者报告1.51–1.99×端到端提速（Fig5）；量化模型AWQ/FP8/W4A16/W8A8的兼容实验是PPL，作者明确把**低位×稀疏联合物理kernel留为后续工作**。同预算A1换FP16 proxy反而更差；A3直接把proxy激活用于output使Qwen3-8B 70%时score由68.6变44.3，支持“代理用来决策、最终值仍按原函数”。

**迁移条件**：当前Wan一条W1/GELU/W2（不是SwiGLU三个GEMV）、Q2还有64→8→4共享依赖、M≈31668不是B=1 decode。仅复制“INT4 proxy→TopK”属于强先例同构。更深问题是原已存FP4编码是否携带无需再做两次proxy的高价值**有符号gate×W2净函数/合法空tile**信息。

### L70 Spark · 预测器是通过重新训练产生的

**一手**：[NeurIPS2025正式论文36页PDF](https://papers.nips.cc/paper_files/paper/2025/file/220d26c7fb478c2a397bd3fef0fe1d6f-Paper-Conference.pdf)，Eq2–4、Eq9–12、§4、附录C/D。

作者重构FFN为分拆输入q的两段与三组共同训练参数：
\[
f(q)=V\left[\operatorname{GELU}(\operatorname{TopK}(K_1^\top q_{[:r]}))\odot(K_2^\top q_{[r:]})\right].
\]
\(K_1\in\mathbb R^{r\times d_{ff}}\)先在较少输入坐标上预测位置；K2及V只计算selected k。作者Eq4 FLOPs/token：
\[
C=2rd_{ff}+2(d-r)k+2dk
=2r(d_{ff}-k)+4dk.
\]
r≈d/2、k≪dff时约为 \(d\,d_{ff}+3dk\)，对应作者同参数预算原FFN \(4d\,d_{ff}\) 的近4倍计算差。

**statistical top-k** 不是精确排序：先根据已算的predictor向量a估 \(\theta(a,k)=\operatorname{mean}(a)+\operatorname{std}(a)\Phi^{-1}(1-k/d_{ff})\)，再用 \(\max(a-\theta,0)\) soft threshold。Theorem3.1假设a元素i.i.d Gaussian，证明激活个数的概率偏差界，**不保证**Wan Q1码、混合残差、量化相关分布满足它。论文Gemma-2 2B d=2304，原门控FFN宽9216，Spark保持参数数目扩至13824，r=1024、k=1106即约8%非零；按原Gemma-2 recipe训练，LLM decode最高CPU1.79×/GPU1.40×，总体2.5× FLOPs reduction。这不是冻结权重算法；两个训练的函数可能发生FFN/Attention功能重配。

**迁移条件**：允许轻量后训练时可重新分配部分输入维度作为计算前预测特征；但从当前Frozen Wan weight code signs推出相同结构的存在性，仍是全新的数学和证据问题。若考虑原Wan，先比较其真正postQ1的signed方向，不要用Spark的Gaussian top-k去假定全局SSE代表功能。

### L71 SpQt · “量化布局能省运算”要看输入token数量

**一手**：[arXiv:2511.04477v1](https://arxiv.org/pdf/2511.04477) §3–6、Fig3–4和Tables2–3。SpQt并不是使用weight sign预测原FFN输出。它采纳TEAL校准得到的本次hidden阈值，通过zigzag把Q4_K类型的量化权重从行组改为便于按**输入通道列**活跃状态读取的pack。在线一次索引收集得到所有非稀疏输入坐标，再用balanced Metal SIMDGROUP GEMV避免原布局的branch divergence与stragglers。这三项（layout，sparsity，load-balance）作者分别设置了strong ablation：Q4K dense、naive original-layout sparse、new layout dense、new sparse无balance、新full kernel。

**重点源事实**：作者§5部署明确说prefill（多token）使用dense，而autoregressive single-token decode才用sparse GEMV。原因包括多token激活支持广散和质量退化；作者dense分支也尚无成熟独立multi-token tiled GEMM实现。因此A19 Pro等Apple设备50% sparsity下最高1.55×**相对于Q4_K dense**的decode提升，不可以据此称Wan当前整段M≈31668上FP4稀疏GEMM也有提速。其Table3中Llama2-7B PPL：dense Q4K 5.28、50% sparsity 5.65、65% 7.39，效果具备稀疏阈值边界。

**机制作用**：这是对我们既有M16×K64后验100%非空观测的强竞争，说明应该同时考虑**token分组+物理布局/packed GEMM**，而不是仅追逐单位置proxy mask重合度。短chunk/单向量实时流可独立形成不同最佳方案。

### L72 SQ-DM · 量化带来的“天然稀疏”其实来自激活函数改造

**一手**：[arXiv:2501.15448原始全文HTML](https://ar5iv.labs.arxiv.org/html/2501.15448) §III/IV及Tables I/II，正式DAC2025。其PDF体积超Web PDF fetch限制，**本轮没有取得PDF截图**，已直接阅读作者论文原始HTML全文。

其对象是**EDM/EDM2 U-Net**，>90%计算在Conv+SiLU，不是DiT FFN。作者先用层敏感度将部分block MXINT8、其他INT4+FP8 scales；随后将原SiLU替成ReLU并对原全精度模型微调（作者<原pretrain 10%时间）后PTQ。ReLU将负值变成**严格zero**且使激活使用UINT4全部非负量级，而原SiLU输出小负尾占用signed INT4码。激活平均zero率约从10%变成65%，部分层85%。

§IV中作者的自定义28nm STONNE模拟硬件有一个dense processing engine和一个sparse engine，各128乘法器。按channel-last映射，把输出zero数量用于判定下一层channel稀疏/稠密；30%是**通道稀疏分类阈值**，而不是直接删除30%幅值；每个diffusion step更新一次频道类别。质量是图像FID：EDM1 CIFAR10 FP16=1.85、MP-only=2.87、MP+ReLU=2.12；总6.91×加速可以拆成量化3.78×与时步稀疏1.83×，能耗节约51.5%是相对作者定义的dense硬件。不是NVIDIA真实GPU、没有I2V主体/运动/背景质检，也没有冻结tanh-GELU上的跳算证据。

**机制作用**：量化适合的符号域可以通过重新训练重构函数，使计算变得严格可省。这是与“当前已量化权重蕴藏天然可跳结构”并行的训练后变换研究，不是同一个问题。

### L73 CodeGEMM · 利用码本结构替换重复内积，公开代码仅单vector走快路

**一手**：[NeurIPS2025原论文/arXiv:2512.17970v1](https://arxiv.org/pdf/2512.17970) §2–4 Eq2–3 Fig3；[官方源码](https://github.com/naver-aics/codegemm)直接读取Python层、CUDA GEMV和PyTorch glue，具体blob版本锁在同目录[source ledger](G11_QUANTIZATION_SOURCE_LEDGER.json)。

对于additive codebook模型每一段长度v的weight是m个codebook centroid向量之和，用code id保存；给当前activation的各v向量段 \(x^{(j)}\)，作者Eq2先计算所有码本候选 \(P_{c,j}=\langle C_c,x^{(j)}\rangle\)，再按每个weight当前code id取Psum与group scale相乘汇总。这是已量化函数的一种**代数重组**（数值求和顺序仍可能不同），不是预测哪个W1/GELU通道无用。

原论文Eq3在output size \(M\gg2^b\) 等前提下，时间复杂度 \(O(MNK\cdot m(2^b/M+1/v))\)，其中build Psumbook也付费；论文报告8B/70B的2bit下对比AQLM分别1.83×/8.93×，具体核度量重点是**GEMV M=1,N=28672,K=8192**。它不是通用FP4 code+scale可直接用的核。

**强于文字摘要的源码事实**：[CodeGEMMLinear.forward](https://github.com/naver-aics/codegemm/blob/main/codegemm/inference/codegemm_linear.py)判断x.numel()/in_features==1时调用codegemm_gemv；否则调用self._gemm，执行codegemm_dequant重构完整weight再torch.matmul。对应[gemv.cu](https://github.com/naver-aics/codegemm/blob/main/codegemm/inference/custom_kernel/gemv.cu)硬约束B=T=1，仅支持(num_codebooks,vec_len)=(1,4)与(2,8)；[codegemm.cu](https://github.com/naver-aics/codegemm/blob/main/codegemm/inference/custom_kernel/codegemm.cu)使用K_TILE_SIZE=32、M_TILE_SIZE=2048和shared LUT，把packed code映射Psum再乘alpha、half2 atomicAdd。官方README的主配置是两组8bit codebook、每组8元素矢量及scale_group128。

**重大迁移修正**：已发表的算法支持codebook-centric算子思路，**其release的真实多token分支却没有Psumbook kernel**。Wan full-M、实时多帧chunk若直接复用该公开实现，都不会走作者的优势路径；单向量/小batch的真实系统应分开评估。此结果应作为 **source-level FACT** 写入计算图，而非仅称“CodeGEMM已经实现大M查表”。

## 机制争论与最小可证伪问题

**C1 前置信息**：weight scale/sign存在≠能够廉价获取aligned-code-product margin。Prox额外INT4 proxy付费、Spark主动训练部分weight计算预测器；必须把获取信息的成本与低位Tensor Core dense投影正面比较。

**C2 真实功能**：TopK交集/激活zero率≠gate×W2多通道净输出。原Wan实际D-Q Q2共享scale令某个组里未选hidden也可能影响入选hidden的量化结果。尽可能在同一q、同expert、同group定义 signed transported influence，而非全局SSE一种证据。

**C3 多token物理成本**：SpQt作者多token退dense + CodeGEMM代码多token退dense共同说明：大M与实时小M对应不同算法/内核空间。原FFNDS M16/K64满tile诊断依然成立；有价值的新方向是能提前预测的**函数组、规律化tile、码本代数分解**，这些尚未被证明对当前模型成立。

**C4 训练引入的结构**：Spark和SQ-DM说明可以通过共同训练/修改激活让计算可稀疏；它们不能直接作为Wan冻结函数天然存在该结构的证据。小规模后训练是可研究的独立成本轴。

## 后续源与计算图挂接

新增G11方法子节点 Q19–Q29（见Graph Agent Bindings），保持原Q01–Q18统一信息图；原论文5条索引 L69–L73 阅读级别升级。检查未来任何idea必须附 [paper-math vs author-actual-source vs Wan-original-SRC vs native-OBS vs OPEN] 证据身份、决策信息的合法时点、实际MAC/访存/layout与完整视频表现。没有获得新GPU/model permission；现有E0实验继续暂停。
