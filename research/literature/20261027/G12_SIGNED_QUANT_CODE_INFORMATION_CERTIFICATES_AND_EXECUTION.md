# G12 | 量化码“计算前粗信息”到有符号功能证书：重排界、可辨识性与编码计算

**日期：2026-10-27。** 继[G11六篇方法深读](G11_FIVE_PRIMARY_PAPERS_EQUATIONS_CODE_AND_COUNTERFACTUALS.md)、[SharQ](G11_SHARQ_FP4_VIDEO_SOURCE_AND_FUNCTION_AUDIT.md)以及[G10实时流式](REALTIME_VIDEO_ARCHITECTURES_AND_FFN_METHOD_AUDIT.md)后，将用户的假说推进成**可严格判定的信息边界**以及不同原理的强竞争方案。下述是数学证明、原始论文、官方源码审阅；全新CPU/GPU/模型、视频实验均为0。原模型仍是Wan2.2 I2V A14B 4step D-Q HiF4 FFN \`q→W1→tanh-GELU→HiF4-Q2→W2→e5-gate→residual\`，不把外部LLM或T2V的成绩冒称现有模型。

## 1. 新信息理论观察：只有block内直方图，依然可得**严格且最紧的配对无关内积区间**

假设输入X与权重W沿每个K维分段 \(G\) 均有**正的、各自在该段恒定的量化scale** \(s_{x,G},s_{w,j,G}\)，且量化码值 \(c_x,c_w\) 已给出，0码作为实数0。段内量化重建点积（bias单独加）是
\[
z_j=b_j+\sum_G s_{x,G}s_{w,j,G}\sum_{i\in G}c_{x,i}c_{w,ji}.
\]
在某个段G里，知道**码字的有符号直方图**，但不读同坐标pair相关性。将signed codes升序为 \(a_1\le\cdots\le a_B\) 和 \(b_1\le\cdots\le b_B\)，由**重排不等式**得到：
\[
\boxed{L_G=s_xs_w\sum_{k=1}^B a_k b_{B+1-k}\le z_G
\le U_G=s_xs_w\sum_{k=1}^B a_k b_k}.
\]
固定这两个multiset，在所有允许的未知坐标配对中，存在反序或同序匹配达到相应极值，因此它是**只凭该两种直方图能得到的最紧普适区间**。将各G界相加并加bias得到 \(L_j\le z_j\le U_j\)。若不同scale边界未对齐，必须将输入/权重尺度的K维分段细分为**scale同时恒定的交集**；若weight采用2D out×K scale，则每W1行依其真实分组拆段。动态block scale必须已先从当前q算出。任何非数值有效scale/特殊码字需独立处理。整个推导**只认证已量化实数点积**，实际native BF16 / FP32累加误差须外加舍入界。

**严格反例和可判定性**：w=(+1,−1)，x=(+1,−1)与x=(−1,+1)的两组signed直方图一致，真点积+2或−2；重排界确切是[-2,2]，任何单靠直方图的算法都无法给出单一正确符号。把范围看作**信息缺失的定量表示**，而非失败。只有 \(L_j>0\) 或 \(U_j<0\) 可对重建点积认证净符号。

### 分层获取信息并收紧区间

1. **离线weight统计**：weight FP4码/scale及每K block有符号16档直方图/有序码表可离线记录，访问仍需收费。在线q须完成本次normalize/AdaLN/Q1编码才能形成input histogram，若完全没有packed Q1码须另增加转换成本。
2. **零配对层**：先取weight+activation直方图，计算上述 \(L_G,U_G\)，不读同位置码字配对；其上下界严格，但可能经5120维多组相加仍跨0。
3. **部分配对层**：挑少数input/weight pair精读，对已核的贡献取实际值，从两个multiset里删除该对，剩余部分再用重排界；允许pair的集合减少，区间只收紧。什么时候精读应由**已有** cheap interval和真实成本决定；这是可选的adaptive proof，不是提前选定算法。
4. **完整配对/点积**：若区间始终无法收紧，则读全码字pair就恢复普通量化GEMM，直接同native Tensor Core对照。**不能把全码扫描并配对得到的sign certificate写作“无需计算点积”。**

**信息代价的算术约束**：每output-row、每new token需要约 \(H(D/B)\) 次直方图区间查询；全 \(M\) token大约 \(MHD/B\) 次。当前D=5120,H=13824,B=32、M=31668，对每FFN约70.04e9 group queries（数学规模，不是计时），每query至少需要signed直方图对应的排序/配对代价（例如按长度32直接计算约32次码值乘加，尽管可利用16档计数压缩），若全量遍历则可能重新达到与W1 dense点积同一量级的算术工作；不能把每组常数看成0。离线weight histogram也有显存价格：原FP4 packed权重 B=32块只用16 bytes，若每块保存16档的6bit计数上限需要12 bytes，约额外75% weight bytes（未经压缩的示意最坏表示），未必胜过直接读packed weights。因此**全W1逐通道打证书通常未必划算**；更可研究已有mask/低维簇、按时间或token共享的函数结构，使候选数显著减少。此处没有预定选择哪种压缩或通道掩码。

### 到GELU和W2还差哪些功能证明？

从 \(z_j\in[L,U]\) 可严格得 \(\gamma_j=\sup_{z\in[L,U]}|\operatorname{GELU}_{tanh}(z)|\)；对GELU负侧**不能只取两个区间端点**：非单调负尾的区间内部可能出现更大的负幅值。经原HiF4 Q2尺度依赖构造包含实际postQ2码值的合法集合 \(\mathcal H_G\) 才能界定
\[
\left|a^\top \Delta u\right|
=\left|\sum_{j\in \mathcal S_{\rm omit}} a^\top\operatorname{diag}(g)W_{2,:,j}\,h_j\right|
\le \sum_j\left|a^\top\operatorname{diag}(g)W_{2,:,j}\right|\sup_{\mathcal H_G}|h_j|.
\]
这里 \(a\) 可以是预先指定的下游功能方向/质量敏感探针；估计它同样有成本，若来自后验完整dense不能用作在线便宜选择。该界按绝对值相加保守地丢失signed cross-channel cancellation；联合相关性是继续研究的**独立信息源**。要跳W1/GELU并保持**原HiF4 group-scale**，可能需同时证明整64组Q2的scale/码作用已可合法确定；旧合成反例证明仅改单一输入会影响整组64个输出，不能由单neuron近零省W1。

## 2. 更直接的计算表达：不要先相信稀疏，再问码字能否重写内积

FP4 E2M1有限幅值 \(\mathcal V=\{0,.5,1,1.5,2,3,4,6\}\)。任意非零码可以精确写成**幅值分层的符号阶跃基**：
\[
c = \operatorname{sign}(c)
 \sum_{\ell=1}^{7} (\mathcal V_\ell-\mathcal V_{\ell-1})
 \mathbf1_{\{|c|\ge\mathcal V_\ell\}} .
\]
因此在尺度常量G上，量化点积可改写为7×7个加权**符号相关和/bitset popcount组合**。分层截断得到较便宜的有符号粗积及基于剩余层的误差上界；全49项则是原量化实数点积的**编码域精确等价代数**（忽略浮点累加次序），不是某种证明成立的省MAC算子。对齐非零mask、符号XOR/XNOR、popcount需读取双方的bit-plane。收益条件是bit-plane/lookup成本、缓存和tile形状足够好，必须比硬件原生FP4 GEMM更快，而不能拿FP16 GEMM当唯一比较器。

**为什么这不是新算法先例空白**：XNOR-Net（ECCV2016）以二值sign做XNOR-popcount近似乘法；BiQGEMM（SC2020）和LUT-GEMM（ICLR2024）已经通过quant codes有限模式预计算共享LUT；CodeGEMM（NeurIPS2025）用additive vector-code Psumbook。新可证伪问题不是“发现FP4的符号位”，而是**在已存块scale/符号元信息内，是否存在足够廉价并能预测gate×W2方向的函数规律**，或者以新编码代数真实超越FP4 Tensor Core。

## 3. 本轮原始引用链：按论文数学与已发布代码区分

### L75 BiQGEMM（SC2020）
[官方SC20原始记录](https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap601.html)，[原arXiv](https://arxiv.org/abs/2005.09904)，作者讨论binary coding compressed weights的**多权重并行读取、有限组合LUT**。本次核的是SC20官方论文摘要及LUT-GEMM的原引文（2024 ICLR）；完整BiQGEMM的公式和作者源未读，明确为**OFFICIAL_PRIMARY_ABSTRACT_AND_SUCCESSOR_CITATION**。最强重叠：离散码导致重复部分积可计算前缓存；它不说明冻结Wan的函数可整组跳过。

### L76 LUT-GEMM（ICLR2024）
[正式18页原PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf) §2.3、§3.1–3.3/Fig2–3（已通过PDF第3页截图核）；作者把BCQ weight表示 \(w\simeq\sum_{b=1}^q\alpha_b b_b\), \(b_b\in\{\pm1\}^D\)，按输入长度\(\mu\)预计算\(2^\mu\)种sign-dot并从bit packed index取值。其中心是**weight-only / A16**、LLM token GEMV，非W4A4；官方称OPT175B 3bit单GPUtoken generation相对OPTQ 2.1×。更关键的[真实源码 kernels.cu](https://github.com/naver-aics/lut-gemm/blob/main/lutGEMM/src/kernels.cu) blob e6950daa734c1b47331730145dbb9d32945f0914：\`matmul\` 在n或m=1调用\`kernel::nqmv\` LUT；大于1时直接\`matmul_useCublas\`，其内部\`getDequantiedWeight(true)\`构造解码weight后调用cuBLAS。和新CodeGEMM**两个代码仓库的same branch结构是独立源级证据**。 [mv_fp16.hpp](https://github.com/naver-aics/lut-gemm/blob/main/lutGEMM/src/cuda/kernels/mv_fp16.hpp) blob 77edd0a01ec6b815e7801451f5aa2f7183efc2b7 构造8位正负pattern的256项 LUT，逐8个K元素查表与scale accumulate。
- 新实质认识：文献说“量化码重构内积”不等于**实现了multi-token GEMM**。CodeGEMM和LUT-GEMM公开实现均只优化vector decode，不应对离线M≈31668或stream多latent大M外推。

### L77 XNOR-Net（ECCV2016）
[原arXiv](https://arxiv.org/abs/1603.05279)、[作者原PDF](https://pjreddie.com/static/papers/xnor.pdf)给出 \(w\approx \alpha\,\operatorname{sign}(w)\), \(x\approx \beta\,\operatorname{sign}(x)\)，内积使用XNOR加popcount并以正scale恢复粗值。这是**sign-only粗幅值预测**的早期充分先例，且要求两侧被特定方式近似；强对照是W1数千项发生signed cancellation时粗值符号缺乏保证。报告加速针对二值CNN而非原GPU FP4 Tensor Core；不能搬至NVFP4自动获得相同速度。本次以作者原摘要/原论文已公开方程为主，未审其官方原始CUDA实现。

### L78 Optimal Quantization for Matrix Multiplication（IEEE TIT2026 / arXiv2410.13780）
[IEEE正式论文与摘要](https://doi.org/10.1109/TIT.2025.3649596)、[原arXiv](https://arxiv.org/abs/2410.13780)：讨论的是**编码预算R下的矩阵积重建失真（MSE/rate–distortion）**，不是“量化签名可保证在线点积正负”。高斯矩阵下推导lower bound、通用nested-lattice编码和约0.906bit/entry相变；作者的一般保证涉及矩阵范数等整体对象。故它给出了**符号/幅值预算并不自动决定目标信息价值**的理论比较，特别当目标是margin或gate×W2的视频质量，MSE最优并不等于选算决策最优。本次只核**正式论文摘要与作者原arXiv摘要**，不冒称完整TIT定理证明已逐页审查。

### L79 MACKO（ICML2026）
[正式PMLR原文](https://proceedings.mlr.press/v306/macko26a.html)、[作者源码](https://github.com/vlejd/macko_spmv)；这是稀疏weight SpMV **存储格式/带宽改造**，不是从本次量化激活code推断FFN mask。作者展示50%非结构weight sparse在FP16 GEMV中相对dense约1.2–1.5×，Wanda Llama2-7B 1.5×；官方代码README说明主要测consumer GPUs 2080/3090/4090、服务卡H100仍需要专门优化。[kernels.cuh](https://github.com/vlejd/macko_spmv/blob/master/src/macko_spmv/cuda/kernels.cuh) blob 1be2c066684598fd5f0a341f6e9dc1dc3fa681fc：\`M_values\` half压缩流、\`M_deltas\` 4-bit坐标差、\`M_row_indices\`，warp shuffle prefix decode并执行FP16 SpMV。对FFNDS的现实判断是**只在single-vector和已存在的稀疏weight码**有此优势，不能据此声称H100 A14B video多token W4A4已解决。

## 4. 新计算流方法：三条竞争命题及对应最低成本下一证据

- **I-1 区间认证（STRICT）**：只读scale+两侧有符号直方图得到 \(z\in[L,U]\)，必要时付部分码字配对钱以收紧，证书要传播tanh-GELU、原HiF4 Q2 group、W2 gate与未来step。代价基线是native FP4 packed GEMM；若全output-row都评估bound很可能不划算，只有明确cheap/shared筛选机制才可能收益。
- **I-2 码字联合预测（STATISTICAL）**：已存code sign/scale与当前输入的信息是否预测一类**有符号下游功能方向**，超过norm/step/时间历史基础基线；控制数据、source、seed、expert，特别关注backlight/camera/background signed drift。需要本次preW1可用而不是posthoc精确标签泄漏。
- **I-3 编码域代数（REWRITE）**：把bitplanes、signed LUT、codebook partial sums视为替代计算表示（LUT-GEMM、BiQGEMM、CodeGEMM），不要求找到可删通道，但必须把multi-token fallback、Q1在线scale、metadata/byte read、codebook storage、额外GEMM与native packed lowbit kernel比较。

**条件性数学结论**：重排区间是**指定多重集合信息下的紧证书**，并不证明现有HiF4/FP4推理可低成本构造该多重集合，更不证明有用的interval sign coverage已经出现在某个Wan block。G12是**OPEN SCIENTIFIC ROUTE**，不是G11 Prox型proxy或SharQ残差路线的新强制选择。历史源/质量结果保持原样，0新原生实验。
