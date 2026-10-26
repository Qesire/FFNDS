# FFNDS · 机制先行文献目录（2026-10-26）

**研究阶段：先系统检索和审阅相关工作，再提出新方法。** 本目录按实际研究对象而非指标名分类；作者原文报告和FFNDS已复现实验分开。对应[33篇结构化索引](PRIMARY_LITERATURE_INDEX.json)、[机制交叉比较](MECHANISM_COMPETITION_MATRIX.md)。前期[2026-10-25五族目录](../../20261025/E0_FFN_INTRINSIC_PRIOR_ART_20261025.md)作为历史记录继续保留。

**阅读深度：**多数条目已核对正式会议/预印本官方摘要；Chipmunk作者技术正文、NVIDIA格式规范、TinyFusion相关原文段落做了更深入阅读。此轮未全面逐篇核验方法伪代码、训练步骤、完整消融或代码，实现核对将由独立文献Agent在同一索引追补；不可把摘要论断当Wan可迁移定律。

**第二轮实际深入记录：** [五篇直接竞争者的原论文/源码核查](FIVE_PAPER_METHOD_AUDIT.md) · [证据层级与准确性账本](METHOD_AUDIT_LEDGER.json)。最重要的纠错：DiffSparse四步实验为FLUX.1-schnell，其Wan2.1-1.3B视频实验为25步；Chipmunk的掩码决策在缩小token均值上运行W1并依赖历史缓存，公开实现含>=10步cache-mask分支。RT-Lynx和Dense2MoE的部分细节只能取自二手自动概述，尚未原文确认。

## 研究逻辑

Wan2.2 I2V四步原生对象是Q1→W1→GELU(tanh)→Q2→W2→gate/residual→40 block→flow→scheduler→完整视频。两来源paired研究、四anchor CPU函数实验、SM120合成packed FP4基元分别回答不同层级的问题。论文中的图像DiT、video attention、LLM、训练后转换及不同采样步数必须注明迁移条件。

**必须分开的五种命题**：激活弱贡献；函数可由更少坐标表示；跨步变化可复用；实际有利于视频终端功能；一个GPU GEMM的执行块可以真实减少。每一项都可能在其它项不成立时成立。SSE、signed cross和频率只是为了检验明确命题的仪表。

## FFN功能、可解释性与传统稀疏（7项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L01 · Transformer FFNs Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/) · EMNLP 2021 | W1 modes induce feature activation; W2 sums learned value vectors；信息：post-W1 for inference | conceptual interpretation；none | Video FFN value vectors may be features in superposition rather than one neuron/semantic role |
| [L02 · Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) · Transformer Circuits 2022 | Sparse features can superpose in non-neuron-aligned coordinates；信息：offline model analysis | representation analysis；none | Do not assume individual Wan GELU neurons equal independent scene features |
| [L03 · Sparsing Law: Towards LLMs with Greater Activation Sparsity](https://proceedings.mlr.press/v267/luo25i.html) · ICML 2025 | Sparsity robustness depends on task and training architecture；信息：post-activation and evaluation | sparse inference on adapted LLM；new model training | CETT/PPL quality and ReLU/SiLU trends do not transfer unmodified to Wan tanh-GELU |
| [L04 · FLAP: Fluctuation-Based Adaptive Structured Pruning](https://ojs.aaai.org/index.php/AAAI/article/view/28960) · AAAI 2024 | Recoverable feature fluctuation, structured selection and mean compensation；信息：offline calibration | static slim matrices plus bias；no retraining | Selected original nonlinear columns + fitted residual readout is close to known FLAP/regression principles |
| [L05 · Wanda: A Simple and Effective Pruning Approach](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html) · ICLR 2024 | Weight magnitude multiplied by observed input activation selects sparse weights；信息：offline calibration | weight sparse；no retraining | Weight-activation score is not guaranteed functional necessity or GPU speed in four-step I2V |
| [L06 · SliceGPT: Compress LLMs by Deleting Rows and Columns](https://proceedings.iclr.cc/paper_files/paper/2024/hash/316648eb8b4ffb6010f531b07848c300-Abstract-Conference.html) · ICLR 2024 | Computational invariance enables narrow dense networks under specific transformations；信息：offline transform | narrow dense GEMM；post-training transform | Nonlinear GELU and QDQ prevent arbitrary hidden basis rotations from being free |
| [L07 · Universal Properties of Activation Sparsity in Modern LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec3c79dc0c2b85532cfd1012a4aaa923-Abstract-Conference.html) · ICLR 2026 | Model scale and architecture affect FFN sparsity robustness；信息：postactivation evaluation | adaptive FFN sparse；analysis | Diffusion LLM with masked tokens is not a video DiT |

## 直接涉及DiT FFN/激活的最强方法（6项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L08 · RT-Lynx: Putting GEMM Sparsity in the Right Place for Diffusion Models](https://proceedings.mlr.press/v306/cong26a.html) · ICML 2026 | Activation N:M more robust than weight N:M; norm compensation and light LoRA；信息：after activation / before next GEMM | native N:M kernel；light LoRA used | Must check whether online 2:4 packing saves original W1, Q2 and BF16 costs on actual Wan |
| [L09 · Chipmunk: Dynamic Column-Sparse Deltas](https://arxiv.org/abs/2506.03275) · arXiv + author method 2025 | Sparse inter-step change of MLP and attention scaled value vectors; voxel reorder；信息：previous-step cached MLP activations | column sparse delta GEMM and cache；training free | Only two same-expert steps in four-step high-high/low-low; no cross-expert cache identity assumption |
| [L10 · Dense2MoE: Restructuring DiT to MoE](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html) · ICCV 2025 | Preserves total FFN capacity while selecting activated experts and blocks；信息：learned router | subset of FFN experts and blocks；multi-stage distillation | Needs model reorganization/training: not the same as static zero-training mask |
| [L11 · OBS-Diff: Accurate One-Shot Pruning for Diffusion](https://arxiv.org/abs/2510.06751) · arXiv 2025 | Timestep-aware Hessian motivated by accumulation; supports FFN neurons and N:M；信息：offline timestep calibrated | structured and N:M pruning；training free pruning | Hessian/time weighting is its mechanism-derived tool, not a universal semantic objective |
| [L12 · Activation Concentration: Column-Level Output Sparsity Across Diffusion Architectures](https://arxiv.org/abs/2606.00567) · arXiv 2026 | Element zero rate overestimates actually empty columns; layout and thresholds matter；信息：postactivation hardware profiles | column/systolic modeling；none | Systolic simulator is not a real Wan Tensor Core kernel; directly relevant to existing scatter |
| [L13 · TinyFusion: Diffusion Transformers Learned Shallow](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) · CVPR 2025 | Immediate low calibration loss fails to predict after-finetuning recovery; learns recoverability；信息：offline learnable masking | reduce full blocks；small finetuning | Motivates function/quality targets before SSE; does not demonstrate training-free FFN sparsity |

## 跨步冗余、动态缓存与少步（9项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L14 · TeaCache: Timestep Embedding Tells Its Time to Cache](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) · CVPR 2025 | Cheap timestep-modulated input difference predicts expensive output difference；信息：input + time before full forward | timestep output cache；training free | Wan only four steps, expert switch and CFG false require conditional validation |
| [L15 · AdaCache: Adaptive Caching for Video DiTs](https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html) · ICCV 2025 | Content and motion heterogeneity guide cache budgets；信息：movement and previous state | adaptive caching；training free | Static appearing background does not establish FFN dispensability or no flicker |
| [L16 · Accelerating DiTs with Token-wise Feature Caching (ToCa)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbe024e0517fe12ac3a8d388b19ff9fe-Abstract-Conference.html) · ICLR 2025 | Some tokens much more sensitive to caching; cache granularity varies by layer；信息：previous step + tokens | token cache；training free | Compare old light/background failures and true FFN-level rather than token-only control |
| [L17 · ProfilingDiT: Model Reveals What to Cache](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_Model_Reveals_What_to_Cache_Profiling-Based_Feature_Reuse_for_Video_ICCV_2025_paper.html) · ICCV 2025 | Foreground/background oriented block functions and step variation；信息：offline profile + previous step | block-feature selective cache；training free | FFNDS background lighting failures contradict simplistic background=unimportant rule |
| [L18 · FasterCache: Training-Free Video Diffusion Acceleration](https://proceedings.iclr.cc/paper_files/paper/2025/hash/518046d86bbc41a0707727c38301ad8e-Abstract-Conference.html) · ICLR 2025 | Naive adjacent feature reuse loses subtle changes; CFG branch redundancy exploited；信息：previous state / both CFG branches | feature cache and CFG cache；training free | Current distilled Wan has CFG disabled so only non-CFG parts transfer |
| [L19 · DiffSparse: Learned Token Sparsity for DiT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html) · ICLR 2026 | Learn layer-wise sparsity with end-to-end optimizer for few-step caches；信息：learned allocation | token and layer sparsity；two-stage training | Very close few-step competitor but still not equivalent to FFN neuron selection |
| [L20 · BWCache: Block-Wise Caching for Video DiT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Abstract-Conference.html) · ICLR 2026 | Observed U-shaped block variation prompts block cache trigger；信息：previous block state | block-wise cache；training free | Four-step expert schedule may not exhibit multi-step U-shape |
| [L21 · HyCa: Let Features Decide Their Own Solvers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html) · ICLR 2026 | Feature dimensions show heterogeneous dynamical evolution resembling mixture of ODEs；信息：step history | dimension-wise hybrid cache；training free as reported | Time interpolation limited by only four steps and expert changes |
| [L22 · MosaicDiff: Structural Pruning Reflecting Pretraining Dynamics](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_MosaicDiff_Training-free_Structural_Pruning_for_Diffusion_Model_Acceleration_Reflecting_Pretraining_ICCV_2025_paper.html) · ICCV 2025 | Pretraining phase learning dynamics informs stage-aware sparse models；信息：offline stage calibration | multiple stage-dependent sparse networks；training free pruning | Pretraining stages and denoising stages cannot be conflated without source review |

## Attention近邻的发现范式与边界（6项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L23 · Sparse VideoGen: Spatial–Temporal Sparse Video Attention](https://proceedings.mlr.press/v267/xi25c.html) · ICML 2025 | Spatial/temporal attention head motifs and online classification；信息：attention online profile | tensor layout plus sparse kernel；training free | Attention patterns not FFN masks; borrow discovery→cheap classifier→kernel process |
| [L24 · Sparse-vDiT: Sparse Attention for Video DiT](https://ojs.aaai.org/index.php/AAAI/article/view/37287) · AAAI 2026 | Attention head topology has partial input stability and layer specificity；信息：offline pattern selection | native attention kernel；training free | Do not equate attention pattern stability with GELU output stability |
| [L25 · SVOO: Attention Sparsity Is Input-Stable](https://proceedings.mlr.press/v306/luo26a.html) · ICML 2026 | Offline per-layer sparsity plus online QK co-clustering；信息：layer profile then online QK | sparse attention blocks；training free | QK interaction has no direct W1/GELU counterpart |
| [L26 · DFSAttn: Dynamic Fine-Grained Sparse Attention](https://proceedings.mlr.press/v306/hu26p.html) · ICML 2026 | Fine dynamic sparsity may defeat coarse block recall despite low attention density；信息：online query-key | fine-grained sparse blocks；training free | Conflicting layer stability conditions must be measured in Wan FFN separately |
| [L27 · SLA: Sparse-Linear Attention for DiT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/df77cca99e2785eafb5b2416d688ae24-Abstract-Conference.html) · ICLR 2026 | High-rank sparse critical weights and low-rank marginal attention differ；信息：attention weights | fused sparse+linear GPU kernel；some finetuning | Suggests heterogeneous solvers but not proven FFN basis decomposition |
| [L28 · QuantSparse: Video DiT Quantization and Attention Sparsification](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) · ICLR 2026 | Quantization+attention sparsity interact; multi-scale distillation and second-order residual；信息：calibrated attention state | quant+sparse attention；distillation | Attention quantization effects and HiF4 FFN Q2 dependencies are not interchangeable |

## 视频功能评测与低位硬件（3项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L29 · VBench: Comprehensive Video Generation Benchmark](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html) · CVPR 2024 | Separates subject/background consistency, flicker, motion and dynamic degree；信息：completed video | quality evaluation；n/a | Keep functional quality components separate from global proxy/SSE |
| [L30 · VMBench: Perception-Aligned Motion Generation Benchmark](https://openaccess.thecvf.com/content/ICCV2025/html/Ling_VMBench_A_Benchmark_for_Perception-Aligned_Video_Motion_Generation_ICCV_2025_paper.html) · ICCV 2025 | Motion quality must be aligned with perceptual dimensions；信息：completed video | perceptual motion metrics；n/a | Farneback proxy alone cannot support real motion fidelity acceptance |
| [L31 · NVIDIA NVFP4 Official Format and Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) · official technical spec 2026 | E2M1 with E4M3 block and FP32 global scales; packed layouts change real cost；信息：packed operand | FP4 Tensor Core GEMM；n/a | FFNDS successful synthetic CUTLASS primitive is not a Wan kernel or arithmetic reduction |

## 交叉检索用综述入口（2项）

| 论文与来源 | 研究发现和决定信息 | 原生动作/训练 | Wan四步需验证的不同假设 |
|---|---|---|---|
| [L32 · Efficient Video Diffusion Models: Advancements and Challenges](https://xie-lab-ml.github.io/efficient-video-diffusion-model-survey/) · arXiv survey/author website 2026 | Classifies distillation, attention, compression, and cache/trajectory approaches；信息：research index | survey only；n/a | Use only as citation discovery; cite original experiments for claims |
| [L33 · Efficient Diffusion Models: A Survey](https://arxiv.org/abs/2502.06805) · arXiv survey 2025 | Algorithm/systems/framework compression and acceleration taxonomy；信息：research index | survey only；n/a | Secondary map not independent evidence for every included result |

## 跨论文比较所得的三条核心研究问题

**同一步活跃性、跨步变化、按输入选择参数是不同机制。** RT-Lynx面向激活N:M稀疏及原生kernel；Chipmunk面向跨去噪step的MLP value增量和缓存；Dense2MoE通过转换/蒸馏保留总容量但少激活专家。它们不能合并成同一个FFN通道得分。相反，Activation Concentration提示元素稀疏在列/tile执行级可能仍完全稠密。对Wan应逐一核对信号何时可得、同/跨expert权重身份、是否能省W1与W2、内核开销及视频质量。

**表示能力、跨输入可预测、原生可执行是三种不同发现。** Geva和superposition启发研究FFN功能坐标；FLAP/Wanda/SliceGPT是要明确查重的传统近邻；现有4anchor/16-neuron的样本内oracle只能说明某些函数空间有能力，尚不能宣称其余12个neighbor获得有用压缩。若要主张共享非线性，应检验新增神经元信息与真实提前可获得的输入坐标。

**视频功能不能只靠短程误差定义。** TinyFusion提醒即时校准误差与后训练恢复不等价；OBS-Diff研究扩散时间误差传播；VBench、VMBench及ProfilingDiT揭示运动/背景/时间一致性的不同敏感性。既有Affine54可见异常是功能现象而非一个频段的SSE最优问题。

## 阅读顺序与归档规范

先深读L08–L13中最直接的DiT FFN论文，再L09与L14–L22对应四步蒸馏时序，随后L01–L07把基本数学与先例查清；系统效率与视频功能沿L23–L31并行。每次新阅读登记paper ID、正式版本、具体公式/章节、线上信息时点、实际训练、CUDA kernel、关键失败消融、本文与本repo的精确交集与差异。文献索引只是工作台，绝不据此自动启动原生实验或预定一个算法方向。
