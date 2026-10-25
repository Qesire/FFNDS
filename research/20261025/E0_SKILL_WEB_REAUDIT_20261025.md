# E0 — AutoResearcher Skill + 独立 Web 复审：修订上轮四路提议（2026-10-25）

**审查基准**：FFNDS 原提议 [E0_REVIEW_RESPONSE_AND_AGENT_GUIDANCE_20261025.md](E0_REVIEW_RESPONSE_AND_AGENT_GUIDANCE_20261025.md)，及保存分析 [E0_RADIAL_PAIRING.md](E0_RADIAL_PAIRING.md)、[E0_SHARED_REFERENCE_MOMENTS.md](E0_SHARED_REFERENCE_MOMENTS.md)、[E0_FOLLOWTHROUGH.md](E0_FOLLOWTHROUGH.md)。核对过的核心脚本为 \`tools/analyze_e0_radial_pairing.py\`、\`tools/analyze_e0_radial_template.py\`，原评议为 \`research/method_discussion/discoveries/E0/gpt6-independent-radial-pairing-judgment-review-20261025.md\`。

**实际使用的 Skill**（来自 FFNDS 已同步、固定的 AutoResearcher v2.4）：\`.agents/skills/mechanism-research/SKILL.md\`、\`.agents/skills/research-expansion/SKILL.md\`、\`.agents/skills/problem-reformulation/SKILL.md\`、\`.agents/skills/web-research-discovery/SKILL.md\` 与 \`core/WEB_RESEARCH_DISCOVERY.md\`。检索日期 2026-10-25。Web 范围包括：Wan 官方源码、LN 官方定义、表示各向异性、DiT 时序/特征缓存、少步扩散稀疏、量化 SmoothQuant、NVFP4 官方硬件格式。阅读层级与链接见后文，未宣称复现他人算法、下载 PDF 全文或执行任何新模型实验。

## 最重要的新判断

原四路方案有效区分了共享方向/条件变化、有限四臂、来源时序及 NetVOI，但过度把研究约束成“已有数学指标更精确化”。更强的解释对象是：

> 某种 FFN 信息由哪一个真实操作产生，什么时刻已可低成本观测，经过哪几条数值/残差/attention/跨步路径起作用，又在哪个原生计算单元具有可重排、共享、近似或完全省略的自由度？

这是一张可检验的**信息时间—执行边界—功能路径**地图；既允许几何、重复结构、量化元数据和时序重构，也允许新的非算子型科学发现。五个开放切面仅为分工，Agent 可以提出超出分工的新问题。

## I. 独立源码与数学核查：六个具体修订

### 1. \`delta_g\` 不是原生有限干预的写回差值

实际脚本逐组计算
\`delta_g = (q2_post[:,64g:64(g+1)] @ W2_panel_g.T) * gate\`。
这是保存 BF16 operand 解码到 FP64 后的 W2 分组方向。它不含完整 BF16 累加、不同组求和、bias、gate 舍入、残差写回。实验中真正的置零动作通过 native Q2 后控制组，二者为不同数学对象。

因此原 Agent B 的“保存的 FFN 写回更新”应改为**decoded FP64 group contribution**。可对其做精确恒等分解，但不能将它当作执行后 \`x'_native-x_native\`，也不能据此推出某组单独的因果边。

### 2. 真实 normalizer 的位置、算子和类型转换

Wan 官方 \`wan/modules/model.py\` 中 FFN 输入使用 \`norm2\`，下一个 block 的 self-attention 使用 \`norm1\`；FFN 后写回经过 residual identity 路径。项目自身的 LightX2V source pin 已在 \`E0_RESIDUAL_GEOMETRY.json\` 标记 \`transformer_infer.py\` 和 layer_norm_weight 实现。Agent 应优先复建**项目当前 LightX2V 版本**的真实读写顺序，官方 Wan 作为交叉对照，不能以通用 pre-norm 模板替代。

对无仿射 ideal epsilon-LN，设 \`P=I-11^T/C\`, \`u=Px\`, \`s=u^Tu\`，则
\`J_LN(x)d = rstd(x) [Pd - u (u^T d)/(s+C eps)]\`，
其中 \`rstd=(s/C+eps)^(-1/2)\`。若存在 norm affine weight，则在左侧乘以对应 diagonal gamma；AdaLN shift/scale 位于真实 norm 之后的下一运算，属于后续链。原生 fp32 norm 和 cast-to-BF16 是有离散数值边界的实现，导数对象为条件连续代理。

### 3. 固定 J 的分解虽精确，却尚未等于实际下游价值

对同一真实 x_i 和已构造 delta_i，令 \`mu=mean_i delta_i\`, \`eta_i=delta_i-mu\`, \`J_i=J_LN(x_i)\`：
\`||J_i delta_i||²=||J_i mu||²+||J_i eta_i||²+2<J_i mu,J_i eta_i>\`。
恒等式精确。需要同时注明：每个 \`J_i\` 随 token 不同；分解是**同一固定 x_i 处的一阶局部响应**，在下一 block norm1 的输入不是唯一消费边，identity bypass、attention、AdaLN 与 downstream nonlinearities 仍会传播该状态。高比例径向响应不等于网络中该信息消失。

### 4. 32 行位置均值不等于整段视频的全局分量

脚本按 source/step/block/group 取 \`uniform_rows\` 的 **32 个位置**，在这些行上做 \`state.mean(axis=0)\` 和 \`delta.mean(axis=0)\`；真实 patch token 网格为 \`21×29×52\`。已有“共享方向”首先是**该固定采样面板的 5120 通道位置均值**；跨六来源相同 seed 和 row-coordinate 策略的留出结果也不是对全部 token、跨 seed 或新视频的可迁移性证明。应检验 row 采样与共同坐标的影响，并核对实际可用的其他保存行，而非直接把它称为视频级共享 bias。

### 5. Star-null 不是两组动作可加性的零模型

四臂 D/A/B/AB 定义 \`a=A-D\`, \`b=B-D\`, \`c=AB-A-B+D\`。\`c=0\` 等价于当前边界两组有限动作可加。令 \`S1=d(D,A)+d(B,AB)\`, \`S2=d(D,B)+d(A,AB)\`, \`S3=d(D,AB)+d(A,B)\`（各 d 为平方 L2 距离），有
\`S1=||a||²+||a+c||²\`,
\`S2=||b||²+||b+c||²\`,
\`S3=||a+b+c||²+||a-b||²\`。
取 D=(0,0), A=(1,0), B=(0,1), AB=(1,1)，\`c=0\` 但 \`(S1,S2,S3)=(2,2,4)\`。三配对和相等是 \`d_ij=v_i+v_j, v_i>=0\` 的特定 shared-reference uncorrelated-error 零模型必要条件，不是动作可加的必要条件。现有解释需要将 star-null 残差、有限交互 c 与响应 Gram 三者并列，避免由一项直接归因 BF16 或 smooth propagation。

### 6. 原引导的离线 CPU 权限语言过宽

当前 E0 允许源码阅读、数学证明、文献研究和对已有保存资产的允许范围内离线数值分析；“禁止新增 CPU 实验”若覆盖此种读取/复算，会阻塞被要求的判别任务。现改为：既有授权范围内可做离线复算；全新模型执行、GPU/生成/训练和扩大的原生实验范围需用户另行指定；不自动扩大研究状态。

## II. Web 深搜使竞争机制怎样改变

以下全部是**外部原始来源的机制对比**，不是已在 Wan 验证的结论。

| 原始来源与阅读层级 | 已知机制 | 对 FFNDS 的判别提示 |
| --- | --- | --- |
| [Wan2.2 官方模型源码](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py)（WanAttentionBlock 定义/forward） | norm1 self-attn；norm2 FFN；AdaLN gate 后的 residual writeback | 任何归一化“零方向”论证都要定位真实消费者和 bypass，结合 LightX2V 固定源复核 |
| [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)（算子定义） | 标准化 / eps / affine | ideal analytic Jacobian 与实际 cast-boundary 不同 |
| [SVD-Cache, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)（官方会议方法摘要、arXiv 摘要） | 主子空间预测、残余子空间复用 | 单一跨位置均值不是 SVD 主通道子空间；但“公共和可变部分不同可预测性”已有近邻 |
| [HyCa, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html)（官方摘要） | 按特征维度采用不同缓存动态模型 | 以“不同分量用不同预测”本身不能构成新意；需真实 FFN 信息/执行机制差异 |
| [RFC, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4019681ef05e8e347c3067d7f973a364-Abstract-Conference.html)（官方摘要） | 用模块输入—输出关系推断增量和是否重算 | pre-FFN 的廉价信息与输出重要性关系存在强先例；须对观测成本/量化/真实动作区分 |
| [DiffSparse, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html)（官方摘要） | 少步 DiT learned token sparsity，含 Wan2.1 | 少步限制不等于稀疏消失；但该工作依赖优化/训练，FFNDS 的轻量后训练约束是差异维度之一 |
| [Token-wise Feature Caching, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbe024e0517fe12ac3a8d388b19ff9fe-Abstract-Conference.html)（官方摘要） | token 的缓存敏感性差别大 | 输出均方误差/全局 cosine 会丢失少量重要 token 的功能敏感性 |
| [SmoothQuant 原论文](https://arxiv.org/html/2211.10438v6)（§4 具体尺度公式/融合） | 静态可融合缩放移动量化难度 | “有尺度”不是廉价在线预测器的证明；静态平滑不同于已生成的动态 Q2 metadata |
| [NVIDIA NVFP4 官方格式](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)（格式、缩放、支持设备） | 16 元素 micro-block FP8 scale + tensor FP32 scale，原生支持 SM10.x | MXFP4/NVFP4 与当前 A100 上 HiF4 QDQ→BF16 GEMM 为不同信息产生/消费边界，必须分开论证可观测性和真实 speedup |

**补充强反例**：大型公共方向可能是不可删的功能偏置；极少 token 可能决定最终感知质量；在四步双专家模型里长轨迹缓存的历史依赖未必可得。此前 FFNDS 已检索各向异性 / rogue dimension / massive activation 类论文，其数学线索仍有效；此处新增更接近“已有输入信息预测未来输出、特征子空间动态、少步稀疏”的实际竞争路径。

### 检索轨迹

使用公开 Web Search，先从 conceptual families（LayerNorm Jacobian、DiT common component、BF16/量化边界）、implementation families（Wan official source、NVFP4 spec）、failure/competitors（few-step token caching、input-output relation、conditional prediction）搜集；进一步按检索结果追到 ICLR 2026 的 RFC、HyCa、DiffSparse，CVPR 2026 SVD-Cache 以及 NVFP4 官方硬件格式和 SmoothQuant §4。已逐一打开 Wan/ICLR/NVIDIA/SmoothQuant 官方 HTML；CVPR 2026 HTML 的深入打开遭 403，使用搜索返回的官方 CVPR 条目和 arXiv 摘要核对其方法定位，并未声称阅读全文。并未完成所有论文的全文算法与官方训练代码再现，后续 Agent 应继续按 WARD 深读与引用追踪。

## III. 修订后的探索组织

围绕**真实信息图**而非四种既定结论建立五条可选并行线：

1. **A 四臂因果边界**：区分 c、Gram、star-null；审计现有边界能识别哪一种连续传播与哪一种 BF16 数值机制。
2. **B 固定真实 norm Jacobian**：明确 decoded \`delta_g\`、同位置固定 J 的三项分解、残差旁路及 norm 实现/舍入边界；对比之前改变 x 的 centered probe。
3. **C 共享来源/时空机制**：32 行采样公共分量、seed/position 协议、step/expert/gate/AdaLN 的竞争来源与不同可检验预测。
4. **D 量化信息的出生时刻**：追踪 HiF4 Q1/Q2 codes/scales/smoothing 的真正读取成本，与 MXFP4/NVFP4 原生 scales 对照；寻找前置可获的统计信息与可物理省略的 GEMM 单元。
5. **E 跨领域机制/计算经济与新颖性**：对照 RFC/HyCa/SVD-Cache/DiffSparse 和硬件受限现实，提出至少一个**新的**数学表示或竞争解释，再从 NetVOI 与可执行计算判断是否值得未来独立验证。

每条线可在各自独立记录里先提出大胆的探索 seed，后精确定位 conditional identity / observed data / source analogy / prospective experiment。不要以“已有数据仅六源”替代科学推导，不以实现近在眼前而要求立即选算子；同时也不要把完全的效率证明设成提出新机制的先决条件。

**即时可完成的最小工作**：B 的精确符号分解与 operand 字段映射；A 的 c、Gram、S1/S2/S3 正交零模型对照；D 的 quantizer metadata 时序/读取账本。它们分别改变几何判断、非加性判断和省算可行性判断。C、E 并行搜寻更强的替代表述。

## IV. 修订结果与权限

原提议的研究目标和 19.794/20.542、8.839/3.590、0.99645 等保存数字保持；**修正的是数据所指称的对象、归因能力、探索分工和文献最近邻**。不修改实验结果、复现实验脚本或 AutoResearcher upstream；不增发 model/GPU/视频实验，不把本次审查算成外部 Agent 独立复现，也不提升 PEER_SUPPORTED。
