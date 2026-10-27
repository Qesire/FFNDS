# FFNDS · 五种计算方法操作 × I/F/C 科学入口 × 当前候选结构

**2026-10-27 · 统一方法分析法 / METHOD_SNAPSHOT_TAXONOMY / STATIC_ONLY。** 依据用户提出的五操作分类、FFNDS AutoResearcher Skills、[G14 全102项论文到真实计算边的映射](../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)、[现行I/F/C证据组合](PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)及[独立A/B/C评议](COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)。这是方法分析与下一Agent指引，不是新实验、独立同行认可或ACTIVE执行注册。

> **核心原则：先说明算法实际改变哪一项计算，再研究它为什么有效、在线是否省掉物理工作、行为是否得到保留。稀疏（sparsity）是多种方法可能产生的结构或执行结果，不是独立、完备的机制分类。**

**方法构造已经更新至v2**：[整合M-A/M-B/M-C现行研究方案](COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)直接运用[G14反向审计](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md)与[先验约束机器关系](../agent_protocol/METHOD_OPERATION_CROSSWALK.json)：A0四位置native恢复已完成，M-B v2已证明末层中心化head的64读出＋1中心方差充分坐标（ALG/SRC），下一项为B0原生u/head/flow neutral状态捕获准备、B1有限敏感性。M-C保留输入充分性/输出秩/实际大M LUT代价OPEN。本分类是可组合**方法语法**，不能替代每条实际科学方法的状态与证据。

## 1. 三层研究体系，避免代号冲突

- **科学问题层（I/F/C）**：I信息在何时可得、何种编码足够；F哪个正常计算和下游响应支撑时空生成、是否能够迁移；C如何真正减少延迟、访存、GEMM、存储与部署成本。这是WHY/WHAT，不与操作类别竞争。
- **方法动作层（O-A/O-B/O-C/O-D/O-E）**：分别是计算选择、函数表示、历史复用、计算代数与学习验证目标。这是HOW，一个论文或方案可同时具备多个操作，各操作需注明是否真的部署执行。
- **候选方案/执行层**：具体科学对象如UB1、Affine54、Whole94、A0窄学生、原候选B敏感坐标及原候选C码本，及其真实source/step/block、QDQ/FP4/kernels和数据。此层要有真正的输入输出/作用位置、精度、预算和指标。

**名称约定**：原三候选保留名称 **M-A（原提案A，紧凑学生）、M-B（原提案B，下游敏感坐标）、M-C（原提案C，编码直接求值）**；本分类的操作总使用 **O-A 到 O-E**。所以“下一位Agent从B开始”精确指 **M-B**，它的主要操作是**O-B＋O-E**，潜在物理实现可有O-D，而不是泛泛研究所有O-B论文。

## 2. 五操作的数学定义、边界和成果形式

| 操作 | 科学/数学动作 | 实际判别标准 | 容易混淆的邻近动作 |
| --- | --- | --- | --- |
| **O-A 选择保留的计算量** | 选择集合或预算：S(x,t,H)属于token、组、层、专家、列及其联合计算子图 | 哪一个算子/乘法/访问被真正省去？选择信号首次可得何时？静态/step-static/input-dynamic分别记录 | 选组但原完整W1已经执行，只是改变W2输入，不等于省掉W1；从校准来源挑数据属于O-E，不是推理算子O-A |
| **O-B 改变函数表示** | 以新基、新分解/结构/隐变量或少维参数近似或重写 f_theta → f_phi | 证明替代函数来自前置合法输入，区分实数恒等与学习近似；记录容量、表示空间、重新量化与训练 | 保持同一W2矩阵、只用特殊kernel执行，是O-D；但低秩因子化改变可部署函数/参数形态往往O-B＋O-D |
| **O-C 复用历史计算** | 借助执行时真正已经存在的 H_<t 来重用或预测当前f(x_t) | 命中/有效历史、cache驻留字节、更新/验证成本和误差随step累积；history必须在目标计算之前可得 | 同一步重新组合恒定预计算表属于O-D；跨step缓存引发选择跳算时通常再加O-A |
| **O-D 改变计算代数或执行协议** | 将选定的函数操作用低位码、预表、融合、稀疏矩阵核、混合精度/布局等求值 | 算子的数值等价类别、packed/fake-QDQ、访存/算术/format转换与真实GPU端到端成本 | O-D可精确、近似或随机数值实现；量化后f的逼近误差需单独评价，不表示数学上完全相同 |
| **O-E 改变学习与验证目标** | 训练、校准、数据/源选择、蒸馏、扰动响应监督、视频质量读出及因果辨识 | 改变哪个estimand/训练优化问题、独立采样单位、监督能否合法获得、验证与推广范围 | O-E本身一般不会自动减少推理乘法；“选训练source”与O-A“选择在线计算子图”有不同时间和对象 |

形式上，一个方法是具有版本的动作组合，而非唯一类：
\[
\mathcal M=\mathrm{Compose}(\mathrm{O\!-\!A},\mathrm{O\!-\!B},\mathrm{O\!-\!C},\mathrm{O\!-\!D};\,\mathrm{O\!-\!E}).
\]
未使用的动作标NONE，不能为了凑全五类把论文混合归档。O-E包含E_train（优化/校准）与E_evaluate（独立验证/因果）两种相似但不相同的角色。

**最小操作指纹 METHOD_OPERATION_FINGERPRINT**：
(1) native producer→operator→consumer真实计算边；
(2) 主操作及辅助操作（分别注明OFFLINE/ONLINE/OPEN）；
(3) 最初可用信息、输入/状态和跨step历史；
(4) 语义等价（exact/approx/learned）、QDQ、dtype与shape；
(5) 减少的实际原运算及新增observer、router、packing、cache、LUT、kernel、训练与读表成本；
(6) 同模型/算子/硬件/适配预算的强baseline；
(7) 已完成SRC/ALG/LIT/NATIVE/VIDEO/SYS与OPEN的证据；
(8) 分辨方法机制与最强竞争解释的下一证据。

只有声明了(1)–(5)才能主张真实计算结构改变。方法不必为了理论发现先做完整视频；若声明完整视频保真和整端速度，则需要来源级真实自由轨迹和系统收据。

## 3. G14原文先例：同名稀疏方法可能是不同操作组合

**本表按真正的论文动作打标签**，不是给每篇论文一个唯一类别。括号内LID/CID是当前G14/补充论文索引；“外部补充”未伪造主索引的新LID；2026-10-28经原方法阅读新增L90–L92，主索引现为92项。

| 论文 / G14身份 | 主要操作 | 同时发生的第二操作与具体事实 |
| --- | --- | --- |
| [Wanda](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html) L05 | O-A 离线选择权重 | O-E：用激活统计构造删权决策，不是推理时自适应激活路由 |
| [Dynamic-in-Few-Step](https://arxiv.org/abs/2607.06631) L34 | O-A 少步分配及结构选择 | O-B结构蒸馏＋O-E恢复目标；有step条件，但不应和逐token在线路由混同 |
| [Prox](https://arxiv.org/abs/2607.27591) L69 | O-A 由廉价代理选择需精算的FFN部分 | O-D INT4代理执行；代理有在线成本，仍要算被选真实W1/GELU/W2 |
| [DiffSparse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5e66b9f710a8e38351da690f3806cba9-Abstract-Conference.html) L19 | O-A 层/Token计算分配 | O-C token缓存复用＋O-E可微分配训练，属于复合方法 |
| [Dense2MoE](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html) L10 | O-B 变为专家函数表示 | O-A token/expert及部分block选择＋O-E蒸馏；付出真实路由/dispatch |
| [Fast Feedforward Networks (FFF)](https://arxiv.org/abs/2308.14711) 外部补充 | O-B 条件树状FFN表示 | O-A 推理只走相应叶子/路径；改变结构同时改变计算 |
| [SliceGPT](https://proceedings.iclr.cc/paper_files/paper/2024/hash/316648eb8b4ffb6010f531b07848c300-Abstract-Conference.html) L06 | O-B 利用网络不变性旋转并生成小dense矩阵 | O-A 离线删维；输出是实际紧凑dense表示，而不只是稀疏权重 |
| [SVDQuant](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html) C06 | O-B 低秩支路吸收outlier后改变函数的计算分解 | O-D 4bit主干+高精支路融合Nunchaku，且O-E校准；支路访存成本不可免费 |
| [TeaCache](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) L14 | O-C 复用前步模型输出 | O-A 根据廉价timestep感知proxy判定何时跳算 |
| [Chipmunk](https://arxiv.org/abs/2506.03275) L09 | O-C 跨去噪step复用慢变化特征 | O-A 仅重算快变column，O-D列稀疏kernel及重排 |
| [SVD-Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html) L39 | O-C 缓存及预测历史特征 | O-B 按SVD主/残空间定义不同的更新函数；主方向EMA预测、残差直接复用 |
| [L2P / Learnable Linear Predictor](https://arxiv.org/abs/2604.26365) 外部补充 | O-C 学习时间状态的缓存预测器 | O-E 学习各去噪step的线性历史组合系数；注意是L2P-Cache，不是其它同名L2P |
| [NVIDIA NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) L31 | O-D block FP4/scale/真实Tensor Core格式 | 数值表示/quantizer不是新的FFN函数架构 |
| [CodeGEMM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html) L73 | O-D 码本求和替代对应量化线性乘法 | 已检查作者代码中单token查表与多token回退；Wan大M不能照搬其快路 |
| [SharQ](https://arxiv.org/html/2606.26587) L74 | O-A 基于当前输入保留部分乘法 | O-D FP4 sparse+FP4 dense residual双GEMM，不能把“稀疏”当作全FFN省算 |
| [LUT-GEMM](https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf) L76 | O-D 量化GEMM按编码查表执行 | 主要作用是算子求值协议，而不是省掉原FFN整层 |
| [SparseGPT](https://arxiv.org/abs/2301.00774) C01 | O-A 离线删权重/结构 | O-E 用Hessian感知校准与局部重构决定权重；数据选择并非推理时动态A |
| [S²Q-VDiT](https://papers.neurips.cc/paper_files/paper/2025/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html) C05 | O-E 显著校准source及稀疏token蒸馏监督 | O-D 最终量化运算属于部署协同；训练选token不是当然的在线推理O-A |
| [Q-VDiT](https://proceedings.mlr.press/v267/feng25q.html) C07 | O-E token相关量化纠偏与时间保持蒸馏 | O-D 量化计算协议＋O-B可学习校正模块；具体以实际模型为准 |
| [G13/AtP*](https://arxiv.org/html/2403.00745v1) L89 | O-E 因果定位与辨识 | 科学证据动作；当前G13 patching并非直接部署的省算方法 |

特别是**L²P/L2P**：这里以用户“复用历史”语境核对为 **L2P = Learnable Linear Predictor / L2P-Cache（arXiv:2604.26365）**；字面L²P可存在命名歧义，当前纸面索引明确到论文标题，暂不伪造G14 LID。

## 4. FFNDS已有路线改用“方法操作签名”解释

| 现有方案或资产 | I/F/C科学入口 | O-A～O-E操作组合（实际/开放边界） | 当前真正问题 |
| --- | --- | --- | --- |
| **UB1 step-static Whole64** | I/F/C | **O-A** 固定step×block完整组；**O-E** 来源能量校准；接consumer时再有**O-B** W2/readout＋R256拟合 | 固定组省了真实W1/W2多少，省掉哪些行为响应？20来源已暴露视频运动显著下降，训练内拟合改善不等于full-video恢复 |
| **A5+HWG历史mask** | I/C＋F | **O-A** token条件选择；**O-E** 评分/指标定义；原W1预代理及tile执行需分别记成本 | 若先做满W1再决定post-GELU mask，不能把上游GEMM当作已跳过；重点是时空函数与合法Q2边 |
| **Affine54折叠/Whole94/R256** | C＋F | **O-B** 仿射＋非线性基替代、消费者重拟合；**O-A** 仅保留54组真实非线性；**O-E** 局部校准与自由视频 | 相同MAC下函数表示能力 vs 质量；真实少数视频模糊/运动有损，不否定整个B操作族 |
| **G11/G12有符号量化信息证书** | I＋C | 当前为**ALG信息结构**，可能导向O-A选择器或O-D低位精确投影；**尚未形成真实部署O-A/O-D加速** | 同一输入多W1行联合约束相对直接p=Wᵀc的净信息/在线成本 |
| **Activation 2:4/RT-Lynx迁移** | C＋I | O-A 2:4选择真实operand，O-D packed布局与SpTC；作者RT-Lynx有O-B/ O-E norm和LoRA恢复 | 合成BF16 online CUTLASS少量收益并不等于原Wan画质与整FFN提速；可尝试异构融合/新执行粒度 |
| **Calibration V2与G13功能定位** | F＋I（提供所有路线证据） | **O-E** 来源/teacher-student/响应/因果和完整视频验证，不单列为部署省算结构 | 真实独立image cluster、同噪声配对、signed后缀与视频运动语义分别保留 |
| **跨step缓存专题** | I＋F＋C | **O-C＋O-A** 依历史合法可得特征预测/跳算，必要时加**O-B/O-D/E** | 仅四去噪step、high/high/low/low两expert；cache复用对象必须拥有相同函数/状态/量化与合法依赖；额外IO/VRAM是否抵过重算 |
| **M-A/A0普通6016窄学生** | C＋F | **O-B** 新非线性计算函数；**O-E** Teacher/Student有限适配和video监督；继承原HiF4的Q2为数值约束，非自动O-D创新 | 原FFN两GEMM MAC约43.52%；A0是工程与强科学比较器，A1须定义额外设计变量 |
| **M-B原候选B：敏感输出坐标** | **F＋I＋C** | **O-B** gate/suffix joint敏感的输出空间与学生预测；**O-E** 有辨识力的同source/同state后缀读出；**O-D** 为可选factored/fused执行 | 首先判别有无低能高影响方向、pre-W1可预测性；folded没有在线W2低秩MAC节约，factored须计额外launch/QDQ |
| **M-C原候选C：编码直接求值** | I＋C | **O-B** 学习完整非线性输出函数与U基，**O-D** 码本查表/内核及输出展开，**O-E** 数据/码本训练 | 输入Rx是否足够，U输出秩是否足够，交互表与读表比连续小MLP便宜吗；不能因为有离散code就自动视为O-A动态选择 |

**A/B/C方法的共同科研竞争**是：当固定相同实际MAC、真实QDQ、training预算时，有限恢复的紧凑学生、下游敏感坐标、以及直接编码函数各自还留下什么可辨别的质量/真实速度差。G13只在需要定位功能作用时作为工具，不是所有新表示的强制前置。

## 5. 新组合空间、优先级和G14导航

**组合I：前置信息预测 O-A ＋新函数 O-B ＋低位执行 O-D ＋O-E校准。** 例如预可得信号决定哪一片新紧凑函数求值、按真实NVFP4/MXFP4等kernel执行。与旧后验“调神经元TopK阈值”的区别是改变了**函数类与合法计算单元**，但所有观察、路由、Q2/packing、kernel及写回必须算在线成本。

**组合II：后缀敏感新函数 O-B ＋有根据的适配/验证 O-E ＋可选O-D分解。** 即当前最优先的 **M-B**。用同source gate-conditioned联合后缀响应定位应保留的方向，在pre-W1 x上学习便宜映射，再评估folded与factored部署。这条科研路线**不以动态mask为前提**。

**组合III：时序合法复用 O-C ＋只重算必要分量 O-A ＋可选O-B输出基 ＋O-D fused实现。** 从Tiny/Chipmunk/SVD-Cache/L2P学到方法，但Wan仅四步双expert，需先以原状态/权重一致性和历史可用性确定缓存的可迁移时间依赖，避免偷用同step未来状态。

**组合IV：直接编码O-B＋O-D，与连续紧凑函数平行竞争，O-E校准。** 即M-C，试验输入统计充分性和完整FFN输出码内歧义，而不是继续尝试用“低位码直方图是否能估得神经元重要性”替代真正函数求值。

**方向排序**：第一轮仍先由Agent完成M-B的METHOD_SNAPSHOT；同时可保留M-A/A0作为matched基线、M-C作为并行新信息计算、以及真正前置O-A＋O-B＋O-D和历史O-C复用的新增候选空间。不是固定五个GPU项目；任何新组合可由独立agent按同样指纹探索。正式proposals/reviews/ENDORSEMENT归[QUEUE](../method_discussion/QUEUE.md)，运行范围归[ACTIVE](../ACTIVE_RESEARCH.json)。

**G14查找顺序**：
1. 在[全论文计算边索引](../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)或图上查原生T01–T16同算子位置，先读一手paper_links的intervention/decision_time/review_depth与源码，明确实际是选择、表示、复用、代数还是学习验证。
2. 在本文件判别该论文有哪些O-A～E，哪项为PRIMARY，哪些为辅助，是否真实online；跨模型论文只能产生LIT迁移关系。
3. 回到原Wan SRC/QDQ/geometry/step专家及旧NATIVE/VIDEO/SYS，构造至少两条有相反预测的机制，并选择证明、现存保存资产或将来受控执行。
4. 将新的方法指纹、first discriminator及数学/数值合同加入独立METHOD_SNAPSHOT，不直接修改原有G14 LID、方法QUEUE或历史RUN。

**结论**：把五操作作为**可组合动作语法**，把I/F/C作为**科学问题空间**，把原生运行/视频作为**证据与物理成本约束**。这样既能肯定传统稀疏与量化的贡献，也允许 FFNDS 研究不依赖稀疏的新函数、编码计算与合法时序复用。
