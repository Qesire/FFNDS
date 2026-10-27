# FFNDS 三种新FFN结构的独立方法评议（2026-10-27）

**状态：INDEPENDENT_METHOD_REVIEW / DISCUSSION_ONLY / STATIC_ONLY。** 评议对象：[A/B/C原方案](COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)。应用仓库ffnds-paper-discovery、ffnds-experiment-evidence、mechanism-research、problem-reformulation、web-research-discovery及数学方法Skill，从问题结构、数学必要条件、已有证据、真实源计算路径及原论文重新评议。本评议**不是**独立第二Agent的支持票、QUEUE认可或实验授权；没有重新训练、执行模型、生成视频或获得A/B/C实际Wan实验结果。

## 0.1 评议过程中主分支新增的原生证据（2026-10-28归档版本）

在本评议进行时，main又合并了[consumer独立20来源评测](https://github.com/Qesire/FFNDS/blob/main/research/20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)、[源级状态与原生生成](https://github.com/Qesire/FFNDS/blob/main/research/20261027/UB1_CONSUMER_DISJOINT20_NATIVE_RESULTS_20261027.md)、[A0 Whole94适配就绪审查](https://github.com/Qesire/FFNDS/blob/main/research/method_discussion/discoveries/E0/compact6016-adaptation-readiness-20261028.md)、[四位置来源稳定诊断](https://github.com/Qesire/FFNDS/blob/main/research/20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md)和[真实GPU fused 2:4数据](https://github.com/Qesire/FFNDS/blob/main/research/20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)。据此更新本评议的证据条件：

- **VIDEO，已暴露开发20来源**：fresh Dense/consumer有20对同源同initial-noise真实81帧视频；相对Dense，消费者RAFT 6.894→1.152（−83.28%），官方dynamic degree二值6/20→0/20，imaging 72.629→71.626（−1.38%）；一致性升高可能伴随静止。另有20条历史neutral，初始噪声字节对齐未证实，勿将其作为同噪声强配对。此次60条视频/20来源、原版八个独特指标不等于正式VBench九维和camera motion，也不是独立未暴露确认集。**这使A/B的运动保留成为更具跨来源一致性的研究动机，不证明B低能高敏方向存在。**
- **NATIVE/OBS，四teacher位置**：旧CAL24源删除时whole64固定支持Jaccard均值0.980–0.998，说明在测量的四位置、该能量统计下，支持对单来源移除较稳定。它让“由于这四位置的来源支持选择不稳而直接失去运动”成为相对弱的解释；**其余156位置、Student-state/轨迹及量化数值仍需单独辨别**。
- **NATIVE-OBS，A100合成2:4**：新增fused online CUTLASS路径W1形状15.684ms vs dense16.247ms，W2形状15.867ms vs dense15.982ms，只有约3.5%/0.7%非交错、五次稳态测量的边际收益。它是**masked合成输入的执行证据**，不是原Wan稀疏化质量、跨机器显著收益或完整视频提速；B-Factored与C-LUT的真实系统成本门因此更加重要。
- **A0已有具体原生启动路径**：Whole94原有可重用bank包含选出的完整64通道组、真实缩窄Q2、原W1选行及由四来源ridge修复的BF16输出权重。新A0适配的量化STE、训练/heldout来源身份、新student状态及权重包可达性都**尚未验证/生产**；需严格区分历史ridge预算与新的训练预算。

本节后面的A/B/C判断在此更新范围内成立：**大量源的新事实是行为退化而不是新方法成功；A更需要强恢复基线，B更需要定义时间行为读出，C更需要物理执行证据。**

## 1. 评议结论

| 方法 | 正确且有价值的部分 | 当前最重要的结构缺口 | 决策 |
| --- | --- | --- | --- |
| **A：可恢复6016维FFN** | ALG预算准确；简单、可导出、成熟迁移路线 | **A本身就是普通6016维学生**；把“同宽度普通学生”设为唯一强对照时，没有可验证的新增算法差别。恢复训练/显存预算庞大 | **A0作为强实现基线；A1另定义有区别的机制增量** |
| **B：后缀敏感方向** | 低能高影响是CoBRAS启发的可证伪假说；gate/跨step作用值得独立研究 | 平均协方差乘平均梯度可能丢失同source关联；折叠W2不会降低W2 MAC；需要定义gate/完整suffix及计算形态 | **优先深入机制，区分B-FOLDED和B-FACTORED** |
| **C：编码直接求值FFN** | 信息不只用于routing，而是直接承担函数求值，构造空间广 | 输入Rx丢失信息与输出U低秩是两个独立不可逆瓶颈；二组LUT可能吞掉全部显存/带宽优势 | **保留高风险并行方向，先估信息下界与表成本** |

统一科学框架依然是[I/F/C](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)；A/B/C不是三个互斥研究入口。UB1/fixed-RR、Affine54、G12结果为历史比较器与构造材料，不能外推为新三方法已经有视频质量提升。

## 2. 共同数学坐标

真实Wan四步D-Q、高高低低两expert、每expert40 blocks，FFN顺序为pre-W1/post-Q1输入5120 → W1输出13824 → GELU(tanh) → Smooth/HiF4 Q2原64→8→4分组 → W2输出5120 → caller e5门控＋残差 → attention/下游block/四个去噪step/21 latent时间槽及81RGB帧。原生fake-QDQ+BF16 GEMM与真实packed FP4不是同一部署语义；新窄学生须另建合法Q2量化尺度，实数折叠不保证浮点/量化逐位等价。来源image cluster是独立统计单位，token/row/step是嵌套，已暴露两源六视频不能变成新独立确认集。

## 3. A：预算成立，算法贡献应从A0/A1开始分开

令D=5120，H=13824，Affine54真实非线性hidden数=3456。原始主投影MAC/token为2DH=141,557,760。Affine54的D²+2D×3456=**61,603,840**；一个h=6016=94×64的普通FFN恰为2D×6016=61,603,840，为dense两GEMM的**43.5185%**。这是ALG恒等，**不包含**真实QDQ、gate、VRAM及launch。两矩阵×80位置共有4,928,307,200参数；两矩阵rank32 LoRA的增量参数约57,016,320，反向图及teacher/suffix预算另计。

**A0（系统基线）**：每expert-block一个6016维学生，post-Q1→新W1→GELU→新Q2→新W2→原e5，明确Whole94或其它初始化和同预算LoRA。只要A0完整视频与原生物理成本成功，便是有价值的成熟迁移和可复用基线。

**A1（需新增真正的设计变量）**：例如保持全局MAC、让各block/expert宽度非均匀分配；或采用同时约束teacher同状态函数与student自由轨迹的固定预算适配。同样要用同结构、同训练参数、同来源及同算力的正确消融；如果A1只是另一个初始化就描述初始化效应。不能拿A0同A0本身作“新方法胜出”对照。

最近直接先例：[TinyFusion CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html)按恢复后质量选结构（作者[源码](https://github.com/VainF/TinyFusion/blob/a147de7c33c2bc12872b9913f459119068b81ac8/prune_by_learning.py)有搜索gate/LoRA和导出重载）；[Dynamic-in-Few-Step 2026](https://arxiv.org/html/2607.06631v1)将四步视频生成、step-specific结构、progressive训练和output rollout联合；[SVDtrunc 2026](https://arxiv.org/abs/2609.04646)对DiT做按block的rank分配与蒸馏；[Dense2MoE ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Dense2MoE_Restructuring_Diffusion_Transformer_to_MoE_for_Efficient_Text-to-Image_Generation_ICCV_2025_paper.html)把FFN转换为MoE；[FastFLUX AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/37237)用线性分支替代和邻层LoRA。**恢复性和可训练新结构本身已有成熟算法家族。** FFNDS更有辨识度的潜力来自原Wan HiF4量化闭包、少步motion保持及公平的完整成本/来源泛化。

首个判别：先冻结A0算子与A1相对A0的一个真实变化；若改变了宽度和训练预算两个因素，就拆开对照。A1的功能解释不能仅归因于减少local SSE。

## 4. B：应把gate-conditioned后缀的样本联合响应作为科学对象

原提案以输出协方差C与平均下游敏感G提出代理
\[
E_{\rm marginal}(P)=\operatorname{tr}[\bar G(I-P)\bar C(I-P)^\top].
\]
这是合法非负代理；但$\bar G\bar C$通常**不等于**同source的真实平均后缀响应误差，因为门控、梯度和输出变化可能相关。若冻结当前caller gate和其它前缀，定义来源/step/条件状态s上的$y_s$、门控对角矩阵$D_{g_s}$及从完整FFN写回到预先定义下游读出的局部Jacobian$J_s$，则一阶代理更应从
\[
E_{\rm joint}(P)=
\mathbb E_s\Vert J_sD_{g_s}(I-P)(y_s-\mu)\Vert_2^2
\]
出发。该式是**小扰动、冻结gate的局部近似**，不是全视频定理；$J_s$应在多token注意力作用空间定义。若扰动来自FFN输入而门控也随之变，必须加入$\operatorname{diag}(y_s)\delta g$，不能简单归入投影误差。CoBRAS为斜投影提供动力系统先例，最优$P=\Phi\Psi^\top$还需说明$\Psi^\top\Phi=I$和source/step稳定性；原G13有限P/N标量不是完整$J_s$或G矩阵。

**重大部署分叉**：若先训练$\Phi\in\mathbb R^{D\times r}$与$B_c\in\mathbb R^{r\times h}$再离线折叠W2=$\Phi B_c$，在线W2仍需$Dh$ MAC，低秩约束只作用于表示/可训练性；若保留**B-FACTORED**的$h\to r\to D$两个乘法，W2的主MAC可降为$r(h+D)$，当$r<Dh/(D+h)\approx2766$时低于原W2。以h=6016、r=512为例，W1＋分解W2共**36,503,552 MAC/token**，相当于A0两GEMM的**59.26%**；新增GEMM launch、数值舍入、量化scale和中间张量不能遗漏。两种部署方案应分别做原生性能判别。现成gate还将固定pre-gate空间$\Phi$变为随token变化的写回方向$D_g\Phi$，可构成新的gate-conditioned几何研究假说。

最近先例：[CoBRAS SIAM JSC2023](https://doi.org/10.1137/22M1513228)状态协方差＋adjoint敏感平衡、低方差重要坐标；[2ndMatch CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html)针对扩散模型时间响应的二阶Jacobian匹配；[OBS-Diff ICLR2026](https://arxiv.org/abs/2510.06751)去噪步Hessian/剪枝；[SVDtrunc 2026](https://arxiv.org/abs/2609.04646)低秩蒸馏。不能将敏感度加权或低秩W2本身视为原创，研究差别应落实在**HiF4门控、原生后缀、分布相关性及能部署的输出函数空间**。

**最小判别**：先检查保存资产有无post-Q1、pre-gate y、e5、代表性后缀有限JVP/有限差，以及哪些由数据重放才能获得（OPEN_BINDING）。用同rank PCA、marginal G/C、joint gate/J、随机空间和A0普通学生做**来源级**相反预测；不能从局部高敏直接命名“运动神经元”，独立视频仍需完整验证。

## 5. C：先做两个与训练无关的可表示性检查

**输入瓶颈**：设z=Rx、完整teacher FFN函数f(x)。若$Rx_1=Rx_2$但$f(x_1)\ne f(x_2)$，固定R和任意大的K、任意码表都不能让同一个码解释这两个真实输出。平方损失下最佳解码残差为$\mathbb E\operatorname{tr}\operatorname{Cov}(f(X)\mid RX)$；若f在真实输入流形上可微、存在$v\in\ker R$且$J_f(x)v\ne0$，则该区域不能精确因子分解为函数d(Rx)。Q2量化不连续时以原生有限扰动/编码碰撞代替导数论断。**需同时报告来源与视频条件分层的码内输出变化，不暗用未来teacher信息。**

**输出瓶颈**：不论编码有多少交互，只要输出$\widehat y=U\alpha(z)+b$且rank(U)=r_o，预测只能位于$r_o$维仿射子空间。对中心化teacher输出协方差C_y的Euclidean重构，最佳rank$r_o$空间残差下界是尾部特征值和$\sum_{j>r_o}\lambda_j(C_y)$；source-conditional gate/后缀作用另需定义。首先分开量化输入信息不足与输出低秩不足，才能有因果性的码本消融。

**交互表资源**：$B=16,K=256,r_o=256$、BF16，单组表$BK r_o\times2$约2MiB；每个二组交互表$K^2r_o\times2=32$MiB/**单expert-block**。四个交互就加128MiB/函数，80个FFN约10GiB额外表空间，还未计R、U、L与其它权重；可用共享字典、低秩交互、按step复用或连续残差探索新的表示，但要计新增算术与访存。

已知近邻：[LUT-NN MobiCom2023](https://www.microsoft.com/en-us/research/publication/lut-nn-empower-efficient-neural-network-inference-with-centroid-learning-and-table-lookup/)可学习质心直接替代算子；[LUT-Diff IEEE TMC2025](https://chrisplus.me/assets/pdf/tmc25-lutdiff.pdf)已经使用Fisher、输出加权码本与移动GPU/CPU系统；[LUT-DiT IEEE TVLSI2026](https://doi.org/10.1109/TVLSI.2026.3732472)使用FP4相关码本、分级LUT和专用片上存储（不同硬件与算子）；[Product-Key Memory NeurIPS2019](https://papers.nips.cc/paper_files/paper/2019/hash/9d8df73a3cfbf3c5b47bc9b50f214aff-Abstract.html)已有乘积键稀疏查表检索。C的差别必须落到“**直接预测Wan完整非线性FFN函数，而非逐线性GEMM的乘加近似**”和在大M实际GPU成本下的可证实收益。

**最小判别**：先从已有源级$x,y$诊断输入碰撞及输出谱（没有就明确需要新producer），再算$R/U$投影和lookup负载，比较同rank连续小MLP与LUT-Diff式逐线性查表；若有明确信息/成本缝隙才创建码本训练。

## 6. 第一轮竞争机制与后续推进

| 决定 | 同等条件比较 | 真正判别的现象 |
| --- | --- | --- |
| A是否应成为方法论文主体？ | A0普通6016学生vs A1明确定义的单一新增变量，固定训练、Q2、来源、算力 | 新变量带来的独立视频质量–wall曲线增量 |
| B低能高敏存在吗？ | PCA/能量、平均G×平均C、sample-joint gate/J、普通学生同rank | 独立source后缀有限响应及动态/外观维度；marginal与joint排序不同时应反思机制 |
| B低秩可加速吗？ | folded vs factored、同量化、同rank和真实M布局 | **真实W2及整个FFN wall**，不是仅算MAC |
| C编码的信息足够吗？ | 投影碰撞、输出尾谱、连续vs离散、有限交互表 | 同码是否有不可恢复输出差，LUT是否抵过读表/展开成本 |

**推进排序**：B优先数学与资产可辨性；A0并行冻结为工程与科学强比较器，A1需要新的可证伪增量才进入方法队列；C先完成必要的信息及物理成本计算，保留不依赖B的探索。任何方法也可以从其它I/F/C源图自由扩展。

**复用资产方案（本轮未新生产）**：来源级manifest与image cluster、Dense/Student同seed和scheduler及quant状态；pre-W1/post-Q1输入、pre-gate y、gate、真实post-GELU/Q2、G13已存有限响应或视频/latents须先验证locator和producer。未来新资产包括A0权重与冻结适配预算、B来源/step共同敏感JVP与factor/fold性能记录、C input-collision/output-spectrum与码表配置；按method version、producer、shape/units、统计身份、数据曝光、可复用/失效条件登记。无新的运行登记，正式GPU执行仅由ACTIVE限定。

## 7. 阅读覆盖范围

本轮核验了FFNDS[跨路线证据](PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)、[近邻论文库审查](PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)、[Calibration V2](CALIBRATION_V2_METHOD_SNAPSHOT.md)、[G12信息研究](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md)。一手扩展：DIFS原文HTML第4节及后缀机制、SVDtrunc原文HTML结构压缩方法、LUT-Diff作者TMC论文§1–2、TinyFusion作者代码、CoBRAS正式摘要、2ndMatch/Dense2MoE/OBS-Diff/FastFLUX正式论文摘要、LUT-DiT和Product-Key正式记录。不是逐篇全文查重结论，更未证明首次提出。ALG与数学必要条件已经评议；三方法的模型质量及真实新部署结果保持OPEN。另行第二Agent review仍须以其真实评议来源写入QUEUE，而不是自动获得投票。
