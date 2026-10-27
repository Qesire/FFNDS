# FFNDS · 部分线性FFN、结构重构与跨步纠错：一手论文核验及Wan迁移边界

> 2026-10-27｜文献与源码交叉研究｜**引用身份**：PRIMARY_TEXT（已阅读原文实质方法）、OFFICIAL_ABSTRACT（正式出版摘要）、VENUE_METADATA（仅发表记录）、PROJECT_NATIVE（本仓库原生RUN）、OPEN（待验证）。本稿作为[Affine54方案](QUANTIZATION_CONSISTENT_FFN_FUNCTION_AND_TRAJECTORY_PROPOSAL_20261027.md)与[UB1方案](UB1_STEP_BUDGET_CALIBRATION_AND_VIDEO_GENERALIZATION_RESEARCH_PROPOSAL_20261027.md)的**共同文献入口**；两份方案不重复长篇论文评论。文献调查不是排除已有方法的原创性筛选器——可直接移植有证据的现有方法。

## 0. 最有用的研究判断

**固定仿射折叠不是待发现的新基本原理。** TARDIS(2025)已把GELU常见区间近似、预折叠D×D、2-bit量化前置预测器、在线异常神经元的“减近似加真实”补偿及专门CUDA实现联合起来；同一团队的LLMFolder已在EuroSys 2026正式发表。后者本轮能核实作者、题名、出版信息及DOI，**尚未取得可核验全部技术公式的全文**，不声称它的算法必然与TARDIS一致或改进在哪一项。

Affine54不是裸线性回归：它已有后Q2神经元仿射分解、原非线性真实54 whole64组、非线性消费者V_S ridge重构、全160位置运行和六个真实视频/实际时延数据；其视频失真属于值得解释的实证事实。下一步可**直接复现或组合**TARDIS异常修正、Model Preserving Compression的选列/消费者恢复、BRECQ式中间组重构、CTEC跨步训练，也可以复查FP4/N:M硬件路线。优先看Wan同源真实收益和信息成本，**不要求新方案必须和论文“刻意不同”**。

## 1. Affine54与TARDIS同属哪种函数表示？

原FFN按已核源码实际为 \`postQ1→W1+b1→tanh-GELU→Smooth2/HiF4 Q2→W2+b2→e5 gate/BF16 residual\`。令 \(a=W_1q_1(x)+b_1,\ q(a)=Q_2(\phi(a))\)，其中Q2实际含Smooth2及多级共享尺度/数值cast。

在保存的实际post-Q2响应上定义 \(q(a)=\kappa\odot a+\beta+r(a)\)。若完整保留\(r\)，这是**恒等分解**；若只保留部分非线性残差，才近似。Affine54将仿射部分折入 \(A\in\mathbb R^{5120\times5120}\)，再保留54组/3456原神经元的实际W1/GELU/Q2，并通过校准得到 \(V_S\) 和bias：
\[
\widehat F(q_1(x))=A_{\rm bank}q_1(x)+b_{\rm bank}+V_S q_S(a_S).
\]
这是实际bank的执行形式，并非全\(q_i\)真实计算后减去补集；W2消费者重构与原b2/门控的边界需根据真实runtime逐层核验。

D=5120、H=13824时，**理想的线性投影MAC密度**为
\[
\frac{D^2+2D(54\times64)}{2DH}=\frac{40+54}{216}=43.5185\%.
\]
这里D×D相当于40个原64组对应的两次投影；不含额外QDQ、GELU、bank权重、kernel调度和内存成本。

真实已测：2个已暴露持出来源×4step×40block=320个局部案例，Affine54平均gated SSE **8.54%**，同理想MAC Whole94 **11.67%**，原affine-anchor input ridge **15.35%**，Affine54在305/320例小于Whole94；two-source×three-arm共六个81帧视频已生成，source12热态原生FFN9.707→4.449秒（2.182×），完整生成50.750→45.313秒（1.120×），但主观可见模糊、运动/细节偏移，**不能声明视频保真已通过**。校准四来源、每函数256行，弱于UB1原24来源校准，但不能把两者误差直接归因到样本数。[E0函数重构](../20261026/E0_FUNCTION_RECONSTRUCTION.md)、[完整局部](../20261026/E0_ALL_BLOCK_RECONSTRUCTION.md)、[真实视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)、[bank源码](../../tools/build_e0_reconstruction_bank.py)、[native hook](../../tools/e0_reconstructed_runtime.py)。

**与TARDIS关系**：
- 相同：逐神经元近似非线性与恒定线性折叠；折叠项可含常量；在必要时恢复原非线性。
- 已知不同执行身份：Affine54用离线固定的54×64真实通道支持/新的V_S，TARDIS用每神经元校准热区与**在线量化预测器**决定出界补算，修正时减去近似、加回实际；TARDIS使用定制CUDA kernel。
- TARDIS使用GPTQ 2-bit压缩W1形成预测器，在线预测本身要计访存及GEMM；原文指出修正路径/sparse索引可能成为主要开销。其7B LLM上的vLLM**1.6×完整服务加速**主要处于自回归/权重IO场景，不可换算到Wan M=31668、A100或HiF4。
- TARDIS原文主要研究普通逐神经元FFN；Wan Q2原64组共享scale，使修正单列时可能改变未选通道数值。若迁移TARDIS，必须明确定义whole64合法group上的固定/动态计算与scale补集作用，而不是把原文CUDA速度无修改视为可部署结果。

[TARDIS v1全文，§5.1–5.4、§6–7](https://arxiv.org/html/2501.10054v1)｜[EuroSys 2026 LLMFolder官方目录](https://2026.eurosys.org/papers.html)｜[LLMFolder DOI](https://doi.org/10.1145/3767295.3769339)。

## 2. 另一条同样成熟的路径：保留真实非线性基与消费者重构

**Model Preserving Compression（Chee等，NeurIPS 2022）**利用interpolative decomposition(ID)在原激活列中选取子集， \(H\approx H_S T\)，于是
\[
HW_2 \approx H_S (TW_2).
\]
其独特之处是基来自真实被保留的激活列，且T折到下游消费者；同时考虑各层结构与免微调。**Affine54的输出矩阵\(V_S\)已体现消费者重构大类思想**，但其构造是逐神经元后量化OLS残差+固定/校准支持、ridge拟合，不能说已完整复现MPC的pivoted QR、ID误差/层预算证明。[NeurIPS官方页](https://papers.neurips.cc/paper_files/paper/2022/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract-Conference.html)、[作者PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/f8928b073ccbec15d35f2a9d39430bfd-Paper-Conference.pdf)。

可直接借鉴的三个组件：合法原Q2组的可恢复性/列选，固定预算下的消费者重构，跨层容量分配。它可以是Affine54的增强或独立于仿射主干的Whole94等竞争表示。

**FLAP（An等，AAAI 2024）**在结构化LLM剪枝中使用激活波动度量、全局压缩比搜索与均值bias补偿。其bias是低成本基础恢复，而不是Affine54同等的D×D+V_S表达；但它的**可恢复性/来源稳定性**比单组原能量更契合UB1新20泛化问题。[正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/28960)。

**BRECQ（Li等，ICLR 2021）**原对象是后训练量化，而不是Wan结构删组；它通过block reconstruction与Gauss–Newton/二阶近似在局部和完整模型之间取中间监督边界，说明压缩中应把下游依赖与训练校准的泛化进行折中。可以直接采用其分块重构思想作为比较器；\(e^\top J^\top HJe\)是常见二阶形式，**非本项目首次发现、且不是全量视频精确等价式**。[arXiv原文](https://arxiv.org/abs/2102.05426)。

**其它邻近先例**：[ICLR2025 Model Folding](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f102c495b31bcc7b2a11c27de75475fc-Abstract-Conference.html)研究不依赖数据而合并权重相似神经元；是相近的结构替代，但**不是与TARDIS“常量折叠”同义的特定算子**。[EMNLP2023 两层FFN近似统一框架](https://aclanthology.org/2023.findings-emnlp.49/)将MoE、PKM和稀疏近似放入更广的函数空间，启发调整计算函数族而非固定54组。

## 3. 跨时间步误差与功能响应：采用已知机制，并从真实Wan改写

**CTEC（Li与Du，AAAI 2025）**证明只减当前量化噪声预测偏差不足以解决累积误差；以蒸馏学习对前一步状态误差的纠正。应用到FFNDS可在冻结true scheduler、相同source/seed的Teacher/Student实际状态上构造离线纠错目标。理想实数Euler流场形式下 \(z_{t+1}=z_t-h_tv(z_t)\)，若\(\delta_t=z_t^S-z_t^D\)，下一状态想回到Dense，则完整模型所需\(v_S^*(z_t^S)=v_D(z_t^D)+\delta_t/h_t\)。**这只是理想整体flow控制目标，不是任何单个FFN可直接获得的目标**；原Wan真实scheduler/precision须逐次复核。CTEC原论文做的是量化扩散/图像，不是UB1/视频实测。[AAAI正式摘要](https://ojs.aaai.org/index.php/AAAI/article/view/34039)。

已测[完整四步finite counterfactual](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)使用
\[
P_t=T_D(z_t^S)-T_D(z_t^D),\quad N_t=T_S(z_t^S)-T_D(z_t^S),
\]
\[
\|\delta_{t+1}\|^2=\|P_t\|^2+\|N_t\|^2+2\langle P_t,N_t\rangle.
\]
是对原生finite map的**严格代数分解**，不是新传播定理。后续实验在其它候选上观察到P/N cross随source/seed变号、local误差改善但视频质量仍可能劣化；它们提示功能相关的时间/行为目标，而**不能直接标成Affine54或UB1本人已有的因果定位**。[新来源与seed反例](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)。

**2ndMatch（CVPR2026）**已研究扩散剪枝后模型敏感性/二阶Jacobian匹配；**OBS-Diff（ICLR2026）**已提供扩散时间步加权Hessian和结构剪枝。它们可直接作为成熟监督/支持选择器。若研究有符号finite secant，
\[
\Delta_S(b;\eta)-\Delta_D(b;\eta),\quad\Delta_F(b;\eta)=F(b+\eta)-F(b),
\]
它是经典有限差分思路，不能仅凭新增一个loss就主张原创性。真实性取决于按实际AdaLN、gate、Q2和前缀状态配置完整比较。[2ndMatch论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_2ndMatch_Finetuning_Pruned_Diffusion_Models_via_Second-Order_Jacobian_Matching_CVPR_2026_paper.html)、[OBS-Diff](https://arxiv.org/abs/2510.06751)。

**TCEC（ICML2026）**又提供了量化模型的误差递推和补偿；较早的arXiv版本闭式索引式在本项目审查中出现与边界条件不一致的代数问题。可采用核心机制但必须独立从原Wan步进推出递推，并以终稿核对，不能不经核查把其推导中的一个特定展开当作定理。[TCEC正式页](https://proceedings.mlr.press/v306/liu26fv.html)。

## 4. 不采用仿射表示的强系统竞争者：RT-Lynx

**RT-Lynx（Cong等，ICML 2026）**不是TARDIS/Affine54的进一步仿射折叠。其原文§4在DiT GEMM输入侧施加在线**2:4半结构化激活稀疏**，使用范数恢复与约2000步LoRA微调补偿，在H20上设计融合稀疏化/压缩/SpTC-GEMM/LoRA输出的专用kernel，作者报告线性层平均最高约1.55×加速。实际测试Qwen-Image、FLUX、Z-Image等图像DiT，并非Wan2.2-I2V四步；不支持称作免训练或已经拥有本视频任务1.55×速度。[ICML正式页](https://proceedings.mlr.press/v306/cong26a.html)、[arXiv逐节全文](https://arxiv.org/html/2605.26632)。

原文还表明未经融合在线N:M动作本身可占较大总运行时间，LoRA也需要成本，故真正突破不是发现“激活可以变稀疏”这一后验统计，而是**合法N:M硬件形状＋在线成本融合＋质量恢复**。这对Wan真实大M/HiF4+BF16值得认真竞争测量，并可能与UB1和Affine54并列，不要求合并成一个复杂算法。

## 5. 新假说与最经济的先行判别

| 可以直接借鉴的研究思想 | 有价值的机制问题 | 最小判别与明确反例 | 何时值得原生部署 |
| --- | --- | --- | --- |
| TARDIS条件热区/动态修正 | Wan各step/block/原64组是否存在跨来源稳定的“近线性工作区”？ | 从可读保存的preW1+postQ2恢复合法组尺度，测按来源/step预测outlier的召回/成本；若outlier打满M/K tile，在线修正不省算 | 预测器+修正+元数据实际成本低于减少的工作 |
| MPC+FLAP/OBS重构 | 为什么原54组或UB1 C_mean的固定支持在不同来源失稳？ | 固定group/MAC、同数据比较joint/ID/FLAP/OBS选择与V_S/rank校准；避免同时改变来源、loss、预算 | 完整原生W1/Q2/W2仍合法，独立视频改善 |
| BRECQ/2ndMatch/CTEC | 局部近似函数和跨步视频行为为何分离？ | 单点值/有限响应/同前缀suffix/完整trajectory多边界分开核对；若相同局部SSE产生不同真实后缀作用，则需新目标 | 已有成熟目标确实改善真实视频且训练成本可接受 |
| RT-Lynx N:M | 更换物理动作能否优于固定whole64/折叠？ | 仿真实际M=31668，核对2:4可执行形状、HiF4量化顺序与LoRA/kernel成本 | 有可交付的实际GPU native kernel和完整video评测 |

**不要预判**“既然TARDIS有预测器，Wan必然需要动态补算”，亦不预判“最后一步预算越多越好”。上面每条路径都可以形成有用的证据、可复现迁移或实际方法。

## 6. 文献覆盖与尚待核验

- **已逐节阅读**：TARDIS v1 §§3–7（线性化、predictor、fixing与CUDA开销），RT-Lynx arXiv v2 §§2–5（activation N:M、LoRA、kernel、图像评测），BRECQ arXiv论文的方法及二阶推导；**阅读作者/正式出版方法与摘要**：Chee/MPC原文主要ID构造、FLAP/AAAI、CTEC/AAAI；用户提供的仓库E0源码与真实RUN另单独核对。
- **出版记录已确认、技术公式待核验**：LLMFolder（Hu等，EuroSys 2026，DOI 10.1145/3767295.3769339）。不可由相同作者/标题推定与TARDIS所有算法一一对应或已经超越TARDIS；取得合法全文后再修订其专门设计。
- **应用界限**：TARDIS主要LLM自回归/I/O，MPC主要分类网络，FLAP主要LLM，BRECQ主要CNN PTQ，CTEC图像扩散，RT-Lynx图像DiT；它们的精度百分比、参数减少及线性层时延与Wan的VBench/4step生成wall均不同量纲。
- **查重原则**：先利用高质量已有构造解决真实模型问题，之后依据具体、可证实贡献如实描述原创性。只因思路已出现而放弃改进是错误的；只因复现成功就宣称全新数学原理也不准确。

**本文件仅为一手文献研究与原实验解释：没有新权重拟合、模型任务、GPU/video运行；不改变ACTIVE和一次性Agent交接。**
