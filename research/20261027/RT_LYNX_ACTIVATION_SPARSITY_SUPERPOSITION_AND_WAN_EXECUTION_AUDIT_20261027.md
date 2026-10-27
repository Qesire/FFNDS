# RT-Lynx 深读：从权重能力、激活使用到可执行2:4稀疏的FFNDS研究重构

**2026-10-27 · ICML2026一手论文/原生Wan证据/数学判别 · LIT/SRC/NATIVE/ALG/OPEN分级。**  
**后续同方向原生算子进展（不回填本文原始文献复核状态）**：[E0 A100激活2:4布局对照](E0_ACTIVATION24_BACKEND_LAYOUT_RESULTS_20261027.md)已在合成BF16算子证明某些**预先选择、压缩**后的布局快于dense，同时默认布局也可更慢。它未测实时在线top2＋压缩全成本、真实HiF4 FFN、全部生成或视频质量；因此R0→R1的下一有效研究是核对Q1后W1、Q2后W2两种实际输入边界，并在匹配精度与best dense/packed基线上计时端到端的选择＋压缩＋SpMM＋QDQ。[跨路线表](PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)保持这一C路线与UB1/Affine54的科学/系统对象清晰分离。

**版本证据**：直接检查 [RT-Lynx arXiv 2605.26632 HTML（已读可访问版本v2）](https://arxiv.org/html/2605.26632) §2–5、Appendix B/E/F/G/H，以及 [ICML2026 PMLR正式页](https://proceedings.mlr.press/v306/cong26a.html)；[arXiv abstract及LaTex正文视图](https://arxiv.org/abs/2605.26632)提供原式、修正数字。arXiv页面还显示2026-08后续版本v3，但本轮未逐公式与v2作差分；PMLR正式标题是 *Putting GEMM Sparsity in the Right Place for Diffusion Models*。未找到可靠的作者可复现CUDA代码仓库，不将正文kernel叙述冒充亲自复现。FFNDS原生路径及观察引用对应已存实验，**无新FFNDS模型/训练/视频/GPU执行**。

## 0. 最重要的科学发展：模型长期的计算潜能 ≠ 该输入调用的计算

RT-Lynx的最强经验发现，不是简单的「激活有更多零」：**对其Qwen-Image等预训练图像DiT，逐权重强制2:4容易损伤能在不同输入被调用的函数容量，而逐token激活强制2:4更易保留当前输出；后者还必须结合合理的范数恢复、轻训练及其硬件算子，才能使画质—速度同时有效**。

它启发FFNDS将三种对象明确拆开：

1. **长期能力**：原始稠密W1/W2储存跨输入/条件/时步可能需要的完整功能库。
2. **当前实际工作**：给定真实postQ1及postQ2、step、expert、gate的GEMM输入，哪些乘加贡献到当前FFN输出？是「此输入可以省」而非「此权重永远不重要」。
3. **物理执行单位**：RT-Lynx逐输入2:4 activation operand由Sparse Tensor Core处理，**与UB1/早期E0沿原64组删输出hidden整组是不同算子/不同硬件稀疏定义**。

这是研究架构上的区分，不代表已证明Wan具备RT-Lynx的可行质量或速度。用户关心的「解释性」应被落实为正常计算在不同输入的功能承担与可干预因果作用，而非把neuron magnitude直接命名为semantic feature。

## 1. 论文实际上执行的算子

设每层输入 \(X\in\mathbb R^{M\times K}\)，保持训练好的稠密 \(W\in\mathbb R^{N\times K}\)，每个token输入行按连续四元组取绝对值最大的两个，定义\(S_{2:4}(X)\)。RT-Lynx算法1：
\[
\bar X=S_{2:4}(X),\quad
s(x)=\sqrt{\frac{\|x\|_2^2}{\|\bar x\|_2^2+\epsilon}},\quad
\tilde x=s(x)\bar x,
\]
\[
\widehat Y=\tilde X W^\top + X(L_A L_B)^\top,\quad
L_A\in\mathbb R^{N\times r},\quad L_B\in\mathbb R^{r\times K}.
\]

- 前项 **在线activation 2:4 sparse GEMM**、norm compensation；权重W保持稠密，这是*输入稀疏*，不是W2永久删列/输出剪枝；LoRA支路在密集Tensor Cores上处理**原X**，而非只预测异常索引。
- LoRA固定预训练骨干，在作者图像DiT数据做Teacher–Student微调；主要训练设置 **20k prompt–image样本、约2k步、rank64**；在线不是训练free的最终方案。Z-Image与FLUX single-stream某些异常敏感projection暂不做稀疏（原文称 selective layer skipping：跳过的是*稀疏化*，不是删除整层）。
- 专用CUDA把online Top2、norm/压缩/2:4 SpTC-compatible layout、sparse GEMM、dense LoRA回写尽量融合；未经融合的online稀疏化成本约16%–59%的相关路径，不可视为免费。
- Qwen-Image基线覆盖QKV、attention Out、MLP Up与Down的image stream；不是仅FFN W2，也不是逐token删整个neuron导致W1/W2同时缩窄。在不同模型/单流部分projection选择不稀疏。

**数字定位（Qwen-Image，MJHQ，FID越低越好）**：Full **21.98**、naive sparse weight 2:4 **51.63**、naive sparse activation 2:4 **35.85**、SA+norm **25.28**、SA+norm+LoRA **21.25**。仅激活2:4也有显著损害；「2:4 activation天然无损」是误读。Qwen-Image另有weighted pruning成熟对手：Wanda40.81、RIA43.02、BaWA39.68、Slim22.25；最后的Slim带LoRA，不应用naive weight组概括所有权重压缩。

**速度身份**：作者H20线性层平均最高约1.55×，在线稀疏GEMM kernel指定形状最高1.88×，模型完整图像生成约1.2×；原论文例Quant(Qwen-Image)基线60.66s，RT-Lynx50.60s，W8A8 54.45s，联合46.04s；Z-Image 8-step与其稀疏方法组合的4.99s→4.17s。都是**作者图像DiT/H20数据**，不代表Wan2.2四步/A100+HiF4有相同比例或视频质量。

原文已显示稀疏激活丢弃的低幅值部分对发丝、轮廓及纹理可能关键，norm/LoRA的重要性直接体现在逐项消融；某些难补层要保留原Dense计算。其解释优于「激活都可以安全清零」。

## 2. 论文的解释性/超位置叙事：证据、假说及强反例

原文§3.1：Qwen-Image权重统计近准高斯且未训练满足2:4；在强制2:4剪重下质量下降。§3.2：token activations明显集中于零附近（作者给出约**5–10% active**，是作者使用的near-zero/active统计，不是原生严格零率），并将其与Transformer在特征superposition下不同token调用不同特征联系起来。

应当区分：
- **LIT作者量测**：幅值分布及2:4剪枝误差/画质比较，属于本文实证。
- **解释模型**：superposition可能提供一个机制直觉，但「每token少量高激活**neuron**」≠「每token只使用少量**语义feature**」。Superposition的原始论证中，多语义feature可共享单个neuron/维度；feature sparse不保证坐标上的neuron sparse，除非表征基与任务分布满足附加条件。比较必要时可用可解释字典/SAE/低秩或因果patching，勿把“单坐标近零”作为语义证明。
- **weight不宜稀疏的精确范围**：只针对**既有稠密预训练权重、未经结构适应且强制2:4权重元素掩码**的严峻困难。不是对所有可学习稀疏权重、结构剪枝、蒸馏/再训练后稀疏、功能折叠的否定。反例为 [Gao等 *Weight-sparse transformers have interpretable circuits*, ICML2026](https://proceedings.mlr.press/v306/gao26w.html)：**从训练阶段约束稀疏权重**能获得更可解释电路，但付出能力—可解释性权衡；其研究对象和RT-Lynx对预训练DiT强制2:4完全不同。
- **更基础的一手解释框架**：[Elhage等 *Toy Models of Superposition*](https://www.anthropic.com/research/toy-models-of-superposition)，讨论feature sparse如何通过重叠表征储存超维特征及其非线性干扰；不能从这里直接推导Wan的token-neuron稀疏百分比或行为语义。

**面向FFNDS的新可证伪问题**：不改权重时，某个神经元/量化组/功能方向是否在source、frame、step之间呈“稀有但重要”的使用？若固定剪去它，哪些之前不出现的功能无法召回；若只在当前输入压缩且保留权重，它在下次条件改变时是否可以重新激活？用真实正常生成与受控干预区分「难以重要」和「只是当前不需要」。无标签时可以只检验有符号后缀函数方向，解释性意义不取决于给每个神经元指定自然语言概念。

## 3. 与Wan2.2 HiF4计算图的两个完全不同的2:4插入点

完整路径 \`norm2/AdaLN→Smooth1/HiF4 Q1→BF16 W1→GELU→Smooth2/HiF4 Q2(64→8→4)→BF16 W2→e5 gate/residual\`，M≈31668、D5120、H13824。

**S1：在Q1后、W1 GEMM输入上对X执行2:4 activation sparse**：
\[
\tilde X_1=S_{2:4}(Q_1(x)),\quad
\widehat U_1=\tilde X_1 W_1^\top.
\]
潜在节省W1的部分计算，但保留全部W1行/输出hidden维度和权重潜能；改了preactivation，因此下游GELU/Q2尺度与W2全部响应都会变化。输入Q1已完成，而norm rescale在Q1之后可能改变原量化可表示点，须明确乘法/cast顺序；若先对Q1之前压缩还会改变Q1共享量化标尺，应单独定义。**此动作并未因原whole64 hidden-group覆盖216/216而失去SpTC 2:4执行机会**。

**S2：在Q2之后、W2 GEMM输入上对已真实量化的post-Q2向量A执行2:4**：
\[
\tilde A=S_{2:4}(Q_2(\operatorname{GELU}(W_1Q_1(x)+b_1))),\quad
\widehat U_2=\tilde A W_2^\top.
\]
这种定义固定本轮完整W1/GELU/Q2已计算的值，**只节省W2 GEMM**，符合Q2原尺度闭包，因为mask在原Q2完成后施加；规范化系数若改变q值需计cast、误差，不可声称保留原原生Q2量化等价。它与先删hidden group省W1+W2的UB1、Affine54 A+真实非线性组是三类不同的实际动作。

**S3：组合S1与S2**：若同时改变W1输入，S2必须使用**新W1/GELU/Q2真实输出**，不能套用Dense保存的旧post-Q2；同时稀疏两个GEMM可得较大潜在计算减少，但状态误差、量化scale及额外LoRA要一起计。

**Q2尺度副作用的源证据**：[原始M的Q2补集](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)显示，修改Q2前某单坐标可能引起补集中其它坐标变化，所测最大坐标的补集/直接范数13.36%–23.63%。这不是对S2的反例：S2在Q2后处理现值，不重新标定其补集；S1或preQ2稀疏则必须检验合法尺度传播。

## 4. 先前“100% tile满占用”的结论需要按物理动作重新解释

[E0原生完整hidden](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)得到：以逐token后验TopK的**原始任意hidden坐标/原64通道组**为对象，M16×K64以上原layout tile近乎/完全全满；真实postQ2严格零率约6.115%–13.441%，这没有带来whole64空块。与RT-Lynx原文约5–10%“有效激活”的near-zero标准不是同一个统计对象，不可直接比较其百分比。

**RT-Lynx的2:4天然在每4个坐标局部留下2个，Sparse Tensor Core按内层结构化编码跳过指定乘法，因而即使一个大M×K64 tile始终非空也可以执行2:4稀疏运算。** 该路径并不依赖某64组在所有token都为零，也不需要全体token共享同一hidden子集。

硬件/信息分层：
- 对UB1/Whole64：真实缩窄W1/W2需要选择合法64组并减少矩阵形状。
- 对后验token TopK/gather：若仍用普通稠密GEMM，100% tile占用是强负例。
- 对2:4 activation+SpTC：性能条件是稀疏操作数的压缩方向/对齐/布局、每四取二、线上选择/元数据、A100 BF16支持与融合真实实现，而不是原whole64 tile空不空。A100具备Sparse Tensor Core硬件，但**作者H20源码/速度不能直接迁移或证明我们已有可用A100 activation-2:4 kernel**。
- 对真实HiF4：Q1/Q2 fake-QDQ和Sparse Tensor Core 2:4 BF16/FP16属于不同表示；同时把原生packed FP4、原模型QDQ、2:4激活SpTC合并为“2×”需要实际硬件调度和数据格式证明。

**计算费用下界提醒**：S2即使理想0成本削掉全部W2，原标准FFN的W1和W2主GEMM各占一半，故整个主FFN线性MAC最多节省50%；现实2:4使W2乘法理论省50%，只对主W1+W2合计省**25%**，还未扣norm、metadata和LoRA。S1与S2都2:4时理想主乘法减50%，但不是在其它一切不变下整个Wan提速2×；与UB1全组剪裁主MAC减50%只是**相同分母下的算术比较**，不等质量/面积/速度保证。

## 5. 为何“补偿”更值得研究，而非立即强行稀疏？

对给定真实W2输入\(a\)，一组2:4激活\(m(a)=S_{2:4}(a)\)加norm缩放\(s\)后，未经LoRA的精确当前输出误差为
\[
e_y=W_2(a-sm(a)),\quad e_g=g\odot e_y.
\]
**输入范数保持不能保证输出方向、signed抵消或某个行为方向保持**。最简单确定性反例：\(a=(1,1,1,1)\)，任选同一2:4规则只保留前两项，则\(sm(a)=(\sqrt2,\sqrt2,0,0)\)；若\(W_2=(1,1,-1,-1)\)（单行），dense输出0、norm-comp sparse输出\(2\sqrt2\)。这是数学反例，不是Wan测量。若输入处于超位置混合基中，单坐标幅值更不足以说明功能保留。

所以可直接复用作者三项成熟恢复机制，但用已知Wan作用对象评价：
1. **Norm compensation**：可作为低成本无梯度对照，不先验认为最佳。比较无缩放、原文每token \(\ell_2\) 缩放、若确有证据才考虑gate/输出感知缩放。
2. **LoRA低秩恢复**：以dense当前输入预测稀疏W2失去的输出，rank与训练集大小控制；作者Qwen-image rank64/2k步及其训练数据规模是图像模型参考，不可直接把24source/256row视为足够Wan视频后训练。也可以考虑只保留原UB1 R256机制作为对照，而不是默认共用补偿有效。
3. **Selective sensitive layers kept dense**：作者在FLUX/Z-Image单流路径跳过特定projection稀疏。对Wan可以先识别哪种(W1/W2, high/low, block, step)强2:4更脆弱，使用同成本更均匀计算分配，而不重新叫“新”方法。
4. **视频行为恢复**： [UB1独立视频运动幅值](UB1_DYNAMIC_DEGREE_RESULTS.md)中两历史来源虽Aesthetic/Imaging/Motion三指标接近，但UB1/ Dense平均RAFT高5%光流幅度分别为38.1%/9.19%，二值dynamic-degree全0。RT-Lynx中LoRA恢复图像边缘纹理不保证Wan恢复视频运动量；也不能从这两例推断UB1全部视频动态减弱。跨step控制可直接参照CTEC与已测finite P/N。

## 6. 一手先例与反解释如何帮助提出新研究

| 原论文 | 吸收的机制 | 我们需要保持的不同身份 |
| --- | --- | --- |
| [RT-Lynx ICML2026](https://proceedings.mlr.press/v306/cong26a.html)／[含原式的arXiv](https://arxiv.org/abs/2605.26632) | activation-input而非weight 2:4、norm、rank64 LoRA、kernel融合、selective layer | H20、FP16图像质量与Wan HiF4/BF16四步视频不同，不预设必须超越它的算法 |
| [Toy Models of Superposition, Elhage等2022](https://www.anthropic.com/research/toy-models-of-superposition) | feature稀疏、混叠、干扰与非线性筛选 | 不直接推出neuron语义或者Wan的5–10% feature数量 |
| [Weight-sparse transformers have interpretable circuits, Gao等 ICML2026](https://proceedings.mlr.press/v306/gao26w.html) | 训练期权重稀疏可能提高解释性 | 反驳“权重必须稠密/稀疏一律不可解释”，并非RT-Lynx对post-hoc预训练DiT剪权结果的反例 |
| [Accelerating Transformer Inference and Training with 2:4 Activation Sparsity, Haziza等2025](https://arxiv.org/abs/2503.16672) | 激活2:4不是RT-Lynx首创；依赖对应activation/硬件 | 主要LLM squared-ReLU背景，不等于Wan tanh-GELU+Q2 |
| [Activation Concentration 2606.00567](https://arxiv.org/abs/2606.00567) | 逐元素集中与按列物理节省有鸿沟，按模型/模态不同 | 该工作研究column skipping/模拟；RT-Lynx的2:4 SpTC正是另一种work unit |
| [WINA ICLR2026](https://www.microsoft.com/en-us/research/publication/wina-weight-informed-neuron-activation-for-accelerating-large-language-model-inference/) | 幅值+消费者权重列范数改进激活选择/误差界 | 仅是LLM基准与另一评分，若用于Wan仍先查实际量化/算子价 |

强竞争解释：
- H-A（当前激活容量）：「当前token只需少数近零外的元素」，S1/S2降低瞬时函数误差；
- H-W（长期能力）：「原权重维持跨source/seed/step的长期罕见函数」，固定删权造成新来源尾部恶化，在线稀疏保留潜能；
- H-Q（量化和signed敏感性）：「当前幅值很小的值经W2/gate/Q2与下游有重要作用」，规范化或简单top2无法保持，需要精确插入点/低秩/敏感层；
- H-C（真实Kernel）：「尽管当前激活可丢，查找/压缩/metadata/QDQ使物理收益为负」，需要真正可用的A100核与端到端计时。

四者可同时成立；不把「解释性」转成未经干预的语义故事，也不把「权重稀疏不能做」作为永久政策。

## 7. 修订后的研究路径及最小辨别工作

**R0：SOURCE/ALG无需新GPU的既存证据登记**  
定位Wan中S1和S2的原始真实输入边界、是否保存Q1/postQ2全向量（不同来源/seed/step/block）、R256/LoRA合法叠加位置、原packed/SpTC 2:4硬件与编译器支持。已有[E0全hidden](../20261025/E0_INTRINSIC_FULL_HIDDEN.md)与[连续token组/tile](../20261025/E0_SHARED_TILE_PRE_W1.md)、[原M Q2补集](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)，可以先重新解读其统计：何者证明原布局满tile、何者真的证明原2:4逐组Top2作用。**没有保留后验完整Q1/Q2值时，不用标量JSON伪造新2:4统计结果**。

**R1：不同对象的正交对照（条件许可再执行）**  
在同一合法BF16矩阵输入上，用相同2:4和匹配代码计算四臂：Dense \(XW^T\)、post-hoc weight 2:4 \(XW_s^T\)、activation 2:4 \(S(X)W^T\)、activation+norm \(sS(X)W^T\)。分开W1输入、W2输入，不把后Q2选择等同preW1；记录token组直接MSE、signed gate输出、各source/step/layer的尾部相对误差。说明weight 2:4是**新引入的固定强制对照**，与UB1整组删神经元完全不同。既有成果足以支持动机，不因此先要求全视频。

**R2：解释性/条件性功能使用（可用现存载荷或合法新观察）**  
同组输出信息测试不同source/seed/frame/step下的稀有激活、激活集合稳定性、静态weight/hidden组删除的最差尾部差、按当次input恢复的条件支持。若要声称概念/功能因果，需要在不改变正常输入路径的充分对照下区分W2 signed贡献、Q2组尺度、BF16写回及实际下游行为，F线可使用G13 O/D/Q/P，但不能要求I/C首先找到语义。

**R3：论文已有补偿的迁移与能力分析（后续授权）**  
先照搬norm-comp、LoRA rank64与sensitive-layer keep-dense为成熟基线，再决定是否值得引入gate-aware、conditional high/low、step预算或精确Q2补集。分开label来自Dense同状态局部还是teacher full-rollout，校准身份、source数和训练预算与UB1/Affine54分别记录。最有帮助的可能是一个**直接有效的已有方法迁移**，不设置必须不同/必须再加模型的门槛。

**R4：物理kernel与真实视频质量（后续授权）**  
对可部署候选在A100的native 2:4 BF16 operand方向、LoRA fusion、norm/packing、HiF4 QDQ及内存读写逐项计时；对照resident Dense、UB1、Affine54、H20作者结果仅作外部背景；同模型/seed/来源/全40层四步完成真实视频，三项历史VBench+Dynamic Degree连续幅度+Subject/Background/Temporal/ROI并列，避免只看单个motion smoothness或二值阈值。全部计算资源按当前[ACTIVE](../ACTIVE_RESEARCH.json)中已冻结范围执行，本研究文档本身不发放新GPU/训练许可。

**路线评价**：RT-Lynx是原FFNDS C线的**新算子族**，并为I线提出「输入功能调用vs长期函数能力」的新问题，为F线提出有符号功能与时空运动量评估。它不替代UB1与Affine54的研究方案，也不要求新方法一定是2:4；固定组、仿射折叠、动态激活、weight retraining是可互相比较的真实不同计算合同。

## 8. 结果、限制与skill审查

- **可实质采用的科学认识**：激活稀疏 vs 权重稀疏须固定相同2:4算子、压缩层与其它恢复手段作对照；近零不是严格零/语义token独占；weight不适合强制2:4是具体后训练稀疏命题；现whole64 full-tile负例不能排除SpTC N:M。
- **必须坚持的系统认识**：RT-Lynx原生只做2:4 GEMM输入operand；当选择postQ2时，W1/GELU/Q2仍全付费；只有实际kernel、observer、norm、LoRA和完整视频确立部署Pareto。
- **新方向可以来自任何I/F/C入口**：原论文精确复现、已知norm恢复、选位置、联合码本或超位置启发的函数重组都应凭实测推进，不以刻意不同取代有效已有方法。
- **可用文献完整性**：原论文已读arXiv HTML正文/附录与正式PMLR摘要；作者核实现成CUDA仓库未取得，尚未在A100复现。ICML Gao 2026与Toy Models作为不同问题的反例和机制背景，不作为已证Wan interpretability实测。
- **当前执行范围**：本次GitHub文档研究与静态审查；无新原生2:4实验、HiF4/Wan模型运行或视频/正式打分；已完成旧资产由原RUN SHA支持，用户之前恢复的既定范围按ACTIVE另行登记。

**Skill顺序复核**：NOTICE=权重长期能力/当前激活使用矛盾；MAP_FLOW=Q1/W1/GELU/Q2/W2/caller；DERIVE=两插入点与signed反例；EXPAND=静态/条件/2:4/仿射/whole64/新代码域；DISCRIMINATE=R0-R2已有证据和最小对照；BUILD_COMPARE=R3-R4成熟补偿/物理系统；RECORD=现存RUN不改、OPEN不伪称NATIVE。采用FFNDS领域Skill及AutoResearcher mechanism-research、research-expansion、problem-reformulation、web-research-discovery审查；准确借鉴先例，不刻意否定其它路线。
