# FFNDS · 量化元信息的决策价值、函数耦合与真实稀疏：恢复研究路线

> **本路线后续跨线综合（2026-10-28）：** [QCFB 量化条件联合函数组束与动作依赖 Q2 研究前沿](INTEGRATED_QMETA_FUNCTION_EXECUTION_FRONTIER_20261028.md)将本文的 NetVOI、signed joint error 和 Q1/Q2 决策时点联合 lowering 为完整 HiF4 64组合法计算动作；本文保留为 v1 数学/文献起点，二者相互补充，不重复登记实测或更改 M-B B0/B1 优先级。

> 2026-10-28 · METHOD_SNAPSHOT v1 / PARALLEL I+F+C RESEARCH · ALG+SRC+LIT+OPEN。新路线恢复，不替换 M-B B0/B1 的既定执行顺序；不修改 ACTIVE_RESEARCH.json，不登记/启动 GPU、模型、训练、视频作业。本文是独立的机制与实验设计，**不是**新的原生 Wan 实测、正式同行投票或已验证新算子。
>
> 研究基础：[I/F/C北极星](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md) · [G14先验](G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md) · [G6D规模代理与审查](../method_discussion/proposals/MD-20261024-QMETA-GRAM-PROXY-G6D/DERIVATION.md) · [G6E Q2面板](../method_discussion/proposals/MD-20261024-Q2-NATIVE-SCALE-PANEL-G6E/v1.json) · [G6F K面板](../method_discussion/proposals/MD-20261024-QMETA-KPANEL-TRANSITION-G6F/v1.json) · [G11](../literature/20261027/G11_QUANTIZED_SIGN_MAGNITUDE_INFORMATION_AND_FFN_FUNCTION.md) · [G12](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) · [M-A/B/C现行方案](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)。

## 0. 新研究结论先行

**量化元信息可能有显著预测/决策价值，但不是通用稀疏的数学必要条件。** 它的候选价值在于，把原本付费才获得的当前张量结构编码成可融合读取的低成本侧信道，使不同计算动作可以在决定时间以前被选择。原先“scale → W1能量 → Top-K”是这一问题的一个很弱的统计子模型。新的中心估计量应是**同成本、同合法决策时间下，量化信号的额外策略收益（incremental decision value）**，目标同时考虑符号、门控、W2交叉项、量化误差、时序功能和物理tile。

恢复三个并行科学问题：
- **I-Q / 信息价值**：Q1码/scale及小型符号统计是否改善 W1 前的真正可执行动作决策？在条件给定普通cheap状态后，额外行动收益是否大于提取成本？
- **I-F / 联合函数**：Q1元信息与 e5 gate、W2 几何、M-B下游读出和M-A紧凑函数结合，是否预测实际对视频有害的有符号缺失而非activation energy？
- **I-C / 物理计算**：Q2码/scale是在 W1+GELU 之后才形成的，但可对 W2 完整K面板、编码域计算、group closure和精度分配形成独立动作；另研究Q1对W1的早决策与编码代数。

三者可分别给出科学发现；工程声称要求 native endpoint 和视频质量。

## 1. 数学重述：新信息 vs 低成本可访问的信息

在固定权重、量化recipe和确定性舍入下，量化码和尺度是原张量/参数的确定性函数。若完整当前输入 X、权重 θ 已经作为条件，则理想香农信息意义上

$$
I(Y;Z_Q\mid X,\theta)=0,\qquad Z_Q=Q_{\mathrm{metadata}}(X;\theta).
$$

此式**并不**意味着量化元信息没有计算价值：访存、kernel producer和数值操作时点决定读取 Z_Q 的成本可能远低于重建某个 W1 / W2 输出。随机舍入需把 RNG state 纳入条件，或独立处理随机条件信息。

定义可允许的决策信息族（保持实际运行时顺序）：
- \(\mathcal F_0\)：离线冻结权重、step/expert、已有的cheap residual/gate/conditioning或历史可用状态；具体字段逐 producer 核实。
- \(\mathcal F_1=\sigma(\mathcal F_0,Z_{Q1})\)：post-AdaLN/Smooth1 Q1编码的当前输入FP4 code、scale及事前融合得到的小统计，**必须在W1前**合法可得。
- \(\mathcal F_2=\sigma(\mathcal F_1,Z_{Q2})\)：W1、GELU、Smooth2、Q2算完之后才能观察；允许指导W2或后续计算，不能追溯指导已支付的W1。
- \(\mathcal F_3\)：完整 teacher W1/W2/head/J/未来flow属于校准 oracle，不是免成本线上输入。

给定同一候选操作集 \(\mathcal A\)（例如固定/动态保留、compact函数、W2面板省算、不同bitwidth等）及损失 \(\ell(a;s)\)，信息集的**最优策略成本**为

$$
R(\mathcal F)=\inf_{\pi:\mathcal F\rightarrow\mathcal A}
\mathbb E_s[\ell(\pi(\mathcal F);s)+\lambda C_{\mathrm{execute}}(\pi;s)].
$$

量化侧信道的在线净决策价值定义为

$$
\boxed{\mathrm{NetVOI}_{Q1}
=R(\mathcal F_0)-R(\mathcal F_1)
-C_{\mathrm{extract}}(Z_{Q1})-C_{\mathrm{route}}.}
$$

如果 \(\mathcal F_0\) 真包含全部 X，且策略和无限算力都免费，则附加确定性 Z 无香农信息增益；这里 R 的可执行函数类、读取预算和物理成本才是关键。比较相同物理动作集时，若Q侧信道只提升相关性却不能改变低成本策略，NetVOI≤0。对于已经完成 Q2 的晚决策，额外获取价值要与 \(\mathcal F_2\) 的真实producer成本分开计价。

### 不可辨识的最小反例（ALG）

一个 MXFP4 block 的两输入 c=(4,4) 与 c'=(4,-4) 可以共享相同scale、幅值多重集合和平方能量；W1权重 w=(1,1) 时，点积分别为2s·4与0。相同 unsigned metadata 不能识别这两类实际动作。若任何合格策略必须在两者采取相反决策，则**至少需要某种区分有符号方向的额外信号**；这只是该具体判别任务的必要条件，不是“所有稀疏都需要符号码”的定理。G12则给出在只见 signed histograms、未见逐坐标配对时最紧的重排内积区间。

## 2. 正常Wan路径与两类合法决策

现行源图：M≈31668、D=5120、H=13824；x∈R^(M×5120) → Q1 → W1 → GELU(tanh) → HiF4 Q2(64→8→4) → 可选alpha/BF16转换 → W2 → pre-gate y → g⊙y + residual → nextnorm/attention/head/flow，四步 high/high/low/low，两套expert×40 blocks。原HiF4路径为**fake-QDQ + BF16 GEMM**；当前CUDA QDQ的嵌套局部尺度并未持久暴露为可免费读取的metadata tensor。Packed MXFP4(32/E8M0)及NVFP4(16/E4M3局部+FP32全局，可能2D权重scale)必须独立指定recipe和平台。

| 量化信息 | 首次有效时间 | 可以影响的操作 | 不能冒充的动作 |
| --- | --- | --- | --- |
| 离线W1/W2 code/scale、范数/符号profile | 部署前 | 离线预计算、固定权重簇/函数表示/缩窄；与当前Q1拼接 | 不含当前输入的在线动态证据 |
| Q1 scale shape、码字及少量签名 | Q1 producer后/W1前 | W1合法早决策、模型库/紧凑函数路由、W1输入2:4、低成本编码求值 | 直接知道 post-GELU Q2 或未来真实后缀响应 |
| Q2 complete-group scale/code/zero/moment | W1+GELU+Q2后/W2前 | W2有效K面板省算、近似精度分配、输出编码计算 | 回溯省去W1/GELU/Q2已付成本 |
| gate/residual/step及已付费历史 | 真实生产时点决定 | 与Q1/Q2融合成conditional risk、相邻时间控制 | 未来teacher/未来step oracle |

HiF4 64组内4/8尺度耦合，之前已有单输入扰动影响全组输出的反例；因此W1前跳组时，不能用“完整dense Q2后删一列”当作原数值执行等价。对W2可在完整Q2后省掉**物理完整K面板**，准确保持其它post-Q2 operand、bias和gate，才是这个动作的局部恒等。

## 3. 新核心机制：Q-Meta → 联合有符号计算风险（Q-Joint）

### 3.1 为什么 W1 energy 回归不是最终目标

G6D的候选 \(p_g(s)=\sum_b K_{gb}s_b^2\)，在假定block间零相关、条件各向同性code时给出 \(\mathbb E\|W_{1,g}x\|^2\)。已证明：
- 若 \(s_b^2=a(x)u_b\)仅有全局幅度变化，则within-token ranking固定；
- 若 \(K_{gb}=a_g b_b+v_b\)可分，则group排序仍固定；
- 若两个组的 profile存在逐块偏序，任意非负scale shape都不能反转；
- scale、unsigned energy不能识别 signed cancellation/GELU/W2/门控/下游需求。
这为先做**scale-shape×权重profile真实rank-crossing检测**提供最便宜的判别，而不是将弱proxy相关性作为成功。

### 3.2 将动作损失移到真实写回或M-B读出

在固定完整 Q2 后、固定当前gate的**W2-only动作**中，将hidden K面板分为合法集合 \(G_1,\ldots,G_J\)。令真实post-Q2/BF16输入为 \(h_j\)，W2对应列为 \(B_j\)，选定下游读出映射 \(P_s\)（最便宜先取identity或经严格验证的 post-gate 通道中心化；offline M-B可另取原生finite/head读出）：

$$
d_{s,j}=P_s\operatorname{diag}(g_s) B_jh_{s,j}.
$$

省略集合 S 的单位置**线性读出损失**为

$$
\ell_s(S)=\left\|\sum_{j\in S}d_{s,j}\right\|_2^2
=\sum_{i,j\in S}\langle d_{s,i},d_{s,j}\rangle .
$$

给当前Q1/Q2可得的便宜特征 Z，在固定读出与gate、相同生产状态下，

$$
\boxed{
\mathbb E[\ell_s(S)\mid Z]
=\sum_{i,j\in S}\left(
\mu_i(Z)^\top\mu_j(Z)+\operatorname{tr}\Sigma_{ij}(Z)\right)
}
$$

其中 \(\mu_j=\mathbb E[d_j\mid Z]\)、\(\Sigma_{ij}=\operatorname{Cov}(d_i,d_j\mid Z)\)。恒等式的关键是**输出方向和跨组有符号相关**；分别估算单组能量再做Top-K会丢失这些交叉项。离线可以只拟合小rank共享输出坐标 \(U^\top d_j\)、条件均值和部分块间Gram，不要求在线构造D维向量/Jacobian或O(J²)完整矩阵。冻结student的实际online算子及成本后，确认该压缩统计是否比scale-only/旧linear proxy带来增量决策价值。

**注意**：这是fixed-Q2、fixed-state、线性P的条件矩公式；若在W1前选择跳算导致保留组Q2 scale随输入改变，必须对**候选真实新Q2执行**重新定义 \(d_j\)及误差，不能直接套用以上分组相加。M-B随状态变化的完整后缀J也不能当在线免费观测。末层B0/B1正好为离线下游监督提供将来可用的合法有限证据。

### 3.3 量化误差与稀疏误差不是两个独立质量预算

固定可比较读出P，令 \(\Delta_Q=y_Q-y_{\rm ref}\)、\(\Delta_S=y_{Q,S}-y_Q\)，则

$$
\|P(\Delta_Q+\Delta_S)\|^2
=\|P\Delta_Q\|^2+\|P\Delta_S\|^2+
2\langle P\Delta_Q,P\Delta_S\rangle .
$$

交叉项可正可负。**量化scale的异常、码字截断和符号结构，可能预测的是“当前还有多少可承受的稀疏扰动”，不只预测hidden activity。** 这连接QuantSparse的联合误差机制、SharQ的量化稀疏分解、M-B gate/suffix及A0量化一致的compact adaptation。研究焦点是有符号交叉是否在原生Wan中可稳定、提前预测，而不是假设稀疏只需避开大值。

## 4. 三条候选执行路径（同主问题、不同决策时间）

### Q-A：Q1早期侧信道 + 合法低成本函数库（M-A/M-B融合）

**构造**：以当前Q1 scale-shape、稀疏符号码字草图、weight-profile低rank特征、可得gate/t作为router输入，决定每个物理tile/step使用（a）固定紧凑M-A0函数，（b）另一共享参数紧凑函数/预算，（c）完整版，或一个合法W1 whole-group结构。选择动作并不是预定只丢hidden channel，也可选择**执行哪一个替代函数表示**。可训练学生中非线性和真实Q2一起定义；不要把无效的任意hidden旋转当同函数。若多套权重库搬运成本高，优先共同驻留共享basis、固定几种形状/按tile分组的动作结构。

**独特预测**：条件给定普通cheap RMS/step/gate后，scale-shape×weight-profile与少量code方向能使同成本策略优于静态A0/step-only；选择结果在原生合法K/GEMM tile上不塌缩成满计算。若只有source固定排序而无动态value，保留其**固定结构/精度**信息，动态路线转向Q2或表示层。

### Q-B：Q2后W2完整K面板 + downstream风险（G6E/M-B融合）

**构造**：Q2 producer 同时输出适合selector消费的有效group scales，必要时附带码字平方和/zero/sign小摘要；按真实post-alpha/BF16有效operand、W2权重、gate构造逐整面板保守上界或条件联合风险，再在**row tile**粒度省掉确实没有执行的W2 K面板。优先试scale-only certificate → cheap signed summary → 少量相关面板联合风险三档，记录证明覆盖率/选择率/支付成本。保守上界为

$$
\|\operatorname{diag}(g)B_Jh_J\|_2
\le\|g\|_\infty\|B_J\|_F
\left(6|\alpha|\sqrt{\sum_{b\subset J}k_bs_b^2}
+\|\epsilon_{\rm cast,J}\|_2\right).
$$

这是local写回界，不是视频质量保证；signed-risk允许更大集合但不是形式证书。**W1已全算**，因此物理节省上限只来自W2；即使理想删除一半W2算术，相等W1/W2算术预算下也最多减少两大GEMM合计约25%（忽略其余开销）。

### Q-C：量化域直接函数/前后不变部分（M-C/G12融合）

**构造**：在原有code/scale下寻找输入函数的低维充分统计、联合code相互作用、可复用bitplane/LUT或step共享表征，直接计算足够的输出功能方向，不一定通过筛选hidden group。G12的直方图区间可用于**低成本预认证**和adaptive refinement；正负码字分层展开可用于编码代数，但是完整49个符号层交叉一般只是重写内积。将码本形状和存储、multi-token聚合、scale/round、Q2 whole-group非线性耦合，以及 output readout sufficiency 纳入成本。与M-C同rank连续模型、普通FP4 GEMM及LUT-GEMM/CodeGEMM公开多token路径比较。

**融合先后顺序不强制**：可先独立发现某个有价值的量化编码算子；也可先用Q1 side-channel提升现有M-A或原G6D；还可从Q2直接研究W2，而不待M-B成功。

## 5. 2026一手论文先验：能继承什么，已被哪些工作占据

| 论文 / 原始入口 | 实际贡献 | 对本路线的REQUIRES / COMPETES_WITH |
| --- | --- | --- |
| OCP MX v1.0；NVIDIA Transformer Engine NVFP4官方格式 | MXFP4 32元素E8M0；NVFP4 16元素E4M3+tensor global，布局/weight2D可变 | 码表、尺度、producer时点和kernel读价须以选定recipe核实；格式名不能充当实测数据 |
| [Prox (2026)](https://arxiv.org/abs/2607.27591) | 有付费INT4 proxy的LLM SwiGLU前中间选择，再只算选中真实投影 | **低位proxy→动态mask→选中组**已是强先例；我们须比较净开销和Wan原生GELU/Q2 |
| [SharQ (2026)](https://arxiv.org/abs/2606.26587) | 适配FP4的在线稀疏主路径+仍然存在的dense残差，近缘Wan2.2-T2V视频 | **稀疏+低位双路**强先例；不直接证明能省W1或跳残差；大M可能被两路成本抵消 |
| [RT-Lynx (ICML 2026)](https://proceedings.mlr.press/v306/cong26a.html) | 稀疏**输入激活**2:4、精度恢复、真实SpTC执行 | 可复现的执行强对照；Q1后W1和Q2后W2须分别审，不将权重稀疏/通道稀疏相混 |
| [QuantSparse (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html) | 视频DiT量化+**attention**稀疏联合恢复 | 量化/稀疏误差交互有近缘先例；其attention方法并未验证Wan FFN Q-meta router |
| [MR-GPTQ (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b87bb4f6346d727b265088235e5bc389-Abstract-Conference.html) | 对FP4 scale量化误差和小block重处理，低位旋转/核 | 量化scale并非无偏稳定强度标签；保持校准/数值上下文与不同format分开 |
| [SpQt / CodeGEMM / LUT-GEMM](../literature/20261027/G12_SIGNED_QUANT_CODE_INFORMATION_CERTIFICATES_AND_EXECUTION.md) | 量化layout与动态跳算/有符号码表计算 | 多数强代码快路针对GEMV/小batch；Wan M≈31668必须真正大M GEMM对照 |
| [DVD-Quant (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3bd28dd5cc4e15f9e019da13cc0c4844-Abstract-Conference.html) | video DiT的自适应bit切换/旋转量化 | scale亦可用于**精度分配**，并非只能挑组；实际recipe/bit分配成本需对照 |
| [SLA2 (2026 v2)](https://arxiv.org/abs/2602.12675) | video attention可学习的sparse/linear选择与低位QAT | “选择计算方式”有跨模块先例；其attention路由仅提供结构启发，不自动迁移到FFN |
| [Deflating the Hessian (Oct 2026)](https://arxiv.org/abs/2610.11315) | 联合低秩恢复与W4A4激活噪声的rank4 PTQ | 支持**joint rather than separable residual/noise objective**的设计；未说明Q1 metadata含在线跳算信息 |
| [OrbitQuant (2026)](https://arxiv.org/abs/2607.02461) / [SyncQ-DiT (2026)](https://doi.org/10.1016/j.neucom.2026.133975) | 旋转/标准化稳定量化及step变化的异常/尺度处理 | 本路线须比较“scale变化反映可决策结构”与“scale变化只是quantizer recipe artifacts” |

阅读等级：G11/G12/SharQ的细节、作者源码与代码路径已在历史仓库分层审阅；本轮新增 DVD-Quant、SLA2、rank4 PTQ、OrbitQuant、SyncQ-DiT仅用公开论文/出版摘要与metadata建立 **LIT_ABSTRACT / OPEN_SOURCE_AUDIT** 类型近邻，不伪称已完成源码验证。G14原有92主+10补论文索引继续为权威映射；此表为针对Q线的**机制过滤**，不以论文数量替代论证。

## 6. 最小可判别实验：先判断值得读取哪些信息，再研究算子

### E0：静态判别，不需要新模型
1. 读取真实weight/量化recipe的有效W1 group×Q1 block profile \(K_{gb}\)，检查特征秩、逐块偏序、实际source尺度形状是否可能导致ranking crossing；只做数学/已具载荷的分析，不从旧scalar汇总制造Q1码。
2. 使用G6D符号反例、G12重排界及Q2 exact panel定义检查协议可辨识性；记录Q1有效输入、Q2 BF16 cast、alpha/e5读出与真实可选择的physical K/row tile。
3. 检查历史payload是否实际存储Q1/Q2 codes/scales与真实配对gate/student/teacher；仓库2026-10-24离线报告明确旧scalar JSON**没有**原Q1码字观测。若只有汇总值则列出最小 future capture producer。

### E1：信息增益试验，需独立冻结 source/资产和真实model scope后执行
独立单位为 image-source cluster，token/step/site嵌套。同source同seed的teacher state比较下列 arms，训练/特征/线上计算明确记账：

| arm | 允许输入 | 同条件比较目标 |
| --- | --- | --- |
| F0 | 固定static组和step-only宽度/精度 | 原样本基础动作 |
| F1 | cheap RMS、step、gate/残差可得摘要 | 不借量化metadata的预测 |
| Q1-S | F1 + Q1有效scale **shape**及weight-profile lowrank | 新动态排序/质量/NetVOI |
| Q1-C | Q1-S +极少signed code/occupancy/zero/m2草图 | 方向性新增行动收益 minus extra read |
| Q1-F | Q1-C +经严格对齐的**offline** M-B下游教师监督、真实student cheap输出 | 是否保留 motion/照明/外观等联合行为 |
| Q2-S | 后W1真实Q2 scales → 完整W2 K-panel bound | W2独立策略（不能与Q1算相同节省） |
| Q2-C | Q2-S + post-alpha/cast码字平方和/符号分布 | 证书更紧、物理skip更多？ |
| ORACLE | 完整真实W1/Q2/W2及下游finite结果 | **只作上界和regret**，不是online方法 |

**因果动作标签**：Q2-W2可从完整Q2 fixed state合法进行精确sum omission；Q1-before-W1必须真实重跑候选缩窄W1+新Q2路径以构造teacher同状态反事实，绝不可用“完整Q2删组”的事后标签代替。所有local响应和视频readout是不同estimand。

### E2：融合方法和部署验证
- Q-A：静态M-A0、同预算step-only compact、同预算Q1-policy compact、普通HP1/NP1 linear proxy、Prox/RT-Lynx可迁移匹配基线；报告独立来源配对regret和cost。训练预算与双expert共享同权重的step处理必须一致。
- Q-B：Q2 upper-bound vs cheap-code-bound vs signed-joint-risk；直接比较保留physical tile占用、有效省MAC、真实kernel开销、full Q1/W1成本；若仅选出token孤立mask而K-tile仍全满，视为没有取得当前硬件路线上可兑现的skip。
- Q-C：普通FP4 dense / BF16 fake-QDQ / 码book变换 / 同rank连续student；逐台硬件、M、权重resident、数据搬运、scale swizzle、quant prepare、GEMM/loss/all kernels核算。
- 最终经过**相同源、四步、80参数FFN×160调用**自由生成，逐source评价主体/背景/细节/光照、video motion幅度与方向、运动平滑、语义动作及真实wall/显存，不把UB1的已暴露20来源视频作为确认集。

### 判别结论与后续研究分叉（在看结果前冻结）
- **scale-shape排序不变**：将Q1-S作为固定特征/精度分配对照，转向cheap signed sketch或Q2完整面板；不是终止全部quantization路线。
- **Q1-S增强排序但没有净策略收益**：先减少读价和router代价；若仍无净值，将它保留为representation机理而非部署方法。
- **Q1-C存在source-heldout净行动收益但physical tile密集**：优先M-A compact函数库、M-C编码和tile-aware aggregation，不反复试token Top-K。
- **Q2准确判别whole K panels且producer read可融合**：可以独立获得W2-only方法；仍需独立四步视频与wall实测。
- **Q1/Q2局部SSE较好，运动却衰减**：与M-B B0/B1及UB1竞争机制合作，区分下游paired J、student-prefix drift、纹理影响RAFT，避免把局部风险等同视频运动。
- **真正native packed FP4 dense比候选快**：记录有效的计算表示/信息结论，工程路线转为better producer-consumer fusion或可重用函数表示。

## 7. 方法实现对象与数值合同（供未来Agent）

| subsystem/module/operator | 输入、shape/semantic | 输出/不变量 | 可替换的前提 |
| --- | --- | --- | --- |
| Q1_METADATA_PRODUCER | 同Wan Q1真正postnorm/Smooth输入、chosen quant recipe | 原Q1量化有效codes/scales及shape和读价；M×(D/B) | QDQ元数据若未expose须付额外materialization；packed与fakeQDQ独立 |
| STATIC_WEIGHT_PROFILE | W1/W2真实effective decoded parameter / B/group layout | 共享W1 block-profile / W2输出几何或简化Gram | 2D scale/weight rotation/bias/Smooth/alpha需要同版本 |
| CHEAP_SIGN_SKETCH | Q1已经存在码/scale；固定O(r) sketch | 实际signed summary；forward因果不含full W1 | 实测读码/packing/可融合性 |
| Q1_ROUTER | cheap state+Q1-S/C，token/row tile × step/expert | 合法学生函数/算子选择及决策成本 | 所选函数真实Q2组、所有native输出与权重resident合法 |
| Q2_PANEL_PRODUCER | 原W1/GELU/HiF4 Q2及alpha、round | 同一W2有效operand上的scale/block摘要 | 不用Q2候选提前省W1 |
| Q2_W2_DISPATCH | Q2 scales/codes、gate、W2 geometry；row tile×physical K | 跳过真实W2面板的执行与局部误差 | 只在完成W1/Q2后选择，bias与保留面板不受改变 |
| FUSED_DOWNSTREAM_READOUT | 真实e5 gate、固定readout、可选离线B finite oracle | conditionally meaningful joint risk | 训练期oracle与上线readout严格隔离 |
| FULL_VIDEO_VALIDATOR | fixed model revision、seed、source cohort和student free-run | source-level视频质量各轴+实际wall/VRAM | 不把teacher-state local替代terminal |

**数值等价分层**：real decoded FP4算术恒等、BF16操作及累加顺序近似、fake-QDQ→BF16 GEMM、packed MXFP4、packed NVFP4分别标注；Q2 local scale变动会影响同一64组的所有输出。真正下游改变以完整BF16原生程序观测，不以一个Euclidean proxy证明视频等价。

## 8. 可复用资产计划（REUSABLE_ASSET_PLAN）

记录的source、模型revision、quant recipe、Q1/Q2 producer、expert/step/block/token axes、frozen arm、cost scope与指标对齐不可省。下面的资产全为**PLANNED / NOT_PRODUCED_IN_THIS_TURN**。

| asset_id | 意义与producer | format / shape / locator proposal | 复用条件 / 失效条件 |
| --- | --- | --- | --- |
| QMETA-OPCONTRACT-v1 | 原生Q1/Q2/cast/gate owner与成本证据；SRC check | Markdown+JSON；research/quant_metadata/.../op_contract | runtime/recipe/pack布局或模型变化即复核 |
| QMETA-WPROFILE-v1 | 权重profile rank/dominance和normalization | arrays：expert×block×output-group×Q1-scale-block；来源与权重版本索引 | quant decoded weight/Smooth/Q1 block变化失效 |
| QMETA-Q1PAYLOAD-v1 | source-paired真实Q1 code/scale/cheap-sketch | source×step×expert×block×sampled token，packed & scale schema | 源/seed/step/coords、时间合同不一致禁止共用 |
| QMETA-Q2PAYLOAD-v1 | true after-alpha+BF16 W2 input与Q2 metadata | source×step×expert×block×token×Kpanel | Q2/alpha/cast/stochastic recipe变化失效 |
| QMETA-ACTION-LABELS-v1 | 同状态各合法动作的local BF16+gate响应和metadata | row×action×site×readout / nested cluster | Q1/Q2动作的因果时点不一致禁止混用 |
| QMETA-POLICY-COST-v1 | selector+quant+physical tile occupancy与matched baseline | arm×M×device×kernel/version times | 设备架构/packed策略/运行batch变化不外推 |
| QMETA-FREEVIDEO-v1 | source级配对生成、各轴质量和wall | MP4、source×seed评测JSON、环境记录 | 训练/校准曝光或参数变化须重算 |

验证状态：**本轮只完成ALG+仓库SRC+一手LIT归纳；没有新增NATIVE/SYS/VIDEO。** 尚未声明metadata sensor可免费提取、方法获独立支持或优于dense。

## 9. 同行协作与优先级

- 本路线以 **I-QMETA / Q-JOINT v1** 的**平行探索**身份恢复，可由另一方法讨论Agent独立提交正式proposal并由不同Agent审阅；本文不代替他人的SUPPORT。
- 保持 [M-B B0/B1正式交接](../agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)为当前已指定的方法执行首项。Q0 (static producer/weight profile)、Q-Joint 数学和prior-art可与B0并行；Q1/Q2实际模型实验须在ACTIVE独立冻结合法scope后执行。
- 优先级 **Q0/信息可用性与rank collapse → Q1 heldout NetVOI与Q2合理界 → Q-A/Q-B/Q-C择优融合 → 完整学生自由视频+native wall**。这里的箭头表示代价最小的判别顺序，不约束其它Agent独立科学发现。
- 研究声明层级：**MATH**（上面代数、必要条件/反例）；**SRC**（已核当前HiF4 W1/Q2 timing）；**LIT**（相关论文已提出具体机制）；**OPEN**（Wan端真正的增量decision value、signed covariance、kernel speedup、视频泛化）。
