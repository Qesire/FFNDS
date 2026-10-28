# FFNDS · 量化侧信道 × 有符号联合函数 × 真实可执行省算：综合研究前沿

2026-10-28 · INTEGRATED_RESEARCH_FRONTIER v1 · ALG/SRC/LIT/OPEN。本文整合现行 M-A/M-B/M-C、Q-Meta/NetVOI、G6D/E/F、G11/G12/G14、A0 与 M-B 四轮评议；是研究机制扩展而非替换现行 M-B B0/B1 优先级。未运行新 Wan/GPU/训练/视频/性能实验；不变更 ACTIVE_RESEARCH.json、独立Agent评议票或旧 RUN。

## 0. 新中心：动作不是“删最弱神经元”，而是省掉可预测、互相抵消且能够物理整块执行的函数束

科学问题：在可访问的 Q1 量化信息下，是否能预测完整 HiF4 64 组的**联合、有符号、门控后及后缀加权的功能贡献**，把低联合风险的多个组**一起**在 W1 之前免算，同时不牺牲视频的真实动作与时间行为？新候选称 QCFB（Quantization-Conditioned Functional Bundles）；它是待判别的机制种子，不是已经成功的方法。

三种科学轴可独立产生成果：I 信息价值（量化 metadata 是否比同成本 RMS/step/upstream state 更能决定动作），F 函数价值（高单组能量但低联合输出/轨迹风险是否存在），C 物理价值（早决策、合法 whole64 组合、row-tile/packing/kernel 是否真的减少了 W1+GELU+Q2+W2）。不需要所有路线都服从 QCFB；Q-STAGE、Q-CODE、M-A/M-B/M-C 保持相对独立和可组合。

## 1. 原生对象与明确先验

原对象 LightX2V Wan2.2 I2V A14B，四步 high/high/low/low；两 expert，每个40 block；每次全模型为 160 个按 step×block 的 FFN 调用。D=5120,H=13824，216个连续完整64 hidden组；流程：post-AdaLN/Smooth1 → Q1 fake-QDQ → W1 → tanh-GELU → Smooth2/HiF4 Q2 64→8→4 → alpha/BF16 cast → W2 → e5 gate + residual → downstream LN/attention/head/flow → student full rollout。原 HiF4 执行是 fake-QDQ + BF16 GEMM，不是原生 packed FP4 Tensor Core。MXFP4 的32/E8M0与NVFP4的16/E4M3局部+全局、不同 weight scale layout 都需要独立数值和成本协议。

已有仓库 R12 用真实HiF4 Q2源码与Python reference核验：**完整ordered64组**及完整行的选择/重排在其固定条件下可交换；组内单坐标选择则不具备此性质，且现存Q2完整形状对照发现13.36%–23.63%的补集尺度作用。来源：research/20261022/R12_HiF4_group_separability.json、research/20261027/E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md。

完整组选择矩阵 S 满足 Q2(S h)=S Q2(h)，前提是各64组量化scale独立、组内相对顺序保留、Smooth2/bias/W1/gain/alpha/cast随组正确选择、不改变其它组的输入与数值recipe。对固定同一 post-Q1 x，定义有效post-alpha/cast组系数 h_j(x)，B_j 是对应W2列，v_j=B_j h_j，则**实数decoded逻辑层面**：

$$ y_Q(x)=b_2+\sum_{j=1}^{216} v_j(x),\quad y_Q(x)-y_{Q,S}(x)=\sum_{j\in S}v_j(x). $$

因此对于**整个hidden64组的提前删算**，原teacher完整后Q2的组贡献可合法用于同输入、同state的离线动作标签；真正在线仍必须在W1之前用前置信息选S。W2 GEMM的BF16归约次序、动态global FP4 scale、重排与不同format会破坏严格bitwise等价，须独立native neutral检验。这个等式不适用于Q1后删除W1输入K面板：其结果改变其他W1行的GELU/Q2 code和scale，完整Dense Q2只可当后验oracle，不能充当动作后实际Q2元信息。

## 2. 由单组分数转为**条件联合组束**

取当前有限原生状态、固定gate与冻结线性readout P（初步使用identity；仅当M-B条件证明真实后缀的LN常量规范时采用postgate channel-centering，未来真实finite/J仅做离线监督）。定义组有符号readout贡献 d_j=P diag(g) v_j。对省略向量 m_j∈{0,1}，固定Q2整组合法动作误差为 −sum_j m_j d_j；故

$$ \Gamma_{ij}(Z)=\mathbb E[\langle d_i,d_j\rangle\mid Z],\qquad
\boxed{\mathbb E[\|\Delta_S\|^2\mid Z]=m^\top\Gamma(Z)m,\quad\Gamma(Z)\succeq0.} $$

Z是**真正W1前可用且计成本**的Q1 side-information（例如scale-shape + few signed code moments）、step/expert与已有状态；不是当场读取完整teacher v/J。真实物理row-tile共享mask时，Gamma必须汇总tile内全部rows对应读出，不能把逐token的可跳率当作physical K/M tile跳率。

两个组同时省略相对单独误差之和恰多 2Gamma_ij。负offdiagonal意味着共同删除时有符号抵消；**被单组能量TopK保留的高能组也可能成对可删**。此事实是普通二次型代数，不据此宣称发现了真实Wan组束或数学首创。

一个反例说明为什么unsigned scale只可能是第一层代理：两个tanh-GELU输出取 d1=GELU(x1)、d2=−GELU(x2)。x=(1,1)与x'=(1,−1)可有相同unsigned code magnitude/scale，但前者 d1+d2=0，后者 GELU(1)−GELU(−1)=1（由tanh-GELU奇偶恒等式）。若真正最优动作不同，单靠scale与unsigned能量不可辨，至少需要便宜的符号方向或其它已付信息。例子是连续代数反例，不冒充已经通过原HiF4 Q2的64组实例。

### 把量化误差变成组束决策的“影子价格”

定义同冻结readout的量化误差 e_Q=P diag(g)(y_Q−y_ref)，并令 d_j=P diag(g)v_{Q,j}。联合省算的相对于**高精参考**误差为 e_total=e_Q−sum_j m_jd_j，得到条件精确恒等：

$$ \boxed{\mathbb E[\|e_{total}\|^2\mid Z]=\mathbb E[\|e_Q\|^2\mid Z]-2b(Z)^\top m+m^\top\Gamma(Z)m},\qquad
 b_j(Z)=\mathbb E[\langle e_Q,d_j\rangle\mid Z]. $$

b表示量化误差与要删组有符号贡献的**条件交互**，而不只是两个独立的误差预算。b可能使特定删算降低局部量化失真；这必须与原量化dense的视频目标、source-level长程质量另行比较，不能由local BF16/MSE改进推断视频等价。它区别于SharQ的两路FP4 GEMM残差恢复和QuantSparse的attention联合重参数化，真正作用在whole64 W1/GELU/Q2/W2组动作；然而“联合量化与稀疏”本身已经有先例，尚无经实证证立的独有论文创新。

**可执行动作优化目标**：

$$ m_T^*(Z_T)=\arg\min_{m\in\mathcal M_{physical}}\left[m^\top\widehat\Gamma_T(Z_T)m-2\widehat b_T(Z_T)^\top m+\lambda\,C_{native,T}(1-m,Z_T)\right]. $$

M_physical应限制完整64组、物理row-tile共用的形状/掩码、已可驻留的少量prepacked execution views或编译器真实支持的packed sparse kernel；C包括quant producer额外输出、code/scale读取、selector/packing/权重resident、kernel launch、W1/GELU/Q2/W2及端到端cost。与native packed dense、现行BF16-QDQ dense、普通static-group A0做各自合法数值与硬件匹配。若成本不可写成各组线性之和，必须实测/近似真实C(action)，不可只使用理论MAC。

可候选估计器：Gamma_hat(Z)=U(Z)^T U(Z)+diag(var(Z))，var≥0，U∈R^(r×216)，保证PSD但允许负offdiagonal；b_hat由冻结teacher high-precision与quantized pair离线蒸馏。是否能比static joint Gram、diagonal、RMS/step、旧HP1/NP1、同cost小MLP提供更低**行动后悔值**是OPEN。online每tile预测U的成本O(rG)仍需计价；不预设r小就一定更快。

## 3. 两阶段动作改变后续信息：Q-STAGE

现有Q1 scale-Gram（G6D）和Q2 panel（G6E）是不同合法决策时点；将它们联合后必须考虑A1改变随后实际能观测到的Z_Q2(A1)：

$$ Z_{Q1}\xrightarrow{A_1}W1_{A_1}\to\phi\to Q2_{A_1}\to Z_{Q2}(A_1)\xrightarrow{A_2}W2_{A_1,A_2}. $$

- A1**完整hidden64**早删：R12条件成立时，保留组的Q2与dense对应组相同，被删组根本不再生产Q2。后段A2可以只在真实剩余组上选W2 K面板，离线使用dense**保留组**元信息有条件合法。
- A1**W1输入K-panel**早删：多数隐藏预激活变化，Q2可能发生非线性scale/code跃迁；真实后段信息是Q2(A1)，不能把未执行dense反事实Q2作为免费router signal。这是G6F所提醒的关键差异。
- A1**紧凑M-A/M-B函数替代**：新的学生隐空间和Q2 recipe可能不同，A2须按学生自己的producer与后缀重新定义信息；teacher Q2不能当学生真实code。

抽象决策函数可以用V_1(z_1)=min_a1 [C_1(a1)+E_{Z2|z1,a1} min_a2 {C_2(a2)+lambda L(a1,a2)}]，但通用Bellman决策不是我们的首创。新研究机会在于组完整性交换律提供**不需要预测被改变Q2的合法可分阶段算子子类**，而W1-K/action改变观测的问题形成明确竞争机制。真实收益仍OPEN。

## 4. 为什么“创新的缝隙”在论文之间，而不是简单串联算法

| 文献或仓库成熟事实 | 已经覆盖的层面 | 仍未闭合的关键接口 | 最强竞争对照 |
|---|---|---|---|
| Prox 2026 / Spark NeurIPS25 | 付费低位代理、学习激活预测、选择性计算 | **Q1原生侧信道→源级合法whole64→负交叉Gamma的联合行动收益** | 同cost代理/learned topk、FP4 dense |
| FANG AAAI26（function-aware grouping） | 上下文功能组、静态/层预算分配 | 条件**符号抵消成束**能否在Wan gate/Q2/物理row tile上增益 | FANG式功能静态组、A0固定94组 |
| Learnable Permutation 2026 / PA-DST | 通过排列获得可执行结构 | 仅许整个64组排列（不能任意神经元旋转）以实现条件组束tile聚集 | 不重排的固定预pack和kernel开销 |
| QuantSparse ICLR26、FPSAttention NeurIPS25 | 视频attention量化稀疏联合、step/tile/kernel | 换成实际**Wan FFN Q1/W1/GELU/Q2/W2**的交叉符号和动作观测依赖 | 原论文相似机制、真实native sparse后端 |
| SharQ 2026 | 动态FP4稀疏主路+真实dense残差 | 整个函数组真的不算、不要隐藏dense补算 | SharQ全开销与native FP4 dense |
| G12与共享输入排列 | 固定线性目标精确重排界与强直接投影对照 | tanh-GELU/HiF4组闭包后能否以低于**直接投影**的成本指导行动 | CROWN/DeepPoly/直接低rank预投影 |
| M-A0当前NATIVE | 4位置、94组compact6016，同teacher局部拟合 | 组束条件选择能否超过全80参数函数的真实student/step-static基线且不灭失运动 | A0 whole-group、固定/step-only方案 |
| M-B最新四轮 | 后缀gate/head函数、规范自由度、source-joint和finite规划 | 让后缀敏感信号做离线监督，并由Q1**在线可预见**，避免teacher J泄漏 | 固定readout、gated loss、B0/B1、商空间损失 |
| M-C / LUT/CodeGEMM | 码本计算/输出低秩、GEMV快路 | whole64 tile无法高效skip时直接把量化码信息变成**可执行函数** | 同rank连续compact、multi-token packed GEMM |

文献证据等级：Prox/SharQ/G12由仓库原文与作者源码Ledger提供；新增FANG、Learnable Permutation、Veda/DFSAttn及DVD-Quant的本轮检索为**官方摘要/元数据层**，未宣称逐页源码核验。原始页面：[FANG](https://ojs.aaai.org/index.php/AAAI/article/view/40021)、[Learnable Permutation](https://arxiv.org/abs/2601.22980)、[QuantSparse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94359ca6e248af69b8b6854668ae9782-Abstract-Conference.html)、[Veda](https://proceedings.mlr.press/v306/han26y.html)、[DFSAttn](https://proceedings.mlr.press/v306/hu26p.html)、[DVD-Quant](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3bd28dd5cc4e15f9e019da13cc0c4844-Abstract-Conference.html)、[SharQ](https://arxiv.org/abs/2606.26587)、[Prox](https://arxiv.org/abs/2607.27591)。

## 5. 最短科学推进（不要抢 M-B B0/B1）

**T0 数学与既有资产/源码核实**：逐site确认原生Q1/Q2真正code/scale producer（原HiF4只有kernel局部临时值），真实postgain BF16和gate/e5读出。确认现有saved tensor是否保存相同source、row、step的完整组贡献；旧OFFLINE_ASSET_QUANT_STRUCTURE_R1只有聚合标量，不能由它反推raw Q1/Q2和Gamma。R12 group-extraction和单坐标 collateral作为已有证据直接复用。对2组toy严格复核、source/rowtile Gamma的定义证明足够，不用为恒等式重复模型试验。

**T1 源级最小判别**：若真实配对组响应资产可用（不擅自以旧视频取代），在冻结source×step×expert×block×physical rowtile条件下估计static joint Gamma和diagonal energy；报告源级 group pairing neg-cross的出现频率、单组energy→联合风险排序反转、跨T的稳定性和物理连续组mask占用。首先回答**真实联合可删结构存在吗**，无需先发明在线router。对新来源的采样权限仍按ACTIVE登记。

**T2 前置条件信息价值**：在已有真实Q1 payload上或冻结最小producer采集后，比较 F0 static, F1 cheap RMS+step, Q1 scale-shape, Q1 + few signed codes, static joint, conditional joint lowrank, matched same-cost tiny head、shuffled-sign/paired-source、oracle后验。**源级**report物理动作regret vs真正NetVOI而非Pearson/SSE; 对每个特征的读价和quantizer layout做独立费用记录。若scale rank collapse或sign sketch没价值，转向Q2 W2-only、static A0 / Q-CODE，而不是重复负样本。

**T3 真实执行与视频**：先约束少数rowtile级可预pack whole64 action class、共享两step专家权重真实运行时视图，证明减少W1/GELU/Q2/W2且所有quant group合法，并计resident权重、QDQ、observer和kernel wall；再做完整160 FFN调用的四步Student free-rollout，图像、主体/背景、光照、纹理、camera/motion、动作事件与独立source尾部质量/完整端到端wall、显存。UB1的20个已暴露开发源appearance -1.38%、RAFT -83.28%为明确的动态退化警告，不能用局部低SSE回避。M-B B0/B1是独立既定优先执行项，QCFB平行做离线/数学和文献，不替换授权。

预冻结**竞争假说与分叉**：H1有负交互但Q1不可预测→研究固定A0恢复/低rank编码；H2逐row负交互而physical tile聚合消失→few action classes/固定紧凑或换可执行单位；H3 scale只是量化配方/amax伪相关→对signed code/等价quantizer view强ablation；H4末层readout低误差却自由视频运动衰减→M-B后缀、Student prefix及多site交互；H5已有native packed dense全开销优于动态策略→保留信息规律不宣称效率方法。以上否定只对对应scope有效。

## 6. 方法与资产合同

- MATHEMATICAL_MODEL：原组分解 → group-selection Q2 commutation → conditional signed Gamma PSD/两组交叉 → relative high-precision error 的 linear b + quadratic Gamma → action-dependent Q2 observation → executable tile action、terminal rollout。
- OPERATORS：Q1_SIDE_PRODUCER [M,5120]→codes/scales/sketch；WHOLE64_ROUTER (preW1, whole 216 groups, per real token tile)→mask/class；SELECTED_W1 [5120→64k]→GELU→Q2 group64；W2_SELECTED [64k→5120]→gate residual；Q2_LATE_PANEL (postW1 actual Q2 only)→W2 Kpanel mask；VALIDATE_VIDEO_AND_WALL (four steps/80 block functions).
- NUMERIC_ORACLES：decoded-real additive group theorem vs BF16 matmul rounding vs original fake-QDQ vs packed MX/NV；同输入neutral source/row identity、old selected groups untouched，group inner-order/smooth/alpha/bias/gate真实路径；source-level video/wall不由局部oracle替代。
- REUSABLE_ASSET_PLAN：QCFB-WPROFILE（离线W1/W2 decoded+quant recipe+valid group index）, QCFB-Q1PAYLOAD（source/step/rowtile码与尺度、producer和额外读价）, QCFB-Q2WHOLE（真实post-gain/cast与组闭包）, QCFB-DIRECTIONGRAM（group输出、gate/readout、tileGamma、量化b）, QCFB-ACTIONLABEL（同state合法完整组选择动作及成本）, QCFB-PAK/KERNEL（真实物理mask、packed shape、memory/clock）, QCFB-VIDEO（81帧、四步、独立source/seed、各轴指标/视频）。逐资产记录producer、model/quant版本、shape/axes/dtype/coordinate、输入依赖、locator、随机性、验证/复用边界及失效条件。**本轮全部为PLANNED / NOT_PRODUCED**，没有新模型native/GPU/SYS/VIDEO结论。

[研究导航](../RESEARCH_NAVIGATION.md)、[量化路线v1](QMETA_DECISION_VALUE_AND_FUNCTION_FUSION_METHOD_SNAPSHOT_20261028.md)、[现行三方法](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)、[M-B四轮评议](../independent_reviews/M_B_20261028/SYNTHESIS_AFTER_FOUR_ROUNDS_20261028.md)为互补入口，而非四份互斥权威路线。正式独立方法提案与同行投票仍由method_discussion/QUEUE按真实独立作者处理。