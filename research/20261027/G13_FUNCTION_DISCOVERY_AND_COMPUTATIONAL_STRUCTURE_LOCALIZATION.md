# G13 · 有意义的FFN输出函数及其计算承担结构

**研究定位：2026-10-27** · 从真实视频行为、FFN写回及跨状态差异出发，发现**哪些输出函数值得保持、哪些原生计算结构实际承担它们**。进一步研究联合函数能否共享、重组、预测或更高效求值。此阶段开放多种数学表示和计算机制；量化code、稀疏mask、proxy、cache和训练后结构为后续设计资源。

**原生对象**：LightX2V Wan2.2 I2V A14B，四步蒸馏D-Q、high/high/low/low双专家、40 blocks。主图：[G0–G13](../../docs/98_端到端计算图与证据注解_20261025.md)；[行为/函数/计算结构分别登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)；[O/D/Q/P函数定位方法学](G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)与[三张源码/行为定位卡](../agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)。具体证据与实验权限以[ACTIVE](../ACTIVE_RESEARCH.json)为准；本次为源码、原文、数学和已有结果的研究整理。

## 1. 发现函数：输出具有什么行为作用？

原FFN的真实信息路径在当前层/step/expert \(l,t,c\) 上写作：

\[
\begin{aligned}
q_{l,t,c}&=\operatorname{AdaLN}_{l,t,c}(\operatorname{norm2}(r_{\rm cross})),\\
h_{l,t,c}(q)&=\operatorname{GELU}_{\tanh}
\left(W_{1,l,t,c}Q_1(\operatorname{Smooth1}(q))+b_{1,l,t,c}\right),\\
u_{l,t,c}(q)&=e_{5,l,t,c}\odot\left[
W_{2,l,t,c}Q_2^{\rm HiF4}(\operatorname{Smooth2}(h_{l,t,c}(q)))+b_{2,l,t,c}
\right],\\
r_{\rm next}&=\operatorname{BF16}(r_{\rm cross}+u_{l,t,c}(q)).
\end{aligned}
\]

\(u(q)\)及真实残差写回由后续block、attention、flow与scheduler消费。HiF4 Q2 的64→8→4组尺度是同一原生数值函数的一部分。实际执行细节/shape以[G2和原生源码](../../docs/98_端到端计算图与证据注解_20261025.md)为准。

**功能研究对象**：给定真实上下文 \(c=\{\mathrm{source},\mathrm{seed},\mathrm{step},\mathrm{expert},\mathrm{block},\mathrm{token\ space-time},z_t\}\)，定义一个FFN残差响应族、其自然条件差异及可关联的生成行为。功能可以是有标签的主体、相机、运动、背景、照明、细节，也可以是未命名的**有符号协作、抵消、相位关系、共变和稳定性约束**。某个神经元或一条语义标签只是潜在的观察坐标；真实功能可以由多个相互作用的通路分布式实现。

功能证据采用三级，支持研究者从任何一级继续发现新问题：

| 等级 | 它告诉我们什么 | 实际所需证据 |
|---|---|---|
| **FUNCTIONAL-OBS** | 真实上下文中的输出响应与视频/latent/flow现象形成值得解释的关联；可发现新的无标签函数轴 | 固定source/seed/step的原生张量或视频、完整证据身份及独立条件变化 |
| **FUNCTIONAL-CAUSAL** | 具体图边和条件限定下的函数，对某个行为有可复核的真实因果作用 | 同状态/合法图路径的限定干预、normal-forward忠实性、必要性/充分性、下游重算和补偿检查 |
| **FUNCTIONAL-IMPLEMENTABLE** | 承担上述功能的计算结构能通过更经济的真实算子实现，维持相应功能与终端质量 | 原生W1/GELU/Q2/W2或层/step计算变化、成本账本、完整视频与实际系统对照 |

这三级体现研究成熟度与不同信息价值：有信息量的FUNCTIONAL-OBS可以推动新理论，FUNCTIONAL-CAUSAL把角色落实到正常计算路径，IMPLEMENTABLE再把机制变成高效执行研究。

## 2. 定位结构：正常计算如何承载这些函数？

沿源图追踪五种互相连接的结构自由度：

1. **输入条件与共同驱动**：同一\(q\)受cross/self-attention、AdaLN、step/expert、时间状态影响，驱动数千个W1行；研究联合响应与条件不变量。
2. **非线性和量化依赖**：W1→tanh-GELU→HiF4 Q2形成条件响应；Q2按原64→8→4共享组尺度。研究同尺度组内外的交互、尺度控制与可合并路径。
3. **有符号输出合成**：多个W2列经\(e_5\)门控成为一个向量值残差函数。观察同相、抵消、跨通道协同、共同空间方向及与行为的对应关系。
4. **真实下游与备份**：写回进入后续normalization、attention/FFN、flow和scheduler；研究后续层怎样继续、转移或补偿某类函数。
5. **时空及跨专家职责**：同一功能可能跨token/latent帧/step/层共享，或由high与low专家条件性分工；四步的状态递推使局部贡献和终端视频行为形成不同尺度的因果问题。

**联合结构的数学线索**：若对固定行为 \(B\) 和同一合法干预定义，子结构 \(A,C\) 的响应存在
\[
\Delta_B(A\cup C)-\Delta_B(A)-\Delta_B(C)\ne 0,
\]
则函数的重要组织单位包含二者的**交互项**。这为研究共同输入、GELU非线性、HiF4尺度耦合和W2 signed合成提供直接入口。还应在正常前向与替换干预中区分已经使用的主通路和可能接管的备用通路。

计算结构可以表现为跨神经元的共同非线性、可共用的子表达式、时空共同/独有更新、跨层回路，乃至训练后重新分配的职责。**先发现什么结构在承担功能**，再考察它能否压缩成原生GEMM/合法tile、缓存状态、编码域计算或新紧凑模型。

## 3. 已有证据提供的具体功能问题

| 已有原生事实与来源 | 新的机制问题 | 可以定位的下一层结构 |
|---|---|---|
| [六来源完整四步signed P/N/cross](../20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)：同输入FFN局部误差减小，与dense全网络有限状态响应放大可以并存；交叉项随source变化 | 哪些signed输出功能使“当前候选改动”和“已继承状态偏移”抵消或增强？ | 真实gate×W2输出到后续block/step的路径，逐层/step条件响应 |
| [全网络有限响应](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)：\(P=T_D(z_S)-T_D(z_D)\)、\(N=T_S(z_S)-T_D(z_S)\)、\(\delta_{t+1}=P+N\) | 对完整变化分解为**既有状态经dense传播**与**候选当前作用**，各自对应什么原生计算？ | 后续attention/FFN/flow/scheduler上的传递与补偿；这些是实际有限状态对的响应 |
| [来源425的交叉seed](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)：同prompt/img两个seed后三步cross符号全负/全正，局部误差位置统计相同 | 哪些输入状态/空间区域/条件组合决定signed作用方向？ | 同源跨seed的联合输出函数，expert与后缀流依赖 |
| [已保存频带观察](../20261026/E0_SAVED_FLOW_SPECTRUM.md)：全域与一个指定频带内的signed交叉项可反号 | 哪些空间–时间联合函数使不同响应方向在相同全局能量下被掩盖？ | 频带作为**现象观测坐标**，沿原生路径继续对应行为/语义或无标签约束 |
| [原HiF4与tile](../20261025/E0_SHARED_TILE_PRE_W1.md)、[既有视频质量发现](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md) | 为什么token级高后验稀疏与实际tile可节省算术、整视频功能形成不同对象？ | 真正的联合函数组织粒度及其物理执行单位 |

已完成**10个独立source身份、12条source×seed自由视频轨迹、1920处嵌套FFN site**。这些事实提供跨情境线索；当前具体单/多神经元的功能归因登记为OPEN。整理现存观测与可用接口，可明确后续某一最小因果判别需要额外采集的具体图边。

## 4. 文献支持的定位工具与不同预测

| 原始来源及阅读级 | 已可借用的机制思想 | FFNDS的可区分问题 |
|---|---|---|
| **L80 Geiger等 Causal Abstraction（JMLR 2025）** [正式论文](https://www.jmlr.org/papers/v26/23-0058.html) | 高层行为、低层表示与干预忠实性之间的机制映射 | 哪个正常计算子图支持某个条件视频行为，而非只支持数学重建？ |
| **L81 Zhang & Nanda Activation Patching Best Practices（ICLR 2024）** [官方](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html) | corruption与指标选择改变component的定位结果 | 同一假说在不同合法对照及video行为指标下是否仍成立？ |
| **L82 Makelov等 Subspace Patching Illusion（ICLR 2024）** [官方](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) | patch可激活normal-forward之外的备用通路 | 如何证明发现的功能子图在真实正常生成时承担该任务？ |
| **L83 TIDE（AAAI 2026）** [官方](https://ojs.aaai.org/index.php/AAAI/article/view/37006) | 图像DiT的time-conditioned语义SAE表示 | 能否从视频FFN正常响应中找出新的随step演变的无标签函数轴？ |
| **L84 I'm a Map / IMAP（CVPR 2026）** [官方](https://openaccess.thecvf.com/content/CVPR2026/html/Jun_Im_a_Map_Interpretable_Motion-Attentive_Maps_Spatio-Temporally_Localizing_Concepts_in_CVPR_2026_paper.html) | 视频DiT中motion概念的“对象×时间×空间”定位 | 如何从motion定位进入gate×W2与原Wan生成状态的因果结构？ |
| **L85 Video ST-SAE（arXiv:2604.03919，ACMMM 2026）** [作者论文](https://arxiv.org/abs/2604.03919) | 视频编码器的时间相关结构与稀疏表示的权衡 | 哪些时空功能因一般重建目标而弱化、又在完整视频行为中重要？ |

本轮按[来源账本](../literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)逐篇记录正式摘要/作者摘要和模型任务；一手机制转移首先形成不同预测，在具体Wan SRC/OBS观察基础上逐步判别。

## 5. 相互竞争的科学解释

| 假说 | 预计观察到的自然结构 | 下一项能区分机制的证据 |
|---|---|---|
| **联合有符号合成** | 多W1行通过GELU、Q2、W2共同维持条件输出方向；单行与群体的功能变化存在交互 | 比较同状态的联合signed W2响应与独立列响应；检查原Q2共享尺度及同源跨seed方向 |
| **下游补偿/备份** | 某层输出变化后，attention和后续FFN在具体状态下增强、减弱或转移该效应 | 锁定合法图边，对已计算正常通路与有限补偿路径设计分开的干预预测 |
| **时空协调函数** | 有些函数维持背景、照明或运动关系，表现为跨token/step条件不变量及有符号时序协同 | 结合完整视频对应区域与无标签状态响应，区分旧state继承和当前FFN注入 |
| **expert和去噪状态职责** | high/low两expert及相同expert不同step在不同输入状态中承担可比较的函数族 | 条件与真实q匹配后，比较相同/不同expert的功能响应、门控和后缀依赖 |
| **解释表示与最小计算单元不重合** | 同一视频行为可以由多个相关/补偿子图承担，且有符号输出空间中的函数基与物理tile不同 | 正常路径干预、必要/充分、跨source迁移及与Q2真实执行单位的对照 |

每条解释可以继续派生新数学坐标；当前研究以**找到真实功能和结构中的差异**为结果，而非按某个指标提前指定解决方案。

## 6. 研究记录与继续推进

独立的[Function–Structure registry](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)维护：

- **BEHAVIOR**：现象、source/seed/video/flow条件、对应真实RUN、可重复性与可观察行为；
- **FUNCTION**：自然输入集合上的输出响应和其有符号/时空关系，OBS/CAUSAL身份、竞争解释；
- **STRUCTURE**：真实原计算图边、非线性/量化/门控/attention/step依赖、可能的补偿关系和实际执行颗粒度。

已有保存证据先用于**功能候选地图**与**图上真实依赖地图**，形成科学上不同的竞争解释；之后选择最低成本的因果判别，并在新授权下执行特定模型动作。随着功能结构明晰，再比较量化代码、LUT、前置proxy、稀疏tile或后训练，围绕**实际需要保留的功能**建立物理代价与质量约束。

**当前授权范围**：论文、源码、数学与已保存材料的解释性整理；新原生CPU/GPU/模型/视频实验依[ACTIVE](../ACTIVE_RESEARCH.json)继续暂停。原实验结果、候选的真实失败边界与正式质量状态各自保留。
