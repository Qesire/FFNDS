# G13 · 先定位有意义的FFN输出函数，再识别承担它们的真实计算结构

**2026-10-27 用户指正后的首要科研任务。** 本文件取代“先研究量化code能预测哪些可跳神经元/函数”的研究排序。量化码是待研究结构形成后可选的**信息来源和计算表示**，并不先验规定要找到什么功能。研究从 LightX2V **Wan2.2 I2V A14B 4step D-Q high/high/low/low** 的现有源码/真实自由视频/保存原生受扰状态出发，沿同一计算图检索机制，不引入训练或新CPU/GPU/模型/视频实验。权限见 [ACTIVE](../ACTIVE_RESEARCH.json)。

**研究对象的优先逻辑**：
\[
\boxed{
\text{真实视频/flow现象与功能差异}
\;\longrightarrow\;
\text{条件性FFN输出函数族}
\;\longrightarrow\;
\text{实现该函数的计算图子结构/相互作用}
\;\longrightarrow\;
\text{可观测性与执行代价}
\;\longrightarrow\;
\text{量化码、proxy、重编码、训练或cache等实现}
}
\]

这不是要求每条新假说通过固定关卡，而是要求**因果与功能的定位对象不被后续选定的量化格式、通道掩码或算子所替代**。可以从源函数或视频反例反向发现新结构，任何新表达均能先作为 OPEN 被研究。

## 1. “有意义的输出函数”是什么？先定义行为，再定位表示

原 Wan 一次 FFN 的原生对象（光滑分支/量化BF16边界以原图G0–G5为准）：
\[
q=\text{AdaLN}(\text{norm2}(r_{\rm cross})),\quad
u_{l,t}(q)=e_{5,l,t}\odot
\left[W_{2,l,t}\,
Q^{\mathrm{HiF4}}_{2,l,t}\!\left(
\text{Smooth2}(\mathrm{GELU}_{tanh}(W_{1,l,t}Q_1(\text{Smooth1}(q))+b_1))
\right)+b_2\right].
\]
写回 \(r_{\rm next}=\mathrm{BF16}(r_{\rm cross}+u)\)，然后被后续同block/深层attention、head、flow/scheduler和下一step读取。原Q2有64→8→4共享尺度依赖，改变隐藏坐标可能改变其它坐标的量化结果。下面所有抽象的“\(\Delta u\)”都需要绑定到实际合法图切面、dtype、状态与输入分布。

**函数对象不是某个通道\(|h_j|\)、一段频谱或一个人为指定的“背景方向”。** 给定条件 \(c\)（原视频图/提示、source、seed、step、expert、block、token时空位置、当前状态 \(z_t\)），一簇自然出现或在真实源码中可定义的FFN响应差异形成**候选输出函数** \(f_c\)。例如同一个状态下、同一组输入变化、不同层间FFN的门控增量场 \(\Delta u\)；其空间–时间行为、跨状态稳定性、符号、与其它组件的补偿关系均是**要观察的特征**，不是默认可独立重建的“功能单元”。

**何时称“有意义”分三个可以分别取得的层级**：

- **FUNCTIONAL-OBS**：某类完整FFN的残差向量响应/条件差异在不同输入source、seed、层/step的合法对照里稳定出现，且对应可观察的生成状态、条件或视频现象（例如摄像机运动、背景亮度稳定、细物体变化）的**相关结构**。其意义是值得解释的可重复功能现象，还未拥有组件因果归因。
- **FUNCTIONAL-CAUSAL**：限定状态、source/seed/控制与干预方式，在该FFN输出/子图的**真实图边**作固定的替换/擦除/有限扰动，重算合法后缀和后续step，得到预注册视频/flow行为变化；检查恢复（sufficiency）、移除（necessity）及其他路径的补偿。这才是“它对该行为有因果贡献”的证据。本轮尚无逐候选新干预运行。
- **FUNCTIONAL-IMPLEMENTABLE**：已定位的因果功能可由明确的更小、更便宜或可共享的**真实物理计算子图**承担，在强matched full-dense参考下符合视频质量、成本和部署约束。与功能识别是独立结论，当前OPEN。

非因果观测依然有独立科学价值；避免把所有观测都压成“无效”，也避免把函数的可解释名字当因果机制。

### 避免被语义标签预设的输出函数库

视频行为维度是**定位候选的观测视角**而非先验真实神经元模块，具体包括：主体几何/身份、运动及方向与速度、相机运动、背景纹理、光照/阴影、时序稳定、细小对象、提示词/图像条件跟随。另设一条**无标签功能发现**：直接从真实\(u(q)\)及保存的\(P,N,\Delta\)中识别有符号共变、变号、跨时空约束、局部/全局消长、不变量和非线性输入响应；之后才通过视频、条件反事实把它们解释为语义。功能可以跨多个标签、跨层实现，也可以只有时间条件而无单个语义名。

**内部替代必须留出变动空间**：同一个可见行为可能由不同的神经元组合、attention与FFN交互、时间/空间缓存承载；同一隐藏单元也可能兼任多个行为。因此研究产物首先是**上下文、分布与干预方式限定的候选函数族及其支持证据**，不是“某个自然唯一的FFN通道组”。

## 2. 真正需要定位的“重要计算结构”：执行关系，而不是当前mask

把原生路径拆成以下可追溯依赖层：

1. **输入及条件关联**：Q1之前 norm2/AdaLN 的 \(q\) 如何依赖cross/self attention、timestep与e1–e6；哪些输入变化/step/expert会改变输出函数？
2. **表示与有符号合成**：同一 \(q\) 经W1所有行的共享输入、GELU非线性，产生隐藏场 \(h\)；经原HiF4-Q2全组尺度与W2列组\(\,g\odot W_2 h\)形成向量值函数；可能发生同相、抵消、互补、条件开关和协作，**不能用单列幅值独立判别重要性**。
3. **真实可执行结构**：计算图上的共同子表达式、同组尺度闭包、矩阵维度/访存tile、可能的shared/subset/低维结构与重算频率。结构只有绑定到实际W1/GELU/Q2/W2、注意力或跨步算子后，才能讨论计算可降。
4. **下游功能传播与替代路径**：FFN gated residual之后的norm/attention、后续40层、当前flow/scheduler与其它3个step；同输入Teacher/Student四臂以及完整dense有限状态响应可分开“本轮输出功能”和“历史状态已偏移后的传播”。
5. **跨输入/时空/层/step共享关系**：计算功能可以是多个token、潜在帧、step、层、expert共同维护的约束；量化码分组、显式稀疏和frame anchor都只是定位后可能采用的**物理表示**。

**结构重要性不等于神经元重要性**。如果某个AB子图的联合效应满足
\[
\Delta_B(A\cup B)\neq \Delta_B(A)+\Delta_B(B),
\]
那么该行为的计算结构包含**相互作用**，只按照A、B的独立SSE/范数/patching效果排序会误判。这里 \(\Delta_B(C)\) 必须在同一合法基准状态和干预定义下表示目标行为响应；非线性闭环下不能无依据地用线性相加。也要检验**补偿/备份路径**：移除A影响小时，可能因B接管而非A没有承担计算。

真正可定位的结构包括：同一q的多W1行联合映射、共享时间状态的跨step函数、空间分布上的协同相位与背景稳态、跨block形成的功能回路、训练改变后的职责分配及静态规律。发现哪一类结构很重要本身就是科学结果，即使暂时没有新内核设计。

## 3. 与已有原生实验的证据关系：现在可定位到哪里

| 原生观察与原始出处 | 当前能识别出的函数张力 | 下一条尚未识别的结构 |
|---|---|---|
| [六source原生 signed P/N/cross](../20261026/E0_SIX_SOURCE_DIRECTIONAL_RESULTS.md)：同输入局部FFN误差较小，可与完整dense状态有限放大并存；来源的有符号交叉项可正可负 | “误差变小”与“视频状态功能改善”不是单调关系；由条件决定当前候选注入与已有偏移是抵消还是增强 | 哪些**具体FFN gated-output功能**产生这个依赖、在哪个block/step注入、如何进入后续source/seed的完整视频 |
| [全网络finite map](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)：\(P=T_D(z_S)-T_D(z_D)\), \(N=T_S(z_S)-T_D(z_S)\), \(\delta_{t+1}=P+N\) | 明确状态传播和本步候选作用的不同数学对象；某些完整轨迹很大但本步\(N\)有抵消作用 | 哪些FFN/attention/condition计算子图造成固定状态响应；整步映射**不是局部Jacobian** |
| [跨seed来源425](../20261026/E0_FRESH_FOUR_SOURCE_SIX_PAIR_RESULTS.md)：同图同prompt不同seed使后三步cross符号全负/全正，局部误差计数相同 | 重要函数不能只由prompt/source标识，也不能只由scalar FFN SSE评分 | 联合输入状态、motion condition、门控、层间响应及空间/时间区域中哪个结构翻转signed效果 |
| [保存状态的频带](../20261026/E0_SAVED_FLOW_SPECTRUM.md)：全局与固定空间band的P/N cross可反号 | 内部功能可能被全局SSE或单频段能量掩盖；频率基可以帮定位**现象** | band是否有对应的真实视频功能/因果计算通路尚OPEN；频谱自身不是语义/因果标签 |
| [全hidden/tiles](../20261025/E0_SHARED_TILE_PRE_W1.md)及[Affine功能失败](../20261026/E0_POST_NATIVE_VIDEO_DIRECTION_REVIEW_20261026.md) | 后验per-token浓缩和可执行tile/terminal video显著错位，旧近似方法因完整视频失配未验收 | 是否存在更自然的函数组、跨通道消元、共表达、时间职能分工、跨层补偿，从而与实际kernel粒度对应 |

**已存数据优先的最小工作**（只整理，不运行新CPU模型或GPU）：登记当前已经原生测到的现象与具体step/source/seed、run identity、数值和对应视频质量级别；把已存 \(u,\Delta,P,N\) 的数据接口列为“可供日后授权读取”，核查有哪些输入/输出没有保存，区分已可纸上证明的函数等价与需要新干预的因果身份。现有12条视频/1920site是**嵌套原生观察位置**，不是1920个独立随机反事实。不可凭它们声称已完成新功能的干预定位。

## 4. 因果功能定位的方法学先例：取能力，也读清外部界限

| 论文 | 已公开的一手方法及对本问题的作用 | 严格边界 |
|---|---|---|
| **L80 Geiger等，Causal Abstraction (JMLR 2025)** [正式论文/64页](https://www.jmlr.org/papers/v26/23-0058.html) | 将神经计算与高层抽象之间的映射放到机制干预、抽象保真和多义特征/线性表示的统一理论里；提醒我们需要**高层行为+低层子图+对照干预**共同定义函数 | 理论框架并未自动给出Wan的函数或重要神经元 |
| **L81 Zhang & Nanda, Activation Patching Best Practices (ICLR 2024)** [官方](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html) | 同一component经patch的“重要性”会随着干预方式、corruption与评价指标而改变；要求合法基准与多指标 | 原实验主体为语言模型；本文只采用方法学启发 |
| **L82 Makelov等，Subspace Patching Illusion (ICLR 2024)** [官方](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) | **行为被子空间patch改变**可以是激活此前休眠并与原路径因果断开的备用通路，因此patch成功不足以证明子空间就是原本运行的功能结构 | 需要独立的原执行图依赖/恢复与消融支持；不能一开始就当作Wan已有该现象 |
| **L83 TIDE (AAAI 2026)** [官方](https://ojs.aaai.org/index.php/AAAI/article/view/37006) | 在**图像**DiT上用跨timestep SAE特征提出几何/物体/细粒度语义分层解释和可干预性 | SAE训练与可解释性不等于Wan**视频**FFN的原始函数因果定位或实际少算 |
| **L84 I'm a Map / IMAP (CVPR 2026)** [官方](https://openaccess.thecvf.com/content/CVPR2026/html/Jun_Im_a_Map_Interpretable_Motion-Attentive_Maps_Spatio-Temporally_Localizing_Concepts_in_CVPR_2026_paper.html) | 针对**视频DiT**将运动概念定位到“什么对象、什么时候、什么空间位置”，使用GramCol/IMAP，无需参数更新或梯度 | concept localization/semantic mask是**观察空间**，并未直接定位W1/W2因果函数或硬件可省结构 |
| **L85 Video ST-SAE (arXiv:2604.03919, ACMMM 2026 accepted)** [作者论文](https://arxiv.org/abs/2604.03919) / [源码仓库](https://github.com/atahandokme/spatio-temporal-sparse-autoencoders-video) | 作者对**video encoder DINOv2/VideoMAE**发现普通hard TopK SAE可能破坏时间相关性，用时空对比目标恢复；有针对性的feature ablation | **不是生成Wan FFN**，但表明“可解释且重建误差低”可能与时间功能冲突，支持把时序/功能做独立判据 |

这些条目本轮基于**正式出版摘要、作者公开摘要与开源项目说明**审视机制；未把完整方法、全作者源或具体Wan数值提升为已核证。原生结构LIT和已运行SRC/NATIVE永远分开。

## 5. 竞争性机制：不是先指定某个函数必须存在

| 可证伪解释 | 若成立预期是什么 | 与其争夺解释的另一结构 | 最少需要的区分证据 |
|---|---|---|---|
| **Hf1 联合有符号抵消/支持** | 同一q上的多个W1/GELU/Q2群形成共变、反相或空间特定的gate×W2函数；个别通道幅值不可靠 | 只是总输出惯性或scale/归一化引发的表观共变 | 原群协变、GELU/Q2同尺度组与W2有符号方向审计；之后限定状态的真实组联合干预 |
| **Hf2 跨层/attention补偿** | 单FFN函数改变后，后续residual/attention可能恢复或改变其视频效果；局部贡献与终端作用不同 | 同步变化只是受扰state传递，无需特定功能通路 | 锁定相同native输入/图边的层间响应，后续一阶假设与有限完整step区分 |
| **Hf3 时空共同/独有功能** | 一些功能维护长程背景/光照/运动相位而非单token局部能量 | 低频集中来自旧state继承、平滑/周期FFT边界 | 独立source/seed与时空区域中固定的无标签函数统计、条件视频对应，然后隔离因果输入 |
| **Hf4 去噪/专家职责依赖** | 函数类型随high/low、步、条件与状态变化；“重要”的时间和层级定义因而不同 | 四步差异只是输入分布/幅度变化 | 同expert的两步对照、明确Q1/GELU/Q2身份、跨step回写状态 |
| **Hf5 可理解功能与最小计算图不是一回事** | 可解释字典方向可能由多个等价/补偿子图承担；稳健的行为保真子图与稀疏latent basis不必一致 | 一条真正分离的简洁功能回路存在 | 双独立识别策略一致性、necessity/sufficiency/跨源稳定性、source-path忠实性与物理代价 |

这些问题允许其它创新功能的出现；五个H不是算法或候选排名。

## 6. 工作台交付结构：功能发现先于量化/硬件

每个未来FUNCTION记录包含：

\`function_id, behavior_definition, context(source/seed/step/expert/block), natural_signal_at_runtime, observed_response, measurement_identity, spatial_temporal_support, evidence_tier, candidate_implementation_structures, competing_explanations, finite_intervention_needed, full_video_metric_needed, available_existing_data, unresolved_assumptions\`。

随后才为其建立STRUCTURE记录：
\`structure_id, exact_graph_nodes_edges, true_quant_Q2_closure, constituent_W1_nonlin_W2_paths, cross_token_step_layer_dependencies, algebraic_or_causal_test, known_native_trace, physical_execution_granularity, offline_observer_cost, training_if_any, cost_and_quality_state\`。

这两个索引先保持**OPEN/OBS/ALG**的诚实等级。**结构中的“重要”由已定位的行为目标、源分布与干预来限定**，不是单凭更大的权重、激活、梯度、谱能量、语义可解释程度或可压缩性。

**新的研究顺序**：
1. 在原生实验观察与自由视频中形成**行为轴候选 + 无标签函数候选**，定位输出端涉及的条件、跨位置/时间变化与有符号相互作用。
2. 追问**哪个计算子结构在正常前向真实地产生该函数**，以及哪些上下文允许替代/联合结构。对已存数据作独立证据分级，明确功能意义的观察/因果两级。
3. 找到**函数与真实W1/GELU/Q2/W2/attention/step执行单位**之间的关系，研究可共享、不变量、有限等价、训练重组及物理操作数。
4. 最后，决定量化code/FP4符号、proxy、历史、LUT、结构化稀疏或小后训练是否提供便宜的提前信息或新的核实现。

此轮只更新研究定位、文献和计算图，模型实验未开始。当前[统一综合](FFN_MECHANISM_LITERATURE_AND_GRAPH_SYNTHESIS.md)应按本文件的优先次序阅读。
