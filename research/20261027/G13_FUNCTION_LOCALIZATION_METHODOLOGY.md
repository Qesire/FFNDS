# G13 · F线函数定位方法学：从行为差异到正常前向的计算承担者

**2026-10-27 · 论文原文、已存原生证据与条件数学推导。** 基于 FFNDS `14129346092aef2c3efed4a565d154f494fed2af` 及后来G13文档状态。[G13研究目标](G13_FUNCTION_DISCOVERY_AND_COMPUTATIONAL_STRUCTURE_LOCALIZATION.md) · [独立函数/结构登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json) · [机器定位卡](../agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json) · [一手文献来源](../literature/20261027/G13_FUNCTION_CAUSAL_LOCALIZATION_SOURCE_LEDGER.json)。

**适用范围**：这是[并行研究总目标](../RESEARCH_GOALS_AND_PARALLEL_MECHANISMS_20261027.md)中F线的因果作用与承担结构定位工具；I线的信息证明和C线的算子/系统构造可采用各自最合适的数学及部署验证，无需先执行O/D/Q/P。

## 0. 本次方法学形成的两个互补定位问题

**作用定位**：在自然视频行为与实际FFN写回之间，找出某类有意义的、上下文限定的响应关系。**来源定位**：沿正常前向的真实图边，从e5门控写回逆向定位W2 signed组合、HiF4 Q2尺度闭包、GELU/W1共享输入和跨层/step传播。功能可以分布式实现；同一正常计算结构也可能在不同source、seed和step下承担不同职责。

研究顺序由问题驱动，允许先从任何已观测的现象或新公式找到突破口。一个典型有效路线是：**O写回作用 → D固定post-Q2贡献 → Q合法pre-Q2干预 → P后缀运输和跨层作用**。四个边界不是预设的单向实验清单：可以由新论文、真实量化源码或时空响应反过来发现其它候选结构。

**本轮材料身份**：新增的是定义、条件代数与未来试验设计。原生模型CPU/GPU、视频与组件级干预新增执行均为0；原[ACTIVE](../ACTIVE_RESEARCH.json)暂停状态保持不变。

## 1. 定义有意义的函数：上下文、行为、作用边与后缀

在一个固定的源实现、权重与随机状态下，记录上下文
\[
c=(\mathrm{model/weight/config},\,\mathrm{source},\,\mathrm{seed},\,t,\,\mathrm{expert},\,l,\,p,\,
\mathrm{conditioning},z_t,\mathrm{rng},\mathrm{dtype}),
\]
其中 \(p\) 为latent token的时空索引。\(u_c\) 表示真实FFN输出；\(H_{c,\partial}\) 表示**从明确干预边界 \(\partial\) 起、采用明确冻结/重算策略的正常后缀**；\(b_k\) 表示第\(k\)种预定义的标量或向量行为读出。

\[
\mathcal L=
(\mathcal C,b,\partial,S,I,\mathrm{donor/base},H,\mathrm{comparison},\mathrm{prediction}),\qquad
E_{k,c}(S,I)=b_k(H_{c,\partial}(I_S))-b_k(H_{c,\partial}(I_\varnothing)).
\]

\(I_S\) 是一份可重放的**完整状态替换定义**：写入的真实边和数据类型、从何处取得替换值、保持/重算哪些依赖，以及当前FFN写回边界前的状态。\(E\) 是未来干预测量的定义。先记录每个source×seed条件下的有符号向量、时空范围和读出，再研究聚合规律。自然关联支持FUNCTIONAL-OBS；独立合法干预支持条件性的FUNCTIONAL-CAUSAL；定位后具备物理减算与视频质量支持再到FUNCTIONAL-IMPLEMENTABLE。

**功能可以先用无标签定义**：\(\mathrm{FUN01}\) 的联合signed写回模式、条件变号、时空稳定性、跨层协调，均可由真实输出和自然变化提出。PCA、SAE、GramCol、频带、W2方向、联合分解和梯度提供候选坐标，正常计算中的承担结构则由源图边、节点、跨Q2分组或跨层路径构成。

**作用签名**可把\((E_{k,c,I})\)在预定有限上下文上比较。近似签名相似意味着该测试范围的行为作用相近；在不断补充样本时保留配对与原始响应差。使用阈值时，其相似关系未必满足传递性；科学产物首先是条件性的**函数候选族及簇间竞争假说**。L80的causal abstraction和graded faithfulness提供高层行为与低层干预匹配的理论语言，并未直接指定本Wan的神经元回路。

## 2. 行为读出：让视频差异有方向、能与替代解释竞争

| 已知现象/功能假说 | 正常视频中可预定义的读出 | 同时衡量的其它作用 |
|---|---|---|
| B01 背景/光照随时间波动 | 固定并经过光流/几何配准的背景ROI亮度时间曲线、时序差、阴影方向；显式登记遮挡、曝光变换和区域追踪有效性 | 主体区域亮度、相机轨迹、纹理/可见性和全局颜色变化 |
| 运动功能与相机运动 | 区分相机/背景全局运动与主体相对运动；记录方向、运动时点、轮廓与条件语义 | 主体身份、背景光照、光流有效性，纹理退化造成的假运动 |
| B02 source425跨seed signed P/N差异 | 固定同一个全步有限映射定义，记录方向、范数、cross以及指定自然视频表现 | 同一seed内部状态、门控/attention输入分布、已有状态传播 |
| B03 全域与子频带cross异号 | 保留空间FFT/频带选择、符号/方向、原始latent张量和相应region，先称“无标签状态响应” | 全域混合、原latent平滑、边界窗/频率选择导致的表观差异 |

这些是**行为指标的具体设计对象**。既有Farneback或相机prompt本身尚不等于相机语义正确性的验收；未来CoTracker/视频VBench需要真实运行身份才登记为已测。source425已有的跨seed变化说明条件异质性：仅改seed同时改变许多内部状态，适合识别函数随状态变化，属于自然对照；未来要操纵单个变量需定义更受控的边界。

区分两类任务：**同状态Dense/候选近似**定位某种近似计算带来的当前作用；**自然Dense条件对**定位原模型在正常运行时承载的功能。两个任务的相同函数预测可以相互校验，其具体donor/patch语义各自记录。donor取自然样本有利于匹配状态，内部拼接的分布相容性仍通过receiver读出、sham对照和状态距离评价。

## 3. FFN原生边界与四种定位操作

当前Wan块的真实计算从cross-attention residual到norm2/AdaLN、Smooth1/Q1、W1、GELU(tanh)、Smooth2/HiF4 Q2(group64→8→4)、W2、e5门控BF16写回，再进入后续block/flow/scheduler。为作条件代数，以一token列向量在同一真实\(q\)、权重、bias和门控下定义

\[
h=\mathrm{GELU}_{\tanh}[W_1Q_1(\mathrm{Smooth1}(q))+b_1],\quad
a=Q_2^{\mathrm{HiF4}}(\mathrm{Smooth2}(h)),\quad
u=\mathrm{diag}(e_5)(W_2a+b_2),\quad
r_{\rm next}=\mathrm{BF16}(r_{\rm cross}+u).
\]

真实内核包含BF16舍入顺序、QDQ dtype及参数的实际布局；该公式是便于讨论的**同一输入实数分解**。

| 操作 | 可精确定位的对象 | 依赖与下游真实重算 |
|---|---|---|
| **O · Output/writeback** | 指定e5门控前/后、BF16 residual前/后的FFN输出或向量子空间，测试功能是否能由该边界改变 | 固定同一base prefix；记录真实写入的边、dtype、bias、e5、残差和真实后缀身份 |
| **D · fixed post-Q2** | 使用当前已计算的\(a\)，改变W2所接受的部分量化后坐标，衡量**直接signed列贡献** | 固定整段\(a\)与原scale；重算真实W2/gate/writeback；这是功能分解的研究操作 |
| **Q · valid pre-Q2** | 在\(h\)上执行明确通道/组修改后，重新走**实际Smooth2+原HiF4 Q2**，得到可从上游执行的功能变化 | 保留原group64→8→4的全部尺度/阈值、影响闭包、W2/gate、BF16与后缀 |
| **P · suffix/path** | 正常全后缀继续执行，或有依据地固定某个receiver/witness区分正常运输、放大及补偿 | 指定冻结的图边及其base/donor取值；自由rollout与冻结路径作为两个实验对象 |

从O观察**作用能否从写回引起**，从D/Q分析**哪些计算路径与尺度依赖可产生这种作用**，从P判断**下游哪种真实状态传播关系最关键**。功能角色与HiF4数值group的执行边界可以互相交错；一个group里可能承担多种功能，多个group也可能共同支持同一个功能。

### 3.1 直接列贡献与pre-Q2连带改变的恒等分解（ALG）

固定\(q,e_5,W_2,b_2\)，并固定上表定义的完整Q2结果\(a\)。对通道集合\(S\)，**固定post-Q2删除动作**定义
\[
(a_{\rm post})_S=0,\quad(a_{\rm post})_{\bar S}=a_{\bar S},
\qquad
\delta u_{\rm post}=-\mathrm{diag}(e_5)W_{2,S}a_S.
\]
一个**pre-Q2动作**通过明确的\(I_S(h)\)产生
\[
a'=Q_2^{\rm HiF4}(\mathrm{Smooth2}(I_S(h))),\qquad
\delta u_{\rm pre}=
\underbrace{\delta u_{\rm post}}_{\text{固定码的列贡献}}
+
\underbrace{\mathrm{diag}(e_5)W_2(a'-a_{\rm post})}_{\text{目标自身剩余作用及组尺度/平滑等连带作用}}.
\]

该式在固定gate和权重的实数向量层面**严格成立**，与具体\(I_S\)是否将目标通道置零无关。为了进一步区分尺度影响，定义
\[
a'-a_{\rm post}=
\mathbf 1_S\odot a'
+\mathbf1_{\bar S}\odot(a'-a).
\]
因此右侧可分成**目标被修改后残余贡献**与**补集受共享尺度、平滑、量化改变的连带贡献**。本式更便于检查组尺度的实际影响。若Smooth2/Q2含跨组或跨通道依赖，以**源码确定的依赖闭包**为范围，64组只是现阶段已确认的一项基础依赖。真实BF16顺序另记录原内核有限差。

**区分预测**：若D与Q导致相近的\(u\)方向与行为作用，可能以目标列的直接signed计算为主；若Q效应集中在补集scale/code变化，存在“**数值尺度调节者**”这一结构身份；若目标group同时控制自身与补集，两种角色可以共存。上述两个代数分量是FFN写回来源账本，在非线性后缀的行为分数上需重新实测其联合作用。

### 3.2 三种非加性来源分离（ALG）

对同一个base、同一边界定义不重叠动作\(A,C\)，标记\(u_{ij}\)为应用/未应用两动作后的FFN输出，\(R_{ij}\)为下一receiver，\(Y_{ij}=b(H(u_{ij}))\)为行为读出。定义
\[
J_u=u_{11}-u_{10}-u_{01}+u_{00},\quad
J_R=R_{11}-R_{10}-R_{01}+R_{00},\quad
J_b=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]
在固定post-Q2的不同列、固定gate和bias、理想实数W2下，\(J_u=0\)；当Q2重算、Smooth2尺度调整或上游W1/GELU共同输入变化时，\(J_u\)可能非零。BF16舍入可带来小的原生非加性。即使\(J_u=0\)，后缀和评分器依然能给出\(J_R,J_b\ne0\)。

**明确反例**：\(u_{ij}=ia+jc\)，\(b(u)=\|u\|_2^2\)，则\(J_b=2\langle a,c\rangle\)，即完全线性的输出列也可产生非零行为交互。因此**联合结构的实验证据由其首个出现非加性的合法图边来定位**；一个标量cross或能量分数可用于发现候选，却属于不同的数学对象。

## 4. 双向替换、群体交互和补偿

取自然条件中目标行为较强的\(x^+\)及匹配的\(x^-\)，令\(Y(x;S\leftarrow x')\)表示在真实\(x\)基线边界把集合\(S\)替换成来源\(x'\)的相应值后重算后缀。方向性效果定义
\[
N_S=Y(x^+)-Y(x^+;S\leftarrow x^-),\qquad
R_S=Y(x^-;S\leftarrow x^+)-Y(x^-).
\]
这是**破坏方向**与**恢复方向**两种不同的上下文作用，分别回答“在强行为背景中替换这一部分会改变什么”和“在弱行为背景中恢复这一部分会改变什么”。是否称必要/充分由明确的行为指标、替代规则和阈值决定。小幅基础行为差\(Y(x^+)-Y(x^-)\)不适合归一化；比值时保留原分母、方向以及任何负值/大于1的原值。

L86指出AND型与OR/备用型结构在两方向patch中的不同反应：AND可能要成组恢复，OR可能因正常备用而对单点破坏不敏感。研究组\(A,C,A\cup C\)、其补集及matched随机/同规模控制，检查条件组粒度下的留一最小性和负作用。**组的集合依赖与个别列的边际效应互补**：找到小组只是一个候选函数承担者，需看其正常存在的依赖与跨条件泛化。

对下游候选支路\(W\)设置witness固定值，可研究
\[
N_{S\mid W=\mathrm{base}}
\quad\text{与}\quad N_S
\]
的作用差，并辅以sham固定、主通路是否被切断、未干预自然轨迹的支路活动和同状态receiver响应。两者差首先是**自由后缀与固定后缀的条件差**；当其它可能原因被区分后，再解释为补偿或替代路径。L88（2026-10-02，arXiv:2610.04017v1）的WISE对成组mediators及witness取分布期望，JuntaLearner寻找有作用的集合，属于LLM电路发现；这里迁移的是可证伪的群体/补偿对象，轻量数学研究先于大规模搜索训练。

## 5. 验证正常执行职责：自然支持、路径连通、行为特异性

L82的subspace-patching illusion展示了干预激活备用路径的可能性，促使“支持正常功能职责”的证据包含四个互补视角：

1. **自然支持与干预相容**：在无补丁的视频/latent正常轨迹上该候选按预测变化。记录base/donor具体source、seed、step、expert、token及其余条件，对替换后的投影范数、分组scale与receiver状态进行校验。
2. **输入输出路径连通**：追踪实际gate×W2变更进入的后续正常receiver，比较整体列组、子空间与互补组。固定post-Q2、固定W2的数学零空间，与经过非线性Q2后的上游零空间是不同对象。
3. **来源与条件泛化**：将已使用的10个source当探索集；新source在候选/指标确定后才能成为独立确认。已有新seed可检验同source的条件迁移，high/low两expert首先作为两套真实权重分别定位。
4. **目标与非目标功能共测**：保留目标行为变化、非目标视频功能、latent/flow及完整视频质量向量。广泛破坏提示通用计算瓶颈，也可作为发现潜在高价值联合结构的线索。

L87在MLP、attention circuit上显示faithfulness对节点/边、粒度、替换值、token位置、方向和集合选择的依赖，因此这六项作为每条具体定位命题的身份保留。独立数学证明、源码追踪与有限干预共同使“正常承担者”的结论成熟；因果效果的选择范围和等价身份均明确标注。

## 6. 在现有资产上定位，未来选择最经济的判别

**现存事实**：10个source、12条source×seed完整生成与1920个嵌套FFN位置；相关采集有完全不同的张量保存粒度。读取现有RUN/trace/视频索引时登记实际存储的\(q,h,a,u\)、gate、后缀状态等切面；标量P/N/cross及FFT频带资料可以发现待解释现象，其是否包含具体FFN分组可从资产清单明确分辨。

未来查找函数可在全40层×四步、相应自然空间/时间位置上发现候选，再深入到W2方向、Q2依赖闭包与多W1行/跨层路径。已有研究支持的PCA/SAE、梯度、SAE feature或频域探针可以用于**离线候选优先级**；与实际FFN分组的原生有限替换、同规模/低分组抽查、跨seed/source迁移共同检查漏检与选择偏差。

L89 AtP*在LLM里研究梯度Attribution Patching的漏检、attention饱和与直接/间接影响抵消，并用随机子集诊断；Wan还包括QDQ阈值和BF16舍入。由此可以把代理梯度/STE作为候选筛选信息，最终用真实数值路径核实条件效应。离线功能探索的额外观测开销可以合理很高；这些信息如果转为**上线路由**，才需要严格计入发生于W1前/后的时点、observer花费与实际可省的算术/访存。

**统计计划**：先逐source内记录各seed方向与强度，再按source作为独立cluster汇总；跨site、token和视频帧记录相关结构。已经用于提出假说的source是探索集，未来外部确认在新source上预先固定行为、group、替换值、指标及读出方向。实验预算由是否能够区分机制及后续单独授权决定。

## 7. 三张有判别意义的定位卡（均为PROPOSED）

### C01 · source425：写入函数变化还是后缀运输变化？

**原生现象**：相同source的两个seed在后3步P/N cross作用符号不同，而局部FFN raw误差计数一致。**H-write**：当前FFN写回向量/门控方向在两个状态下有实质差异。**H-transport**：写回作用类别类似，但当前state的attention/后缀对相同方向的运输不同。

**当前静态任务**：从现有source×seed RUN索引具体列出可配对的postgate输出、receiver、latent与视频文件及其缺口。**未来的有限判别**：分别在各自原生base状态定义一致的改动规则，从O边界比较\(u\)立即差、首个receiver、step_post及完整行为；若研究交叉seed donor，则将其作为不同的相容性干预。现有P/N已经分离整步历史状态响应与当前完整候选作用，可用来限定这一新观测最有价值的图边。

### C02 · 直接输出贡献还是HiF4尺度调节？

**H-direct**：候选组的signed W2列贡献直接承担目标输出函数。**H-scale**：组改变共享Q2尺度/平滑，使补集code变化而实现同一或相反的行为作用。

**当前静态任务**：精确确定Smooth2/Q2/group64的原实现影响闭包、原W2与gate的顺序。**未来的有限判别**：同base比较D与Q两个合法动作，记录\(a'_S\)与\(a'_{\bar S}-a_{\bar S}\)、gate×W2直接/连带分量、后缀行为与同规模比较组；以§3.1精确分解解释差异，保留真实BF16有限差。

### C03 · 背景照明稳定函数还是上游状态/解码的表现？

**H-coordinate**：部分跨时空位置的联合FFN写回协调背景亮度/纹理变化，单token局部能量缺少其功能信息。**H-propagation**：既有state偏移经attention/flow/解码带来背景显现变化。

**当前静态任务**：在已保存自由视频与latent/flow中配对背景区域、相机/主体轨迹、条件/seed及实际视频质量身份。**未来的有限判别**：对功能性分布式组、同规模随机组与非目标区域做明确的O/Q/P对照，采用自由后缀与有条件witness固定，并观察目标及非目标视频读出。若只改变latent频带而未改变预定视频行为，先命名为**latent状态响应功能**，再研究其与背景意义的条件关系。

三卡均以**独立竞争性预测**作为成果；后续要启动哪一项由已存证据能区分什么、源码事实和实验许可决定。

## 8. 定位结果如何自然形成高效计算研究

一个有效函数定位命题最终提供：**要保留的行为作用、正常执行该作用的计算子图、该子图的条件依赖、执行之前可取得的信息、可改变的真实算术/内存/调度工作**。不同发现可能导向：
- 多组W2 signed作用相似：共享求值、重新参数化或训练职责重组；
- W1/GELU/Q2共同非线性：寻找真正的有条件joint update而非孤立通道分数；
- 某组为HiF4数值尺度调节者：分离尺度维护成本与实际有价值的函数更新；
- 跨block/step的补偿/替代：联合而非单点的功能保留及可能的真实时间复用；
- 作用分布跨latent帧：研究因果时空函数组织及适合实时chunk的执行单位。

函数的价值、真实因果身份与物理实施分别以FUNCTIONAL-OBS、FUNCTIONAL-CAUSAL、FUNCTIONAL-IMPLEMENTABLE推进；公开作者的机制能力和本Wan实际测量保持各自的身份与论文阅读等级。正式全视频质量与真实延迟由原指标协议验收。

## 9. 相关原论文及确切迁移内容

| 来源 | 本轮已核的一手方法 | 在FFNDS中启发的不同定位对象 |
|---|---|---|
| [L80 causal abstraction, JMLR2025](https://www.jmlr.org/papers/v26/23-0058.html) | 理论总览与分级faithfulness；仓库既有更细阅读账本 | 作用身份依赖高层读出、干预与低层映射 |
| [L81 patching best practices, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html) | 方法与metric/corruption敏感性；仓库既有阅读记录 | 写清base/donor/metric位置后再解释作用 |
| [L82 subspace patching illusion, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) | 补丁可能激活未在normal-forward中起作用的路径 | 自然使用+功能输出连通及备份路径判别 |
| [L86 How to use and interpret activation patching, 2404.15255v1](https://arxiv.org/html/2404.15255v1) | §2–4、AND/OR、noising/denoising、corruption/metric的实用对照 | 双向N/R是两个独立作用读出，并作成组/补集比较 |
| [L87 Transformer Circuit Faithfulness Metrics Are Not Robust, 2407.08734v1](https://arxiv.org/html/2407.08734v1) | §3表1六个消融方法轴、§4–6及任务相对的faithfulness | 组件粒度、目标、替代值、token、方向、集合都属于“定位身份” |
| [L88 Slaying the Hydra, 2610.04017v1, 2026-10-02](https://arxiv.org/html/2610.04017v1) | §2 WISE组合效应、§3 JuntaLearner的分布/梯度集合学习、§4指标 | 组与witness对照揭示备用与多点相互作用 |
| [L89 AtP*: efficient scalable localization, 2403.00745v1](https://arxiv.org/html/2403.00745v1) | §3.1梯度局限、direct/indirect cancellation、§3.2漏检诊断 | 允许梯度预筛，但以真实QDQ及有限路径检查重要结构 |

L83–L85的图像DiT语义、视频IMAP与视频编码器时空SAE用于发现有标签/无标签函数坐标，保留其原任务身份。L86–L89是原本论文中**LLM电路方法**的来源；本轮提出的四边界、量化差项、三卡及Wan视频功能对应均为**FFNDS自己的ALG/OPEN方法学**。

阅读与检索日期：2026-10-27。以原arXiv HTML、JMLR正式介绍、官方ICLR论文页面和原生Github研究来源为主；本次没有运行作者实现或新模型。

## 10. 状态与交接

本研究方法学优先帮助Agent**提出新的可证伪功能、真实计算结构和最有信息价值的下一证据**。输出可映射到[定位卡](../agent_protocol/FUNCTION_LOCALIZATION_METHOD_CARDS.json)、[旧行为/函数/结构登记](../agent_protocol/FUNCTION_STRUCTURE_RESEARCH_OBJECTS.json)以及[主计算图](../../docs/98_端到端计算图与证据注解_20261025.md)。

**当前活动**：数学、现有证据、源码、原作者论文、科研讨论和图谱整理。**新CPU/GPU/模型/视频执行暂停**；实验许可由[ACTIVE](../ACTIVE_RESEARCH.json)负责，历史实验原始RUN/哈希保留。
