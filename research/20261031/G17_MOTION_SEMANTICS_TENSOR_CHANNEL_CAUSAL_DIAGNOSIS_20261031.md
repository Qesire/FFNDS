# G17 · 从直接张量/通道对比定位运动—语义损失：证据审计、四边界因果诊断与低成本组梯度

2026-10-31 · SCIENTIFIC_DESIGN / SOURCE_AUDIT / ALG_DERIVED / NEEDS_INDEPENDENT_EXPERIMENT_REVIEW / NOT_EXECUTED

**本轮目的**：回应G16外部评议并重排机制研究优先级：先解释已实测的运动衰减与FFN功能差异，再决定是否值得扩大Q-MARG。研究对象只限Wan2.2-I2V FFN（不优先替换Attention、step或原生T2V模型）。**现有文档/聚合指标不等于已经读取并比较原始Dense/Sparse时空通道张量；下文区别真实历史测量、ALG机制与拟议新producer。** 基础：[G16](G16_SICR_V3_FUNCTIONAL_MARGINAL_STATIC_FFN_REVIEW_20261031.md)、[G13 四边界功能定位](../20261027/G13_FUNCTION_LOCALIZATION_METHODOLOGY.md)、[S2 21τ统计](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)、[A5-HWG fullM](../20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md)。

## 1. 真正的科学问题和可用历史事实

H1（**选择缺失**）：所遗漏的post-Q2群组在运动主体/物体交互ROI内具有低全局能量但高有符号、时空结构化的写回，对最终动作重要；反面：ROI差异只是token密度、噪声、纹理或光照分布造成，功能不是高选择性通道。
H2（**历史冻结**）：纯复用旧Y、S0/SICR只更新小组，使**跨去噪step**需要的运动改变未进入当前写回；反面：FFN局部变化足够，运动衰减主要来自student prefix、下游attention的状态迁移或数值重排。
H3（**损伤传播**）：local少量同状态错误与video动作坍缩不成比例，后续层/Low阶段/gate/attention在特定语义状态放大它；反面：问题已经全部发生在当层局部输出，后缀只线性传输。
H4（**组间协作**）：移动物体的函数方向由多个组signed交叉项承担；单通道大范数高不等于真实贡献；反面：所谓“运动组”效应源于随机组数、总输出能量和干预造成的分布外激活。

**真实历史观察**（没有本轮新增计算）：
- [UB1 20源](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)视频RAFT 6.894→1.152（−83.28%），dynamic_degree官方binary 6/20→0/20；已证明该方案运动衰减，尚未定位因果组。
- [BF16 value10 vs DCAC10 20源](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)Dense RAFT 7.0580、value10 3.1805、DCAC10 3.1007；训练AC相对SSE 0.051229→0.047267，**AC局部改善未转成运动收益**。不是“所有高频保留无效”。
- [S2 21 latent τ、10图全160 Dense](../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)gY去τ0 lag1≈0.048，r95≈16.96；固定xy且稀疏8位置，未随物体流移动；不能据此判断真实运动在高/低频或通道本身是否稀疏。latent τ0异质性不可与τ1…20混合。
- [P64 High/Low 两开发源](../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)Low-only相对Dense imaging -2.120/-1.483，H-only -0.008/+1.912；这是**固定HHLL时段策略效应**，expert和时间无法因果剥离。
- [A5/HWG HiF4三源E0](../20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md)全80历史对×58策略，共13,920条指标；1344行matched postQ2与源边界逐bit校验。提供局部signed/组能量/空间tile的强比较资产，但3源均已曝光、无候选完整视频，**随机行不足以直接标定运动主体语义**。
- [source898/96 ROI](../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md)已完成Farneback 5×5/远距低流静态邻域前审，但898最大流落在背景植物，不能冒充主体ROI；96 HiF4 A5 matched capture失败，不能对该来源补造Q2证据。
- 原始fullM matched、全τ fixed xy及部分历史视频位于远端HF/仓库索引；**本次仅通过GitHub审计文本和来源路径，没有取得原始大张量payload或运行新张量统计**。数据生产与访问清单必须逐artifact核对，禁止把存储的摘要当作新通道结论。

## 2. 严格区分两种“时间”，两种对照状态

索引k=0..3是去噪步HHLL；τ=0..20是video latent时间槽；f=0..80是81个decoded帧（真实VAE时间映射须核源码）；l=0..39层；x=(τ,r)为latent时空坐标；b=1..216为hidden K64组，c为单hidden通道。注意τ0是初始条件槽，不参与普通latent邻差统计；视频时间与去噪step绝不可混为一谈。

**分离的差分**：
\[
 D_{\mathrm{step}}h_{k_1,k_0}(\tau,r)=h_{k_1}(\tau,r)-h_{k_0}(\tau,r)
 \quad\text{（同expert两次不同去噪状态）},
\]
\[
 D_{\mathrm{video}}h_k(\tau,r)=h_k(\tau+1,\pi_{\tau\to\tau+1}(r))-h_k(\tau,r)
 \quad\text{（同一步中物体运动对齐后的latent时间变化）}.
\]
\(\pi\)来自Dense视频tracking/flow映射到**经确认的latent patch网格**，含遮挡/置信度与camera residual；没有可靠对应时仅标固定xy差分，不叫motion-aligned。两个D不具有交换假设，二阶交叉差分可独立观察。

**两类对照**：
- TF（TEACHER_FORCED / 同输入）Dense FFN完整计算 vs 在**完全相同当前输入/历史donor/gate**上的post-Q2静态组遗漏，定位该位点产生的本地误差；不是自由视频。
- FR（FREE_ROLLOUT / 学生自由轨迹）自然Dense路径vs真实Sparse学生，自该源初始noise配对运行全过程；“Dense张量−Sparse张量”此时包括全部上游状态迁移。必须额外做同学生输入的Dense counterfactual \(f_D(\tilde x)\)，才能把当前局部函数误差与state drift分开。

**精确差分分解（各边界同shape）**：
\[
 f_S(\tilde x,\tilde C)-f_D(x,C^D)
 =
 \underbrace{f_S(\tilde x,\tilde C)-f_D(\tilde x,C^{D|\tilde x})}_{\text{相同当前输入、合法对应历史下的算子/缓存差}}
 +
 \underbrace{f_D(\tilde x,C^{D|\tilde x})-f_D(x,C^D)}_{\text{输入/历史轨迹差}} .
\]
其中 \(C^{D|\tilde x}\) 必须由**真实可执行的Dense counterfactual历史定义**，不是任意插值；如果Dense函数本身无cache可依赖，则可去掉C参数并仅对当前f_D(x)分解。不要把历史和当前混合归为同状态“纯算子误差”，要单独注册如何构造完整合法前缀。

## 3. 实际逐张量/通道分析——从不需要语义标签的结构到有监督的可检验意义

比较的**真实图边**：pre-W1 u、Q1 code/scale/重建、W1输出v1、GELU h_preQ2、post-Q2 h、W2 pre-gate Y、当前gate g、postwrite g⊙Y、后续Attention receiver、当前step预测flow/velocity、下一step latent及最终video。仅在payload真实存在时计算该边；要留source/seed/k/expert/l/τ/r/数值族/当前student-vs-Dense身份与dtype。

### 3.1 单通道结构（候选定位，并非“运动神经元”判定）

对同一步同层、同固定匹配token集合，记录：
- \(h_c\) 与同专家跨step \(\Delta_{\rm step}h_c\) 的mean、MAD、p50/p90/p99/p999、正负比例、零码率、Q2饱和/尺度统计；按source/τ0/τ1:20/主体/背景/相机位移分层。
- 视频时间的固定xy与光流对齐时间差的能量 \(E_c^{xy},E_c^{flow}\) 和时间lag结构；时频分解需保留复相位/方向，不以平均FFT幅度代替运动。
- \(P_{\rm obj}(h_c)-P_{\rm bg}(h_c)\) 的来源外条件效应，必须匹配照度、token数、速度幅值、纹理/遮挡；标签ROI由冻结的Dense-video tracking/segmentation预先产生。任意K64 group可能混合多个功能，报告选择后偏差和source稳定性。
- 视频运动方向\((u,v)\)、光流散度/旋度、相机全局变换 vs 物体残余、加速度/事件onset等分开；纯RAFT幅值不能确定动作语义。

### 3.2 FFN W2组贡献及**有符号取消**

对同专家合法两步，令post-Q2各状态hidden为h0,h1，**pre-gate**组增量（不重复gate）：
\[
 P_{i,b}= (h_{1,i,G_b}-h_{0,i,G_b})W_{2,i,G_b}, \quad
 \delta Y^{pre}_{i,S}=\sum_{b\notin S_i}P_{i,b}.
\]
当前写回缺失\(E_{i,S}=g_{1,i}\odot\delta Y^{pre}_{i,S}\)。post-Q2的差必须先分别Q2后再相减：\(Q_2(h_1)-Q_2(h_0)\ne Q_2(h_1-h_0)\)。对于真实Sparse遗漏（预Q2 group跳算），共享scale/重建还可能改变已保留组，应采用G13 Q边界闭包诊断，不能假定上述post-Q2诊断与preQ2原生操作天然一致。

对对象/背景mask m，定义组和跨组内积:
\[
 E_{b}^{m}=\sum_{\tau,r}m_{\tau,r}\|gP_{i,b}\|^2,\quad
 G_{bc}^{m}=\sum_{\tau,r}m_{\tau,r}\langle gP_{i,b},gP_{i,c}\rangle.
\]
准确分解\(\|\sum_{b\notin S}gP_b\|_{m}^2=\sum_{b,c\notin S}G_{bc}^{m}\)。记录正/负cross、净输出能量、与Dense视频动作ROI位置关系、源间方向稳定性。**高能但彼此抵消并不表示单独保留哪一个具有正的最终视频价值**。

### 3.3 时间相位、运动对齐与语义状态

同一候选组在固定xy时间AC高，但沿物体运动轨迹对齐后AC可能低；反过来也成立。定义：
\[
 E_{\rm fixed}(b)=\sum_\tau\|V_b(\tau+1,r)-V_b(\tau,r)\|^2,
\]
\[
 E_{\rm warped}(b)=\sum_{\tau,r}q_{\tau,r}\|V_b(\tau+1,\pi(r))-V_b(\tau,r)\|^2,
\]
附带valid mask、flow方向、参考Dense运动方向及timestamp map。这里V_b可为postwrite \(gP_b\) 的latent格局，**是内部响应结构，不是帧级光流**。必须比较移动主体、静态背景、因相机移动而有背景flow、光照变化控制；若没有对应，就只写**固定位置时序响应**。

对于具体语言动作（例如鸟挥翼、鱼游动、相机推进），先冻结prompt动词、动作对象mask、起止时间、方向和可观察行为。定义对象/事件在video中的真实跟踪或视频语言读出；然后看组响应是否具有**自然变化、跨源迁移、有限干预中的选择性**。只见到高相关不能命名motion semantic channel。

## 4. 低成本全层通道归因：修复G16 double-gate并消除朴素逐组W2前向

**更重要的新ALG推导**（适用于固定W2线性输出、pre-gate边界、给定参考学生/Dense状态与可微读出）：

令 \(f_q\) 是末latent/定义好的连续功能读出，\(Y_i=h_iW_{2,i}+b_i\) 为第二次同expert调用的**pre-gate W2输出**。定义
\[
 A_{i,q}=\nabla_{Y_i}f_q\in\mathbb R^{M\times D},\qquad
 B_{i,q}=A_{i,q}W_{2,i}^\top\in\mathbb R^{M\times H}.
\]
\(A\)已自动把当前e5门控和**所有真正后缀**的局部导数包含一次；绝不能将g再乘到pre-gate P组上。对所有216个K64组一次分摊：
\[
 \boxed{z_{i,b,q}
 =\sum_{m=1}^{M}\sum_{c\in G_b}
 (h_{1,i,m,c}-h_{0,i,m,c})\,B_{i,q,m,c}.}
\]
证明：Frobenius双线性内积恒等式
\[
 \langle A,\Delta h_{G_b}W_{2,G_b}\rangle_F
 =\langle A W_{2,G_b}^{\top},\Delta h_{G_b}\rangle_F.
\]
因此所有group共用**一份W2 transpose backprojection**，再按group streaming求乘积和，而不是为每个组materialize一块M×D输出并另做一个W2 GEMM。若通过自动微分拿到post-Q2 hidden梯度，该backprojection可能已由后向W2计算提供，需核真实autograd/显存与checkpoint路径，不能重复算作免费。VJP仍需高代价全网络后缀与可能的历史缓存重放；先pilot少量k/l/ROI而非全80×216。

**数学身份**：z是当前状态下关于pre-gate Y的导数对冻结组post-Q2增量的**一阶方向导数**；不表示分组跳算后Q2共享scale会保持此Δh，也不表示最终VBench指标可微；用真实finite group intervention检查不光滑QDQ和远离参考状态的影响。若观察的是**首次调用的历史缓存错误**，必须从增广态传播包含第二步读取该缓存的路径，不能把仅第二次Y节点的A误当为第一次缓存路径敏感度。

**原版G16 §2.2的纠错**：它先定义v=g⊙ΔhW2又说P包括gate，潜在重复gate。当前G16已于本轮修复为pre-gate v及pre-gate P。本G17只使用一种边界身份，防止两次计数。输出基于Q2后固定h，preQ2跳算动作需单独走G13 Q闭包。

**关于成本**：当前M31668、H13824、D5120、一源80位点的朴素每组显式输出约2MHD×80≈359 TFLOPs（一次整个W2投影级，非不可避免的下界）。转置adjoint每个功能读出仍需全网络backprop及W2^T，量级和峰值需真实测量。静态前向功能归因不需要为**每个组**进行后缀反向，然而读出数p、新来源n、student状态/τ布局和VJP存活图开销仍可能高，故将Q-MARG降为少数候选组的验证工具，先看直观局部诊断。

## 5. 四边界的干预梯度：把“运动语义”从相关提升为限定因果

阶段A（只读tensor，DATA_DIAGNOSTIC）：全40层×4步尽可能读取原生matched数据，先按每源计算不带语义的通道/组AC、DC、signed格局；然后在已有真实视频可有把握跟踪的ROI上计算结构差异。特别对 source898 bird/wing、source96 camera（HiF4 96 Q2缺失）、source979已曝光发展源作**功能不同的反例**；不得从高flow ROI事后认定运动主体。先保留每源热区与原始region图，不泛化。

阶段B（同输入函数差，必须另审真实新执行）：选择少量对照source×step×layer×region×K64组，分别干预：
- **O** post-W2 pre-gate 输出：把当前学生指定组贡献修复成合法Dense同状态贡献；sham原值、等norm随机方向和同组数random/energy/signed强对照；
- **D** 固定post-Q2 hidden：将选定通道替换成匹配合法Dense donor，保持其余h、W2、当前g，观测后缀；明确这是**受控干预边界**，不是preQ2可执行跳算；
- **Q** 修改真实pre-Q2 hidden后重走Smooth2和真实Q2的**依赖闭包**，记录保留组被共享scale影响的连带项，单独比较Q/D差；
- **P** 让后续40层/后续去噪全自由传播 vs 固定某一真实receiver或witness路径；比较第一接收者/当前step/下一step/final视频输出。不能仅看末层或同输入单层。

**双向修复**：在Sparse student state把某组恢复至Dense同学生输入得到的响应；同时在Dense state实施对应稀疏删除，测恢复与损伤是否方向相反。采用配对source×seed、冻结token/τ/ROI、单组/K64组群及其联合、相同group count/no-op/sham/dose-response；如果移植导致明显OOD（h/g/Q2尺度差），只报告该受控替换效应，不能称自然功能承担者。

阶段C（真实动作因果，必须另审完整视频）：针对已观察到的动作、光照、相机、物体出现/遮挡和语言文本条件构造**目标与非目标双读出**。同noise/prompt，完整学生自由视频分别比较主动修复/删除后的主体轨迹、相机背景flow、对象交互、动作发生时间/方向、时序闪烁以及八轴原版VBench合法值。仅当自然Dense响应存在、干预引起**选择性的动作变化**、在其他来源/seed重现、非目标内容保持较好，才能有条件称“与动作语义有关的分布式功能方向”，**不能仅凭通道相关命名运动神经元**。

## 6. 资产可用性与具体补采合同

| 资产/证据 | 当前能直接回答 | 不具备的决定性量 |
| --- | --- | --- |
| 10图×160 Dense包21τ统计 / raw BF16 | 固定xy AC/lag/频带、τ0异质、按位点层间变化，存在实际保存原张量和SHA | 随移动主体的真实连续ROI；对应学生state；完整因果组视频 |
| A5/HWG 3图×80 matched步对×58策略 / HiF4 | 局部组输出signed Gram、K64 occupancy、postQ2匹配抽样行与静态支持，按源/层/专家复算 | 运动对象精确τ/r轨迹与全M真实student state；96 Q2失败采集 |
| UB1 20来源 Dense/consumer配对视频 | 最明显运动坍缩现象、source-level 8轴与RAFT、困难源分类、行为候选 | 与配对视频完全同RUN的全40×4 FFN pre/postQ2、gate、学生input、选组局部张量 |
| P64 source898/96 4臂实际视频 | High/Low策略对照与已观察动作/相机画面、连续RAFT五窗、contact sheets | 对应same-state逐通道原生tensor完整追踪与动作标签验证 |
| ROI预审 898/96 | Farneback运动极值/远距控制，识别背景误标风险 | 验证后的主体/背景/相机拆分跟踪，视频81f到latent21τ映射和遮挡置信度 |
| G13 O/D/Q/P记录 | 有原则、边界、条件因果与signed补集差代数 | 新的真实介入、后缀receiver/学生视频RUN身份 |

新增捕获建议**不是全量永久存储两次M×H所有层**：
1. 先保存全160位点轻量各通道/组统计：数值范围、signed energy/交叉sketch、DC/AC/flow-aligned差、对应source×layer×step×τ×ROI计数，chunk streaming，不额外长期复制所有h。
2. 原始指定source/seed/候选site与ROI保存post-Q2 h0/h1、当前Q1/Q2真实量化码/尺度（可得才记录）、gate、preW2 Y、pre/postwrite和下一receiver、真正学生prefix哈希；确认完全匹配model/conditioning/seed/scheduler。
3. motion semantic ROI由**已冻结Dense完整81帧**产生：主体track、背景相机运动估计、遮挡、对象关系与动作段；使用真实VAE时域下采样核/patchify映射到latent21τ，不能直接把f除4近似当精确卷积中心。
4. 原先局部组统计强候选与弱候选在同样物理K64预算下进行少量有限组干预，先CPU不含模型的代数自检，再依独立真实Agent预审注册新GPU数据/视频。
5. 所有source split按image cluster，τ/row/frame/seed作为嵌套；898/979/96/1040及现有10/20来源均按历史曝光用途登记，不用于新盲确认。

**单个最小判别**：source898 bird(真实主体运动)与source96 camera(相机运动背景)，每source选一个High与一个Low代表层，先用已有视频固定主体与背景ROI；再在那些层上对历史原生有matched h的真实来源做O边界限定干预。若缺对应同RUN tensor则**先只读回已有asset并登记MISSING**，随后新审采集。对照每层两个组（local signed高/低、相等输出范数随机）、O/D/Q/P和sham，以同初始noise/模型/数值族配对记录stage boundary与最终video。此最小设计用于机制**开发**，绝不是来源外功效确认。

## 7. 前沿文献及与本研究的关系（仅作为机制先例）

- [Motive: Motion Attribution for Video Generation, ICML2026](https://proceedings.mlr.press/v306/wu26aq.html)：运动加权/掩码的梯度数据归因，证明可以将motion与appearance的评价对象分开；其归因单位是**训练数据片段**，不是FFN K64通道、也不是本仓已有实现。
- [AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bahmani_AC3D_Analyzing_and_Improving_3D_Camera_Control_in_Video_Diffusion_CVPR_2025_paper.html)：相机信息在部分DiT层中可probe；说明层分工是可检验机制，不证明Wan FFN层已自然分离相机与对象运动。
- [FloVD, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jin_FloVD_Optical_Flow_Meets_Video_Diffusion_Model_for_Enhanced_Camera-Controlled_CVPR_2025_paper.html)：背景光流与相机/对象运动的区别；可用于建立相机残余与主体track的读出对照。
- [Motion Concept Unlearning in Video Diffusion Models, arXiv2609.36832, 2026-09-29](https://arxiv.org/abs/2609.36832)：通过Wan2.1-T2V等模型的动作概念条件干预发现cross-attn与时间编码的不同作用。说明**跨Attention的动作语义**需要跟踪，不能因为本项目研究FFN就假定动作功能由FFN单独承担。
- [Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bitrate-Controlled_Diffusion_for_Disentangling_Motion_and_Content_in_Video_ICCV_2025_paper.html)：内容/运动分离和离散瓶颈的先例，提供竞争方法但不是原Wan自然FFN channel已解耦的证据。
- [Causality in Video Diffusers is Separable from Denoising, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Causality_in_Video_Diffusers_is_Separable_from_Denoising_CVPR_2026_paper.html)：在**causal autoregressive视频扩散**中发现层/去噪分工；其架构与本项目非AR Wan4step不同，不可照搬其层位置结论。

## 8. 判定顺序、失败标准与G16修订关系

**本轮实际完成**：核对已有仓库公开文本、来源/作用边界、历史已完成质量；推导W2-transpose所有组一阶归因恒等式；定位G16 pre/post gate双计问题并在G16修复；提出具体张量/通道/ROI/干预对照。**未**取得HF全量payload、未执行新tensor统计、未运行任何GPU/模型/新视频、未验证任何motion semantic组或native速度。

**现在的科学顺序**：
- D0历史资产直接张量/通道**read-only**比较，明确来源、数值路径、坐标、时间差定义；高ROI错配立即修正，记录真实缺口；
- D1用固定pre-gate边界与W2^T做小范围post-Q2组影响诊断，先验证直接有限差，不遍历80×216；
- D2申请真实少量O/D/Q/P因果组干预，区分本地删除、量化闭包、门控与后缀传播；
- D3若动作语义因果证据成立，再修改Signed-HWG/Qmeta的静态组选组规则及Low阶段预算，接V1 E2缓存原生系统门；这些轴可并行设计，不把D0作为V1可执行工程验证的数学前提。
- D4完整视频机制确认/来源外确认和原生成本，决定论文主贡献是否是motion-semantic static FFN action conservation或仅缓存系统优化。

**停止/反例**：
(1) 高相关但有限干预无选择性动作变化→仅FUNCTIONAL-OBS；
(2) postQ2修复有效但合法preQ2 Q2闭包失败→不具可执行FFN动作；
(3) 少数组修复也能恢复RAFT但主体轨迹/事件不对→并非运动语义保真；
(4) 信号只在曝光source出现→重新采集、不得来源外宣称；
(5) 学生轨迹整体漂移/Attention承担动作改变导致局部channel归因不稳定→保留为状态依赖函数，停止“固定运动通道”叙事；
(6) VJP成本过高、不优于仅局部signed统计的少数动作直接有限试验→不要扩大Q-MARG；
(7) 全视频动作有收益但真实FFN/system wall未下降→机制研究有效，暂非部署省算。

遵守[每新scope独立预审](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)：真正不同Agent评议、作者修订/recheck、ACTIVE新冻结scope、RUN/EXIT收据才能执行新GPU/CPU科学实验及视频；本文件不代替这些许可。