# FFNDS · 三条新函数路线的现行整合研究方案（M-A / M-B / M-C）

**2026-10-29 v3 · 实测驱动修订。** [完整评议与推导](../20261029/RESEARCH_ROUTE_REVIEW_AND_REVISION_20261029.md)及[125项标量/有限代数检查](../20261029/RESEARCH_ROUTE_REVIEW_SCALAR_AUDIT_20261029.json)：B0/CAL24/B1固定池、B-H、同head A0、B-F末层、A0全80与六视频均已有结果。当前A0视频明显退化；B-H H128准确率落后A0；B-F有局部小幅改善与逻辑MAC下降，SYS未证实。原始BF16末层固定范数转移是定点正证据，B-F学生则仍分子误差主导。下一步按第7节区分两种机制，A100 BF16与RTX PRO 6000原生NVFP4并行；QCFB先做可复用资产T0与有限归一化风险。本次只改研究路线，不新增ACTIVE运行。

> **平行创新交界层（2026-10-28，不替代三方法权威正文及现行M-B后续任务）：** [QCFB/Q-STAGE跨 QMeta×M-A/M-B/M-C 统一前沿](../20261028/INTEGRATED_QMETA_FUNCTION_EXECUTION_FRONTIER_20261028.md)与[typed prior crosswalk](../agent_protocol/QUANT_FUNCTION_EXECUTION_FRONTIER_CROSSWALK_20261028.json)具体推导原HiF4完整64组的动作闭包、条件有符号联合删除风险、量化误差交互和动作依赖Q2信息。它是研究种子，尚非任何原生部署结果或同行支持票。

> **2026-10-29 · INTEGRATED_METHOD_PLAN v3 · CURRENT RESEARCH PLAN / MIXED ALG+SRC+NATIVE+LIT+OPEN。** 此文件是三方案**现行权威整合入口**。首次2026-10-27讨论版本可由[原始合并提交dd21c539](https://github.com/Qesire/FFNDS/blob/dd21c539ca6695fb4b470fb21ba459234ff7a588/research/20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)审阅，旧版本时间点的推测不再作为当前操作指令。修订依据：[G14反向审计](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md)、[原独立评议](COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)、[五操作框架](FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)、[现行M-B数学快照](B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)和[历史A0四位置pilot](../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md)。研究方案和方法讨论不代替[ACTIVE研究与运行授权](../ACTIVE_RESEARCH.json)，本次编辑没有运行新的模型、实验或视频。

## 0. 研究目标、三层组织和现在的成果边界

**科学问题**：Wan视频生成FFN正常计算如何承载信息、时空生成行为和真实物理工作？能否通过**新的函数表示、合法先验/历史和可执行算子**在保留完整视频能力的同时减少实际计算？

| 层 | 科学身份 | 当前展开 |
| --- | --- | --- |
| **I / F / C** | 分别问信息何时/以何形式可得；正常函数与后缀动力学为何；执行怎样真实低成本 | 三个可并行的科研入口 |
| **O-A / O-B / O-C / O-D / O-E** | 选择需要算的量 / 改变函数表示 / 历史复用 / 计算代数 / 学习与验证目标 | 方法动作可组合；**稀疏是一种可能的结构/结果，不是独立路线** |
| **M-A / M-B / M-C** | 具体候选及原生实现 | **M-A0** 普通紧凑学生；**M-B** 后缀敏感低成本函数；**M-C** 整层编码直接求值。名称不与操作O-A～E混淆 |

**下一位方法研究Agent继续 M-B 的实际机制判别**：复用已完成的原生末层捕获、CAL24配对固定池、B-H/A0/B-F和原始BF16四角/固定gain结果。优先核查跨来源/深度的实际consumer和校准覆盖，分别检验分子预测与分母偏差。M-A0全80局部恢复及视频已完成，当前视频质量未通过；它仍是匹配局部基线。M-C保持独立开放；历史失败不扩展为整个家族失败。

**历史HiF4诊断边界**：LightX2V Wan2.2 I2V A14B，四步高/高/低/低、2个expert×40 blocks；post-Q1输入 $x\in\mathbb R^{M\times5120}$ → W1 (5120→13824) → GELU(tanh) → Smooth2/HiF4 Q2 (64→8→4) → W2 (13824→5120) → pre-gate $y$ → e5门控及BF16残差 → 后续attention、最终LN/head、flow和去噪状态。现有本地计算是fake-QDQ＋BF16 GEMM；真实native FP4与它的格式和耗时分别建证。4 denoise steps、21 latent时间槽和81 RGB视频帧是不同的研究坐标。

## 1. 历史起点与证据边界（最新结果见v3评议）

| 证据 | 实际观察 | 对设计的推动 |
| --- | --- | --- |
| [UB1 consumer20视频](UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md) | 20个**已暴露开发来源**，fresh Dense对照下RAFT均值6.894→1.152（下降83.28%）、二值dynamic6/20→0/20；imaging均值仅下降约1.38% | “静态接近＋动态响应大幅改变”是区分自然功能、可观测性/纹理及student轨迹的理由；**并未证明存在运动专属低rank方向** |
| [四位置CAL24来源能量](../20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md) | 固定whole64支持leave-one-source Jaccard均值约0.980–0.998；同位置已保存gate无来源变化 | 同source共同敏感新信息更可能来自**真实J_s和状态**，而不是重复平均常数gate² |
| [A0 compact6016 pilot](../20261028/COMPACT6016_RECOVERY_PILOT_RESULTS_20261028.md) | **已实际完成**4个(step,block)×energy94/random94，每arm每位置128次更新；train16/已暴露heldout8，energy94局部gated relative SSE1.061%–2.546%，random94为1.463%–3.649%；native narrowed Q2和8份回读权重 | M-A0可以恢复teacher状态上的局部函数，应成为**matched实验基线**。四site/旧来源≠80个函数、学生自由轨迹、完整视频或整端速度 |
| [M-B末层v1](B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md) | **ALG+SRC已完成**：final LN→调制→64维head的连续有限充分统计、Jacobian和finite identity；实际head权重/shape已有已核checkpoint资产 | B不必从盲目全5120维加权PCA起步：优先用真实末层readout建立能被合法原生有限差判别的函数坐标 |
| [C5合成A100 online](E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md) | 合成BF16 2:4 fused online相对dense仅约3.5%/0.7%中位数余量 | 任何“低MAC/稀疏/查表/低rank”都要加上观察、量化、packing、kernel、读表、布局、驻留及系统墙时 |

**验证立场**：上述新增的A0/NATIVE数字可以用于同一当前文档，但不同pilot和UB1的训练来源/状态不可直接并为一种显著性证据；原A54少量视频失配、source-level motion和C5执行证据均限定各自版本。

## 2. G14先验正式升级为构造条件，而非参考书目

对每一候选，必须写出**原生计算边→已知论文先验→数学或源码约束→新表示自由度→竞争解释→最小判别**：

| G14/文献输入 | 会约束哪条具体新方法 | 目前可用的研究修正 |
| --- | --- | --- |
| **T04–T06；L02 Superposition、L50 NerVE、L06 SliceGPT** | M-A/M-B/M-C都穿越W1/GELU/HiF4 Q2 | 原hidden坐标并非独立语义单位；非线性会重排谱；**只允许明确合法的重基/替代**，不能把任意hidden旋转穿过GELU和原生64-group Q2当作恒等 |
| **T07–T09；L82、L86–L89 patching/幻象/组作用** | M-B主张保留正常计算承担的重要方向 | 分开轨迹观察、原正常路径受控有限干预和替代函数可实现性；**能引发响应不自动说明该方向是原生正常承担者** |
| **T09–T12；L39 SVD-Cache、L21 HyCa、L60 SCD** | M-B敏感方向、可选O-C历史复用 | 三个不同属性：方向能量、下游作用、**时间可预测性**；低能快速振荡可以重要，保留、预测/复用、重算是不同运算 |
| **T02/T14/T16；L34 DIFS、L47少步时序塌缩、L56 Self Forcing** | M-A局部恢复、M-B训练目标 | teacher同状态准确≠student四步自由轨迹准确；若变更全网络架构允许attention职责迁移，须与**冻结attention**作不同estimand |
| **T06/T13；L71、L73 CodeGEMM、L74 SharQ、L76 LUT-GEMM** | M-B因子化、M-C LUT、O-D低位和O-A选择 | 大M Tensor Core GEMM与单token/GEMV码本fast path、真实QDQ与native FP4分开；以**实际线上计算边**计净省算 |
| **T08论文覆盖稀疏（现行图关键薄弱处）** | M-B末层门控、LN/head后缀 | 采用实际SRC + B v1导出的有限等式建立可检验方法；缺少外部同构论文时保留开放问题、针对性深读与强基线 |

**G14关系分类**：REQUIRES是实现/可辨识条件，CONSTRAINS是已知数学/数值限制，COMPETES_WITH是相同成本对照，PREDICTS_DIFFERENTLY是争议机制的对向预测。见[机器方法—先验交叉索引](../agent_protocol/METHOD_OPERATION_CROSSWALK.json)，图节点/论文证据不直接变成新模型已完成实验。

## 3. M-B：当前首要方法——从正常后缀到可编译函数

### B-0. 科学对象及已经证明的结构

基础假说：存在由正常FFN承担、下游具有重要作用的坐标，这些坐标能从合法的pre-W1输入等当前已可得信息以较低成本预测，且对整个视频质量有益。

将此拆成**三个独立可判别命题**：
1. **B-NATURAL**：方向在自然Dense轨迹及匹配原生有限响应中确有下游影响，区分patch-induced dormant backup；
2. **B-PREDICTABLE**：该方向对应的所需函数可由当前$x$、合法residual和conditioning推算，而不偷用完整teacher pre-gate y或将来step；
3. **B-COMPUTABLE**：新函数lower成实际BF16/Q2/factored/fused/native FP4等算子后，**全付费在线成本**优于强dense/紧凑学生，仍保持source-level完整视频能力。

B-NATURAL可以独立成为机制结果；B-PREDICTABLE＋B-COMPUTABLE才是可部署结构主张的核心，不应把这三者合成单一局部SSE门槛。

**B-MARGINAL对照**与**B-JOINT候选**分别为（固定当前写回gate、相同source/state、相同公共中心$\mu$、$R=I-P$）：
\[
E_m(P)=\operatorname{tr}[\bar G R\bar C R^\top],\quad
E_j(P)=\mathbb E_s\|J_sD_{g_s}R(y_s-\mu)\|^2,\quad
G_s=D_{g_s}^{\top}J_s^\top J_sD_{g_s}.
\]
[独立严格二维反例](../method_discussion/discoveries/E0/conditional-sensitivity-pairing-and-balanced-projection-20261028.md)证明边际与联合相反排序甚至可达100倍差；然而**在旧四位置CAL24同一个step/block的gate对来源固定**，真正新的配对判别需要变化的**后缀$J_s$与状态$y_s$**。真实多token后缀的$J_s$一般跨token，末层单tokenhead只是控制性锚点；上述目标是一阶敏感代理，不是BF16程序恒等式。

### B-1. 当前最强起点：block39最终head的有限充分统计

[既有B-v1已证明](B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)：固定该step最终head modulation，令$u=r+D_gy$为post-write状态，$P_0=I-\mathbf1\mathbf1^\top/D$、$z=P_0u$、$\sigma^2=\|z\|^2/D+\epsilon$、$T=W_{head}\operatorname{diag}(1+scale)$，则连续读出
\[
F(u)=Tz/\sigma+W_{head}\,shift+b
\]
由**64个线性head投影、mean(u)、||u||²**决定（最多66个标量）；一阶$J_u=\sigma^{-1}T(P_0-zz^\top/(D\sigma^2))$，真实离散BF16下应做原生有限差。此结论针对**每个expert的最后一个block和固定条件的head读出**，不是“整网只需66维”或原FFN+Q2可无条件绕过。

**将其转为两个真正可比较的新函数构造**：
- **B-H / readout-sufficient student**：预测目标是$\widehat q,\widehat m,\widehat n$，它们分别代表该末层post-write状态的$T u$、mean(u)、||u||²；预测器仅接收当前post-Q1$x$、已可得$r$、gate/time conditioning，产生同一head流形的读出。用真实student路径加上原生post-norm/head等价/近似求值，而不是从已付费完整W2得到这些统计后再宣称节省W2。
- **B-F / generic gated-output factorization**：每expert/block以新的可训练紧凑表示$\widehat y=\Phi c_\phi(x)$或$\widehat y=\Phi B_c Q_{2,\phi}[\mathrm{GELU}(W_{1,\phi}x)]$替代原pre-gateFFN，并保留原真实gate/residual。该因子化结构可用于其余78个非末层FFN，但效果与实际后缀权重需要重新验证，不能移植末层65量充分性，具体Q2缩窄/量化闭包需重新建立。
- **共同反例**：原生full y可由head读出压缩成几个标量，并**不**推出这些标量由post-Q1 x以低成本精确得到。尤其$||r+D_gy||^2$含残差交叉及W2二次项：若仍通过fullW2或巨大的Gram求得，B-H只是离线分析而不是效率方法。

将**时间预测/历史复用O-C**作为独立后续组合：四步high/high/low/low专家变换，只有历史同形状且条件、时间和数值可用时才允许使用；L39低能振荡残差的reuse现象不能自动传到当前Wan。

### B-2. 配对竞争解释及阶段结果

| 竞争机制 | 不同预测 | 首个有分辨力的动作 |
| --- | --- | --- |
| 普通PCA/平均敏感度已足够 | 同rank/同source独立fold上，B-JOINT没有稳定改进 | 在原生真实same-state J/y捕获后对PCA、随机、marginal、配对、打乱配对作等rank比较 |
| 真正联合下游依赖存在 | 来源配对的敏感方向与打乱来源配对会改变真实末层finite response排序 | **先取得同一状态u/head/flow，进行中性重放，再冻结有限差方向和幅度** |
| 低能高敏但学生难预测 | oracle低rank代理好，pre-W1学生难以保留该结果 | matched模型宽度、训练预算、quantization与输入可得性比较；避免把oracle写成部署成果 |
| 仅是head/LN放大而非自然功能承担 | 同状态post-write有限差有变化，但自然trajectory/完整视频不同或实验基线效应翻转 | 正常状态+成对有限反事实、多个读出与来源；随后独立完整40×4新学生video |
| 光流可观测性、量化/状态漂移是主因 | RAFT与语义ROI/相机行为不一致，或student rollout偏差远大于同状态head替换 | 完整free-rollout/来源级运动、纹理及attention/flow状态拆分 |

**现行方法进度，不重置**：
- **已完成**：数学与源码身份；B0原生捕获/neutral；CAL24同坐标96flow；B1固定16D/rank4/18候选；B-H H128、同head A0和B-F两个末层expert pilot。配对与打乱同选PCA，不能据此判定一般paired机制无效。
- **已有局部结论**：B-H对A0均0/8胜；B-F head误差较A0低约3.1–4.3%、逻辑MAC为59.26%，但主要剩余项是分子预测。原始BF16保留headspan投影相反地以分母误差为主，source0冻结gain转移到source12有实际改善。
- **后续开放**：跨深度真实consumer、固定gain与动态补全的净价值、学生状态上的有限损失、独立来源完整视频和实际FFN/系统成本。末层65统计不自动是中间attention后缀的充分表示；新工作逐scope登记，不重复已完成B0/BH。


### B-3. 编译与实际成本的两个分支

设$D=5120,h=6016,r$为输出中间rank：

| 原生模型方法 | 理论主MAC/token | 部署含义 |
| --- | ---: | --- |
| **M-A0 / B-FOLDED** | $2Dh=61,603,840$（dense原13824宽为141,557,760） | W2折叠$D\times h$后**在线MAC与A0相同**；折叠可能仅改善表示与参数优化 |
| **B-FACTORED** | $Dh+r(h+D)$ | W2路径rank限制$r<Dh/(h+D)\approx2766$才有理论MAC减少；多1次GEMM/kernel与量化舍入 |
| **B-FACTORED, r=512** | **36,503,552**（A0的59.26%） | 是ALG预算，需在$M=31668$、真实QDQ、存储、launch和matched BF16下取得SYS证据 |
| **B-H末层直接读出** | **取决于实际学生和统计构造** | 精确连续head sufficient stats≠计算复杂度界；不能先完整W1/W2/Gram然后声称跳过 |

原生BF16、HiF4 fake-QDQ、进一步native NVFP4/MXFP4都应拥有独立数值等价合同。新O-D改写若与M-B的科学函数发现正交，可单独作为计算结果推进，而非必须等视频因果全证实。

## 4. M-A：固定预算紧凑函数，不再作为没有实验的原始假说

**M-A0普通学生已实际验证局部恢复**：每expert/block width=6016=94×64，相比dense两个主GEMM占**43.5185%**，与旧Affine54$D^2+2D\times3456$等MAC。80套W1/W2两矩阵总参数4,928,307,200；rank32 LoRA覆盖两矩阵约57,016,320增量参数。**已有pilot**比较能源选组和random选组，在4site、train16/heldout8已暴露source相同预算得上述局部恢复；该结果与完整Wan动态能力或所有位置的恢复是不同命题。

**M-A1真正新机制选择空间**（只选择一个主设计变量做每版验证）：
- **nonlinear-capacity allocation**：相同全模型总MAC但$h_{expert,block}$不同；对L50的nonlinear spectra、L34的step异质性和M-B的suffix作用提出**相反宽度分配**预测，与均匀A0相比。
- **rollout-aware adaptation**：冻结FFN-only LoRA路径，与保持其余变量一致、改用同预算student-prefix/terminal rollout监督比较；L34/DIFS的输出rollout是最强同类先例，独立新贡献需额外明确原Wan量化边界、来源/行为约束。
- **whole-model role transfer**：如果允许attention/norm参与训练则是另一模型类和另一estimand，不可把全网络共同适配的收益归于FFN独有的函数压缩。

**A0下一步**：保留全80恢复、字节审计及六视频的真实资产与负结果，作为函数/状态/目标机制和M-B/C的匹配对照。当前energy94视频在两个暴露来源的RAFT仅为Dense约13.5%/14.5%，MUSIQ均值46.94对76.02；不再默认重复全80训练或将其列为质量已通过候选。新的目标/状态变体先作有判别力的有限比较。

## 5. M-C：直接预测完整量化后非线性函数，而非另造线性GEMM查表

构造$\widehat f(x)=U d(k(Rx))+b$，必要时包含有限交互码表；这一方案是**O-B函数替代＋O-D编码求值＋O-E训练验证**。最先明确三个独立自由度：

1. **输入信息充分性**：若$Rx_1=Rx_2$且真实$Q2/GELU/W2$函数输出不同，编码不能同时精确表达两者。平方损失最优预测器不可约平均风险为$\mathbb E\operatorname{tr}\mathrm{Cov}(f(X)\mid RX)$，离散码$k(RX)$只会进一步缩小可分辨输入。
2. **输出空间限制**：只输出$\operatorname{col}(U)$时，中心化teacher输出均方重构残差至少是协方差尾谱$\sum_{j>r_o}\lambda_j$。门控与后缀作用属于另一个损失/可辨识问题，不能直接用Euclidean尾谱断言视频表现。
3. **真实线上物理代价**：编码投影、量化/查找、交互表、输出展开与GPU内存层级全付费；$K=256,r_o=256$的每二组BF16表=**32MiB/expert-block**。CodeGEMM原作者公开实现的快速查表路径主要在单token，不能据此推出大M真实Wan快于Tensor Core dense。

**新科学设计自由度**：在保留input sufficiency的情况下，把码book容量分配给经过**真实量化后非线性耦合**且下游作用高的函数交互；同rank连续小MLP、A0、最强公开LUT-GEMM同类算子为匹配强臂。M-C目前没有Wan训练/视频/SYS，先用现存$x/y$资产（真实身份先核对）与输出谱、编码冲突及驻留成本给出最小判别，再决定是否登记实验。它可以独立于M-B成功或失败。

## 6. 三条路线共享的方法协议：先验、测试和科研分层

**必须对每个候选冻结的对象**：
1. **操作指纹与源图**：I/F/C科学问题；O-A～E主/辅动作；G14 T03–T16定位；真正动作在信息可得前还是后；原Wan shape、quant、gate、后缀状态与候选替换后的shape/参数/计算；
2. **DERIVATION_GRAPH**：证明确切等式、边界和反例（PROVED/CONDITIONAL/OPEN），具体source/原文先例的**REQUIRES/CONSTRAINS/COMPETES_WITH/PREDICTS_DIFFERENTLY**关系；
3. **独立来源资产**：image-cluster级训练/验证/未暴露确认，seed/row/step嵌套；真实post-Q1/pre-gate/gate/Q2、teacher/student同坐标、时间与expert身份、原生producer SHA。旧CAL24的6144行不是6144独立源。
4. **相同条件比较**：各自recipe内部的原始BF16 Dense或真正NVFP4 Dense（HiF4单列诊断）、M-A0同训练预算、同rank随机/PCA/marginal/paired、相同用时/数据/算子边界的最佳近邻；oracle teacher target与在线student分开报告；
5. **分阶段证据**：ALG证明→SRC/numeric neutral→source-level finite/local→真实完整40×4 student自由视频（主体、背景、成像、连续motion、相机/ROI、prompt）→在线FFN及整机墙时/VRAM/系统Pareto。每个阶段回答自己的命题，不把数学实验先验变成所有创新种子的强制GPU门；
6. **竞争机制**：真实作用 vs patch dormant；低能高影响 vs source bias；数量敏感 vs student预测不足；QDQ数值 vs 后缀放大；RAFT可观测性 vs 真实动作改变；MAC减小 vs packing/访存抵消；
7. **开放性**：新研究可以在任何I/F/C位置提出新的表示或历史复用/算子组合，不要求从M-A/M-B/M-C三者选一个；作为开放新方法独立写入discovery、竞争/审阅按QUEUE，运行按ACTIVE。

## 7. 当前研究顺序（2026-10-29现行下一步）

| 轨道 | 下一项与最小对照 | 状态及边界 |
| --- | --- | --- |
| **M-B / A100原始BF16** | 核验CalibrationV2跨来源/深度及实际consumer；同一候选bare、冻结fixed gain、合法动态norm修正、同成本分子修正 | **READY_STATIC设计**；low39一转移来源成功不等于全80；新模型scope未注册 |
| **B-F函数预测** | 固定基底/输入下先比Y目标与finite/head目标，再单独改变合法输入；保存teacher投影参考与A0/BH强对照 | **PARTIAL_NATIVE**；区分表示、信息、训练目标；大oracle差距不单独证明瓶颈 |
| **状态与视频** | a+p+c四角及signed交叉/step权重；同预算teacher/student-state；需要时有界完整视频，最终独立image-cluster确认 | 当前A0六视频质量未通过；新候选VIDEO/SYS仍OPEN |
| **RTX PRO6000原生NVFP4** | 独立R0访问/模型/packed后端→R1数值→R2 Wan Dense→R3候选→R4完整视频/速度 | 与BF16并行；遵循[现行双轨政策](../agent_protocol/BF16_NATIVE_NVFP4_EXPERIMENT_MIGRATION_20261028.md)；历史primitive不等于当前设备已分配 |
| **QCFB/Q-STAGE** | 已保存V2/W2/gate绑定后T0，有限归一化/尾部风险；T1合法前置信息与净价值；后端特定闭包 | 数学/资产并行，不把缺失packed Q1作为全部T0阻断，不直接移植HiF4 group64 |
| **M-C** | 信息/输出表示、真实M与查表/编码成本的独立最小判别 | OPEN，不要求其他方法先成功 |

**核心决策**：从已完成实测继续，优先回答“哪类读出误差、在哪些实际状态和消费者上、可被何种低成本函数修复”。[完整评议](../20261029/RESEARCH_ROUTE_REVIEW_AND_REVISION_20261029.md)给出方差联合动作恒等式、归一化反例、量化global-scale闭包条件与最小对照。模型/GPU运行仍依ACTIVE冻结scope；这份路线更新没有改变历史RUN、资源授权或独立评议票。
