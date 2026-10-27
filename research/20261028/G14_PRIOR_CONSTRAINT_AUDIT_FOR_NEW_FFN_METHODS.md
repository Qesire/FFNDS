# FFNDS · G14全论文计算图对新FFN方法的反向先验审计

**METHOD_SNAPSHOT_REVIEW / 2026-10-28 / SRC+LIT+ALG+OPEN。** 基线仓库研究对象为LightX2V Wan2.2 I2V A14B四步D-Q、两个high/low expert、每expert40 blocks、native BF16 GEMM与HiF4 fake-QDQ。本轮回看[99项论文到16个真实计算边的G14映射](../agent_protocol/PAPER_METHOD_FOCUS_GRAPH.json)、[202节点/413边原生图](../agent_protocol/GRAPH_AGENT_BINDINGS.json)、[五操作×I/F/C映射](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)、[三候选独立评议](../20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)以及[下一Agent首先研究M-B的任务](../agent_protocol/NEXT_AGENT_B_SENSITIVE_OUTPUT_FIRST_20261027.md)。从既有计算图中找到**真实数学约束、尚未利用的可迁移先验、潜在近邻先例和能判别新方法的构造**。这是静态审计；没有进行新本地模型推理、训练、GPU算子、缓存或视频实验。不更改ACTIVE、历史RUN或其他Agent同行评议。

## 1. 总裁决：不是整体违背先验，而是尚未把图谱转换成方法的必要条件

**有直接张力、应在新结构冻结前解决的事项**：
1. **M-B**以pre-gate输出子空间/局部Jacobians定义保留函数，G14 L02/L50/L54/L82/L86–L89已指出非线性、分布式表示、门控及休眠备份路径的复杂性：输出可影响/可patch行为≠该方向是原模型正常路径的必要承担者；任意hidden基旋转也不穿过GELU与原生64-group Q2。
2. **M-B**以平均敏感矩阵乘平均协方差作为代理，忽略source、gate、状态、下游敏感度与当前实际输出变化的条件相关；其最关键的一阶目标应能直接区分这种依赖。
3. **M-A**普通6016学生是可训练新函数，但固定80个均匀宽度及局部teacher-state恢复并不会自动满足DIFS/L34、NerVE/L50和L54关于step异质性、非线性容量及跨模块职责迁移的机制；需比较恢复后的完整四步行为及学习预算。
4. **M-C**输出U低秩、输入Rx压缩与码表近似是三个额外的信息瓶颈；G14 CodeGEMM/L73、LUT-GEMM/L76、L71硬件执行先例显示单token GEMV快路不能直接推出Wan大M GEMM加速。

**尚未充分利用的有益先验**：L39将变化能量与时间可预测性分开、L60揭示视频时间推理与多步去噪可能职责分离、L47提示few-step temporal collapse的训练机制、L54提示训练时其它模块可能承接FFN职责、L82–L89提供功能定位的反例/作用边界。这些可以为当前M-B的设计创造新的假说和比较器，而非再做旧mask误差统计。

## 2. G14的覆盖结构：论文数不等于关键机制已闭合

G14当前有**89主论文＋10补充、16个T01–T16真实计算焦点**，其paper_links已经标注focus_nodes、机制、intervention、decision_time、review_depth和Wan迁移分歧。注意这些是**LIT/SRC关联边**，不是本项目观察到的Wan因果/执行结果。原生图202节点/413边也不能直接替代数值方法证明。

- **T08：e5门控与残差写回只有3项主论文链接**（L74 SharQ；L82 subspace intervention illusion；L86 activation patching），但**M-B的机制轴恰好落在gate×W2及下游接收**。这反映需要对该边做定向原文/源码重建，不能以G14总99篇判断gate-sensitive方法已充分验证。
- **T03：Q1与输入仅2项主论文**，T05：GELU hidden为7项，T07：W2 signed readout为8项；比较T10空间时间attention的24＋3项、T11历史缓存28项，说明当前图谱的“文献密度”偏向attention/cache，不均等于M-B所需的原生分布和敏感度证据。
- 书目读取深度混合了OFFICIAL_ABSTRACT、SPEC、原作者正文/方法、部分代码深审。**不能将具有G14链接理解为论文全部公式/代码/强先例已经过严格复核。** 后续M-B应对真正约束其定义的论文升级原文方程、实现及适用条件；特别注意L54是小规模训练Transformer，不是Wan原生因果证据。

## 3. 对M-B影响最大的六项图谱先验：发现→修改

### P0-B1 · 非线性改变功能基，不能只研究W2后静态谱

- **原图边**：T03 Q1→T04 W1→T05 GELU→T06 Q2→T07 W2→T08 e5。**L50 NerVE**指出非线性可重组协方差谱/有效维数；**L02 superposition**提示高阶功能可以分布在非神经元基；**L06 SliceGPT**依赖特定旋转不变量，而任意hidden旋转不能自由穿过GELU和有共享scale的HiF4 Q2。
- **M-B目前**：从pre-gate $y=W_2Q_2(\mathrm{GELU}(W_1x))$上学习固定共享输出$\Phi$，再从pre-W1输入$x$预测坐标。这是合法的**替代函数类假说**，但即便存在良好$y$低秩输出空间，也不保证学生的$(W_1,\mathrm{GELU},Q_2,W_2)$在同一预算下**可以生成**那些坐标。
- **新数学对象**：分开 (i) teacher output的可保留性，(ii) 从合法$x$与真实Q2可得函数的**可预测性**，(iii) 折叠/因子化后的有限精度**可部署性**。至少设一个相同r、不同hidden h的函数预测基线，观察非线性/Q2前后谱变化（若保存资产存在）；不可对量化隐藏空间任意进行线性旋转后宣称精确函数等价。

### P0-B2 · source-joint门控敏感不同于平均G×平均C

- 定义单一可配对source/step/state的真实写回$D_{g_s}y_s$和从该写回到**预定义**读出$o_s$的一阶后缀$J_s$。在冻结gate且$P^2=P$时，设计局部代理
  \[
  E_j(P)=\mathbb E_s\|J_sD_{g_s}(I-P)(y_s-\mu)\|^2,\qquad
  E_m(P)=\operatorname{tr}[\bar G(I-P)\bar C(I-P)^\top],
  \]
  其中$G_s=D_{g_s}^{\top}J_s^\top J_sD_{g_s}$、$\bar G=\mathbb E_sG_s$、$\bar C=\mathbb E_s[(y_s-\mu)(y_s-\mu)^\top]$。**数学含义**：$E_j$保留同样本矩阵$G_s$和偏差$(y_s-\mu)(y_s-\mu)^\top$的联合分布；$E_m$把两者的期望拆开，仅在对每个所考察投影相关交叉协方差为0等条件下相同。
- **可手工核验的二维反例，ALG/PROVED**：令4个等概率情形$(y,G)$分别为$(e_1,\mathrm{diag}(10,100))$、$(-e_1,\mathrm{diag}(10,100))$、$(e_2,\mathrm{diag}(1,0))$、$(-e_2,\mathrm{diag}(1,0))$，所以$\mu=0$、$\bar C=\mathrm{diag}(0.5,0.5)$、$\bar G=\mathrm{diag}(5.5,50)$。取秩1正交投影$P_1=e_1e_1^\top$、$P_2=e_2e_2^\top$：
  \[
  E_m(P_1)=25,\quad E_m(P_2)=2.75;
  \qquad E_j(P_1)=0,\quad E_j(P_2)=5.
  \]
  因此边际方法偏向保留$e_2$，联合方法偏向保留$e_1$，给出**相反排序**。此反例不证明原Wan方向满足此相关性，但说明数学上确实必须先查联合分布，再讨论改进。
- **作用边界**：$J_s$为多token/下游flow的线性响应算子，不是视频RAFT值对每个FFN token的梯度。若扰动来自pre-W1或改变上游state，gate也变化，微分应包括$D_y\,\delta g$等项；后量化、后gate局部插入的线性化与学生完整rollout要分开评价。

### P0-B3 · 功能可影响性不等于原本自然承担作用

- **L82、L86、L87、L88、L89**分别提供patch illusion、基线/方向/AND-OR备份、intervention granularity、交互组/旁路、梯度筛选假阴性。它们支持：一条方向的patch可能让视频发生变化，却不一定由该方向在正常forward中承担同一行为。
- **M-B改法**：分开 **FUNCTIONAL_OBS（轨迹上的相关）**、**FUNCTIONAL_CAUSAL（合法、匹配原生状态的有限干预作用）**、**FUNCTIONAL_IMPLEMENTABLE（从已可得$x$学习的经济替代）**。不能用一个JVP/一次PCA解释“运动神经元”，也不要因功能定位证据暂不充分而否定新的计算方法；可分别取得科学规律和系统成果。
- **最小区分**：在自然Dense状态上取同一读出与预先冻结来源，考察删除/注入及自然扰动的方向、baseline、cross-term与source稳定性；新学生自由轨迹作为独立结果，不能拿不同model state任意拼接。

### P1-B4 · 门控子空间旋转是已存在的代数作用，不是凭空增加的信息

- 若$\widehat y=\Phi c(x)$，当前e5 gate输出$D_{g_s}\Phi c(x)$，因此有效写回基随token/source/step变化，**这一等式在Dense已有gate中就成立**。它本身既不构成新路由、更不证明低rank产生新的运动功能。
- 真正可证伪的设计是：是否存在**低存储、少额外访存**的共享$\Phi$，利用已有gate的实际分布，在同源下游readout中保留优于普通/PCA子空间的行为方向；以及能否从pre-W1信息训练出系数。该结论需要对gate-conditioned和gate-blind基线实施配对比较。

### P1-B5 · 时间变化能量、时间可预测性与下游作用是三轴而非一个分数

- **L39 SVD-Cache**区分平滑主方向与低能振荡残差，对后者直接reuse而非简单丢弃；**L21 HyCa**指出特征演化可能异质；**L60 SCD**在因果视频扩散架构中把时间推理与多步帧内渲染分离。L60证明的是其研究的causal架构，不是当前非AR Wan 4-step中的同一FFN已有“时间专家”。
- **M-B的可用新方向**：对每个候选输出子空间同时记录$(\mathrm{variance},\mathrm{suffix\_response},\mathrm{time\_predictability})$，并考虑时间读出（跨视频帧/对象）与外观读出（单帧/纹理）分别构造$J_s$。如果预测困难但行为作用高，应**重点保存或重算**而非简单削减。是否采用跨step O-C需另证明相同expert历史合法、信息可在当前计算前获得和IO/VRAM成本。
- **最小区分**：先看已有时间槽/step的source配对统计能否评估方向变化与同源readout，不要求全rank巨大Jacobian；预定义相机/对象/背景读出再检查方向稳定，避免用一个图像流幅值代表视频动作。

### P1-B6 · 输出矩阵的低rank表示要和真实执行相连

- 对A0宽度$h=6016$、$D=5120$，离线折叠$W_2=\Phi B_c$仍有$Dh=30,801,920$ MAC/token；保留factored $h\to r\to D$有$r(h+D)$ W2 MAC/token。只有$r<Dh/(D+h)\approx2766$理论W2 MAC才下降；$r=512$时连W1共36,503,552 MAC/token，约为A0原主投影成本59.26%。
- **L71、L73、L74/L76与C5 A100真实观测**都说明执行形状、在线packing、双GEMM、读表与大M Tensor Core利用率不能外推；折叠与因子化有不同的BF16舍入与中间输出。若要主张M-B是真实低成本计算，必须记录$M=31668$、Fused vs staged、量化scale、VRAM与完整FFN/video，否则目前只是可用的函数表示假说。

## 4. 对M-A（可恢复窄FFN）的先验修正

**A-1：容量是经过非线性形成的函数，不是单纯的hidden数。** L50 NerVE和L02 superposition提示pre/post-GELU谱以及W2 joint response可能决定各block的可恢复难度。建议对A0 6016宽普通学生另设**同预算不同hidden分配**与初始化/训练范围控制；非均匀宽度本身已有结构分配先例，创新需具体到原Wan问题。

**A-2：少步模型应该看整个rollout而非仅同一teacher状态**。L34 Dynamic-in-Few-Step已有step-specific MoM+progressive/output rollout，并需要特定推理引擎；L47 Adaptive Video Distillation提出few-step运动/时序塌缩的独立解释；L56 Self Forcing通过Student历史曝光处理分布迁移（原生AR视频，与此Wan4step不同）。新A0必须在同训练预算明确teacher同状态、student free rollout、源/seed独立性，且不能把DIFS报告的整端速度直接当为Wan本项目实测。

**A-3：改变FFN函数可能改变相邻Attention的学习职责**。L54小模型实验提示训练中的架构变化可诱发计算转移；若A0只允许FFN LoRA、固定全Attention，其假定更接近“在冻结消费者中迁移FFN函数”，与“训练全网络使Attention分担FFN作用”是两个不同estimand。提出竞争假说，不将小模型结果冒充Wan已存在的替代路径。

## 5. 对M-C（编码直接求值）的先验修正

**C-1：信息瓶颈先于码本规模。** 当学生只见$z=Rx$时，$f(x)$能否近似因子分解为$d(Rx)$是前置可辨识条件；同码内teacher目标分布的不可约均方风险$E\operatorname{tr}\operatorname{Cov}(Y\mid Z)$与输出$\operatorname{col}(U)$秩$r_o$的谱尾误差是不同限制。可利用G12共同输入固定方向精确投影、同rank连续小MLP作为基础强基线。

**C-2：量化非线性是整体对象。** 如果C声称替代完整W1→GELU→64组HiF4 Q2→W2，则teacher输出的真实scale/rounding是目标函数的一部分；它不是把每个输入组的GELU相加的等价重写。L50的非线性谱变化与G12的组间依赖正好提示编码交互项的必要条件，但不证明二组表足以逼近。

**C-3：硬件先例的反面最强。** L73 CodeGEMM公开作者实现重点是单token LUT路径、多token可退回dequant+matmul；L76 LUT-GEMM、L71低稀疏率GEMV、L74 SharQ双FP4 GEMM具有不同适用形状。当前Wan大$M$需完整比率 $(\mathrm{projection}+\mathrm{encoding}+\mathrm{table\_gather}+\mathrm{output\_expand})/\mathrm{dense\ GEMM}$，而不是只有MAC压缩比。对$B=16,K=256,r_o=256$，BF16每二组表32MiB/expert-block；读表缓存和80套FFN驻留不容忽略。

**C-4：真正的创新自由度**是学习同时适配下游作用的**函数充分编码**，而非重做一个线性乘法查表器。是否成功应与同源同rank/同cost连续函数、M-A学生及最强LUT线性内核比赛；无需以M-B成立为前置。

## 6. 五操作框架与计算图先验如何连接

| 图谱中的先验事实 | 科学问题I/F/C | 可能导致的新操作组合 | 关键竞争解释/最小判别 |
| --- | --- | --- | --- |
| L02/L50：非线性/分布式函数 | I+F | **O-B＋O-E**，可选O-D | 函数rank是否可由更窄真实GELU/Q2学生重建 |
| L82/L86–89：干预不必忠实于正常计算 | F | **O-E**研究/因果，而非强制O-A | 同一自然路径与不同baseline的finite反事实是否一致 |
| L39/L21：变化能量与时间可预测性分离 | I+F+C | **O-C＋O-B**，可再O-A | 高频但低能的方向是否必须原样计算/缓存 |
| L60/SCD：时间推理与渲染分工 | F+C | **O-B＋O-E**，可能O-C | 当前非AR Wan的分工是否真的存在，不能直接照搬 |
| L34/L47/L56：少步分布与训练状态 | F+C | **O-B＋O-E**，可配O-A | 等训练预算下teacher-state vs student-rollout支持何种保真 |
| L71/L73/L74/C5：执行边成本 | C | **O-D**与O-A/O-B组合 | 同M/量化/内核下online净时延是否真的下降 |

**不应让五操作的代码本身限制原创性**：许多成功算法横跨O-A/O-B/O-C/O-D/O-E，一个操作即使有先例，使用条件/结构的新数学也可能带来贡献；同样，同一分类标签不证明两个算法重复。

## 7. 针对下一位从M-B开始的Agent：最少优先级改动

**第一步：先校正科学对象与论文锚点。** 在G14重点读取T05（L02/L50）、T06原生HiF4 Q2、T07、T08（L82/L86）及T09/T12/T15（L39/L47/L60/L87/L88），而不是只读CoBRAS/2ndMatch/OBS-Diff。记录原文实际参数、数学不变量、可用状态及适用模型；阅读深度不足的升至公式/作者源码，不因论文数量替代证据。

**第二步：先用本文件二维ALG反例检验B-MARGINAL与B-JOINT的区别。** 再定义normal-path conditional读出、gate固定/变化条件、multi-token下游动作和old CAL24资产可辨性。把B的三义划清：需要保留的函数、可以由student预测的函数、可被硬件省算的函数。

**第三步：与强替代假说竞争。** 低能高敏、Q2数值/Student状态漂移、光流可观测性、attention代偿、跨step变化可预测性要有不同来源级预测。受限数据可先给数学/CPU分布判别设计，不能从20源RAFT指标直接宣布B因果机制。

**第四步：独立产出版本化METHOD_SNAPSHOT。** 每个新候选写PROBLEM_STRUCTURAL_SIGNATURE、DERIVATION_GRAPH、FLOW_LEDGER、scientific object dictionary、数值等价与open bindings、first discriminating evidence、实际online成本和REUSABLE_ASSET_PLAN。方法提案及同行投票遵守QUEUE，实际运行遵守ACTIVE。原有M-B为下一Agent首项任务的用户指令不变。

**成果定位**：本轮完成的是G14先验对新方法的**可证伪审计和数学条件强化**（含E_m/E_j的ALG反例），不是原Wan实测发现或真实独立Agent对新方法的认可。
