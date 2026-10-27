# FFNDS · 下一位方法研究 Agent 的起点：B 下游敏感函数坐标

**用户当前指令：下一项新方法研究先从 B 开始。**  
**状态：NEXT_AGENT_TASK / DISCUSSION_ONLY / NOT_EXECUTED。** 这是一项有限的**下轮优先任务**，不把B变成所有未来Agent的永久唯一方向，也不改变I/F/C科学框架、ACTIVE运行授权、已完成实验或QUEUE的独立评议身份。

**统一五操作术语（更新）**：先读[五操作×I/F/C科学分类](../20261027/FIVE_OPERATION_METHOD_LENS_AND_IFC_ROUTE_CROSSWALK.md)和[机器方法映射](METHOD_OPERATION_CROSSWALK.json)。这里用户指定的“B”是**M-B候选（下游敏感函数坐标）**，不是操作分类中的O-B整个函数表示家族。M-B当前为**O-B改变函数表示＋O-E学习/验证目标**，B-Factored/Fused才有可额外讨论的O-D执行改写；不要求加O-A稀疏选择或O-C跨步缓存。Agent如认为O-A/O-C有关键增益，可作为竞争性新候选，但先完成当前M-B数学方法交付。

## 0. 明确分工和交付

给下一位研究Agent的直接要求：

> 首先阅读[原始A/B/C提案](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)、[独立科学评议](../20261027/COMPACT_FFN_ABC_INDEPENDENT_METHOD_REVIEW_20261027.md)与[当前证据组合](../20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md)。**只以B为这轮首个新方法主体**，从真实Wan FFN的gate-conditioned下游响应、低能高敏方向和低成本函数表示构造出可证伪的竞争机制；先数学、源码、论文、保存资产，后选择真正有区分力的实验。A0普通6016维学生是匹配基线，C编码函数保留独立候选；不要以A训练或C查表试验替代本轮B的首份交付。

首份主要交付为独立的 **B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md**（放在research/20261027，工作日期/版本按实际更新）；同时输出最小数据/算子判别计划及明确的OPEN绑定。已有论证为ALG/SRC/LIT，未进行的采集/拟合/视频为OPEN。

## 1. MAP FLOW：完整而必要的科学计算对象

重建LightX2V Wan2.2 I2V A14B四步D-Q、high/high/low/low两expert×40 blocks的真实边界：
norm2/AdaLN → post-Q1输入 x (M×5120) → W1 (5120→13824) → GELU(tanh) → Smooth2/HiF4 Q2(64→8→4) → W2 → pre-gate y (M×5120) → caller e5门控/残差 → attention和其余block/flow → 四去噪step → 21 latent时间槽/81帧输出。记录逻辑tensor形状、dtype、QDQ相邻舍入/尺度、gate广播、操作顺序、expert和step身份及可复用数据源。

冻结真实统计单位：独立image/source cluster，其下才是seed、prompt、row、token、block、step。分开teacher输入状态和学生自由轨迹；完整consumer20源有真实视频但已开发暴露，不是新盲确认。旧CAL24原生V1=post-Q1、V2=post-Q2、Y=pre-gate，gate另存；查明具体producer和有效row坐标后复用。原G13 finite P/N不足以推断完整Jacobian。真实代码绑定未知时标OPEN_BINDING，不凭同名同shape猜测。

## 2. DERIVE：三种竞争机制及数学义务

在同source、同state、同step、同block下，选预先定义的下游读出 o_s，pre-gate输出 y_s，固定gate的对角算子 D_gs，从writeback到所选readout的局部Jacobian J_s。研究秩r投影 P=ΦΨᵀ，ΨᵀΦ=I。

**B-MARGINAL**：原平均代理 E_m(P)=tr[ Ḡ(I−P)C̄(I−P)ᵀ ]，Ḡ=E[D_gᵀJᵀJ D_g]。保留其作为成熟比较器；推导它何时近似同source联合结果、何时因样本相关性改变排序。

**B-JOINT**：提出同source联合的一阶代理 E_j(P)=E_s ||J_s D_gs(I−P)(y_s−μ)||²。明确它是冻结gate、小扰动和所定义读出下的一阶近似，不是完整视频损失恒等式；若输入变化导致gate也变化，必须另外计算diag(y)δg。多token cross-attention让J_s是涉及M×D写回的下游算子，不能把单token梯度均值直接当作完整后缀。

**B-GATE-SUBSPACE**：研究固定pre-gate基Φ经动态gate变为D_gsΦ的有效写回空间。独立提出它何时有可预测的source/step变化、何时只是已有gate重参数化的精确相同作用，及怎样用pre-W1可得输入预测新函数坐标（不可偷用完整teacher hidden/未来运动信息）。

**首个强推导与反例**：用明确的充分条件或低维反例证明 E_m与E_j何时一致、何时不同排序；区分PCA/能量、marginal、joint和随机同rank基。构造相反机制：source分布漂移、原生Q2数值、纹理/RAFT观测偏差、suffix传播，而不先把运动减少归因于低能高敏。提供能从已有保存资产产生不同预测的最小实验或清楚列出缺失producer。只有真实观测变化足以区分后，才讨论运动功能标签。

## 3. BUILD：比较两种真正不同的低成本实现

**B-FOLDED**：以 ΦB_c 折叠形成D×h的W2，h=6016，D=5120。推理仍付完整Dh=30,801,920 MAC/token的W2，作用在于参数表示/质量，不因rank小自动省算。

**B-FACTORED**：保留h→r→D两级W2，MAC/token=r(h+D)，当r<Dh/(h+D)≈2766才减少W2主MAC。r=512时，连同W1合计36,503,552 MAC/token，为A0完整两GEMM的59.26%；需额外考虑kernel发射、中间tensor、真实BF16/HiF4 QDQ、权重量化及内存驻留。输入必须在pre-W1时真实可得，不得先做完整FFN再投影。建立folded和factored的数值/部署独立身份及rank Pareto，不把ALG加速预算写成SYS效果。

A0普通6016学生、同rank PCA、随机基、平均G/C与同source联合G/C分别作为合理对照；训练预算、teacher/student状态、量化和源级抽样要匹配。未来如需fit，先冻结协议再实际捕获，不根据已看结果改变confirmatory准则。

## 4. 一手论文检索与G14原图

在[G14论文动作映射](PAPER_METHOD_FOCUS_GRAPH.json)、[计算图](../../docs/98_端到端计算图与证据注解_20261025.md)、[近邻审查](../20261027/PARTIAL_LINEAR_FFN_PRIOR_ART_AND_TRANSFER_AUDIT_20261027.md)中定位：CoBRAS平衡降阶/adjoint、2ndMatch扩散Jacobians、OBS-Diff时步二阶、SVDtrunc低秩分配与蒸馏；并主动拓展gate-conditioned observability、source/step-conditional directions、低秩算子融合相关原文及作者源码。记录每篇论文**真实改变哪个计算节点、何时获得信息、使用的训练/在线物理工作、可迁移的数学条件、与B的差异**。允许借鉴已成功方法，创新主张应落在原Wan门控/后缀/数值执行的可检验新构造。

## 5. 第一份产出验收（数学方法阶段）

所写METHOD_SNAPSHOT须包含：
1. 问题结构、读出定义、source×state条件、现实可得信息、shape和符号表；
2. MATHEMATICAL_MODEL + DERIVATION_GRAPH（关键条件、公式、反例、PROVED/CONDITIONAL/OPEN）；
3. 真实producer→Q1→W1→GELU→Q2→W2→gate→suffix的FLOW_LEDGER与数值等价边界；
4. B-MARGINAL、B-JOINT、B-GATE-SUBSPACE的竞争预测及最强外部先例；
5. B-FOLDED/B-FACTORED的MAC、rank、GEMM/launch、QDQ和VRAM物理预算；
6. 现有资产是否已足够的逐项核验，缺失项明确OPEN_BINDING；下步最小独立source配对arms及适当PASS/FAIL/INCONCLUSIVE判据；
7. REUSABLE_ASSET_PLAN：每件未来JVP/finite response、基矩阵、rank/量化和视频资产的producer、source/state/shape、存储及复用失效条件。

**优先复用证据**：[consumer20真实运动](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)、[CAL24四位置source稳定性](../20261028/UB1_CAL24_SOURCE_ENERGY_PROFILE_RESULTS_20261028.md)、[finite suffix](../20261026/E0_WHOLE_FLOW_COUNTERFACTUAL_RESULTS.md)、[A0 Whole94真实Q2准备](../method_discussion/discoveries/E0/compact6016-adaptation-readiness-20261028.md)、[合成fused 2:4执行成本](../20261027/E0_ACTIVATION24_FUSED_CUTLASS_RESULTS_20261027.md)。这些分别是历史VIDEO/NATIVE/SRC/SYS，不等于B机制已得到直接证实。泛化需独立来源；正式视频质量需相同noise、完整40×4、真实分维评价和物理成本。

**协作与执行身份**：首份方法由下一Agent独立完成，若形成成熟新提案可进一步按[method_discussion协议](../method_discussion/README.md)独立投递；QUEUE仅登记真实proposal/review、不得凭这份任务伪造peer support。数学、论文、源码和已保存资产优先推进；新CPU/GPU训练/生成须按ACTIVE已有精确许可或新的冻结范围另行登记。这份任务本身不授予额外GPU/实验权限。
