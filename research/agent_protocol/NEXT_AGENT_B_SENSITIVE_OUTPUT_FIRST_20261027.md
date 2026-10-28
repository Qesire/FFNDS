# FFNDS · 下一位研究Agent的现行任务：从已完成的M-B方法继续

**2026-10-28 · CURRENT / CAL24_B1_COMPLETE / B-H_PILOT_COMPLETE。** 用户指定M-B优先；[当日任务台账](CURRENT_TASK_LEDGER_20261028.json)和[ACTIVE](../ACTIVE_RESEARCH.json)分别导航依赖与真实运行权限。[CAL24实际结果](../20261028/B_CAL24_SOURCE_PAIRED_DISCRIMINATION_RESULTS_20261028.md)完成24来源×4步96flow、byte audit、固定字典paired判别和B0四case的16次full-M选中行neutral。固定16维span、rank4、18候选中paired/shuffle/marginal/PCA/head-only同选PCA；CPU M256不是native floor，一般paired机制仍OPEN。

**当前续作（2026-10-29 v4）**：优先将[多来源×全深度×视频优先合同](../20261029/FULL_DEPTH_MULTISOURCE_VIDEO_FIRST_METHOD_CORRECTION_20261029.md)落实为每个新方法的全160-site可执行表示和≥20独立image-cluster的配对完整生成。已有B-H、同head A0、B-F、原始BF16 fixed-gain用于机制和候选构造，低39单来源增益不作为方法晋级。先复算20-source UB1、Affine54 320-site、compact6016视频的分维异常与来源尾部，再从视频失败定位具体step/block/consumer，模型/源/seed和真实SYS作为新的方法判据。NVFP4 Pro6000独立并行且按ACTIVE精确scope执行。\n\n## 0. 已完成的工作：作为下一任务的输入，不重复创造

1. [现行统一研究方案 v3](../20261027/COMPACT_FFN_REPRESENTATION_THREE_METHOD_PROPOSAL_20261027.md)是M-A/M-B/M-C的**唯一现行研究路线图**，其中G14先验已转化为数学/数值/执行约束；原独立审计[在此](../20261028/G14_PRIOR_CONSTRAINT_AUDIT_FOR_NEW_FFN_METHODS.md)保留作为推导来源。
2. [M-B方法快照 v2](../20261027/B_GATE_SUFFIX_SENSITIVE_COORDINATES_METHOD_SNAPSHOT.md)已证明固定G的加权投影闭式解、paired vs marginal的反向排序反例、block39末层LN/head的连续Jacobian和有限充分统计；源代码/头部权重的SRC身份及U28/I19已建立。同状态BF16 head有限响应与固定候选配对比较已实测；[B-H H128 pilot](../20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)及同head A0对照已完成；B-F末层pilot已完成；一般配对机制及B-F视频/系统效能仍OPEN。
3. [A0全80函数局部恢复](../20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.md)：160学生训练、已暴露train16/heldout8和metadata bank已完成；独立原始字节审计和两来源六视频v2均实际exit0，quality v4已完成且未通过；人工全播与公平测速仍待验。
4. 既有[consumer20质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)证实当前UB1固定RR在20个开发来源的连续RAFT大幅下降；不能将其当作M-B函数成功/失败。
5. [G14计算图](PAPER_METHOD_FOCUS_GRAPH.json)已有98项主论文＋10项校准，配[五操作机器交叉表](METHOD_OPERATION_CROSSWALK.json)。新研究重点是T05/T06真实GELU与Q2、T07 W2、T08 e5门控、T09及T12视频/后缀、T13核、T15因果；不再以“先广泛扫摘要”代替当前已证明的数学结论。

## 1. 已完成B0的技术合同（复核用）

**B0-PREP（已完成实际V3捕获；保留原对象定义）**：依据v2快照第4、7、8、9节，完成两份已暴露source×step0/high和step3/low共4个Dense flow捕获的**源码实现/producer定位、资产SHA核验、标准协议与中性等价检查**。遵循source image-cluster身份；采集每一真实状态的full-M post-write $u$、原gate、residual、head调制与head/flow输出、绑定seed/噪声/调度器/两expert、真实Q2数值、shape/dtype/stride。任何缺失producer先按真实源码定位或标OPEN_BINDING，不把CAL24的post-Q1、post-Q2或pre-gate Y假设成full u。

**B0原生运行身份（已完成，历史合同）**：真实4次Dense flow的scope、源、exit与SHA已登记，复核时读取原RUN，不把下文老协议重新登记为待运行任务。新的模型/视频实验仍须按ACTIVE冻结真实source、预算、代码与观测对象。

**B0接收标准**：neutral/no-op调用必须保持原函数输出与正确shape/dtype、保存真实配置/状态/hash、来源/运行计数匹配；如果原生FFN/head无法重放则记录真实差异与对应数值边界，不盲改扰动幅度或上移研究指标。

## 2. B1数学和数值判别：直接利用先验，构建最小可区分对照

- 从已证明的末层身份 $F(u)=q/\sqrt{v+\epsilon}+const$ 出发，其中$q=T P_0u\in\mathbb R^{64}$，$v=\|P_0u\|^2/D\ge0$，**连续域65量**即可重建该固定modulation/head读出；与v1的$Tu$、mean、norm²的66量坐标互为表示。注意这不是在线低成本实现的证据，也不是BF16 bitwise identity。
- 在同来源、同state、同幅度、BF16有效$\delta$条件下做full-M有限响应，比较 **head-direction、centered head-null、mean-shift及纯方差/径向方向**。不把同shape/同dtype的旧Y当作状态$u$；实际后缀有限实验需要真实producer。
- 以G14 L82/L86–L89为先验，区分自然路径响应、激活休眠备份的patch幻象、state漂移和纹理/RAFT可观测性的替代解释。
- 同rank/同可观测信息，比较**paired joint、shuffled joint、marginal、PCA、random**；旧CAL24同位置gate近似固定，重点从随source/state变化的真实$J_s$寻找新联合信息。二维反例已证明数学可能性，**不是**Wan中已观测的源依赖。
- 完整读出需要multi-token/后续attention时，不以末层one-token 64D head充当40个block下游的同一个Jacobians；末层是第一个可核的科学边界。

**B1后续判断**：同source finite改变若超过原生neutral舍入且跨source保持有方向性，可进入新的配对基科学假说；数值噪声同量级按INCONCLUSIVE记录，先改观测和量化范围；若排序不改善，分析是否更适合条件基、残差/时间坐标或另一个科学对象，而非反复调rank/SSE阈值。

## 3. B2直接提出两类合法可部署函数（O-B＋O-D）

**B-H，末层head-sufficient替代**：尝试仅从当前pre-W1输入$x$、已存在的残差$r$、原gate/time推算$(\widehat q,\widehat v)$，再求$\widehat F=\widehat q/\sqrt{\widehat v+\epsilon}+const$。与“先完整算teacher FFN，再提取head统计”严格区分；需要估学生网络、非负variance参数化、真实量化、kernel和泛化成本。其作用于block39/head，不自动可移植到前39个block。

**B-F，一般块的gate/suffix-sensitive低rank函数**：学生$y_\phi=\Phi c_\phi(x)$，其中$c$可由新的合法W1/GELU/Q2缩窄函数或另一个有数值定义的算子求得。与A0同来源、参数量、训练预算、BF16和数据暴露条件比较；如果在同rank及同来源上优于PCA/普通窄学生，才归因给suffix-sensitive先验。FOLDED不会减少在线W2 MAC；FACTORED仅在$r<2766$有理论主MAC下降，r512连W1为36,503,552 MAC/token（A0的59.26%），新增kernel/GEMM/VRAM都需实际核验。低能高敏、时间可预测及正常功能身份分开检验。

**末层因子化实数代数**：若$y=\Phi c$，令$a=P_0r$、$B=P_0D_g\Phi$，则$q=Ta+(TB)c$，$Dv=\|a\|^2+2c^\top B^\top a+c^\top(B^\top B)c$。这个恒等式揭示了$B^\top a$的$O(Dr)$和Gram项$O(r^2)$、gate变化与BF16成本：不许因为“输出只需65量”就宣称在线已省去W2。bias归并到r需要同时计量。可据此探索direct 65统计student，或合并读出/分解的fused kernel，二者都标OPEN直到实际代价及数值核验。

## 4. 完整项目交接和实际文件产出

首个可验收交付**不再是重新创建M-B方法快照**（已经存在且有v2数学）；应提交：
- 复用[B-H真实训练结果](../20261028/BH_DIRECT65_H128_PREDICTABILITY_RESULTS_20261028.md)及原RUN/exit，复用已完成同来源同head A0比较及原M数值验收，避免重复；
- B1科学对象的预先冻结读出/方向/扰动量与相反预测，具体实现计划，若有实际原生RUN必须单独落原始receipt；
- B-H/B-F两个真正不同的学生计算图、参数/数据/在线成本以及与A0的matched对照；G14论文的真实作用边、先验限制及新自由度一一匹配；
- 按科学方法输出**M-B snapshot的下一版本变化摘要**，并以Graph Agent的节点/有类型关系记录新SRC/ALG与OPEN条件；不要凭任务计划伪造MODEL/NATIVE/VIDEO/SYS或Agent PEER_SUPPORTED。

M-C编码求值和O-C历史复用保持活跃的独立研究空间。后续若M-B需要新训练、全深度和完整视频，先冻结参数/来源/状态与比较组、真正独立的新video source和实际物理成本；方法提案须由真实提出者通过独立review/QUEUE推进。所有操作从这个现行v2和已完成P30/B-v1开始，不再将2026-10-27原始A/B/C初审作为新一轮执行任务。
