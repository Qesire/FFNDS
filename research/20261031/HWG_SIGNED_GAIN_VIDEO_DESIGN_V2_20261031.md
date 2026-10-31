# HWG Signed Group + Gain：数学与真实视频实验设计 v2

2026-10-31 · **AUTHOR_REVISED_DESIGN / INDEPENDENT_REVIEW_REQUESTED / NOT_EXECUTED**。本文件不代表独立Agent已批准、ACTIVE已注册或执行GPU。历史证据锚点：FFNDS commit 3d7a1a0b5673305d2a86502fba8a1e245fcf9078。

## 科学发现与研究问题

[VIDEO] 真实Chipmunk 2来源×High/Low四臂实验发现，Low的局部P64相对L2较低，却更显著损害成像/I2V；2来源的High/Low视频策略交互方向不一致：[真实因子质量](../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)。
[VIDEO] 历史Y-only的两源RAFT下降且鸟翼出现拖影，说明时间更新有必要：[Y复用质量](../20261030/CHIPMUNK_HIGH_Y_REUSE_QUALITY_RESULTS_20261030.md)。
[VIDEO] UB1固定支持造成20来源连续RAFT约6.894→1.152，主体一致性反而改善：[UB1完整质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)。
[OBS] A5-inspired HiF4三源CAL1 signed静态mask在代理L2优于增量HWG全局选择，但原E0没有新自由视频：[HiF4 E0](../20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md)。

**目标**：在相同真实可执行的FFN组计算预算下，比较离线对角选组、带符号联合选组和保留组有限读出重构，检验是否改善Wan全160位置自由视频质量、动态与真实系统成本。局部Gram/SSE/最小二乘/MAC仅是诊断或能力界，唯一方法验收来自完整真实视频各指标与端到端实测。

## 1. 计算图及数值身份

冻结BF16模型采用Wan2.2-I2V-A14B Lightx2v 4step：步[1000,750,500,250]，High/High/Low/Low，D=5120，H=13824=216×64，40个Block，4×40=160 FFN调用，2×40=80次合法同expert历史增量，M=31668，81帧、16fps、目标480×832。逐字段参照[原生视频协议](../../config/e0_chipmunk_p64_factorial_video_20261030_part0_frozen.json)，包含checkpoint、prompt、裁剪、噪声、offload等。

**BF16原生路线**：没有HiF4激活Q1/Q2。令u为当前原生FFN的真实pre-W1输入（已包含对应norm/AdaLN），h=原生W1→tanh-GELU→pre-W2激活，Y=hW2+b2为pre-gate值；如源码额外缩放则必须原样保存。对同一expert、block，首次Dense算Y0并存旧h或u，第二次执行：

\[
\widehat Y_1=Y_0+(h_{1,S}-h_{0,S})B_S,\quad
\widehat{\mathrm{FFN}}_1=g_1\odot\widehat Y_1.
\]

Y0必须在gate之前；b2只出现一次；当前e5、残差和完整后缀走原模型顺序。不同expert/block不复用旧状态。选组在线不允许使用全部当前h或Dense Y。

**HiF4路线是后续独立迁移**：激活含Q1/Smooth2/Q2，差分为两个分别量化后的post-Q2激活之差；不得写成Q2(差分)，也不得把HiF4 E0直接重命名为BF16方法评测。迁移须重新校准，验证HiF4原K64量化闭包、数值、packed W1/W2和相关成本。

## 2. 三种可辨识的函数类

第b个完整64通道组对增量的当前gate后贡献v_b=g1⊙[(h1,Gb-h0,Gb)W2,Gb]；定义d=Σ_b v_b。所有来源先cluster等权，再在内部按真实空间/latent时间抽样方案聚合；每个expert×block单独学习和冻结支持。

**D = DIAG80**：按校准后的独立组能量 \(G_{bb}=\mathbb E\|v_b\|^2\) 取80组，使用原W2。D是BF16下新离线组评分对照，**不是**历史A5原参数producer或在线A5近似的忠实复现。

**G = SIGNED80**：令 \(G_{ab}=\mathbb E\langle v_a,v_b\rangle\)，\(p_a=\mathbb E\langle v_a,d\rangle\)，在同80组预算下近似最小化 \(\|d\|^2-2p^\top m+m^\top Gm\)，\(m\in\{0,1\}^{216}\)。固定tie、source归一化、forward greedy＋same-k 1-swap；属于受限组合求解器，不宣称全局最优。G/D均使用原始W2。**相反预测**：signed跨来源抵消若不稳定，G的局部收益可能消失于学生视频。

**R = GAIN80**：与G逐site **完全同一mask**，仅令保留组读出 \(B_{G_b}=\alpha_bW_{2,G_b}\)。拟合

\[
\min_{\alpha\in[0,2]^{80}}\;\mathbb E_s
\frac{\|\sum_{b\in S}\alpha_bv_{s,b}-d_s\|^2}{\|d_s\|^2+\epsilon}
+\lambda\|\alpha-\mathbf1\|^2.
\]

无盒约束时 \(\alpha=(G_{SS}+\lambda I)^{-1}(p_S+\lambda\mathbf1)\)；加入盒约束须用**凸二次规划**求解，直接clip无约束解不称受限最优。λ仅用cal-validation设定。增益可离线折入W2，但权重副本、FP4重新量化、packing、BF16数值/原生kernel仍需真实验证。**相反预测**：拟合低L2可能仅放大振幅，导致视频运动过冲、纹理闪烁、源外泛化失败。

G与CAL1 signed static、R与UB1 ridge consumer/TARDIS等存在明确亲缘；不可把Gram greedy或线性回归本身作为首次创新。无约束满维 \(B^*=X^\dagger T\) 只作为离线能力界，不进入第一轮原生视频动作。

## 3. 缓存和MAC的严格区分

四步全部Dense的W1/W2主投影分母=8H：
- **两个专家第二次均稀疏**，H+Y：\(R=1/2+(K_H+K_L)/(4H)\)；X+Y（重算选中旧W1）：\(R=1/2+3(K_H+K_L)/(8H)\)。
- H=13824，K_H=K_L=80×64时分别 **68.5185%和77.7778%**；64/96与96/64保留总160组，主投影预算相等，kernel实际耗时未必相等。
- **仅一个专家第二次稀疏，另一专家直接Dense**：H+Y比 \(0.75+K/(4H)\)，X+Y比 \(0.75+3K/(8H)\)。K=5120时 **84.2593%和88.8889%**。不得把不干预的Dense专家也支付三次W1。
- FULL_DELTA216 H+Y的主投影比例1，只作同数值族数值控制。此缓存族首次两个Dense意味着理论FFN主投影最多省50%，不是全模型加速上界。

[四来源H/X完整视频字节相同](../20261030/CHIPMUNK_INPUT_VIDEO_H_X_BYTE_EQUIVALENCE_20261030.md)已验证有限执行路径等价，但[原生完整成本](../20261030/CHIPMUNK_INPUT_VIDEO_NATIVE_RESULTS_20261030.md)中带observer与CPU offload下的H/X可能比Dense慢；局部packed节约并不等于系统收益。H+Y用于首轮机制，X+Y是显存/计算系统对照，不自动重复生成两套独立科学样本。

## 4. P−1：新的真实实验开始前，先处理来源与校准

1. **真实图像簇分割**：记录image_id、数据集revision、全部crop/prompt变体、输入图像SHA、near-duplicate、历史使用情况。校准训练、cal-validation、P0/P1开发、P2确认互不交叉。旧1040、898/979等已曝光，不能称全新heldout；P2新图需在所有历史实验中未暴露。
2. **BF16校准资产**：逐expert×40 block验证同源同seed两步原始pre-W2 h0/h1、Y0、gate、真实token坐标、学生/教师身份；旧HiF4采样资产不代替BF16。初步12训练独立图＋4独立cal-validation图是**待审配额**，不是现成已确认cohort；抽样应覆盖21 latent时间槽、自然raster、动作/背景等，校准样本行不是独立图像n。数据缺失则另审新的原生采集。
3. **算法冻结**：支持、α、λ与tie策略只用训练＋cal-validation，不借P0/P1/P2评分后验调优再冒充同一确认；候选可以在P1探索并版本化，P2前全部冻结。带gate的Gram须验证跨来源负交叉、零分母、PSD数值稳定和selection的来源外复算。
4. **物理preflight**：真实部分W1与W2输入维度、组packing、偏置gate、缓存信息截止；任何先执行full-M当前W1的评分只能作为付费oracle，不得报真正跳算。
5. **正式门槛**：现只完成作者自审。每个实际CPU/GPU/数据采集/视频/系统实验各自提交原源码与方法相反预测，经真正不同Agent独立review、作者修订与reviewer recheck后，才单独冻结ACTIVE、资源、job、输入SHA/seed、RUN/EXIT。历史scope的2卡约束按该实验原协议遵循，不预写新scope的固定并发许可。

## 5. 视频生成：阶段、对照、判决

**P0 原生门 / 两个已曝光开发来源×seed42**：898鸟叼鱼动作、96相机平移，完整图像/prompt/conditioning按[历史源清单](../../config/e0_chipmunk_input_video_sources_20261030_v1.json)严格读取；96带新增 camera 指令，绝不可与原未经修改prompt视频当同条件Dense。每源生成 DENSE、同数值族P64、DIAG80、SIGNED80、GAIN80，**最低10条视频**；另两个控制CACHE_DENSE_NEUTRAL、FULL_DELTA216按必要做数值中立重放，确需完整视频时最多另4条，**P0上限14**。必须全4step×40FFN；中立路径要求同prompt/seed/noise的Dense states/video byte-equal；FULL_DELTA216比较当前同输入数值残差并预定有限容差，BF16重排下不预设byte-equal。记录原生W1/W2跳算实证、质量、成本和失败资产。P0不通过则先重新独立评议方案，而非扩大生成数。

**P1 因果机制 / 另8个与校准不重叠的曝光开发图×seed42**：9臂 DENSE、P64、DIAG80、SIGNED80、GAIN80、GAIN_64_96、GAIN_96_64、GAIN_HIGH_ONLY、GAIN_LOW_ONLY，**最多72条视频**，可按P0科学结果删减。DIAG→SIGNED改变支持；SIGNED→GAIN保持同mask改变读出。80/80、64/96、96/64固定两个专家保留总160组但重新拟合各自支持/增益，判别预算方向。Dense/High-only/Low-only/GAIN80为完整2×2，其中GAIN80就是**BOTH**；逐源逐指标给 \(I_Q=Q_{\mathrm{Both}}-Q_H-Q_L+Q_D\)，表示整段学生自由轨迹的策略交互，**不是**组间内积交互的因果证明。P1里的方法和参数探索不得泄漏P2。

**P2 确认 / 真正未曝光图像簇**：先对预期视频效应、Dense跨seed/source方差、可接受质量非劣界及资源进行样本量复审；最初提议24个独立图簇、每簇两个真正独立seed、主体动作/相机/小物体交互/低运动背景各约6图，这不是统计功效保证和官方ImageSuite。五臂DENSE/P64/DIAG80/SIGNED80/GAIN80，共上界**240**；P1如有明确专家预算证据，在P2开始前可预冻结一个不均匀预算臂，上界**288**。P0+P1+P2三阶段理论最大14+72+288=**374**条，只是有条件上限，不表示GPU申请或已运行。固定验证图后禁止利用其结果挑λ、mask、组数、指标或方法。

**多轴评价**：对合法的source子集调用原版VBench八轴，I2V subject/background按真实类型分开；camera_motion与temporal_flickering有实际合法执行则报告，否则NOT_MEASURED。全部源做连续RAFT stride2 40帧对、五固定窗口，以及独立的主体/相机/背景运动、事件/文字条件、细小物体、边缘/光照闪烁、重影和完整81帧审阅。RAFT过大可能来自碎片，主体一致性高可能来自静止，不能用单指标判断成功。单seed与两seed自定义官方函数调用不能冒称官方VBench五独立生成leaderboard。

**统计与成本**：以image cluster为独立n，先源内seed平均后来源等权；逐轴配对绝对/相对变化、来源级bootstrap95%区间、类别分层和最差来源尾部；非劣界在P2前通过开发组/独立视频评审冻结，不从P2调整。视频真实成本含带observer与无observer分开测的冷/热wall、ABBA/随机顺序、FFN CUDA、GEMM shape次数、物理tile、cache/packing/IO/prefetch、峰值GPU/host、全系统wall和显存。质量多轴与真实系统成本构成Pareto；局部SSE/MAC仅报告代理。

**预定义可能推翻机制的结果**：G局部优于D而视频不优，说明当前signed选组没有验证视频收益；R局部更好但运动/闪烁更差，说明受限组读出泛化或动态行为可能不合适；64/96并不优于96/64，则专家预算方向不能沿历史两源推断；所有视频质量可接受但FFN/system wall不降，则仍属表示探索而非成功加速；P2未重现P1结果，应记录暴露开发条件的失效并提出下一验证，不用P2再训练后宣称独立确认。

## 6. 与原研究的关系与交付范围

本方案继续原HWG的组重要性直觉，并以CAL1 signed group、UB1 ridge consumer、Chipmunk缓存增量为明确竞争先例。它不要求新增Attention/step跳算或LLM复现。方法创新空间在**同专家时间增量＋可执行静态K64组＋相同mask受限读出＋专家预算在全视频传播中的可证伪作用**。这次只写入设计、自审与机器计划，不改变历史RUN、模型、已冻结协议或ACTIVE。下一实际行动是P−1资产/来源和BF16 matched状态覆盖预检，再申请真正独立实验复审。
