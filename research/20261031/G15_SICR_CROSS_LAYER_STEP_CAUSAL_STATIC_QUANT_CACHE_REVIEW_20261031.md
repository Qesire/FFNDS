# G15 · SICR：静态信息约束残差 FFN 的全层—跨步耦合数学与独立评议

2026-10-31 · **INDEPENDENT_SCIENTIFIC_REVIEW / ALG+SRC+LIT / NEEDS_REVISION / NOT_EXECUTED / NOT_GPU_APPROVED**  
审查锚点：FFNDS [576af1c05c16a2052e436cf3016117f5224c342b](https://github.com/Qesire/FFNDS/commit/576af1c05c16a2052e436cf3016117f5224c342b)；被评方案：[Signed Group + Gain v2](../20261031/HWG_SIGNED_GAIN_VIDEO_DESIGN_V2_20261031.md)。这是独立于原方案作者自审的研究推导，并非仓库协议要求的“真实不同 Agent 的审阅/修订/recheck”已完成；不修改 ACTIVE，不创建实验批准、RUN/EXIT 或 GPU 额度。建议决定：**REVISE**。

## 0. 首要科学判断与版本边界

**核心改写**：FFNDS 需要求解的是“固定可执行 FFN 结构、可用量化信息和最小历史状态共同约束下的整段视频函数逼近”，而不是独立地提高80个 site 的局部 SSE。局部 signed Gram 是有价值的结构代理，但影响最终视频的还包括跨层/跨去噪步传播、gate、self/cross-attention 的空间/时间 token 混合、低噪声阶段的非均匀敏感度、历史缓存反馈、量化舍入与饱和、学生状态迁移、视频解码与光流/语义观测。

**证据**：
- [三来源全40层双expert HiF4 E0](../20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md)：CAL1 signed static相对L2 0.443230，A5-inspired delta 0.456756，都是已有曝光来源/teacher-state代理，**没有对应新自由视频**。
- [P64真实两来源四角](../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)：Low-only对imaging约 -2.120/-1.483，High-only约 -0.008/+1.912（分别来源898/96）。这支持Low/后期更敏感的工作假说；expert身份与去噪时段完全共线，不提供独立的expert因果效应。
- [UB1 consumer20](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.md)：连续RAFT 6.894→1.152，动度二值 6/20→0/20，展示局部函数近似可能改变运动。高一致性不等于运动保真。
- [历史缓存](../20261030/CHIPMUNK_HISTORY_ENCODING_RESULTS_20261030.md)：4个局部case可优化旧h clone；[H/X全视频字节一致](../20261030/CHIPMUNK_INPUT_VIDEO_H_X_BYTE_EQUIVALENCE_20261030.md)限于所测4来源与路径；[真实系统](../20261030/CHIPMUNK_INPUT_VIDEO_NATIVE_RESULTS_20261030.md)曾出现缓存与额外计算使FFN wall劣于Dense。不可用MAC直接宣称速度。
- 本研究假设冻结LightX2V 4step（1000/750/500/250），High/High/Low/Low，每expert 40 blocks，共160 FFN调用、80同expert第二次更新，M≈31668，D=5120，H=13824=216×64；其他scheduler或量化后端均须重新建立恒等合同。

## 1. 明确时间、状态和作用边界

用 k=0,…,3 表示**去噪步**；用 l=0,…,39 表示**Transformer block深度**；用 τ=0,…,20 表示21个latent视频时间槽；用 r 表示空间token。视频最终81帧（16fps）与latent τ不是同一坐标。令e(k)∈{High,Low}，分别为 HHLL，第一次同expert调用k=0/2，下一次为k=1/3。

令 x_{k,l} 为含全部[τ,r]位置的块输入，u_{k,l} 为当前norm2/AdaLN之后真实pre-W1输入，g_{k,l} 为当前e5门控；数值族 Q 的真实 FFN（省略必须由源码核查的其他scale/cast细节）为

\[
 h_{k,l}^{Q}=\mathcal Q_{2,k,l}\{\phi(W_{1,e,l}\mathcal Q_{1,k,l}(u_{k,l})+b_{1,e,l})\},\qquad
 Y_{k,l}^{Q}=\sum_{b=1}^{216}h_{k,l,G_b}^{Q}W_{2,e,l,G_b}+b_{2,e,l},
\]
\[
 x_{k,l}^{+}=x_{k,l}+g_{k,l}\odot Y_{k,l}^{Q}.
\]

RAW_BF16路线取Q1、Q2为恒等；HiF4现行Q1/Q2为当次动态fake-QDQ、W静态QDQ后BF16 GEMM；原生MXFP4/NVFP4须分别移植，不得把QDQ与packed实测混为一类。注意gate通常受当前输入/条件调制，其导数不能在传播中冻结。

定义完整网络F_k(z_k,c_img,c_text)和真实deterministic scheduler \(z_{k+1}=\mathcal T_k(z_k,F_k(z_k))\)；如果具体scheduler允许，才简写为 \(z_{k+1}=a_k z_k+b_kF_k(z_k)\)。所有a/b、CFG、终止步、噪声/conditioning必须以真实冻结runner为准。给定相同初始噪声、图、prompt后定义可配对反事实。

## 2. 严格的跨层、跨去噪步及学生状态传播

### 2.1 不依赖线性假设的精确误差分解

Dense路径z_k，候选路径\(\tilde z_k\)；定义\(d_k=\tilde z_k-z_k\)。在同一输入和scheduler状态（若scheduler有记忆，纳入扩展状态）下，有精确恒等式

\[
 d_{k+1} =
 \underbrace{\mathcal T_k(\tilde z_k,F_k(\tilde z_k))-
 \mathcal T_k(z_k,F_k(z_k))}_{\text{输入轨迹变化的传播}}
 +
 \underbrace{\mathcal T_k(\tilde z_k,\tilde F_k(\tilde z_k,\tilde C_k))-
 \mathcal T_k(\tilde z_k,F_k(\tilde z_k))}_{\text{当前学生状态上的真实干预余项}}.
\]

随机算法要求配对随机数/内部状态并纳入状态。**不能**将第二项用仅从Dense teacher捕获的误差直接替换；这是同状态代理和自由学生视频之间的必需差异。

如果网络和scheduler可微（量化边界只能在固定码区间内分段处理），则沿Dense附近线性化：
\[
 d_{k+1}\approx A_kd_k+ B_kr_k,\quad
 A_k=D_z\mathcal T_k + D_F\mathcal T_k\,D_zF_k,\quad
 B_k=D_F\mathcal T_k ,
\]
其中\(r_k=\tilde F_k(\tilde z_k,\tilde C_k)-F_k(\tilde z_k)\)。余项包括Jacobians变化、非线性、高阶项和量化切换；真实验证必须报告线性预测误差。

### 2.2 同一步内部的gate、下游Attention与后缀

在第(k,l) FFN的pre-gate输出中注入\(\epsilon_{k,l}\)，令\(J^{\rm suffix}_{k,l}\)为从该FFN写回到最终网络输出的完整下游导数，定义
\[
 L_{k,l}=J^{\rm suffix}_{k,l}D(g_{k,l}).
\]
梯度/有限差必须包括未来block的self-attention全token混合、cross-attention、norm/AdaLN、gate及残差；若把gate看作状态函数，单块完整输入导数含\(D(Y)\,Dg\)项及\(D(g)\,DY\)项。此处\(D(g)\)只是**同状态、pre-gate局部注入**的直接线性映射，不能把全路径gate固定。

对最终latent的一阶单site影响：
\[
 P_{k,l} =
 (A_{3}\cdots A_{k+1})\, B_k\,L_{k,l}
\]
（空乘积为I；算子乘积按反向时间顺序理解；各J均在同一冻结Dense轨迹上取）。于是
\[
 d_4\approx \sum_{k,l}P_{k,l}\epsilon_{k,l}+R_{\rm nonlinear}.
\]
如实际scheduler F_k与latent联合作用依赖内部状态，则以完整状态的雅可比重建P，不能照抄简式。

### 2.3 真正跨步的**历史缓存路径**

仅\(d_k\)不构成闭合状态：第k=0/2次同expert的历史FFN缓存会被第k=1/3次消费。令扩展状态
\[
 \xi_k=(z_k,C_{High,0:39},C_{Low,0:39},q_k^{sched}),
 \quad
 \tilde\xi_{k+1}=\tilde{\mathcal A}_k(\tilde\xi_k).
\]
缓存仅在合法expert时点存储/更新/释放；未初始化缓存必须显式标无效。其线性化为
\[
 \begin{bmatrix}d_{k+1}\\ \delta C_{k+1}\end{bmatrix}
 \approx
 \begin{bmatrix}A_{zz}&A_{zC}\\A_{Cz}&A_{CC}\end{bmatrix}_k
 \begin{bmatrix}d_k\\\delta C_k\end{bmatrix}
 +\begin{bmatrix}r_k\\\eta_k\end{bmatrix}.
\]
其中\(A_{Cz}\)刻画上一步学生状态如何写入历史，\(A_{zC}\)刻画缓存再利用对当前输出的影响，\(\eta_k\)包括缓存量化/编码失真和构造误差。跨High→Low段不直接传旧High缓存，但High的latent扰动通过Low专家传播。对这条扩展链应用JVP/VJP而不是构造天文规模的稠密Jacobians。

### 2.4 二次耦合核与可判定的非加性

定义最终latent或其他固定可微读出R上的局部二次代理\(\mathcal L_{\rm proxy}\simeq \frac12\|R\,d_4\|_Q^2\)，\(H=R^\top QR\succeq0\)。其一阶扰动代入得
\[
 \mathcal L_{\rm proxy}\approx \frac12\sum_i\epsilon_i^\top K_{ii}\epsilon_i+
 \sum_{i<j}\epsilon_i^\top K_{ij}\epsilon_j,\quad
 K_{ij}=P_i^\top H P_j.
\]
**i,j跨全部160 site，并可进一步按(τ,r)展开**，故交叉项包含跨层、跨step、空间token和视频时间槽。带符号交叉项可能相消或增强；不能预设为独立项之和，也不能根据局部loss排序全局最优。若最终非线性任务损失本身展开，还存在二阶模型Hessian及一阶梯度项；上述K是Gauss–Newton型PSD读出代理，不等于VBench真实损失曲率。

不需要收集完整80×80×M×D核。**直接用四角干预测量非加性**：
\[
 J^{out}_{ij}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]
这里Y可以是同初始noise配对的末latent、预先冻结的动作/纹理/语义读出，或来源级视频指标；Y_{ab}代表开关两个真实干预后各自自由rollout的结果。其解释是**整个干预策略的因果交互**，不能据此证明内部两个组的Gram耦合。组内同状态\(v_a^\top v_b\)及全视频四角必须分别报告。优先抽样：同step近/远层、High↔Low跨step、同expert跨步复用、不同latent τ/空间邻域；必须有远距离/异号的反例。

### 2.5 精确误差上界与近似的地位

若在固定邻域内\(\|A_k\|\le a_k\)、\(\|B_kr_k\|\le b_k\)，则一阶或Lipschitz递推给
\[
 \|d_4\|\le \sum_{k=0}^3\left(\prod_{t=k+1}^3a_t\right)b_k
\]
（实际非线性界须对整个路径给Lipschitz常数）。此界保守且丢失符号抵消；用于认证最差扰动，而不是用于排序视频感知质量。强近似应比较带符号的\(K_{ij}\)和经验后缀反事实。

## 3. 视频时间耦合：运动、细节、相机与读出

在最终解码视频\(V=D_{\rm VAE}(z_4)\)上区分**去噪步k**与**物理视频帧f=0,…,80**。潜时空Jacobian \(P_{k,l}[(\tau',r'),(\tau,r)]\)的非对角项来自Attention和后续消费者，不能把latent时间槽当独立行；21个latent槽不是21个独立样本。

预先冻结至少四类功能读出（均是离线/诊断代理，VIDEO仍决定采用）：
1. **外观/结构**：帧间及相对Dense配对的多尺度图像特征差\(R_{\rm img}\)；局部纹理与小物体ROI独立统计。
2. **运动轨迹**：预定实体/背景/相机光流或跟踪读出\(R_{\rm mot}\)，比较\(\mathrm{Flow}(V_{sparse})-\mathrm{Flow}(V_{dense})\)及幅值比、方向/轨迹；不要最大化单独RAFT幅值（噪声也可增大）。
3. **时序误差**：以Dense流固定时间扭曲算子\(W_{f\to f+1}\)，构造\(r_{f}= (V'_{f+1}-W V'_f)-(V_{f+1}-WV_f)\)；另外计算对光照/边缘的时频差与突发分布。这是固定Dense流的比较读出，无法替代模型独立估计的动作真实性。
4. **文本—事件**：语义事件、主体与交互对象、动作开始/方向、I2V条件遵循，预注册类别及文字指令；训练时代理可用冻结特征，验收仍要完整视频/人工抽查。

形式上\(H_{\rm proxy}=\sum_a\lambda_aR_a^\top R_a\)，含多视角和多时滞算子。所有\(\lambda_a\)只在cal-validation/dev上选择；不保证读出涵盖真实视频质量。必须记录运动衰减/过冲、时序闪烁、语义漂移**各来源尾部**。高频低能量、低幅值高后缀敏感的方向可能特别重要，单独PCA/SSE会漏掉。

## 4. 静态支持的最小缓存：严格代数与信息下界

在每个expert×layer，冻结支持S（原HiF4整64组闭包的子集）及固定α_b。首次同expert full Dense：
\[
 Y_0=b_2+\sum_{b=1}^{216}h_{0,b}W_b,\quad
 C^\alpha_0=Y_0-\sum_{b\in S}\alpha_bh_{0,b}W_b.
\]
第二次仅计算保留组：
\[
 \hat Y_1=C^\alpha_0+\sum_{b\in S}\alpha_bh_{1,b}W_b
 =Y_0+\sum_{b\in S}\alpha_b(h_{1,b}-h_{0,b})W_b.
\]
**ALG EXACT（实数加法下）**。α=1时缓存就是\(b_2+\sum_{b\notin S}h_{0,b}W_b\)。pre-gate缓存、b2一次、当前g1调制，当前residual与suffix照原执行。缓存历史来自当时**学生**状态，不得代入Dense历史；不同expert/block不可互用。同样公式可按RAW_BF16及分别post-Q2的HiF4调用成立，但不能把\(Q(h_1)-Q(h_0)\)写成\(Q(h_1-h_0)\)。

用M=31668、D=5120、BF16、40层和单个当前活动expert估算：
- 旧h完整+pre-gate Y：约44.70 GiB逻辑；
- h_S(K=5120)+Y：约24.16 GiB逻辑；
- **仅C0^α：约12.08 GiB逻辑**，对44.70 GiB减少约73.0%；额外构建中间量、权重、allocator峰与host/offload另算。

**构建代价必须付款**：若首次Dense之后又独立执行Σ_S αh0W，是额外selected W2计算；优选首次Dense W2按组累加时同步构造Y0/C0，须通过实际无observer原生测试。如果仅缓存C0但首次Dense输出按不同分组顺序求和，BF16数值非byte-equal，须报告与reference的误差。

信息论界必须严格限定：
- 若要求对任意旧激活、实数下连续线性编码精确保存\(C_0\)，所需自由坐标数至少为\(\operatorname{rank}(W_{2,\bar S})\)，至多D；当前遗漏维度H-K=8704≥D，有可能满rank D，但**尚未实际测rank**。
- 数据流形上若允许有损预测，给定当前可合法取得的Z，最优平方误差预测器是\(\mathbb E[C_0\mid Z]\)，不可约风险\(\mathbb E\,\mathrm{tr}\,\mathrm{Cov}(C_0\mid Z)\)。这只是**0额外缓存**条件下的统计极限，不是具体Wan估值。
- 若限定r维线性投影保存，以校准协方差Σ对剩余残差做PCA，理想同分布重建MSE为谱尾和\(\sum_{j>r}\lambda_j\)；当功能损失涉及跨层读出，应改为加权子空间优化，不能据此声称低能方向可忽略。
- 含byte预算的率失真\(R_{C|Z}(D)\)描述理论压缩前沿；没有来源外样本、数值后端和具体codec，不能报实际bits下界。

## 5. **重要新纠错**：动态量化增益与单缓存并非天然兼容

若第二次调用的量化摘要Z1决定\(\alpha_b(Z_1)\)，所需历史项是
\(\sum_{b\in S}\alpha_b(Z_1)h_{0,b}W_b\)；首次缓存构建时**尚不知道未来Z1**，因此一般不能预先合并成一份\(C_0^\alpha\)。将旧α直接代入会令相同旧/新激活时出现虚假变化。强静态应首先使用固定α。

可检验的**固定支持＋低秩条件读出**独立扩展：
\[
 \alpha_b(Z_1)=\alpha_b^{(0)}+\sum_{j=1}^{r}\theta_j(Z_1)u_{j,b}.
\]
首次Dense缓存
\[
 C_0^{(0)}=Y_0-\sum_{b\in S}\alpha_b^{(0)}h_{0,b}W_b,\quad
 R_{0,j}=\sum_{b\in S}u_{j,b}h_{0,b}W_b.
\]
第二次：
\[
 \hat Y_1=C_0^{(0)}+\sum_{b\in S}\alpha_b^{(0)}h_{1,b}W_b+
 \sum_{j=1}^{r}\theta_j(Z_1)
  \left(\sum_{b\in S}u_{j,b}h_{1,b}W_b-R_{0,j}\right).
\]
在实数下此式与对保留组的同α(Z1)历史增量**严格等价**，而实际使用(r+1)个D维缓存对象。若r=1，逻辑缓存已约24.16 GiB；r=2约36.25 GiB（仍需核算计算r路贡献和meta observer）。静态mask/物理形状可以固定，但动态系数额外乘法和数据重用不能预称免费。全程固定α版本是系统主路线；条件α版本需先证明元信息有净价值后独立进入原生执行。

## 6. 量化元信息的信息—截止—动作合同

原HWG重要假说是**静态W量化的符号/尺度几何×当前A编码的有序符号/尺度，是否可以以低成本预测需要保留的功能方向**。最新BF16 D/G/R没有直接检验该命题，应作为基线而非替代科学核心。

真实HiF4含64 signed S1P2 codes +32bit unsigned scales（层级64→8→4）；其CUDA fake-QDQ当前写出的是signed BF16重建值，code/scale未持久暴露。W1/W2为离线静态量化，当前Q1在W1前、Q2在W1/GELU后动态产生。详见[格式审计](../20261028/HIF4_SIGNED_STORAGE_AND_QDQ_SEMANTICS_AUDIT_20261028.md)、[时序审计](../20261028/WEIGHT_ACTIVATION_STATIC_DYNAMIC_NATIVE_FP4_TIMING_AUDIT_20261028.md)、[QMeta合同](../20261028/QMETA_OBSERVATION_TIMING_AND_EFFICIENT_USE_20261028.md)。

理想Q1码域可近似写\(\hat x_j=s^A_{g(j)}c^A_j\)（HiF4应展开分层尺度；原Smooth/cast另计），则某一W1输出有
\[
 a_i=\sum_j (s^W_{ij}c^W_{ij})(s^A_{g(j)}c^A_j)+b_i.
\]
幅值尺度给范围，符号与**具体位置**决定方向，W静态结构确定哪种位置组合影响保留组/后续GELU；同幅值、同正负数量但符号位置重排可能改变动作。单独scale-only信息不足是ALG反例，并不保证signed sketch在Wan一定有收益。

**强静态（主路线）**：S、α、W2 layout均离线冻结；Q1/Q2量化元信息可用于校准统计及离线选组/功能设计，线上不需要额外router。因此信息再丰富也只能通过冻结的结构表现。**结构静态（独立支线）**：S固定、α在低维基中变化，线上只消费合法Q1 producer可用的短摘要Z1；必须付出code/scale提取、跨CTA归约和额外残差读出成本。Q2未来未计算组的元信息不允许决定当前pre-W1跳算。

按同一HiF4数值路径消融：W-static/step-only→A1幅值分布→ordered signed code→W×A1短投影→组间signed coupling→固定支持在线Q1摘要；设置**同成本普通BF16摘要**基线和paid oracle。统计可定义值信息
\[
 \mathrm{VOI}(Z\mid S)=R^*(S,\varnothing)-R^*(S,Z)
\]
其中R*包括在既定动作集合及实际获取成本下的最佳条件风险；只有**扣除合法获取成本后的净VOI**才可能带来部署价值。使用0-cache条件方差和cross-source regret检验额外信息。HiF4不与NVFP4/MXFP4的scale语义/算子闭包直接等同，NVFP4 W/A block16 FP8scale+global，MXFP4 block32 E8M0scale，各另建native合同。

## 7. 从局部signed Gram到全160位点的静态预算

给定每个合法更新site i=(e,l,k)、K64组集合S_i、缓存参数θ_i、真实量化数值族Q与来源s，定义**当前学生输入上的同状态**余项
\[
 \epsilon_i(s)=\tilde Y_i(\tilde x_i,\tilde C_i;S_i,\theta_i)-Y_i^Q(\tilde x_i).
\]
不能先用teacher-state ε_i替代学生余项；gate应按当前位置注入或明确边界。局部组增量向量v_{i,b}定义须明确pre/postgate并匹配所选择的输出目标。

建议三个层级：
- **L0**：统一来源加权/归一化的局部Gram；相同sample权重下 \(G_{ab}=E\langle v_a,v_b\rangle/(||d||^2+\varepsilon)\)，且\(p=G{\bf1}\)。原v2 D/G用绝对内积、R用归一化误差，须统一主目标以便分辨支撑 vs 增益。
- **L1**：用选定后缀VJP/JVP或同状态finite intervention估计各site方向的下游敏感性与跨层耦合；比较只对角\(\sum_i \epsilon_i^\top K_{ii}\epsilon_i\)和有符号交叉\(\sum_{i\ne j}\epsilon_i^\top K_{ij}\epsilon_j\)的来源外预测能力。K无法直接全存，只估预定window、source-balanced sketches和高风险对。
- **L2**：在可执行候选集合上，使用student free-rollout及视频/系统wall评价并推翻错误代理。

全局形式（不是立刻训练一个巨大优化器）：
\[
 \min_{\{S_i,\theta_i,C_i\}}\;
 \underbrace{\mathbb E_s \mathcal L_{\rm video}(V_{\rm cand},V_{\rm dense})}_{\text{最终真实目标}}
 +\lambda_{\rm tail}{\rm CVaR}_\beta(\mathcal L_s)
\]
\[
 \text{s.t.}\quad\sum_iK_i\le B,\quad
 {\rm PeakMemory}(\{C_i\})\le M_*,\quad
 {\rm SystemWall}\le T_*,\quad
 S_{e,l}\subseteq\{1,\ldots,216\},\quad\text{whole K64, frozen}.
\]
其中quality包含多轴非劣条件，不能简单合成单一VBench分数；CVaR仅是预先规定的风险代理和来源尾部约束，不是统计保证。预算是第二次合法更新site的组数总和；若每层/专家不同K，成本需按kernel shape、padding、host IO、cache构建重测。全局预算探索要比较**High K < Low K / 相等 / High K > Low K**，再允许专家内部40层不均匀；早/晚层和高/低SNR混淆须说明。已测两来源并不足以冻结64/96为最优。

除组间cross-term，还需量化误差与结构删组交互：
\[
 e^{total}
 =\{\tilde f_Q-f_Q\}+\{f_Q-f_{BF16}\},
\]
总损失有二者的交叉项；Q1/Q2非线性还使\(\tilde f_Q-f_Q\)本身随Q态改变。量化误差、稀疏误差、缓存误差均可与suffix敏感性相关。后续student input drift不是可直接与这些同状态误差简单加和的“第五个独立随机源”，必须沿扩展动力系统传播。

## 8. 方案族与可证伪预测

**候选S0 / SICR-STATIC（优先机制）**：
- 40×2个expert-block分别离线冻结K64支持S和固定α，用HiF4编码统计/静态W几何、跨来源signed耦合及少量后缀敏感约束优化；
- 第一次同expert Dense直接构造单份C0^α，第二次只算原合法保留组，并按当前gate写回；
- 调整High/Low与各层预算但**不对输入做在线组选择**，缓存只在同expert合法第二步消费；
- RAW BF16 D/G/R、no-cache static、Y-only、历史P64、HiF4 signed-static分别保持数值/预算身份，不能跨数值路径直接比较为元信息因果增益。

**候选S1 / SICR-QREADOUT（条件型、另审）**：
- 不改变S/组算子shape，利用Q1现有码/尺度生产时获得低维有序signed摘要，计算低秩θ(Z1)；
- 缓存(C0^(0),R0,j)保证严格增量语义；固定α及同预算普通BF16摘要是强对照；
- 只有source-heldout净VOI、native无observer系统和全视频均通过才视为有价值。若r≥1失去缓存优势或metadata observer延迟大，应保留S0而非为了符号信息牺牲系统成本。

**候选S2 / SICR-COMPRESS（缓存研究）**：
- 对C0^α进一步考察固定低秩/低精度编码及无缓存连续替代；按conditional variance及带后缀读出的weighted spectrum筛选维度；
- 必须验证突发、细小物体、位移、光照/相机运动、Low阶段和极端来源；旧compact6016/affine54失败不能充当新codec的绝对否证。

**可推翻机制的具体预测**：
P1 若后缀敏感配置比局部Gram好但student-video更差，说明当前后缀代理失真或训练集泄漏，应回到有限四角与量化/运动读出定位。P2 若固定缓存单向量与h+Y理论同函数但数值显著不等，优先查bias/gate、QDQ顺序、BF16重排、C0构造/状态时点。P3 若Low保护在更多来源和双seed不优，历史2来源只支持条件敏感，不能宣称通用预算规律。P4 若ordered signed Qmeta对同成本BF16短摘要无regret收益，量化信息的独特部署优势目前不成立；但离线静态W×A统计仍可继续研究。P5 若全层视频质量好而系统wall不降，则归为函数表示证据，暂不宣称加速。

## 9. 执行前实验计划（**DESIGN ONLY**）

**E-A 代数/CPU只读验证**：生成小维数确定性恒等实例，检查C0^α及低秩动态α公式、bias一次、gate当前、rank/PSD/同权p=G1、来源归一化；校准/学生两个state的标签分别核准。若需读取已保存模型大张量作新的数值实验，仍受新scope独立审查约束。

**E-B 现有资产复用**：现有3 HiF4来源、80层对×多臂与Chipmunk真实视频对应的确有字段进行来源级 signed-correlation、low/high预算和层分布统计。对不具有合法当前Q1原code/scale或完整student prefix的字段写MISSING；不能从QDQ BF16重建码反向声称原metadata已导出。使用旧数据做开发、不得称确认。

**E-C 定向新增采集（审查后）**：以多独立image cluster、两独立seed、全部40层×四步、21 latent时间槽、自然raster/同flow object tubes为单位建立matched Dense/单site/双site/学生自由轨迹；预注册采样的跨层pair与High→Low pair、动作/背景/相机/低运动及小物体分层，采集gate、norm2/preW1、pre/postQ1、pre/postQ2、pre/postW2、cache、initial noise和已冻结scheduler参数/精度/代码SHA。**先source split再校准**，row/τ/step不能当作独立n。

**E-D 数值与物理门（审查后）**：完整160位点 neutral=Dense真实重放，首次Dense+第二次全组Δ应按预设误差界；单缓存与原h+Y路径验证同数值族（数值重排不预设byte-equal），检查首次构建C0、当前pre-gate/e5、split W2和权重副本。一次GPU原生局部和完整无observer wall都测；候选跨层分配必须显示真正物理K64/Mtile跳算，记录W1/W2、Q1/Q2、cache build、host offload、memory、GEMM shape、flush/overlap/critical path。既有两个Dense首次调用决定主W1/W2理想算术最多省50%；80/80、K=5120仅有约68.5185%主投影比，不含其他成本。

**E-E 真实视频确认（审查后）**：P0最小门采用Dense、原v2公平固定signed对照、S0固定gain单缓存、必要的Low保护与no-op（依资源审后冻结）；完整4step×40block、相同noise/prompt/conditioning。P1需针对明确机制使用配对High-only/Low-only/Both、局部/跨层预算、量化元信息消融；P2在从未曝光image-cluster、双独立seed上冻结支持、增益、缓存与后缀代理权重；官方VBench-I2V只按合法source与其独立采样合同使用，另报告连续RAFT40对、运动主体/相机拆分、动作/文字条件、时间闪烁、重影、小物体、最差来源及全81帧完整视频审阅。P0/P1不能充当P2的独立n。

每个新CPU/GPU/采集/视频scope必须按[2026-10-29独立预审状态机](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)：真实另一Agent审查+明确反例→作者修订→reviewer recheck→ACTIVE冻结→启动前SHA/资源复核→真实RUN/EXIT。该审查备忘录**不批准任何scope**。

## 10. 与最新前沿原文的关系（仅确切作用，不冒称复现）

- [Li & van der Schaar, ICLR 2024, On Error Propagation of Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b465dd58ac50e1b0b22894fd581f62f-Abstract-Conference.html)：模块扰动→累积扰动；本文件拓展到FFN site与缓存增广态，不表示继承其定理全部条件。
- [Liu et al., ICML 2026, Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models](https://proceedings.mlr.press/v306/liu26fv.html)：stepwise量化误差传播放大与纠偏；其论文图像任务结论不等于本项目视频校准成功。
- [Peng et al., ICLR 2026, ERTACache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/84d395725a9b40cb4a49d84478ac24c7-Abstract-Conference.html)：区分feature shift与step amplification；本项目重点是**固定FFN组与缓存构造**，不是照搬动态调整去噪步。
- [Haghighi & Alahi, CVPR 2026, SenCache](https://openaccess.thecvf.com/content/CVPR2026/html/Haghighi_SenCache_Accelerating_Diffusion_Model_Inference_via_Sensitivity-Aware_Caching_CVPR_2026_paper.html)：latent/timestep敏感度支持分时预算，但该论文动态缓存时序，不能直接证成静态支持。
- [Liu et al., CVPR 2025, TeaCache](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html)；[Liu et al., CVPR 2026, D2Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_D2Cache_Second-Order_Delta_Caching_for_Higher_Video_Diffusion_Acceleration_CVPR_2026_paper.html)：输入时间步调制、变化率/二阶差提示历史增量不能仅看瞬时幅值；四步同expert仅两次调用，不自动拥有三点二阶缓存历史。
- [Chen et al., CVPR 2026, SVD-Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)：主子空间平滑、残差低能高波动，支持将变化能量/可预测性/下游作用分别测量；不证明低能残差可以永久丢弃。
- [Cui et al., ICLR 2026, BWCache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Abstract-Conference.html)：block之间的step依赖差异，支持非均匀层预算，而非证明当前Wan 40层的相同分布。
- [Wan-Video/Wan2.2 official](https://github.com/Wan-Video/Wan2.2)：High负责早期结构、Low负责晚期精炼；本项目Low P64两来源需要独立确认，expert×time尚不可因果分离。
- [OCP MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)和[NVIDIA NVFP4 spec](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)：block32 E8M0与block16 FP8+global有不同编码语义，不提供当前Wan hot-path元信息免费访问。
- [VBench, CVPR 2024](https://github.com/Vchitect/VBench)：运动平滑与dynamic degree不是同一个目标，评估需分维度，不能让静止视频靠一致性通过。

**Novelty边界**：组合优化signed Gram、线性固定gain、特征缓存、敏感度建模和跨步误差传播均有先例；SICR只是本项目**待验证新组合假说**。候选学术贡献可能在于“原生signed Qmeta可及时获得的有限信息→固定物理组支持→最小历史充分状态→全160 FFN因果传播与视频质量”的联合可执行闭环，而非单个公式或命名。

## 11. 独立评议的可交付结论

**DECISION = REVISE（可继续研究与方案修改；执行许可 = NO）**。

原Signed/Gain v2应保留为BF16、相同K/相同mask的强因果对照，但应在正式大规模P0/P1/P2前：
1. 修复D/G/R主局部目标不一致问题、缓存C0固定gain表示与真实成本、Qmeta分角色/时点。
2. 用上述augment-state与带符号K_{ij}开展可证伪全层假说；独立采集不足的量不许臆测。
3. 把Low保护与40层非均匀预算作为**待实测的竞争机制**，而非固定数值结论。
4. 以有限真实视频、来源级运动/细节/语义及无observer物理成本为方法的唯一最终验收证据。

本文件只提供数学、论文先例与后续审查条件；**没有在本轮运行GPU、CPU科学实验或生成新视频**。
