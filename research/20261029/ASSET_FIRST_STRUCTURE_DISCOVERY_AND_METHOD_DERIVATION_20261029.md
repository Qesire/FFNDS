# FFNDS · 从已存证据与新增原生采集共同推导方法

2026-10-29 · METHOD_SNAPSHOT · **已保存数据复算 + 数学推导 + 原生计算路径 + 文献映射**。本轮不是新GPU、生成、训练或视频评测。科学对象为Wan2.2 I2V A14B、四步high/high/low/low、40 Block、每个完整视频160次FFN执行；现有原BF16、旧HiF4 fake-QDQ、待跑原生NVFP4严格区分。

本报告按 **真实观察 → 可重复的结构判断 → 竞争解释 → 可实现函数 → 最便宜的判别证据 → 完整视频与系统成本** 构造。旧M-B末层读出和signed误差保留为解释工具；新研究不预先把“补算/消除SSE/符号抵消”放在中心。

数值审计：[资产交叉复算 JSON](ASSET_FIRST_METHOD_DERIVATION_RECALC_20261029.json)；直接依据：[原BF16八来源160位置实物](ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.md)、[全80物理函数rank960](RAW_BF16_FULL80_R960_SPECTRUM_RESULTS_20261029.md)、[连续tile实测](../20261025/E0_SHARED_TILE_PRE_W1.md)、[Q2 64组闭包](../20261025/E0_SHARED_TILE_CLOSURE.json)、[UB1二十来源视频/四步误差](UB1_20SOURCE_FULL_DEPTH_VIDEO_STATE_EVIDENCE_RECOMPUTATION_20261029.md)。

**扩大研究的执行解释**：本文提供旧资产中导出的四个竞争机制，不是算法候选或数据规模的天花板。若旧8源BF16的256随机行无法推断视频时间结构，应主动采集全40层×4步的时空tubelet；若shared-r960留源不稳定，应进一步获取来源分层/新状态；若需要native NVFP4的W/A量化尺度、全局尺度或动态码字，应采集真实producer。新物理问题或论文先例可以开启第五、第六条机制线。科学问题与反例决定何时扩大数据，而不强制在已有数据上寻找可用答案。详见[数据采集合同](../agent_protocol/EVIDENCE_DRIVEN_ACQUISITION_20261029.md)。

## 1. 原始计算图和信息时点

对固定expert e、层l、step t以及token行i，原始BF16（权重依行向量右乘）：

\[
a_{tli}=x_{tli}W_{1,el}+b_{1,el},\quad
h_{tli}=\operatorname{GELU}_{\tanh}(a_{tli}),\quad
y_{tli}=h_{tli}W_{2,el}+b_{2,el},\quad
r^+_{tli}=\operatorname{BF16}(r_{tli}+g_{tle}\odot y_{tli}).
\]

D=5120, H=13824, M=31668=21×29×52，其中21是视频**latent时间槽**，与4个去噪step不同。原BF16无Q1/Q2。旧HiF4在W1前Q1、GELU后Q2，4bit code**带符号**，当前fake-QDQ最终输出BF16并做GEMM。NVFP4/MXFP4必须另建原生W、A scale/packing和物理闭包。

原BF16 corpus：8不同图像簇×4step×40layer=1280数据包，每位置256真实同坐标行，包含W1输入、post-GELU h、W2 y、g、残差前后。6源校准、2源旧开发，**均为历史曝光**。8源中同一step/block的gate、shift、scale一致；给定冻结的step输入与原checkpoint，这是提前可用的条件系数。同一expert使用相同W1/W2执行其两次step，step差异来自输入分布和调制，非同一expert换权重。

## 2. 观察A：在输入迁移上，跨步共享函数反而比逐步单拟合稳定

原BF16 r960，每个expert×block一个物理W2，保持完整W1/GELU/b2/gate，只有W2采用低秩两次GEMM。

| 单位为source-equal normalized-postwrite相对SSE | shared80 rank960 | step160 rank960 | 数据身份 |
|---|---:|---:|---|
| 6校准源训练、全部80物理函数 | 0.004553 | **0.001845** | 同source训练 |
| 6校准源LOO、预设8物理锚点 | **0.012858** | 0.014393 | shared改善10.66%，6/6来源 |
| 2旧开发源、全部80物理函数 | **0.015753** | 0.017850 | shared改善11.74%，2/2来源 |

**严格解读**：LOO训练折互相重叠，LOO锚点只有8个，且旧dev被方法研究曝光。这是发现“逐步独立拟合可能过拟合或共享表示更稳”的实测线索，不能当作全40层视频验证。比对的两种因子可能在输入子空间、优化和正则上不同，不能直接推断“不同step无需自适应”。

**竞争机制**：（H1）同expert物理W2恒定，step拟合捕获短样本偶然方向，增加估计方差；（H2）step特有方向真实存在，但6源训练不足，未被可靠估计；（H3）局部LN/gate响应筛掉部分误差，学生自身状态和下游attention会改变排序。以同基、同秩、source-held以及全160位学生自由生成区分三者。

### S1 / CSFB：共享核心基 + 收缩的step特化读出

令共享左因子U属于R^(13824×r)，共享右因子V属于R^(r×5120)，step专用细修正ΔV_t与权重α_t∈[0,1]，则

\[
\widetilde y_t(h)=hU\left(V+\alpha_t\Delta V_t\right)+b_2.
\]

其中α=0精确退化为现成shared80比较器。**对固定step预先融合右因子**，运行仍是两次低秩矩阵乘法；增加step相关存储，但不额外增加第三个乘法。真正新假设是：**step差异集中在共享中间子空间的低维读出系数，而不是需要完全不同的表示子空间**。

校准损失按来源先等权、来源内按token/step聚合：

\[
L=\frac1{|S|}\sum_{s\in S}\sum_{t:e(t)=e}w_t
\frac{\sum_i\|g_{tle}\odot h_{stli}(W_{2,el}-U(V+\alpha_t\Delta V_t))\|^2}
{\sum_i\|g_{tle}\odot (h_{stli}W_{2,el}+b_2)\|^2+\epsilon}
+\lambda\sum_t\|\Delta V_t\|_F^2.
\]

这只是候选的可解释**离线训练/校准目标**，视频保真由free rollout证明。已有shared80与step160因子的基可能不一致，直接相减不是合法ΔV：必须先投影到同一个U再拟合右因子。可以先α=0直接做已有BF16共享rank960视频基线，再在不看验证视频前冻结α/λ；不通过局部验证不得自动叠加gain补丁。

**物理账本**：W1原始=DH=70,778,880 MAC/token；W2原始=DH；低秩W2=(H+D)r。r=960时W2原始MAC保留25.6944%，整个FFN MAC保留 **62.8472%**，即函数级MAC省37.1528%。实际需计完整160次W1、两次W2 GEMM、mem materialization、权重常驻/加载、BF16舍入以及端到端Latency。现有shared80因子导出2.91GB，step160导出5.82GB；不能把2.91GB全当每step每GPU都常驻或节省的实际显存。

## 3. 观察B：便宜的信息应当预测“哪个计算动作值得执行”，而非精确重建整个teacher

同一BF16真实h在**W1/GELU已经执行、W2尚未执行**时可用。我们要决定的是哪一组低秩函数坐标值得执行，不是先预测全部原始W2输出。已有r960可拆成rank640核心＋rank320可选块（第一实验在同一已导出、排序经校准的rank960基上划分；拆分前先校验因子语义与秩排序）。

### S4 / FRANK：基于动作收益草图的可变FFN输出秩

令B0=U0V0，B1=U1V1，a∈{0,1}，以完整原始W2为离线参考：

\[
\widetilde y(h,a)=h(B_0+aB_1)+b_2,\quad
e_a(h)=h\left(W_2-B_0-aB_1\right)\operatorname{diag}(g).
\]

动作增加rank块后减少的真实**局部平方误差**为Δ(h)=||e0||²−||e1||²。若Δ可被廉价预测，就有可学习的真实决策对象。取部署前固定、独立标准高斯的随机投影Ω∈R^(5120×k)，提前构造

\[
P_0=(W_2-B_0)\operatorname{diag}(g)\Omega,\quad
P_1=(W_2-B_0-B_1)\operatorname{diag}(g)\Omega,
\quad \widehat\Delta(h)=\frac{\|hP_0\|^2-\|hP_1\|^2}{k}.
\]

**数学恒等式**：对任何固定向量v，EΩ[||vΩ||²/k]=||v||²，因为E[ωωᵀ]=I。因此EΩ[Δhat(h)]=Δ(h)。它是对离线探针抽样的无偏性；一个固定probe部署到新来源时可能产生源级系统误差，必须source-held检验。k独立高斯探针的单个平方范数估计相对标准差为√(2/k)：k16约35.4%，k64约17.7%；差值估计波动取决于两个error向量的相关性，需实测负收益选中率/错失重要tile的尾部。

该在线探针**仅消费已有的post-GELU h**，耗费约2Hk的MAC/token；k16为442,368 MAC，约现成r960 W2低秩MAC的2.43%，约额外rank320 MAC的7.30%（仅算术比率，未含探针内存访问/路由/packing）。根据空间tile将行Δhat先求和，再以同一tile统一是否执行B1；两动作的执行必须使用真正高效的可变rank批处理，否则固定r960会更快。

**这条路线不默认补算Dense**：基础秩始终是压缩函数，可选动作只执行额外rank块。它从Prox等论文借鉴“提前获得低价行动信息”的研究范式，但不是对原神经元做TopK。第一项最小实验利用现成8源×160位×256行h/W2/r960，比较真实benefit与k16/32/64 probe、step-only、恒定rank、oracle的**source-held regret**，连带估算false negative、tile共享与成本。若条件收益小于probe开销，保留静态shared-r960，停止此路。

潜在额外研究：如果得到中低层对运动有真实作用的受保护函数，可把Δ的读出度量换成离线冻结的有限响应，而不能在实际推理时把Dense未来视频/真实后缀Jacobian当作免费的输入。

## 4. 观察C：原生时间与tile冗余是层异质的，而且真正的物理选择单位受量化控制

旧HiF4连续M128实验只有2暴露source与5个Block，其视频时间信息不能直接从raster的M128推断。按Block 0/10/20/30/39顺序，**同token块平均变化占创新能量**分别为65.27%/9.84%/13.60%/14.3%/77.8%。25%列保留的后验共享对角对创新SSE分别11.98%/38.99%/42.21%/40.12%/7.57%。中层与首尾层呈明显结构异质；只在末层证明共享显著有偏。

### S2 / FTFD：函数条件的时空计算单元发现

分清去噪step t、latent视频时间τ、Transformer层l。对真实匹配的局部视频token群T，定义正常的同一FFN输出：

\[
Y_{T,t,l}=\mathbf1\mu_{T,t,l}^{\top}+E_{T,t,l},
\quad\mathbf1^\top E_T=0,\quad
\|Y_T\|_F^2=|T|\|\mu_T\|^2+\|E_T\|_F^2.
\]

这只是**精确发现性分解**。研究各层/step在运动对齐/空间邻接下的共享均值、低rank剩余和token-specific剩余；只有实际证明其中一类可被廉价表达/复用，才构造对应的计算动作。完整1280包原BF16随机256行有利于跨来源、跨层统计，却没有充分的同空间位置×21时间槽配对；旧M128 raster也不等同于运动对齐。下一数据需求是**全40层的匹配视频时间token**，先用existing row locator检查能复用多少，缺口再做冻结的被动Dense采集，保留真实step/row/τ语义。

现有FIS-DiT已经提出少步模型frame interleaving，所以**固定交错帧不是新贡献**。可竞争的创新是**FFN函数级的layer-dependent时空残差组织 + 真实前W1可得条件分组 + 对GELU/Q2/Attention后缀作用的守恒**。和FIS-DiT、Chipmunk、均匀帧/Block路由比同条件成本，保留视频主体动作、相机与背景。

### S3 / QFAR：恢复“真正早决策”的可执行集合，而不是继续做任意列能量TopK

旧HiF4下，完整64隐藏组是与原Q2 64→8→4尺度结构相容的单位；216组×64=13824。原连续tile中，25%**任意列**保留的Q2 group64闭包覆盖**100%的组**，其原生W1工作并未按列稀疏率省掉。直接25%**完整64组**保留的旧对角准则，其创新relative SSE=55.89%，而任意列对角准则为28.18%。这个实测淘汰了“先任意列TopK再按密度计算W1成本”的假设，留下**是否存在可提前预测的、真正低风险整组联合动作**。

对固定当前post-Q1 x和完整Q2原数值组（含signed code、b2、gate、cast），
\[
y_Q=b_2+\sum_{j=1}^{216}v_j,\quad
d_j=g\odot v_j,\quad
\Delta y(m)=-\sum_j m_jd_j.
\]

在与物理相容的完整组mask集合A64上，比较合法pre-W1信息Z（step/expert、权重静态表示、已付的Q1 code/scale、低成本上游特征）驱动的source-held**行动风险与净成本**。先用真实保存的组贡献判断合法动作oracle是否存在，再检验廉价的pre-W1符号/尺度信号能否**提前**预测该动作；历史有符号Gram与相互抵消在此只充当离线判别工具。若整个合法动作集的oracle都无法满足视频对应的保留风险，转S1/S2/S4；若oracle可以而cheap-Z不能预测，研究观察时点/代价而不是调更多同一组打分。

**格式隔离**：BF16 corpus没有Q1 FP4；HiF4是signed S1P2 fake-QDQ；原生NVFP4 W4A4的局部16组、动态A scale和global scale产生不同闭包。S3在PRO6000上必须重新证明可执行动作的producer、布局和物理成本，不能移植HiF4的64组数值。

## 5. 符号消除和GELU代数：保留为竞争机制，不预设为目标

旧连续token分析中，同一个GELU_tanh严格分解
\[
\operatorname{GELU}_{\tanh}(a)=a/2+\psi(a),\quad
\psi(a)=\frac a2\tanh\left[\sqrt{2/\pi}(a+0.044715a^3)\right].
\]

其linear-half和nonlinear-even函数的跨步创新经过W2投影后的cos在5个观察Block约**−0.64至−0.82**。这是实测**强负交互**：忽略其中任何一项都可能严重高估或低估真实变化。可以研究成对/联合函数结构、等价折叠和量化传播，却不以“把signed cross变负”直接作为视频质量目标。原先Affine54/Whole94/补偿实验的失败只适用于其特定版本；任何新折叠都要先与这条负交互和native非线性数值条件匹配。

旧UB1 20来源全深度free视频中，RAFT候选/Dense来源均值比0.1672，18/20来源运动减少；有6来源成像分数提高同时运动减少。该已测负例为S1/S2/S3/S4设定真实**动作及运动保留**终端，而不是给BF16新方法附加凭空的同一量化损失。旧Affine54在320个site中318局部绝对operator error较teacher-prefix低，依旧有大latent drift，警告任何路线都不能用局部误差宣布成功。

## 6. 可判别的对照、优先队列和视频准入

| 优先级 | 方法 | 当前资产/可启动动作 | 首次相反预测 | 本轮实测状态 |
|---|---|---|---|---|
| **P0 视频基线** | BF16 shared80-r960 vs step160-r960 vs同BF16 Dense | 两套因子均已存在；完整160位置原生函数已定义 | 原始BF16是否能用低秩W2保持视频动作；shared局部优势会否延续到真实轨迹 | LOCAL/SRC，VIDEO/SYS未测 |
| **P0 资产诊断** | S1 共享基＋step-shrunk | 8来源全80因子和256行；需在共同U下学习ΔV | H1共享稳定 vs H2少量可泛化step特化 | 现有r960源级迁移复算，新增收缩OPEN |
| **P0 资产诊断** | S4 可变rank的行动价值草图 | 同8来源h/W2/r960/gate，无需新生成 | 按收益选rank是否优于固定rank，probe成本是否真低于额外rank | 代数无偏性已证明，真实regret未测 |
| P1 条件发现 | S2 时间/空间FFN函数单位 | 旧2源5层连续M128；8源随机256行×160位 | 中层运动配对后是否有可重用共享函数结构 | 5层异质性OBS，全深度匹配token仍OPEN |
| P1 早期量化 | S3 物理完整Q2组行动 | 旧HiF4组闭包、CAL24/saved true h及原quant SRC | 新whole64联合选择是否在真实source-held优于对角策略 | 25%旧对角负对照，组束算法尚OPEN |
| 辅助 | M-B / 有符号响应 | BF16末层head、E0签名、UB1四步signed | 实际正常路径哪些方向影响视频动作 | 机制解释/有限干预，不是所有主线前置要求 |

**顺序原则**：S1/S4/S2/S3是并行可竞争研究问题。第一条完整视频测试优先使用**现成BF16共享r960因子**——它是操作结构的可靠基线而非新论文首创；之后以video+system证据筛选真正的方法贡献。不要用“末层先证明”作为下一候选必须经过的漏斗。S4的rank依赖原已付W1，能省W2但不省W1；S2和S3有潜力实际省W1，但证据更早期。

**视频级正式验收**：所有候选必须在真实4×40 site中执行；首先≥20不同image clusters的同BF16 Dense/候选paired full-video，image/prompt/初始noise/seed/scheduler一致，记录主体动作事件、条件语义、背景/光照、相机、时序、RAFT及画质向量和真实全wall、FFN、packed/observer/offload成本。对新候选以预冻结未暴露的image clusters和独立seed确认，旧8个BF16 corpus来源和20个UB1来源均属历史开发曝光。原生NVFP4 lane另设matching Dense和数值基线。所有执行按ACTIVE冻结具体sources、GPU预算和运行身份；本轮未新增执行权限或作业。

## 7. 可复用资产与失效条件

- **BF16 full-depth raw8**：producer来自20261029 corpus，HF manifest定位；数据是原始BF16 teacher h/W2、门控、r/u，source+token+step绑定；仅同checkpoint/同precision/同row和变换语义可复用。不能用作旧HiF4 Q1或原生NVFP4 A4 scale。
- **BF16 shared80/step160-r960**：producer来自20261029 spectrum；factor身份需和原W1/W2及gate/Step绑定；原已导出因子可以作为候选但不是新视频结果。若变更checkpoint、numeric backend、rank基对齐或调制时点，标stale或重新验证。
- **旧HiF4 M128**：2来源×5层连续raster、同expert step差与Q2 64组依赖；仅支持相应结论，不能声称40层代表性或运动对齐。
- **旧UB1 20来源paired视频**：旧HiF4开发集条件及前向身份明确；适合负例、动作评估设计；不能和原BF16共享Dense性能分母。
- **本次结构交叉复算**：JSON包含source-held差、样本规模、物理MAC和旧组闭包；输入全部是旧报告公开标量，未读新增GPU Tensor，任何新实际测量需另生成producer/receipt。

### 最强文献对照与原创性边界

FIS-DiT已做few-step frame interleaving；Chipmunk已做跨step列增量；SVD-Cache已有主/残子空间异质；Prox已有代理路由；SliceGPT/SVDQuant已有表示压缩，QCFB已有whole64 signed风险数学。**shared-r960、普通低rank和随机投影无偏本身不构成原创贡献。** 真正可能成立的新贡献须在跨步共享功能的**泛化结构**、probe的**实际可执行决策价值**、视频FFN**有条件时空函数单位**或量化元信息的**合法提前行动**中给出与这些先例不同的机制、对照、完整视频及物理加速。

主要原文：[FIS-DiT](https://arxiv.org/abs/2605.11869)、[Chipmunk](https://arxiv.org/abs/2506.03275)、[SVD-Cache](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html)、[Dynamic-in-Few-Step](https://arxiv.org/abs/2607.06631)、[Prox](https://arxiv.org/abs/2607.27591)、[SliceGPT](https://proceedings.iclr.cc/paper_files/paper/2024/hash/316648eb8b4ffb6010f531b07848c300-Abstract-Conference.html)。
