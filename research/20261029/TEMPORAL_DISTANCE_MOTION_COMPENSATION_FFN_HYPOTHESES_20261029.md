# MTC / TDIC · 视频时间距离、运动对应与FFN低秩时间创新补偿

2026-10-29 · **SCIENCE SEED / ALG + SRC + LIT + existing OBS + OPEN**。这是独立可竞争的机制研究方向，与S2-FTFD协作，且允许演化为新的S5方案；不是经过另一个Agent支持的正式方法提案，不是已部署motion compensation。本文件基于已审的BF16计算图、真实视频负结果和原论文建立判别式。**本次仅建档与推导：0新增tensor/GPU/模型/视频/质量运行，0新ACTIVE模型授权。**

**后继实测更新（独立于本文创建时的0运行声明）**：E1[十新图完整21τ×8xy×全160原生及CPU观察](RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)、E2[同Dense-h BF16因子及原W2 M168舍入对照](RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)均已按ACTIVE注册并实际exit0。shared门后DC/AC相对误差为0.02932/0.18415，原W2小M的AC误差约0.000052；这支持M1同数据/同秩表示对照作为下一项实现。E3运动邻域、Student prefix与M2运输均仍OPEN；未新增同行支持票。原创建时的E1/E2状态保留其时间身份，现行状态以ACTIVE与上述RUN为准。

## 0. 科学问题与值得立刻解决的矛盾

**观测O1（VIDEO）：** 原始BF16、完整4去噪step×40 block、20个已暴露开发image clusters、60条81帧Dense/shared80-r960/step160-r960视频实际生成。来源等权连续RAFT幅值Dense=7.05802、shared=3.45723、step=3.36421；均值之比分别0.48983/0.47665，两候选各19/20来源低于Dense。FFN CUDA单次观测约1.49×，完整生成仅约1.057×/1.042×，额外因子显存使峰值上升。八轴官方评分在实验自身的单seed、非官方五次独立采样条件下已执行，质量未通过。证据：[VIDEO20_Native](RAW_BF16_FULL160_VIDEO20_NATIVE_RESULTS_20261029.md) / [VIDEO20_Quality](RAW_BF16_FULL160_VIDEO20_QUALITY_RESULTS_20261029.md)。不能把运动衰减唯一归因于W2的时序信息损失。

**观测O2（稀疏时空OBS）：** 8个历史暴露source、1280个来源×step×block包、26,389个同xy跨latent时间配对，时间相同但xy不同为对照。归一化平方距离：真实W1输入0.66234 vs 0.80855，GELU h 0.63910 vs 0.77085，W2 Y 0.59292 vs 0.71617，gY 0.34187 vs 0.41323。同坐标优势在W1输入就已存在，不能直接说是FFN创造的时序冗余。只有2511个相邻τ配对、**0条完整21τ轨迹**；此OBS无运动对应标签。证据：[S2跨τ观察](RAW_BF16_S2_TEMPORAL_PAIR_OBS_RESULTS_20261029.md) / [真实row重放](RAW_BF16_TEMPORAL_ROW_REPLAY_20261029.md)。

**问题改写：** 是否存在可以从当前廉价输入/已付计算提前识别的**运动对应和时间创新**，使有限的FFN W2函数预算优先保留决定视频运动的变化方向？“让特征更相似”不是目标；目标是降低实际计算且保留Dense的动态内容。研究先分清 (i) 静态坐标跨τ接近，(ii) 运动对齐后的对应，(iii) 当前FFN误差与时间创新之间的关系，(iv) 学生自由轨迹因果传播。

## 1. 原生计算对象、坐标和信息时点

- 固定真实图像簇s、去噪步t∈{0,1,2,3}、expert e(t)、block l∈{0,...,39}，每次FFN有视频latent时间索引τ∈{0,...,20}及网格p=(y,x), 0≤y<29, 0≤x<52；token行号 τ·(29·52)+y·52+x。**τ与去噪步t不是同一个时间轴；21个latent槽不是21个生成视频帧。**
- 原BF16路径：V1（W1前，D=5120）→ W1(5120→13824)→ GELU(tanh)→ h∈BF16^13824 → W2(13824→5120)+原b2 → gate e5逐通道乘 → 与残差r按原BF16次序写回u → 后续Attention/Block/去噪步。原BF16没有FP4 Q1/Q2；HiF4 fake-QDQ及原生NVFP4不能直接复用同一个数值恒等式/基线。
- 当前对照shared80-r960和step160-r960替换**W2**为两次BF16 GEMM：z=BF16MM(h,U)，ŷ=BF16Addmm(b2,z,V)。原W1、GELU、bias、gate与写回保留。其输入h在W1/GELU之后可用，因此此时的判断不能称作“节省W1”。若希望W1前决策，必须从V1或更早的张量建立**单独**预测器，并计入成本。
- Wan是一个完整视频token并行的FFN调用，不是严格自回归按τ逐帧解码。所谓“从τ−1传播到τ”是**同一次FFN内部的调度、重排/同步和数据复用动作**；不能默认上一τ输出已免费算出、缓存存在或时间顺序串行。运动对齐在W1前和W1后做，能省的算子与物理代价不同。

## 2. 三种“时间距离”及其正确解释

同一数值边界的特征f_{t,l,τ,p}（先测V1和h），定义归一化距离

d_same = ||f_{τ,p}−f_{τ−1,p}||² / (||f_{τ,p}||²+||f_{τ−1,p}||²+ε).

在固定**同一source、去噪步t、block l**下，对候选空间邻域N(p)定义

q*(p,τ)=argmin_{q∈N(p)} d(f_{τ,p},f_{τ−1,q}),    d_align=d(f_{τ,p},f_{τ−1,q*}).

比较 (i) 固定xy，(ii) 同τ、异xy距离匹配负对照，(iii) 邻域最优，(iv) 相同代价随机邻居，(v) 显式来源视频光流/AMF的**离线oracle**，以及往返一致性、遮挡和纹理歧义。记录 argmin 偏移、匹配margin、次佳候选、位置、τ间隔与source tails。**d_align变小只说明该边界特征有更近的邻居；它本身不是物体运动向量、真实光流或动作保真证明。** 需用可获得的原始视频运动/人工动作注释进行验证。

新10来源全21τ×8xy原始teacher采集part0已在ACTIVE登记为RUNNING_FIXED_SCOPE，但**8固定xy只含少量预置横向邻对照，不能支持任意N(p)的局部匹配、真实运动轨迹或遮挡测试**。需先核实其实际RUN/EXIT；若要估计N(p)搜索/实体flow，新增连续空间邻域、完整网格的分层采样或真实视频运动对应producer，并注册一个有限新scope。不拼接不同step、source或teacher/student状态的行去制造配对。

## 3. 对W2误差的精确分解：哪里有可补偿的自由度

在**同一step、expert、block且相同gate向量g**的连续算术代理模型中，令原W2矩阵W、rank-r矩阵B=UV，二者均保留同一个b2。令E=W−B，hτp是post-GELU真实特征。教师输出与候选的门控差定义为

e_{τ,p}=g ⊙ [h_{τ,p} E].

对任一跨latent时间候选对应点q，线性恒等式为

**e_{τ,p}=e_{τ−1,q}+g⊙[(h_{τ,p}−h_{τ−1,q})E].**

它将“可传播的前时刻误差锚点”与“时间创新造成的新误差”分开；即便q匹配错误，这个代数式仍成立，但创新可能不小。若两个位置的gate不同，则需额外加

(g_{τ,p}−g_{τ−1,q})⊙[h_{τ−1,q}E]，

此处不能默认为0。更一般的带gate形式统一为 eτp−e_prevq=gτp⊙(ΔhE)+(gτp−gprevq)⊙(hprevq E)。原始BF16二GEMM、bias cast、门控乘、写回的舍入使这一表达**不是**原生计算逐bit恒等式；离线需比较连续代理与真实typed后处理。原b2在同源同一步的W2误差里抵消，**但在Dense Y的时间DC/AC或跨不等bias配方比较时不能随意丢弃**。

**数据可辨识性边界：** 若vB=0而vW≠0，则h和h+v经B得到相同输出却有不同Dense W2结果。仅靠压缩后的视频/特征距离无法普遍恢复缺失方向；补偿必须消耗尚有该方向的信息（V1、h、额外因子/专家、真实原W2锚点或其它可靠可观察量）。**e_{τ−1,q}本身不可免费在线得到**，因为它使用了未知的原W2真实输出；任意“残差运输”算子必须声明锚点计算、存储、可用性与刷新次数，或用一个明确的低成本误差预测器替代。

## 4. 不预设必须补算：三个竞争的可构造机制

**M1 / Temporal-innovation weighted representation（无运动warp、无Dense补算，优先）**：
按同一xy完整21τ拆分 teacher gated Y（含bias、gate）与候选的时间均值DC及中心化AC。在原rank960矩阵/参数预算下，引入预冻结的DC-AC权重及跨source权重，在原post-GELU h上**重新拟合同一物理W2低秩函数**，保留相同两GEMM与完整W1；与普通输出weighted rank、S1共同U+step readout、同rank不变输出误差对照。最先验证“旧r960是否把表达能力投入持久外观而损害时间变化”。这不是改变视频中实际运动幅度的后处理；必须最终保留完整视频运动与画质。

**M2 / Aligned temporal innovation factor（运动对齐，增量支路或不同表示）**：
用真实且在决策时点可见的cheap V1/低维h投影建立q*(p,τ)，并按在线tile统一，令Δh=hτp−hprevq。比较固定xy增量、对齐增量与相同cost的随机邻居；训练小型U_AC V_AC预测 g⊙Δh E 或直接预测运动敏感的Dense输出变化。**不允许偷用教师e_prev锚点**，可选机制是无锚点的同预算AC基重建，或另行计费的稀疏真实W2锚点/可靠预测锚点+累积误差控制。明确anchor refresh、warp、遮挡、新生区域fallback、同步、跨tile通信与BF16原生两GEMM rounding。增加分支不等于节约总MAC；必须对比同实际计算预算的静态增秩/同rank重拟合。

**M3 / Motion-conditioned budget routing（先找收益，再选行动）**：
当匹配距离/时间创新能在**W2之前**提供稳定source-held行动收益且可在物理M tile汇总时，可用它决定静态rank、可选rank或其它精确合法单元，联合S4进行oracle-vs-cheap predictor regret/false-negative。W1之后只可能直接省W2；要省W1还需单独在V1之前可见的廉价指标与真实稀疏物理算子。最优后验Dense-error oracle仅作上界，不能替换部署选择器。

三个机制可以都落空，不因此默认恢复大规模补算。出现可靠的不同机制可新增路线，不把M1–M3当候选天花板。

## 5. 强竞争解释与能够改变结论的观测

| 机制/解释 | 明确不同预测 | 最小判别 |
| --- | --- | --- |
| H-A：低秩优先保留静态/DC，丢失AC | 同Dense h上，门控AC误差显著大于相同定义的DC误差或aligned AC gain下降；仅重新权重拟合即可改善 | 完整21τ相同source/step/block的真实Y/h/g＋shared/step因子BF16重放；比较DC/AC方向、误差及每source tails |
| H-B：低秩局部已足够，损伤来自student prefix分布迁移 | Dense-h上的AC保留尚好，但free-student h分布/Attention/调度流明显偏移 | 同一模型同source/seed同坐标teacher/student真实state，对前缀逐步替换的已注册有限干预 |
| H-C：图像纹理/画面变化影响RAFT而非动作 | RAFT下降，但主体事件、光流遮挡质量、人审相机/对象方向或其他动作指标不随之下降 | 20来源先做人审与ROI事件；额外motion/control sources做未暴露配对 |
| H-D：运动对应能剥离位置位移，留下低维创新 | 对齐后Δh和W2误差AC残差较固定xy明显减小，并且低成本匹配与物理成本有净收益 | 新空间邻域producer；matched same-source matched layer Δτ+邻域/随机/光流oracle |
| H-E：差异主要由gate、残差、typed BF16误差驱动 | pre-gate AC改善不对应postwrite u或完整视频收益；rounding扭转动作收益 | 原Y/g/r/u真实BF16输出、typed factor forward/postwrite和相同成本中性对照 |
| H-F：高τ相似性只是静态纹理 | 相似优势出现在W1输入，运动ROI与静态背景相同或静态更强；对齐后的动作收益不超过静态策略 | foreground/background/camera分层，V1 vs h/Y/gY/u，源等权非token等权 |

**数学排雷：** 每个21τ向量zτ，定义μ=meanτ zτ、Aτ=zτ−μ，则 Σ||zτ||²=21||μ||²+Σ||Aτ||²。对误差亦然。报告**相对DC/AC误差、signed cross term、AC aligned gain、方向余弦与近零分母缺失率**，不能只测 AC 能量比或SSE。时间均值和运动对齐残差是两种不同的分解；先完成固定xy对照，再选择是否投入高成本运动oracle。

## 6. 最短证据路径：先数据判别，再批准执行

**E0（已完成、不重复采集）**：20源60视频官方质量/RAFT，8源26389稀疏same-xy配对；已存在的S1同秩低秩因子与原始BF16 code/forward校验。

**E1（已有ACTIVE采集在途）**：先核实原BF16 10新来源21τ×8xy×全部160位置part0实际RUN/EXIT及中性回放；后继part1只有完整证明后由ACTIVE注册/启动。现有设计是**静态xy时间轨迹**，不是运动对齐轨迹。取得完整真实 h/Y/gate/r/u 及真实row identity后做固定xy DC/AC，按source计数。

**E2（推荐的第一个新判别，CPU/BF16数值重放，尚未授权执行）**：冻住真实来源、因子、gate、168行/packet的读写SHA，原生BF16两GEMM在完全相同Dense h上重放shared80与step160。对固定xy DC/AC和全层最严重source tails做逐source、step、block报告。加独立固定Dense full-M W2 control以校验M168 vs 原M31668的shape/bias/reduction差异时，另行计实际1600次原W2 GEMM，不冒充免费。与[原有DC/AC设计](../method_discussion/discoveries/E0/raw-bf16-r960-temporal-dc-ac-discrimination-20261029.md)复用同一scope，避免重复拟合/重复观测。

**E3（运动对应数据缺口，需新producer、现状OPEN）**：按E2显示的高风险layer/step/新source运动分层，采真实**同源同t同l的局部空间搜索邻域**、前后latent-τ坐标，以及可定义的flow/camera/遮挡oracle；同时冻结随机、同xy、邻域最近邻、光流oracle、cycle/occlusion控制，记录成本。当前固定8xy不是完整搜索空间。采集可以扩大到全160位和更多图像簇，来源不是总量上限。

**E4（表示与计算实验，后续OPEN）**：在固定rank960/相同预算下对比普通共享rank、step特化、AC加权表示、静态额外rank、运动匹配支路、源/层/step常数路由。所有α/λ/rank/邻域选择在拟合或独立development层确定；预留从未用于模型选择的确认来源和seed。新机制通过真实收益与硬件可执行性再申请4step×40block完整student配对视频与8轴+连续RAFT+动作/相机方向+时序/闪烁+公平warm SYS。评价单位是独立图像簇，不是τ、block、token或视频数。

**停止/扩展准则（构造性）：** 若同Dense-h AC没有比DC更大的损失，则优先捕获student偏移与下游消费者（H-B）；若E2有显著AC损失而E3不见对齐收益，优先M1无warp的同预算表示；若E3具有方向稳定且廉价的对齐收益，才优先M2/M3原生执行；若视频行为与RAFT代理矛盾，扩展动作分层与人工/语义验证，不用单一光流指标代替目标。

## 7. 原论文、最相近先例与差异（2026-10-29核查）

1. **TokenFlow**, Geyer et al., *TokenFlow: Consistent Diffusion Features for Consistent Video Editing* (2023), https://arxiv.org/abs/2307.10373 。视频编辑中用内部diffusion feature的跨帧对应传播特征；它建立对应传输的直接先例，不是Wan W2误差补偿的现成证明。
2. **FRESCO**, Yang et al., *FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation*, CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FRESCO_Spatial-Temporal_Correspondence_for_Zero-Shot_Video_Translation_CVPR_2024_paper.html 。帧内/帧间对应和显式特征更新；与本研究的运动对齐机制相近，但任务为视频翻译。
3. **FlowVid**, Liang et al., CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.html 。warp与不完美光流控制，为oracle的遮挡/错配风险提供先例。
4. **VidToMe**, Li et al., CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.html 。时序对应token合并，构成“按距离直接复用计算”的强近邻；本候选必须超越一般token matching。
5. **VMC**, Jeong et al., CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_VMC_Video_Motion_Customization_using_Temporal_Attention_Adaption_for_Text-to-Video_CVPR_2024_paper.html 。相邻noisy latent的残差表达运动，提示差分与运动有关；不等于原FFN W2误差。
6. **DiTFlow**, Pondaven et al., CVPR 2025, https://openaccess.thecvf.com/content/CVPR2025/html/Pondaven_Video_Motion_Transfer_with_Diffusion_Transformers_CVPR_2025_paper.html 。从跨帧attention提取patch-wise AMF用于运动转移，证明可在DiT内部构造运动线索；该信息不自动免费可见于FFN。
7. **AdaCache**, Kahatapitiya et al., ICCV 2025, https://openaccess.thecvf.com/content/ICCV2025/html/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.html ，和 **FasterCache**, Lyu et al., ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/518046d86bbc41a0707727c38301ad8e-Abstract-Conference.html ：主要利用**去噪step缓存和运动条件策略**，不是直接在固定step内部对21个latent视频时间位置做FFN输出误差分解。
8. **SVD-Cache**, Chen et al., CVPR 2026, https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html 。主子空间预测/残差复用，强先验是异质时间行为；其“time”主要是DiT迭代去噪过程，与本候选需要精确区分。
9. **OmniCache**, He et al., arXiv预印本2026, https://arxiv.org/abs/2607.23844 。已将帧内、帧间、运动与去噪步等多维相似缓存放在一个框架，**最接近“利用时空距离避免重复计算”的先例之一**。不能把最近邻匹配、运动warp、源帧复用本身称为本研究的新贡献；必须给出独立的FFN W2误差运输/创新表示及同预算动态保真证据。

**创新判别：** 将**固定t、l、原始W2函数差E**分解为可识别的时间创新误差，并在未执行Dense W2、保留原W1/GELU/gate写回的严格在线信息与计算预算下，证明一个实现能比同实际MAC/延迟的shared/step rank与现有对应缓存保留更多完整生成运动。任何论文新颖性结论以进一步作者源码对照和实际学生视频/硬件结果为准。

## 8. 记录状态与协作

- ID：**MTC-TDIC-20261029**；科学标签ALG/SRC/LIT/OBS(existing)/OPEN。S2的可组合时空探索，不替换S1/S4/S3，不把“补算”设置为目标。
- 图锚点：V1输入(F06)→GELU后h(F08/F09)→W2(F09)→门控及写回(F12/F13)→视频消费者(G20)；以图谱登记OPEN假说，不增加已实测因果边。
- 讨论队列：**探索种子**，不是独立Agent正式proposal或SUPPORT票。实际多Agent另行提交不同方案/独立review；实验必须按ACTIVE注册并有真实RUN/EXIT，不能将此文当作启动命令。
