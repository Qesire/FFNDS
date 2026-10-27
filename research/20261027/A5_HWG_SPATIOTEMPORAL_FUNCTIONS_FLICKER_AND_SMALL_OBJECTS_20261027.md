# A5+HWG的背景闪烁与小物体模糊：视频FFN的时空功能、条件激活与低成本计算

**2026-10-27｜AutoResearcher机制研究、问题重构、文献核验｜SRC/HISTORY/VIDEO-OBS/ALG/OPEN**  
**研究目标**：从已观察的A5+HWG视频异常，发展对正常Wan2.2四步视频生成FFN的可检验认识，并帮助UB1、Affine54和RT-Lynx激活2:4形成更强的计算—质量研究路线。**此文件不是新视频/模型实验报告**。它与[RT-Lynx一手研究](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)、[当前完整质量合同](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)和[研究导航](../RESEARCH_NAVIGATION.md)互链。真实RUN身份、已注册执行范围均不改变。

## 0. 新认识：视频FFN的重要性不能只以「一次输出有多大」定义

**观察**：历史A5+HWG视频中，主体的大结构与主要动作可以保持，但背景亮度/纹理出现不稳定，小物体或细节有模糊与变化。**这是历史视觉描述，不是现有仓库中已完成逐帧光流对齐、语义背景ROI/小物体实例量化的结果**。与之后独立候选Affine54的视频运动变少、细节模糊，以及UB1两来源的连续运动差相互呼应，但**不能将不同方法的现象、机制和视频身份合并成同一个实验**。

从中提出三个可研究的正常生成功能（**不是已鉴定的单神经元语义类别**）：

1. **结构持续**：保持物体/背景的大形态、位置、构图和主要动作；
2. **细节存在**：让小物体、边缘、微纹理和低对比度结构在当前画面可辨认；
3. **时序协调**：同一物体/背景材质、照明及遮挡沿视频时间平滑而可信地演化。

它们可能由重叠的FFN+attention/normalization/decoder路径承担。重要的新问题是：**保住低阶/宏观视频结构所需的计算，与保住细节及跨帧函数协调所需的计算，是否在信息价格、动作粒度或物理支持方面不同？** 这可以解释为何同一稀疏/折叠原型在粗略视觉或VBench单维接近Dense时，细节、背景、动态仍然改变。

## 1. 重建A5+HWG的真实定义与历史身份

### 1.1 当前可核的核心计算

A5为基于HiF4量化W1符号/分组尺度的行系数\(\beta_j\)的preactivation代理（形式可记为）
\[
\widehat a_{rj}=b_{1j}+\beta_j\sum_g a_{jg}
\sum_{i\in G_g}\operatorname{sign}(W^{\mathrm{eff}}_{1,ji})\,z_{ri}.
\]
HWG在给定step×block×expert/gate作用范围中为可预先计算的W2 gated列范数
\[
c_j=\|g\odot W_{2,:,j}\|_2,\qquad s_{rj}=|\operatorname{GELU}(\widehat a_{rj})|c_j .
\]
阈值/预算得出每个token×hidden-channel的mask \(m_{rj}\in\{0,1\}\)；**A5代理用于支持决策，原有效W1/GELU/Q2/W2承担被保留值的计算**。但是**历史A5完整视频探索执行中W1已完整计算、后续hidden才mask**，因此其约44.3%逻辑零值绝不等于W1或整体FFN省算44.3%；当年特定配置没有得到必要的全系统稀疏加速证据。

### 1.2 历史变体身份要分别处理

- 早期A5探索的\`s1=.7,s2=.436\`在历史formal20的Dense与A5中，aesthetic **0.633222991→0.618013695（相对下降2.401886%）**、imaging **0.735781345→0.733149200（下降0.357735%）**、motion **0.985179632→0.987165826（相对上升0.201607%）**；三指标不满足历史逐项≤2%的全部门槛。**此报告中的标签为A5实验配置，不强行认定它就是后来固定A5+HWG/τ的完全一致执行版本**。[原仓库边界](../../docs/04_前期探索与证据边界.md)。
- 冻结细粒度版本定义为A5→GELU→HWG→\(\tau\) threshold，而某旧V1.1试运行实际使用parent TopK；threshold与TopK不可因方法族相同就重标为同一已验收实验。历史mask/score/阈值位一致性必须回溯原manifest才可作新的原生因果比较。
- 背景闪烁/小物体模糊是历史A5+HWG有关的视频观察，原私人执行视频完整文件、两方法逐帧配对和精确ROI评分**未在当前Git树直接提供为可核全帧资产**。因此此处以HISTORICAL-VIDEO-OBS登记，作为生成科学问题，不用想象中的逐帧统计描述为新实测。
- 后续[Affine54六视频](../20261026/E0_RECONSTRUCTED_VIDEOS.md)有来自两个已暴露来源的真实全81帧、局部视觉与后验运动数据；[UB1原本三维分数与独立运动复核](UB1_DYNAMIC_DEGREE_RESULTS.md)是其它方法/来源，帮助评估同一种质量定义的局限，不用于代替A5单独对照。

## 2. 时空函数两轴：闪烁和模糊不能都叫「高频损失」

视频的不同质量现象至少有空间频率\(k_s\)和视频时间变化\(k_t\)两轴（分析前需作相机/主体运动对齐，未对齐的真实运动会进入时间高频）：

| | 空间低频或大尺度 | 空间高频或微结构 |
| --- | --- | --- |
| **视频时间低频/持续** | 整体构图、稳定背景亮度与主体位置；过度静态也可能是质量问题 | 持续存在的小物体、细边缘、稳定微纹理；**模糊常首先在这里发生** |
| **视频时间高频/不规则** | 背景照明闪白、色彩/曝光抖动、非物理全局变化 | 纹理闪烁、边缘抖动、遮挡边界抖动及小目标突然出现/消失 |

**推论A**：一个模型可以保持空间低频主体结构，却损失空间高频物体；这不需要mask本身逐帧变化（固定支持也可能删除长期弱细节）。  
**推论B**：背景可能在空间低频上高时间频率闪烁，而小物体在空间高频上长期模糊；两者不是同一简单频带病灶。  
**推论C**：小物体的“模糊”与“跨帧位置/身份漂移”还需分离。若轨迹偏移但清晰，逐帧Dense对齐的RGB误差也会显得高频很大；若对象在所有帧稳定模糊，则时间闪烁可以很小。

现存[完整latent频带观察](../20261026/E0_SAVED_FLOW_SPECTRUM.md)证明有不同子带、有符号交互；但21个latent-time切片、周期FFT和81帧RGB不一一等价，不能据该谱宣布A5闪烁来自某个FFT子带。

## 3. 一个可证伪的代数构造：动态支持切换对时间函数的扰动

**定义的条件**：固定同一去噪step \(k\)、FFN层\(\ell\)、真实post-Q2操作数与gate，token \(r\)在**latent视频时间/空间**上的对应点为\(r'\)（可来自Dense辅助运动场的warp；不是把去噪step当视频帧）。只有在post-Q2裁剪、不重新Q2的明确动作下，若 \(c_{rj}=g_r\odot W_{2,:,j}q_{rj}\)，候选省略的当前门控输出向量为
\[
E_r=\sum_j(1-m_{rj})\,c_{rj}.
\]
对两个通过合法运动匹配的token \(r,r'\)，严格代数上
\[
E_{r'}-E_r =
\underbrace{\sum_j(m_{rj}-m_{r'j})c_{r'j}}_{\text{同值参照下的支持切换项 }E_{\rm switch}}+
\underbrace{\sum_j(1-m_{rj})(c_{r'j}-c_{rj})}_{\text{原函数/条件变化项 }E_{\rm signal}}.
\]
这一式子是**同一后Q2输出坐标的线性分解**。如果实际操作在Q2之前、gate变化、BF16 cast或完整后缀，必须在真实图边重新测有限差；不能从它直接得出每帧闪烁或严格因果。若有运动warp插值，额外记录warp/遮挡及空间插值误差。

**可证伪的新预测**：
- **H-SWITCH**：当\(|s_{rj}-\tau_j|\)在阈值附近，少量连续激活变化即可使\(m\)跳变；在低运动背景的匹配token上，mask时间total variation及\(E_{\rm switch}\)对新增flicker有强预测，而同预算固定/滞后/连续软支持可以降低fluctuation。
- **H-PERSIST**：即使mask持续不变，某些幅值小但跨token/帧联合相干的函数方向被稳定丢弃，会让小物体或运动细节长期消失；此时平滑mask不恢复该功能，应改变权重敏感/联合功能支持或恢复计算。
- **H-QNUM**：对数值分组位于preQ2的动作，共享scale使未动元素也改变，Q2/BF16连带可能加剧时序非光滑；对于A5历史post-Q2 mask不能把preQ2连带当既成原因。[原M Q2合法直接/补集观测](E0_Q2_DIRECT_COLLATERAL_FULL_SHAPE_RESULTS.md)只说明该算子的实际机制可能存在，非A5原生视频因果。
- **H-PROP**：W2 gate输出很小的误差经后续self-attention、跨帧token关联、流场step更新或VAE改变成背景/小物体可见偏差；需要同前缀局部干预与下游观察，不能由最后视频单独确定是哪层负责。
- **H-CONTROL**：原视频种子/运动、相机运动、遮挡、VAE解码及编码差异解释部分观察；若同identity Dense/候选配对且运动补偿后异常消失，则不会继续归因A5 mask。

代数允许存在**小范数但大时序变化**的反例：\(E_r=\varepsilon v\)、\(E_{r'}=-\varepsilon v\)，各帧仅\(\varepsilon\)误差，差分却为\(2\varepsilon v\)；相反固定\(E_r=\varepsilon v\)在所有帧有同样局部误差而时序差为0。高时间差≠必然可见视频闪烁，需真实后缀/解码敏感性；却证明**只看各帧误差范数不会确定帧间误差函数**。

## 4. 与RT-Lynx、UB1、Affine54的实质连接

| 历史/新方法 | 计算保留或删去的对象 | 对时空功能的研究预测 |
| --- | --- | --- |
| **A5+HWG fine mask** | 每次token×hidden的后验/预测支持，历史W1计算已支付 | 对动态阈值变化敏感；支持切换可能在真实时空correspondence上改变函数方向 |
| **RT-Lynx输入2:4** | 在W1或W2 GEMM输入四元组只算Top2、原稠密权重保留 | weight长期罕见能力可保存，但同一路径的弱细节和时序一致性仍可能受当前2:4硬选择损伤；norm/LoRA与不稀疏的敏感层提供成熟恢复 |
| **UB1 64组静态支持** | 按step预定删除整个W1/W2隐藏子网络 | 没有逐token硬切换也能损失稀有小目标函数及不同来源的运动幅度，故H-SWITCH不是唯一原因 |
| **Affine54固定仿射＋54组真非线性** | 更低成本的固定函数类，不依赖每token动态mask | 在缺少mask切换时仍能有视频模糊、主体运动减少或额外背景变化，说明容量/签名/跨步作用亦重要 |
| **新C线2:4原生kernel** | 指定operand本次乘法，保持dense权重 | 可提出视频时空功能与实际物理成本Pareto的独立项目；原HiF4 whole64空tile与SpTC 2:4逻辑不同 |

**根本研究改写**：FFN中“当前激活很小”“长期罕见功能”“当前能少算”“不会破坏时空协调”是四个不同谓词。单一\(s_j=|\hat h_j|\|gW2_j\|\)只处理当前输出贡献的对角替代，无法直接证明时序保真、保留长期功能或真实计算收益。这个发现能同时解释为何视频外观近似与完整生成行为分离，并不要求已有方法都是错误的。

## 4A. 新增UB1运动衰减：视频时间误差至少有两个相反方向

[UB1正式运动衰减复核](UB1_MOTION_ATTENUATION_APPEARANCE_DYNAMICS_CAUSAL_AUDIT_20261027.md)进一步提升本研究：两个已暴露formal20源的UB1/ Dense连续RAFT top5幅值比为**38.13%（犀牛，38/40对较低）**和**9.19%（住宅，40/40对较低）**，其中犀牛视觉观察到头/后腿动作减少。原版二值动态分数四条全为0，Motion Smoothness历史正式20均值约0.9839并没有暴露这种差异。**UB1固定group支持，并无A5类逐token阈值翻转**，所以该观察独立支持：正常生成时序函数改变不要求在线mask切换。

时序至少存在两种截然不同的行为：**过度抑制变化（attenuation/freeze）**与**额外非物理变化（flicker/jitter）**。在固定张量边界上用线性latent时间差分\(D_\tau\)和\(e=Y_S-Y_D\)，动态能量变化为
\[
\|D_\tau Y_S\|^2-\|D_\tau Y_D\|^2
=2\langle D_\tau Y_D,D_\tau e\rangle+\|D_\tau e\|^2.
\]
对真实变化负向的压缩扰动可能产生attenuation；高时间频率不相关的扰动可能产生flicker；两者可以同时存在，也可能被真实attention/去噪后缀重组。**这是数学类别，不是用RAFT幅值推算出了真实signed cross**。A5闪烁历史描述没有独立原始ROI分析；UB1的RAF​T衰减可能因模糊导致可观测性下降，同样仍要单独检验。不能把这两个不同候选/视频的结果当作共享原因已定位。

新最省判别：对原视频先区分动作要求（grazing vs houses）、相机/主体方向与sharpness/RAFT置信；然后分别研究\`mask switching→flicker\`和\`固定支持/有符号时序函数→attenuation\`。VideoJAM(ICML2025)对静态外观与真实运动目标的区分、Motive(ICML2026)的motion-weighted训练梯度都已提供强先例；在FFNDS首先可以作为**评估/来源选择/轻后训练的成熟方案**，不刻意要求不同。


## 5. 从旧资产可开展的最小判别：优先区分现象

**S0｜身份核对（现有源码与历史资料）**  
追溯A5视频究竟来自TopK历史pilot、A5 s1/s2版本还是已冻结A5+HWG+τ，记录input/prompt/source/seed、frame/FPS、与Dense paired SHA、真实mask插入点及Q2方式。若只剩早期视觉报告、视频本体/每token mask未保存，则明确不做重新计算的“新定量发现”。下一阶段不把同方法族实验写成同一冻结结果。

**S1｜视频级二维质量分解（存在原MP4时条件执行）**  
原帧分别采集：camera/global motion对齐背景ROI、主体和小目标ROI、遮挡mask/置信度；参考Dense运动与candidate各自估计都报告，以免把真实运动错误抹除。分别记录(i)背景运动补偿luma/颜色帧间残差及其时间差分；(ii)小物体的局部相位/edge sharpness、对象保留率与identity轨迹；(iii)全局subject/background语义一致、motion smoothness、dynamic degree连续光流而非只使用二值结果。先使用已有[逐帧指标合同](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)和[既存six-video回填](../20261026/E0_EXISTING_VIDEO_QUALITY_BACKFILL.md)，语义ROI若无可靠来源则作为人工review或独立语义标注，不能把低运动像素直接标为背景。

**S2｜时空mask分析（只有合法同帧token映射与原mask载荷时）**  
将\`denoise_step k\`与\`latent temporal slot n\`分开，按同一模型step和block/专家、同空间token的真实运动correspondence计算：每组mask Jaccard / xor-rate / temporal TV、阈值margin\(|s-\tau|\)、被删W2/gate signed贡献、时空共同支持集中度和组/SpTC 2:4合法work。与Dense后Q2真实输出对照：soft/fixed mask **必须密度、真实执行成本匹配**。若flicker与mask切换无关或相干弱贡献更强，则降低H-SWITCH并研究H-PERSIST、H-PROP。

**S3｜合法因果判别（以后单独注册）**  
在冻结真实step/layer、相同prefix和gate、合法Q2闭包下，比较：原动态mask；时间一致但相等token预算的mask；固定支持/静态优秀基线；只恢复低运动背景ROI或已定义小目标有关函数方向的局部原Dense计算（恢复的观察/路由成本计价）。测**same-input postgate、nextnorm、完整后缀/flow、终端视频**多级结果；小规模单步可证机制，完整视频用于质量与性能验证。ROI可解释动作须避免依赖Dense未来视频在推理时获得“免费oracle”。

**S4｜若显现可稳定复用的函数**  
数学/系统上研究(a)共享时空支持或滞后路由，(b)跨帧可重用的有符号背景功能，(c)稀有小物体功能的轻量恢复，(d)2:4输入算子及量化码表示；直接复用RT-Lynx norm/LoRA、TARDIS条件异常回退、CTEC跨步纠错也完全合理。先证明信息什么时候可得、选法是否有物理收益；没有必要刻意创造与论文不同的loss。

## 6. 论文近邻：从哪项成熟研究吸收什么？

- **[VBench CVPR2024原论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)、[官方代码](https://github.com/Vchitect/VBench)**：将Temporal Flickering、Subject/Background Consistency、Motion Smoothness、Dynamic Degree、Imaging Quality分开；temporal flicker官方优先用筛选的静态视频，不能把背景ROI自定义度量冒充官方VBench。原算法Dynamic Degree二值可掩盖连续运动差，仓库[UB1两来源原版RAFT复核](UB1_DYNAMIC_DEGREE_RESULTS.md)提供实测实例。
- **[FADE, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_FADE_Frequency-Aware_Diffusion_Model_Factorization_for_Video_Editing_CVPR_2025_paper.html)**：视频模型不同注意力构件承担不同频域先验、用频率引导保持视频时空结构；支持按真实attention/step区分机制，**这是视频编辑先例而不是Wan FFN删算验证**。
- **[STDD, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yao_STDD_Spatio-Temporal_Dual_Diffusion_for_Video_Generation_CVPR_2025_paper.html)**：明确建模时间序列噪声/信息与空间联合传播；启发“时间功能不等于逐帧静态功能”。没有对当前FFN神经元做因果分类。
- **[FloVD, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jin_FloVD_Optical_Flow_Meets_Video_Diffusion_Model_for_Enhanced_Camera-Controlled_Video_Synthesis_CVPR_2025_paper.html)**：将背景光流及相机/对象运动分开；为背景对齐评价提供成熟直觉，不能视作Wan新的观测。
- **[SAGA, ACCV2026](https://arxiv.org/abs/2607.08020)**：用latent acceleration频谱控制自回归视频生成的temporal jitter；可借鉴时间二阶差分诊断，**其chunk-wise autoregressive因果累积机制与这里四步同步I2V不同**，不迁移其训练/采样结论。
- **[RT-Lynx, ICML2026](https://proceedings.mlr.press/v306/cong26a.html)**：区分长期dense weight与当前activation 2:4，norm/LoRA能提升质量，但输入范数保持≠有符号时空函数保持；详见[专题审查](RT_LYNX_ACTIVATION_SPARSITY_SUPERPOSITION_AND_WAN_EXECUTION_AUDIT_20261027.md)。

论文是机制工具，不是必须“另辟算法”的限制。凡是成熟方法在原Wan真正有效，直接复现/迁移就是建设性结果。

## 7. 最终可接受的研究结论与图谱对象

- **历史事实（HISTORY/VIDEO-OBS）**：A5系列存在背景/细节时序异常的早期观察，A5对应formal20方案至少Aesthetic单项未过旧门槛，早期细粒度mask未证明真实省算；现存仓库并未增加A5完整时空逐帧量测。
- **严格数学（ALG）**：支持切换项与被删函数变化项可以在post-Q2线性输出边界分解；即使相同单帧误差范数也可能具有相反的时间差表现。
- **跨模型研究问题（OPEN）**：正常Wan FFN是否存在保护小物体、背景照明和视频时序协调的可再现函数方向？从当前激活/权重/码本/历史状态能否低成本识别它们？是否能改用更合适的物理单元执行？
- **对质量协议的影响**：增加跨帧运动对齐背景亮度与纹理、连续光流/动作量、小物体ROI/身份持续三个**互不代替的观察通道**；缺视频或合法ROI时标NOT_RUN/OPEN，不以局部SSE或VBench Motion Smoothness代替。
- **对可部署方法的影响**：继续用UB1/Resident Dense/Affine54强基线，允许RT-Lynx 2:4、稳定动态支持、函数折叠、跨步训练各自作为独立候选。研究I/F/C仍然平行，G13用于F线因果定位而不是全部算法的前置门槛。

**Skill复核顺序**：NOTICE（主体基本稳定但背景/小物体异常）→ MAP_FLOW（A5→HWG→mask、Wan Q1/W1/GELU/Q2/W2/gate与latent-time轴）→ DERIVE（支持切换/静态遗失/时空双频）→ EXPAND（背景协调函数、细节维持函数、2:4另种执行单位）→ DISCRIMINATE（同源配对、真实motion warp与mask反事实）→ RECORD（保留历史实验身份、不要伪造新视频）。只改研究文档；是否需新实际模型实验由[ACTIVE](../ACTIVE_RESEARCH.json)及明确注册范围决定。
