# S2 独立理论、竞争机制与实验设计

2026-10-29 · author `/root/s2_theory_design_independent` · 独立研究，未读取协调者新误差拆分实现或审阅；无模型、GPU或大张量分析。本文是 ALG/SRC/LIT 与既有 OBS 的科学建议，不是同行投票、运行授权或新增运行结果。Pilot 已由上游在本委派到达之前冻结执行，本文不是其事前审签。

## Original graph / ALG

[SRC] 针对现有原BF16 LightX2V normal forward，[固定源码 infer_block/infer_ffn/post_process](https://github.com/ModelTC/LightX2V/blob/ed146db64d0b6130c669285fc1aa0b280a6a4823/lightx2v/models/networks/wan/infer/transformer_infer.py) 与本地 [bf16_observer.py](../../../../tools/bf16_observer.py) 相互核对：time embedding 与 block modulation 合成六组 shift/scale/gate；self-attention 经 gate 写回，cross-attention 读 norm3/context，其输出 C 首先 in-place 加到 self-attention 后 P 得 R；V1=modulate(norm2(R), shift, scale)，Z=V1 W1+b1，H=GELU-tanh(Z)，Y=H W2+b2，U=BF16_add(R, BF16_mul(Y,g))，然后下一 block 的 norm1/self-attention；末层则进入 head、flow/scheduler 及后续去噪步。源码另有 smooth norm 与敏感精度分支，因此等式只描述目标原BF16 recipe，具体算子 cast/fusion 必须由记录及 native byte check确认。W1 逻辑 5120×13824、W2 runtime 13824×5120，bias各一次，V1/R/Y/U 5120、H 13824。21×29×52=31668 token；τ是视频latent时间，非denoise step。原BF16无激活FP4 Q1/Q2。

[SRC] 四步 high/high/low/low 为两套不同完整专家参数；同一 step/block、source、seed、recipe、grid及真实row才能把 Dense 与 student 当成对应控制，哪怕语义由于自由轨迹发散不再对应。不同 denoise step 的相同 τ/xy 是相同网格地址而非同一状态；跨 high→low 不得套同一 W2、gate 或直接减作同函数误差。坐标匹配不自动是运动物体匹配。当前两个学生只替代 W2，真实 h_S 从学生完整前缀计算，不能用 h_D 冒充。

## OBS: 已有数据说什么

[原21τ统计](../../../20261029/S2_FULL21TAU_TEMPORAL_STATISTICS_RESULTS_20261029.md)：十图1600包、τ1–20，gY lag1 0.0483、平均 r95 16.961/19；去τ0前后的差分峰均比3.321/1.800。中心化和差分自身可使相邻差分cos约−1/2；不能凭它推断往返运动。[同Dense-h factor复算](../../../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md) shared-r960 gate后 DC/AC相对SSE 0.029321/0.184154，原W2 M168舍入AC 0.00005245；这是同状态与同原bias/gate的局部证据，旧shared因子不能与后来新10拟合混同。原残差写回 rowLN(u) AC能量比1.006但方向误差0.0441，证明残差可掩盖幅度变化，不能直接推出视频恢复。

[新10拟合](../../../20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_FIT_RESULTS_20261029.md)：value10→dcac10训练AC相对SSE 0.051229→0.047267，DC 0.001085→0.001775，τ0 0.007648→0.010980；[20开发源视频](../../../20261029/RAW_BF16_TEMPORAL_VALUE_DCAC_VIDEO20_QUALITY_RESULTS_20261029.md) source等权连续RAFT幅度相对Dense 0.450623/0.439311，属于不同数据与自由轨迹、不同指标，不能解释成同一量从0.051直接映射至0.45。训练方向改善而视频没有改善可能因为源/ROI分布、被牺牲的τ0/DC、误差方向/交叉、student前缀或下游敏感性中的任意组合；现存数据未量化各因果份额。RAFT含相机/纹理估计效应，需运动方向、遮挡与人工/语义对照。

## 竞争机制及反预测（OPEN）

| 机制 | 支持与相反的可观测预测 | 混淆和最低对照 |
|---|---|---|
| H1 源/区域及被牺牲方向 | 固定h_D，新的未拟合对象ROI或τ0上dcac10相对value10 的门后有符号目标误差恶化，且差异在同h_S上仍在；反之各区和source上统一改善则削弱H1 | 原同一校准ROI的误差不能代表外源；同source配对两因子、分τ0/DC/AC、source-held、等秩及固定g/r，报告尾部而非挑一site |
| H2 student前缀/后缀放大 | L很小或在两候选间收益同向，但P、I或后缀有限响应差异大且自由状态恢复明显；相反若固定状态局部L已解释跨源退化而恢复student前缀后响应变化小则削弱 | L/P/I不是因果归因，P包含上游所有差异；选相同局部注入范数、随机方向和匹配位置对照，做真实后缀干预 |
| H3 运动对应低维 | 在先冻结且独立估计的可靠对应上，holdout残差/预测误差较same-xy和随机位移显著更小，遮挡区差异消失或反向；相反无改善或仅训练匹配上改善则削弱 | 使用Dense输出事后flow是oracle，不是在线信号；噪声步、遮挡、纹理重复和邻域边界会造假；两个方向一致/遮挡mask、随机位移及跨τ打乱、同搜索容量和heldout成对测试 |
| H4 gate/残差与量化边界 | Y误差差异被g投影缩小而U的实际误差和方向由残差主导；反之 gate后及U方向差异保留并可致大后缀响应则弱化纯数值解释 | 固定r/g对比自由r/g、同M原W2及真实乘加舍入；r掩盖能量不证明无因果作用 |

若H3失败，更强的替代表示是假设**可预测的是条件响应方向而非可搬运token**：源 held-out 的同状态误差对有限后缀响应有稳定低维读出，而实际轨迹对齐无收益。对每个step/block用独立source估计有限响应敏感方向或随机方向控制，比较同预算面向输出影响的测量；不能将后验真实后缀或Dense输出当免费在线路由器。

## L/P/I 与真实数值修正

[ALG] 固定一个expert/site、输出坐标、同函数定义及同bias，D(h)=hW+b，Q(h)=hA+b，A是候选完整有效矩阵。令 Δh=h_S−h_D，E=A−W：

L=Q(h_D)−D(h_D)=h_D E；P=D(h_S)−D(h_D)=Δh W；I=[Q(h_S)−D(h_S)]−[Q(h_D)−D(h_D)]=Δh E；于是 Q(h_S)−D(h_D)=L+P+I。这个分解是以D为锚的路径选择，不是唯一的Shapley或因果归因；P和I可强烈抵消。标量反例 W=1,A=0,h_D=1,h_S=2，则 L=−1,P=+1,I=−1，平方误差无法分摊成三个平方之和，交叉项必不可少。若两函数bias不同，L增加Δb、I中Δb抵消；不要省略真实b。g相同且逐元素固定时门后差是 g⊙(L+P+I)。不同门与残差：连续真实写回差 = (r_S−r_D)+g_D⊙(L+P+I)+(g_S−g_D)⊙Q(h_S)；也可用 g_S 锚但要改变额外项，记清锚选择。不同shift/scale早已改变h_S，条件门差必须另计，不能再将其全算给P。

对每个臂定义 ε_j=U_j−[r_j+g_j⊙Y_j]，采用精确表示已记录BF16数值的实数减法，则真实U差等于上式加 ε_S−ε_D。ε包括该实际乘、加、cast及kernel路径的全部差；L/P/I可以直接对实值连续函数计算，或定义为真实typed reader的四角差，此时恒等式仍逐元素成立但 I 包含GEMM非线性舍入，不能称纯线性交互。分开记录原大M、采样小M及两个factor GEMM的类型、bias、舍入；只读captured Y 不能得到 D(h_S) 或 Q(h_D)，需另付原W2和候选读出计算，选定四角、张量顺序和硬件并计账；native Y/byte neutral作真实锚，误差补项由实际值显式测出。不同 arm 的每个gate/shift/scale须采验，即使预期同值也先逐字节验证。

指标：按源/step/block/ROI分层写 L、P、I 向量范数与成对 signed cross 2⟨L,P⟩等，和其相对总误差；τ0单列，τ1–20 FFT低中高频同时给复交叉谱、coherence及相位，仅在能量大于预定义阈值的频带报告相位；多源配对方向与source簇bootstrap，注明短T与非周期边界/窗泄漏。先算Δτ误差与空间lag、冻结方向的配对cos与遮挡可见率；用同规则的原W2 M形状对照和τ置换控制，避免中心化造成的假负相关。仅比较频带能量会丢失预测方向、相消和频间耦合。

## 可执行实验与预算（建议，尚未执行）

1. **已冻结链路pilot**：898/96是先前已曝光Dense视频按高subject/camera RAFT挑选；5×5最大Farneback ROI可能是相机/背景，鸟图确落右下植物；中央5×5是预注册空间锚，不等于主体mask。两远距低流是候选负对照，可能仍含运动和遮挡。three arms、160 sites、21τ、去重≤52xy、真实normal forward。确认每source×arm真实row、h/Y/r/u、g/shift/scale、step latent、parent checkpoint/seed、原始bytes与观察无扰；已有10状态byte-neutral且0 decode只支持该身份，不是新视频成功或因果结果。对选定相位误差/空间对应，额外需要明确匹配支持边界与mask；缺关键attention消费者时标NOT_MEASURED，不用rowLN冒充。
2. **轻量先验判别（新CPU范围）**：仅从已捕获固定数据计算两个学生 h/V1/r/u 及其对Dense同row的差、τ0/频谱、相邻位移可见范围；若oracle位置来自未来视频，注明并用预注册非流动/随机lag控制。不重跑模型、无新拟合；无法生成缺失四角Y。输出预定每source×step×block，分开两个独立来源，不计算虚假的token置信区间；失败或缺张量即退出，另注册采集。
3. **四角离线reader（另冻结）**：对每arm的实际 h_D/h_S、匹配expert/site，用原W2与固定value/dcac因子各算对应四角，保留bias，记录采样M及full-M的原生舍入和typed postwrite。对每site compare L/P/I及差值、交叉频谱、gate/r/round余项；无原W2 checkpoint或header/SHA/shape不合立即退出。52×21×160=174720行/源/arm；原W2约 70,778,880 MAC/行，即约12.37 TMAC/源/arm/单次读出，另加因子两GEMM、两臂控制、IO及对应权重读取，具体计数按去重实际行和复用重算；离线付费oracle并非在线成本。
4. **来源扩容（另注册）**：至少12个区别于pilot的独立图像簇，预分对象运动、相机平移、静镜局部动作、背景流、遮挡/突发，允许一图多标签但聚类单位是图像。图像SHA、训练历史与seed独立；已有20视频是开发暴露，真正确认须新source及独立seed。先低成本标定subject/object box、camera全局流、前后向一致性、遮挡及高/低流、特征匹配可见性，再固定坐标，新增高运动主体/遮挡和低流负对照；报告按图簇等权及分层coverage，不将多τ/多site当独立图。最终以精度和尾部持续扩充，12非上限。
5. **后缀因果（另注册）**：预先基于跨源、两臂共同规则挑选高/低响应sites并在新源复核，保持同source/seed完整调度；以原postwrite可控地注入或移除固定L方向，含零注入、反号、等范数随机和同坐标原生复原，后缀全重跑及成对latent/flow/video。一次只改变清晰边界；替换整个student prefix是另一干预，不保证产生自然状态。若对称±响应非线性、gate drift或随机控制同样有效，不能宣称特定方向因果；需要新范围完整视频语义/动作/相机/时间质量和真实FFN/全系统时间才可谈可部署性。

数据量：V1+H+Y+R+U每row 2×(4×5120+13824)=68608 bytes；最多52×21×160行约11.99 GB十进制/source/arm，2源×3约71.9 GB，不含g、metadata、latent、额外attn及临时buffer。预算按真实去重、读写峰值和已有免费空间再冻结；GPU session上限2，并非本文授权。全160/21τ 保留；同ROI选择通过前不可推断主体代表性或在线运动成本。

## 最近原论文与作者源码

- [TokenFlow 原论文](https://arxiv.org/abs/2307.10373)及[作者实现 tokenflow_utils.py](https://github.com/omerbt/TokenFlow/blob/master/tokenflow_utils.py)：基于输入视频已提取特征的帧间对应传递编辑feature、pivotal frames；其源码有pivotal pass和跨帧attention等机制。是视频编辑框架，既有整段输入及后验对应；非W2近似误差，也不证明生成时在线match免费。
- [FlowVid CVPR论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.pdf) 使用首帧warp、遮挡与空间条件，训练视频编辑模型，将不完美flow作为辅助；[作者仓库](https://github.com/Jeff-LiangF/FlowVid) 本次目录仅README，无法核实现。该论文提醒遮挡而非证明Wan latent对应。
- [AdaCache ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Kahatapitiya_Adaptive_Caching_for_Faster_Video_Generation_with_Diffusion_Transformers_ICCV_2025_paper.pdf) 与[作者 STDiT3 实现](https://github.com/AdaCache-DiT/AdaCache/blob/main/opensora_base/opensora/models/stdit/stdit3.py) `compute_next_step`：相邻计算残差的归一化距离、基于时间相隔token差范数的MoReg和阈值codebook确定下一次刷新；是 denoise-step feature cache，不是 21τ运动位移查找或W2因子误差运输。首尾刷新与状态成本须计。
- [SVD-Cache 原论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Forecast_the_Principal_Stabilize_the_Residual_Subspace-Aware_Feature_Caching_for_CVPR_2026_paper.html) Eq4–11：离线feature SVD、在线刷新投影，principal EMA、残差复用；[作者库](https://github.com/BlackMaple1203/SVDCache) 本次仅 LICENSE，无法核可执行代码；见既有 L39/G14和[G8公式记录](../../../literature/20261027/G8_FOUR_METHOD_PRIMARY_GRAPH_AUDIT.md)。论文的t是跨采样step的feature，非本研究的latent τ。
- [OmniCache ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Chu_OmniCache_A_Trajectory-Oriented_Global_Perspective_on_Training-Free_Cache_Reuse_for_ICCV_2025_paper.pdf) Eq6–8 定义缓存预测与正常denoiser的输出差，并以跨step相关/轨迹曲率近似及滤波修正。四步双专家不可自动继承论文的长轨迹/缓存能力；本次未确认作者可执行仓库，不能标CODE_CHECKED。其缓存误差和本同输入W2 L/P/I不等价。以上作为既有文献的局部核对，未新增G14节点或宣称文献图已更新。

## 建议与开放分歧

优先完成已冻结pilot的观测完整性，再用另登记四角reader区分同状态误差与学生状态交互；尤其不要因流对齐效果正面就称主体对应或因果证明。若空间支持不足，明确返回边界不可辨并在后续来源采集对象/遮挡；最低成本替代是先用已捕获同坐标的有符号误差、τ0和学生轨迹差锁定方向，再确定是否付费四角。需要全M原W2参考校正：小M差异可被测量抵消，但不能直接用先前 shared 的数值替代两新学生的量化边界。

与上游规格的明确补充/差异：① 中心ROI是空间锚而非主体；补充主体mask、相机分离及前后向遮挡；② L/P/I不是可加的平方贡献或单独因果归因，真实 typed 四角含数值交互；③ 对应由Dense完整视频推得是后验oracle，需公开其信息截止及搜索/边界；④ 不预设H3成立，若同等搜索容量的holdout无增益，转向低维后缀响应与因果干预；⑤ 高噪声step与τ0单列，四步high/low不共享W2；⑥ 旧byte-neutral与0 decode不能证明新pilot全域成功。尚未审阅协调者新代码，以上不声称实现存在具体bug。

## 实现审阅附记（后续委派，独立于先前设计）

针对尚未冻结的 `tools/e0_s2_spatial_prefix_replay.py` 与 `tools/raw_bf16_spatial_prefix_diagnostics.py` 及其测试，静态审阅发现以下解释风险：`component_geometry` 将整个频带所有频点、空间和通道的 `conj(teacher)*candidate` 加权相加，再计算一个复相位与相干度；它是**汇总复向量**的角度，不是各频点相位或逐频coherence。反例：参考信号含等能量频点k=3和4，候选在k=3同相、k=4反相，则两个频点均完全相干，但复交叉总和为零，汇总相位为None、相干度为零；若略微不等，汇总角会跳变。保留每频点复交叉和两侧能量，或明确字段为band pooled complex coherence并且不要声称逐频相位；新增bin统计时计入实际FFT/CPU与输出限制。原测试只使用单频，未覆盖此反例。

协调者已说明正增加native full-M/small-M逐频带与ROI舍入桥；审阅时这一工作尚在进行，无法对新版本宣称完成。当前构造以各状态自身gate/r及typed写回进行四角代数，所以得到的prefix包含真实残差和gate差，interaction也包含gate及typed数值效应；这些不能在报告中等同纯函数矩阵交互或前缀因果效应。现有 captured full-M 输出与小M再算误差须分频/ROI显式比较，并以真实 EXIT/ledger核验完成，不从中途RUN记录推断成功。

验证：`python3 -m unittest tests.test_e0_s2_spatial_prefix_replay tests.test_raw_bf16_spatial_prefix_diagnostics` 中5项replay测试通过；系统Python缺numpy，诊断模块未能导入，因而未运行其测试。未改审阅实现或测试。文献同步：现有L15 T10/G17、T11/G02提升至论文与STDiT3指定局部代码审阅；L39 T09/F13、T11/G02、T16/P20保留既有公式深度并登记作者库未见可执行代码。TokenFlow、FlowVid、OmniCache未找到当前索引LID，未增加图节点。`python3 tools/graph_agent_bridge.py validate` 静态检查通过；不证明模型数值或视频结果。

### 修正后的独立复核（绑定前）

现版逐bin输出及band低相干phase置空覆盖此前双频反例；full-M对small-M的原生锚桥以 `native_error = replay_error + candidate_bridge − teacher_bridge` 保留有符号交叉，形式正确，分别覆盖gY、typed u和rowLN。每位置两candidate×(四输入×两时间视图)×(LPI及native桥) = 32次rFFT，与160位置/part的5120限额一致；每位置7个重放加3个原生anchor，共10次gY和rowLN，各1600/part。`.venv/bin/python -m unittest tests.test_e0_s2_spatial_prefix_replay tests.test_raw_bf16_spatial_prefix_diagnostics` 本次独立执行13项通过。这是静态及小型合成检查，实际native RUN、稀有数值和资源耗时须由真实EXIT验证；频带的复相位依然是汇总量，应按逐bin字段解释。
