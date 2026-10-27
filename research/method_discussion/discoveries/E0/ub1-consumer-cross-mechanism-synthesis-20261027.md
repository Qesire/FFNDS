# UB1消费者恢复之后：表示覆盖、校准几何、状态传播与运动可观测性的竞争解释

2026-10-27 · 独立科学综合 / DISCUSSION_ONLY。阅读原论文与作者源码，重建既存实验的对象；本稿没有重算既存数值、运行模型/GPU/视频/评分、拟合参数或修改执行协议/ACTIVE。当前20源40视频仍以其独立RUN为准，本稿不预报其结果。

**值得继续追究的结构是：同一固定支持能通过消费者重构取回大量训练输出信息，但这种可解码性尚未等同于跨来源、跨状态的动作保真。** 第一竞争解释甚至不必涉及“运动功能”：四步训练问题的N/K几何不同，step0的读出比其他步更难插值；另一解释才是被固定删去的非线性函数覆盖、学生状态偏移，或纹理改变了运动估计。它们给出不同的下一实验，不能用一个83%数字替代区分。

## 1. 本轮已经知道的四个对象

| 已有证据 | 精确对象与结果 | 对后续机制的实际约束 |
| --- | --- | --- |
| [固定RR全160 bank](../../../20261027/UB1_FIXED_RR_CONSUMER_BANK_RESULTS_20261027.md)，P28 | 原24训练来源、每源每site M256；固定54/108/108/162 whole64支持、alpha及原R256，仅改变W2/bias。来源×site均权gated相对SSE 0.055510→0.009434，训练内下降83.01%。step0新残余相对SSE均值占四步合计约90.1%。 | 旧scalar读出没有充分利用保留表示；这是训练内恢复量，90.1%的分母是已报告的相对误差均值，不是视频动作能量或真实flow总误差。 |
| [六条完整视频](../../../20261027/UB1_FIXED_RR_CONSUMER_VIDEO_RESULTS_20261027.md)及[八轴官方函数](../../../20261027/UB1_CONSUMER_OFFICIAL_QUALITY_RESULTS_20261027.md)，P28 | 两个已暴露formal20来源、seed42、各Dense/neutral/candidate。两条neutral及fresh Dense均逐字节复现历史四latent/视频；candidate的MUSIQ更接近Dense；连续RAFT candidate/Dense为0.635、0.223，仍为部分恢复；六条官方binary dynamic degree均false；犀牛aesthetic下降。 | 原生钩子身份已由精确控制检验。外观、运动幅值、美学、prompt动作语义需要并列；八轴自定义子集、一源一真实seed，不是官方五采样九维榜单。 |
| [保存latent的signed几何](../../../20261027/UB1_CONSUMER_LATENT_GEOMETRY_RESULTS_20261027.md)，P28 | 两源四步，candidate相对UB1改变量与UB1误差的cos均负；末步candidate/UB1误差范数比1.0016、1.0110。 | 变化含纠偏分量，同时有其他能量；这是轨迹状态的代数观察，未把直接FFN作用、Dense对偏移状态的响应或真实动作分开。21槽属于latent视频时间，不是四个denoise-step。 |
| [原M一致性](../../../20261027/UB1_NATIVE_CONSISTENCY_RESULTS_20261027.md)，P26；[12/12匹配CV](../../../20261027/UB1_MATCHED_RECOVERY_RESULTS_20261027.md)，P27 | 16保存site的原M31668缩窄W1/GELU/Q2与Dense选列逐位相同；同RR gated gap0。此前CV在四site、两折各12训练来源，读出后另拟合RR，与本轮fixed-RR bank不同。 | 原M的已测Q2路径差异已缩小为次要解释。不能从M160差异倒推原M错误，也不能继承旧CV约39%的持出局部SSE改善。 |

固定bank协议SHA为`d35b0d1233b762236eb75d98a3f93484b639766266280b139008d4fef26d88fe`；六视频协议`ba4a4f8f03a3cdd8c2c796d054f0ff1847e8eab61a6d3e11b1ac477b3710f8a5`；八轴`6dce8f177eab984f501bcae55de8982de3da2ebbb02a1072146ce202a906df7e`；latent几何`16f570b59d81729724e20666a484ae0ed405ebcde73c2c3e9553fc539996a6cf`。这些都是各自已完成范围。新的20源40视频协议`ad81fffb73256c14e65c4221d255080072bcc9902cf420b541e4da4f61254ca6`负责扩大开发分布检验，不改变旧结果身份。

## 2. 先解释“为什么step0剩得最多”：N/K是强对手

[求解源码](../../../../tools/ub1_fixed_rr_consumer_math.py)实际将`E=float(Y)-float(BF16(main+original_RR))`作为目标，以`F=BF16(alpha·selected_postQ2)`为特征，中心化FP64 dual ridge，惩罚固定0.01 trace/N，最后折回BF16 W2/bias；[构建器](../../../../tools/build_ub1_fixed_rr_consumer_bank.py)再以真实BF16分支相加及gate评价。它没有拟合学生自由轨迹或运动标签，原RR没有再次拟合。

在**自由截距、实数线性读出的参考问题**中，令中心化特征为`F_c∈R^(N×K)`，则

\[
\operatorname{rank}(F_c)\leq\min(N-1,K),\qquad
\widehat E_c\in\operatorname{col}(F_c).
\]

| Step | 本轮训练行数N | 所选特征宽度K | 实数参考问题允许的几何 |
| --- | ---: | ---: | --- |
| 0 | 6144 | 3456 | centered rank至多3456；中心化样本空间一般留有至少2687维正交残差方向，目标是否占据这些方向尚未测。 |
| 1、2 | 6144 | 6912 | K>N−1；若特征满行秩，无正则参考问题可插值任意中心化训练目标。 |
| 3 | 6144 | 10368 | 同样允许满行秩插值，且特征方向的识别更依赖样本/正则。 |

这已经给出“step1–3训练误差更容易很低”的非运动解释。此前12/12CV每折N3072，连step0也有K>N−1，且RR与目标不同；两个实验的插值几何不可混用。6144行还嵌套于24个来源，它不是6144个独立场景，来源多样性与矩阵代数秩是不同量。

**ALG条件边界**：上表没有估计实际秩、最小奇异值、目标投影或泛化误差。正ridge、BF16权重/加法、QDQ、来源相关性与固定RR都使真实问题更复杂。样本残差子空间不是HiF4视频的容量下界，也不推出step0具有专属运动神经元。反过来，step0未充分拟合也不能证明换λ就能补回缺失函数。

输出重构本身有成熟先例。[Model Preserving Compression，NeurIPS2022，§§3–4及Appendix I](https://arxiv.org/html/2108.00065v2)用post-activation选列与插值矩阵传播读出补偿；其单层泛化论证要求同分布i.i.d.样本，不能直接套到跨视频状态的6144嵌套行。作者还展示过剪枝集中缺少一个类别时仍能保留该类表现，提醒“校准未见某类”不自动等于函数必然消失。[作者`getID/pruneIDLayer`](https://github.com/jerry-chee/ModelPreserveCompressionNN/blob/205984eb3cc40b15d55596a32a72b19b0a41bf4f/id_utils.py#L7)确实以pivoted QR选原非线性列、用`nextlayer.weight @ T.T`传递修正。UB1当前是固定whole64支持的ridge，不是照搬该自由选列算法。

## 3. 五种跨I/F/C竞争机制及相反预测

| 机制 | 构造性解释 | 与最强对手不同的预测 | 能使该解释降级的观察 |
| --- | --- | --- | --- |
| **H-GEOM：读出/采样几何**（I/C） | step0特征宽度小于训练行数；其他步低训练误差部分来自易插值的几何，而非已恢复全部视频函数。 | 训练与整来源持出之间的差距依step改变；扩大同来源行数与增加独立来源的作用不同。新来源的相同Dense状态误差就可能很大，不必等学生偏移。 | 来源持出、同状态下各步皆稳定低误差，仍出现强动作衰减；几何就不能解释主要视频差异。 |
| **H-COVER：长期非线性函数覆盖受限**（I/F/C） | step0固定删去75% hidden组；保留特征可以重构常见输出，却未必覆盖稀有输入条件需要的有符号响应。固定R256从完整postQ1读取信息，但仅提供既定线性修正，不能任意替代被删的非线性响应。 | 即使在Dense同状态、原M、独立来源上，固定支持也可能在某些变化方向上残留误差；保留所有W1行槽的输入K向稀疏可能呈不同来源尾部，而不要求某个神经元对应一个动作概念。 | 同状态函数误差已很小，主要失配只随学生prefix出现；或换同预算表示后尾部/方向毫无差别。 |
| **H-STATE：值拟合与有限状态响应分离**（F/I） | 训练读出在Dense保存状态上匹配值；它同时改变其他状态上的切向/有限响应。每步先前偏移被attention、其余block与flow继续处理。 | 同一算子在Dense状态与candidate状态的误差大小/方向不同；Dense对两状态的有限映射响应也可能放大已到达误差。负latent交叉项与末步距离增大可以并存。 | 同状态交叉中的算子差跨两状态稳定，Dense状态响应弱，而新来源teacher状态已失配很大；优先转向覆盖/来源而非状态失配。 |
| **H-SOURCE：24来源覆盖了外观，未充分覆盖动作条件关系**（I/F） | 稀有动作、镜头/主体关系与低纹理等条件在校准中分布不均；随机256行也未保留完整时序关联。 | 误差在独立图像cluster、prompt动作类别或纹理条件中集中；相同来源增加行数可能不能替代增加不同来源。运动归因相似度与“平均光流最高”排序可不同。 | 充分匹配来源分层以后，相同误差模式仍主要由state决定；或固定读出跨原分布和新增来源都稳定。 |
| **H-TRACK：运动可追踪性与真实动作改变混合**（I/F/测量） | 成像/边缘纹理变化改变光流特征匹配；Dense中背景纹理漂移、镜头运动也可进入top5。 | 固定已知位移的appearance干预会改变RAFT幅值；物体轮廓/高置信点的相机相对位移与全帧RAFT可能不一致。 | 原速盲播放、主体几何轨迹和不同可观测性检查都显示同向持续位移衰减，且不是整体镜头差异；仅测量解释便不足。 |

这些机制可以同时存在。H-GEOM和H-SOURCE是统计/信息解释，H-COVER是函数类解释，H-STATE是动力学解释，H-TRACK是观测解释；任何一项都不单独授予视频质量或速度结论。

### 一个有用的signed重写

对于既有latent记`u=zUB1−zDense`、`d=zCandidate−zUB1`。写成

\[
d=-a u+d_\perp,\quad \langle u,d_\perp\rangle=0,
\qquad
\|u+d\|^2=(1-a)^2\|u\|^2+\|d_\perp\|^2.
\]

负cos只要求a>0。距离仍可因为沿u过冲，或正交能量超过抵消项而增加；这两种情形不等价于“λ太小”。而`d`来自整条不同rollout，它不是某个FFN直接注入量。对21槽的固定线性时间差`Dτ`可写相同恒等式，但不能将RAFT当线性算子代入。以上是ALG说明，没有对现有latent另做投影计算。

## 4. 论文和作者代码实际改变了哪些判断

**VideoJAM，Chefer等，ICML2025。** 本轮读到[正式出版页](https://proceedings.mlr.press/v267/chefer25a.html)和[arXiv v2 §§3、4.2–4.3、5.3–5.4、Appendices B/C](https://arxiv.org/html/2502.02492v2)。它扩展联合appearance/motion输入与预测，在推理使用模型自身motion预测作Inner-Guidance；Appendix C采用100个去噪步，并只在前半程施motion guidance。其消融中去掉motion预测/引导会损伤运动，局限还包括小物体运动表示不充分。**迁移含义**是研究正常表示是否对时序扰动敏感，以及粗运动何时确定；不能据此把Wan四步的high/low噪声expert命名成motion/detail分工，也不能把它视作无额外调用的FFN读出修改。检查了作者项目页和论文链接，本轮未取得可确认的作者推理/训练代码，算法阅读与代码复现身份分开。

**Motive，Wu等，ICML2026。** 核对[正式出版](https://proceedings.mlr.press/v306/wu26aq.html)、[v2 §§3.2–3.5、4及Appendix D/G](https://arxiv.org/html/2601.08828v2)和[NVIDIA作者方法页](https://research.nvidia.com/labs/sil/projects/MOTIVE/)。其作用是通过motion-weighted梯度做数据归因/精选：AllTracker提供位移/visibility/confidence，loss位置mask用于归因，训练/query共享噪声与timestep降低排名方差，再做投影和来源聚合。Appendix D专门反驳“只是挑高motion视频”：最高/最低influence可有相近平均运动量。**迁移含义**是考虑来源与动作关系，而不是立即给校准loss加一个运动权重；完整Motive还要付tracking、梯度和投影成本。作者页面/论文未提供本轮可核的实现仓库，未把该方法标成CODE_VERIFIED，更未把其Wan1.3B/5B结果当作本Wan14B四步结果。

**BRECQ，Li等，ICLR2021；CTEC，Li/Du，AAAI2025。** [BRECQ §3.2](https://arxiv.org/pdf/2102.05426)通过block reconstruction折中层间依赖和少量校准数据的泛化，正文明确更大的重构边界也可能过拟合，因而“把loss移到视频终点”不是免费修复。[作者`GetLayerInpOut`](https://github.com/yhhhli/BRECQ/blob/e455d62e93c70351961f8991c913b59435bd165f/quant/data_utils.py#L92)在`asym`分支保留FP输出、重新运行quantized prefix取得输入；[block实现](https://github.com/yhhhli/BRECQ/blob/e455d62e93c70351961f8991c913b59435bd165f/quant/block_recon.py#L68)确实消费该缓存，而不是只在README提出该词。[CTEC原PDF Eq.14、Algorithm1、Table3](https://ojs.aaai.org/index.php/AAAI/article/view/34039/36194)明确区分逼近原预测与纠正前步状态误差，其只更新量化参数的纠错配置还可能变差。两者原任务是CNN/图像扩散，不是Wan视频；本轮未取得CTEC作者代码。

这里有三个不同的离线teacher问题：`fD(xC)`是恢复当前学生状态上的原函数；配对`fD(xD)`是针对原trajectory的不同目标；再要求native下一latent回到Dense轨迹还要考虑真实scheduler。它们不可一律叫“用student数据校准”。[DAgger的AISTATS2011原文](https://proceedings.mlr.press/v15/ross11a.html)提供收集learner诱导状态、再查询expert的成熟解释，但其顺序决策保证不能直接转写成四步Wan的质量保证。

**FLAP（既有L04）和RAFT（VBench既有L29相关测量）。** [FLAP作者`BiasGPT`](https://github.com/CASIA-LMC-Lab/FLAP/blob/3bb57db3449dd2fa04a5c2192de80e87e33be2b1/lib/layerwrapper.py#L36)记录输入均值/波动；[compress](https://github.com/CASIA-LMC-Lab/FLAP/blob/3bb57db3449dd2fa04a5c2192de80e87e33be2b1/lib/prune.py#L129)将遗漏输入的均值贡献折入bias。它是原理参照，当前组级CV只是FLAP式打分。另一方面，[RAFT作者`CorrBlock.corr`](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/corr.py#L47)明确以两帧feature dot-product构建相关体。纹理/模糊改变估计难度因此是有计算基础的竞争解释；代码本身并不证明这里的幅值下降全由模糊造成。均匀无纹理图上不同真实平移可产生同一观测图像，是一般的运动不可辨识反例；再换一个依赖外观的tracker也不能自动得到物理真值。

## 5. C/I并行路线：保留所有hidden行槽的输入2:4是不同实验对象

UB1近似为固定`S`上的原非线性特征加固定RR，新consumer只改变这些特征的读出。C3相关输入动作则可写作

\[
\phi_j^{K}(x)=\operatorname{GELU}(w_{1,j}^{\top}S_{2:4}(Q_1(x))+b_{1,j}),\quad j=1,\ldots,H.
\]

它保留全部W1权重行和GELU通道槽，每个函数却只见被保留的输入，所以**没有保留每个原始hidden值**。若W2也做输入2:4，必须对这个新W1/GELU/Q2产生的值再选；不能代入Dense旧postQ2。UB1原whole64选组保持组内Q2尺度，K向W1稀疏则可能联动全部后续GELU与Q2尺度，函数覆盖/数值误差的形状都不同。

[RT-Lynx既有L08，v2 Algorithm1/§§4.1–4.4](https://arxiv.org/html/2605.26632v2)提供激活2:4、范数恢复、dense-input LoRA及融合kernel的成熟先例；其naive activation稀疏也会模糊，不支持“保留行槽便天然保住动作”。原作者图像DiT/H20质量和速度、[本仓库C3静态/合成组件](activation24-online-components-independent-review-20261027.md)、Wan原生视频是三个证据身份。C3尚不提供Wan值误差或视频证据。

**可证伪预测**：若长期覆盖占主导，同总算术预算下，固定删hidden组与K向输入稀疏可能在不同来源/动作条件下呈不同尾部，consumer读出很难消除这种差异；若误差主要是输入尺度与Q2非线性联动，K向方案可能更差，即使所有行槽都在。两者都允许静态重组、共享变量/编码域计算、条件支持或成熟补偿成为后续动作。

成本比较要使用共同分母：仅W1或仅W2做2:4，只将两主GEMM合计MAC减至约75%；两处都做才是理想50%，另有observer、选择、norm、压缩、布局、RR/LoRA、全宽GELU/Q2和驻留代价。两处2:4的总main-MAC可与UB1四步平均50%接近，但UB1每步25/50/50/75%的分配不同；不控制步间预算，运动差不能全归因于表示。C3组件的预pack速度也不能抵扣在线压缩费用。这是OPEN的I/C路线，不是要在当前40视频里替换任何算子。

## 6. 等20源后只选一项：32次真实full-flow的2×2同状态交叉

**优先候选实验（尚未登记、未实现、未执行）**：若20源的来源级结果表明消费者成像改善在更多来源存在，但物理动作/相机相对轨迹仍有系统差异，那么先区分“当前算子差”与“此前状态已经走偏”，再决定是否增加容量、换数据或训练目标。该实验利用现有完整postlatent，不需要拟合、新支持、重训RR或生成新视频。

如果最终选择此实验，预固定disjoint20枚举`0,1,10,11`：cow eating hay、seal swimming、stage door、lighthouse；分别来自两个subject和两个background来源。四图与训练24/旧formal20零image-SHA重合，每图cluster是独立单位；它们仍是已经暴露的DEVELOPMENT，不重称盲确认，也不据20源谁改善最大来换这四图。所有四对baseline必须完整成功且noise文件逐字节一致。若需要真正新确认分布，另冻结未被研究使用的来源，不能更换这四图却沿用当前小实验身份。

复用本轮Dense/candidate的**post-step0作为step1输入，post-step2作为step3输入**，每source两个denoise位置。对每个位置分别运行下表四个原M、原40block的真实map，输入条件、model/cache、gate、scheduler、BF16/QDQ、consumer bank全部固定：

| 完整单步算子 | Dense前缀状态 zD | Candidate前缀状态 zC |
| --- | --- | --- |
| 原Dense | A=TD(zD)：自身状态精确重放控制 | C=TD(zC)：Dense对偏移状态的响应 |
| 当前固定consumer | B=TC(zD)：没有此前candidate状态漂移的直接算子差 | D=TC(zC)：自身状态精确重放控制 |

`T`包含真实整步flow及原native `step_post`。**四格都是新的原生调用**：保存latent没有保存flow；不能由前后latent倒算一个近似flow冒充实测。A/D先分别与原保存的下一postlatent逐raw-tensor SHA一致，任一失败即保留并停止该source，不自动重试。条件编码、timestep/mask、scheduler其它必要状态都应绑定，不能只复制一个latent便假定重放已经成立。原[E0 finite-map实现](../../../../tools/e0_whole_flow_counterfactual.py)可作源码/代数参照，旧候选的结果和BF16边界标签不能冒用于本FP32 scheduler latent。

在FP64分析真实native输出，定义

\[
P_D=C-A,\quad N_C=D-C,\quad N_D=B-A,\quad
I=N_C-N_D=D-C-B+A.
\]

\[
\boxed{D-A=P_D+N_C},\quad
\|D-A\|^2=\|P_D\|^2+\|N_C\|^2+2\langle P_D,N_C\rangle.
\]

`P_D`是Dense对已经到达的不同状态的有限响应，`N_C`是当前学生状态上的直接算子差，`I`检验算子差是否随prefix状态改变。对完整latent和固定21槽一阶差分Dτ各报告三项能量、signed cross、cos与闭合误差。它们属于同一个source内的嵌套观察：先给四个source完整结果，再给来源均权paired summary；不对1280个FFN位置或槽数作独立显著性检验。

**预先区分的结果**：

- 若`N_D`小、`N_C`大且方向不同，`I`大，支持H-STATE的状态依赖；若`P_D`也放大已有输入差，优先研究真实student-state覆盖/有限后缀响应，而不是只重解旧teacher目标。
- 若`N_D≈N_C`且同状态误差在两个输入上都较大，状态专有解释减弱，来源/表示/读出问题更值得先查；这个结果仍不足以宣称不可恢复容量下界。
- step1的`P_D`来自此前仅一个denoise-step的差异；step3的`P_D`包含此前三步。两位置可区分进入末步前已形成的差与末步新增算子差；它没有追踪同一早步干预到终端视频，故不声称已经找到“step0决定动作”。
- 若四源方向异质，完整保留异质性；不再加source/step/block挑选赢家，也不凭latent距离决定视频优劣。

**提议的明确增量预算**：4source×2step×2state×2operator＝32次真实full-model flow；其中16 Dense、16 candidate，合计1280个完整FFN site、2560次main FFN linear、1280次RR linear、2560次FFN QDQ、10240次non-FFN linear与10240次non-FFN QDQ。另计最多4套原生source条件编码、setup/驻留/预载，不将其隐入“0模型调用”；0新拟合、0optimizer、0新完整视频/VAE decode、0官方评分。保存32份原FP32 postlatent及完整receipt，输入bank/旧latent只引用。建议单A100顺序执行、CPU8、resident≤64GiB、输出≤0.5GiB，compute900s/outer1200s作为待协调者实物评估的上限草案；资源与源代码均须新协议冻结/ACTIVE登记后才能执行。每次整flow预扣，失败保留预扣与实际部分计数。

这是比重新拟合160site或再做多臂完整视频更小的**状态机制判别**，不是最低成本的运动测量校验。如果20源差异主要表现为纹理/跟踪可观测性疑问，应先选择已有视频的已知位移appearance控制与主体几何/原速播放检查，暂不选上述32-flow实验；如果多轴质量普遍改善，则继续独立来源质量/成本验证，不因为末latent L2略大就再改参数。先看20源，再选择哪一种对当前分歧有信息价值；当前不执行其中任何一项。

## 7. 检索与证据账本

检索日期2026-10-27；实际使用web search/open/find及只读作者GitHub原文件/API。查询族包含VideoJAM joint appearance/motion与作者实现、Motive motion attribution/common randomness/data influence、CTEC state error correction、BRECQ asymmetric reconstruction、MPC interpolative decomposition/pruning-set shift、FLAP、RAFT correlation、RT-Lynx K向激活稀疏。沿VideoJAM正文/附录、Motive Appendix D反解释、MPC理论/缺类实验、BRECQ论文→实际缓存代码做了后续追踪；搜索结果中的同名SEC2024 VideoJam和第三方重实现未作为作者模型代码引用。

本轮亲自读到的方法范围：VideoJAM v2 §§3–4、5.3–5.4、B/C；Motive v2 §§3.2–3.5、4、D/G及作者方法页；MPC v2 §§3–4、8、I及`id_utils.py`；BRECQ v2 §3.2及作者`data_utils.py/block_recon.py`；CTEC正式PDF Eq.14/Algorithm1/Table3；RT-Lynx v2 Algorithm1/§4；FLAP作者均值/波动及compress；RAFT作者correlation。它们支持的分别是PRIMARY_PAPER或AUTHOR_SOURCE，不是本仓库已运行迁移结果。

作者代码版本为：MPC `205984eb3cc40b15d55596a32a72b19b0a41bf4f`；BRECQ `e455d62e93c70351961f8991c913b59435bd165f`；FLAP `3bb57db3449dd2fa04a5c2192de80e87e33be2b1`；RAFT `2888e15a51fa41140771d3f498ed8023cff098d1`。通过作者repo提交API及原文件取得，未运行其代码。VideoJAM/Motive/CTEC/RT-Lynx未取得本轮可核作者运行实现，保留该阅读边界。

图与文献仅复用既有P26/P27/P28/P29、F04、G21及L02/L04/L08/L11/L29作导航；没有为本文新增LID或节点。本文的构造是把既有恢复、signed状态观察与不同信息/执行单位放在可证伪竞争关系中；条件秩论证、正交分解和finite-map恒等式是标准数学工具，不包装为新定理。最终后续动作由当前20源真实RUN和单独质量结果决定。
