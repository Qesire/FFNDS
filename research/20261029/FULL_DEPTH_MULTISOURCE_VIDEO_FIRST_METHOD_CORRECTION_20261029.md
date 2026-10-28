# FFNDS · 全深度、多来源、视频行为优先：研究目标和方法准入的实质修正

> 2026-10-29 · 独立复核 / METHOD_CORRECTION v1 · ALG + 已发表仓库NATIVE/VIDEO证据 + DESIGN。研究基线 `7852cc2`。这是对当前方法选择和评价流程的修正，不篡改旧RUN、模型/视频或ACTIVE；本次未新增原生模型、GPU、完整视频、打分作业。独立意见，不计作其他Agent的同意票。

## 0. 研究目标与立即决策

**新方法的研究主估计量首先是，在给定原始image/prompt/seed/recipe下，真实替换全部适用FFN并完成四步自由生成后，对视频质量向量和物理成本向量的影响。** 对本Wan路线：四步高/高/低/低；每步40 blocks；每条完整路径160个step×block FFN实例（两个expert共享block编号，但不是80个顺序执行的Block）。

`local gated-SSE`、末层`q/sigma`、单点`J`、单来源固定gain、teacher-state神经元响应、MAC/token，均属于**解释与构造证据**；用于辨别失败机制、设计和优化算子，而非替代视频级方法选择。算法方法的选择与晋级从真实生成和实际成本出发，再解释其机制。

### 已有的反向证据（分别保持真实方法、来源及数值口径）

1. **UB1 fixed-RR：20个已暴露开发来源，20条同noise Dense/consumer真实配对，完整160 FFN位置和四步81帧**。来源级平均连续RAFT 6.893859→1.152446（**ratio of means=0.16717，减83.283%**）；Imaging 72.629357→71.625891（**−1.3816%**），官方二值Dynamic Degree 6/20→0/20；Subject Consistency 0.960219→0.980857。重新按原JSON的20个image-cluster源级记录复算：**18/20来源RAFT值下降；15/20保留比例低于0.5；10/20低于0.25；12/20来源Imaging下降；其中6/20出现Imaging上升而RAFT下降**。来源保留比例的算术平均约0.49577，不能与源均值比0.16717混用。20来源仍均属历史暴露开发集，单source单seed，无人工最终验收。见[UB1质量](../20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.json)和[原生生成](../20261027/UB1_CONSUMER_DISJOINT20_NATIVE_RESULTS_20261027.md)。
2. **Affine54全深度原生双轨：2个历史开发来源、每来源160 site**。在320个site中**318处student-prefix上的raw FFN绝对误差比teacher-prefix更小、2处相同**，而真实四步latent差异增长；这否定了“局部改进足以筛选视频方法”的这一次具体策略。该v3不是UB1的20来源对照。见[320site原生实测](../20261026/E0_PAIRED_FULL_DEPTH_RESULTS.md)。
3. **A0 compact6016全80函数：**历史exposed heldout8，局部gated-SSE energy94 0.12976554 vs random94 0.13812655（**相对减6.05%**）；完整生成后的六视频只覆盖**2个暴露来源×3臂**，energy94连续RAFT相对对应Dense仅0.13507/0.14467，MUSIQ两源均值46.944 vs Dense 76.018。这是另一方法的开发失败，不能作为所有FFN重参数化的普遍结论。见[局部](../20261028/COMPACT6016_FULL80_RECOVERY_RESULTS_20261028.json)、[质量](../20261028/COMPACT6016_CONSUMER_QUALITY_RESULTS_20261028.json)。
4. **M-B low39 fixed-gain：**在原始BF16末层、一个已暴露转移source12，同状态head误差下降93.67%/82.45%，own-state末步Flow误差下降69.59%；**全深度全视频质量、真实节省、跨来源泛化均未建立**。不把它作为方法质量成功票。见[原始fixed-norm](../20261028/ORIGINAL_BF16_FIXED_NORM_TRANSFER_RESULTS_20261028.md)。

## 1. 将研究估计量从局部误差转为视频级条件收益

以**独立image cluster**为抽样单位，固定视频条件 `s=(image,prompt,noise_seed,model/backend,scheduler,conditioning)`；其生成版本 `V_D(s),V_M(s)` 必须来自匹配的真实完整生成。令分维视频质量变化向量
```text
DeltaQ_s = (imaging, aesthetic, I2V-subject, I2V-background,
            temporal_flicker, motion_smoothness, continuous_motion,
            subject_action, camera_motion, semantic_relation, paired_perceptual)
```
各指标只在定义适用的场景/来源上计数，`NOT_RUN`保留缺项而不填零。静态prompt不应强制产生运动；行动prompt既要检查保留幅值也要检查主体行为/相位/方向。绝对RAFT幅值不是运动语义真值；同时以ROI跟踪、camera/global motion、纹理/锐度/匹配置信度检查低RAFT是否因模糊失配。

研究对象是来源条件下的**视频质量分布、最差来源尾部、以及实际速度/显存Pareto**，而不是各项指标随手加权的总分。效能主张需同时给出：
- `paired_quality_delta`：同一source/seed/recipe下生成质量和image-to-video忠实性变化、来源级分布及不确定性；
- `paired_dense_fidelity`：逐帧RGB+感知差、时序/运动向量、光照、对象与事件时间关系、可信ROI和相机分量；
- `real_cost_delta`：本机完整 FFN boundary/GEMM/QDQ/pack/offload、整机wall、显存、预处理、测试开销；冷/热区分。

对每项使用在见结果前冻结的方向、可容忍退化边界、失败尾部要求和统计模型。边界优先来自配对Dense重复seed的自然离散程度及用户认可的场景功能容忍度。不得看到候选质量后改阈值，也不得以局部error/单一美学分数为整个质量替代。

## 2. 多来源设计是方法实验的首项，而非最后的装饰

- **开发比较**：复用已有20来源多维视频与全深度证据来设计机制和覆盖；新候选的**正式开发级方法排序**按至少**20个不同image clusters**的完整Dense/Student匹配视频（包含实有场景分层），避免只挑source0/12或block39。历史已曝光数据可用作开发，明确其曝光、prompt、seed与模型recipe身份。
- **预注册确认**：从未用于训练、方法提出、选择或阈值校正的独立来源形成新确认集；以现有CalibrationV2的**48来源、20 subject/20 background/8 camera**作为**目标设计规模**，实际可用源与类别必须先作数据可行性/去重/权限检查，不可伪称目前已具备48个独立来源。若不可用，必须在观察候选结果之前冻结替代规模和推断范围，不能看完效果再缩减来源。
- **多seed和正式协议**：开发阶段同来源配对至少独立两seed以量化随机波动（若资源受限，先让20个来源都有首条视频，不以两个来源多seed取代多source）；正式VBench-I2V每image-prompt需5条**真实独立生成**，与开发自定义官方函数调用分开。每个来源source-weighted聚合，多prompt/seed/80帧/160site只为嵌套重复观测。
- **跨域分层**：对象/动作（主体轨迹、接触、事件）、静态背景与光照、小物体与高频细节、camera-pan/zoom、遮挡与高低运动；同时保留难例、长尾。静态和动态prompt的`motion loss`定义不同。
- **同条件比较**：Dense与候选同image、prompt、seed、noise、分辨率、81帧/16fps、4step、高低expert、attention、量化recipe、模型/checkpoint；改变浮点类型/调度器/原生NVFP4时为**另一recipe内部的Dense对照**，不横跨两种recipe归因。

**方法探索阶段可以从任意数学/信息/算子起点开展，但一旦要在候选间做“哪个方法更好”的实证排序，必须有覆盖全部160 site的完整自由生成与多来源视频级结果。** 两来源的原生定位只回答机制问题；新候选的首个实验不应再默认低39单点选择。

## 3. 真实计算图的全深度响应归因

每个执行source×seed的40×4 site，按实际原生计算流收集：
`x_D,x_S`（norm2/真正Q1后）→`W1,GELU,Q2,W2`→`gate/residual`→完整block/跨token attention→同step head/flow→scheduler poststate→81帧视频。

局部四角为：
```text
a[k,l]=f_S(x_D)-f_D(x_D)
e[k,l]=f_S(x_S)-f_D(x_S)
p[k,l]=f_D(x_S)-f_D(x_D)
Delta f[k,l]=e[k,l]+p[k,l]
```
分别按原生writeback/整block及跨step forward state保存norm、方向、signed内积、source与空间位置分布；`||a||`或`||e||`变小并不构成正面的视频结论。每个site还应记录实际执行/跳过的MAC/tile、QDQ、访问及成本。

完整的后缀响应，采用**对照混合路径的有限差**而非假设某层的teacher-state Jacobian足以表征视频：
```text
F^M_0 = Dense 完整模型
F^M_160 = Student 完整模型
F^M_j = 前j个被替换的step/block已使用Student，其余仍Dense
DeltaOutput = sum_j [Output(F^M_j)-Output(F^M_{j-1})]
```
这是固定输入和相同初始随机性时的望远镜恒等式；中途重算随前缀改变，数值差必须来自同一有限native程序。全160项逐视频回放成本很高，因此它是**正确的数学归因坐标**，不是要求160×来源的完整视频运行。先收集全部160 site的轻量观测，再预注册跨step、跨深度、跨expert/consumer的少量混合干预，选择以source-stratified诊断与极端site为依据；在独立source验证所观察的贡献方向。必须测到完整40层之后与四步之后的输出/视频，不能仅凭末层head排序。

四步轨迹的flow误差有在当前student state计算的模型差`v_S(z_S)-v_D(z_S)`及Dense动态有限响应`v_D(z_S)-v_D(z_D)`，二者带符号累计。真实作用最终以解码视频的时序/语义分组诊断；gradient/proxy只是寻找机制的工具。

## 4. 竞争候选与程序化决定

**全深度开发阶段的比较器**：Dense；已完成且视频失败的原UB1/compact6016作为同recipe历史失败下界（不可跨backend直接当随机基线）；同计算预算的A0；新的M-B/B-F、QCFB或其他独立新方法。比较必须是40×4同condition同预算；对尚不能编译全深度的单点B-R或末层gain，只标`MECHANISM_ONLY`，不作为全视频方法票。

1. **先用既有视频做机制证据对齐**：汇合UB1的20-source VIDEO与Affine54的320-site trace、compact6016的两源质量，逐方法保持自己的source/recipe；报告运动幅值、成像、主体/背景、scene异常以及缺失camera/temporal-flicker/人工评价。各原型不混作同一个因果实验。
2. **构造能全160 site执行的低成本候选**：分子/分母/归一化结构可帮助不同位置的consumer/训练目标设计；验证实际高/低expert的所有40块。仅末层修复可作为有信息量的受控消融，但与完整学生在相同source/seed条件下比较贡献。
3. **首个可排序的模型实验**：20个开发source按scene strata有同条件完整视频（初轮每源1seed覆盖，追加第二seed），完整160-site真实writeback/flow/latent诊断及视频的多维评分；同budget Dense/A0/新候选，伴随真实FFN和完整生成wall。
4. **失败时直接回推分层机制**：由视频异常（运动失幅、闪烁、主体错位、小物体消失、camera错误）→source/ROI/step → 160-site有符号功能响应 →前置信息/训练目标/算子更改；不同失败场景分别提出相反预测与最小混合路径干预。不要一出现局部改善就启动全80训练或宣称质量改善。
5. **确认与科研主张**：开发选定候选之后，冻结模型、阈值、来源、指标，再在新独立source执行真正多seed全视频确认；同时报告source bootstrap/场景层尾部与实际系统速度和存储/预处理成本。若没有对应质量数据，必须具体列出未完成的产品级功能与缺失的原始视频/指标，不评价候选视频保真为通过。

## 5. 资产与作业身份

复用资产必须按 `asset_id, producer, method version, source image cluster, prompt/seed, step/block, quant recipe, state type, shape/dtype/coordinate, saved metric method/version, locator, reuse scope, invalidation` 登记。已产生的UB1 20×2真实视频、consumer metrics、affine320-site、A0六视频、原BF16两视频、CAL24按语义分别复用，不能合并成新方法已验证资产；未来每个新视频都绑定真实run/cost/quality。

当前状态：**本轮仅重算仓库标量、阅读源码/协议并修正研究设计，真实新模型/GPU和多源新视频均NOT_RUN。** 新训练或原生生成按 `research/ACTIVE_RESEARCH.json` 注册冻结scope；这份文档本身不代表已经运行或自动批准新作业。

## 6. 下一Agent的第一项可验收产物

输出一个**完整160-site×多image-cluster×视频指标的矩阵**（行=方法×真正独立source×seed；列=四步site coverage、运动/语义/画质/时序分维、Dense配对、真实成本、证据身份），并列明已实现且可运行的下一候选。对现存数据以`COMPLETE`标已得的指标，缺的`NOT_RUN`，不能填“local pass”。

随后将现行M-B的数值几何改成全深度可执行的函数构造与验证目标，正式候选比较统一使用视频级质量与系统Pareto；低39相关局部代数留作机制线并行，而非再次抢占方法选择主线。

**成熟评价参考**：[VBench CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)、[官方VBench代码与custom-input说明](https://github.com/Vchitect/VBench)、[VBench-I2V官方5次真实生成规范](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md)。研究行为按照[现行视频与全深度政策](../agent_protocol/VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)执行。