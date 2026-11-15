# 受控复现与重做方案：先恢复原 FFN 算法，再验证 Wan 适配及部署变体

状态：DESIGN_ONLY；本轮仅阅读和设计。没有运行模型、拟合、回放、视频生成或远端写入。方案使用 scientific-research-skillkit 的 experimental-design / research-implementation。当前目标是 Wan FFN；本文没有把“原论文完整系统复现”设为移植前置，也没有把原系统 attention 成功算成 FFN 成功。

## 1. 本次必须重做什么

上一轮审计足以发现方法身份和部分数学目标差异，但不足以让旧视频表成为已核实的原方法对照。只找回 producer 记录也不充分，原因至少有三项：

1. ICC 两残差是部署变体。局部 CAL 上接近四态不等于自由轨迹等价；须有四状态完整 Wan 闭环与同因子两残差配对。
2. Chip Global-P64 不是作者 selector、proxy、支持粒度、状态机的数值参考；同仓 packed/Triton zero-diff 也不等于作者 kernel zero-diff。
3. 原方法可能受 rank、校准统计和校准样本预算影响。只重跑旧 r512、CAL4、所有层稀疏的组合，再次得到负结果仍不能排除较忠实配置。

新的证据链分三层，分别给出 PASS/FAIL，而不是一次“复现完成”：

| 层级 | 必须实现/执行 | 能支持的主张 | 不支持的主张 |
|---|---|---|---|
| L0 原 FFN reference | 固定作者源码，作者原形状/dtype/状态机，可执行正对照与模块 oracle | 原 FFN 算法及作者模块已完整复现 | 原论文 FID、整模型速度或其他模型迁移成立 |
| L1 最小 Wan 端口 | 原算术/selector 尽量不动；明确必要接口、时钟、专家、尾组适配；原生全图 neutral 与闭环 | 方法在当前 Wan FFN 接口下正确工作，已测 Wan 质量/成本 | 作者原完整系统或其最优前沿被重现 |
| L2 部署/研究变体 | ICC 2R、native-rounding/packed、Global-P64、预算/范围变体 | 单项改变的质量、数值和系统代价 | 这些变体等于作者原方法 |

原论文整系统复现是另一条支线：ICC 包含 attention；Chip 的原 Wan 视频配置主要是 attention。可以做官方 Wan2.1 同源 Dense / official-config 小配对确认端到端接线，但明确记录 sparse-MLP 调用数为 0；不能由这两条视频宣告 Chip-FFN 已验证。无必要把完整 FLUX、LLM、50K-image FID 或全 VBench 作为 Wan FFN 前置。

## 2. 固定对象、数据与随机性

模型固定当前 BF16 全权重 LightX2V Wan2.2 I2V A14B 四步，实际时间 [1000,937.5,833.333…,625]，HHLL、CFG off、832×464×81、16 fps、torch_sdpa、拟使用A100 80GB、相同block offload。硬件、CUDA/PyTorch和集群入口均为PLANNED_RESOURCE_UNVERIFIED，本会话没有验证可执行GPU环境。所有近似输入为当前真实 pre-W1，返回 pre-gate，当前 gate 仅乘一次。高低专家的权重 SHA 沿已核 B_BINDINGS。

### 2.1 使用现行 VBench v4，而不是把旧七源升格为新确认集

已按现行 v4 元数据形成 COHORT_PROPOSAL.json，真实图像 bytes、source cluster、全部拟合祖先与近重复关系仍需执行前绑定。metadata 列表不是完整输入冻结。本计划不把历史曝光设为已确认“是”或“否”：historically_unexposed=null，history_exposure_status=NOT_VERIFIED。

- CAL11，seed42：vbi_1005,1039,0895,0937,1094,1082,0966,0304,0648,0456,0032。
- CAL_EXT11，seed42：vbi_1024,1064,0901,0922,1107,0967,0958,0152,0760,0448,0040。与 CAL11 合为 CAL22。
- DEV11（方法/预算选择），seed42：vbi_1013,1040,0886,0947,1109,1084,0950,0176,0800,0392,0056。
- 固定技术/机制 DEV3：vbi_1013、0886、0800，seed42；全图neutral固定vbi_1013、0800；DEV主Dense队列中这两条实际执行hook-bypass/instrumented native capture，随后plain native、ICC forced-dense、Chip forced-dense分别各跑这两源，必需额外6条；Q1/GP64无法复用已验收Dense fallback绑定时各再2条，neutral额外上限10。第一条有效capture计入正式资产，不另开“smoke后重采”。
- EVAL22 proposal，seed43/44：见下表。它是拟建立的本周期来源留出，不自动称历史盲集。

| 官方类别 | EVAL record | 原始图像（官方提示词另存COHORT_PROPOSAL） |
|---|---|---|
| abstract | vbi_0024 | a close-up view of a sea fan in the water.jpg |
| abstract | vbi_0016 | a blue and white smoke is swirly in the dark.jpg |
| animal | vbi_0916 | a white rooster standing in a grassy field.jpg |
| animal | vbi_0876 | the otter is standing in the water.jpg |
| architecture | vbi_0168 | the golden gate bridge in san franscisco is lit up by the setting sun.jpg |
| architecture | vbi_0344 | a dark alleyway with a bus driving down it.jpg |
| food | vbi_0963 | a stack of pancakes covered in syrup and fruit.jpg |
| food | vbi_1086 | A boiling pot cooking vegetables.jpg |
| indoor | vbi_0528 | a large waterfall in the middle of a building.jpg |
| indoor | vbi_0504 | a living room with a couch, table, and a window.jpg |
| multiple-human | vbi_1047 | a man and a woman sitting on a bench playing instruments.jpg |
| multiple-human | vbi_1069 | two young women studying in a library.jpg |
| other | vbi_0951 | A teddy bear is climbing over a wooden fence.jpg |
| other | vbi_0955 | a snowman wearing a santa hat and scarf.jpg |
| plant | vbi_1077 | a close-up view of a plant with spiky leaves.jpg |
| plant | vbi_1079 | a field of orange flowers near the ocean'.jpg |
| scenery | vbi_0816 | a foggy landscape with trees and hills in the distance.jpg |
| scenery | vbi_0696 | a lake surrounded by mountains and flowers.jpg |
| single-human | vbi_0988 | a woman with curly hair is drinking a beer.jpg |
| single-human | vbi_0994 | a man in a hat sitting in front of a brick oven.jpg |
| transportation | vbi_0921 | a blue fishing boat is navigating in the ocean next to a cruise ship.jpg |
| transportation | vbi_0927 | a city bus driving down a snowy street at night.jpg |

EVAL 从 v4 reserve 的各官方类别按 SHA256("FFNDS_REPLICATION_EVAL_V1_20261116|"+exact image_name) 升序前两个独立图像簇选取，保留原先固定的首11，再按相同顺序取第二项；未读任何新输出，不是看第一批结果后扩样。保留官方类别，包括看似不合语义的归类。执行前按原图/近重复簇而非 record ID 去重；同图不同 prompt/crop/seed 必须同角色。冲突只按事先固定的哈希次序替补、版本化，并在看输出前完成。

### 2.2 校准规模：保留 256 作为参考，不把 CAL11 当原论文默认

ICC 原论文校准数量为 256 图像。迁移到 Wan 需要视频输入，但把数量缩成 11/22 是预算干预，不能归入“必需接口改动”。方案固定嵌套 CAL11→CAL22→CAL64→CAL256：CAL22 后，以独立literal FFNDS_REPLICATION_CAL_EXTENSION_V1_20261116在每个官方类别内哈希排序，按冻结类别顺序轮转取样、类别耗尽则跳过，从排除DEV/EVAL的来源簇加入42、234个来源。候选 256 名单与替补顺序已写 COHORT_PROPOSAL；簇绑定后才成为可执行 scope。

- CAL 只运行 dense denoising 并收统计，不做 VAE decode、不评分。
- 一次捕获保留每来源的原始统计，从同一资产构造所有前缀；禁止为 CAL11/22/64/256 各重采一次。
- ICC在同一次dense capture保存CA的作者mag统计与CD的四边界增量统计。CD为u/z/h/y四边界按作者顺序得到的per-source全token统计：abs-delta→token mean→每样本channel min-max→来源累计平均→sqrt→max_ratio=1000裁剪；CA按作者mag入口、同样归约/裁剪合同收集绝对激活，保留实际调用/时间权重，不用CD统计代替。不能拿P512的均值悄悄替代全token统计。
- 校准“sample”明确为一条来源×seed 视频；不把 81 帧或 31668 token 算成独立 256 样本。
- 同时保留每来源统一 PANEL512 的配对 U、当前 gate、row IDs，供 Q1 和局部风险读取；固定每来源512行，来源权重相同。
- Q1 所有 A0、normalization、regularizer scale 与新旧候选解都只来自当前 CAL；不能带入旧七源/G19 fit 祖先后仍称当前来源留出。正式称 Q1r2-CAL256；保留 CAL11 曲线而不伪装参数未改。
- 11/22/64 的结果是校准预算消融；主参考使用 256。若资源不足以完成256，诚实止于已完成预算并收缩结论，不能仅凭11和22接近宣布与256等价。

## 3. L0：直接跑作者模块，先建立真正 oracle

### 3.1 ICC

锁定 ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4。直接使用原 SpMlp 与分解/统计入口，先在作者 DiT-XL/2 形状 d1152/m4608、作者 FP32+TF32，以及相应 PixArt FP16 配置做模块正对照；无需先跑整套图像 FID。

检查原生 anchor、increment、四缓存不变性、bias、GELU、因子顺序、zero Δu、周期边界。将作者执行与独立公式 oracle 分开保存。统计 fixture 必须包含：四边界不同尺度、常量通道、非零极小范围、跨样本 min-max 与先平均再 min-max 不相同的反例、ratio=1000 截断。各 rank 从同一确定分解中取前缀，不重复做完整 SVD。

验收不能只有相似指标：源码调用/形状/state 路径一致；FP64 独立公式在小矩阵误差≤1e−10相对量级；确定性backend/same operation graph要求同输出；合法atomic索引顺序差异不据mask SHA单独判错，作者原数值与独立高精度式的差异另表。不能用因子逐项相等核 SVD，因为奇异向量符号/退化子空间不唯一；核重建矩阵及真正前向。

### 3.2 Chipmunk

锁定 sandyresearch/chipmunk@b5e8d1c3da6bddc7ca9007c48e91b82087c11490，使用真实 SparseDiffMlp callable、原 BF16 Triton mm1/mm2 与 compiled author selector。作者当前 setup 支持 A100 的 portable CUDA utilities；不得以“H100-only”跳过尝试，也不预先用本仓 packed 替代。

先在作者原模块 shape/dtype，N 可整除128、offload关闭，执行可重复正对照；再把同一模块接到 Wan FFN 权重做维度迁移。明确 SparseDiffMlp 不是 nn.Module，构造时会缓存转置权重；不能假设普通 model.to()/offload 自动移动其内部列表。先固定引用设备、再检验部署迁移，避免边测算法边修权重生命周期。

必须覆盖：

- dense anchor → 连续至少3个 sparse hit；各通道 last-update state，而非永远回到 dense anchor；
- step 9→10→11→12 的 refresh 与 index/proxy 复用边界；
- first_n_dense_layers=2，full_step_every=10，top_keys=.3，random_keys=.05，256对齐；
- 编译 selector 的索引、实际 K、状态变化。保留前1024列估阈值与 activation 位表示关联 RNG；
- all-selected/no-update 的分支 oracle、完整128组，以及 batch/layer/request reset。
- 不预设所有零增量/全列更新在 native BF16 与作者 kernel 间 bitwise equal；作者 W1/GELU 和 W2/cache 的舍入顺序先作数值基准，不能把已知数值边界差异直接叫算法错误。

原生代码正对照和本仓端口必须导出同一次调用的输入、实际inds/counts、缓存前后、输出，逐边界找first_bad_boundary。选择器验证关注合法索引集合、计数及RNG合同；atomic生成的合法索引顺序可能不同。算术等价对照必须向两实现注入同一份实际inds/counts，不用mask SHA必同当唯一oracle。这样以后端改变造成的差异不会被局部质量均值掩盖。

### 3.3 已定位的原 LayerCounter 问题如何处理

root 的 UPSTREAM_COUNTER_TRACE.json 显示固定作者 LayerCounter 在进入最后调用坐标前 reset；40层4步最后 call159 看到(0,0,0)，不是(3,0,39)。原码 reference 如实保存该行为，适配端不能复制缺陷再把错调度算成算法失败。

Wan adapter 显式传 scheduler global diffusion step 与本次 layer context，cache key=(request,expert,local_layer,CFG branch)。first2dense 按每专家 local layer 判，不能用80专家层注册成一次实际只走40的全局计数。首次进入新专家强制 dense。默认保留 global step，per-expert local clock 仅为调度消融。修复的资格来自独立调用轨迹，不需生成一批错误视频来“证明最后一层错了”。

## 4. L1：只引入必要 Wan 接口适配

### 4.1 作者默认范围分别保留

- ICC：原周期p=2，全40层，D4共80 anchor+80 hit。
- Q1既有方法：全40层，同样80 anchor+80 hit。
- Chip：作者 first2dense 保留；全160调用为 **8 fixed dense+76 anchor+76 hit**。保留 global step/default refresh，专家冷启动确保 step2 的新专家 dense。
- “三者共同只改层2–39”属于共同支持控制，不是原参考。它放入 DEV 单项消融，不替代正式方法 Pareto。

全图所有160站点都被审计，并不意味着160都稀疏。

### 4.2 不重写已有 voxel；只补 MLP 的真尾组

直接复用原 voxel_chunk_no_padding 与 reverse。源码已支持 T/H/W 不整除4/6/8的尾区域；本任务21×29×52不需要自写compact permutation。对可整除与不可整除几何均核 bijection/inverse、row ID 和输出重排，不加入ghost token。

真正需处理的是 N=31668=247×128+52 的 MLP token group。最保守 L1 为前247整组直接作者 kernel、最后52真实行 native dense；标 DENSE_TAIL52 并计成本。这保护原128块计算不被改动，但尾组本身是明确适配。VALID_TAIL52 仅作一个后续变量：按52真实行均值、正确mask/载入/写回，不把zero padding除以128。

### 4.3 ICC 四态不要因80GB放不下而被2R取代

单专家40层四状态约89.4GiB，A10080GB无法全GPU驻留。以无损 CPU cache offload 和有界GPU staging 保留四态数据与作者运算；每个expert结束即释放缓存，防止两专家同时90GiB驻留。复制不改 dtype、不量化、不改变算术结合顺序。host峰值上限192GiB，GPU上限76GiB；预估不满足则 STOP_RESOURCE，不能自动切2R后仍叫reference。

四态 offload 可用于完整质量和实际成本测量；它的时延是这个80GB可行实现的结果，不声称作者最优 offload 性能。两残差是随后与相同因子配对的 L2。

### 4.4 全图 neutral 与闭环

在既定两个DEV来源上比较4条实际执行路径：native Dense、hook-bypass Dense、ICC forced-dense、Chip forced-dense。hook-bypass使用DEV主Dense队列中这两条正式capture；另3路径各2条，共额外6次完整D4。不能把ICC/Chip forced-dense合成一个方法名省计调用。DEV的instrumented native capture须在两个plain新跑验证输出等价后才作为DEV Dense baseline。Q1与GP64只在相同已验收Dense fallback producer代码/精度/调用绑定时复用neutral；若本轮路径改变或身份不齐，各另加这两源2条，计入neutral保守上限10，不偷藏额外pilot。同初始 latent、相同16fps/输出dtype/调度/offload。逐步 latent 和最终 latent 必须相同（确定性后端下位级相等）；若后端本身非确定，先冻结其自重复包络，不能随候选输出放宽阈值。首个真正有效 capture 属于正式预算。

本scope对ICC作者TF32/FP16到Wan BF16的precision适配进行原模块与同输入局部对照，不纳入另提的4条FP32-FFN自由生成；因此不声称precision适配对完整视频的影响已被排除。FP32缓存的宿主容量/精度方案需单独明确，不能把BF16权重转FP32误称恢复原始FP32权重。

完整近似闭环中必须喂候选自己的当前 U；teacher-input replay 不得充作自由生成。记录四态/两残差零增量偏移，但不以这一项替代视频对照。

## 5. rank、校准、代理与部署的单项消融

先模块/同输入回放，再 DEV，最后固定正式臂。完整视频不做所有因素的笛卡尔积。

### 5.1 ICC

局部/离线主候选恰为10个：SVD/CA/CD三家族×r={512,768,1024}的9点，加CD四态r1408。CA使用作者绝对激活统计，CD使用作者四边界增量统计，两者不能互相代替。r512是历史重现点，768/1024覆盖较高作者相对预算，1408是**完整 anchor-hit 周期**接近 Q1 的补点。不是按 hit-only MAC 挑“公平 rank”。

每token令 F=2dm、i(r)=2r(d+m)、a=d²：
\[
\rho_{\rm ICC4}=(F+i)/(2F),\quad
\rho_{\rm ICC2R}=(F+2i)/(2F),\quad
\rho_{\rm Q1}=(F+2a)/(2F).
\]
r512时四态0.568519、两残差0.637037；四态r1408约0.688426，接近Q1的0.685185。两残差相同预算附近约r704，登记为部署局部补点，不把r1408直接搬到2R后称同预算。

固定因子比较4S/2R；固定r与数值路径比较SVD/CA/CD；固定统计公式比较CAL11/22/64/256；不在一次“改进ICC”中同时改这三件事。ratio1000 与 unclipped 是检查历史差异的 DEV 诊断，原参考始终1000。

### 5.2 Chipmunk

主参考保留作者原proxy、selector、voxel、numerics。以下单变量先在同输入固定 true-M 数据上检验：

- DENSE_TAIL52 → VALID_TAIL52；
- author kernel arithmetic → Wan-native rounding，mask必须相同；
- 原 selector → exact top-K，必须先保留原实际每块K以只改变索引选择；固定K5120是另一预算变化；
- 作者 voxel → raster，保持valid rows、proxy与selector一致；
- global step → per-expert local clock，只在覆盖step≥10的原模块序列/已有O40回放比较；D4 HHLL冷启动且各段<10时D/H与mask重算相同，不安排无辨识力的D4视频；
- 相同voxel、相同exactK下比较全局/块级支持；两者从同一块score集合获得支持，避免旧Global-P64与新Block-P64布局不一致的混杂；
- blockmean → 每块P64 RMS，明确更昂贵；不能叫同信息预算优势。

原 Global-P64-K5120 作为单独算法臂复现，不能挂在“作者kernel优化”名下。full-hidden hindsight/oracle只能诊断，不进入在线速度表。

### 5.3 有限DEV队列及明确选择规则

离线/同输入10点先全部保留；每家族在r768/r1024之间按DEV11全80位点、P512、来源等权F64门后绝对MSE选较低者，若数值差小于预先由CAL neutral固定的数值包络则取r768。此处包络只处理数值不可分，不是科学质量非劣阈值。r512始终保留，不被局部筛选淘汰。

因此进入完整DEV视频的四态ICC恰为7点：SVD512+该家族768/1024较好点；CA同；CD同；额外CD1408。7×11=77条。再加Dense/Q1/Chip作者/GP64各11条=44条，总计121条D4。所有用作选择的真实完整输出与被淘汰点都保存。

每家族正式点在它的两个DEV完整视频候选之间选择：主指标为7个官方适用DEV来源的I2V subject均值（vbi_1013/1040/0886/0947/1109/1084/0950，seed42，来源等权），优先较高者；严格数值相同则按完整周期MAC较小者、仍同则较小rank。I2V background在另外4个适用来源、运动及其余质量轴完整报告，但不事后改成筛选条件。当前没有已授权/已冻结的质量margin，因此不发明5%质量门槛，也不把这个单主指标选点称为全质量Pareto最优。CD1408始终单列，不参与挤掉本家族512/768/1024点。

CAL22预算闭环固定为两臂：Q1-CAL22、ICC-CD所选rank-CAL22，分别在DEV11 seed42生成，共22条；父配置CAL256的同源输出来自121条主DEV，不重复跑。CAL11/64曲线以已保存统计/局部风险为主，不再各开11条闭环。

额外必要适配/部署闭环上限24条，具体不再扩网格：CD双残差（同主CD因子）在DEV11=11；SVD512双残差在DEV3=3；Chip valid-tail在DEV3=3；Chip native-rounding在DEV3=3；共同38层支持的Q1与ICC-CD在DEV2（vbi_1013/0800）各2=4。合计24。若tail等在同一实际inds/counts下逐元素等价，可以取消该项质量重跑；不将省下的额度自动增加其他变体。布局只做数值/成本与必要局部诊断，clock不安排D4视频。

统计抽样诊断固定在现有六sentinel：同一CAL前缀、同rank、同四边界公式，比较PANEL512统计与full-token统计的scale/factor action和DEV局部风险；重用同次采集，不新增完整视频。此项只解释局部抽样影响，不声称排除所有层/自由视频差异。

## 6. 固定九臂正式 Wan D4 比较

在DEV按上节规则冻结三个家族点，再进入EVAL22×seed43/44；等周期MAC的CD1408预先固定，不做EVAL内搜索。

| arm | 定义 | 必要用途 |
|---|---|---|
| DENSE | 原生Wan | 共同输出及速度参照 |
| Q1 | Q1r2-CAL256、既有PANEL512拟合、完整40层 | 当前数据祖先清楚，未假称full-token拟合 |
| I4-S | ICC四态SVD，DEV选定r，原p2 | 普通SVD原算法 |
| I4-A | ICC四态CA，DEV选定r，CAL256，ratio1000，p2 | 作者绝对激活channel-aware分支 |
| I4-C | ICC四态CD，DEV选定r，CAL256，ratio1000，p2 | 作者增量channel-aware分支 |
| I4-C1408 | ICC四态CD1408，CAL256 | 与Q1完整周期MAC接近的强基线 |
| I2-C | 与I4-C同因子/同rank、双残差BF16 | 完整闭环部署消融 |
| C-AUTHOR | 原作者机制+已登记最小Wan适配，首2层dense | 两种Chip实现中的作者式版本 |
| C-GP64 | 本仓Global-P64-K5120，原40层策略 | 第二种Chip实现，明确算法变体 |

Reuse-Y只作已有数据/DEV局部机制对照，不占正式EVAL臂；如另做视频必须占24条消融的已取消槽位并在输出前修改scope，不能无限附加。SVD双残差的闭环在DEV3已单列，不与CA/CD正式参考相互替代。

正式共 **9×22×2=396条D4视频**。每条unique video/latent/initial-noise identity保存，不复制文件名伪装多seed。此为官方函数自定义子集，不能称完整VBench-I2V榜单。

统计主单位是22个原图簇，seed43/44源内平均；token、layer、step、44条视频都不扩大独立n。保留逐源 paired differences、源等权平均、范围和leave-one-source-out。每类别2源；类别分层的不确定性仍很粗，不把44条视频当44个独立源。DEV仍每类别1源，不对DEV做退化类别内bootstrap。

指标不以一个加权总分掩盖失败：预先指定 I2V subject/background 为保真轴，subject/background consistency、MUSIQ/imaging、aesthetic分列；官方适用维度缺失记NOT_APPLICABLE而非0。运动同时报告direction cosine、连续幅值比及动态阈值结果；静态Dense分母低于CAL冻结floor的来源不计算相对幅值比。RAFT沿原40对采样帧方案，source内先算，再源等权，冻结分辨率和帧索引。

“更好”需说明是哪条质量—成本维度，未预先确定科学可接受margin时只报告效应和不确定性，不事后发明非劣界。

## 7. D4/O40 与 weight×state：放在正确性之后，避免大规模重复

### 7.1 先用已有数据排除数值问题

复用现有BWS的6来源、6站点、D4/O40端点、common D4 G_ref与已拟合参数，做新模块与旧有效普通SVD/Global-P64的回归检验；既有raw风险和每cell输出能直接复用则不再算一遍。四态/两残差对照用相同因子。这个阶段不新增native trajectory。

PANEL512能检验ICC/Q1的样本行函数，却不能重建作者true-M blockmean，也不能冒充完整M native性能。对缺少原proxy充分统计的Chip作者臂明确DATA_MISSING，不能通过512行缩小问题后宣告原模块在真M成立。

### 7.2 唯一有上限的新增 O40 capture

若正确实现后，解释仍需比较原生状态且缺少作者proxy/连续时刻输入，最多新增 **6条O40**：CAL3=vbi_1005/0895/0648，DEV3=vbi_1013/0886/0800，均seed42；第一条检查通过即计入这6条，禁止另有smoke重采。

保留原O40权重、scheduler、CFG和真实expert switch。O40为40求解步，若每步cond/uncond两次调用，则每轨迹80模型前向，不按“40次”低估成本。绝不追加256条O40校准；本小支线是机制回放，不替代主D4 CAL256。

一次capture保存：

- 所有40时刻、层4/19/34的统一P512、真实row IDs、current gate、作者voxel块均值、Global-P64真实64行、调用/CFG/expert context；
- 为同输入跨权重精确重算和间隔诊断，保存高噪步{0,1,2,4,8}、低噪步{15,16,17,19,23}的full-M U（仅以上3层）；
- 所有solver latent/timestep，小数据可直接保存；视频不必decode，生成终点与checkpoint identity保留。
- 这些步骤在既有O40 recipe中专家归属需执行前核查；若expert不符，按原scheduler真实分段、预先登记相同相对位置，不能看质量后改端点。

D4对应full-M/块统计尽量在CAL256与DEV Dense capture同时得到，不另开D4重采。若已有O40 full inputs确实足够，6条新增数自动降为0；不是“有旧数据”就默认够用。

### 7.3 分开两个estimand

**部署主问题**：固定主D4校准参数，比较这些参数在不同状态上的可迁移性。这回答“已部署方法对状态变化有多敏感”，不在每状态重拟合。

**机制2×2**：在共同CAL3的D4/O40混合状态上，各权重一次拟合/分解，参数固定跨两个μ；每W重算自己的anchor/target。主风险继续用共同D4 G_ref、同rows、相同source聚合，native gate仅附表。计算
\[
d_m(W,\mu)=R_m(W,\mu)-R_{Q1}(W,\mu),
\]
\[
I_m=[d_m(W_D,\mu_D)-d_m(W_O,\mu_D)]
-[d_m(W_D,\mu_O)-d_m(W_O,\mu_O)].
\]
这个小CAL3诊断与主CAL256参数不是同一个实验；必须各有参数manifest，不能混表。

新加入C-AUTHOR与ICC-CD，旧普通SVD/GP64若执行身份仍完全一致则复用。既有BWS已说明W效应较小，但未评价新原作者臂，不能用旧null替代新臂。

### 7.4 还剩哪个状态因素值得分离

同一O40 dense轨迹、同W/CFG/anchor，在high0→{1,2,4,8}、low15→{16,17,19,23}上离线比较单anchor有限增量，新增native轨迹0。它检验固定轨迹上更远端点的难度，不叫“纯训练蒸馏效应”，也不是作者多hit策略的部署表现。

同一捕获序列另做逐步多hit重放，保留Chip列级更新和原refresh；Q1/ICC若改成同refresh，明确命名调度干预。P512重放不产生视频质量或速度结论。仅把Δu按α缩放是虚拟弦诊断，也不能冒充真正改变solver步长。

若要单独识别CFG或真正solver步长，需要另冻同W、同起点、同噪声区间的native反事实求解；本计划不自动发出这种新任务，更不把W_O×D4或W_D×O40的非原生全生成当必要2×2。交叉FFN同输入回放已足以定义当前weight×state estimand。

## 8. 明确复用与必须重做的边界

| 现有资产 | 复用方式 | 不能替代的新增证据 |
|---|---|---|
| 原论文/固定作者代码/旧审计 | 定义与差异清单 | L0新鲜执行reference |
| CAL11/DEV11 P512 bank | 相同input/row/recipe时供拟合与函数评分 | 全M作者CD统计、块mean、实际P64行缺项 |
| 已核SVD因子 | checkpoint/rank/precision/driver一致时直接复用；较高rank可一次分解取前缀 | 作者CD producer真实统计与ratio路径 |
| 旧Q1参数 | 方法回归与历史表 | 当前周期source-heldout主表；除非全部fit祖先合规且CAL定义一致 |
| BW/BWS结果/6条O40 | 作为原范围事实、已有方法回归和相同raw cells | 新作者Chip proxy、多hit状态、CD新臂缺失 |
| 历史七源×2seed视频汇总 | 保留历史结果与差异；有真实video/身份时可重评 | 新模块L0/L1闭环、当前cohort与timing |
| 旧GP64同mask数值验证 | 缩小本仓backend风险 | 作者compiled selector与原kernel等价 |
| 旧timing描述 | 容量估计、预算先验 | 当前九臂生产时延与严格配对 |

新闭环不能只给“producer已补”；至少九臂的受控Wan生成、4S/2R配对、作者Chip/native状态验证必须有新鲜运行证据。新代码影响哪些臂就重跑哪些；不因某个selector改动无差别重跑全校准库。

## 9. 成本、停机条件与交付

### 9.1 预算按真实动作计

以下是提案上限，不是已授权/已执行的调度；运行前用首个有效正式样本估计时间和磁盘，不能虚报精确GPU小时。

| 阶段 | 最大native轨迹/生成次数 | 额外工作 |
|---|---:|---|
| L0模块 | 0整模型 | 原shape/dtype、状态机、少量full-M module；1 A100 GPU-hour提案软上限 |
| CAL D4 | 256，全部前缀且不decode | 缺项才新采；四边界full-token统计+Q1用P512一次保存 |
| DEV主队列 | 121 D4 | 7四态ICC候选77；Dense/Q1/Chip作者/GP64共44 |
| CAL22闭环消融 | 22 D4 | Q1与主ICC-CD各DEV11；父CAL256输出复用 |
| 适配/部署消融 | ≤24 D4 | CD2R11+SVD2R3+tail3+rounding3+共同38层4 |
| native neutral | ≤10 D4 | DEV含hook-bypass2；plain2+ICC forced2+Chip forced2必需；Q1/GP64不能复用已验绑定时各2，条件不触发扣除4 |
| EVAL主表 | ≤396 D4 | 9arms×22sources×2seeds；重复点去重，缺失参考不以额外变体补位 |
| timing技术重复 | ≤60 D4完整调用 | DEV3×{Q1,I4-C,I2-C,C-AUTHOR,C-GP64}各对Dense做ABBA；每对4次完整调用 |
| D4失败重试 | ≤8实际执行 | 仅有已定位可修复的技术失败；保留失败，耗尽即INCOMPLETE；不允许因质量不好重试 |
| O40补充 | ≤6实际执行总数 | 包括检查与失败尝试，不另开smoke；CFG通常每条80前向；不自动第7条 |
| 原模型小闭环 | ≤2组pair=4原模型生成 | ICC原生小配对、Chip官方原生小配对；记录各自真实step/CFG，非Wan FFN成功替代 |

统一上限：**889条D4基础执行（256+121+22+24+10+396+60），另最多8条D4技术重试；最多6次O40和4次原模型生成单列。** 不是把396写成总成本，也不与ICC专项的旧724建议叠加。全部是待执行提案上限，不构成GPU/集群派发授权。可兼容既有资产逐条抵扣，取消的阶段不挪作新实验。

D4最多897次×4=3588模型前向（失败按上限估，实际另记）；O40若双CFG最多480次，合计本Wan2.2队列≤4068次模型前向。真实失败可能只运行部分step，记录actual NFE。原模型4条按其步数单列。所有warmup须来自已计入的正式DEV或模块预热，不藏额外完整推理。


资源均为PLANNED_RESOURCE_UNVERIFIED：提案1 GPU/job、A10080GB；GPU live/reserved上限76GiB、host RSS192GiB；临时可用磁盘至少600GiB作为CAL256 P512与有限O40原始输入的保守提案门限，按真实manifest累计并在每来源后检查。只保存科学有用payload；全M不在256来源80层全量落盘。

CAL捕获若无法在16 GPU-hours提案内完成、离线拟合/分解超过16 GPU-hours、整体超过64 A100 GPU-hours，暂停并提交真实耗时/剩余工作；不能悄悄缩样本、减少作者统计、漏掉最弱结果，或自行扩机器预算。这些是保守停止上限，不是性能承诺。

正式九臂均有EVAL未插observer的生产计时；60次技术重复只覆盖明确列出的五个候选，I4-S/I4-A/I4-C1408没有额外ABBA，不隐去其EVAL时间，也不把其精度说成同样的技术重复。生产时间主表取EVAL真实生成，统一包括/排除文字编码、VAE、加载和预热边界，另给denoising core。首次compile、首次权重load分别报告。CPU cache offload、copy、selector、packing、anchor额外投影均在其应在的范围内，不从某候选单独扣掉。ABBA只估技术波动，不能把重复执行增加独立来源n。source×seed顺序按已冻结RNG打乱，各臂共享initial noise但缓存请求间清零。

### 9.2 STOP / PASS 条件

- METHOD_SPEC / BINDING缺口：有明确运行对象但关键算法未定，停止相应臂，不用替代实现后继续称原法。
- L0任何selector/state/shape/counter不一致：先定位first_bad_boundary，修正后仅重验受影响边界；不进入大CAL/视频。
- Native neutral不通过：停止全部approx质量归因，优先修adapter/offload/context。
- OOM/资源不足：记失败与real peak，保留样本，停止该路径；不把4S静默换2R。
- CAL统计前缀使用了DEV/EVAL或更大CAL祖先：本拟合INVALID，不能保留有利结果后标小预算。
- EVAL生成失败：报告失败原因与配对缺失，不按视觉好坏替换source；修复后完整重做受影响的同一source×seed×arms并版本化。
- 性能差异小于技术波动：TIME_INCONCLUSIVE，不用局部GEMM估计替代端到端加速。
- 质量明显下降：这是已核实现的真实结果，完整报告；不因为它不利于原论文或Q1而自动增加rank、改scope或再选seed。

最终交付为 MODULE_CONTRACT/REFERENCE_RUN、WAN_ADAPTATION_DIFF、COHORT_BINDINGS、FACTOR/CAL_MANIFEST、CALL_TRACE、每视频input/noise/config/weights/output绑定、原始逐源指标、无observer timing与失败账本。真正“完成”逐层绑定命令、exit与artifact，而非只靠文档声称。

## 来源与本设计的依据

- 上轮联合审计：ffnds_review/research/20261116/ICC_CHIPMUNK_WAN_FIDELITY_AND_FFN_STRUCTURE_20261116.md。
- 当前v4：Qesire/FFNDS@e7a1e6b7a6babb5cf742072cc07a9961af854416 的 config/g20_vbench_restart_cohort_v1_20261113.json、research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md。
- 原VBench metadata：Vchitect/VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/vbench2_i2v_full_info.json。本目录保存metadata副本与COHORT_PROPOSAL。
- 既有BWS设计/绑定/结果：ffnds_review/reports/20261114/g20_two_track_core/BWS/v1。
- Chip原模块/A100/voxel事实由chip_fidelity直接核固定作者源码；counter反例由root的UPSTREAM_COUNTER_TRACE及独立源码审阅支持。
- 所有新增阈值、候选、预算与source角色均为本次待执行的设计，未写成已经运行的实验结果。

