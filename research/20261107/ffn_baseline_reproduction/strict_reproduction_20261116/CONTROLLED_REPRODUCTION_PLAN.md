# 下一步实验方案：完整 FFN 复现、Wan 适配与方法差异

日期：2026-11-16。整合基线：FFNDS `5a5b84a04f71aa1fa6c426febf614a1a4c1d2492`。状态：**设计已整合；新模型/GPU运行为0**。本文是下一阶段唯一执行设计；[预算](EXPERIMENT_BUDGET.json)、[数据角色](COHORT_PROPOSAL.json)和[ICC校准配方](ICC_CALIBRATION_RECIPE.json)提供机器参数。作者定义及代码定位沿用[统一代码引述](CODE_CITATIONS.json)，详细算法分别见[ICC参考](ICC_STRICT_REFERENCE.md)、[作者校准](ICC_AUTHOR_CALIBRATION.md)和[Chipmunk审计](CHIPMUNK_UPSTREAM_AUDIT.md)。这些专项文件解释机制，不另起实验队列。

## 1. 本轮要得到的答案

本轮先建立可核验的 ICC 与作者式 Chipmunk FFN 参考，再在当前 Wan 上比较 Q1r2、ICC、作者式 Chipmunk 和 Global-P64。依次回答三个问题：

1. **实现与迁移是否正确？** 原论文/作者代码规定的校准、统计、缓存、选择器、数值顺序和状态分支是否完整；Wan 接线与必要适配各自有什么执行证据。
2. **在相同部署任务上，各方法如何交换质量与成本？** 固定五臂、同DEV11、同seed42，测自由生成质量、完整FFN成本和可配对的端到端时间。
3. **差异出现在哪一环？** 用同输入回放区分ICC两级近似、秩与缓存表示，检查两种Chip的实际支持/数值行为，再结合既有BW/BWS约束权重与状态解释。

与两条研究主线的关系：本轮优先补齐B线比较的可信基础；A线已完成的新算子实验继续作为资产，下一项算子构造由本轮实际残差结构决定。本轮不重开A/BW/BWS，也不把当前对照乘成新的算子网格。

| 竞争解释 | 最有区分力的观察 | 本轮如何取得 |
|---|---|---|
| 旧差异来自实现/校准偏离，或来自算法在目标域的近似能力 | 作者oracle、采集trace、同因子4S/2R与完整三家族结果 | 阶段1–3与E4；旧→新同时改多项时只称参考修正的组合效应 |
| ICC误差主要来自第一层近似穿过GELU，或第二投影对真实隐藏增量的低秩限制 | 分项向量、门后能量和交叉项 | 对选定主ICC的66个局部格记录两级误差，不增加视频 |
| 两种Chip的差异来自支持/代理，或数值与缓存执行 | true-M选择结果、同indices/counts的算术对照、实际支持数量 | 两源六位点回放及模块验收；整视频差不能归给单一改动 |
| 差异更多关联权重变化，或去噪状态组合 | 已完成BW/BWS的固定比较 | 保留各自估计对象，不新增O40；不能据此排除蒸馏的一切作用 |

五臂是已暴露DEV上的开发比较。ICC与Q1校准来源尚未配平，局部选择也使用DEV11；因此结果支持规定处理流程的比较，以及固定因子的误差定位，不承担来源留出的总体优胜或纯算法因果结论。

## 2. 固定比较对象

目标为当前 **Wan2.2 I2V A14B、四步蒸馏权重、BF16、CFG off、832×464×81、16fps、torch_sdpa、相同block offload**。实际调度时间约为 `[1000,937.5,833.333…,625]`，专家为HHLL；执行读取scheduler真实数组。FFN维度d=5120、m=13824，有效N=31668，patch grid为21×29×52。所有方法消费当前norm/modulation后的pre-W1输入，返回pre-gate输出，宿主当前gate只乘一次。

| 主臂 | 固定身份 | 范围与重要区别 |
|---|---|---|
| Native Dense | 相同权重、采样器与数值配置的原生FFN | 主质量与成本参照 |
| Q1r2 | 原VBench CAL11协议、固定拟合/导出/在线执行器 | 全40层；80 anchor+80 hit；优先复用合法参数 |
| ICC | SVD/CA/CD三家族完整局部评估后选一族，r1408，作者四状态 | 全40层；80 anchor+80 hit；主参数来自ICC_CLEAN_CAL11 |
| Chipmunk Author | 作者block128 proxy、compiled selector、Triton与持续缓存 | 每expert首2层Dense；8 fixed-dense+76 anchor+76 hit |
| Chipmunk Global-P64 | 当前FFNDS变体，64真实行、全局exact-K=5120 | 全40层；独立算法身份，不称作者实现的另一后端 |

ICC四态r1408的完整anchor-hit主MAC比约0.688426，Q1约0.685185；这是算术预算接近。r512只作为同分解前缀的局部历史参照。作者Chip保留原top_keys=.3、random_keys=.05、period10、mask-cache=2和首2层Dense，实际counts与全部额外工作计入成本；不人为制造共同38层或多强度视频网格。

代码定位：`ICC-U01–U07/U12`、`CHIP-U01–U12`、`CHIP-F01–F05`。Q1的数学/数值入口可核到固定[b_common_replay.py的fit_q1/q1_predict](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py)，但BWS的实际参数是CAL3 mixed；主Q1必须另闭合CAL11的producer、A0祖先、lambda、PCG、导出与全80参数身份，不能用BWS参数代替。

## 3. 数据与已有资产

| 资产/集合 | 本轮角色 | 当前状态与动作 |
|---|---|---|
| ICC_CLEAN_CAL11 | 作者式CA/CD主校准 | 11条真实clean clip或可追溯编码x0待绑定；优先找旧CAL图片原片，否则另选同规模真实视频 |
| ICC_CLEAN_CAL3 | 两项必要适配的小诊断 | 从上述11个来源按预定hash顺序取前3；不新增来源、不按结果挑选 |
| ICC_CLEAN_CAL_EXT11 | 可选扩至22的统计预算诊断 | 默认关闭；仅在近似臂视频前明确冻结是否执行，扩集不重抽前11 |
| 原VBench CAL11 | Q1原有拟合及其资产复用 | 保留原图文/轨迹身份；不能改名为ICC clean视频 |
| DEV11 | 五臂主视频与局部开发比较 | 固定11个来源、seed42；保持历史暴露身份 |
| DEV3 | E4及条件E5/E6视频 | vbi_1013、vbi_0886、vbi_0800；是DEV11子集 |
| 两个neutral来源 | 宿主核验及必要true-M输入 | vbi_1013、vbi_0800；同样属于DEV11 |
| VB_DENSE22 | ICC/Q1的同输入局部目标与既有数据 | 实际是CAL11+DEV11的U/Y/gate面板，不能作为CAL22或clean x0 |
| BW/BWS与A归档 | 已完成结果及受限的机制约束 | 按原source、权重、状态、参数、数值身份复用，不重做旧scope |

DEV11固定为：vbi_1013、1040、0886、0947、1109、1084、0950、0176、0800、0392、0056。原Q1 CAL11固定为：vbi_1005、1039、0895、0937、1094、1082、0966、0304、0648、0456、0032。完整元数据以[COHORT_PROPOSAL.json](COHORT_PROPOSAL.json)为准。

真实clip记录包含原始视频、帧区间与fps、空间裁剪/缩放、整段一致的水平翻转、实际文本、首帧条件、source/近重复组、VAE及x0身份。ICC与DEV在来源组层级隔离。图片、生成末端、首帧复制和零填充条件不自动满足真实clean输入要求；现有缺项见[资产审计](ICC_CALIBRATION_ASSET_AUDIT.md)。若只能取得已编码x0，还需有可重载的同一次raw μ/logvar或真实clip，以完成posterior配对诊断。

所有来源、时间、噪声、条件dropout和posterior噪声在看对应结果前固定。各source派生独立RNG记录；增加EXT11不改变前11。CAL3顺序为`sha256("icc-clean-diag-v1|"+source_cluster_id)`升序，去重后取前3；最终3个ID在数据绑定时落盘。

## 4. 执行顺序与阶段产物

| 阶段 | 输入 | 执行动作与输出 | 推进依据 |
|---|---|---|---|
| **0 绑定与实现准备** | 固定源码、现有bank/参数、真实clip清单 | 完成actual producer/config/weights/data/resource绑定；实现原模块reference、Wan adapter和作者式collector | 能定位真实文件与调用入口；缺项逐项保留，设计不冒充已运行 |
| **1 原模块与宿主核验** | 作者函数/核、少量原shape输入、两个neutral来源 | 有限模块oracle；两源native/plain与adapter forced-dense；首次hook同时捕获true-M输入 | 模块、接线、状态、尾块和资源通过实际核验；有效Dense产物直接计入主55 |
| **2 作者式校准与适配诊断** | ICC_CLEAN_CAL11、原生VAE/条件/scheduler | CA/CD最多66 NFE；CAL3两个配对干预最多30 NFE；四边界逐源统计、调用trace、敏感性报告 | 所有11源/两专家/两统计家族完整，采集过程与统计oracle一致 |
| **3 分解、局部比较与冻结** | 主统计、旧DEV面板、两个true-M来源 | 三家族六位点；r512/r1408及4S/2R；主ICC选择；补齐选中族全80因子；如启用则做CAL22诊断 | 三家族候选齐全；按唯一局部准则选定；冻结全80参数、E5/E6及数值路径 |
| **4 固定五臂自由生成** | 已核验端口与冻结参数 | 完成DEV11五臂共55槽，扣除阶段1及真实兼容旧产物；输出视频、latent、实际调用与资源记录 | 五臂按相同质量协议评分，失败/缺失如实登记 |
| **5 有限消融与解释** | 主4S父视频、已冻结触发 | 必做E4三条；E5/E6各最多三条；汇总逐源质量/成本、误差结构和旧证据约束 | 形成当前配置内可复核结论，明确尚未由本轮区分的解释 |

阶段0的数据绑定和模块实现可并行。缺clean视频时，可继续原模块、Q1/旧资产核对、普通SVD与neutral准备；CA/CD采集及完整三家族选择等待真实输入。阶段1的Dense/neutral是方法无关的输入生产，可以先于阶段3；**近似臂自由视频开始前**冻结主家族、所有参数和条件分支。选择只使用预定局部指标，不依据Dense观感或近似视频结果调参。

阶段1的hook-bypass Dense虽属于主55，但先执行并唯一计账，不形成“主视频前置又依赖主视频”的循环。此前已暴露DEV仍按开发集解释，不因暂缓查看新视频而改称盲测。

## 5. 阶段1：完整机制与最少宿主核验

### 5.1 模块验收清单

| 对象 | 必须覆盖的行为 | 最小可区分的检查 |
|---|---|---|
| ICC四状态 | u/z/h/y anchor、两级低秩、真实GELU/bias、hit不更新anchor、p2及请求重置 | 原SpMlp同输入/因子逐边界；zero-delta；连续两hit；full-rank小例；导出重载 |
| ICC统计/分解 | 四边界、abs-before-mean、逐样本min–max、sqrt/cap1000、scale eps一次、反缩放 | 直接作者函数oracle；非零min、常量向量、零delta、两不同source；比较factor action，不要求合法SVD因子逐位同号 |
| 作者Chip | blockmean/preactivation proxy、原selector、256对齐、持续cache、selective writeback | 同完整输入的proxy与selector；固定实际inds/counts比较h、delta、W2 delta、y；合法索引与计数 |
| Chip时钟 | dense/首2层/hit/refresh/mask reuse/新请求 | 一条小模块轨迹覆盖0/1/2/9/10/11/12，包含step10刷新后step11旧mask复用；不生成七条视频 |
| Chip数值/设备 | 作者Triton舍入、compiled selector、活动W2转置、缓存槽和offload | 原provider=triton的小型执行；一条expert/layer切换；逐段同mask对照；默认常驻与无损offload等价 |
| Global-P64 | 真实64行、post-GELU RMS、全局exact-K5120、既有刷新/释放 | 固定helper/indices与历史source；不拿其同mask验收冒充作者Triton |
| Q1r2 | 原拟合目标、A0祖先、PCG停止、F32导出/BF16执行、anchor常数和gate | 同CAL11的收敛记录与参数身份；zero-delta与实际预测入口；缺合法全80参数才按同协议补拟合 |

选择器保留已识别的对抗边界例：13824列、前1024为1、随后每1024一列为0、quantile=.7、random=0、multiple=256，检查reported count内没有未写sentinel。若实际触发源码缺陷，保留原参考失败、修复diff及同输入检查；修复后的safe port独立命名，不以非法结果评价算法质量。无关原模型系统/LLM榜单不作为FFN模块验收前置。

### 5.2 两源neutral与true-M捕获

每个neutral来源核native plain、hook-bypass、ICC forced-dense、作者Chip forced-dense。Q1与Global-P64的forced-dense只有在相同producer/数值/输入路径已实际验收时复用，否则各补两条。两个hook-bypass属于55主槽；额外plain2+ICC2+Chip2及可选Q1/GP64各2，**额外最多10条D4**。

同一路径比较逐步flow/latent与最终latent。确定性同算术要求一致；若原后端存在合法非确定性，先在小模块规定自己的数值包络。E5使用的局部风险容差与这些模块误差保持同量纲，在方法结果之前记录；近似结果出现后不放宽。

这两条hook同时保存六位点`expert∈{H,L} × layer∈{4,19,34}`、各自anchor/hit的**完整真实pre-W1 U**，共24张`[31668,5120]` BF16张量；保留raster token坐标、grid、voxel映射，以及P512原生Y/current gate和GP64行索引。U本体7,782,727,680 bytes（约7.248GiB）；含必要目标、索引与小trace的档案上限9GiB，单tensor顺序写出，额外capture staging≤1GiB。不保存所有层full z/h/y。

若旧hook没有这些full-U，就保留这两个主Dense槽用于一次正式捕获；不能先抵扣旧P512，再额外开未记账capture。作者Chip selector使用full-M；不能把P512评分行当作全部选择输入。

两源×六位点共12个anchor-hit pair，作者Chip和GP64各一次，共24个full-M局部预测格。算术同mask检查只取两源×两expert×layer19的4 pair，使用完整输入实际选出的mask，在voxel后完整块0/123/246及尾52上检查，不裁小输入后重抽selector。

## 6. 阶段2：ICC作者式校准

### 6.1 主配方

原依据为作者[CA/CD采集循环](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py#L54-L101)，对应`ICC-U08–U11`和`ICC-CAL-W01–W08`。每条真实clip经原生Wan编码得到clean latent；I2V首帧/零帧编码与mask是独立条件，仅目标latent加噪：

\[
x_\sigma=(1-\sigma)z_0+\sigma\epsilon.
\]

| 分支 | H专家 | L专家 | 每source调用 |
|---|---|---|---:|
| CA | 随机均匀选index0或1，fresh-noise后一次dense forward | 随机均匀选index2或3，fresh-noise后一次dense forward | 2 |
| CD | fresh-noise到0，H调用→真实step0→H调用1 | fresh-noise到2，L调用→真实step2→L调用3 | 4 |

每source最多6次，CAL11最多66次。所有调用使用待压缩的同一冻结蒸馏checkpoint，FFN保持Dense；每次运行同时取得当前expert全部40层统计。CD第二输入来自原生step_post，保留FP32先减σv再加σ_next v的顺序；两次之间不重新加噪、不reset、不跨expert。CA后端点必须fresh-noise，不能由CD第二状态替代。

主VAE模式是`wan_native_normalized_mean`；文本条件保留作者p=.1的Bernoulli dropout，映射为实际T5空字符串编码，保留I2V视觉条件，CD pair条件不变。batch1、epoch1、不丢尾批；四边界全有效token、source等权，完整保留作者统计顺序及cap1000。统计累计采用已声明的FP32路径，并对同值作者函数做数值检查；原dtype敏感性在模块中记录。

永久保存逐source/expert/layer/family的四边界通道统计、count及RNG/条件/时间身份；CD首调用的完整激活只作当前pair临时缓存，逐层消费后释放。CA/CD首调用仅在实际操作数、时间、噪声、条件、模型和dtype完全一致时共享，预算不依赖共享成功。旧rollout CA/CD统计及其因子不混入新主统计。

### 6.2 必做CAL3配对适配诊断

用户要求必要适配也有消融。两项存在输入分布变化的映射用CAL11内预选3源检查；主配方始终保持不变。

| 单因素分支 | 固定什么、改变什么 | 新增NFE上限 |
|---|---|---:|
| VAE mean→posterior | 固定clip变换、I2V条件、t/ε、dropout及模型；仅目标z0从原生mean换为正确归一化posterior样本 | 12 |
| text↔empty | 固定主z0、t/ε和视觉条件；将主分支实际context在真实prompt/真正空字符串之间翻转，CD两次同条件 | 18 |
| D4时间支持核查 | 记录既定随机CA覆盖和CD同expert唯一pair；对照scheduler实际数组 | 0 |

Posterior从**同一次原生encoder的raw μ/logvar**构造：FP32的`η~N(0,I)`，`raw_sample=μ+exp(0.5*clamp(logvar,-30,20))*η`，然后整体应用相同Wan通道mean/std，再按主输入dtype进入原生add_noise。η使用独立固定RNG并记录；不把已归一化μ与raw logvar混合，不由已舍入normalized μ逆推raw μ。clamp依据固定LightX2V的sample规则，完整调用须避开其带非默认scale时的μ/logvar不一致：[VAE sample源码](https://github.com/ModelTC/LightX2V/blob/9cb6929d83814ff28c4e1386532298fe93a96af1/lightx2v/models/video_encoders/hf/wan/vae.py#L818-L823)，对应`ICC-CAL-W07`。

在H-CD的σ0=1处，目标latent不依赖z0；I2V/context/ε不变且实际输入身份相同，则整pair直接复用。H-CA若抽中index0也可复用。故posterior每源新增至多H-CA1+L-CA1+L-CD2=4，3源至多12；条件互换每源至多6，3源18。复用前检查实际输入；身份不符先定位构造错误，不用额外模型调用绕过。

两分支各自相对主配方一次只改一项，不合并成2×2网格，不重新分解因子，不增加视频。这30次产出编码/加噪对照、四边界统计和流场预测的配对差异；固定因子的误差分解使用下述DEV既定66格。敏感性是连续结果；大变化记为适用范围限制，不挑一个效果更好的分支替换主方法。

主校准66+诊断30=**最多96 NFE**；CAL22累计132+同一份诊断30=**最多162 NFE**。这项诊断检验输入和统计对适配的敏感性，不能证明重新拟合后的因子或终端视频不受影响。D4部署域有明确合同，但本轮没有建立原DiT时间支持与D4支持的因果等价，也不自动添加50时点校准。

## 7. 阶段3：有限局部比较与主ICC冻结

### 7.1 局部任务清单

所有ICC/Q1评分使用DEV11、六位点、固定P512行、同一native Dense目标和当前gate。在评分开始前冻结统一输入manifest：两个neutral来源用阶段1新full-U捕获的对应P512/64输入与gate，其余9源使用配置一致的既有bank。若新旧两源记录逐值及身份相同可直接复用；若不同，则以新捕获为该两源的唯一局部输入，不先算旧格再重复。按(source, expert, layer, family, rank, cache_mode, factor/input身份)缓存结果，同格只计算一次。

局部canonical anchor/target统一为**当前权重、BF16、P512形状的native_ffn重建**：每个pair一次生成z_a/h_a/y_a与z_b/h_b/y_b，合计132个端点，沿固定[own_w_targets调用](https://github.com/Qesire/FFNDS/blob/e7a1e6b7a6babb5cf742072cc07a9961af854416/reports/20261114/g20_two_track_core/BWS/v1/source/b_common_replay.py#L113-L123)核实真实实现。capture/旧bank中由full-M生成再抽行的Y另存为Y_fullM_panel，不能覆盖或冒充Y_local512。在共同两源上利用两份已存在的Y报告GEMM形状对应的Dense数值差；这不增加Dense调用。两级误差拆分使用同一local512合同。

| 项目 | 有限数量 | 说明 |
|---|---:|---|
| ICC CAL11局部预测格 | 3家族×2秩×2缓存×6位点×11源=792 | 每格为一次完整anchor→hit预测；包含主选择的198格 |
| 主选择子集 | 3家族×r1408×4S×6×11=198 | 是792的子集，不能再加198 |
| 可选CAL22局部预测格 | CA/CD×r1408×4S×6×11=132 | 普通SVD不重复；主家族不因此重选 |
| Q1与Global-P64 P512基线 | 各6×11=66，合计132格 | 身份一致的旧格直接复用；不是再采132条轨迹 |
| 作者Chip/GP64 true-M局部格 | 2方法×6位点×2源=24 | 使用阶段1 full-U；与P512执行身份分开 |
| 原生局部anchor/hit构建 | 2端点×6×11=132次P512 Dense FFN | 若未有完整z/h，用同一输入/权重构建并供各格共享，不逐候选重复 |
| 主ICC误差拆分 | 6×11=66格，最多132次B2 factor action | 仅选定族r1408/4S；F64解释计算另计，非新推理精度臂 |
| 固定数值模块对照 | 4 pair×2缓存×2 FP32/TF32模式=16附加格 | 选定族r1408，两neutral源×两expert×layer19；native BF16 anchor/输入/因子值固定，只换缓存构建及增量算术；主BF16对应格复用792子集 |

四方法同输入机制表取共同的两个neutral来源、六位点和同一P512目标；ICC/Q1直接取各自66格的对应子集，作者Chip/GP64由真实full-M选择与执行后取相同行。方法实际数值路径和工作量仍分别标注，不能用P512时间比较full-M时间。作者Chip两源均值不与ICC/Q1的11源均值直接排行；11源ICC候选选择与两源机制表各自保留。主视频才统一11源。

共同两源表以Y_local512为统一局部目标，同时用已存Y_fullM_panel补充误差归约并报告Dense数值差；不把这个差额单独归因低秩或selector。ICC家族选择始终采用Y_local512，额外归约不参与换主家族。

为避免选择之后重跑预测，198个主选择格首次计算时保留三个家族的实际BF16 ĥ_b/Ŷ_b，并保存132个native端点的h/Y及4个固定数值诊断pair所需的z_a。其BF16张量payload约6.015GiB，临时离线缓存上限7GiB；选定和误差拆分后删除非选家族的大张量，保留完整分数、身份和主家族必要中间量。其它rank/cache格只保留预定统计，不再复制同批anchor。

16格数值对照只检验固定native BF16操作数/anchor/因子值的FP32增量算术：无损升精度后，Δu、Δh、GELU、缓存加回以及2R的anchor投影/减法均按FP32执行；因子仅升精度，不换数学值或重新分解，诊断输出最后cast回BF16，再用同一F64门后评分。绑定实际torch/CUDA、matmul精度与TF32控制；若开/关实际走同一内核，记录未分辨，不要求人为产生差异。该对照不重生成FP32模型的anchor或视频。

这里没有新增全网络NFE。模块小fixture、同mask子块检查、CPU统计及VAE/T5编码按有限任务ID另记实际耗时；局部格也不是独立source数。没有预登记的附加rank、随机panel、参数网格或完整O40回放。

### 7.2 唯一选择准则

对source s、expert e、layer l，定义实际门后误差

\[
R_{s,e,l}(m)=\frac{1}{Pd}\sum_{n\in\mathrm{panel},c}\left[g_{s,e,l}\odot(\hat Y_m-Y_D)\right]_{n,c}^{2},
\]

此式P=512、d=5120；P是评价行数，与完整视频N=31668分开。用F64归约，先层等权、再expert等权、最后source等权。只在三家族r1408四状态的198格上选择平均风险最低者；完全相同则按SVD→CA→CD固定顺序。r512和2R只诊断，不参与主家族/秩选择。全部候选分数保存；缺任一家族不启动选择，不以SVD填补未采CA/CD。

六位点先做3族×2矩阵=36次Wan大矩阵分解；选定族补其余74位点×2=148次，总主预算184。一次分解保存足够的r1408因子，r512取同一前缀。选中族覆盖全部80个expert/layer参数，另外两族保留六位点完整证据。原作者函数的shape验收另有6次分解；Q1缺兼容参数时最多80个固定规则PCG拟合，不搜索lambda/迭代/rank。

CAL22若在近似视频前明确启用，只追加11个真实source并做CA/CD六位点，额外24次分解，累计208；若E5触发，仅补原选中族其余148次，累计最多356。旧rollout因子不能抵扣；同权重普通SVD按真实数值与保存秩复用。主表始终用CAL11。

### 7.3 用同批结果解释误差

对固定当前gate与同一个Dense anchor，记录真实增量d和预测增量d̂的夹角余弦ρ、范数比a及误差：

\[
\frac{\|\hat d-d\|^2}{\|d\|^2}=1+a^2-2a\rho.
\]

它把方向与幅度贡献分开；分母为零或低于预定数值分辨率时记NA，同时保留绝对误差。只利用已有预测，不训练新校正。

对选中ICC，实数算术下有

\[
y_b-\hat y_b=(W_2-B_2)(h_b-h_a)+B_2(h_b-\hat h_b),
\quad \hat h_b=\mathrm{GELU}(z_a+B_1(u_b-u_a)).
\]

理想式第一项描述第二投影未覆盖的隐藏增量，第二项描述第一层近似经GELU产生的偏差。实际BF16记录使用可执行的分解：令A=B2_F64(h_b−h_a)、B=B2_F64(ĥ_b−h_a)，其中ĥ_b直接取主BF16路径，所有实际BF16原值先无损升F64再作差；则E_read=ΔY_native−A，E_first=A−B，E_num=(Y_b−Ŷ_b)_actual−E_read−E_first。每格两次B2 action，共132次（每次两级因子GEMM，共264次）；不额外执行F64 Dense W2。

E_read是**含native Dense数值路径的读出残差**，不能直接称为纯几何的(W2−B2)Δh。三项用相同当前gate记录能量和有符号交叉项，按F64舍入尺度检查总误差闭合；E_num由残差定义，闭合本身不是机制的独立验证。若E_first较大，支持检查第一投影/非线性；若E_read较大，支持检查隐藏增量、读出近似及原生数值路径；交叉项则揭示单项误差与总误差之间的抵消或增强。这些是在固定参数与输入上的结构解释，是否构成新数学规律仍需后续证据，当前不预写结论。

## 8. 必要适配如何验收，哪些进入视频消融

| 适配 | 本轮证据 | 对应视频安排 |
|---|---|---|
| pre-W1/pre-gate、当前gate、bias、expert/request/CFG状态 | 同输入oracle、160-site trace、两源neutral | 不把错误接线当算法消融 |
| flow加噪/真实step、I2V条件、D4时间/专家 | 原生函数及输入身份检查、采样覆盖表 | 不额外造50步反事实 |
| VAE mean/posterior、class-null→empty-text | CAL3两项配对统计诊断 | 无新因子/视频；保留终端影响未测的范围 |
| batch1/source等权、统计精度、scale epsilon/solver | 同值作者oracle及固定小矩阵/逐源统计检查 | 不逐项添加参数网格 |
| voxel往返、无损offload/W2生命周期 | index round-trip、同输入同mask与真实资源记录 | 等价保持由执行证据判定 |
| ICC四状态→两残差 | 同因子r512/r1408局部对照 | **E4必做DEV3三条**，父4S来自主表 |
| Chip尾52：Dense尾→valid-count尾；作者舍入→native舍入 | 相同真实输入/同mask的单因素局部检查 | E6最多选一个合格且对解释有影响的变量 |
| CAL11→22 | 同配方追加源；固定主家族/rank/4S局部比较 | E5按下述条件最多DEV3三条 |

E6在近似臂视频前按固定优先顺序tail→precision选第一项：局部实现合法、与主参考非等价且影响当前适配解释。若没有合格项，E6=0。oracle错误直接修复；不花三条视频裁决错误代码。若两项都需闭环，本轮只覆盖第一项，另一项终端影响保留未决。CAL3敏感性不自动生成全80新因子后挤入E6。

E5仅在已执行CAL22诊断、主家族是CA/CD，并满足以下之一时触发：主ICC相对固定Q1的局部平均风险差在11→22变号且超过预定局部数值包络；或`|R22−R11|≥0.5|R11−RQ|`且超过同量纲包络，并在DEV3至少两源同向。好转、恶化均可触发；保持原家族/rank/4S，不根据CAL22重新挑主臂。SVD被选中或CAL22关闭时E5=0。

## 9. 五臂视频、质量与实际成本

主55槽为5方法×11来源×seed42。阶段1的两条hook-Dense及合法旧产物按真实(source, seed, path, execution_id)抵扣；同一输出可以服务多个分析，但不存在的执行路径不能抵扣。任何自由视频都从其自身前序状态继续，teacher同输入回放不替代自由生成。

记录每源的I2V主体/背景保持、时序主体/背景一致性、imaging/MUSIQ、美学、运动连续性和连续运动幅度，沿已绑定指标实现、帧采样与分辨率；质量缺项记缺失。相对Dense的运动幅值在Dense分母过小时不稳定，报告绝对量与NA；平滑不直接等于动作保持。对DEV3的五臂作相同时间位置接触表与连续播放，记录具体动作、主体形态、背景/光照、相机及闪烁现象；这是解释性观察，不能从少量帧宣称通用质量优胜。

以source/近重复组为分析单位，逐源配对报告、等权均值、范围和leave-one-source-out。token、层、帧和多个算法不是独立样本。DEV实际若少于11个独立簇，报告真实簇数；不补换输出不佳来源。未事先给质量非劣margin时，不宣布质量非劣，只给各轴差值与取舍。

成本同时包括：anchor/hit/fixed-dense实际次数，W1/W2及低秩/选择器/proxy/voxel/缓存搬运，单次生成墙钟，CPU/GPU峰值，离线校准与分解。首个有效任务就是正式记录，不另开额外warmup视频。主视频默认无dense-shadow等重observer；采集hook的时间不充当plain速度。若旧视频时间不可比，只有两条新plain与近似臂能配对，就只报告这两来源速度。微小差异记未分辨，不追加60次计时。

## 10. 总预算与资源

| 类别 | 主路径 | 可选全部触发后的上限 |
|---|---:|---:|
| 作者式主校准 | 66 NFE | 132 NFE（22源累计） |
| CAL3必要适配诊断 | ≤30 NFE | 同一份≤30，不因扩22重做 |
| 校准及诊断合计 | **≤96 NFE** | **≤162 NFE** |
| 主五臂视频 | 55 D4槽 | 55 |
| 额外neutral | ≤10 D4 | ≤10 |
| E4同因子2R | 3 D4 | 3 |
| E5、E6 | 0 | 各≤3 D4 |
| 完整视频合计，重试前 | **≤68 D4** | **≤74 D4** |
| Wan大矩阵分解 | ≤184 | CAL22局部208；含E5全80最多356 |
| ICC局部预测格 | ≤792 | 加CAL22的132，最多924 |
| 其它局部工作 | Q1/GP64 132格、true-M两Chip24格、原生P512构建132次、主ICC B2 action132次、固定数值模块16格 | 固定同一清单，不自动加网格 |

所有视频、校准和适配诊断共享**最多3次技术重试**。视频重试≤4 NFE；校准/诊断重试的单位为CA单调用或CD不可分两调用pair，≤2 NFE。失败统计事务先回滚，沿同样输入/RNG重试，不重跑整source六调用；质量差不触发换seed/来源。

令`r_video+r_cal≤3`，所有条件打开时

\[
V\le74+r_{video},\quad C\le162+2r_{cal},\quad
NFE_{video+cal}\le4(74+r_{video})+162+2r_{cal}\le470.
\]

完整视频最高77只在三次重试都用于视频时达到。470不含局部FFN、模块检查、VAE/T5、分解/PCG、搬运与质量评分，也不是等墙钟预算。此前440上界缺少这次明列的30次必要适配诊断，现由470替代。旧99/897条方案均无活动额度。

拟用单任务A10080GB，真实设备、并发上限、host内存与临时盘须从当前会话绑定，不由历史路径推定。ICC单expert40层四状态逻辑payload约89.395GiB；作者Chip38层h/y/proxy/indices缓存约43.2GiB，均不是实测峰值。ICC须有无损host/临时存储及有界staging；一次只保留一个source/expert的CD首调用缓存，按层读取释放。作者W2转置按活动层物化/释放，不能常驻80份额外副本。

第一份有效校准pair、模块核验和正式分解用于记录真实峰值/耗时并预估剩余固定任务，不另行重复pilot。资源不足时保留已完成产物与明确缺项；不通过P512替代全token统计、2R替代4S、降rank或近似SVD来伪装完成原参考。永久保存充分的因子、逐源统计和数值身份；临时大激活及时释放，避免保存22源全量中间张量。

## 11. 旧证据、Wan特点与蒸馏解释

| 已有证据 | 本轮保留的认识 | 对新参考的边界 |
|---|---|---|
| A固定新算子比较 | 既定Q1残差、已测候选的改善/退化及执行成本 | 不加入本轮主臂，不据此默认重拟合Q1 |
| BW | 旧SVD-r512-2R与GP64在固定状态、own-W目标/拟合下的权重替换影响 | 不自动迁为新CA/CD-r1408或作者Chip的结论 |
| BWS | 混合CAL3、既定D4 gate下权重与复合状态的交互；旧配置中状态切换影响大于平均权重替换影响 | 不与BW绝对风险拼表；状态同时含时间/轨迹/输入分布变化，不能单归因步长 |
| 当前新局部与视频 | 规范化参考在本Wan部署域的误差、支持行为、缓存与质量/成本 | 不能推断原模型/原论文系统、未跑家族或所有视频FFN的总体结果 |

由这些结果讨论视频FFN时，重点看同一token坐标跨去噪时刻的增量、GELU两级作用、空间分组支持和当前gate后的写入。与LLM的比较限于计算图、可复用状态、token组织和硬件工作点的理论/文献对照；本轮没有LLM对照实验，不作跨域性能结论。

关于蒸馏，先把“训练后的权重变化”与“实际求解/状态分布变化”分开陈述，再明确BW/BWS各自真正固定了什么。若新参考的关键机制仍需原/蒸馏模型配对才能辨别，在最终报告中给出这一未决问题与最小后续反事实；本轮新O40=0，不从一个相关发现自动扩成新原模型生成队列。

## 12. 完成标准与移交

最终交付四张结果表和一份解释：①原算法/必要适配验收；②三家族与缓存/秩局部风险及两级误差；③五臂逐源质量；④离线与在线成本；解释稿把观察对应到竞争假说，保留失败、缺失与未决范围。

执行器只维护一份实施记录与产物manifest，关联actual producer、依赖、输入/参数/配置、运行日志、exit和输出；无需给每次阅读、统计或小检查建立新的审批层。按仓库已有流程对真实代码/数据/资源作独立执行前复核并登记有限ACTIVE，再推进已授权研究目标。设计审阅本身不是运行验收。

当前立即任务依次为：**绑定11条真实clean输入与CAL3；完成作者模块及Wan collector/adapter；将两源neutral/full-U捕获落实为主Dense槽；执行66+≤30校准与诊断；完成局部选择和全80参数；再生成剩余主视频及有限消融。** 截至本文发布，这些新的模型任务均未执行，旧A/BW/BWS继续保持已完成身份。
