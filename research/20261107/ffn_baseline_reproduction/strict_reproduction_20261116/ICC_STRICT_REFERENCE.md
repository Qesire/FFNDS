# ICC 严格 reference、必要适配消融与有限重做规格

日期：2026-11-16。状态：**STATIC_DESIGN / NO_MODEL_RUN / NO_GPU_ACCEPTANCE**。本轮环境没有已确认的 GPU、torch 或集群执行入口；下文是可冻结的实施规格，不是已通过的运行记录，也不声称当前拥有一张或两张 A100。

依据用户最新优先级：完整、正确复现优先；原始四状态 ICC 必须作为主 reference，不能为省内存用两残差替代后仍称原法。旧 scope 中“只做 SVD512/2R、不做作者复现、不扩校准”的范围描述保留其历史效力，不限制本次新目标。新执行仍应绑定实际代码、数据、资源与项目要求的独立复审；设计工作可以继续，无须为旧范围再次询问用户是否要完成已明确要求的复现。

**预算与执行唯一入口：** 本专项只定义ICC语义与验收，所有执行并入 [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md)。旧CAL256、七个完整DEV点、九臂EVAL22×双seed均已撤销；ICC主11真实视频作者式校准66次，可选22累计132；另在其中3源做必要VAE/文本配对诊断至多30次，主/扩展合计96/162。三家族r1408局部选择一个，主视频55条；原99条D4总账已按校准NFE/视频分开。固定源码逐项引述见 [ICC_CODE_CITATIONS.md](ICC_CODE_CITATIONS.md)。

固定来源：ICC CVPR2025 / arXiv:2505.05829v1；`ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。既有 FFNDS 证据基线 `e7a1e6b7a6babb5cf742072cc07a9961af854416`。具体已核公式、源文件行号、旧结果边界见 [此前论文与实现审计](../../../20261116/ICC_CHIPMUNK_WAN_FIDELITY_AND_FFN_STRUCTURE_20261116.md)。

## 1. 完成的含义：原法、移植、变体分开验收

必须分别交付以下记录，不用一个“复现通过”标签覆盖所有层次：

1. **作者算子reference通过**：直接原SpMlp、统计和分解函数与端口的同输入正对照；作者原形状/精度、四状态和所有有效分支完整。原DiT/FLUX整模型、FID不作为本轮前置。
2. **Wan原生接口通过**：native、hook-bypass、ICC forced-dense在真实逐步latent一致；全部80站点因子/权重/专家匹配，160次实际调用无误。
3. **三家族实现完整**：SVD、CA、CD全部有原函数oracle、四边界统计和六站点同输入风险；按固定规则选定一家构造全80处因子进入主视频。未选择家族的全模型质量仍未测。
4. **适配证据明确**：必要接线/缓存隔离/搬运先用模块和neutral；4S→2R做同因子DEV3闭环；校准与数值适配先局部。没有对应闭环的分支不声明视频级等价。
5. **方法比较完成**：五臂DEV11×seed42的开发对照及实际资源记录。与局部选择使用同源，不能叫确认性留出。完整复现不以Q1获胜为条件。

若四状态有效、两残差明显劣化，旧2R结果属于该变体的负结果，应撤回对ICC的推广。若某一CA/CD配置改善，必须以修正后的强基线更新Q1比较。若正确四状态在已测试全部有限配置下仍差，结论也只针对这些Wan工作点，不宣称ICC不存在更好rank/调度。

## 2. 主 reference 的严格算子合同

采用列向量，\(z_s=W_1u_s+b_1\)、\(h_s=\phi(z_s)\)、\(y_s=W_2h_s+b_2\)。anchor缓存 **u_s,z_s,h_s,y_s四个未压缩张量**，命中执行

\[
\widehat z_t=z_s+B_1(u_t-u_s),\qquad
\widehat h_t=\phi(\widehat z_t),\qquad
\widehat y_t=y_s+B_2(\widehat h_t-h_s).
\]

- 首级缓存是pre-GELU z，不是h；第二级输入是post-GELU h；y是pre-gate。保留W1/W2偏置与tanh-GELU。增量分支不另加bias。
- hit不更新anchor四态。禁止递推更新近似状态、偷换teacher-anchor或把y替换为另一轨迹的Dense输出。
- 当前norm/shift/scale由宿主产生u；返回y后由宿主**当前gate**乘一次并加残差。gate不得用于改变原ICC因子或CD左右scale；gate可以用于单列的评价指标。
- 因子采用作者默认 `left=U_r Σ_r`、`right=V_r^T`，先右乘再左乘。换成对称平方根分配虽实数等价，也改变BF16数值路径，不能静默替换。
- 每个 `(request,expert,layer)` 分别缓存；高/低专家因子与权重严格绑定。四步40层应有160次FFN调用：80 anchor、80 hit。缺cache、错shape、错expert直接报错，不能静默回退dense并继续当既定方法评分。
- anchor优先沿原生FFN路径读取真实u/z/h/y，返回原生y，避免用另一个重写FFN的数值差异混入reference。

固定作者代码依据：`DiT/models/models_sp_step_decomp.py` 69–109（算子）、238–239（GELU）、268–283（当前调制/gate）、`DiT/adapters/sched_sp_step.py` 9–22（p2）。

### 调度与作用范围

作者ICC图像主实验p2是全层anchor/hit交替，MHSA和FFN同步校正。当前任务主表明确命名 **ICC-FFN/Wan-D4-HHLL**，只迁移FFN，并保持原生attention。

Wan D4每个专家只有两次调用：0/1 high，2/3 low。主reference用专家内A/H，跨专家强制重新anchor。p1为native/neutral对照；p≥3在每专家两步内与p2相同，或会错误跨专家复用，不构造虚假的period扫描。保留最后一步hit，因为作者ICC没有强制末步dense。

ICC与Q1的D4调用呈A/H；Chip保留其period10并在专家首次调用建立cache，在D4也呈A/H，但首2层仍Dense。若另设共同p2或共同层范围，它是**共同Wan预算对照**；其它默认warmup、refresh、作用层数应另列，不重命名为各自原作者默认。若未来增加“低专家最后一步dense”等策略，它是新的覆盖率/调度臂，不可替代这里的原法主reference。

## 3. 必须覆盖SVD、CA-SVD、CD-SVD

三者共享四状态在线路径，区别只在离线B1/B2来源。

| 家族 | 四个统计边界 | 统计信号 | 主reference |
|---|---|---|---|
| SVD | 无校准依赖 | 原始W1/W2 | 原始权重的截断SVD |
| CA-SVD | u、z、h、y | 每边界绝对激活幅值 | 左右双侧缩放后截断SVD |
| CD-SVD | u、z、h、y | 同专家真实配对端点的激活差幅值 | 左右双侧缩放后截断SVD |

不能以SVD代表CD，也不能只做CD而跳过CA。作者论文并未发现三家族在所有情形下同一方法最佳。

### 作者统计流程必须逐步对应

对每个sample、每个边界：

1. CA：`abs(x).mean(token)`；CD：`abs(x_hit-x_anchor).mean(token)`。不是RMS、不是只取pre-W1、不是只取某个token均值之差。
2. 在该sample的通道维取min/max，计算 `(mag-min+1e-6)/(max-min+1e-6)`。常量通道幅值的特例得到全1，不应错成全0。
3. sample/source等权平均；所有前缀使用相同source排序。目标视频可用batch=1，避免不同大小尾batch被按batch等权误加权或drop_last丢数据。
4. 累计平均后取平方根。
5. 左右scale分别显式 `clamp_min(max/1000)`；保存原始幅值、归一化累计值、sqrt后值和clamp后值的identity。
6. 对scale再加作者的1e-6，分解 \(LWR\)，撤销左右缩放。epsilon只按一次分解定义添加，不能在多rank循环里隐式累计原作者原地加法的副作用。
7. W1右/左scale来自u/z；W2右/左scale来自h/y。y始终pre-gate。不得把gate权重、相对方差或Q1目标偷偷并入“作者CD”。

依据：作者 `models_stat_svd.py` 14–45、85–95；`func_decomp.py` 78–101、136–173；`utils_decomp.py` 44–84；`gen_decomp.py` 158–177。现有FFNDS `weights(max_ratio=None)` 默认不足以证明作者配置，因此新producer必须显式传1000并记录。

### 采样方法遵循作者，数量沿现有量级

主校准必须使用真实clean视频编码后前向加噪。CA在合法时间均匀抽样后单次Dense调用；CD从加噪起点第一次Dense调用，执行真实native step，再作同专家第二次Dense调用。旧native-D4轨迹端点不再作为主采集。完整配方及代码对应见[ICC_AUTHOR_CALIBRATION.md](ICC_AUTHOR_CALIBRATION.md)。

Wan使用原生归一化mu latent、flow加噪/一步、目标D4 grid。CA在H{0,1}/L{2,3}随机取时刻；CD仅H0→1/L2→3，每pair条件不变。保留作者10%条件dropout并映射为空文本T5编码，I2V图像/mask保留，不做CFG双前向。真实clip、首帧条件和完整目标latent必须分开；首帧零填充是条件，不是clean视频。

主ICC_CLEAN_CAL11为11source，最多66 NFE；可选22source累计132。batch1/epoch1，不丢尾批，前缀不重抽。当前仅有VBench图文/FFN面板，真实clip/x0尚需绑定；见[资产审计](ICC_CALIBRATION_ASSET_AUDIT.md)。不能用生成末端latent、静止复制或旧rollout替代主真实视频数据。

## 4. 精度必须拆成推理、统计、分解三个轴

作者DiT报告TF32计算、FP32张量；PixArt报告FP16。Wan原生BF16是宿主数值合同，不能只贴“原文精度”标签掩盖区别。

### 主路径与必须对照

- **作者参考验收**：在固定upstream的原生dtype/TF32设定，使用同一真实操作数和同一因子执行原SpMlp与我们的四状态reference。统计函数用原代码直接返回值作oracle。至少包含两个非零bias、GELU正负区间、零增量、真实层/专家操作数。
- **Wan主路径**：原生BF16 u/z/h/y、BF16 inference factors、FP32 SVD。统计主reference使用FP32统计算术，输入是实际native BF16值的无损转型，并与上游函数在这些FP32输入上的输出比较；这是显式的校准数值约定，不把它混同于BF16中先做差再转型。
- **统计精度消融**：同一记录的BF16源值，比较FP32统计与按BF16自然计算的统计，固定其余项。在既定模块oracle和捕获的逐源统计上比较scale；本轮不为此重新分解或新增factor-action/风险网格。完整视频影响仍未测。
- **在线精度消融**：固定选中族r1408、两个neutral来源×两expert×layer19共4个pair，固定native BF16 anchor、输入和因子数值；仅缓存构建/增量算术转FP32，4S/2R各测TF32开/关，最多16个附加模块pair。主BF16对应格已包含在792局部格中。记录cast边界，目标仍同一native BF16；这检验被指定算术的敏感性，不是整个FP32-FFN部署臂或新选参候选。
- **两残差消融**：必须使用与四状态完全相同的因子、输入、调度和dtype，仅换缓存代数。FP32与BF16分别读出差异。不能通过更换rank、stats或因子精度给2R补偿后仍称单因素消融。

作者原shape/dtype的模块正对照与Wan同输入算子精度对照必须做。本轮不新增原模型完整系统或FP32-FFN整视频网格；局部精度差异不能证明其视频影响为零。条件适配闭环仅允许一个已定位单变量、DEV3三条，未评价分支保持UNRESOLVED。

### 验收oracle，避免以质量代替正确性

1. 同dtype、相同操作顺序、shape/layout/backend的作者reference与端口应逐中间张量bitwise一致。若upstream自身在固定环境下非确定，先定位/记录该非确定性；不能事后放宽到“视频看起来差不多”。
2. 小规模FP64解析例覆盖非零bias、零增量、常量统计、跨expert错误与缓存生命周期；代数误差采用预先写死的数值容差，不以科研效果作为PASS。
3. zero-delta四状态应回到原anchor；两残差若不为0，作为实测数值消融保留，不修改reference掩盖。
4. `rank=full` 的因子化在有限精度下也不一定逐位等于原dense GEMM，不能将它设成虚假的bitwise oracle。
5. Wan neutral至少2个既定来源，比较160次门前/门后边界及最终latent；MP4容器字节可受编码元数据影响，neutral主oracle用latent与解码帧，MP4 SHA作产物身份。

## 5. 四状态内存方案：保留算法，再选择实现

N=31668,d=5120,m=13824、BF16、一个专家40层的缓存同时驻留：

- 四状态：89.394836 GiB；每层2.234871 GiB。
- 两残差：44.697418 GiB；**是独立变体，不是四状态资源不足时的reference回退**。
- 以上不含模型、因子、attention/GELU临时区和allocator峰值。FP32四状态再翻倍，不能默认192GiB主存一定足够。

### 可接受的四状态执行路径

**路径A：单GPU＋无损host cache。** 将四个BF16状态按层存入host，hit按需预取并消耗释放；传输只复制原始数值，不量化、不改cache内容。先使用最简单可审的同步reference，验证后增加有界双缓冲/异步预取，并再次检查neutral与算子action。两专家四步总逻辑cache D2H+H2D约357.579GiB，必须计时，不能视为免费。pinned memory只保留有限活动缓冲，不盲目锁住全部90GiB。主存是否同时容纳模型offload、89.4GiB cache和工作区必须实测预检。

**路径B：后续资源备选，不与本轮默认单GPU/host路径同时实现或计时。** 经实际核实可采用两GPU分层， 例如每GPU保留20层的cache约44.7GiB，绑定层权重和cross-device hidden传输；inactive expert按既定策略卸载。理论空间估算不等于设备可用或一定能装下。若用两卡作速度比较，Dense/Q1/其它臂必须有同等资源背景，或明确这只是“使原法可执行”的资源成本点，不能与单卡速度作无条件胜负。

更细的逐层host预取也可用，前提是四状态数值和读取顺序一致。任一路径失败就保留 `RESOURCE_BLOCKED_REFERENCE` 与失败资源账，不把2R结果冒充reference。不得按六层成功宣称全40层运行完成。

### 校准统计不必永久保存四状态全张量

按作者式CA单次/CD相邻两次调用生成四边界统计，保留输入真实clip解码、VAE编码和I2V编码，无输出VAE解码或生成视频编码。可以临时host保存anchor四态，hit算完统计立即释放。

本轮使用直接保存首调用四边界、第二调用逐层消费的路径。只存u再重算z/h/y会增加本预算之外的full-M FFN工作，保留为后续资源变体，本轮不自动启用；当前统计也不以P512替代全token。

## 6. 固定rank与周期预算

三个家族主rank统一1408，r1=r2；不做独立r1×r2、逐层rank或r512/768/1024视频搜索。r512只是同一分解前缀下的历史局部参照。四状态的hit/dense主MAC为2r(d+m)/(2dm)，完整A/H周期则为(F+L)/(2F)，不能只按hit成本宣称公平。

| 配置 | 完整A/H周期主MAC / Dense |
|---|---:|
| ICC4S-r512历史参照 | 0.568519 |
| ICC2R-r512历史执行 | 0.637037 |
| Q1r2 | 0.685185 |
| ICC4S-r1408主预算 | 0.688426 |

r1408约比Q1多0.473%主MAC，是对齐后的近邻预算，不是作者原建议值或实测等时延。三家族按六站点DEV11固定局部规则选一个进入主视频；所有局部结果保留，未选家族全视频质量不外推。实际选点和离线分解数量只采用统一方案。

## 7. 校准与选点：主CAL11、预算消融CAL22

主参数在任何新增近似臂自由视频结果可见之前固定ICC_CLEAN_CAL11；方法无关Dense/neutral可以先行，产物计入主55。CA/CD保留四边界、有效全token、样本内归一化后来源等权、sqrt与max_ratio1000；两者不能以P512或只含u的旧bank替代。原生VAE、flow、D4时间grid、条件dropout映射及统计精度均在因子清单记录。

只有同一作者式配方与真实输入身份均匹配的统计才可复用。主11源最多66、可选22累计132个单模型调用；不采完整CAL轨迹，旧rollout/P512不补造缺失的作者式统计。不能把已有512行面板称为原全token统计。数据项与图像簇身份、fit祖先、源码与随机性均须绑定。

三个家族固定r1408，在统一输入manifest的六站点×DEV11×同P512上评分；两个neutral来源以新full-U capture的对应panel为准，其余九源使用兼容旧bank；按来源等权门后绝对MSE选一家，严格同分按SVD→CA→CD。r512仅从同次分解取前缀作历史诊断；不新增r768/1024视频网格。选定家族全80站点导出，其余家族不重复构造全模型参数，不因此省掉算法实现或冒称它们的视频已测。

可选ICC_CLEAN_CAL22只做相同作者式配方的预算干预；SVD复用同一因子作无校准负对照。若局部出现预登记的排序/风险差改变，才给**原选定家族**、同rank/4S增加DEV3的CAL22闭环；不换Q1主参数、不改主表。若仍不稳定，本轮记未解决，不自动扩64/256。

## 8. 两残差、有限视频与成本

主ICC始终四状态，r1408完整周期主MAC比0.688426，Q1为0.685185；不是等时延结论。两残差同因子在r512/r1408均做局部检查，选定主因子仅补DEV3闭环三条，4S父结果复用主视频。两残差增加anchor投影，不能用其hit成本代表整个周期。

五臂主视频55槽位保持；作者式校准66/132 NFE与CAL3必要诊断≤30分别列账，合计96/162。完整视频基本68、条件全开74，与校准/诊断共享3次技术重试（全用于视频时77），两类去噪器调用联合上界470。唯一计数为统一方案及 [EXPERIMENT_BUDGET.json](EXPERIMENT_BUDGET.json)，没有第二ICC队列。当前没有GPU环境或新模型运行证据。

四状态约89.4GiB的单专家BF16缓存需真实验证host offload/staging容量和生命周期；硬件、RSS、pinned、GPU allocated/reserved、传输及工作区以实际测量记。资源不够报告未完成，不能以2R替代四态主身份。参数分解、统计和模块回放也单独记录耗时，不能因D4次数少而把离线成本隐去。

## 9. 实际执行证据与复用

每个因子清单记录：固定代码与producer、家族/rank、四边界scale来源、min-max/sqrt/ratio/epsilon、SVD driver/full_matrices/TF32、因子拆分与cast边界、checkpoint/expert/layer、CAL数据角色和哈希。每次运行绑定真实输入/初噪、调度、缓存状态、gate、退出码及产物。代码里存在函数、规范写了1000，都不等于运行用过。

原A/BW/BWS保持完成：普通SVD512双残差与Global-P64结果继续按实际身份引用。旧CA/CD或author-style视频缺producer时标未核，不覆盖原FFN模块与Wan严格端口的待办。相同身份的Dense、Q1、统计和neutral逐条复用；更改CAL来源后不能借用旧参数视频充当新arm。ICC若绑定新真实视频而Q1保留旧图文CAL11，须披露各自校准协议/数据，不能称逐源配平或将差异独立归因算法。

本轮局部家族选择与主视频同源，结论是开发对照。完整方法、校准分布、精度、全模型质量与实际时延分别按已完成证据下判断；未选家族、未跑适配和原论文整系统保持相应未测状态。
