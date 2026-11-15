# ICC 严格 reference、必要适配消融与有限重做规格

日期：2026-11-16。状态：**STATIC_DESIGN / NO_MODEL_RUN / NO_GPU_ACCEPTANCE**。本轮环境没有已确认的 GPU、torch 或集群执行入口；下文是可冻结的实施规格，不是已通过的运行记录，也不声称当前拥有一张或两张 A100。

依据用户最新优先级：完整、正确复现优先；原始四状态 ICC 必须作为主 reference，不能为省内存用两残差替代后仍称原法。旧 scope 中“只做 SVD512/2R、不做作者复现、不扩校准”的范围描述保留其历史效力，不限制本次新目标。新执行仍应绑定实际代码、数据、资源与项目要求的独立复审；设计工作可以继续，无须为旧范围再次询问用户是否要完成已明确要求的复现。

**预算与执行唯一入口：本文件是ICC方法与正确性专项，不是第二执行队列。所有样本数、候选臂、消融与计数最终并入主代理的 [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) 单一总账；本专项早期7臂/724次的独立预算建议已经撤销，不与合并预算叠加。统一EVAL为22 source×2seed×最多9臂，即396条轨迹，重复臂按真实身份扣重。**

固定来源：ICC CVPR2025 / arXiv:2505.05829v1；`ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。既有 FFNDS 证据基线 `e7a1e6b7a6babb5cf742072cc07a9961af854416`。具体已核公式、源文件行号、旧结果边界见 [此前论文与实现审计](../../../20261116/ICC_CHIPMUNK_WAN_FIDELITY_AND_FFN_STRUCTURE_20261116.md)。

## 1. 完成的含义：原法、移植、变体分开验收

必须分别交付以下记录，不用一个“复现通过”标签覆盖所有层次：

1. **作者算子 reference 通过**：固定 upstream 的 `SpMlp.forward_upd/forward_inc`、统计和分解，在原始数据类型及相同真实操作数上与我们的 reference 对齐；作者原工程至少一次完整去噪执行成功。小规模运行只证明工程/算子闭环，不冒称复现其50k图像FID主表。
2. **Wan 原生接口通过**：同 checkpoint、输入、初始噪声、scheduler、attention backend，native、wrapped-native、forced-dense ICC-wrapper 的层输出与最终latent一致；请求与专家缓存生命周期正确。
3. **Wan 四状态三家族完成**：SVD、CA-SVD、CD-SVD 均有完整80站点因子、真实四状态自由轨迹视频及原始指标/成本。这里的“完整”是当前Wan任务的全深度执行，不是只在六个站点回放。
4. **适配消融完成**：两残差、小校准、数值精度分别与四状态/充分校准 reference 对照；只有对应数据才能解释旧结果。
5. **方法比较完成**：有限候选在选择集冻结，独立source集合比较；保留失败、较差和资源受限点。完整复现不以Q1获胜为成功条件。

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

### 目标域采样属于必要迁移，预算属于单独选择

作者用256个图像，随机采原图像扩散时间，再以真实相邻去噪调用构造CD。Wan主校准使用当前真实D4、同专家0→1与2→3，不能对不同expert权重的输出做“时间增量”。CA在每expert两个合法端点等权累积，CD在每expert合法端点对累积，均在native Dense轨迹采集。

这些是迁移到目标推理分布的统计定义；**256个目标视频条件与256个原论文图像不等于同成本或同任务**。应把两者各自的source数、模型调用、token数、耗时列出。

## 4. 精度必须拆成推理、统计、分解三个轴

作者DiT报告TF32计算、FP32张量；PixArt报告FP16。Wan原生BF16是宿主数值合同，不能只贴“原文精度”标签掩盖区别。

### 主路径与必须对照

- **作者参考验收**：在固定upstream的原生dtype/TF32设定，使用同一真实操作数和同一因子执行原SpMlp与我们的四状态reference。统计函数用原代码直接返回值作oracle。至少包含两个非零bias、GELU正负区间、零增量、真实层/专家操作数。
- **Wan主路径**：原生BF16 u/z/h/y、BF16 inference factors、FP32 SVD。统计主reference使用FP32统计算术，输入是实际native BF16值的无损转型，并与上游函数在这些FP32输入上的输出比较；这是显式的校准数值约定，不把它混同于BF16中先做差再转型。
- **统计精度消融**：同一记录的BF16源值，比较FP32统计与按BF16自然计算的统计，固定其余项。先在六个预登记站点比较scale、因子action及heldout风险；如要声称对完整视频无影响，必须另有完整轨迹证据。
- **在线精度消融**：四状态 BF16 与四状态FP32/TF32计算在相同输入、相同原权重值、相同数学因子下比较，显式定义cast边界；同时保留pure-FP32/TF32关闭的数值参考。不能把FP32自己target与BF16自己target混成一个“精度改善”数。
- **两残差消融**：必须使用与四状态完全相同的因子、输入、调度和dtype，仅换缓存代数。FP32与BF16分别读出差异。不能通过更换rank、stats或因子精度给2R补偿后仍称单因素消融。

原dtype的完整作者工程运行与Wan同输入算子精度对照是必做验收。**本轮统一预算未采纳完整Wan FP32-FFN自由轨迹。** 以下仅是未来另行决定的可选诊断，不自动进入当前队列：2个来源×1seed，运行 `Dense_FP32_FFN` 与 `ICC4S_FP32_FFN` 两个额外臂，FFN出口写回宿主BF16；它们相对已有BF16 Dense/ICC构成2×2精度对照，共4次D4。此臂改变FFN算术，需独立标识、不能替代BF16主reference。若资源不满足，不谎报完成，也不从局部对照宣布全视频精度影响已排除。

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

**路径B：经实际核实的两GPU分层。** 例如每GPU保留20层的cache约44.7GiB，绑定层权重和cross-device hidden传输；inactive expert按既定策略卸载。理论空间估算不等于设备可用或一定能装下。若用两卡作速度比较，Dense/Q1/其它臂必须有同等资源背景，或明确这只是“使原法可执行”的资源成本点，不能与单卡速度作无条件胜负。

更细的逐层host预取也可用，前提是四状态数值和读取顺序一致。任一路径失败就保留 `RESOURCE_BLOCKED_REFERENCE` 与失败资源账，不把2R结果冒充reference。不得按六层成功宣称全40层运行完成。

### 校准统计不必永久保存四状态全张量

一次native校准轨迹同时流式生成CA/CD四边界统计，无VAE解码、无视频编码。可以临时host保存anchor四态，hit算完统计立即释放。

另一种更省主存的统计producer只保存anchor u，并在相同完整N形状、原生dtype/backend下重算anchor z/h/y；先对一个已计入校准预算的来源验证重算与原始四边界一致。它增加校准计算但不改变在线reference；必须计成本。不得用512行重算代替完整N再宣称full-token统计等价。

## 6. 有限rank集合与公平选择

主reference家族为SVD/CA/CD，统一 `r1=r2`，不进行80层各自调rank，也不同时扫独立r1×r2。主离线集合：

\[
\{\mathrm{SVD},\mathrm{CA},\mathrm{CD}\}\times\{512,768,1024\}.
\]

额外登记 **CD-r1408四状态** 作为接近Q1完整周期MAC的强补点，共10个主候选。r1408不是作者建议值；它的作用是避免“只给ICC更低计算量，再宣布Q1更精确”的比较偏差。

| rank | Wan hit/dense | 四状态完整A/H周期比 | 解释 |
|---:|---:|---:|---|
| 512 | 0.137037 | 0.568519 | 历史点；接近作者DiT-r128相对hit预算 |
| 768 | 0.205556 | 0.602778 | 接近作者DiT-r192预算 |
| 1024 | 0.274074 | 0.637037 | 接近作者DiT-r256预算 |
| 1408 | 0.376852 | 0.688426 | CD补点；Q1完整周期为0.685185，主MAC差约0.47% |

因子分解按同一校准权重一次求到最大rank，再按前缀截断。不同rank不重复估计统计、不重新调max_ratio；CA/CD不同校准前缀仍是不同factor identity。所有family/rank点均公布，不只留胜者。

### 候选选择只发生在DEV

先在所有80站点、统一512真实配对行的DEV11局部回放中评价10点，报告source等权风险、high/low、每层异常和数值差；这一回放是候选筛选，不能替代完整N视频。

完整DEV视频固定保留三家族r512；SVD和CA各加768/1024中局部风险较低的一点；CD加768/1024中较低一点并保留r1408。最多7个四状态候选，全部在相同DEV11×seed42上运行。每家族最终选一配置用于后续EVAL；CD1408预算匹配点即使不是该家族质量最优，也另列到EVAL。不可看EVAL再改rank、统计、dtype、周期或层范围。

选择指标在读取这些视频结果前冻结，采用共同项目的I2V主体保持及运动/质量护栏规则；不要事后按最有利指标换赢家。若尚无合法非劣/护栏阈值，报告完整指标向量，并事先固定选择的主指标及来源适用集合；不能临时编造5%之类阈值。局部最佳与视频最佳不一致本身须报告。

主表分为“作者机制的忠实迁移点”“固定参数历史点”“主MAC近似匹配点”；有限10点不等于搜索到全局Pareto最优。

## 7. 校准数量：256参考预算，小校准是消融

为避免继续以CAL4/11的小样本结论否定作者方法，主参考使用**256个经去重、角色绑定的目标域source条件**，每源一个固定生成seed、一个真实native D4轨迹。256是与作者图像统计数量相对应的参考预算，不声称二者计算量、统计有效样本量或数据分布相同。

预先形成嵌套前缀11/22/64/256；若既有CAL11/预留CAL_B11身份有效，保留为前22，不把旧DEV转为CAL。其余按输入信息选择，不能按任何方法结果选“容易校准”的来源。所有统计在一次256轨迹采集内流式累积，可得到四个前缀，不需要重复模型前向。

- **主统计**：CAL256全token、四边界、每source等权。视频内大量相关token或多个seed不能代替256独立来源。
- **全80局部校准消融**：CA/CD比较CAL22与CAL256的统计、factor action和局部风险，固定数值/选择参数。本轮CAL22完整DEV视频只运行CD（同选定rank）与Q1两臂，共22条，父CAL256视频复用；CA不另增加完整视频，不能声称已排除CA闭环对校准数量的敏感性。
- **11/64前缀**：先在预登记六sentinel站点统计scale稳定性、subspace/action和局部风险曲线；若仅做到六站点，明确不能据此断言全80校准已饱和。
- **Q1公平访问**：同一个CAL256名单与源权重、同一真实端点和512行面板均可供Q1拟合；不把新版ICC的256来源与旧Q1的11来源不加说明拼表。共同CAL256 Q1作为新主比较器，历史Q1作为独立冻结参考。Q1自己的正则/求解配置冻结，不在DEV额外调参。

保留每source四边界归一化向量，CA/CD合计约5.8GiB量级（不含格式开销），即可复算前缀；无需永久保留full-M z/h。若同时保存全80、每source512行的U0/U1/Y0/Y1供Q1与后续局部分析，BF16 payload上界约400GiB/256来源，必须明确申请和核对存储，不能称“只存几个统计向量”掩盖这一资产成本。已有正确面板尽量复用，但full-token统计不能由不充分面板补造。

若实际资源只能完成CAL22，结论必须写“小校准适配”；不能把它当充分完成的256参考，也不能由其劣化推广原方法。无须无限扩到512/1024校准来源：256后没有自动增加的分支。

## 8. 选择集与独立source评价

建议复用现有DEV11作开发选择集，固定新EVAL22（11个语义类别各尽量2个经验证独立source）与2个嵌套seed。所有同图、近重复图、裁剪、提示变体和seed属于同一个source角色。EVAL必须与CAL256、DEV11及各方法所有拟合祖先不交叠；不能把更换seed当新独立来源。

旧cohort记录含历史暴露声明，但应沿source和历史评价manifest核对具体暴露，不因见过数据集名或元数据就声称全部结果已暴露，也不因新建EVAL标签就恢复历史盲性。若仅满足新的拟合/选择隔离，称“冻结后source-out评价”；只有确有记录证明尚未观察方法结果，才称盲确认。没有必要机械要求换外部数据库。

EVAL22×2seed足够产生可复核的44条配对轨迹/臂和22个source统计量，**不预先保证足以证明非劣、显著优胜或覆盖所有失败尾部**。先源内平均两个seed，再source等权；保留所有原始配对差、类别覆盖、置信区间与失效案例。某一指标只适用于部分类别时，其独立n相应减少，不把22硬套给所有指标。区间不够窄就是INCONCLUSIVE，不能无边界加样本；任何后续扩大是新的明确问题与有限预算。

合并主规格的EVAL臂为：native Dense、Q1_CAL256、选定四状态SVD/CA/CD、CD-r1408预算匹配点、同选定CD因子的two-residual消融、作者式Chip reference、Global-P64，共最多9臂；22 source×2seed为最多396条轨迹。CD-r1408与选定CD若相同，只运行一次，不为凑臂数另找新参数。Chip定义与验收由其专项提供。本文件仅规定ICC侧正确性。

ICC式naive FFN缓存p2保留为方法验收/开发阶段的作者reuse对照，不自动增加一个EVAL臂；它应直接缓存pre-gate y，不用其它旧G0名称推断身份。是否生成其完整DEV视频与所占预算由合并主规格统一登记。

### 必须保存的末端指标

- 与任务匹配的I2V主体/背景、视觉质量、时序稳定性；motion读出至少同时有方向、模长与残差，避免静止方法在smoothness上“获胜”。
- 输出失败与缺失按预定规则计入分母，不能删除坏视频后重算均值。
- 所有方法使用同一metric code/model/预处理与源适用集合。已有视频/latent可评分时只补指标，不重生成。
- 速度采用同硬件/资源、同attention/offload/decoder背景、probe-free路径；分解/校准、anchor、hit、cache传输、峰值和完整模型耗时分别报告。原法host-offload的低速是该资源配置结果，不以通用慢reference冒充已充分优化的作者极限速度。

## 9. 最小适配消融与结果解释

| 消融 | 固定项 | 唯一变化 | 最小覆盖 | 能解释什么 |
|---|---|---|---|---|
| 4S→2R | 因子、rank、CAL、dtype、轨迹配置 | 缓存代数 | 统一EVAL中的选定CD/同因子CD2R；历史r512开发对照作为合并主规格可复用项，不另开队列 | 缓存重排对自由轨迹影响；不同rank结论不互相替代 |
| CAL22→256 | 家族、rank、4S、dtype、p2 | 校准来源前缀 | CD同固定rank的DEV11×1seed；Q1另按主账配对；CA仅全80局部比较 | CD/Q1闭环与CA局部校准敏感性，不外推CA视频 |
| FP32 stats→BF16 stats | 原始值、公式、source | 统计算术dtype | 六sentinel、CAL11/DEV11；有完整视频主张需再绑定视频 | 统计精度贡献 |
| 作者dtype→Wan BF16算子 | 同真实值、同因子与边界 | 操作数与计算dtype | 全80真实操作数；作者工程闭环 | BF16适配的局部数值影响 |
| BF16→FP32-FFN rollout（本轮未采纳） | 4S、同因子/输入及宿主出口 | FFN算术精度，同时有Dense控制 | 仅未来另行明确范围；本轮不含这4条 | 数值变化经真实轨迹的传播 |

历史两残差不是“错误实现”的同义词；它有真实内存动机。其身份应保留为工程变体，并由配对消融决定是否可作为生产替代。若2R在已测域足够接近，仍要报告其额外anchor投影和BF16零增量偏移。

不增加独立r1×r2、每层rank、max_ratio网格、gate-weighted-CD、CA/CD融合、重训练或新period候选。这些是新的方法或后续优化，不是补齐原法所必需。

## 10. 并入单一总账，避免pilot与失败被藏在预算外

**唯一正式计数以 [CONTROLLED_REPRODUCTION_PLAN.md](CONTROLLED_REPRODUCTION_PLAN.md) 为准，本专项不再维护另一个总执行预算。** ICC三家族、数据生产、neutral、small-CAL、dtype、2R消融与Chip主臂共享同一source与生成账，不允许分别按专项重复派发。统一EVAL上限396条，已匹配Dense/Q1/候选产物和重合的CD1408按实际身份扣重。

CAL256采集不VAE解码、不生成视频；DEV/EVAL若已有完全匹配产物，不因新报告名称重跑。所有smoke/preflight若完成一次全D4，都计入合并账；能作为正式条件复用时只占一次。失败与重试仍计资源，不刷新版本重新获得“免费pilot”。

离线计算的结构性估算供主账核对，而不是额外授权：三家族全80需480个linear分解（每家族80站点×2矩阵），最大rank前缀复用。若合并主规格确实启用CAL22两家族全80和11/64的六站点消融，相应分别增加320和48个分解；已存在同统计identity的因子直接扣除。rank、缓存形式、回放共享同因子，不重复分解。未被主规格采纳的建议不自动进入执行队列。

资源缩减应优先消除候选重复、复用既有产物或缩小末端统计主张，不可删除原法四状态/CA/CD后仍称完成同一任务。任何未完成分支明确标识。

GPU小时不能凭旧混协议时长硬填。执行冻结前用已存在producer receipt和首个已计入正式条件的耗时/峰值建立剩余工作估计，资源不足按公开未完成范围停，不自动转成更弱方法。当前只知道旧scope曾给出至多2GPU的规划上限；不能据此断言现在有2GPU可用。

## 11. 复用与必须重做的边界

| 已有资产 | 复用条件 | 不满足时的最小补救 |
|---|---|---|
| 原模型/学生权重 | checkpoint SHA、tensor shape/dtype、专家/layer键一致 | 找回原文件，不重新训练 |
| SVD512因子 | factor payload、原weight SHA、merge方式/solver/dtype匹配 | 只补缺失/不匹配层；更高rank一次分解后保留前缀 |
| 历史CD因子 | producer完整源码、4边界配对、max_ratio/eps、CAL身份和文件SHA齐 | 找回记录；不能恢复则重建CD因子，不把旧视频标成新因子结果 |
| CAL/DEV native面板 | 真正同token IDs、same source/seed/step/expert、原生dtype/模型/backend，角色合法 | 只补缺失字段/条件；512行不能伪装full-M统计 |
| CA/CD full-token统计 | 可核producer、每source四边界向量与计数、dtype/eps/cap一致 | 缺哪source就补哪source一次native采集 |
| 历史视频 | 方法/因子/cache-mode/所有数值参数、输入/噪声/encoder/scheduler绑定齐，指标可比 | 只重做不合法的臂/条件；能补指标不重生成 |
| 旧2R视频 | 可继续作为2R历史结果；新4S与其同identity配对时可复用 | 不能充作4S主reference；因子变了需新生成该方法 |
| 旧Q1视频/参数 | source role、拟合祖先、CAL配置及A完全相同 | 新CAL256 Q1是新arm，不能沿用CAL11视频 |
| 旧局部R1b/BW/BWS | 保留其原估计对象与源绑定 | 不回写旧结果；可作diagnostic，不代替新完整视频 |
| 历史timing | 相同资源/背景、probe-free、可核原始重复 | 只补公平timing，不重写已合法质量评价 |

发现代码binding缺失不自动意味着历史计算有误；也不能把找不到脚本当作“理论等价所以已验证”。先找回，不能找回再按最小缺口重做。发现第一处数值/状态边界错误时，修复对应版本并使其下游产物失效；不默认全部历史资产失效。

## 12. 每个实际运行必须留下的验收包

`METHOD_SPEC.json`（family/rank/cache/precision/period/模块范围）、固定upstream与本地源码identity、checkpoint/因子/scale/CAL manifest、source角色与历史暴露记录、当前输入/初始噪声与真实timestep、neutral逐边界结果、各层anchor/hit计数、RUN/EXIT/失败账、原始逐视频与逐source指标、cache bytes/peak/传输与timing。

每个实验结论沿实际artifact绑定到实际代码；不能用“规范写了1000”“代码有four_state函数”“计划256校准”推断对应执行发生。所有尚未运行的验收保持UNVERIFIED。主报告在上述事实完成前，应写“严格reference正在建立”，不能继续以已知2R劣化宣判原ICC失败。
