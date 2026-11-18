# 当前响应驱动 FFN 算子的实验方案

日期：2026-11-18。实验结果基点：f89e4fae868ac8d21b3b8f0da47616f43b0c954a；本次修正替代 a2a3f7e2d802f26dee04652c2f328dc6cd8a2b6c 中的局部 MSE 准入和9+6视频分支。

目标是判别当前真实观测是否能让自己的紧凑算子获得固定读出缺少的响应，并在学生自由生成中保留其价值。[算子设计](OBSERVED_SECANT_OPERATOR_DESIGN.md)规定数学、训练、缓存和BF16顺序；[机器规格](observed_secant_experiment_spec.json)记录队列、预算和分支；[独立审查](OBSERVED_SECANT_INDEPENDENT_REVIEW.md)记录本次设计与参考核心的实际复核。

本轮方案改为：既有资产上的局部诊断；两类既定构造各自全80位置的 static/online × DEV3，共12条主开发视频；完成后最多3条六位置作用范围诊断，总上限仍为15条。局部 MSE、增量余弦及其 oracle 均不决定某一构造能否进入视频阶段。推导与证据见 [MSE筛选复核](MSE_SELECTION_REASSESSMENT.md)。

没有启动新模型实验。FP64参考核心已测试，CAL producer与Wan端口仍需按本方案实现和绑定。原独立审查只覆盖其记录的旧文件SHA与9+6方案；本次12+3修正已由独立agent重新审阅并通过设计复核，见审查记录文末。该通过状态不覆盖尚未实现的原生producer。当前文档不更改已关闭的上一轮P0—P2，也不自动启动旧P3/P4。

## 1. 中心问题、构造和固定比较

两类构造：

- 观测条件读出：16原生隐藏通道、配对32特征、64当前完整FFN观测；从固定全输出V_stat在线更新V。
- 满秩状态模式：4原生隐藏系数模式、64当前完整FFN观测；从固定theta_stat在线更新共享theta，再编译当前矩阵。

两类构造基于不同表示假设，均保留一次完整作用范围的配对视频比较。实施顺序可先读出、后模式；前一类的局部误差或视频结果不取消后一类。只因绑定、实施、数值有效性或实际资源问题无法执行时，记录确切执行状态，不把缺失结果解释成方法失败。

当前 CAL 的 MSE 通道选择、greedy16、作用度量 POD 和正则定参保持冻结，明确作为欧氏响应拟合的构造控制。本次撤销的是未经验证的质量筛选权，不假装这些结构已按视频功能优化。若两类视频都没有建立收益，结论仍只覆盖这些具体特征/模式和拟合规则，不能推出所有状态条件化算子都无效；本轮不另加损失、rank或probe搜索。

要区分：

| 竞争解释 | 判别 |
|---|---|
| 只是旧Q32限制输出 | 同一新X的固定Q32与固定全输出读出 |
| 固定表示已足够 | 同一表示、同缓存的static与当前probe online |
| 特征有用但64观测辨识不足 | 每fold query标签oracle与真正64probe预测的同query差距 |
| 单次残差秩较高但状态族简单 | rank32读出与4个可满秩模式 |
| 六位置不足以产生已有视频效果 | 0392的P64-6与已有P64-80，以及观测读出6/80作用范围 |
| 教师局部收益未传入学生轨迹 | 自由视频内自身输入的离线query审计与最终完整视频 |

所有oracle仅用于结构判断，不能成为部署臂。局部改善、视频改善与实际加速分别报告。

## 2. 资产复用与学校位置

### 2.1 已有数据范围

| 资产 | 已核对的范围 | 本轮用途 |
|---|---|---|
| 上一轮原始BINDINGS | CAL11+DEV11、6位置、132端点对/264个P512文件 | 首轮局部拟合与判别 |
| 严格Q1 | 80个expert/层矩阵 | 共同基线 |
| L64 | high/low×04/19/34 | 仅六位置基线 |
| Q1_CAL_PAIRS | 11 CAL×80位置=880端点对 | 为两类全80构造分别生产参数 |
| BANK_MANIFEST | 声明22源、每源160捕获位置 | 复用原始U/Y/gate/坐标；其余74位置引用需扩展核对 |
| 已有DEV3视频 | Dense、Q1、Global-P64，以及旧六位置L0/L+/Q32/N32 | 直接复用已绑定媒体/评分 |

CAL11：

0032、0304、0456、0648、0895、0937、0966、1005、1039、1082、1094。

DEV11：

0056、0176、0392、0800、0886、0947、0950、1013、1040、1084、1109。

以上均使用完整source ID前缀vbi_。DEV3固定0392、1013、0800，全部seed42。

学校已有记录根目录：

    /public/home/scc/pb23061276/projects/wan-quant/work

已有locator：

    g20_v2_capture_20261113_rev2/stage/output/BANK_MANIFEST.json
    g20_v2_capture_20261113_rev2/stage/output/bank
    ffnds_strict_reproduction_20261116_336b7e1
    g20_two_track_core_20261114/A_Q1_RESIDUAL/v1/stage/output/params

上一轮原始BINDINGS SHA256：

    8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6

公开摘要会把路径缩为basename，不能把摘要中的high04.pt等名字当完整locator。实际worker通过学校原始BINDINGS和manifest定位，不推断其他主机home目录。先核实这些已登记文件是否可读；若某引用不存在，输出确切缺失清单，不自动生成新的Dense轨迹补洞。

### 2.2 共同配方和基线

Wan2.2 I2V A14B蒸馏HHLL四步、CFG off、BF16、81帧、832×464、16fps。真实latent token网格21×29×52；21为latent时间轴，81为解码帧轴，4为去噪步轴。沿用严格复现的同图像、prompt、裁剪、初噪和权重身份。

共同A：六位置L64，其余74严格Q1。扩展全80改变条件分支的作用位置，不改变这份底层A映射。

两类方法的特征/模式必须各位置独立由同一CAL规则产生，不复制六位置的参数到其余74位置。全80绑定先复用已登记880 CAL端点；缺文件状态为ASSET_NOT_AVAILABLE，新的Dense采集若确有必要需单列范围与预算，不能混入12/15视频数。

## 3. 局部阶段：0条新完整视频

### 3.1 两折共同query

每source的P512，用UTF-8字符串

    observed-secant-20261118|{完整source ID}|{十进制global token ID}

的SHA256排序，相同digest以global ID打破平分；前256为A部分，后256为B部分。分割在该source的所有层/expert共用。

- fold1：probe候选池A；query=B256；
- fold2：probe候选池B；query=A256；
- 每个候选池按排序后的global ID用NumPy default_rng(seed42)的冻结排列选64行；
- 几何64也只能从同一候选池选；
- 全512无标签X可用于当前Gamma和几何白化，query真实标签只做评分；
- 每个token恰好做一次query；两fold属于同一source的重复测量。

固定64、几何64和比较算子都在相同query上评分，不能各自删掉自己的64行后比较不同的448行总体。

历史Global-P64的真正64行不等于P512前64，也未必包含在P512内。本轮P512内的原生通道更新只叫panel-P64，K=5120；它是FP64同端点方向机制参照，不继承strict Global-P64数值。若另列其BF16结果，单列局部producer身份，且不用于候选准入。视频P64-6则必须使用既有严格Global-P64的真实固定64行规则和原生实现。

### 3.2 两条数值轨

结构轨：将冻结BF16输入/权重转FP64，\(A_{64}\) 为冻结部署BF16 A的FP64提升，计算数学F、有限割线和表示/观测误差分解。

部署轨：使用设计文档冻结的C0、Xb/Db、真实64行BF16 observer、动态读出/矩阵编译与舍入顺序。query目标使用同坐标原始full-M捕获的BF16 Y_pre_gate，当前gate在比较时施加。小批量probe结果与full-M标签的差异单独记录。

通道/模式结构选择只用CAL结构轨；最终先验、正则选择和数值验收用部署轨。旧结果的FP64 risk和本轮原生risk不混排。

### 3.3 必须计算的局部对照

| 对照 | 是否可部署 | 用途 |
|---|---|---|
| 共同固定A | 是 | 基准 |
| 同新X、CAL固定Q32读出 | 六位置可部署；本轮仅局部 | 分辨输出空间限制 |
| 同新X、CAL固定全输出V_stat | 是 | 观测读出严格静态控制 |
| 同新X、fixed64在线读出 | 是 | 观测读出在线臂 |
| 同新X、geometry64在线读出 | 可实现；本轮仅局部 | 单独测观测几何，不形成视频网格 |
| 同X、当折256 query标签oracle读出 | 否 | 连续特征类在同query上的局部欧氏误差下界 |
| 4模式、theta_stat | 是 | 模式构造静态控制 |
| 4模式、fixed64在线theta | 是 | 竞争构造 |
| 4模式、当折256 query标签oracle theta | 否 | 连续模式类在同query上的局部欧氏误差下界 |
| panel-P64 | 仅同panel机制参照 | 当前原生响应方向与抑制作用 |

固定Q32一臂在同一新X上用CAL拟合，不拿旧N32的不同传感器作为唯一对照。零先验读出仅做附带局部分析，不新增视频臂。

每fold的oracle直接在该query256标签上求同一部署Xb或Db/模板的连续最小二乘，再在同query评分。它是有意使用query标签的表示诊断，不是泛化或部署臂。拟合全512后只评query不一定给出query下界，因此不作该替代。连续oracle的读出/系数再经过冻结量化/编译，另存量化后结果；后者是一个可计算候选，不宣称已找到BF16离散模型的全局最优。full-panel表示谱可作为附带统计，但不代替上述query oracle。

### 3.4 主要统计与P64方向

逐source×site×fold保存：

- gated MSE \(E=\|(\widehat Y_1-Y_1)G\|_F^2/(N_Qd)\)；
- 同目标相对风险，分母为 \(\|(Y_1-Y_0)G\|_F^2\)；
- 增量余弦、预测/真实增量范数比；
- oracle floor、online相对static增益、条件数/有效秩、tau/lambda、更新幅度；
- 模板预测与实际量化/编译预测差异；
- 各source的两fold先合并，再六位置等权，最后11个source等权。来源为独立统计单位，不把fold、层或token计成样本数。

在统一FP64方向表中，令 \(R_A=(Y_1-Y_{\rm base})G\)，\(D_P=(Y_{\rm panelP64}-Y_{\rm base})G\)，候选修正为D。直接计算：

\[
g(D)=2\langle R_A,D\rangle-\|D\|_F^2.
\]

再把R_A和D同时投影到当前gated Q32及其正交补，分别给出内外增益。它回答“P64/候选降低当前局部欧氏残差的贡献是否在旧Q外”，不再用原始残差75%漏出替代这一问题。这仍是MSE的方向分解，不把正贡献解释成已证明的视频功能价值。候选与P64方向余弦只作辅助；视频功能需要完整生成判断。

### 3.5 局部诊断不再决定视频资格

撤销旧方案的 MSE 改善至少2%、至少8/11来源获胜、单来源回退不超过10%及均值不差于A四项视频准入。它们没有被现有证据验证为视频质量预测规则；也不以余弦、范数比或另一未经验证的标量替换。

局部阶段仍回答三个有界问题：

- 相同 query 上，online 相对 static 改变了多少 gated MSE、方向和幅值？
- 同类连续 oracle、量化后的 oracle 候选与真正64probe预测之间有多大差距？
- 这些差距来自观测辨识、当前欧氏表示类、量化/编译，还是基线/坐标错误？

所有 oracle 结论限定为当前 query、特征/模式类及欧氏目标。报告连续数值和具体差距；如使用分类名称，带上 EUCLIDEAN 限定，不把“该类欧氏 oracle 没有收益”写成视频功能无改善空间。静态表示收益与当前观测的额外收益分别归因。有限的局部 MSE 退步本身不关闭任何一类视频。

两类进入视频前的要求仅是可执行性：80位置资产和参数绑定正确，冻结的真实BF16/FP32计算、cache/current-gate和probe信息边界有效，输出及求解数值有限，static控制与自身online数值路径一致，资源与运行范围能够兑现。缺失绑定、非有限值或实现错误先修复/报告；不用FP64 oracle代替实际数值路径验收，也不把技术未就绪当成科学负结果。既有独立复核和ACTIVE登记仍用于实际模型运行。

## 4. 两类完整作用范围：12条主视频

对每一类构造，均执行以下全80比较，全部seed42：

| source | 观测条件读出 | 满秩状态模式 |
|---|---|---|
|0392|readout-static-80、readout-online-80|modes-static-80、modes-online-80|
|1013|readout-static-80、readout-online-80|modes-static-80、modes-online-80|
|0800|readout-static-80、readout-online-80|modes-static-80、modes-online-80|

共12条。每类的两臂使用全部80位置各自CAL参数、相同底层A映射、同缓存provider和同数值顺序。只在online启用当前64行观测和求解。static不执行无用probe。每个新臂从相同初噪重新生成自身anchor，不跨臂借缓存。

复用原Dense/Q1/Global-P64-80同源视频；不重跑ICC、Chipmunk作者臂或旧四个候选。原全80基线必须沿用其已绑定配方与作用范围，不能拿六位置臂冒充。

这12条用于直接判断两种不同表示中条件化的实际作用，不按局部MSE先挑“赢家”，也不因先完成的一类已获收益而取消另一类。全80绑定缺失或原生实现无效是执行状态，分别修复/报告，不记成科学负结果。

## 5. 完成全80之后的可选范围诊断：最多3条

范围诊断固定使用观测读出，不按视频或局部最优分数改换family。仅当以下条件同时成立时执行：

- 两类全80的12条视频均完成，完整连续视频人工判断已完成；HUMAN_REVIEW_PENDING不代表失败，也不触发本分支；
- 至少保留一个有功能价值的候选，或已明确记录一个仍需区分的六位置/全80覆盖问题；在生成本分支之前写明该问题及本三臂如何区分；
- 三个臂的资产、原生数值路径和预算有效。

新增0392的3条，全部seed42：

1. readout-static-6；
2. readout-online-6；
3. Global-P64-6。

全部仅作用在high/low×04/19/34，其余74位置严格Q1。P64-6使用原Global-P64 K5120、真实固定64token与同一packing/缓存行为，只改变作用位置。复用既有0392 Dense、Q1、L0、N32、Global-P64-80及本轮readout-80；每臂仍有自身学生anchor。

这一步只用于作用范围诊断，不能倒过来门控已经预定的12条全80。若P64-6与readout-6都不能保留完整作用范围的效果，只能说该六位置配置没有重现对应的全80表现，不能由此证明某个机制无效或识别其余74位置的单独因果贡献。若没有值得继续区分的范围问题，不运行这3条；坏MSE本身不构成触发理由。

## 6. 视频指标、人工判断和解释

沿用[QUALITY_LINEAGE](../../reports/20261118/state_conditioned_ffn_operator/output/QUALITY_LINEAGE.json)绑定的原始指标和尺度：I2V subject/background、subject/background consistency、MUSIQ原始分、美学、运动平滑、时序闪烁等。旧dynamic_degree缺乏区分力，不作为优化目标。

补充连续RAFT位移幅值、相对Dense固定网格的EPE/幅度/方向，但不将“光流更小”“运动更平滑”自动判为好。固定像素EPE不等于有物体对应的轨迹误差。

完整MP4保留给用户连续播放。0392的Dense晕眩重影是需要保留的既定效果；不能把消除该效果判为改善。先隐去方法标签播放，再记录：

- 相对Dense额外破碎：0 无明显新增；1 短暂局部；2 反复结构破坏；3 持续严重破坏；
- 既定效果是否保留；
- 是否新增主体/动作/背景结构严重退化。

每类以自身matched-static为主要控制，完整视频功能判断与逐维自动指标同时保留。功能候选需要在0392改善额外画面破碎并保留既定晕眩重影/动作，另外两源没有人工确认的新增严重主体、动作或背景退化。人审记录具体时间段、破碎等级及判断依据；不从缩略图替代连续播放，也不要求某个自动指标达到任意固定增量才能承认功能改善。

旧方案的 subject consistency +0.01、三源 I2V 均值回退不超过0.003/单源0.01、MUSIQ均值回退不超过0.5，仅保留为旧方案历史参照，当前不作为准入、关闭family或保留方法的硬阈值。逐源原始量、差值和任何实质退化仍完整报告，不通过删除不利指标维持结论。

若人审与自动指标冲突，明确记录 METRIC_HUMAN_DISAGREEMENT 和具体证据；不以一个标量自动判成功/失败，也不强行把冲突汇总成单分数。完整人审未完成时记HUMAN_REVIEW_PENDING。DEV3已经暴露，这些是开发观察，不是统计非劣或泛化证明。

同时报告Dense、Q1、P64的逐源完整向量。若static与online都改善，只把额外的online-static差异归因于条件化；若仅局部改善、全80未建立收益，则结论限本构造和本轮开发范围，旧终端校正路线仍未被实验。

## 7. 每条视频保存什么

主生产保留MP4、final latent、各step hash、初噪/权重/参数身份、每位置anchor/hit计数、当前gate边界见证，以及：

- 64个probe global token ID；
- 当前有效秩、条件数、lambda/tau；
- 当前V或theta、固定prior及模式/特征身份；
- 各hit真实probe残差与基线预测；
- 固定query64的U1、当前gate，以及该token的anchor必要状态和候选预测。

query64按真实global ID固定选择，排除新方法fixed64与原Global-P64 canonical64的并集；同source的static/online及P64-6使用同一query ID清单。生产中只保存，不计算额外query Dense。若输入与cache落盘会扰动计时，标明该视频属于研究路径。

在视频结束后对保存query独立计算64行Dense，以分辨教师局部→自身student状态的迁移。该teacher有自己的64行batch形状和producer身份，不叫full-M shadow；此前CAL的64/full-M数值差异为其解释提供参照。它不是免费推理部分，也不计入主视频延迟。追加query教师运算有固定上界：

- 主12视频共有960个candidate hit（两family的static/online合计），每hit最多64个query，合计至多61,440 token-FFN离线观测；
- 可选3条范围诊断再加12个readout hit和6个P64-6 hit，共1,152 token-FFN；
- 全部15条的离线query观测上界为62,592 token-FFN。

不用这些query重新拟合本轮参数。本轮只有两类固定主比较和一次范围诊断分支，不按出现的视频问题逐层调参。

## 8. 预算与执行顺序

| 阶段 | 新完整D4 | 新NFE | 目的 |
|---|---:|---:|---|
| 既有资产六位置CAL/DEV局部 |0|0|欧氏拟合、表示与在线信息诊断|
| 观测读出全80 DEV3 static/online |6|24|第一类完整作用范围|
| 状态模式全80 DEV3 static/online |6|24|不同表示假设的固定比较|
| 0392三个六位置臂（可选） |≤3|≤12|完成全80后的作用范围诊断|
| 合计主路径 / 上界 |12 / 15|48 / 60|开发范围|

主链FFN位置调用为1920 / 2400；额外probe单列：

- 主12条中的6个online全80臂：480次完整64token FFN，30,720 token-FFN；
- 可选readout-online-6再加6次，上界486次，31,104 token-FFN；
- P64-6只在可选分支发生，最多另有6次64token W1/GELU观测，不是完整FFN probe；
- static臂零probe调用。

离线拟合及P512 replay、指标评分、query教师回放不伪装成0成本，但不新增DiT轨迹。当前GPU并发按有效两卡上限，不因设计改变集群调度规则。两类CAL生产均按既有880端点绑定，不能将另一类的全80参数生产藏为“条件免费”。

执行顺序与停止优先级：

1. 资产或数值路径无效：报告确切执行问题；只修该问题，不下方法负结论。
2. 记录两类局部诊断，有限的MSE改善不足或退步均不取消其全80视频。
3. 按固定配置完成两类共12条全80；一类的质量结果不关闭另一类。
4. 完整人审完成后，根据第5节的范围问题决定是否执行唯一3条诊断；人审pending不作为失败。
5. 汇总完整视频功能、逐维指标及局部/学生query关系，结束当前构造范围；不自动开启旧P3、区域bank、rank/probe/loss网格、新来源或更多seed。

上述12/15及对应NFE是有效完整实验配置数；它取代旧方案的9条主路径与6条条件备选，不是含任意失败重启的总消耗上界。本轮自动额外完整重启预算为0；若基础设施失败确需重新执行完整生成，应先冻结独立的有限重试范围与额外预算，失败尝试和实际消耗逐项计账，不从本表推导无限续跑权限，也不能用重试挑选视频样本。新的Dense资产采集、性能专用重复和独立确认均不包含在本轮预算内。

## 9. 执行工件与状态

首个原生生产视频同时承担完整gate/cache/作用位置见证，不另造前置整视频队列。纯数学单测、真实P512路径验收和视频验证承担不同职责。

建议最小工件：

    BINDINGS.json
    CAL_PARAMETERS.json
    CAL_TUNING.json
    LOCAL_QUERY.csv
    LOCAL_STRUCTURE.csv
    BF16_ACCEPTANCE.json
    FAMILY_EXECUTION_STATUS.json
    GENERATION_MANIFEST.json
    QUALITY_PAIRED.csv
    HUMAN_REVIEW.json
    STUDENT_QUERY.csv
    DECISION.json

大参数、完整视频和张量留在学校真实归档；仓库保存摘要、校验身份、命令与明确locator。沿用现有ACTIVE登记和执行前独立复核流程；本文件的设计审查不是尚未完成的native producer运行审查。

既有复用入口：

- [p0_bindings](../../tools/ffnds_state_operator/p0_bindings.py)：原始身份/学校路径；
- [fit](../../tools/ffnds_state_operator/fit.py)：端点/权重加载与来源聚合；
- [native](../../tools/ffnds_state_operator/native.py)：六位置/其余Q1分流、当前gate见证；
- [prepare_p2](../../tools/ffnds_state_operator/prepare_p2.py)：生产范围和配方；
- [参考核心](../../tools/ffnds_observed_operator/core.py)：本轮新数学核心。

实现者需要新增CAL producer和BF16/Wan port，不能把现有NumPy文件直接称为可提交Slurm的推理入口。

## 10. 本轮之后的确认与性能

DEV11/DEV3已暴露，本轮属于开发和机制检验。

如果保留了构造，下一独立范围应冻结唯一方法，比较Dense、Q1、Global-P64与候选的全80行为。20独立图像簇×1配对seed×4臂为80条D4；若按每prompt5个独立样本的口径则为400条D4。来源至少排除本轮CAL/DEV，历史曝光另记，不能自动声称全项目未见。

这些是后续确认的明确成本，不自动进入本轮。系统计时使用去除研究落盘/逐层哈希的同一缓存驻留策略，包含probe、FP32动态编译和求解；分别给出FFN与端到端时间/峰值。MAC表和带研究hook的视频耗时都不能作为已经获得加速的证据。

