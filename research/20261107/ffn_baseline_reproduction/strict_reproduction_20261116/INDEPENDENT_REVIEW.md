# 固定代码引述与精简复现：独立复审

审阅者：`/root/strict_repro_review`。日期：2026-11-16（北京时间）。本审阅沿用 research-implementation / experimental-design 的核验原则，读取实际固定源码、引述文件和本轮精简设计；没有 GPU、模型生成、分解或拟合运行。

## 1. 审查状态与对象

**最终判定：可以发布为已独立审查的固定源码证据与精简开发实验设计。初审要求均已修订并回核，见第7节。该判定不是模型验收，也不是可直接派发的运行scope。** 本轮不把已撤销的897次方案当成待执行要求，也不恢复CAL256、EVAL22、O40或原系统完整生成。

已审对象：`CHIP_CODE_CITATIONS.md/.json`、`ICC_CODE_CITATIONS.md/.json`，以及当前正式目录的 `CONTROLLED_REPRODUCTION_PLAN.md`、`EXPERIMENT_BUDGET.json`、上层 `STRICT_REPRODUCTION_AND_REDO_20261116.md`。引用审查使用引述JSON中实际local_path所指字节，而非仅相信生成器写入的verified字段。

## 2. 引述与重要主张的独立核验

独立纯Python检查共 **83个短引述**：Chip 63个、ICC 20个。全部满足相应文件SHA256、固定Git blob、原始UTF-8/CRLF位置、quote字符切片或行区间与commit永久链接；没有发现逐字引述错误。Chip两份旧tools快照的额外末尾LF按已明示规则规范化，未静默改动源码。ICC的12个被引用文件均在source_files中登记。

| 主张 | 实际代码依据与审阅结论 |
|---|---|
| 官方Wan示例只接attention稀疏 | run.sh→实际cuda-test YAML→WanModel注册链闭合；YAML的MLP关闭、注册is_mlp_sparse=False、FFN仍Linear/GELU/Linear。支持这一固定版本的官方入口判断，不证明作者从未在其它代码上试验MLP。 |
| 原SparseDiffMlp可优先复用 | 官方provider分派含BF16 Triton mm1/mm2，selector/copy仍需compiled utilities；不能把A100作为必须重写算法的理由，也不能把源码支持说成当前环境已编译通过。 |
| Chip状态机不同于单次anchor→hit helper | dense存h/y/proxy；重算mask才选择性更新proxy；每次hit更新隐藏列与输出缓存；step≥10才能出现指定reuse；first2常Dense。原源码dense分支不清indices，10刷新后11可继续使用9的mask。 |
| 原Triton与本仓packed并非相同舍入 | mm1在FP32累积+bias后GELU再BF16；mm2先将delta结果转BF16再加cache。本仓helper先BF16 Linear后GELU，W2 FP32结果与y.float相加后再转BF16；同mask只能隔离选择器，不能自动消除算术差别。 |
| ICC四状态、两级校正与2R差别 | 原forward_upd保存两级输入/输出；forward_inc按输入差、低秩、加anchor、真实GELU、第二级差/低秩/加anchor执行，hit不更新anchor。2R重排实数等价，但改变有限精度与anchor成本。 |
| CA/CD必须分别实现 | stat_upd的mag与delta_mag来源不同，四边界均参与；token mean-abs→逐sample min-max→batch/source累计→sqrt→左右cap→加权分解的顺序有源码依据。普通SVD不能替代CA/CD的实现验收。 |
| 256不是256完整轨迹 | gen_decomp默认256 dataset items、batch16、epoch1。CA外层每batch调用一次ddim_sample，CD两次相邻调用；因此对应16/32次批调用、256/512个item-evaluations，不能乘成256条完整50步生成。原代码也没有来源簇独立性证明。 |
| BWS是已有局部执行证据 | 固定runner实际走普通SVD-r512/2R与Global-P64 helper。publication/EXIT支持既有局部replay执行身份，同时明确未测视频质量与性能，不能据此认证新的4S/CA/CD/作者Chip全生成。 |

关于256的纠正直接影响设计：上轮我认可的强制CAL256门槛属于当时过强的实验范围，不是原算子正确性的必要条件。此次将其修正为显式CAL11目标域配置与有限CAL11→22敏感性检查，有代码和任务目标依据。仍不能据此宣称11个条件足够代表作者校准分布。

## 3. 精简方案的科学逻辑

可以保留完整模块定义，同时只让一个预先选定的工作点进入全模型比较。三个家族均需完成原算子/统计/分解验收，并在固定六site、同DEV11/P512/current gate上比较；按已经冻结的聚合规则选一个r1408，构造选中家族全部80处因子。未选中家族只有模块/局部证据，不能在结论中提升为已测完整视频质量。

CAL11→22改变当前native-D4采集分布内的来源预算；它不检验原ImageNet real-image q_sample随机t分布与Wan轨迹的等价性。主CAL11和主家族在全视频前固定，条件CAL22结果不能回写主表。CA的固定端点选择也须保存在统计manifest，不能看风险后改为双端点均值。

五臂共用DEV11、seed42，可以回答当前开发问题。DEV参与家族选择，因此完整视频均值与来源删除敏感性是开发证据；不使用“独立确认”“无偏泛化优势”或“完整Pareto最强”标签。指标适用来源少于11时，独立n相应减少。

必要的实现错误应先按oracle修复；合法的tail/rounding变化才作为单变量适配对照。四态资源不足仍标RESOURCE_BLOCKED，不以2R替代。完整原系统生成、新O40与多seed不是本轮模块验收前置。

## 4. 初审提出的具体修订

以下保留初审依据；已完成处理见第7节，不是另一个待执行清单。

### R1. E5必须对应真正被改变的主家族

若选中SVD，CAL22不改变其因子，E5必须为0。CA/CD其它候选排序变化可以报告为局部敏感性，但不应触发一套相同SVD视频。

即使选中CA/CD，风险变号或“一半差距”的条件也应针对**已冻结主家族**；单纯另一个家族变好不等于主家族预算干预已有辨识力。第三条件应写出数学定义，明确是任一方向的变化量，还是朝ICC-Q1差距缩小，并要求超过同量纲的neutral数值包络。主CAL11/主家族始终不替换。

### R2. E6只能是一个预先确定的合法适配对照

建议固定tail→Chip native-rounding的优先顺序，在主视频结果可见前只选一个已由必要局部检查发现非等价影响的合法方案；没有符合条件项则E6=0。另一个未评价分支保留UNRESOLVED，不把三条当任意搜索额度。与作者oracle不符的主实现错误必须先修，不能通过E6视频的好坏决定是否忽略错误。

### R3. neutral复用必须按具体执行路径扣除

两个来源各有plain、hook-bypass、ICC forced、Chip forced、条件Q1 forced、条件GP64 forced六条路径；最坏12次，其中hook2已计入55，额外上界10正确。

但旧plain视频可以同时支持质量表与plain-neutral证据，不能冒充新hook-bypass路径。若仅有旧plain质量视频，不能先从55扣掉全部11条，再把仍需新跑的两条hook忘记记账。按path/execution_id扣除实际重复，99上限无需增加。

### R4. 引述和采集措辞

- Chip U07应写“更新选中隐藏列h，并将其W2增量加到对应token的完整输出向量y”，避免误读成只更新选中输出通道；同步写清dense refresh不清旧indices及9→10→11的验收关系。
- 初稿Chip supporting_artifacts引用了旧3361字节工作副本的counter trace。应改绑上轮已纠正的正式trace：source SHA256为`cb9ffab57832ce69141ecd914fbb3f2c873c26bdca31eee1f7e2b41c01764288`，trace SHA256为`2c9cf100a5599c8d6b1d19ce53e015e51c86c21a27838930335ea8b943db8879`，并保留源码身份修正说明；无需新跑模型。
- “无VAE”应写成“不做输出VAE解码/视频编码”；I2V原生条件编码或可核验的编码缓存继续保留。

## 5. 数量和数学预算核对

基本D4上界 `22 CAL + 10额外neutral + 55主表 + 3同因子2R = 90`；两个条件分支各3，加全局技术重试3，总上界 **99**。一条D4按4次模型前向计，最大396次NFE，实际失败可能只完成部分步；E0模块、离线SVD/PCG、同步搬运成本另记，不能称全部工作仅396次等价标准推理。

r1408的四态anchor-hit主MAC比仍为0.6884259259；Q1为0.6851851852，差约0.473%。原ICC层范围为80 anchor/80 hit；Chip保留first2后为8 fixed-Dense、76 anchor、76 hit，合计160。较少实验不改变这些方法定义和成本分母。

离线分解按拟定唯一对象计算：主CAL11三家族六site两矩阵为36；CAL22的CA/CD六site两矩阵为24；选中主家族其余74site两矩阵为148，总 **208次Wan大矩阵SVD**。若E5确实触发，主家族CAL22其余74site最多再148，合计 **356**。SVD选中时没有CAL22重分解；r512从同次分解取前缀。原shape正对照和Q1固定PCG另列，不把额外solver、端点或权重网格藏入同一预算。

## 6. 真实证据状态

本轮新增执行只限于文档/源码字节、引述、元数据与算术的CPU检查；模型/GPU/正式视频/校准拟合新增数均为0。引述验证证明“这些代码字节表达了这些操作”，不能替代author module、selector、neutral或自由生成的实际运行。

历史A/BW/BWS的既有执行身份和原结论不改写；原897方案保留为已撤销设计记录。本轮没有新执行授权请求，也没有新增审批层级。

## 7. 最终合并回核

最终回核直接读取正式候选目录 `ffnds_review/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/` 及上层主文，确认以下修订已落盘：

| 初审问题 | 当前处理 | 状态 |
|---|---|---|
| E5可能对SVD或无关家族生成无效视频 | 仅主CA/CD可触发；风险条件仅针对冻结主家族；其它家族排序单列局部观察；SVD时E5=0 | 已关闭 |
| “解释一半”含义不明确 | 明定 `abs(R22-R11) >= 0.5*abs(R11-RQ)`，超过同量纲数值包络，DEV3至少2源同号；允许改进或恶化触发，主CAL11不变 | 已关闭 |
| E6可变成自由搜索或容忍实现错误 | 在任何主视频结果可见前，按tail→Chip native-rounding固定优先级选一个合法非等价方案；oracle错误先修；无符合项则0，未测项UNRESOLVED | 已关闭 |
| neutral复用可能重复扣除 | 按(source,seed,path,execution_id)抵扣；plain不能替代hook；全Dense减11须两条必需hook也已有有效复用 | 已关闭 |
| Chip输出更新措辞与refresh状态遗漏 | U07明确选中隐藏列h与完整输出y；明确10刷新不清indices、11可复用9的mask | 已关闭 |
| counter又引用旧工作trace | 已改绑正式trace及正确源SHA `cb9ffab...`，保留历史身份修正；本轮复用而未声称新增运行 | 已关闭 |
| “无VAE”可能误删I2V条件编码 | 明确保留原生I2V编码/合法缓存，只省输出VAE解码与视频编码 | 已关闭 |
| 旧大实验或额外离线计算残留 | 主文、ICC专项与预算统一主CAL11/局部选一个r1408；原897方案已撤销；208/356次Wan SVD、原shape6、Q1最多80固定PCG分别登记 | 已关闭 |

### 实际独立检查

1. 正式 `EXPERIMENT_BUDGET.json` 的stage上限之和为99；90+6+3一致；新O40、额外整模型计时、原系统完整生成均为0；`runnable=false`、`new_model_runs=0`。
2. 离线分解208=36+24+148，条件追加148后356，r512共享分解前缀；没有增加precision/endpoint/sampling因子网格。
3. 统一 `CODE_CITATIONS.json` 已对36个固定文件、83个quote再次独立核对，全部通过；注册主张数37。检查包含实际源字节、Git blob、quote字符区间及行区间。两份历史tools快照仅使用已登记的单个末尾LF修正，没有静默规范化CRLF。该检查仍是CPU文件核验，新增模型运行0。
4. 关键主张另按第2节阅读上下文核其含义。自动引述通过不被当作算法、数值、质量或速度通过。

### 本次审查的文件身份

| 文件 | SHA256 |
|---|---|
| STRICT_REPRODUCTION_AND_REDO_20261116.md | 5557708a39221b954316b9c85f237a43f0d8e1a9ee5229acffd23f9bb6d08ff8 |
| CONTROLLED_REPRODUCTION_PLAN.md | b9489f4f7ed958d6545d44ddcc42ed4f09a3c394ed1943997a8a5b2fc6cb713c |
| EXPERIMENT_BUDGET.json | 0f13df0faa6a07417dc9a01c5136f688dcac8137011504c101dda3197a8a15e8 |
| CODE_CITATIONS.json | 44be5801cc49869e15f3338adf20ab51f2ba78d70aadc5e311eb4b88c0860179 |
| verify_code_citations.py | 268062d7086f2af492faa800a40f810431289668adb39340b49e6e4e9fd77245 |
| CHIP_CODE_CITATIONS.md | e4a1b7919dd056729906c3113daf64f4e5441667e7e7f978762b629cc580294e |
| ICC_CODE_CITATIONS.md | e0ef17c43667236539f50c95c2ff8ebe05dd0757c0e3289f4d6ae4707c44a29d |
| ICC_STRICT_REFERENCE.md | e4fc2e27ee718df9457bc67f6efe3c5bad409a4fb7fc388f3f5f4fdb3332f1d6 |

**可发布。** 这套设计保留了需要验证的算法组成，并使每个新增完整生成对应一个具体问题。当前资源、实体输入、编译、模块oracle、全图neutral、全部选中方法的自由生成仍未验证；开发集选择与小校准的结论范围已经收窄。实施时只能按真实receipt更新这些状态，不能用本评议替代执行。

历史897次提案的独立审阅保留在[9def固定提交](https://github.com/Qesire/FFNDS/blob/9defbbaa9112eee954c24ae92c91831fd2dd69c4/research/20261107/ffn_baseline_reproduction/strict_reproduction_20261116/INDEPENDENT_REVIEW.md)，不与当前设计叠加。
