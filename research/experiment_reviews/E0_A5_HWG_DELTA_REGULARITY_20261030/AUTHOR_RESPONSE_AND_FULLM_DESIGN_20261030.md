# E0预审答复与完整全层设计（未执行）

作者 `/root`。回应[真实独立科学预审](SCIENTIFIC_PRE_REVIEW_20261030.md)。状态 DESIGN_REVISED，具体runner、实际采集/验证结果、输入SHA、预算与private job仍须冻结后由独立reviewer精确recheck；本文不是ACTIVE授权或已测结果。

## 已完成的数学修订

ACCEPT：零参考范数、零方向和零delta能量的未定义比例/余弦返回null并携带zero标记，保留绝对误差。ACCEPT：flatten相邻明确包含行尾/τ边界，另外按真实grid分别统计同τ水平/垂直、同空间相邻τ及其配对数。11项小张量fixture实际PASS，包含物理tile饱和、稳定tie、shuffle原坐标恢复、尾部padding、signed抵消推翻对角能量排序、full-delta舍入差与遗漏项负交叉、零分母与真实邻居计数。它们是M1代数/软件证据，未读取真实科研tensor。

ACCEPT：实际Chipmunk P64使用全M seed42固定randperm的64个真坐标，当前小行W1/GELU实际执行并计费；旧h来自经原生样本核对的同身份fullM重放。移植到HiF4 postQ1输入且ranker仍按原raw-h RMS，版本单列；不改成采集1344行内另选64。

ACCEPT：postQ2 true-delta×HWG对照称paid diagonal-energy selector，不称signed最优oracle。原生Q2为各状态分别量化后作差，b2仅在oldY保留一次。FP32 paid-W2投影关闭TF32、固定最高matmul精度；actual BF16 reference和full-delta舍入差分别保存。全M没有保存R，故全M postwrite不冒充可测；实际sample R上的原dtype gate/write检查另列。

REVISE：采样行先筛查是有界可行方案，但本任务要补足真实fullM规则性和全40层。当前计划保留四个source各80同expert pair的完整M31668，不以筛查替代全M。允许按source独立part并行和有限批次完成；每个实际projection有独立预扣、计数、CUDA/host事件和shape，不把所有矩阵乘折成“一次统计”。现有采集+独立读取验收是输入前置。

## 冻结前拟采用的完整同预算因子

所有比较固定K5120；原K64为80/216整组。8个角色为A5-inspired current+HWG、delta+HWG、delta-noHWG、原P64 raw-h RMS的明确HiF4输入迁移、校准静态、固定随机、oldY-only、paid true-postQ2-delta diagonal-energy。原A5参数producer缺失，旧threshold/正式视频保留原身份，原threshold新重放仍MISSING_ORIGINAL_PARAMETER_PRODUCER；不把新adapter称原A5。

每对拟58臂：7种ranker的共同全M K64（7）；三个A5 ranker与paid diagonal ranker在raster/shuffle×M16/M32/M64、K64（24）；同四ranker的逐token单channel及逐token完整K64（8）；同四ranker在global或raster M64、K16/K32的诊断（16）；原P64共同全M单channel K5120诊断（1）；oldY-only（1）；新的mean-input-per-M64 delta+HWG（1）。只有oldY没有矩阵投影，故每对57次paid稀疏投影，四源×80对共18240次，另有320次full-delta参考投影。K16/K32和逐channel不能自动获得原HiF4闭包或物理skip标签。

上述是待freeze的确切范围，不是已发起作业。fullM各状态重放共640次W1、GELU、native Q2、W2；全量A5代理另640次BF16 proxy W1/GELU；P64另320次64行W1/GELU（原logical-channel及K64闭包两臂共享score，不混方法身份）；mean-M64另640次约495行的proxy W1/GELU。量化、路由、统计、norm、排序、mask和packing分别计数。输入Q1已在采集轨迹中真实支付，不在本重放中重复声称省下。

## 代理信息与成本

A5-inspired由已绑定有效W1确定sign、input64均值绝对尺度和row-LS beta；旧/新proxy preactivation各用一次固定BF16 linear，再分别tanh-GELU作差。该具体舍入路径冻结，不能将两次BF16输出差冒充直接FP32线性恒等。当前/增量评分先除真实smooth2，再可选乘当前gate对effective W2的列敏感度；真Δ目标仍是postQ2 V2差。全token代理普通Dense GEMM只作为被完整收费的可运行数值版本；需要低价sign/scale算子才能另谈效率。

四步中每expert第一次必须Dense，只有step1/3可使用其自己历史。K5120主MAC比例为0.5+0.5×5120/13824=0.685185；四次额外全M proxy W1再加0.5，变成1.185185，尚未含HWG/cache/packing。此ALG算术已经揭示普通Dense proxy版本的成本问题，不是实测latency。

新增mean-input-per-M64是独立动作：每tile旧/新V1先求均值再算proxy/GELU差、选择同80个K64组，与原逐token代理同K比较。正负输入变化可相互抵消，GELU(mean)不等于mean(GELU)，作为明确反例检验；四步理想额外proxy主MAC约0.0078125，真实尾部、mean、pack和kernel必须实测。没有通过E0和另审E1前不称可用加速。

## 来源、资产与结果解释

来源898/96拟用于开发静态组选取；1040/979仍是历史暴露开发源，不是盲确认。静态对照进一步加强为 `CAL2_SIGNED_FORWARD_GREEDY_K64_V1`：在两校准source真实1344匹配行上构造216个K64组的门后FP32贡献、保留signed cross的Gram和与原BF16 Y变化的内积，各source按自身当前门后Y能量归一化再等权。前向greedy每次按确切二次目标的边际改善选一组，固定选80组；这是指定可行的强静态基线，不宣称全局最优子集。实际采样覆盖是21个预选τ-tile，不将校准当fullM均匀样本。另行有限注册的fit包含160次216组batched W2、160次Gram/target-dot、80次greedy及真实cache/读写成本；不需要重新跑Wan或读取dev数据。真实输出artifact和SHA在开发source运行前冻结，禁止看其结果改组。cal自身结果与两个dev分开，不将320层对当独立n。来源等权向量和嵌套source bootstrap只描述这4个开发source，不能外推未见来源。

保存每site有效权重/代理参数身份、真实输入/样本重放核对、score recipe、全量compact mask（逐channel用bitpack，组mask保留row layout、permutation和真实tail）、真实随机seed、可确定重放依赖、同K指标/支持/物理原布局occupancy、局部空间/τ及signed质量/舍入项、所有GEMM成本和RUN/EXIT。不会保存多份fullM代理浮点中间值来挤占GPU/磁盘；可重放来源、参数与确切计算recipe显式绑定。模型/cache沿HF复用，不新下载。候选实际selected W1/Q2/W2和完整学生质量仍分别进入另审E1/E2。

## 具体实现自审及第二轮独立预审答复

有限入口是 `tools/a5_hwg_registered_static_fit_20261030.py` 与 `tools/a5_hwg_registered_delta_discovery_20261030.py`，共同使用 `a5_hwg_evidence_io_20261030.py` 和已经冻结的采集资源/读写守卫。本节为作者自审及对 `SECOND_IMPLEMENTATION_PRE_REVIEW_20261030.md` 的答复；精确输入协议及私有作业尚未冻结，状态仍为修订中，未获E0实际运行授权。

ACCEPT：父链从实际 producer RUN/EXIT、独立 VERIFY RUN/EXIT 出发，再读取 producer RUN 原绑定的 observed capture manifest，逐项比较其全部160个 site→metadata 原始引用；payload/fullV1 引用只能来自这些 metadata。新协议不能用同路径、同schema的新SHA替代原来独立验证过的文件。已增加同路径改SHA的拒绝反例。

ACCEPT：HiFloat4/quant 首次导入前检查唯一真实根目录，显式加载已冻结的真实 backend 后，在第一次科学重放前检查普通源码、扩展与namespace来源；最后再次检查。错目录 shadow 的软件反例单列。启动/资产读期间也检查同一个 assigned-GPU monitor，不增加模型计算；只读CPU验证不使用GPU。

ACCEPT：静态组选取针对FP32 batched组投影、FP32 Gram/target-dot后转FP64归一化的已舍入二次式。每source另付一次所选支持的直接FP32 W2，对实际采样target以FP64求SSE，并保存Gram预测与直接向量SSE的闭合残差、全部80次边际。新增近抵消小反例展示真实能量4可在FP32 Gram求和时变0；不称实数精确或全局最优。新增160次selected direct W2和160次校准操作数H2D进入固定计数。

ACCEPT：全M入口显式增加输入/当前gate H2D、proxy分数的fullM abs/sub/div/mul、mean扩展、proxy参数D2H和一次真实坐标设置的收费范围；sample核对包含index_select/D2H，support统计包含所有原布局occupancy/Jaccard/邻居扫描，signed范围包含FP32权重转换及FP64范数/内积/临时张量，feature AC包含其均值与FP64归约。所有调用边界同时记录同步CUDA event区间及host wall；CPU等待可能包含在event区间中，不能当kernel活跃时间相加。整作业wall、显式文件预扣/实际读取、独立raw tensor内存SHA字节、opaque cache loader I/O UNKNOWN、输出文件写入/文件SHA和raw SHA分别记录。调用数不冒充完整FLOPs或部署速度。

全M每source预算拟为1 GPU/4 CPU、76GiB累计CUDA reserved上限、96GiB RSS、64GiB host forecast及128GiB可用内存保留、单host最多2个model、全session最多4GPU；显式输入读256GiB、HF新增32GiB、单文件512MiB、6小时/外层加120秒、单次失败不重试。两个expert原bias从已核整文件SHA的原checkpoint精确读取header与80个BF16 slice；不使用cache bias替代原runtime bias。输入约为95.5GB配对张量+56.2GB cache+57.2GB原checkpoint SHA，低于256GiB，其余源码/清单/小slice另扣。每pair把所有掩码和proxy参数合写约224MiB的一个文件，80个pair约18GiB，另留manifest/调用/失败预算。逐channel诊断占主要掩码体积，原坐标、shuffle与尾组均保留。

峰值推导按真实存活对象而非单GEMM：两状态V1/Z/H/Q2input/V2/Y约7.73GiB，五个fullM score约8.15GiB，proxy参数/两个preactivation、delta、full FP32输出及有效权重等固定对象约4–5GiB；最坏逐token K1稳定完整排序需额外完整index/order、能量/掩码及pad临时，signed诊断的FP64双份hidden范数临时另约6.53GiB，两者串行。保守给固定和最大临时合计≤56GiB，再给allocator/workspace约20GiB余量，以实际累计reserved门限为最终失败边界；这不是已测峰值。结束每pair释放大对象，compact掩码D2H合计约224MiB/host，不驻留所有pair GPU张量。原生导入和大M shape下的实际中性、数值和峰值均仍待获准真实RUN验证。

## 最新收尾比较范围：保留来源96失败，改用三来源

用户最新要求“你快比较完收尾啊”：只完成当前核心等预算比较、汇总提交后结束，不展开E1新算子或E2新候选视频。原四来源采集身份保持；来源96的PART1_NS_V3已实际EXIT1，0个generation phase，错误为assigned GPU monitor的TimeoutExpired，显式资产读取约14.4GiB；不自动重跑。该失败不是候选质量失败，也不能记为第四来源数值完成。

本批拟比较成功采集及独立VERIFY闭合的898/1040/979，全部双expert×40层fullM。静态校准改为898/1040，明确新版 `CAL2_898_1040_SIGNED_FORWARD_GREEDY_K64_V2`；两者在任何E0数值结果产生前指定，979保留为历史暴露开发来源而非盲测。四来源初始设计、原898/96校准草案和来源96失败原样可追溯；最终表须给出三来源实际结果及第四来源失败/未测，不能继续写四源均值。动态/随机/P64/paid/mean等58arm和每source80对预算不变，合计拟240对、13680次candidate FP32 W2加240次full-delta参考投影；61项M1/软件fixture实际通过，E0仍未登记未执行，等待精确输入闭合及独立复审。

收尾CPU汇总另有 `a5_hwg_registered_core_summary_20261030.py`：只在三份E0实际EXIT0后逐一核其RUN/EXIT/原协议、80site×58arm及完整operation序号，以单遍文件SHA和typed raw SHA独立验证240个compact mask/proxy文件。先按每源80对等权，再按三个image cluster等权；固定1024次source bootstrap仅描述这三个曝光图，其中两个为校准、一个为开发，不当成盲确认。保存逐source/site/arm的支持、真实物理布局、signed cross、误差及paid CUDA event区间/host wall CSV。拟CPU1、GPU0、RSS512MiB、显式读96GiB、HF新增256MiB、2小时+120秒、单次不重试；其prepare也只绑定实际闭合输入，不授权启动。三项纯软件fixture已独立于原61项通过，涵盖source不随嵌套样本数加权、bootstrap可重现/缺失来源拒绝，以及即使更新整文件SHA仍拒绝错误raw mask。CPU汇总同样待精确协议/不同评议者/ACTIVE新范围。前两expert checkpoint各113,512字节header的M0检查确认全部80个bias key存在，FFN.0/2分别为BF16[13824]/[5120]；未读取bias或weight数值。
