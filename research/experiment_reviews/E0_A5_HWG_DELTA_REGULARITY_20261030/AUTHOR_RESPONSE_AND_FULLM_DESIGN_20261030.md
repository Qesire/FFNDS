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

来源898/96拟用于开发静态组选取；1040/979仍是历史暴露开发源，不是盲确认。静态每site在两校准source的实际postQ2 diagonal-energy归一化后等权排序；真实输出artifact和SHA在开发source运行前冻结，禁止看其结果改组。cal自身结果与两个dev分开，不将320层对当独立n。来源等权向量和嵌套source bootstrap只描述这4个开发source，不能外推未见来源。

保存每site有效权重/代理参数身份、真实输入/样本重放核对、score recipe、全量compact mask（逐channel用bitpack，组mask保留row layout、permutation和真实tail）、真实随机seed、可确定重放依赖、同K指标/支持/物理原布局occupancy、局部空间/τ及signed质量/舍入项、所有GEMM成本和RUN/EXIT。不会保存多份fullM代理浮点中间值来挤占GPU/磁盘；可重放来源、参数与确切计算recipe显式绑定。模型/cache沿HF复用，不新下载。候选实际selected W1/Q2/W2和完整学生质量仍分别进入另审E1/E2。
