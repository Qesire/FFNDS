# 下一步整合实验方案：独立最终审阅

日期：2026-11-16。审阅基线为 main `5a5b84a04f71aa1fa6c426febf614a1a4c1d2492`。当前对象是该基线之后的完整下一步方案，不沿用早期 99/897 条方案的活动额度。

## 1. 结论与审阅范围

**结论：当前版本可作为有限、可辨识且预算明确的下一步实验设计发布。** 主链条已经统一为作者模块与 Wan 接口验收、真实数据作者式 ICC 校准、固定输入局部比较、冻结主方法、五臂自由视频以及有限消融。它能够回答在给定 Wan 部署条件下，不同完整 FFN 路径表现在哪里不同，并用局部误差和适配诊断约束解释。

该结论是设计及静态一致性审阅，不是模型数值验收、已完成复现、资源可用性证明或执行 scope 批准。本轮新增模型/GPU运行 **0**；真实输入、实现产物及资源仍待绑定。所有“通过”均指下述静态检查，不能转写为作者模块、Triton 核、Wan neutral 或完整视频已经通过。

主要依据为 [统一方案](CONTROLLED_REPRODUCTION_PLAN.md)、[机器预算](EXPERIMENT_BUDGET.json)、[作者校准配方](ICC_CALIBRATION_RECIPE.json)、[数据角色](COHORT_PROPOSAL.json)、[ICC 参考](ICC_STRICT_REFERENCE.md)及[作者校准说明](ICC_AUTHOR_CALIBRATION.md)。下方最终身份覆盖九个文件；文件路径相对 `research/20261107/ffn_baseline_reproduction`。

## 2. 已修补的科学与实施缺口

### 冻结顺序不再自相矛盾

两个 Dense/hook 捕获是主 55 个槽中的两项，可以先于 ICC 家族和 E6 冻结执行，为 neutral、真实 full-M Chip 及局部共同输入提供数据。家族、参数、CAL22 开关及 E5/E6 选择最迟在任何新近似臂自由视频输出可见之前冻结，且仅依预登记局部规则。这样既不要求先有未选择的 ICC 才能构建输入，也不允许看近似视频后反向选择。

P1 的依赖已明确为代码、权重、DEV 输入、接口实现和宿主资源；真实 clean CAL11 及 CAL3 原始 μ/logvar 是 P2 的额外依赖。缺少真实视频不能被解释成所有静态模块工作都必须停止，也不能用普通 SVD 填补 CA/CD 后宣称完成三家族选择。

### 作者式校准与必要适配有独立身份

主 CA/CD 从真实 clean 视频编码或可追溯 x0 起步，经 native flow 前向加噪。CA 在每专家两个合法 D4 时点均匀选一个；CD 使用该专家唯一合法相邻 pair，第一次 Dense 调用后执行原生 Euler 续步，再作同权重、同条件第二次调用。第二点不重新 q-noise、不 reset、不引用旧自由 rollout。作者条件 dropout 映射为真实空文本 embedding，I2V 条件保留，CFG off 仍是单次模型前向。

目标 normalized mean、class-null→empty-text 及 D4 支持均显式作为 Wan 适配。预先选定 CAL3 上的 posterior 与文本互换分别相对固定主配方变化一项，不把诊断结果用于改选主配方。posterior 的 raw μ/logvar、FP32 η、clamp 和整段规范化顺序已明确；H-CD 在 σ=1 且完整输入一致时可复用整 pair。诊断仅支持编码、统计及流场预测敏感性，不能证明重新拟合后的因子或终端视频等价。

旧 rollout 统计及其 CA/CD 因子不能抵扣新采集和分解；原图文 CAL11/EXT11 不是自动获得了真实视频身份的 clean 数据集。作者 256 个校准样本也不解释为 256 条完整采样轨迹；本轮数量明确沿 11/22 源量级。

### 局部选择、共同机制表和目标张量分开

三家族的完整模块与统计 oracle 保留。主选择仅使用 DEV11 × 六位点 × 三家族、固定 r1408/四状态的 198 格，属于 792 格总局部网格的子集；r512 和两残差只作诊断。选定族补齐全部 80 个 expert/layer 参数，未选族仅有规定六位点的局部证据，不能写成全部三家族都完成了视频级评估。

两个 neutral 来源的新 full-U 捕获提供对应 P512/64 输入及当前 gate，其余九源用身份兼容的既有 bank；评分前冻结一个统一输入 manifest，每格只算一次。canonical anchor/target 均由 own-W BF16、P512 形状 native_ffn 构建；捕获的 full-M Y 单独保存为 Y_fullM_panel，不能混作 Y_local512。共同两源表同时报告已有两种 Dense 目标之间的数值差，避免把形状及运算路径的差额直接归因低秩或 selector。

作者 Chip 的 true-M 局部证据只覆盖两个共同来源。它与 ICC/Q1 的比较取共同两源子集，不能把两源作者 Chip 均值与十一源 ICC 均值直接排序。最终五臂视频统一 DEV11，但仍是开发比较。

### 数值分解不再超过其证据能力

局部风险分母为 P×d，其中 P=512、d=5120；完整 token 数 N=31668 用于主校准及真实 M 路径，不能混入 panel 评分分母。

主 ICC 66 格的实际误差拆分使用保存的 BF16 原值先升 F64 再作差，保存的真实 ĥ 不重新预测。A=B2_F64(h_b−h_a)、B=B2_F64(ĥ_b−h_a)，据此定义 E_read、E_first 和剩余 E_num。每格两次 B2 action，总 132 次 action，即 264 次因子 GEMM。E_read 包含 native Dense 数值路径，不能称为纯几何投影误差；残差定义使闭合不是独立机制验证，只在 F64 舍入尺度检查能量及带符号交叉项。

额外精度对照封口为选定族 r1408、四个固定 pair、两种缓存、两个 FP32/TF32 设置，共 16 个模块格。固定 native BF16 anchor/输入/因子数值，仅指定缓存与增量算术升精度；不新增纯 FP32 模型、因子重拟合或自由视频网格。BF16 对应格复用 792 的子集。

## 3. 预算独立核算

下表均为当前设计上界；实际复用只能凭完整输入、因子和执行身份抵扣。

| 项目 | 独立核算与解释 |
|---|---|
| 主作者式 CAL11 | 11×2 专家×(CA1+CD2)=66 次模型前向 |
| 可选 CAL22 | 22×2×3=132 次累计前向，前 11 源不重抽 |
| 必要 CAL3 诊断 | posterior≤3×4=12；文本互换≤3×6=18；合计≤30 |
| 主/扩展校准含同一诊断 | **96 / 162 NFE**，均为重试前上界 |
| 主五臂 | 5×DEV11×seed42=55 个 D4 槽，含两个提前 Dense 捕获 |
| 基本视频 | 55+neutral 额外≤10+同因子 2R 三条=**68** |
| 条件分支 | E5≤3，E6≤3；全开合计≤**74** 条，非必跑 74 条 |
| 技术重试 | 全局共用≤3；视频一次≤4 NFE，CA/CD 原子统计事务一次≤2 NFE |
| 视频最大数 | **77** 仅在三个重试全部分配给视频时成立 |
| 联合前向上界 | 4(74+r_video)+162+2r_cal≤**470**，r_video+r_cal≤3 |
| 主 ICC 局部格 | 3族×2秩×2缓存×6位点×11源=**792**，含 198 主选择格 |
| 可选 CAL22 局部格 | CA/CD×r1408×4S×6×11=132；ICC 总≤924，不重复普通 SVD |
| 其它局部计算 | Q1/GP64 P512 各66；作者 Chip/GP64 true-M 各12；共享 native P512 端点132；B2 action132；精度附加16 |
| Wan 大矩阵 SVD | 主 CAL11 **184**；可选 CAL22 局部诊断增加24至**208**；E5 最多再增148至**356** |
| 其它离线拟合 | 原作者 shape 分解6；缺兼容 Q1 参数时至多80个固定规则 PCG 拟合，不新增搜索 |
| full-U 捕获 | 24个 BF16 U 张量，payload **7,782,727,680 B**；归档≤9 GiB；额外 staging≤1 GiB；不加 D4 槽 |
| 候选中间量 | payload **6,458,179,584 B**，约6.015 GiB；临时上限**7 GiB**，不与 full-U 归档混算 |

共享重试枚举已核对：三个视频重试时联合 470 NFE；三个校准重试时为 74 条视频加 168 校准/诊断 NFE，即 464。不得同时把 77 条视频和额外三个校准重试相加。CA 单调用、CD 两调用作为不可分统计事务，失败回滚后用相同随机记录重试；不是按整源六次前向重复另计。

470 是去噪器前向调用数上限，不是总算力或时长上限。VAE/T5 编码、模块 oracle、真实 M selector/核、离线 SVD/PCG、F64 factor action、搬运和质量评分另账。不能以旧“99→77 条视频”推导总计算下降。大矩阵 SVD、四状态缓存及实际 GPU/CPU/offload 代价仍需真实资源绑定；当前不能称成本已经换算为可执行时长。

## 4. 结论必须保留的限制

1. **局部选择与视频都来自已暴露 DEV11。** 预登记选择规则避免看新近似视频调参，但不把这十一源变成独立留出。本轮能报告选中实现的开发表现，不能声称确认性优越、全域最佳 ICC 或事后定义非劣界。
2. **ICC 与 Q1 的校准来源目前没有逐源匹配。** ICC 用真实 clean 视频，Q1保留旧 CAL11 图文/轨迹拟合身份。共同局部输入能够控制评估状态，不能消除拟合分布差异；五臂主结果首先是完整 pipeline 比较。Q1 全80参数/producer未绑定前，也不能用 BWS 混合 CAL3 参数冒充原身份。
3. **多个历史到当前的变化同时存在。** 旧 ICC512/2R 到当前 r1408/4S/新校准的差异是组合变化；E4只隔离缓存形式，rank前缀只提供局部对照。作者 Chip 与 Global-P64 同时改变代理、支持集、选择器、算术和层覆盖，五臂差异不能压缩成单因素原因。
4. **必要适配证据按层级解释。** CAL3和16格精度诊断不等于完整视频影响为零；E6仅按冻结的 tail→precision 优先顺序覆盖一个合法、非等价运行时变化，其余终端影响保持未决。oracle错误先修复，不作为质量实验候选。
5. **旧 Wan/蒸馏证据按原身份使用。** A/BW/BWS不重开，不改其数据、参数、运行与结论。BWS的CAL3及D4/O40复合差异不能与BW的CAL11绝对风险混池，也不能解释为纯步数/CFG效应；本轮不新增O40、原系统整视频或LLM训练。Wan与LLM的联系限于计算图、复用机制和硬件条件，不能借旧证据声称跨域质量或蒸馏因果结论。
6. **来源是统计单位。** 实际来源簇、近重复和已暴露身份必须保留；token、帧、六位点和多方法配对不能扩大独立样本数。若实际簇少于11，应报告有效簇数，不能静默更换不利结果。

最终措辞也已统一：ICC_AUTHOR_CALIBRATION §5 明确 CAL22 只比较既定 r1408/四状态得分，主家族和秩维持 CAL11 冻结；r512 仅属 CAL11 诊断，没有额外秩搜索。

## 5. 实际完成的核查与待落实项

本轮纯 Python 静态核查 **19/19 通过，实际退出码0**。核查覆盖66/132/30/96/162/470、共享重试各分配、55/68/74/77视频数、局部格和分解数、存储payload、主采样/条件/统计元数据、冻结时序，以及队列/ledger的历史已完成字段。与main基线快照比较，A/BW/BWS既有关键完成记录和判断字段保持不变。详见[独立预算回执](INDEPENDENT_BUDGET_CHECK.json)。

本轮没有重跑未改变的代码引述 verifier，没有重跑历史 counter，也没有把文件存在或静态公式通过当作模型运行。固定代码引述身份随最终清单记录；其既有核验记录继续按原提交和原执行身份使用。

下一步执行仍需落实：11条真实clip或可追溯x0及来源簇；CAL3同编码器raw μ/logvar或可重新编码的真实clip；文本、首帧I2V、噪声与时间/条件记录；实际checkpoint、native solver/encoder/collector/adapter字节；Q1原CAL11全80参数及producer身份；真实硬件、dtype/kernel和offload能力。随后按统一阶段产生模块oracle、160次FFN trace、neutral、统计与分解、选择表、完整视频及成本回执。以上是方案内既有工作，不另建审批层级，不在资源未证实时宣称可派发运行。

## 6. 最终覆盖文件及身份

以下路径相对 `research/20261107/ffn_baseline_reproduction`。内容审阅覆盖方案、预算、配方、cohort、ICC专项和两个入口；代码引述文件仅记录固定身份，不重复既有核验。机器可读版本见[最终文件身份](FINAL_REVIEW_FILE_IDENTITIES.json)。

| 相对路径 | 字节数 | SHA-256 |
|---|---:|---|
| `strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md` | 33709 | `913f032c6c1c73e6f015004dde3620a638774c5ef814abf969a02fa53f0383e4` |
| `strict_reproduction_20261116/EXPERIMENT_BUDGET.json` | 11901 | `73ef13b3066e9988954f0feceab7ebcfdaee851e258745800a9f76904a0e29ba` |
| `strict_reproduction_20261116/ICC_CALIBRATION_RECIPE.json` | 8016 | `d2a1b278c9b62b491586e7d53c8e9aec995942c004faf12a946cd20d6ab39f32` |
| `strict_reproduction_20261116/COHORT_PROPOSAL.json` | 16356 | `805026d67fd58db232d6d9d77e70e99e017f3ec79de003c2d5d7e8465f20458d` |
| `strict_reproduction_20261116/ICC_AUTHOR_CALIBRATION.md` | 20726 | `17bd99bd1ab93093e68b08d20b7560653dcfd86e6138840e6f2e73a78fd1c614` |
| `strict_reproduction_20261116/ICC_STRICT_REFERENCE.md` | 18741 | `b2e03519183886fc0465db77dd052a241d488fec6e75b3bb3dd122c31af22f8f` |
| `strict_reproduction_20261116/CODE_CITATIONS.json` | 104433 | `3777e3ecfa0bbc8f02a2fc947360d2f07c2f3b448b60efe04d57963736701bdb` |
| `strict_reproduction_20261116/README.md` | 4112 | `0e60389f22ceeb198568dfaf2736dcbadafed5052618c4a854f0951d2037db5c` |
| `STRICT_REPRODUCTION_AND_REDO_20261116.md` | 3224 | `80a88fdb5df1cbd8a7b70c85ac1071fc5ba09a48e71873ded582884a9aa73cb8` |

审阅回执绑定预算 SHA-256 `73ef13b3066e9988954f0feceab7ebcfdaee851e258745800a9f76904a0e29ba`，配方 SHA-256 `d2a1b278c9b62b491586e7d53c8e9aec995942c004faf12a946cd20d6ab39f32`；联合前向上限 `joint_nfe_cap=470`，`new_model_runs=0`。若上述核心字节变化，当前身份清单不自动覆盖变化后的版本。
