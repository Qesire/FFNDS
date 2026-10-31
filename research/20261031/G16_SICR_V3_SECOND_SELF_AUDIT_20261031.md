# G16 · SICR v3 第二轮自查：数学、计算边界、因果、系统与新颖性

2026-10-31 · AUTHOR_SECOND_SELF_AUDIT / STATIC_ALGEBRA_CHECKED / NO_NATIVE_RUN / NO_VIDEO_RUN / NO_INDEPENDENT_RECHECK

对应[G16具体修订](G16_SICR_V3_FUNCTIONAL_MARGINAL_STATIC_FFN_REVIEW_20261031.md)，原始[G15](G15_SICR_CROSS_LAYER_STEP_CAUSAL_STATIC_QUANT_CACHE_REVIEW_20261031.md)。本审查是同一助手对用户提供的外部评议逐项采纳后的**第二轮自查**，**不是仓库协议要求的真实不同Agent independent recheck**。

## A. 逐项纠偏（原审稿人的主张与本次处置）

| 外部意见 | 处置 | 具体变化 |
| --- | --- | --- |
| SICR只是框架 | ACCEPT | G16 §2提出Q-MARG，明确真视频边际estimand、可算z/r、精确代理边际与组更新算法；不给算法或视频收益背书 |
| 单缓存与Chipmunk增量同函数、非首创 | ACCEPT | S0改称单残差充分缓存，定位为执行/内存候选而不是独立算法创新 |
| C缓存构建可吞掉收益 | ACCEPT | G16 §3给出REDO selected / SPLIT disjoint / DUAL accumulator三条物理路径，68.5185/77.7778均只算W1/W2 MAC |
| S1需额外缓存 | ACCEPT | 动态α必须增加历史投影R0,j，明确(r+1)个D张量和获取/构建成本 |
| “最小缓存”未证 | ACCEPT+CORRECT | rank下界限于任意旧h与线性精确编码；α≠1应分析全B_alpha而不只遗漏组W2 |
| 跨层J/K不是算法 | ACCEPT | 不构造稠密K；用p个预定义readout后缀VJP/有限差的组投影z、当前r与有限成本选择规则 |
| 论文新颖性过宽 | ACCEPT | 对TOPLAS98/Chipmunk/ICC/CacheQuant/QuantSparse/SVOO/MC-Sparse/FlowQuant/Test-time sparsity/GOC/SVD-Cache进行最近机制交叉；仅声称未验证特定问题空间 |
| 缺最终视频价值目标 | ACCEPT | G16 §2.1给出各轴真实反事实边际，§5多轴约束；局部代理只排序 |
| High/Low误差传播缺判别 | ACCEPT | V3强制同K H/L 80/80、64/96、96/64与单个非均匀层预算、全学生视频；承认expert与step共线 |
| 大规模实验过早 | ACCEPT | V1/V2/V3先证伪，再确定新独立P2来源和功效；不改ACTIVE |

## B. 确定性代数/计数自查结果（独立于Wan、不是GPU科学实验）

在V8隔离环境以固定种子20261031的LCG生成250套小维数样本：7个组、每组一个D=4输出向量、固定支持组索引{0,2,5}、独立随机历史/当前贡献与增益。每套检查：

1. 固定gain：C=Y0−Σ_S αP0、C+Σ_S αP1、Y0+Σ_S α(P1−P0)三者的实数恒等式，最大欧氏差9.992007221626409e−16；
2. 低秩动态gain：α=α0+θu，C0+Σ_S α0P1+θ(Σ_S uP1−Σ_S uP0)与Y0+Σ_S α(P1−P0)相比，最大差1.0007415106216803e−15；
3. 错误旧缓存漏掉R0=Σ_S uP0时，错误表达与真值之差等于θR0，最大恒等式残差9.10291908440698e−16；
4. 边际二次代理：||r||²−||r−z||²=2⟨r,z⟩−||z||²，最大绝对差8.881784197001252e−15；
5. 四步W1/W2理论比：H=13824、K_H=K_L=5120，理想无重复组W2构建为0.6851851851851851，额外首次selected W2构建为0.7777777777777778；
6. M=31668,D=5120,40层,BF16单C缓存12.08038330078125GiB；完整h+Y缓存44.697418212890625GiB。

上述是**小矩阵代数和算术自检**，而不是Wan真实h/gate/QDQ、缓存构建kernel、视频输出或速度的检查。随机矩阵没有测试BF16舍入、group64闭包、空间token、物理rank、sparse kernel和真实系统内存峰。执行未绑定Github Action/RUN，不能称正式复核收据。

## C. 数学命题分类：哪些已经证明、哪些尚未证明？

| 命题 | 身份 | 证据/缺口 |
| --- | --- | --- |
| 固定α的单C表示等于历史Y+组增量 | ALG EXACT（实数） | 直接展开；250套小维数自查 |
| rank(B_alpha)给出一般α下精确线性历史表示下界 | ALG CONDITIONAL | 令B_alpha=EF，rank(B_alpha)≤rank(E)≤r；前提旧h任意且线性编码/解码；不是bit最小性 |
| 动态α要求额外历史投影（在给定α基展开时） | ALG CONDITIONAL | 展开α=α0+Uθ；特定U可能退化、可合并或存在其他编码；不能宣称r+1是全算法普适最小维数 |
| 全层线性读出代理下精确边际公式 | ALG EXACT FOR SURROGATE | 固定参考z和α=1时直接平方展开；不能外推真实Q_j(V) |
| 随机/已定义R中的后缀z能够预测视频动作 | OPEN | 需来源外真实视频排名、反例、非光滑输出与student drift |
| meta比同成本BF16短摘要更有信息价值 | OPEN | 当已有完整x,W,recipe时，确定性meta没有额外香农信息；需证明低成本/截止时点优势 |
| Low真实质量更敏感且应保留更多组 | VIDEO HYPOTHESIS | 既有2来源P64支持条件性观察，尚无来源外普遍最优预算 |
| E2/E3能无额外selected GEMM构造C并提速 | SYSTEM HYPOTHESIS | 需native完整代码、kernel计时、copy/offload/pad计费；68.52%仅理论主MAC |
| 最终视频Pareto优于Chipmunk/TEAL/SharQ | OPEN | 原生部署与真实视频完全未执行 |

### 形式更严谨的rank证明

固定α时，令B_alpha=diag(β)W2，C−b=hB_alpha。若编码为hE∈R^r、解码为(hE)F，对**所有**h∈R^H精确，必有EF=B_alpha，故rank(B_alpha)≤r。反过来rank因子分解允许r=rank(B_alpha)的精确线性存储（忽略浮点、读/算成本）；所以这是**这个受限数学问题的维数最小值**。真实数据流形、量化有限字节、计算预算和读写/预测侧信息均改变问题。

### 非线性/量化硬边界

- P_i、J_i若来自BF16或HiF4局部数值流，分段导数在码区间边界不支持全局线性结论。
- 同一步FFN gate当前值在局部pre-gate注入下可当固定线性系数；跨输入/自由学生轨迹其导数与Attention/Norm/Adapter/Scheduler共同变化。
- source×seed相互独立程度必须按独立image cluster构建；一个视频的81帧/21 latent槽/40层绝非独立来源n。
- 凡称“static”，需实际mask及编译weight-layout静态；读出增益可以动态，但若用未来Q1决策必须明示时点、临时量与历史投影。
- 后缀z若只在部分层采集，算法全80位点不能私下补零或声称已完整优化；未采集位置应保持预先冻结的保守基线或标MISSING，不将缺失当0影响。

## D. 最强反例与主动失败路径

1. **高能互抵**：z_a=+u, z_b=−u，单组对角分数会让出预算，联合边际考虑r可以得到不同选择；但真实非线性后缀可能破坏抵消。
2. **投影失明**：R z≈0但少量帧的小物体身份完全改变；所有p维线性读出都可能盲区，真实视频检验必须有语义、连续时间与视觉审阅。
3. **首步成本反噬**：拆W2为two GEMM带来的低利用率/多launch、额外同步或新W2副本使系统wall更慢；不得从理想MAC写实际加速结论。
4. **qmeta冗余**：给定x/W/recipe时Z_Q是确定性函数，独立增量信息0；如不优于同成本ordinary BF16 summary，则不支持Qmeta创新假说。
5. **教师与学生状态错配**：teacher上计算的v/z在学生前缀发生Q2切换、gate调制后排序失效；需要完整free-rollout。
6. **High/Low共线**：冻结HHLL不能隔离专家权重与去噪step敏感性；只报告预算策略而非专家因果。
7. **双阶段独立样本不足**：CAL1 source1040、E0 898/979、Chipmunk 96等均为历史曝光；不凭3/4来源宣称来源外泛化。
8. **高成本校准**：全M长序列的VJP可能超过前沿方法允许校准成本；给出实际GPU秒/访问量，并与更廉价的Signed Gram/P64比较。不能用后续O(N A p)得分计算掩盖特征生产代价。

## E. 论文新颖性审查的积极结论与限定

现有最接近研究已包括：
- TOPLAS 1998 static cache pruning；CVPR2025 Increment-Calibrated Caching；CVPR2025 CacheQuant；
- ICLR2026 QuantSparse（视频attention）、ICML2026 SVOO（离线层敏感度）、2026-10 MC-Sparse（sparse attention metadata/cache residual）、2026 FlowQuant（feature-flow与cache/quant）、CVPR2026 Test-time Sparsity（动作扩散）、ICCV2025 GOC（gradient temporal correction）、CVPR2026 SVD-Cache。

链接与方法差异详见[G16 §6](G16_SICR_V3_FUNCTIONAL_MARGINAL_STATIC_FFN_REVIEW_20261031.md)。目前可提出但尚未证实的独特问题是：Wan 4step FFN 的**完全静态K64组**，使用合法时点的signed scale/code与少量全层后缀观察，可否预测来源外视频功能边际，达到Chipmunk/TEAL/SharQ/局部SIGNED不能实现的质量—实测wall—显存Pareto。

这个问题符合论文实验假设的粒度，**尚不是论文贡献结论**；若三项必经门（信息独特性、真实native成本、视频功能保真）任何一项失败，应重新定位为机制负证据或限制范围，而不是继续扩大宣传。

## F. 第二轮自查决定与下一步审查门

**SELF_AUDIT_DISPOSITION = REVISE_WITH_TESTABLE_ALGORITHM**（作者自查通过“可作为新候选设计”门，未通过“新方法可部署/论文新颖性”门）。

继续阅读与代数推导可以进行。实际实验按顺序：
1. V1先审原生cache构建E1/E2/E3与同函数控制，明确首步两W2输出生成是否真的有效；
2. V2随后审少量source×group的p后缀投影是否物理可付、来源外能否预测视频组边际，量化摘要与同成本BF16特征竞争；
3. V3再审同K High/Low+跨层预算的独立来源视频与多轴质量；
4. 确认实验P2另行设计统计功效、图像簇与完整成本收据。

实际CPU科学实验、GPU、data acquisition、native benchmark、video仍需**真实不同Agent**独立review/recheck、ACTIVE冻结与实际RUN/EXIT；本轮没有执行这些实验。
