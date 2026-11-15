# ICC / Chipmunk：先完整复现，再重做 Wan 比较

2026-11-16（北京时间）。依据 FFNDS `8fbc81411d2840709166923788612165beb15ee0`、本轮用户最新要求及再次核对的作者默认分支。当前为**源码审计、接口检查与重做规格**；本轮模型/GPU运行数为0。详细上游审计、设计复审与输入提案在 [本轮材料](strict_reproduction_20261116/) 中。

## 1. 本轮更改的判断

**需要重新建立并执行作者参考基线。** 过去“核心公式一致”“已有同mask数值比较”不足以保证整个复现完整：它们没有同时证明作者配置、所有状态分支、校准生产器、数值执行顺序和完整自由生成均与所引用的方法相符。

旧视频、A/BW/BWS与各自RUN/EXIT继续保留；已完成范围不重新标成未完成。旧ICC结果明确属于所运行的SVD/CD、rank512、双残差和当时校准配方；Global-P64属于FFNDS变体；历史author-style Block128在producer/编译扩展/数值链未闭合前不能升格为作者完整复现。关于末级低秩、代理相消和复合状态差异的代数/局部发现仍有效，但不能据此认定原作者方法已被公平证伪。

本轮先把下面三层分别做完整，再讨论哪一层出现质量损失或高成本。

| 层次 | 实际对象 | 完成条件 |
| --- | --- | --- |
| 作者参考 | 固定论文与代码版本，调用作者原FFN模块、分解器、selector、kernel和状态机 | 原形状/精度的可执行正对照；覆盖dense、hit、refresh、多hit、cache和bias；记录论文与代码不一致之处 |
| 最小Wan端口 | 只补Wan宿主接线、真实调度/专家隔离、必要shape与设备生命周期 | 与参考同输入/同状态对照；原生neutral；全部站点trace；有选择余地的适配单项消融 |
| 实验/部署变体 | ICC双残差、Chip Global-P64、exact-K、数值重排、性能优化 | 从已验收端口一次改一项；分别报告质量、实际成本与改动原因 |

这里“完整FFN复现”包括被比较FFN算法的全部组成。ICC整篇方法也处理attention，Chipmunk论文的视频系统只稀疏attention；FFN-only移植不能称为完整论文系统复现。作者原模块的正对照必须做；完整FLUX/DiT论文性能表和LLM基准不作为本次FFN移植的额外前置。官方Wan2.1系统的小配对可以核接线，不能替代MLP验证。

## 2. Chipmunk到底在Wan上做过什么

本轮重新读取仓库默认分支：Chipmunk默认是`master`，HEAD仍为`b5e8d1c3da6bddc7ca9007c48e91b82087c11490`；较旧的`main`不能替代它。ICC默认也是`master`，HEAD仍为`dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4`。

| 问题 | 依据原文/当前代码的答案 |
| --- | --- |
| 作者是否做过Wan？ | 做过Wan2.1 T2V、50步。论文的约2.67倍是Chipmunk与STA/step-cache等组成的系统结果，不能当作FFN-only倍速。 |
| 论文在Wan上验证了稀疏MLP吗？ | 附录D.1明确Wan/Hunyuan最终视频配置只使用稀疏attention；MLP稀疏在FLUX中使用。 |
| 原Wan示例是否已接MLP？ | `WanModel`只注册`SparseDiffAttn`，`is_mlp_sparse=False`，FFN仍为普通Linear→GELU→Linear。只改YAML不能接通MLP。 |
| 默认脚本实际运行哪个配置？ | `examples/wan/run.sh`使用`configs/cuda-test/chipmunk-config.yml`，其中attention top_keys=.1；根目录YAML的.15不是同一个运行配置。两者均关闭MLP。 |
| 原码直接支持I2V吗？ | 作者issue回复说尚无I2V性能数据，建议补初始化和`inference_step`。当前默认分支的I2V入口仍缺这两项；文件存在不等于接线完整。 |
| A100是否迫使我们改写MLP算法？ | 不会。官方已有Ampere/Triton路径；保留作者`SparseDiffMlp`、Triton两级MLP和compiled selector/copy工具。实际环境是否编译通过另以运行证明。 |

原始依据：[论文D.1](https://arxiv.org/html/2506.03275v1)、[官方仓库](https://github.com/sandyresearch/chipmunk/tree/b5e8d1c3da6bddc7ca9007c48e91b82087c11490)、[作者I2V回复](https://github.com/sandyresearch/chipmunk/issues/2#issuecomment-2977261223)、[官方MLP](https://github.com/sandyresearch/chipmunk/blob/b5e8d1c3da6bddc7ca9007c48e91b82087c11490/src/chipmunk/modules/mlp.py)。逐层调用链见[上游审计](strict_reproduction_20261116/CHIPMUNK_UPSTREAM_AUDIT.md)。原T2V的硬编码静态mask几何也存在乘积不符：32760被映射到21×45×80=75600，21840被映射到21×30×52=32760；接线时应从真实patch grid导出并验证，不能照搬这张表。此为源码发现，未声称本轮实际启动过原系统。

## 3. ICC正式参考必须恢复四状态与完整分解配方

以列向量记号简写单token，原生FFN为

\[
z_s=W_1u_s+b_1,\quad h_s=\operatorname{GELU}(z_s),\quad y_s=W_2h_s+b_2.
\]

作者命中路径保存并使用四状态：

\[
\hat z_t=z_s+B_1(u_t-u_s),\qquad
\hat y_t=y_s+B_2\left[\operatorname{GELU}(\hat z_t)-h_s\right].
\]

`B1/B2`各自用两次低秩linear执行，保留两次减法、加回缓存、中间GELU、因子顺序及原有anchor不随hit更新的规则。bias在完整锚点中进入；增量路径不重复加bias；Wan当前gate仍在模块外执行。原DiT的无效dropout/identity norm可按真实配置映射，不能删掉任何有实际作用的运算。

**SVD、CA-SVD、CD-SVD均建立参考，不能只用普通SVD代替论文主张。** CA/CD四个边界是`u,z,h,y`；CD按作者顺序做配对差绝对值、token归约、样本内通道min-max、batch/来源聚合、平方根、左右`max_ratio=1000`约束、加权SVD与反缩放。先读并执行作者函数，再接入Wan真实边界。历史`weights(max_ratio=None)`的默认值不能当作已采用作者截断；真实生成参数必须写入因子清单。

分解精度不只写“FP32”：还要绑定作者实际`torch.linalg.svd`调用、driver、full_matrices、TF32开关和因子拆分。先以直接原函数为oracle；若换成仓库的`gesvd`/reduced工程路径，单列对照。SVD向量可有符号或退化子空间的不唯一性，因此比较重建与对真实输入的factor action，不要求不同合法分解的因子SHA相同。scale的epsilon仅在一次分解定义中加入；多个rank从同次分解截取，不能对共享scale反复原地加epsilon。

两残差

\[
q_1=z_s-B_1u_s,\quad q_2=y_s-B_2h_s
\]

只在实数代数下等价。它改变BF16运算顺序，并增加anchor低秩投影。因此四状态是正式参考，双残差是使用**同一组因子**的单项消融；二者均需自由生成比较，不能以局部CAL误差接近替代。

四状态40层、当前单专家、BF16缓存约89.39GiB，仅是逻辑payload。优先用无损host offload/逐层预取或经核实的设备分配保留原计算；显存不足不能成为省掉参考基线的理由。记录GPU allocated/reserved、CPU pinned/RSS、PCIe传输、工作区与权重。当前会话没有可用GPU或已连接的集群执行入口，尚未验证任何这类资源方案。

原始依据：[ICC原论文](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Accelerating_Diffusion_Transformer_via_Increment-Calibrated_Caching_with_Channel-Aware_Singular_Value_CVPR_2025_paper.html)、[作者SpMlp](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/models/models_sp_step_decomp.py)、[分解脚本](https://github.com/ccccczzy/icc/blob/dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4/DiT/func/func_decomp.py)。完整实施规格见[ICC专项](strict_reproduction_20261116/ICC_STRICT_REFERENCE.md)。

## 4. Chipmunk正式参考不能遗漏的机制

| 机制 | 作者开源reference应保留 |
| --- | --- |
| 代理 | 每128个真实token分组；当前input均值经W1+b1，与缓存preactivation均值作绝对差 |
| 选择器 | 作者compiled selector：前1024列估计阈值，全列阈值选择、random补选，实际count对齐256；不换为全列quantile或exact-K |
| 默认强度 | top_keys=.3、random_keys=.05；记录实际各块count而非把.3当精确保留率 |
| 层策略 | 每个expert局部前两层保持Dense；reference不改成全40层稀疏 |
| dense刷新 | MLP默认period10；前两步dense是attention的规则，不能抄进MLP |
| mask复用 | block_mask_cache=2，step<10仍每次重算；同时覆盖10前后的refresh/reuse边界 |
| 状态更新 | 每次hit更新选中post-GELU值及输出缓存；仅重算indices时更新相应proxy缓存；多hit按逐列历史状态运行 |
| 数值 | 官方Triton的FP32累积→GELU→BF16以及第二投影的delta写回/缓存相加顺序；本仓packed BF16路径另作消融 |
| 布局 | 直接复用作者voxel/reverse函数，保持token身份；在FFN-only宿主上用局部置换/逆置换保护原生attention，须做等价及成本对照 |
| 物理执行 | 原Triton稀疏GEMM、selector、cache写回和必要搬运均计入；不用dense gather原型速度代表作者kernel速度 |

特别注意：作者dense refresh没有清空旧indices，step10刷新后step11可能复用此前mask。先忠实记录其行为；如修改，另列变体并消融。不能只测试D4的一次anchor→hit，就声称已覆盖作者整个状态机。

## 5. 最小适配清单及其验收

| 改动 | 为什么需要 | 验收/消融 |
| --- | --- | --- |
| Wan pre-W1/pre-gate接线 | 当前norm/modulation、bias、gate和权重所有权属于宿主 | no-op与Dense回退须与原生同路径；逐站点核输入、输出、gate只使用一次 |
| expert/request/CFG缓存key | 两专家权重不同，跨请求/CFG串缓存不合法 | 首次进入expert建立dense状态；cache键与真实scheduler trace逐项相符；不把非法串缓存当质量消融 |
| 显式step/local-layer计数 | 原singleton不能直接表示双专家；已发现末次调用前reset边界 | 保留原类reference；显式scheduler context与160真实调用比对；原/修复小轨迹对照，修复不静默归入原码 |
| MLP尾52行 | N=31668=247×128+52，原block_mean不能reshape全部N | 第一版247整块调用原核，最后52行Dense并计成本；与valid-only尾组方案单项对照；不将补零行混入均值 |
| 视频空间排列 | 作者已支持不整除的3D几何 | 直接用voxel_chunk_no_padding及逆函数；无丢失/重复行，先验证排列往返；无需另造排列算法 |
| 模型offload/转置权重 | SparseDiffMlp的W2转置在普通list中，不自动跟随宿主设备迁移 | 在权重已落设备时构造/重绑；逐层核权重SHA、设备、转置内容；比较常驻与无损offload的输出及成本 |
| 原精度→Wan BF16 | 原ICC图像路径与Wan数值recipe不同 | 保留原精度正对照、同输入BF16与更高精度诊断；原因子顺序不随手改为fused addmm |
| CAL数量/秩/保留率 | 这些是统计与预算选择，有多种合法值 | 有限CAL/DEV选择与下述预算曲线；不能标成无需验证的必要接口改动 |

**本轮已执行一项无模型的纯Python接口检查。** 原`LayerCounter`类原样提取，40层×4步×1调用的最后一次实际call159应是(step3,layer39)，类给出(step0,layer0)；50步×2CFG调用的最后一次也出现相同边界。若把80层注册到一次只运行40层的时钟中，160次有120次坐标错配。[原始trace](strict_reproduction_20261116/UPSTREAM_COUNTER_TRACE.json)。这证明需要明确计数适配，不证明论文作者实际发表运行因此失败；GPU模块尚未执行。

保留前两层Dense时，D4应记录**8固定Dense + 76 anchor + 76 sparse hit = 160站点**。ICC/Q1按其自身原层策略可以是80 anchor+80 hit。正式Pareto报告各方法真实策略和总成本；共同38层范围是另一个机制控制，不静默写成ICC原reference。

## 6. 先修实现，再开展有限重做

### 6.1 作者代码与最小端口的正对照

一次准备覆盖以下真正不同的分支，不把每项检查再拆成重复完整视频。

1. 原模块/原形状/原精度正对照，固定实际编译二进制和输入；ICC核四状态和分解器，Chip核原selector与原Triton。
2. dense回退、零增量、固定mask、全列/空列，以及有变化的连续多hit；分别比较输出和所有被更新/应保持的缓存。全列稀疏路径与native dense可能因作者融合舍入不同，必须先用同算术reference解释，不能伪要求BF16逐位等于另一算法。
3. Chip selector检查有效count、范围、填满与对齐；保留随机策略。原子顺序可能影响补齐列/索引顺序，不把重复索引SHA必同作为唯一通过条件。同一实际indices注入原kernel与适配器，独立核算术。
4. global step、expert冷启动、request重置、局部首层策略、refresh与mask reuse全过程；零漏挂、零错误fallback。
5. native-noop→native-Dense→有状态Dense的完整同seed/初噪闭环；逐层trace可随这一批同时采集。复用匹配产物，不再为相同输入另跑独立smoke。

实现错误先修复，不能进入性能/质量比较后再当作方法局限。对应检查通过只证明所覆盖语义与数值，不自动代表自由生成质量通过。

### 6.2 校准和有限参数选择

继续使用现行VBench v4的CAL/DEV身份，保留旧CAL11、扩展CAL11形成CAL22；DEV11仅开发，历史七源不升格为盲测。**本轮正式参考完成CAL256全有效token统计**；11/22/64仅作为嵌套预算消融，不能凭小前缀的早停跳过256。四个ICC边界在同一次Dense D4前向流式累计，不做VAE/MP4，不持久化所有全token大tensor。256对应作者论文的参考样本规模，但图像变视频后并非相同工作量；将来若部署采用较小预算，应以参数/风险稳定性和DEV结果说明，并明确作为预算适配。Q1享有相同CAL256访问与已冻结的拟合规则；保留其512行拟合时应披露，不伪称与ICC全token统计相同。

作者原统计使用全部有效token时，不能以固定512/64行面板默认为等价；若为了资源抽样，full-token统计是参考、抽样是独立预算消融。用于ICC统计的四边界不能以旧只有U或P64的资产补猜。完整来源与图像簇去重、prompt/image/预处理/seed/初噪hash必须在实际运行绑定；当前[输入提案](strict_reproduction_20261116/COHORT_PROPOSAL.json)只冻结元数据选择，未伪称实体输入已验。

ICC有限候选为SVD/CA/CD×rank{512,768,1024}；另增加一个CD四状态rank1408的近整周期MAC匹配点。对每一家族在DEV按冻结的质量约束与实测成本选择，保存所有被淘汰点。Chip保留作者默认点；top_keys的有限变动及Global-P64只在DEV确定，不能用评估集挑最优。

为什么需要较强ICC预算点：令F=2dm，ICC hit成本L=2r(d+m)，Q1额外投影a=d²。一个anchor+hit的比值为

\[
\rho_{\rm ICC4S}=\frac{F+L}{2F},\quad
\rho_{\rm ICC2R}=\frac{F+2L}{2F},\quad
\rho_{\rm Q1}=\frac{F+2a}{2F}.
\]

| 实际缓存与秩 | 完整周期主MAC / Dense |
| --- | ---: |
| ICC四状态r512 | 0.568519 |
| ICC双残差r512（历史路径） | 0.637037 |
| ICC四状态r1024 | 0.637037 |
| Q1r2 | 0.685185 |
| ICC四状态r1408 | 0.688426 |

四状态约r1383.78才在这个算术口径下匹配Q1；r1408是对齐后的近邻。它不是实测时间相同，也不保证更高rank质量一定更好。只测r512再声称相同预算下Q1优于ICC，依据不足。

### 6.3 适配消融与闭环评估分开

必做单项：ICC四状态→双残差同因子；作者精度/算术→Wan BF16路径；CAL规模/全token与抽样；尾Dense→valid-tail；原布局→局部FFN置换；原模块常驻→无损offload。布局、尾处理等在可证明逐元素等价时，先做数值和真实成本验证；不是每个等价搬运都需要重生成整套评估视频。有数值/支持变化的方案再做有限DEV自由生成对照。Chip全局scheduler与expert局部时钟在D4的A/H形态相同，不能靠再生成一套D4识别；应在覆盖step≥10刷新/复用的连续状态轨迹中比较。

Chip分位+随机→exact-K→Global-P64/Block-P64以及first2→all40是变体消融，用来解释差别，不能替代作者reference。新算法候选暂不扩展。 抽样统计的必要诊断使用同次CAL的full-token与PANEL512，在六个既有sentinel、同rank/同配方下比较factor action和DEV局部风险，不新增完整视频；只支持该局部范围。完整Wan FP32-FFN自由轨迹不在本轮预算内，当前精度消融只覆盖原模块/同输入边界，不能据此排除视频级精度贡献。

评估集在参数锁定前按输入元数据规则选取22个source，11类各2个来源，每源seed43/44；与CAL/DEV按图像簇分离。第二批来源按同一输入哈希顺序预先选定，不是看第一批结果后加样本。当前没有证明其全部历史未暴露，所以只称本周期拟合/选择留出集；若文稿要声称真正未见来源的确认，再按历史曝光清单核定并更换，不能用命名掩盖曝光。

闭环评估最多9臂×22source×2seed=396条D4：Dense、Q1r2、ICC四状态SVD/CA/CD各一个冻结点、ICC-CD四状态r1408、ICC-CD双残差同因子、Chip作者最小端口、Global-P64。若某点重复，复用同一产物；不能把缺失的正式参考换成额外变体来凑齐。完整臂选择、DEV消融与条件预算见[实验设计](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md)。396只计这一评估批，CAL、DEV、neutral和计时复跑单独记账，绝不冒充总预算。 统一全阶段提案上限为889次D4基础执行，加最多8次技术重试，共897；另外O40最多6次、作者原模型小配对最多4次单列。该上限包含CAL256、DEV121、CAL22闭环22、适配24、额外neutral至多10、EVAL396与计时60；兼容资产逐条扣除，不把取消额度转给新候选。资源、真实输入与producer尚未绑定，所以不是已派发任务。

每条候选自由运行全部步/层，不能注入Dense教师输入。配对同checkpoint/预处理/prompt/seed/初噪/采样器/attention/数值族。指标包括既有8个质量分项、连续光流方向与幅值、主体/动作/闪烁盲看；二值dynamic_degree不替代运动质量。source是独立单位，先聚合两个seed，再做配对差、来源分布与区间；44视频或160层不能写成独立样本数。n=22也不预先保证非劣或总体优胜的检验功效，区间过宽就报告未定，不观察结果后无限扩容。

### 6.4 原始多步模型与蒸馏问题放在正确位置

优先完成D4目标模型的严格端口。作者模块的多hit/refresh轨迹是必测；现有BW/BWS用于提出预期并复用状态，不再重新拟合其已完成表。

O40的新增原生轨迹只用于补当前资产没有的连续调用/状态边界，数量须单列冻结。原始权重O40、蒸馏权重D4、CFG与步间隔共同变化，不能把其差异单称蒸馏因果。只有完成参考后，才据实际缺失因素设计分离对照。不能用蒸馏假说解释一个尚未完整接通的复现。

## 7. 速度与资源必须按实际可执行路径比较

所有方法在同一已核实硬件/软件上测完整anchor+hit/整段生成，含proxy、selector、packing、权重搬运、cache读写/预取、首次参数准备和工作区。冷启动、预热后FFN、denoise与完整生成分别报；关闭重型observer后的技术重复另记运行数。

ICC四状态的host offload若是唯一可执行参考，就如实计传输开销；不能把单层算术时间当完整方法速度。随后允许双残差或更好的合法offload作为独立优化臂。Chip同理必须直接给官方Triton合理实现机会，不使用dense indexing原型的慢结果惩罚作者方法。

在GPU/host容量、真实输入实体、编译环境和producer补齐前，本文件不是可直接派发的ACTIVE scope。既有独立评议规则继续适用：提交具体版本、真实不同Agent审查修订、冻结有限范围后执行；这一轮的设计审阅不预填RUN/EXIT，不增加审批层级。

## 8. 哪些成果复用，哪些必须重新做

| 资产 | 本轮处置 |
| --- | --- |
| A/BW/BWS已完成分数、参数与收据 | 原身份保留；用于开发诊断，不重跑相同问题 |
| 旧Dense/初噪/图像/参数/视频 | 所有关键身份与本轮完全匹配才复用；不因日期旧而重做，也不只凭同source/seed就拼表 |
| 旧ICC四边界统计/因子producer | 若找回且参数/字节/实际执行全部闭合可复用；缺边界或未知max_ratio则按新配方重建 |
| 历史author-style Chip | 源码、selector二进制、调度、neutral、输入身份闭合才可认作对应运行；只有汇总时重新建立reference |
| Global-P64 | 继续保留已验证变体身份；新共同评估条件改变时再生成匹配结果 |
| 新参考完整视频 | 完成模块/端口验收和参数冻结后重做；不以补文档代替缺失的运行 |

执行后的解释顺序固定为：先确认实现正确完整，再确认必要适配的影响，再比较合理预算下的质量/时延/内存，最后才讨论低秩约束、空间分组、少步动力学、视频与LLM的差别及蒸馏关系。若作者reference修正后追平或超过Q1，应据实改写结论。
