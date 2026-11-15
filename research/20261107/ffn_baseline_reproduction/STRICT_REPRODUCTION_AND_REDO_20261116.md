# ICC / Chipmunk：先完整复现，再重做 Wan 比较

2026-11-16（北京时间）。依据 FFNDS `9defbbaa9112eee954c24ae92c91831fd2dd69c4`、本轮用户最新要求及再次核对的作者默认分支。当前为**源码审计、接口检查与重做规格**；本轮模型/GPU运行数为0。详细上游审计、设计复审与输入提案在 [本轮材料](strict_reproduction_20261116/) 中。

## 0. 本次细化：建立代码引述，并缩减生成

当前执行设计以 [精简实验总账](strict_reproduction_20261116/CONTROLLED_REPRODUCTION_PLAN.md) 为唯一入口。**撤回上一版897次D4提案；本轮五臂×DEV11×一个seed，主视频最多55条。校准、neutral、适配消融和重试合计最多99次D4尝试，匹配资产逐条抵扣。** 这是开发对照；不再另开EVAL22、第二个seed、新O40、原论文整系统生成或60次计时复跑。

完整性由作者模块与实际端口的验收建立：ICC四状态、SVD/CA/CD及作者Chipmunk的proxy、selector、kernel、所有状态分支均保留。三种ICC分解按固定局部准则选择一个r1408四状态点进入视频表；未进入的家族只有模块/局部证据，不声称其终端视频质量已测。

**纠正256的含义：** 作者`num_samples=256`数的是图像数据项。CA每批一次随机时间点模型调用，CD每批两次相邻时间点调用；默认batch16、epoch1分别为16/32次批模型调用，不是256条50步轨迹。原真实图像加噪分布与Wan原生D4轨迹仍有迁移差别，CAL11→22只能检查后者的预算敏感性。主CAL11固定，取消强制CAL256，不能用大视频token数替代来源广度论证。

代码证据按“主张→固定commit/函数/行号→短引述→最小验收”整理：

| 对象 | 可核验入口 |
| --- | --- |
| ICC四状态、三家族、统计生产器、256语义、历史2R路径 | [ICC_CODE_CITATIONS.md](strict_reproduction_20261116/ICC_CODE_CITATIONS.md) |
| 官方Wan入口、作者MLP全机制、必要接口、Global-P64实际调用 | [CHIP_CODE_CITATIONS.md](strict_reproduction_20261116/CHIP_CODE_CITATIONS.md) |
| 机器可读来源及短摘录 | [CODE_CITATIONS.json](strict_reproduction_20261116/CODE_CITATIONS.json) |
| 原始字节、行号、摘录与链接核验 | [verify_code_citations.py](strict_reproduction_20261116/verify_code_citations.py) · [CITATION_VERIFICATION.json](strict_reproduction_20261116/CITATION_VERIFICATION.json) |

引述核验只证明代码来源与摘录准确，不把静态代码存在写成实际数值验收通过。本轮新增模型运行仍为0。

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

## 6. 精简后的实施顺序

### 6.1 完整机制由模块和同输入验收覆盖

ICC直接使用原SpMlp、统计与分解函数建立oracle；核四状态、bias、GELU、两级低秩、anchor固定、CA/CD四边界和左右缩放。三家族都实现，r512和r1408从同一次分解取前缀。Chip直接使用作者SparseDiffMlp、compiled selector与官方Triton；原shape/dtype正对照、固定真实indices算术对照、连续多hit和step9→10→11→12必须覆盖。已有计数器CPU trace复用；错误状态不需要另生成一批视频来重证。

必要适配分别检查：真实pre-W1/pre-gate接线、当前gate一次、request/expert/local-layer隔离、voxel往返、尾52、转置权重/CPU缓存生命周期。可证明等价的搬运用同输入数值对照；支持集或舍入改变的变体先局部评价。主Chip保留作者舍入与dense-tail，不为了省核验替换为本仓packed路径。

### 6.2 同一批捕获供校准与局部诊断复用

主参数统一CAL11。CAL_EXT11只用于CAL11→22统计/局部预算消融。仅在旧资产缺少充分统计时补采，最多22条native D4、不做输出VAE解码/视频编码；每次同时收集CA/CD四边界全有效token统计、来源权重及Q1所需P512、row IDs、当前gate。P512不能补造缺失的全token统计。

CA主定义在每source/expert的两个合法端点中按固定输入哈希取一个；同次捕获可计算双端点等权版本作敏感性检查。CD取high0→1、low2→3，绝不跨专家相减。两者都保存统计归约、min-max、sqrt、max_ratio=1000及分解数值合同。它们是明确的Wan目标分布适配，不声称与原图像q_sample校准分布等价。

### 6.3 固定一个合理预算的ICC进入完整视频

令F=2dm、L=2r(d+m)、a=d²，一个anchor+hit周期的主MAC比为


\[
\rho_{\rm ICC4S}=\frac{F+L}{2F},\qquad
\rho_{\rm ICC2R}=\frac{F+2L}{2F},\qquad
\rho_{\rm Q1}=\frac{F+2a}{2F}.
\]

当前d5120/m13824下，四状态r512为0.568519，旧双残差r512为0.637037，Q1为0.685185；**四状态r1408为0.688426**，其周期主MAC约比Q1高0.473%。所以新主ICC固定r1408，r512只做历史同输入诊断，不再扫多rank完整视频。这是主算术预算接近，不能当实测时延相同。

SVD/CA/CD在既有DEV11、high/low×layer{4,19,34}、相同P512及当前gate上比较F64门后绝对MSE：层→专家→来源等权；严格同分按SVD→CA→CD固定顺序。选中的家族再构造全80站点因子并冻结，随后才开始主视频。此规则只有六站点开发证据，不保证选出全质量最优ICC。其他家族的分数完整保存。

### 6.4 主视频55条，全部D4尝试上限99

| 项目 | 新D4尝试上限 | 唯一目的 |
| --- | ---: | --- |
| CAL11/22缺项补采 | 22 | 得到四边界完整统计；不做输出VAE解码/视频编码、同捕获多用 |
| Dense、Q1、选定ICC4S-r1408、作者Chip、Global-P64 | 55 | 5臂×DEV11×seed42的自由生成开发对照 |
| 两源neutral额外路径 | 10 | plain、hook-bypass及各adapter的Dense语义；主表两条hook-Dense已计入55 |
| ICC同因子4S→2R | 3 | DEV3完整闭环；四状态父输出复用主表 |
| 条件CAL22闭环 | 3 | 仅当局部预算变化足以改变判断；固定原家族/rank/4S |
| 条件适配闭环 | 3 | 仅一个已定位的非等价适配；未评价分支保持未定 |
| 技术重试 | 3 | 已定位可修复失败；不能用于换seed/候选 |
| **合计** | **99** | **基本90，两个条件分支6，重试3；不是必须跑满** |

方法/参数/输入完整匹配的旧视频、统计和neutral逐条抵扣；主Dense11及两个必需hook路径全部可复用才减11，Q1/Global-P64的neutral各完全符合才能各减2。缺少绑定前不报预计实跑数量。所有失败尝试计入总账，取消额度不改派新候选。

没有EVAL22或第二个seed；DEV既用于局部选择又用于完整视频，因此本轮结论限定为开发结果。独立单位是source cluster，不是token、帧或层。保留逐源配对质量、连续运动和来源删除敏感性。原始多步模型与蒸馏因果问题继续使用BW/BWS的现有证据，本轮不新增O40；未充分的状态/数据问题如实留待下一轮。

## 7. 速度与资源必须按实际可执行路径比较

所有方法在同一已核实硬件/软件上测完整anchor+hit/整段生成，含proxy、selector、packing、权重搬运、cache读写/预取、首次参数准备和工作区。冷启动、预热后FFN、denoise与完整生成分别报；本轮不额外安排完整模型计时复跑；正式产生且关闭重型observer的运行记录实际时间，缺匹配时间的旧视频只复用质量。计时样本不足时不宣布微小速度差已分辨。

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
