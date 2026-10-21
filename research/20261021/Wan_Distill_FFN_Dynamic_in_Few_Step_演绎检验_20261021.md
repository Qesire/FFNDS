# Wan Distill FFN Dynamic Sparse：Dynamic-in-Few-Step 研究包的演绎检验

日期：2026-10-21，北京时间。方法记录：DIFS-WAN-DF-R1。使用 mechanism-research 的计算流重建、先演绎、反例检验与研究记录流程。

**判断：有条件地值得借用其“在实际剩余轨迹上评估干预”的思想；不支持照搬 U 形预算、从头扫描所有掩码、直接叠加 A5+HWG，或重新尝试普通小秩补偿。当前最有价值的用途是解释并改进 UB1 的跨来源画质退化。它尚未解决廉价输入动态支持选择，也尚未证明能通过你的联合验收。**

这次读取了指定 ZIP、旧交接包、后来发现的 `repo_updates_20261021/files` 更新、对应运行报告与源码，并联网核对论文。历史结论以较新的实际记录为准。重新计算了六份质量比较的相对下降、UB1 三次完整 FFN 总计时比、调度器系数与 AR2 C 风险比；见同目录 `evidence_arithmetic_audit.json`。这些是本轮文件/算术核对，不是本轮重跑 GPU、原视频评分或大张量回放。未启动新训练、模型生成或服务器作业。

**一、首先纠正研究起点：已有成果比 ZIP 假设的更进一步**

| 已有结果 | 证据所支持的事实 | 对新包的直接约束 |
|---|---|---|
| UB1，密度 25%/50%/50%/75% | 真实缩减两次主 GEMM；全局主 MAC=50%；完整 FFN 1.546111×；原20开发集 aesthetic/imaging/motion 下降 0.873850%/1.344750%/0.007365% | 按步结构预算、真实 packing、低秩修正都已实现；不能作为这次新贡献重新发现 |
| UB1，来源分离补充20 | 三项下降 1.933715%/3.165968%/−0.094191%，imaging 失败 | 当前缺口是跨来源画质；原20通过不等于项目已完成 |
| TF2，同一 D-Q 状态 | 四步 flow 相对 RMSE 为29.4041%/15.5424%/16.7728%/11.9241%；单步更新后为2.6326%/2.5564%/6.4403%/11.6769% | flow、更新后 latent、最终质量不是同一个重要性指标 |
| UB2，25%/25%/50%/100% | 同状态末步 flow/post 误差归零，但原20/补充20 imaging 分别下降2.217994%/4.651736%，均失败 | 已经否定“看到末步误差最大，就把 step1 预算继续移至末步”的这一个策略 |
| LR2，末步最后10层 R512 | 原生局部 CV 通过；补充 imaging 仍下降3.050580% | 提升局部补偿容量不充分，不能把 rank4/8/16 当未探索的主要突破口 |
| 新 AR2，非旧超时作业 | 两来源、8 scopes，C-dynamic/C-static 风险比0.6141–0.8097，C-dynamic/random为0.5610–0.6817，局部门槛通过 | 输入动态支持确有局部机会；“平均通道排名稳定，所以动态无用”不成立 |
| PR1、W1低秩/PCA、RB2、HP1/2、NP1 | 尚无廉价在线选择器通过；RB2四bank oracle只多保留约0.56–0.79个百分点C；NP1略逊匹配线性头 | rollout 离线搜索不能自动补上 pre-FFN0 在线预测缺口；C-score也不是重构损失 |
| OD1，固定 D 输入的 O/D 局部对照 | 两图、block20 的8 scopes；O/D 的 top108 C保留比例最大差约0.0016423个百分点 | 不能继续把“蒸馏已普遍消灭局部稀疏冗余”当已证实前提；此结果也不代表O整网 |

原20与补充20目前都已经参与开发。新方法若根据这些结果调整，就不能再把它们称作未见测试。六份质量表的来源单位均是每批20个来源，五个评分文件名副本不是五个独立视频。

旧的 A5、R11/R12/R15、R21B/R26、AR1 仍有用：A5正式美学下降2.401886%且完整计算W1；影子执行速度不能拼接到其他候选视频；R15半宽+rank512闭环失败；任意重排/重新QDQ会改变保留值；AR1跨种子gain显著减少局部幅值收缩。这些教训已部分被UB1吸收，不应重新绕过。

**二、论文与低成本扩展必须区分**

原论文在 Wan2.1 T2V 上联合更新蒸馏学生与按步结构门控；Output Rollout 为 fake-score 网络提供最终生成样本。你的 ZIP 则提出冻结 Wan2.2 I2V 权重，以前向 rollout 搜索掩码的扩展，两者优化问题不同。论文也报告了冻结蒸馏权重后只优化动态mask的负结果；这对迁移构成警示，但不是所有离线搜索都不可能成功的证明。14B 的剪枝额外加速为1.20×，测量边界是 DiT、不含VAE；它不是你的“50%主FFN、完整FFN≥1.5×”验收结果。来源：[原文 §4–5](https://arxiv.org/html/2607.06631v1)。

本报告不把论文中的整网结构图当成你 FFN-only 的天然冗余谱，也不把按去噪步变化等同逐token输入动态性。这里只把该论文作为方法启发与最近对照，不宣称原创性审计已经完成。

**三、实际对象与计算路径**

目标主对象是 D-Q：Wan2.2-I2V-A14B-LightX2V-Distill-4step 加已有 HiF4 smooth/有效权重与原生 QDQ；D-BF16仍是蒸馏D，原版O是另一个权重对象。40 blocks、输入/输出维5120、hidden维13824。连续64通道组成216组。step0/1为high expert，step2/3为low expert。这里不是token级MoE路由。

每位置形状为 U,V1∈R^(M×5120)，H,V2∈R^(M×13824)，Y,Delta∈R^(M×5120)；M取实际视频token数，保存的511/512行只是采样，不得当完整原生M。gate按实际广播形状使用。

已核对相关源码的语义路径为：SA/CA残差 → norm2与时间调制 → smooth1/cast/QDQ1 → FFN0 → GELU(tanh) → smooth2/cast/QDQ2 → FFN2 → gate/cast/residual → 后续blocks/head → flow → scheduler → 后续去噪 → VAE与质量指标。

用行向量约定省略数值cast时，D-Q可写为：

\[
V_1=Q_1(U/s_1),\quad H=\operatorname{GELU}_{tanh}(V_1\widetilde W_0^T+b_0),\quad V_2=Q_2(H/s_2),\quad Y=V_2\widetilde W_2^T+b_2.
\]

实际路径在除法、QDQ、线性、增益和残差之间都有规定的精度与cast；此式只表达算子依赖，不宣称合并运算逐位等价。有效权重形状分别为[13824,5120]与[5120,13824]，分析导出的转置布局不能混用。

UB1还包括位置专属支持K、输入端gain、折叠bias、从post-Q1 V1计算的rank256双线性残差。其主路径分别使用 W0[K,:]、b0[K]、s2[K]、W2[:,K]；残差在FFN2输出后、原gate之前相加。不能把UB1简化成裸mask后再声称新方法胜过它。

源码定位：`step_budget_runtime.py:create_runtime/apply`，`trained_packed_runtime.py:apply`，`reduced_rank_residual.py:predict_y`；LightX2V保存副本 `transformer_infer.py:366–429`；`Wan22StepDistillScheduler.step_post`。完整路径在文末。当前服务器是否仍逐字对应这些保存版本，本轮未重新连接核验，记为OPEN_BINDING；不影响下述带明确前提的代数结论。

**四、可以直接演绎解决的六个问题**

**D1：按步重要性必须包含真实scheduler，但scheduler权重本身不够。**

配置索引[1000,750,500,250]经过shift5，实数公式为：

\[
\sigma(q)=\frac{5q}{1+4q},\qquad \sigma=[1,15/16,5/6,5/8],\quad \sigma_4=0.
\]

忽略舍入的更新为 x_(i+1)=x_i+β_i F_i(x_i)，其中

\[
\beta_i=\sigma_{i+1}-\sigma_i=[-1/16,-5/48,-5/24,-5/8].
\]

同一flow误差在固定输入上的直接更新幅度系数，末步为首步10倍；平方误差系数为100倍。实际模型输入时间为约[1000,937.5,833.3333,625]。原生代码分别计算FP32乘加，再cast回BF16，不能用合并β式替代后宣称逐位相等。

若连续近似下 A_i=I+β_i J_(F_i)，K_(i,l)表示某FFN写回扰动到该步flow的导数，则

\[
\delta x_4\approx\sum_{i,l}(A_3\cdots A_{i+1})\,\beta_i K_{i,l}\,\delta\Delta_{i,l}.
\]

β只是其中一个因子。未知的是下游方向、放大、抵消与状态依赖。末步full FFN只能令该步在**同输入**下的算子差为0，无法令此前造成的输入差消失：其输出差仍为 T_3^D(x_3^S)−T_3^D(x_3^D)。UB2正好给出了该误推的实测反例。

QDQ与离散支持在阈值处不光滑，这个Jacobian式只作连续局部机制解释，不能当50%裁剪误差的严格上界。D4提供不依赖可微性的替代。

**D2：局部最小误差、单组评分和最终质量，三者没有一般保序保证。**

固定post-Q2激活与有效权重，令每组门控贡献 c_b=Σ_(j∈b) a_j(W2[:,j]⊙gate)。删除集合S的误差为 −Σ_(b∈S)c_b；故

\[
\|e\|^2=\sum_{b\in S}\|c_b\|^2+2\sum_{b<c,\,b,c\in S}\langle c_b,c_c\rangle.
\]

C_b=||c_b||²仍忽略组间交叉项。加上gain、折叠bias及RR后，误差还包含这些项与删组误差的内积。支持变了，原策略的最优gain/RR也不再保证最优。因此必须评估实际部署算子的增量，不能将AR2 C叫作最优支持oracle。

局部到终态保序的构造反例：后续线性映射L=diag(100,0.01)。扰动e1=(0.01,0)比e2=(0,1)局部小100倍，但终态范数分别1与0.01，排序颠倒。相同原理对不同step/layer均成立。

此外，潜变量L2不等于VBench质量。现有C24→UB1已经显示：原20末态latent相对RMSE从38.72%增至42.80%，视频指标却改善。故“local MSE→terminal latent MSE”只是换了代理，不足以证明最终画质改善。rollout阶段必须最终关联实际decoded video与原质量分项；时序指标作为补充，不能替换imaging硬门槛。

**D3：小幅单组探针不能外推到50%组合。**

对离散删除变量r作形式上的二阶展开，损失可能为 L(0)+b^T r+½r^T H r。单组探针不提供所有H_bc；稀疏一半不是无穷小扰动。即使在纯线性可加模型里，两个终态误差d0=(1,0)、d1=(−0.9,0)合计范数0.1，单独消除d0后剩余范数反而0.9。该例严格否定“消除最大的单项必然降低总误差”，不证明你的模型一定依赖这种抵消。

现有模型总计4×40×216=34,560个step-layer-group位置。ZIP从dense逐点扫描再组合，既昂贵又缺少当前50%部署上下文。更合理的起点是UB1；以后若搜索，先比较总主MAC和辅助成本匹配的有限预算交换，重新测组合后的真实轨迹，而不是由单组分数相加作验收。

**D4：四步结构允许一个不训练、不求大Jacobian的精确轨迹分解。**

将每步完整网络+scheduler记为T_i^D与T_i^S，S固定为UB1。所有必要状态（包括scheduler、RNG、cache语义）包含在x中，或在各臂保持匹配；令x_i^D为dense前缀状态。定义

\[
H_i=\Psi\big(T_3^S\circ\cdots\circ T_i^S(x_i^D)\big),\qquad i=0,1,2,3;\quad H_4=\Psi(x_4^D).
\]

Ψ可取终态latent，或确定性的完整VAE解码。于是H0是全UB1，H4是全dense，H1/2/3依次使用1/2/3个dense前缀步骤与余下UB1后缀。令d_i=H_i−H_(i+1)，直接望远镜求和得到

\[
H_0-H_4=\sum_{i=0}^{3}d_i.
\]

证明只有相邻项相消，不需要小误差、可微性、FFN线性或QDQ可交换假设；每个T可以保留真实BF16/QDQ操作。差向量在FP64累计仅有差分统计的舍入误差，不改变被测模型算子。

进一步，终态SSE严格包含 Σ||d_i||² 与2Σ〈d_i,d_j〉。对任意固定标量评分q同样有 q(H0)−q(H4)=Σ[q(H_i)−q(H_(i+1))]，可直接考察imaging变化，而不强迫潜变量L2替代它。

这是**按指定替换顺序的干预分解**，不是顺序无关的因果贡献、Shapley值，也不是最优预算分配。hybrid本身可能使用更多计算，只作诊断，不能作为通过50%验收的新候选。

为什么比TF2多一步：TF2每次都恢复dense状态，测的是单步输出；这里让扰动走完真实UB1后缀，保留后续放大和相互作用。若已有可正确恢复的dense状态与两端参考，每来源只增加H1/H2/H3，即3+2+1=6次denoiser evaluation、3次解码。四个已有TF2来源需24次新增evaluation；这是不含初始化、重放控制、解码与评分的计算计数，不是墙钟承诺。若恢复状态不足，必须从完整前缀重算，不能套用这个成本数。

**D5：新增动态模块的延迟余量只有约3%，而后W1选择不满足完整FFN目标。**

UB1历史同边界均值：T_D=10177.568613ms，T_U=6582.688544ms。若不减少现有计算，要保持1.5×，新增耗时必须满足

\[
T_{extra}\le T_D/1.5-T_U=202.357198\text{ms}=3.074081\%\,T_U.
\]

它只是该硬件、输入、三次重复点估计下的预算余量，不是保守置信保证。新增路由、packing、padding、残差或同步都要支付；如果新策略另省了计算，需重新测量净收益。不能把两种加速比相乘。

若两主GEMM原本等成本，完整算完W1，只把W2减半，理想上限为1/(1/2+1/4)=4/3，还未计QDQ与路由。因此依赖完整H/C在线选择的原型不能直接回答完整FFN≥1.5×。按步静态支持能预先缩减W1；逐token动态支持仍需一个足够便宜的pre-W1选择器，已有负结果正落在这个缺口上。

**D6：rank4/8/16不是尚未尝试的补偿原理。**

在固定输入特征X、目标残差E、样本权重、支持与未正则化平方损失下，先对X、E作相同的加权中心化并左乘权重平方根。以下X、E表示变换后的矩阵，P_X为X列空间的正交投影。最优rank≤r线性预测误差为

\[
\min_{\operatorname{rank}(B)\le r}\|E-XB\|_F^2=\|(I-P_X)E\|_F^2+\sum_{j>r}\sigma_j(P_XE)^2.
\]

因为两项正交，第一项是输入特征不可表达的误差，第二项是可表达部分的秩截断误差。增加rank不能消除第一项。小秩函数类包含于大秩函数类，因此同条件的最佳训练重构不能优于更大秩；这不比较不同正则化、拟合算法、量化或泛化误差。

你已经有R256、条件R512、联合重建、学生状态教师、量化适配以及NF1等结果。现有实现还使用ridge与近似低秩分解、BF16部署，不能把上式当实测最优值。但这些结果足以降低普通小秩方案的优先级。除非改变特征信息、拟合状态或明确优化目标，并给出可区别于已有试验的预测，否则ZIP的E5主要是已知方向的重复。

**五、对ZIP各部分的取舍**

| 包内建议 | 本项目判断 | 必要修改 |
|---|---|---|
| step-specific规则支持 | 可复用的已实现基线 | 以UB1为incumbent；与其完整补偿及实测成本公平比较 |
| rollout复评 | 最值得保留，条件接受 | 先做D4的少量hybrid诊断，再决定是否搜索；最终看decoded质量 |
| U形或越晚越多计算 | 不接受作为先验答案 | 不移植论文形状；UB2是本项目的现成反例 |
| dense单组扫描→greedy | 不宜按原样开始 | 当前UB1附近、限候选、检查组合效应；不能从dense小扰动外推50% |
| A5+HWG Hybrid | 暂不优先 | 明确是否输入动态，是否在W1前选，是否还有≥1.5×净速度 |
| 小秩adapter | 低优先级 | 必须先解释它与RR1/C24/LR2/OP1/NF1的结构差别 |
| 原版/蒸馏大对照 | 可做后续科学问题，非本轮前置 | OD1只覆盖局部，不支持全模型论断；先解决D-Q自己的失败 |
| temporal/background约束 | 有辅助价值 | 保留实际视频帧轴、运动对齐；不能用motion平滑掩盖模糊与运动停滞 |

包内 `two_linear_ffn` 使用默认GELU，而目标是GELU(tanh)；toy重排只处理线性权重，不含smooth/量化分组/残差。其“全1精确回归”是待验证要求，不是已有Wan接入证据。64整组packing已有局部及UB1部署控制可复用；不能由这些控制推出任意permutation或任意新support都逐位等价。

**六、下一步只保留一个小诊断，尚未执行**

建议先固定UB1全部参数与现有四个TF2开发来源0/5/10/15，执行H1/H2/H3，不拟合新mask、不改预算、不加路由。复用dense与UB1端点前必须确认同输入、原生dtype、精确调度阶段、conditioning、RNG及runtime状态；恢复模式先复现纯dense与纯UB1控制。模型hook恢复后检查下一请求无残留。全部对照不允许用强制恢复的dense最终视频冒充主动候选质量。

只区分三种机制：

1. **局部排序失配**：若早/中步在后缀传播后的质量影响超过末步，TF2本地误差排序不足以指导预算，继续针对这些步骤做等成本交换；若同向且接近，不能把“轨迹信息”当新的主要收益来源。
2. **组合相互作用**：若d_i内积相对于总误差有显著量级，或者一项恢复降低局部误差却使decoded质量变差，后续必须整组/成对评估，不再用单点得分相加；如果这些项很小，优先简单的独立近似。
3. **质量代理失配**：若latent变化排序与实际imaging/aesthetic变化不一致，搜索评价必须更接近decoded输出；不能仅加大latent-MSE权重。若一致，保留低成本latent指标作筛选，但最终质量验收仍保留。

这里四来源只作描述性机制诊断，不设虚假的总体显著性或成功概率。数值影响应超过重复运行/评分的实测波动；具体噪声容限必须在下一次模型实验前由原生控制固定。本报告未把这些待测量写成观察结果。

若有明确可操作的排序差异，才登记一个**主MAC与辅助成本匹配的有限预算/支持交换**。保持完整FFN≥1.5×、三项原质量逐项下降≤2%的目标；同FLOPs不等于同延迟。若只是降低稀疏目标才达质量，则应明确标为另一条质效前沿，不称原50%目标已完成。

后续评估在开发集改进后再用未使用来源确认；不能在预先冻结的确认集上反复选择候选。对“静态按步策略”和“输入动态策略”分别写结论。已有AR2支持继续研究输入动态性，但这份包没有独立给出通过的廉价在线实现。

**七、可复用资产与复用边界**

本轮只生成文档、状态与算术审计。以下是下一诊断的资产计划，均未在本轮产生：

| 资产 | producer、格式/坐标 | 允许复用与失效条件 | 当前状态 |
|---|---|---|---|
| D-Q/UB1端点及完整恢复状态 | 原历史生成器；BF16状态、step/expert、源图/prompt/seed、conditioning及RNG记录 | 同runtime、输入、schedule、精度下作为端点；换checkpoint/cache/支持/seed或恢复缺失则失效 | 历史索引可见，大张量本轮未重验 |
| H1/H2/H3 hybrid终态 | 计划的固定UB1后缀；每来源3份latent与decoded video，保存switch_i | 可复算差向量与固定质量分项；不可当50%部署候选 | NOT_PRODUCED |
| d_i与内积矩阵 | FP64离线差分；每来源4向量/4×4内积、同一latent或像素坐标 | 只解释这一替换顺序与冻结算子；换metric或观测坐标需重新计算 | NOT_PRODUCED |
| 原生恢复控制日志 | 完整输入/输出、dtype、状态边界；JSON | 确认本次resume路径；不能覆盖其他新hook或cache路径 | NOT_PRODUCED |
| 按来源质量表 | 相同官方worker/版本；源ID×hybrid×三个指标，保留原单位 | 源为统计单位；不能以文件别名/帧/token扩充样本量 | NOT_PRODUCED |

每项未来资产应附producer、方法版本、依赖locator、shape/dtype/坐标、seed、运行参数、确定性条件、存储位置与验证命令。本轮不计算或用摘要哈希代替语义核验；历史记录里的哈希字段未作为本轮身份或证明依据。现有采样张量仅支持相应边界的离线局部分析；不能拿旧AR2采样行、C24学生输入或OP1局部教师冒充UB1完整状态恢复点。

**八、证据定位与最终状态**

方法包：`<LOCAL_WINDOWS_HOME>\Downloads\Dynamic_in_Few_Step_Low_Training_Research_Pack_20261021.zip`，重点为 `02_research/`、`03_experiments/` 与参考 `src/difs/ops.py`。包内STATUS明确无真实Wan结果；demo未用于结论。

最新本地证据根目录E：`<LOCAL_WINDOWS_HOME>\Documents\Codex\2026-10-20\qign\work\repo_updates_20261021\files`。

- E下 `docs/12、24、28–41` 的对应编号文档：新AR2、UB1、TF2、UB2、LR1/2、OD1、NF1、NP1及相关阴性结果。
- E下 `reports/worker_20261020/formal20_UB1_comparison.json`、`disjoint20_UB1_comparison.json`、`UB1_complete_FFN_comparison.json`、`UB1_joint_engineering_acceptance_RUN.json`：UB1联合开发验收及补充失败。
- 同目录 `LR2_formal20_COMPARISON.json`、`LR2_disjoint20_COMPARISON.json`、`UB2_formal20_COMPARISON.json`、`UB2_disjoint20_COMPARISON.json`：两次新候选的完整质量结果。
- 同目录 `native_shift5_scheduler_audit.json`、`ar2_decisions.json`：本轮重算对应的调度器与局部风险字段。
- E下 `tools/step_budget_runtime.py`、`trained_packed_runtime.py`、`packed_reference_runtime.py`、`reduced_rank_residual.py`、`resident_ffn_runtime.py`：当前保存的部署数值路径和计时边界。
- LightX2V相关源码根：`<LOCAL_WINDOWS_HOME>\Documents\Codex\2026-10-20\qign\work\upstream\contracts\LightX2V\lightx2v\models`；对应 `schedulers/wan/step_distill/scheduler.py` 与 `networks/wan/infer/transformer_infer.py`。本轮按保存源码与报告相互核对，未以最新main替换历史运行。
- 早期证据：`<LOCAL_WINDOWS_HOME>\Documents\Codex\2026-10-20\qign\outputs\Wan22_FFN_HANDOFF_20261020/evidence`，尤其context_20261011与V5 AR1。旧交接“AR2未完成、尚无联合通过”的描述已被上述新矩阵/UB1结果部分更新，不能继续当最新状态。

数学/机制判断：**ACCEPT_CONDITIONAL**。精确轨迹分解与成本约束成立于写明的前提；“rollout搜索可解决UB1跨来源画质”仍为**OPEN**。论文复现：**NOT_RUN**。新Wan/GPU实验：**NOT_RUN**。研究创新：**DIAGNOSTIC / CANDIDATE，未声称原创或有效性已验证**。
