# P0 Chipmunk · Wan2.2/A100 作者机制受控移植：最小可审方案

作者：s2_theory_design_independent。状态 **SRC/ALG + 实验设计，REVIEW_REQUESTED**；无新运行、代码、批准或性能结果。此路径是作者选择性更新机制的 Wan2.2/BF16/SM80 受控移植，**不是作者 Hopper kernel 硬件忠实复现**。作者原 H100 范围保持独立待实现；当前授权节点无 Hopper 的盘点是硬件约束，不是对方法的否定。

## 操作与信息时点

对固定 expert、block、source 和 token 排列，记输入为 `X_t [M,D]`、参数 `A [D,H]`、`B [H,D]`、偏置 `b1,b2`；Dense 本层为 `h_t=GELU_tanh(X_t A+b1)`、`Y_t=h_t B+b2`，再按 Wan 原路径执行 gate、残差及其 BF16 写回。缓存的是同层同 expert 上次的 **pre-gate** `h_c,Y_c` 和 selection 需要的参考均值；严禁不同 expert、block、source 或 token 索引串用。每 expert 第一次访问必须 Dense 并初始化；四步中同 expert 后续访问才可使用其缓存；切换 expert 时未初始化的一方先 Dense。重置 source/seed/形状/模型或失败后不能沿用缓存。是否保持 high cache 直到退出及是否连续访问由真实调度证据决定，不能默认两个 expert 可以交叉共用。

对每连续128行 `g`，当前已可用 `X_t` 和先前缓存得到组均值代理，明确付费计算相应全部 H 个 W1 输出并比较作者式历史参考，再固定的 top/random 规则选择 `S_g`。作者 `src/chipmunk/modules/mlp.py` 的 proxy 以128 token均值、上次缓存激活/均值、top0.3+random0.05、counts_multiple_of=256 生成候选；严格公式及 BF16 rounding 在实际实现前对固定 commit `177fdefc4803e621fd8a02aad491812e9d1ecc94` 源码复核并冻结，不能用输出真误差作为选择 oracle。原作者 Wan 示例 MLP 未接入、论文 Wan D.1 为 attention 路径，见 [作者路径审计](../../../experiment_reviews/B1_CHIPMUNK_FLUX_MLP_ONLY_AUTHOR_MECHANISM_20261029/v1_author.md)。作者 FLUX MLP-only 有 disabled-attention counter、终止 reset 等缺陷；本移植明确以直接 `(expert, step, block, source)` 映射管理生命周期，不把修复称作者原样复现。

连续实数模型中，组内选择集共享而每 token 值独立：

`h'_i,j = GELU(X_i A_:,j+b1_j)` 若 j∈S_g，否则 `h_c,i,j`；
`Δh_i,j = h'_i,j-h_c,i,j`；`Y'_i = Y_c,i + Σ_{j∈S_g} Δh_i,j B_j,:`。

然后原位更新 h_c、Y_c；**b2 已含在初次 Y_c，只加入一次**，当前 gate 每步重新应用于 Y'，不缓存 gate 后结果。S=全 H 在精确实数下得到当前 Dense；实际 BF16 GEMM、GELU、差分和两次输出取整会使其与原 Dense 不逐byte等同。固定两种明确路径：BF16 激活差分+FP32累加并一次BF16写回（物理快候选），FP32差分/累加诊断参考；与原 Dense kernel同输入输出另比较，记录 max/relative L2、finite、全更新/偏置/gate/residual；不宣称形式恒等即 byte-neutral。设冻结 neutral 容差需审稿人确认并基于原生同M误差标度，不能运行后调阈值；超界直接 FAIL。FP32 差分经 BF16 tensor-core输入量化仍改变数值，若要严格 FP32 则 TF32/IEEE 算法、速度与误差单独列明，不混作 BF16等价。

## 代价与最强反例

M=31668、D=5120、H=13824，两个 dense GEMM 逻辑 MAC 为 `2MDH = 4,482,851,143,680`；若选中实际比例 p 且不计 padding，proxy dense W1 额外比例约 `1/256`，两个主 GEMM约 p，合计 `p+1/256`；尾组 52 行、256对齐、分组变长、随机并集、筛选、读写与 kernel tile 使实际比例改变。p=0.35 时粗略 0.354，而不是已测速度。每组必须记录实际有效和填充索引/重复消除/每kernel tile；缓存Y读写、h读写、代理W1、topk/random、同步、分配和 W2 布局均计入完整FFN边界。BF16 GPU tensor core 与 TF32 或 FP32 路径精度、算子和速度分别报告。

每 expert 的40层 BF16 `h+Y` 缓存为 `2×31668×(13824+5120)×40 = 47,993,487,360 B = 44.697 GiB`，不含 proxy、选择、workspace、原runtime双缓冲、模型、attention、VAE和allocator余量。双expert同时驻留约89.4 GiB，已超80 GiB级设备；单expert也必须实际测原runtime峰值与余量后才可判断是否共存。若真实调度证实每 expert 的两个步骤连续且不再回访，可在边界释放整个前expert缓存；这只降低同时常驻，不解决单expert约44.7 GiB问题。将层缓存放host或按层H2D/D2H是数值上可等价的存储策略，但全部流量/同步和新增host资源需计价，不保证快。缓存 BF16 Y 有累计舍入误差，FP32 Y更耗显存；只缓存 X 并重算 h 或 Y节省存储却增加 dense计算，不是免费等价。

固定35%示意：每个组若通过 Torch gather materialize 两个权重切片，约 `2×5120×4838×2 = 99,082,240 B`，248组累计约24.57 GB/层的**切片产物**，还需读取原权重及再次读切片；此为逻辑体量估算，不是测得 HBM 流量。Dense完整W1+W2权重约283.1 MB，cache复用和实际内存事务决定真实差距。不同组选择不同通道可能使稀疏gather无法复用权重；当前低数量的有效乘法仍可能比dense慢。不要以逐组Torch权重materialization作为默认加速实现。

竞争预测：H1 同组跨步改变集中，代理选择覆盖主要的逐token改变，误差低且两个稀疏kernel与代理/cache全边界的实测耗时小于原Dense；H2 块均值因相反 token 改变抵消而漏掉重要通道，或选中集跨组不重合、padding与访存使质量或性能失败。可构造两个 token 的输入差反向、组均值为零但各自激活发生变化的反例；代理分数小并不约束逐token误差。大权重相乘、gate放大、残差舍入和学生闭环扩散均使小局部h误差不必对应小视频差异。只测Dense teacher相邻状态无法推断自由student。

## 可审的最小两阶段 producer 设计（未登记）

**A 软件/算子原型**：新独立 producer + 两个 Triton BF16 kernel（A100 SM80，Torch负责管理和 selection），不改作者代码/冻结runner。kernel1 按每组索引直接gather W1列，`X@A_selected+b1`，tanh GELU，读旧h生成 `Δh` 并写选中h；kernel2 将 `Δh@B_selected` 以FP32累加到预 gate cached Y，再BF16写回。每组尾行mask、通道padding mask、每个output tile独占写回，避免跨 tile/race/重复加bias。delta存储和索引按组有界；显式检测重复索引、越界、alias、stale缓存。selection和每项buffer流量/算子均真实计数。编译与预热独立于计时；相同输入对原Dense、全更新、空更新、固定稀疏、反向抵消和expert错配进行控制。必要时先做小形状纯软件控制；合成误差不能替代真实模型输入证据。

**B 最小真实输入决策**：优先一层、一个同expert相邻step的两次输入，128个连续实际token，真实同block W1/W2/b1/b2及两步gate，限定一个开发来源/seed，并在同GPU同输入对照：原Dense、first dense+全更新neutral、真实代理与固定稀疏、等数随机负对照。禁止以未来teacher逐token真实误差选通道；可另存标为oracle的离线误差上限，不能计入可执行方法。强制参考两步均完整Dense，比较局部 pre-gate/gated 输出与实际 selection，测 selection+两个 kernel+cache完整计时及真实tile/weight访问；先预热、交错重复、event与同步wall各记，编译/加载分别记。此仅支持单层局部而非全160视频或泛化。若结果可辨，再另立全 M 扩展检验和独立全视频复审，不能由此自动注册。

真实资产：模型身份来自 `config/e0_s2_new12_dense_baseline_part0_20261029_v2.json` 的 high/low safetensors 及 SHA、原生源码与包闭包；keys 按实际 checkpoint header 冻结，而非假设转置。已有 [whole-FFN桥](../../../20261029/S2_WHOLE_FFN_NATIVE_BOUNDARY_BRIDGE_RESULTS_20261029.md) 的 source911/456、Dense/value四视频只在 (0,0),(1,20),(2,20),(3,39) 每点保存32行，并非连续128行同expert同block相邻step；不能伪造为此probe的真实匹配输入。须另审、登记原正常forward被动producer：例如固定 source911 seed42 的 expert high block20 在 step0/1 相同连续128坐标保存 W1输入 `V1` 与 gate、行坐标和模型/phase SHA，同时原视频与10状态逐byte neutral；读入真实权重的地址和两个输入同时绑定。若缺实物、模型/环境或匹配坐标，保持 BLOCKED/NOT_MEASURED。单组不能证明多组内存复用或全M速度，且新producer本身也须完整独立审查及 ACTIVE。

候选有限预算供审稿而非授权：采集与单层kernel分成不同scope；kernel单 A100、CPU4、GPU≤24GiB、host≤64GiB、wall≤900s、outer≤1020s、HF≤512MiB；编译/核加载、输入权重和完整事件计数明确上界并预扣，失败保留RUN/EXIT，不自动重试。单组工作量小于真实M，禁止把其速度外推大M。静态实现前先冻结详细预扣（含重复和warmup）与独立评审；生成开销、物理内核trace或显存观测缺失即保持 UNKNOWN。所有执行决策以 ACTIVE 中实际批准为准。

## 后续资产核查更正

上述“须另审登记原正常forward被动producer”的说法过于严格：现有 spatial52 已保存真实匹配的 V1，可先用于一个**非raster重排128行**的单函数局部probe，无须新增采集。详细逐文件身份、组顺序和未观测边界见 [spatial52 可用性及固定probe补充](p0-chipmunk-spatial52-paired-input-probe-20261029.md)。若要原始连续128行或完整M的真实组行为，仍须新采集；不得把所选混合空间/时间组当作者原生raster组。
