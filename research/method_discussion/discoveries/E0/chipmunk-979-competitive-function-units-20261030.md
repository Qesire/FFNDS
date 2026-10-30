# 979：从全局隐藏幅度改为投影与轨迹的竞争机制

状态：独立 DISCOVERY / ALG / SRC / 已存 OBS；不是 SUPPORT 票、实验批准或新运行。对象是 Wan2.2 I2V A14B、原 BF16、四步中 high step 1 的 40 个 FFN，其他 120 位点 Dense。无 heldout、无人工连续播放验收。本文只利用既有 L09/C20 原文源码账本，不声称新增论文深化或 G14 落图。

## 现象与计算对象

[VIDEO](../../../20261030/CHIPMUNK_INPUT_VIDEO_QUALITY_RESULTS_20261030.md)：四个已暴露 source、seed42，898/96/1040/979 候选与 Dense 的连续 RAFT 比为 1.0155/0.9751/1.1318/0.7876，979 的 MUSIQ 差 −6.382。RAFT 测的是图像光流估计，MUSIQ 是图像质量模型；二者都不能单独证明运动或某个 FFN 单元的因果缺陷。AI 接触表显示 979 手脸姿态差异，不能替代连续播放或人评。[H/X 40状态与四视频字节相同](../../../20261030/CHIPMUNK_INPUT_VIDEO_H_X_BYTE_EQUIVALENCE_20261030.md)，逻辑 history 44.70→24.23 GiB，限定于这些运行；局部全M老输入重算与保存 h 的非零差也说明不应推广为数值恒等式。实测 FFN 时间 H 7.2919s、X 7.5745s、NONE 6.4101s，未证明整体加速。

[SRC/ALG] 令第 i 行同 token 两步激活变化 d_i = h_i(new)−h_i(old)，W2 的第 j 列为 w_j，当前 gate 为 G_i。以选集 S 表示的理想近似是 old Y + Σ(j∈S) d_ij w_j，残差 e_i = G_i Σ(j∉S) d_ij w_j。实际 BF16 delta 舍入、投影及加和路径有各自误差，必须按真实算子复核；old Y 包含 bias，delta 只线性投影一次。P64 以64个实际匹配 token 的 d RMS 选一个全 M 共同 K=5120/13824 mask；选择前须已经付64行当前完整 W1/GELU、旧h读取或旧x重算。执行账本见 [owner](../../../tools/e0_chipmunk_input_video_owner.py) 和 [full-M packed](../../../20261030/CHIPMUNK_FULLM_PACKED_NATIVE_RESULTS_20261030.md)。

## 三种可竞争的解释

1. **局部异质性 / 掩码公共化（H1）**：64行的全局幅度目标由高能区域主导；979 某些局部区域或时空片段的通道重要性不同，统一 S 对它们产生更大的遗漏。预测：相同状态下，空间/时间分区的同K后验 mask 明显降低各组尤其尾部的门后误差，且 P64 能事先预测分组差；若不同组最优集合高度相同或分组无收益，H1 被削弱。已有 raster 分组 hindsight 在 898/96 block20 四案例降低门后L2 8.26–10.90%，random grouping 也优于 global，说明组数自由度和空间邻近必须拆开比较；不是979证据，也不是在线速度结果。[本地结果](../../../20261030/CHIPMUNK_RASTER_GROUP_ORACLE_RESULTS_20261030.md)。

2. **有符号 W2 组合（H2）**：最重要的单位是共同投影后的残差，单列 RMS 省略了 w_j 的方向和 d 的跨列协方差。精确平方误差包括 Σ_j d_j²‖G w_j‖² 和 Σ_j≠k d_j d_k⟨G w_j,G w_k⟩；后者可能放大或抵消。预测：固定相同的 d RMS、K及局部划分，使用已有64行 signed 信息与当前 gate 或静态 W2 方向进行的投影动作，能降低同状态的门后误差/尾部误差；如果同预算多个有符号方法都没有增益而局部 mask 有明显增益，支持H1相对H2。现有 source898 单128行后验中 signed cross 占遗漏投影能量约36–37%，而门加权列 oracle 仅比幅度 oracle 低2.75%；两者的集合不同，既不证明因果亦不否定更好的联合动作。[证据](../../../20261030/CHIPMUNK_SAVED_H_DISCRIMINATION_RESULTS_20261030.md)。既有 P64 signed complement 在两个 high 案例改善而 low 一例略变差，提示适用性依状态而异，非统一解。[证据](../../../20261030/CHIPMUNK_SIGNED_COMPLEMENT_RESULTS_20261030.md)。

3. **后缀传播/交互（H3）**：哪怕同状态门后误差相近，不同位置和方向进入残差、后续 attention/FFN 与去噪后，会经非线性后缀产生不同视频；979 的偏差可能是小误差的高敏感方向，也可能是多位置交互。预测：将固定同状态、同一已保存动作误差注入后缀，不同位置/方向的最终视频响应差异大；配对单组干预的效果不能线性相加则出现交互。与H1/H2并非互斥：先验误差和后缀放大可同时存在。此前两源 HIGH/LOW/BOTH 指标交互有正有负，但动态 policy 的 step3 mask 随前缀改变，因此并非固定扰动的纯交互估计。[factorial](../../../20261030/CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)。同源自由轨迹 latent 相对SSE跨步变化也混合误差和参考能量变化，不能独立证明放大。[轨迹](../../../20261030/CHIPMUNK_P64_STATE_GEOMETRY_RESULTS_20261030.md)。

## 两个实际可执行的动作候选，互相作为强对照

**A：采样投影补偿（保留共同 mask）**。复用 P64 已算的同token signed d，得到遗漏列均值 m=(1/64)Σ d_i, m_S=0；计算 c=W2 m，按行加至 oldY+selected delta，再乘当前 gate。它改变 W2 的作用单位：保留 K 列全M W1/W2，并增加一条 H→D 的 W2 GEMV 和全M输出广播；不能声称减少这条补偿支路。静态 W2 可预持有；m 在当前64行 W1/GELU 和旧h可得后首次可得，不能由未来完整 h 生成。计入抽样、旧h/shadow、均值、zero、GEMV、广播、mask/索引、权重搬运、oldY/cache与同步；不将已有诊断 hash 当作必需线上操作。若 P64 漏采稀有区域，或采样均值符号错误，补偿可恶化；已选 mask 依赖同样本，均值不能标无偏。可加一个等成本随机符号/零补偿对照和 full-current-mean 付费后验上界，但后者禁止线上使用。

**B：固定静态分区、多组采样共享预算（改变 token×列动作）**。离线冻结两或四个坐标时空分区，分配总计64个相同已付的采样行，组内按本组RMS给每行K=5120独立选列；在各组对应 token 子矩阵上分别执行选中 W1 和 W2，仍保留各 token oldY 和 old h/x。按相同可索引 token 坐标比较等大小随机分区、原 global 64 行mask。分区固定在当前 FFN 前可得，本组d在采样 W1/GELU后可得；单组样本下降带来的估计误差是强反例。账本包括分组内输入/输出 gather/scatter、K列权重重复打包、若干小M GEMM、索引、缓存、kernel launch及 P64采样，确保每行同K但不假设有效硬件速度。粗分区可跨 tau、y、x，不能把 raster 连续线性分组称为运动对象；额外 full-current h 分组 oracle 只能作付费诊断。

A、B可在 **同 K、总64行、同当前student状态及相同输出参考** 下竞争；A的额外GEMV与B重复pack的代价分别实测，另报等实际时间的Dense/原P64比较。若 A 改善而 B不改善，优先H2；反之优先H1；两者都改善或均失败不能排他，保留H3和样本/数值路径解释。A 的均值广播无法修复纯零均值局部误差；B 无法保证跨列抵消，因而可互为具体反例。

## 最小下一 producer 与固定干预

1. **新被动采集，需逐scope独立复审、ACTIVE和RUN/EXIT**：979 与 898/96/1040 的相同seed42、原BF16，同 recipe Dense和 high-only student 自由轨迹；记录 40个 high step1 FFN 的真实同 token old/current 输入、old h/Y、当前 gate、实际64行坐标及mask，至少对预先按层深均匀抽样的位点保存全M current h 与当前W2输出用于后验诊断；产出坐标 `[tau,y,x]`、expert/step/block、dtype、checkpoint/code SHA及 neutral。全部40位点的轻量门后误差/尾部统计与采样摘要要覆盖；全M捕获用于裁定采样盲区，成本需预先预算，不能从64行推断全M误差。source为独立单位，site/row均嵌套；既有898/96 block20 Dense全M资产只提供跨模型条件开发对照，不含979和student自由态。
2. **固定状态反事实**：逐相同 student 输入和 old cache，比原P64、A、B、等组随机、全K及同设备Dense；汇报逐source/位点、总体和分区尾部误差、signed cross、采样与全M偏差、相同mask实现误差及BF16重算底。须包含 full-M hindsight作为额外付费诊断，不作为候选。若 no-action、全K 或被动 capture 不满足事先冻结的 neutral 容差，停止解释。未确定捕获/干预是否字节中性时，标数值容差而非推断字节相同。
3. **固定真实完整视频**：从相同 dense-prefix 状态、已冻结的位点集合与已保存的同状态 mask/扰动构造 single-site、成对及联合后缀，记录最终状态和视频；比较成对结果与单项总和的有限差，保留控制前缀与后缀数值路径。先在979和对照源定位方向，再以原P64/A/B各自在40处高step1自由选mask生成同source/seed完整视频，测独立运动、画质、主体、语义及持续人工播放；不能把冻结mask诊断当作在线policy泛化。固定动作的单点和多点显著差异支持H3；同时至少扩充未暴露图像簇与独立seed开展冻结确认。新实验必须另审，视频阈值、动作选择、预算、顺序和检验先冻结。

## 原论文与源码身份

[L09 Chipmunk](../../../literature/20261026/CHIPMUNK_SOURCE_READING.md) 和 [原作者/TEAL账本](../../../literature/20261029/P0_CHIPMUNK_TEAL_PRIMARY_SOURCE_LEDGER_20261029.json)：作者是跨步动态列 delta，blockmean W1 proxy 与原 Wan2.1 配置有自己的执行条件；本 P64 与 Wan2.2 四步是研究迁移。C20 TEAL 的当前矩阵输入阈值与作者单token sparse GEMV 是不同信息时点和执行对象，不直接为大M BF16提供加速。本文没有新增/深化论文，G14无需由本文宣称已更新；若root纳入新动作图，需自行登记对应原生anchor和状态。
