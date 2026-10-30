# A5+HWG → 历史增量通道：规则性、门控函数保真与物理可执行性（Agent 执行任务）

**请求时间**：2026-10-30。**用户明确要求 Agent 实际执行研究**。对象仅为 Wan2.2 I2V A14B 四步全40 Block FFN 的训练自由历史增量动态稀疏。任务首状态 **ASSIGNMENT_REQUESTED / DESIGN_DRAFT / INDEPENDENT_REVIEW_REQUESTED**。本文是一项可交付的研究与实现任务，**不是主 Agent 已接手、不同 Agent 已审查、CPU/GPU 已执行、实验批准或质量证明**。执行时复核当前 HEAD、ACTIVE 和资源。

## 0. 第一动作：Agent 接手并分工

主 Agent 接手后在任务 Issue 回复实名/真实 agent_id、代码 checkout 的 rev、数据和权限现状，以 `AUTHOR_SELF_AUDIT` 生成本次具体实验自审。真正不同上下文的独立 Agent 阅读本任务、原作者A5/Chipmunk论文源码与旧数据提出反例；主 Agent 逐条修订、独立 Agent re-check，严格使用 [实验前独立复审状态机](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)。保留每轮审查及批次方案于 `research/experiment_reviews/E0_A5_HWG_DELTA_REGULARITY_20261030/`，只有批准后在 ACTIVE 注册有限 CPU/GPU scope、重新核对输入/代码/seed/预算、真实RUN/EXIT。**GitHub issue 不会自动启动Slurm任务；不得将本任务状态升级为 RUNNING。**

这里安排**两个并行职责**，不是预填独立支持：理论/审查 Agent 独立推导信息时点、强反例、计量和量化闭包；执行 Agent 实现已有资产预检、同预算离线比较、全来源/全层统计、必要的新采集及原生算子。独立实验审核角色不能由主 Agent 自己冒充。保持与 P0 SharQ/TEAL 共享资源计划，不挤占没有获得许可的GPU。

## 1. 重建原始对象和版本身份（必须做）

阅读并核对：
- [A5+HWG真实公式、身份与旧失败](../20261027/A5_HWG_SPATIOTEMPORAL_FUNCTIONS_FLICKER_AND_SMALL_OBJECTS_20261027.md)：A5 为 HiF4 W1 符号/分组尺度近似 preactivation，HWG 为 \(\|g\odot W2_j\|\)；历史约44.3%逻辑零值在**完整 W1 之后**，不代表实际 W1 省算；旧 threshold 和旧 parent TopK 不能混作同一个正式版本。
- [A5最早正式三指标](../../docs/04_前期探索与证据边界.md) 及原 `evidence/context_20261011/04_实测记录.md`，确认真实历史版本、单独快照和原 video/mask/protocol locator；如果不能拿到冻版源码，**将新方法标 A5-inspired adapter 而非“原A5复现”**。
- [Chipmunk真实全M输入/付费P64](CHIPMUNK_FULLM_CAUSAL_PROXY_RESULTS_20261030.md)、[局部后验分组与非随机raster优势](CHIPMUNK_RASTER_GROUP_ORACLE_RESULTS_20261030.md)、[视频四臂八轴](CHIPMUNK_P64_FACTORIAL_QUALITY_RESULTS_20261030.md)；本任务和 Chipmunk 作者原始 mean-RMS 代理不同。
- [TC1固定当前贡献支持组内复用负结果](../../reports/worker_20261020/TC1_coherence_summary.json)；不能把该旧测试当增量 mask 的已测试证据。
- [TEAL全M16/32当前支持并集上界](TEAL_WAN_FULLM_ROW_GROUP_SUPPORT_RESULTS_20261030.md)、[HiF4原64组闭包](../../docs/69_HiF4分组边界与空间动作的可执行布局.md)。
- [完整八轴和旧 UB1/S2/SharQ 对照](WAN_FFN_FULL_QUALITY_AND_MOTION_MECHANISM_COMPARATIVE_AUDIT_20261030.md)、[旧文件身份索引](WAN_FFN_CROSS_METHOD_ASSET_PRESERVATION_MANIFEST_20261030.json)。

数值族分开冻结：A5/HiF4 的 W1 符号与scale、Q1/Q2 均属于旧 `HiF4 fake-QDQ+BF16`；Chipmunk 的原 `BF16` 存量张量可作为**先验、debug和BF16机制对照**，但**不能悄悄直接作为 HiF4 A5 的同数值族因果验证**。新 HiF4 匹配tensor不足时，单独登记producer真实补采，并留存BF16独立对照。两expert high/high/low/low、每expert40块、\(M=31668,D=5120,H=13824=216\times64\)，四步共160 FFN位置；\(21\)个 latent-time 坐标不是四个去噪step。

## 2. 冻结科学对象/数学问题

同一专家和block的先前实际完整 FFN：
\[
a_0=W_1 x_0+b_1,\ h_0=\mathrm{GELU}_{\tanh}(a_0),\ Y_0=W_2h_0+b_2.
\]
当前输入 \(x_1\)、当前gate \(g_1\)，可从历史同身份 \(x_0\) 计算低价 A5：
\[
\widehat\delta a_j=\operatorname{A5}_j(x_1-x_0),\qquad
\widehat\delta h_j=\mathrm{GELU}(\widehat a_{0,j}+\widehat\delta a_j)-\mathrm{GELU}(\widehat a_{0,j}).
\]
近似delta代理避免将 \(\mathrm{GELU}(\Delta a)\) 误写为 \(\Delta\mathrm{GELU}(a)\)；若仅保存历史 \(h_0\) 而没有 \(a_0/x_0/\widehat a_0\)，则另算缓存或明确代理版本。在同expert W/b相同才有上述输入差分; high→low绝不能复用彼此history。对单channel的原始HWG增量评分
\[
s^\Delta_{rj}=|\widehat\delta h_{rj}|\cdot\Vert g_{1,r}\odot W_{2,:,j}\Vert_2.
\]
在线评分需清晰标价：对当前gate逐行计算HWG不可当预计算常数，完整W2 \(\Vert\cdot\Vert\) 或Gram/分组 signed output可能很贵；离线 full-W2 oracle 是**已付信息上界**，必须与在线合法选法分开。若使用precomputed step/block类gate近似，需列出质量成本与真实时点。

按选择集合 S 仅当前部分 W1/GELU/W2 delta 输出
\[
\widehat Y_1=Y_0+W_{2,S}(h_{1,S}-h_{0,S}),\qquad
\widehat u_1=g_1\odot\widehat Y_1.
\]
单专家首次完整计算与缓存旧h/Y均要计成本；第二次不能偷用当前完整 \(h_1\)、\(Y_1\) 选择S。原HiF4模型的Q1/Q2激活尺度、Q2的64/8/4层级和BF16舍入在正式数值检查中严格对齐；上述连续算术恒等只是算法定义，不是QDQ逐bit保证。

**三个竞争假说**（独立review应补至少一个难以同时满足的反例）：
- H1：相比原A5当前贡献，A5增量在真实跨步分布更集中，特定 K 下保留更多 gated signed 当前变化/终态质量，导致有效更新通道减少。
- H2：增量能量虽集中于单通道，但分布于绝大多数原64-group和M16×K64 tile，因此**逻辑稀疏不形成规则原生省算**，与TEAL/TC1旧负证据相近。
- H3：增量的时空局部相关性允许分组共享（M16/M32/M64，连续raster与随机shuffle比），在合法64组边界和相同实际MAC下降低门控写回误差；若需全当前h/W2求分组评分，优势只属于paid oracle。

特殊反例：\(\widehat\delta h\) 范数低但某些W2方向经gate、下游consumer影响巨大，或 \(\widehat\delta h\) 近似噪声使支持翻转并诱发闪烁；空间RMS高并不能保证运动事件保真。保持zero-delta/历史Y-only、random、static好组和同K Chipmunk作为强对照。

## 3. 先做资产预检，然后开展真正实验

**M0 / 数据覆盖地图（Agent立即执行，纯文件检查）**：构造 `source × expert × step-pair × block(0..39) × row/time/position × numeric-family × tensor-boundary` 表；检查raw x0/x1、原/代理a0、旧h0/Y0、current h1/Y1、g1、W1有效权重/Q1参数/W2/Q2 scale、source/seed/condition、teacher与student身份、existing raw HF路径、是否可独立验证原输出与中性条件。标 AVAILABLE / MISSING / INCOMPATIBLE / NEED_NEW_PRODUCER。保留原Sha或文件身份不覆写; 现有registry**不是实体备份**。不得只因目前block20最完整就将研究永久限制为该层。

**M1 / 代数与小张量验算（无科学实验冒充）**：同输入差分、bias抵消、GELU、gate变化、W2 signed组合及BF16/HiF4 Q2合法整64group/原scale；证明原Dense中性与历史缓存首次调用等价路径；独立用非oracle信息重放 `A5-current, A5-delta` 输出支持，固定K并冻结所有随机tie及group order。原有效Q1/Q2的post/pre掩码的含义必须拆开。

**E0 / 一次完整全层/双expert多来源分层的离线发现实验（需独立review+ACTIVE CPU scope）**：
- 同K or 同MAC对照 `A5-current+HWG`, `A5-delta+HWG`, `A5-delta-noHWG`, `Chipmunk-mean-RMS P64`, `static-best calibration group`, `random`, `oldY-only`, `true-current-delta posterior oracle`。原A5源码不可恢复则在name/科学结论单列版本。
- granularity: 单token×hidden, 全M共同mask, 同空间raster token tile(M16/32/64), shuffle row negative control; 原hidden 64整组、K16/K32也作为**诊断参照而非自动HiF4合法核**。每组输出mask稀疏率、非空K64比例、实际全零/跳过M×K tile数、行组column-union、连续组长度、各tile支持Jaccard和时间支持切换率、source/expert/block tails；计算理论W1/W2主MAC与观测selector/pack/offload代价。注意对于A5原固定阈值不天然相同K；实验中保留原阈值臂和等K臂两套结果，不能替换历史身份。
- quality: 当前\(\Delta h\)捕获能量、门控signed输出相对L2、符号交叉与direction cosine、postwrite、局部空间/latentτ动态AC，最终消费者有限差如可复用；mean/std与独立来源cluster bootstrap，禁止把token/block/step当独立n。报告proxy vs exact\(\Delta h\)排序差异及依赖的已付数据量。使用reader-visible每源×40层×两expert图/CSV；若因资产缺少某些层则**主动补采对应真实状态**并标明新增scope，不退化成只有首末/单block。
- cost: shared mask冲突、K64组数与M分组、cache h+Y真实大小、A5对\(x_1-x_0\)的计算时延、HWG gate perrow成本、W1/W2 selected GEMM完整时间。纯oracle错误不能代替native成本。

**E1 / 原生执行可达性（另审scope）**：E0找到真正可执行的稀疏结构后，以BF16/HiF4各自neutral及same-format Dense，对照native grouped selected-W1/GELU/selected-W2-delta、packing、cache, call数、Q2 group closure、operator FFN event 与 FFN-only speed；若实际仍做full W1或W2，只叫数值模拟；CPU逻辑MAC≠原生GPU净收益。

**E2 / full160 81帧student验证（另审scope）**：至少多源、多独立seed，H/L/两者因子，且有source held-out；同seed/初始噪声、同模型/codec/权重、真实完整40×4、Neutral/oldY-only/Chipmunk等K/static/A5-delta候选，**完整八轴**以及连续RAFT逐时间窗、camera_motion(如能部署)、flicker、小物体/主体轨迹、背景照明、相机运动与真实事件变化。没有的指标标NOT_MEASURED。保存完整81帧、4step flow/latent和所有原生计时。不要只凭RAFT幅值宣布“保住运动”。

## 4. 预先建立 REUSABLE_ASSET_PLAN，保留全部旧对照

| 资产类别 | 必须保存的科学身份 | 复用和验收 |
| --- | --- | --- |
| `A5_FULL160_BOUNDARY_COHORT` | source/seed/expert/step/block/latentτ/grid/token selection, x0/x1, preQ1/postQ1 a0/h0/Y0/current gate, quant scale, backend, dtype | 新旧数值族分开，不混BF16与HiF4，要求真实source路径及读取后的文件级校验 |
| `A5_PROXY_AND_SCORE` | A5源码版本、W1 proxy参数、旧/新preGELU代理、delta score、HWG模型和支付成本 | 校准/测试分开，记录离线/在线是否合法 |
| `DELTA_MASK_FAMILY` | 每arm每source×site×tile的mask索引、组、tie policy、随机seed、K、native byte layout、有效率与实际skip单位 | 等K比较，保存原版threshold arm；旧资料原样保留 |
| `FFN_SIGNED_AND_COST` | BF16/HiF4同身份dense输出、oldY+delta的gate/postwrite、error decomposition、CUDA事件/CPU时间/路由/pack/cache开销 | 付费oracle隔离，same-format计时 |
| `VIDEO_QUALITY_REPLAY` | 初始noise、81帧MP4、四步state、8轴逐来源原值、RAFT时间轴、event/camera/ROI、visual contact sheets、RAW RUN/EXIT/失败 | 完整student质量和物理收益不可从局部代理直接推断 |

资产plan/record记录producer、具体revision、schema、shape/coordinate/dtype、输入依赖、运行配置、位置、是否可确定重放、适用与失效条件、验证状态。既有[876条SHA索引](WAN_FFN_CROSS_METHOD_ASSET_PRESERVATION_MANIFEST_20261030.json)仅做追溯，不改变或删除任何旧视频/状态/失败，真实复制需在可访问HF的机上用[备份验证脚本](../../tools/verify_preserve_wan_ffn_comparison_assets.py)执行并留下本批RUN/EXIT。

## 5. Agent交付与PASS/FAIL/INCONCLUSIVE

- **先交付**：M0详细可复用资产覆盖与原始 A5 源码身份、方法图和明确的数据缺口、主Agent自审+真实独立review请求（未接手不能伪造）、可运行的M1静态验证源码。
- **E0完成**：真实CPU/必要GPU exit0，完整覆盖或显式报告必要新producer、各来源/40层/H-L分布的同预算 mask规则性和signed输出误差，完整RUN/EXIT、产生资产清单。报告明示在哪里比原A5/Chipmunk更规则、哪里失败。
- **E1通过**：实际可跳过合法的FFN W1/W2计算；按同数值族Dense公平计入代理、cache、gate、pack和物理kernel，明确净收益与显存成本。
- **E2通过**：全160真实生成的每来源八轴与事件/时序/相机/成本Pareto，在新source上确认、无明显质量崩溃，保留中立对照和失败。
- **FAIL也必须保留**：若增量只减少元素而不减少原K64/真实tile，就具体记录哪个层/专家/来源产生饱和与因果替代；若代理成本大，分辨A5/HWG贡献，转向group prior或稀疏值重构，不能降维到只看末层、只看局部L2。所有状态和指标以实际证据而非文档任务声明更新。

**该任务不得擅自抢占已冻结 P0 GPU 任务**。经review和ACTIVE/资源登记后，按实际可调用显卡资源分配；旧scope2卡限制按其记录执行，新scope不预写固定2卡或4卡承诺。
