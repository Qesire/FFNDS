# G17 v3 · 完整视频—全深度张量—A5/HWG同状态比较联动与生成代码归档审计

2026-10-31 · METHOD_SNAPSHOT_EXTENSION / CODE_ARCHIVE_STATIC_AUDIT / DESIGN_ONLY / NATIVE_TENSOR_NOT_REREAD / INDEPENDENT_EXPERIMENT_REVIEW_REQUIRED

## 0. 审计身份、结论和原则

本次静态审计核查FFNDS main文件树与四份真实冻结配置中的\`repo_sources\`文件路径：当前Git tree完整列出4,857 blob、1,463 Python脚本（非GitHub搜索结果的抽样）。抽取的四份冻结配置分别为P64 factorial part0、A5/HWG HiF4 matched capture part0、A5/HWG E0 source898和原BF16 video20 part0。逐配置记录的FFNDS内部路径项及当前树存在性为197/197、14/14、28/28、48/48。**这些检查是当前Git路径存在性，非冻结运行源码逐字节等价、更不是全仓库每个RUN覆盖证明**。

已找到核心生成、拦截、张量采集和质量运行入口；尚不能宣布所有生成代码完整归档。已明确缺口：原始A5完整参数producer与compact参数尚未找到；外部LightX2V/HiF4/W4A4的repo_sources使用运行环境的\`.runtime/code/...\`文件约束，Git中保存的是上游固定版本＋部分delta恢复机制，不是每次RUN的整个运行时代码原树；真实HF视频、原生fullM张量与模型/编译产物本轮不可读。**保存了源码文件路径≠实测使用的是同样源码≠整个生成环境能重新构建。**

现行视频机制合同：[G17 v3完整视频优先](G17_V3_VIDEO_FIRST_FFN_MECHANISM_FULL_ROLLOUT_DESIGN_20261031.md)。本文件将**真实视频行为、沿生成状态传播的张量对比、HWG系列组信息和物理执行**组成互为约束的证据三角；绝不能只讨论局部L2或仅视频RAFT。

## 1. 科学逻辑链的三条相互约束证据

\[
\underbrace{\text{真实81帧完整视频} \;\to\; \Delta Q_{\mathrm{event/motion/temporal/imaging/text}}}_{\text{行为与最终质量}}
\quad\Longleftrightarrow\quad
\underbrace{\{\delta u_{k,l},\delta r_{k,l},\delta z_k,\delta V\}}_{\text{全深度、学生状态与传播}}
\quad\Longleftrightarrow\quad
\underbrace{\{\Delta h,\;g,\;W_2,\;m_b,\;\sum_b gP_b,\;K64\text{可执行组}\}}_{\text{A5/HWG及正常FFN算子}}
\]
三者有不同证据身份。VIDEO为最终动作/语义/画质检验；TENSOR提供竞争机制区分；HWG对比提供**哪些信息值得保留、何时可低价可得、是否省真实W1/W2**的算法候选。VIDEO发现并非必须等待所有Tensor完成；Tensor可以先基于已保存合法资产比较，形成下一次视频干预的可辨识预测。**所有新方法最终必须完整4step×40FFN生成。**

**核心待证伪假说**：缺少某些当前FFN fresh写回可能经有门控的receiver与去噪动力学改变动作进展和主体时序，同时静态身份未必同幅度受损。强竞争：纯容量降低/成像退化、相机/背景flow混淆、student-prefix与后缀放大、缓存错配（仅适用于cache）、量化闭包与真实kernel数值路径。试验要让这些假说在视频行为**及**内部计算边界均给出相反预测。

## 2. 全深度Dense/Student原生张量比较（T系列）

**T0 配对合法性**：同source、同image/prompt（注意额外camera instruction）、同seed与初始噪声、同weight、scheduler、precision、numframes、artifact来源。原生HiF4 fake-QDQ+BF16与raw BF16不同数值族；可分别分析但不能把两者的张量直接相减当因果变化。将所有sample带\`source_cluster,run_id,arm,seed,k,expert,l,latent_tau,y,x,quant_family,teacher/student,producer\`。完整调用坐标为4step×40layer=160；跨同expert同层历史对是80，而不是160个独立第二次调用。

**T1 原生同输入函数误差**：对同一个**学生实际输入**分别运行合法Dense shadow和相同输入下的具体稀疏操作；比较preW1（包括norm/AdaLN身份）、W1 preactivation、GELU、Smooth2/Q1/Q2（仅HiF4）、W2 pre-gate Y、g、postwrite gY、residual。No-cache的遗漏对象是当前\(\sum_{b\notin S}h_{1,b}W_{2,b}\)；Cache-delta的遗漏是\(\sum_{b\notin S}(h_{1,b}-h_{0,b})W_{2,b}\)；函数重表示按实际\(\tilde f\)输出。不得混写。局部误差按\`source×step×block×ROI×quant_family\`、signed方向、p99/p999及数值闭包记录，不单给全局SSE。

**T2 教师与学生前缀/状态漂移**：对原始自由Dense与自由Student在同一个真实物理边界取\(\delta r_{k,l}\)、该step预测flow/velocity、latent_step与下一step输入；逐层输出是受上游全部变化影响的观察差，**不能直接说是当前FFN的因果份额**。匹配教师输入与学生输入的Dense shadow，分别记录同输入局部函数差、状态条件差及有符号交叉；不同层统计不能在未控制依赖时相加“根因百分比”。冻结有效的tensor时间映射、21个latent τ与81解码frame映射，以真实geometry验证，区分τ0。

**T3 动作语义对齐**：先由已有真实视频的主体/背景/相机分层与事件起止建立冻结ROI和track；鸟898最大Farneback flow落在植物背景，不能冒充主体ROI；96若HiF4 Q2缺失登记MISSING。记录对象track失败、主体消失、动作缺失及遮挡，不将其从分母中默默排除。对内部响应同时计算fixed-xy与合法配准后的时间差、符号/方向/强度/相位，结合主体速度、camera residual、事件onset。低warped变化可能代表正确随运动输运，不自动表示不重要。

**T4 机制路径裁剪**：在视频中确实发生恢复效果的指定site进行O输出边界、D固定postQ2、Q完整preQ2原生闭包、P后缀/receiver定位。干预必须进入完整四步自由生成，最终视频变化是验收。若只是当前同状态局部有限替换只能登记TENSOR_CAUSAL，而非VIDEO_CAUSAL。首尾及中部层都要覆盖，且层数/预算匹配；不以单source/末层得出全局机制结论。

## 3. A5/HWG系列的张量比较（H系列，和视频主线并行）

### H0 已存资产与限制

已完成HiF4 A5-inspired fullM：来源898、1040、979；1040用于CAL1，898/979已曝光，seed42；每来源80同expert层对×58策略，共13,920行；数值为原HiF4 fake-QDQ+BF16同状态teacher，\`M=31668=21×29×52\`、\`H=13824=216×64\`。全局K5120整64组的现有相对当前输出误差：calibration_static 0.443230、a5_delta_hwg 0.456756、a5_current_hwg 0.465695；有局部signed与物理tile统计。**这些数据无法取代自由学生视频，也不是原A5忠实版或原生skip计时**。96采集失败，不能当成功源。参照 [E0完整报告](../20261030/A5_HWG_DELTA_SUPPORT_REGULARITY_RESULTS_20261030.md) 与 [M0覆盖](../20261030/A5_HWG_M0_M1_READINESS_20261030.md)。

### H1 必须并列比较的算法/信息对象

在**相同group、K、数值族、参考state与可复现原始行坐标**下，至少并列：
- 当前A5×HWG：\(|\widehat h_1|\,\|gW_2\|\)；历史增量A5×HWG：\(|\widehat h_1-\widehat h_0|\,\|gW_2\|\)；增量A5 without HWG。
- 静态calibration signed联合组（CAL1）；当前真实postQ2增量的paid-diagonal上界；原P64 \(\Delta h\) 抽样代理；random及oldY-only零更新。
- 新的HWG signed-group共同选组与受限gain（若使用BF16新路线，**单独校准/采集/视频**，不得重命名HiF4结果）；同mask的gain/no-gain对照，明确gain、cache、mask分别影响什么。
- 无缓存的whole-K64 current-value选组与cache-delta等K竞争，以及representation替代FFN算子（至少保留独立路线，不预设补算）。

上面A5_CURRENT、A5_DELTA等代理输入可能已执行全M W1，是**付费分析**；保留paid/oracle与在线真正前置可用信息两个版本，后者才能申请方法部署。

### H2 张量比对坐标与统计

针对每个\`source × expert × block × τ × y × x × group64\`，保存或在原资产上只读求：
1. **估计误差**：真实postQ2 \(\Delta h\) 与A5代理的signed correlation、符号翻转、p50/p90/p99/p999、scale/饱和、ranking overlap/Jaccard，按source/layer/ROI/step分层；
2. **函数误差**：每组 \(P_b=\Delta h_b W_{2,b}\) 经当前gate后的有符号向量、局部Gram \(G_{bc}\)、互抵与总体合成；当前值与增量在不同方案中分别定义，**post-Q2真实差必须分别量化后再相减**；
3. **规则性与物理**：K64/M16/32/64真实tile occupancy、raster vs shuffle、跨空间行组与latent τ的支持稳定性、不同专家预算及channel support overlap、实际W1/W2跳算/packed kernel/selector/gate成本；
4. **张量→视频对应**：按既有真实视频中的主体事件/相机/背景ROI和五段时间窗，比较Dense/Student差异来源与组选组分布，记录**反常失败**：局部L2低却动作更差、RAFT大却主体事件错误、选择支持高相关却完整视频不恢复；
5. **跨方法边界**：HiF4 A5三源与BF16 Chipmunk/S2旧来源只能做**来源/现象层级的外部一致性**；若要相同输入数值相减，需要新审的同数值族、same-run matched producer。

### H3 检验A5/HWG的因果机制而非增加分数

先通过T+H在某个video失败source冻结可辨识预测：例如“低噪声某层/某ROI被丢失的fresh signed写回方向比随机同K组对动作更关键”。在**同一真实完整视频生成入口**比较P64 / 原A5-inspired / signed static / signed incremental / gain同mask / no-cache / random；固定可执行K64与计算成本；全部真实四步自由传播。现有[HWG Signed+Gain视频v2](HWG_SIGNED_GAIN_VIDEO_DESIGN_V2_20261031.md)可作D/G/R候选合同，但其当前标志为设计/无GPU授权，不能冒充已生成10/72/240视频，且其主设定是BF16，不可用HiF4 E0统计自动替代。若某组局部代理好而视频失败，要用视频失败反过来改变核心机制和可得信息假说，不继续只调分数。

## 4. 生成代码保存性：当前Git和RUN配置实际检查

### A. Git树/冻结配置的确定事实

- 当前完整Git tree：4,857个blob、1,463个\`.py\`；\`tools/\`923个文件（均Python）、\`scripts/\`15、\`tests/\`440个Python。Git tree返回\`truncated=false\`。这些是当前Git内容统计，并非原RUN环境的完整文件数。
- 4份代表性冻结配置在当前Git中**仓库内部文件路径**存在性：P64 factorial 197/197，A5 HiF4 matched capture 14/14，A5 fullM E0 source898 28/28，原BF16 video20 48/48。只验证存在，不证明执行版本的逐字节身份。
- 上述配置分别还绑定889、1007、1007、888个外部或运行时条目，很多位于\`.runtime/code/LightX2V\`、\`HiF4_Sp\`、\`.runtime/inputs\`等；这些**不是本仓库普通源文件**。不能将repo_sources清单当作源码已全部随Git保存的证明。
- [upstream_audit/README](../../upstream_audit/README.md)记载2026-10-20恢复基线：固定LightX2V/W4A4/HiF4提交 + 92差异文本文件与5补丁，未包含完整上游源码、编译对象、模型、数据；恢复工具可供单独源完整性检验。此范围不能推定2026-10-29/30运行时未再改变。
- [A5/HWG M0/M1](../20261030/A5_HWG_M0_M1_READINESS_20261030.md)明确原A5参数producer及compact参数未找到，新E0采用A5-inspired版。**原版A5完整生成代码/参数：MISSING_OR_UNVERIFIED；不可宣称忠实复现。**

### B. 已核验存在的关键入口（只代表Git保存，不代表运行成功）

| 实验/任务 | 已在Git的可检查源码入口 | 精确生成/复现状态 |
|---|---|---|
| 已有Dense/UB1视频 | \`tools/ffnds_run_registered_video.py\`, \`tools/run_ub1_consumer_disjoint20.py\`, \`tools/ffnds_stage_registered_i2v.py\`, \`tools/ffnds_video_metrics.py\` | 实验专用driver存在；运行仍依赖冻结LightX2V/HiF4、外部模型及HF输入 |
| Chipmunk global-P64全160 | \`tools/chipmunk_wan_global_p64_driver_v2.py\`, \`tools/chipmunk_wan_global_p64_tap_v2.py\`, \`tools/e0_chipmunk_wan_global_p64_owner_v2.py\` | 本仓源码存在、历史真实视频RUN闭合；非新site-restore runner |
| Chipmunk High/Low 2×2视频 | \`tools/chipmunk_p64_factorial_driver.py\`, \`tools/chipmunk_p64_factorial_tap.py\`, \`tools/e0_chipmunk_p64_factorial_owner.py\` | 三种新phase + reuse Dense/Both的真实生成入口存在；**不要重复生成已完成四臂** |
| A5/HWG HiF4匹配采集 | \`tools/a5_hwg_matched_capture_20261030.py\`, \`tools/a5_hwg_matched_capture_namespace_v2_20261030.py\`, \`tools/a5_hwg_matched_capture_relocation_v3_20261030.py\` | 原生采集相关代码存在；source96缺采是运行失败，不是将其标为有数据 |
| A5/HWG统计、选组与重放 | \`tools/a5_hwg_e0_native_replay_20261030.py\`, \`tools/a5_hwg_registered_delta_discovery_20261030.py\`, \`tools/a5_hwg_registered_static_fit_20261030.py\`, \`tools/a5_hwg_registered_core_summary_20261030.py\` | 已保存A5-inspired与CAL1及58臂E0的核心代码，**不是原A5参数producer** |
| Signed-HWG/Gain新视频 | \`config/hwg_signed_gain_video_v2_20261031.json\` 与研究方法设计 | **DESIGN_ONLY；没有由此证明已存在可执行D/G/R完整视频runner** |
| G17所需任意site修复并继续四步 | 本次未在代码中确认具有该功能的通用runner；旧专用driver已有固定High/Low | **OPEN_BINDING / NEEDS_IMPLEMENTATION**，不可复用旧Dense/UB1-only入口冒称已支持 |

### C. “全部生成代码保存”的严格PASS合同

对每一个**实际生成**的实验来源/运行，建立一个代码与数据可重建登记（可用现有RUN/配置/冻结代码绑定；不得在本轮假冒过去缺失的源码）：
1. **源码入口闭包**：原始运行入口、实际CLI与参数、配置/条件预处理、全部FFN hooks/kernel、量化/QDQ、offload、scheduler、decode、metrics代码；给出每项当前仓库文件或可重建的固定上游基线＋明确补丁，以及对应原run的版本身份。
2. **冻结配置和实际运行区分**：冻结计划中的\`repo_sources\`路径存在，只是覆盖候选证据；还须由实际RUN日志或部署记录确认实际导入源码、动态修改、运行时环境版本、third-party依赖与运行后变更。外部private worker/profile仅存摘要或标识，缺实体不得称重建成功。
3. **生成数据闭包**：原始prompt/image crop、seed/初始噪声、model weight及量化cache identity、precision/driver/GPU、完整81帧MP4、4步状态、FFN实际作用位置、RUN/EXIT、指标/可视资产及存储locator。Git没有存储大权重和HF全部payload是外部依赖，并非Git数据完整。
4. **真实可重建验证**：在隔离环境按合法已冻结版本重建源码并检查导入、neutral Dense函数/数值、跨step状态与视频；浮点/设备差异时使用预先规定容差，不用简单“文件同名”通过。模型/大tensor不可访问时只能标\`SOURCE_ARCHIVED_PARTIAL / RUN_REPLAY_UNVERIFIED\`。
5. **结果矩阵**：每个具体RUN给\`IN_REPO\`、\`RECONSTRUCTABLE_UPSTREAM_PENDING_VERIFY\`、\`EXTERNAL_NOT_ARCHIVED\`、\`MISSING_ORIGINAL_PRODUCER\`、\`DESIGN_ONLY\`、\`REPLAY_VERIFIED\`中的恰当状态。保留原始旧RUN身份，修复缺口时新建版本，不能回填假成功。

## 5. 与下一步实验审查的对接

**第一并行工作包（不新增GPU）**：
- VIDEO：沿已完成的898/96四臂及High-oldY真实MP4做连续播放＋完整81帧动作/事件/背景/相机分段，依已记录原始路径在真实有资产的环境执行只读分析（新CPU研究统计仍按独立预审执行）。
- TENSOR：按HiF4与BF16分别核对应源同RUN、同状态Dense/Student在FFN写回、第一receiver、step flow/latent与动作ROI的实际保存情况，制作\`source×step×block×boundary\`覆盖/缺口表；先使用历史全160数据，缺失再申请新的native producer。
- HWG：优先复用真实HiF4 3源全M 58臂与CAL1，逐source、High/Low、layer、τ与原始raster做signed/取消、estimate-vs-real、mask稳定/真实tile与motion-ROI对照；**这是科学主证据工作包**，不是video之后才考虑的附属项。BF16 Signed-HWG/Gain另建same-format采集/视频，不与HiF4混跑。
- CODE：为真实完成的Chipmunk P64四臂、A5 matched/E0、BF16 video20、UB1和HiF4源恢复分别编制\`run→entrypoint→repo_sources→external reconstruction→model/cache/inputs→RUN/EXIT\`闭包矩阵。首先处理A5原版producer缺失和外部RUN树的逐run源码版本。未完成时不可称所有生成源码保存完成。

**需要明确的第一个视频实验**：从已有四臂中选一个动作受损且能复核的source，在同数值族中做指定**当前FFN实际执行**恢复的完整free-rollout；对照High-only、Low-only、matched random与等预算组修复。视频行为改变与同一次RUN的T/H张量一同落盘。真实GPU/CPU实验前按[独立审查协议](../agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)由另一真实Agent评议/recheck并冻结ACTIVE，不得将本文视为执行许可。

**本轮审计实际范围**：Git tree、冻结配置及仓库文本/已完成报告；没有读取HF原始fullM张量、未检查远端worker当前文件、未重建所有原运行时、未运行任何新scientific CPU/GPU/视频。对无法验证的保存情况维持UNKNOWN/MISSING，而非推定PASS。
