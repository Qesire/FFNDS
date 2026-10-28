# FFNDS · 问题驱动的开放式数据采集与机制判别

**2026-10-29 · CURRENT RESEARCH PROTOCOL · DISCOVER → REQUEST EVIDENCE → ACQUIRE → DISCRIMINATE → REFORMULATE/BUILD**

现有资产是科学发现和防止重复计算的起点，**不是可研究数据、采集规模、采集变量或方法空间的上限**。对尚未观测的source、完整层/step、运动配对token、原生量化动态码和真实学生状态，需要主动设计新的原生producer、source cohort和有限受控实验。新研究可直接由原论文、源码、数学反例、业界算子或新的物理瓶颈提出，**不要求先证明现有8源、旧20源或五个观测层中存在某种结果**。

原AutoResearcher四技能配合使用：mechanism-research找真实计算约束，problem-reformulation变换科学对象，research-expansion开放新方向，web-research-discovery核实先例和独立竞争解释；ffnds-experiment-evidence保证source、teacher/student、数值路径、视频与性能证据一致。

## 1. 必须主动启动补采设计的条件

1. **关键变量缺失**：没有真正latent视频时间—空间对应token、Q1或Q2动态code和scale、实际W1/W2物理tile、后缀输入/输出或student自由轨迹，就构造该边界的真实native producer/capture；先补观察渠道再设计推断。
2. **机制无法区分**：至少两个竞争解释对特定source/step/block/scene或输入状态有相反预测时，就选择这些分歧位置采集最小反事实或自然数据；已有数据无法回答即申请新观测，不能靠修辞维持OPEN。
3. **来源数量、分层或状态不足**：真实image簇有限、训练与验证已暴露、关键动作/相机/静态控制缺席、teacher与student不一致时，扩充独立image clusters、种子和状态对照。256行×160位置不是256或160个独立来源。
4. **生成质量与局部指标冲突**：视频动作丢失、时序事件、背景/小物体、相机、语义不一致时，采集成对视频ROI与帧级事件、对应4步student-prefix和原生计算记录；然后进行有限干预寻找真正承担者。
5. **后端与量化改变**：HiF4 fake-QDQ、原BF16、原生NVFP4/MXFP4分别采集真实W静态与A动态scale/code、量化producer和scale swizzle、layout、路由/packing/GEMM和wall；旧whole64闭包不自动迁移到native K16/K32。
6. **新证据产生更强反例**：若新增来源改变结论，定位子群、层、step、motion条件或舍入阈值并继续分层采集；样本规模和新变量由科学问题决定，允许在已有S1–S4之外提出新研究方向。

**循环**：发现问题 → 两种相反预测 → 确认旧资产可回答的部分 → 定义新的source、token、原生tensor或学生状态 → 冻结有限RUN范围与预算 → 实际采集 → 以image-cluster为独立单位分析 → 判断继续采、转向新函数或进入完整视频。新采集是科研主动作之一，和分析已有资产具有同等优先级。

## 2. 当前最有价值的新增资产

| 路线 | 决定性缺口 | 新数据采集内容 | 竞争机制判别 |
|---|---|---|---|
| **S1 · 共享/step特化函数** | BF16仅6个训练来源、6折共用样本、8函数锚点、2旧开发；全部历史暴露 | 不同image clusters，分层动作幅度、相机、背景/前景、光照、静态控制；原BF16所有4×40位置的真实W1输入、GELU、W2、gate、归一化后状态与来源信息。按需扩容到12、24、48甚至更多独立来源；只把这些数字当实验设计点 | 逐step独立拟合的训练优势是小样本方差，还是step确实需要另一低维函数子空间？校准、开发与真正首次确认分源 |
| **S4 · 动态秩的行动收益** | 已有低秩因子和原BF16 teacher h；尚无来源外在线草图regret与学生状态 | 扩展source×seed×40Block×4step的post-GELU输入、真W2输出、低秩分块实际收益、物理row tile、在线probe代价，teacher/student两轨 | 提前便宜观察能否预测额外rank的真实收益，并在同实际执行单位上超过固定rank；若特定层/step有分歧继续采 |
| **S2 · 时空函数组织** | 旧只有2来源×5Block的raster M128，新BF16随机256行不形成足够的21个latent时间位置配对 | 全40层×4去噪step，带原始latent (time,y,x)标识的**时空tubelet**，覆盖21个时间槽×29×52空间网格；每个source选运动目标/背景及跨时间同位置对照，保存真实preW1、h、W2、gate/残差。需要时采集更多source或运动对齐特征（在线获取成本单列） | layer-invariant共享 vs 层特异共享；普通raster邻接 vs 真正的时间/运动配对；采集不足就补齐，尤其不能只看block0/39 |
| **S3 · 原生量化先验与早决策** | 旧whole64仅HiF4 fake-QDQ；native FP4动态A元信息未通过一致的全图检查 | HiF4及原生NVFP4/MXFP4分开：实际Q1/A1 code和scale及其首次可读时点、W1/GELU、Q2/A2 code和scale、whole组/物理tile贡献、完整gated output、原生producer/layout、同步和kernel耗时；覆盖两expert和全部四步/层，逐source记录 | 合法可执行的preW1省算动作是否存在；便宜的signed/scale/weight×activation信息在同成本下是否比step-only/cheap BF16真的增加行动价值 |
| **跨路线VIDEO/SYS** | BF16 shared80/step160-r960尚无自由生成；旧UB1视频已暴露、motion退化；只有有限microbench | 新的完整160site Dense/候选配对，动作事件/语义/相机/主体/背景/时序/画质、81帧视频和4步latent与学生自身状态；同源同噪声和真实FFN/kernel/整机wall、显存。需要时增加新image簇与seed | 局部teacher拟合能否变成学生完整生成能力；何处功能发生变化；动作与画面是否同步保留；实际计算是否减少 |

**新source从哪里来**：检查真实dataset/revision及可读取的原始输入，扩展成带scene_stratum、image_id、prompt_id、conditioning_id、exposure_history的source-bank；可使用新数据集或新采集源，不受旧VBench消费20图像集合限制。校准、开发、冻结确认须按真实image簇隔离，旧图换prompt/crop/seed仍属原簇。若某scene尚无足量source，继续扩展采集并标明暂缺来源，不能用row/seed冒充source数。候选的20开发来源只是第一次视频研究设计，后续可根据来源级效应、尾部分歧扩大；官方5样本仍按官方独立生成协议执行。

**覆盖结构**：机制研究的深度以4step×40Block的全图为默认；每个位置可按明确设计存真实时空tubelet和抽样row，避免一次性将31668全部token×每层所有tensor写入巨量文件。重点是采集变量的语义足以区分两个机制，而不是一概增加张量体积。离线Dense计算未来光流可以用于发现/监督，但若在线方案依赖这个信号，必须单独构造低成本真实producer并计费。

## 3. 将采集落到实际执行，而不是只写OPEN

本仓库已存在可复用的source-bank、calibration-plan、native geometry、video-plan、cohort preflight与registered-runner，详见 [EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md](EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)。使用它们做真实cohort和固定计划；如研究需新的时空tubelet、动态Q1/Q2捕获或原BF16学生因子runner，就先实现或扩展对应**原生producer**、输出每source的真实tensor locator、checkpoint/numeric/path、frame/step/block/token坐标、实际成本。旧HiF4/UB1 runner不自动支持新的BF16和NVFP4候选。

每一项采集在 [ACTIVE](../ACTIVE_RESEARCH.json) 中单独冻结有限执行范围：真实source清单和暴露历史、模型与量化recipe、teacher/student、token采样、producer接口及形状、具体需要的全深度覆盖、CPU/GPU/存储/时长预算、对照条件、成功和停止标准。资源依 [DYNAMIC_GPU_RESOURCE_POLICY_20261028.md](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md) 使用当前实际能分配并独立运行的最大有效GPU数，不沿用历史2/4卡上限。原scope结束可按新结果继续注册下一批，仍记录实际RUN与失败，既不默许无限循环，也不被第一批预算设成永久科学范围。

**获取增量证据的标准**：本次采集能否让最强两个假设出现不同预测并被source-held数据区分？是否观测到真实算法需要的变量与时点？是否检查包含新source的尾部及失败例？若否，改变采样分层、状态、tensor边界或数学对象，再继续采集。没有必要提前设固定数据规模上限或将“旧数据不支持”当成结论。

## 4. 状态

本文件登记采集方向，不声称已运行新模型/GPU、已新增未暴露source或已经观测到native NVFP4在线activation metadata。原始历史RUN不变。对应机器可读需求及状态见 [ACQUISITION_QUEUE_20261029.json](ACQUISITION_QUEUE_20261029.json)。原研究路线、[统一Agent交接](ASSET_FIRST_AGENT_HANDOFF_20261029.md)、[数据使用协议](DATA_SAMPLING_CALIBRATION_EVALUATION.md)及[视频指标协议](VIDEO_GENERATION_METRICS_AND_FULL_DEPTH_POLICY.md)继续生效。