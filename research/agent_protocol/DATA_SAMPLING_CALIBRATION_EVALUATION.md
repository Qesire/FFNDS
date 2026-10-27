# FFNDS · 采样、校准、评测与数据使用协议 v1（2026-10-27）

**科学目标**：让每一次实验明确回答一个可判别问题；让来源覆盖、行采样、模型状态、生成随机性和视频功能彼此可追溯；以完整视频质量和真实系统成本评价可部署机制。I/F/C 可分别开展探索，实验精度与主张层级相称。

**适用范围**：新建和修订的实验协议、Agent 提示词、校准资产、确认集、VBench 与自定义视频评测。历史 `CAL12/CAL24/UB1/R17/E0` 的配置、RUN、视频和既有度量保留原样身份，审计和重评均另建版本。本文件是未来共同规范，不表示已经补跑历史实验。

## 1. 已核实的具体差异

| 对象 | 实际实现 / 数据 | 科学身份与改进动作 |
| --- | --- | --- |
| `evaluate_vbench3.py` 与 `evaluate_disjoint_vbench3.py` | 20 个来源各只生成 **1 条**视频，随后复制/硬链接为 `prompt-{0..4}.mp4`，供历史 VBench worker 匹配文件名 | 历史三轴结果是 **20 个独立生成样本的单 seed 内部比较**；五个名字不是五次独立采样；标 `HISTORICAL_REPLICATED_FILENAME_SINGLE_GENERATION`，不以100条独立视频或官方五次采样 leaderboard 对外呈现 |
| CAL12 → CAL24 | 12×512 → 24×256：**每个 FFN 位置同为 6,144 行** | 来源广度与每源行密度共同变化。后续用固定总行数与固定来源数的正交对照辨别；`rows` 不是独立来源数 |
| OP3 | 每折12来源/3072训练行，与冻结UB1 24来源/6144行基线相比 | 局部 fitted-state 的结果保留为对应折与目标的结论；增加同训练数据预算对照，区分状态目标与统计功效 |
| `audit_calibration_coordinate_exposure.py` | 已记录 Dense/Student 在 3,840 scope 上复用相同实际抽样行坐标 | 配对坐标降低方差，但用于训练后的评估不因 Horvitz–Thompson 权重自动获得独立验证身份；另设来源级留出 |
| UB1 运动补评 | 两个 source 的 Dense/UB1 连续光流比分别 1:0.381、1:0.0919；官方二值 Dynamic Degree 均0 | 官方二值与连续运动证据并列保存，专门检验动作指令/相机运动，不以 Motion Smoothness 推断运动保持 |
| R17 fresh official workers | 对每个 source×arm 的唯一视频用官方评价函数和 `video_list=[one path]` | 真实 **one-video-per-source 的自定义子集官方函数调用**，与每条件五次采样的标准 leaderboard 不是同一统计协议 |

证据：`tools/evaluate_vbench3.py`、`tools/evaluate_disjoint_vbench3.py`、`tools/run_fresh_official_metrics.py`、`config/calibration24_ffn_operands.json`、`tools/audit_calibration_coordinate_exposure.py`、`research/20261027/UB1_SOURCE_GENERALIZATION_AUDIT_20261027.md`、`research/20261027/UB1_DYNAMIC_DEGREE_RESULTS.md`。

## 2. 数据对象与分割：以来源为独立观察单位

每个来源先登记 `source_id`、实际 `image_id`、`prompt_id`、`dataset/revision`、`image_name`、`scene_stratum`（subject/background/camera/动作/静态等）、`conditioning_id`、`exposure_history`、`role`、真实 locator。跨来源分割优先按图像/原始视频簇隔离，再按prompt重用、相似画面/near-duplicate作人工或可重现检查。来自同一图像的不同seed、裁剪和不同prompt不能横跨确认集和训练/开发集。镜像/别名归为同一来源簇；独立seed是源内重复观测。

**用途版本**：
- `DISCOVERY`：少量有代表性source进行机制反例、源代码/数值定位，结论称局部有界现象。
- `CALIBRATION`：仅用于支持选择、修正拟合、observer统计/训练，记录真实 Student/Dense 分布和每位置数据量。
- `DEVELOPMENT`：用于模型/预算/超参/指标探索和比较，所有看过的结果均视为暴露。
- `FROZEN_CONFIRMATION`：在候选、指标、数据选择、聚合和门槛冻结后一次性评价来源级留出；使用后进入已暴露历史。
- `OFFICIAL_BENCHMARK`：采用该 benchmark 固定的 Image Suite、维度、采样次数和权重，完整报告覆盖与环境；独立于项目自定义诊断分数。

`source_id` 是经数据集版本绑定的语义标识，`image_id` 指向同一原始视觉内容，`conditioning_id` 指输入图像+prompt+裁剪/预处理。历史SHA可继续用于原始RUN内容完整性，本协议新增 manifest 不依赖摘要值推断独立性。确认集 eligibility 由**全部历史拟合/选择/评测暴露**决定；旧disjoint20已被分析，不再是新方法的首次确认集。

## CALIBRATION V2 方法升级

新实验优先以 [CALIBRATION_V2_METHOD_SNAPSHOT](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md) 和 [注册设计](../../config/calibration_design_v2_20261027.json) 描述校准对象、来源×行数配额、状态及停止条件。旧稿中的12×512/24×256/48×128是候选预算，只有真实可得操作数/新拟合/独立视频验证后才能从注册计划转为科研证据。组支持稳定性、联合signed作用、bias、RRR谱、Student轨迹和运动是不同估计对象，数据充分性独立判定。

## 3. 校准：独立考察来源广度、空间抽样和模型状态

真实 Q1→W1→GELU(tanh)→HiF4 Q2(64→8→4)→W2→e5→后续 flow 是校准语义。每条校准记录包含 `source_id, seed, step, expert, block, source_state, tensor_boundary, grid, sampled_row_ids, n_rows, population_rows, row_draw_id, inclusion_rule, dtype, quantization_revision, operand_locator`；使用与实际学习/部署一致的QDQ、gate、后缀。预先写明只采数值还是用于拟合。

**最小有辨识力的因素实验**（满足真实可得source数量后冻结，而非自动执行）：在 **6,144 行/位置**下比较 12×512、24×256、48×128，另比较 24×256 与 24×512（行数增加），同时固定实测QDQ路径、拟合容量、超参选择规则、seed/cohort 的抽样规则。来源数和类别覆盖分别报告；有效统计单位按source聚合，嵌套step×block×row为重复观测。若48个来源未实际取得，保持 `NOT_COLLECTED` 而非补重复行。

同时比较在相同数据/预算上的 `teacher_input`、`student_input` 原生同输入误差、以及全40层×4步真实free-rollout；对拟合状态分布变化采用同一批成对来源与独立的确认来源。抽样行复用是可登记的 **common random numbers** 设计，估计由学习状态决定的统计量时注明其依赖，抽样设计上的无偏性需与该依赖共同论证。

## 4. 视频生成、VBench 与统计口径

每个 source×seed×sample_index×variant 必须是可追溯的真实生成 `run_id`、唯一视频产物 `asset_id`，以及初始噪声绑定、scheduler、expert、步数、帧数、FPS、像素分辨率、随机数生成器设备/重置语义。Dense/候选配对保持图像、prompt、噪声、seed、scheduler及所有非干预参数完全一致；同一个seed的重运行记录为重复测量，不升级独立n。

**使用两种明确命名的评测模式**：
1. `CUSTOM_PAIRED`：指定来源各产生至少一个真正生成的视频，调用原版官方 metric 函数可称“官方函数·自定义子集评分”，标注实际维度、独立视频数、输入模式、缺项，适用于诊断和内部比较。
2. `OFFICIAL_I2V_5SAMPLE`：遵循所锁定 VBench-I2V 版本的 Image Suite/原始维度、**每条件五次真正独立采样**、相机子集和官方汇总脚本；每维报告期望/实际source、video和score覆盖，符合协议后才比较官方 leaderboard。

官方 VBench-I2V 的 `i2v_subject/i2v_background/camera_motion` 与 `subject_consistency/background_consistency/motion_smoothness/dynamic_degree/aesthetic_quality/imaging_quality` 分轴报告；`temporal_flickering` 等额外指标以独立自定义附加轴注册。裁剪/分辨率保持 Image Suite 的自适应原则，锁定实际图像预处理与视频采样规则。单维 `NOT_RUN` 是缺测，**不是数值0**。官方二值 Dynamic Degree 同时配一项事先定义的连续 motion magnitude、时间窗口、ROI/相机动作分层，附逐视频信息。

**聚合与不确定性**：先计算同一source/prompt多 `seed × sample_index` 真实重复观测的均值/分布，对同一 `image_id` 内source/prompt等权，再等权聚合独立image cluster；另报告每类image cluster和source的 n、效应与尾部。以来源为cluster进行bootstrap置信区间（少量来源时描述为主）；不同seed/step/block/row不能增大独立来源n。Paired delta同时给绝对差、百分点（百分制）、相对百分比 `100×(candidate/dense−1)`；分母为0时相对比记 `UNDEFINED_ZERO_BASELINE`。度量方向、mask/帧采样/权重、缺测处理和阈值事前冻结；多方法择优与最终确认相分离。真实性能同时包含完整FFN wall、end-to-end generation wall、显存、驻留/packing/校准代价与可信Dense baseline。

## 5. 资产、预检和Agent执行

新实验登记 `REUSABLE_ASSET_PLAN`：数据切片、原始/抽样状态、行坐标、模型配置、真实run/noise、逐视频指标、统计表、文件索引及历史暴露记录的 producer、shape/units、版本、locator、可复用范围和失效条件。预检时先使用 `python tools/ffnds_cohort_preflight.py --manifest <path>` 检查来源泄漏、视频伪重复、Dense配对、校准坐标及模式身份；它是**静态/文件级预检，不替代官方模型评分**；即使五个不同 run ID 的样本检查通过，也只证明一个条件的记录一致性，正式VBench-I2V榜单可比性仍须独立核对完整Image Suite、图像处理、九维覆盖、官方权重与汇总。结果中的 `official_comparability_verified=false` 明确标示该边界。确认集和 leaderboard 采样为与校准和本地模式发现不同的协议，不能由修改文件名来满足样本数。

**正向科研提示**：先说现有资料支持的科学发现，接着写可区分的来源/随机性/校准/指标问题；优先调用已有真实数据与成熟实现，选一个最低成本能改变解释的检验。回答固定包含：`观察对象 → 已确认实现 → 竞争解释 → 独立数据单位与分割 → 冻结的计量/判据 → 实际运行状态与资产 → 下一有效行动`。无证据的环节标 `OPEN` 并给出能使其成为证据的具体动作；已有负结果严格限定在被测试的对象和抽样范围。

## 5.1 真正参与运行的代码合同（本轮实现）

[执行手册](EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md)连接以下入口：
`ffnds_experiment_plans.py source-bank/calibration-plan/video-plan` → `ffnds_native_geometry.py`（实际Native grid）→ `ffnds_validate_calibration_operands.py`（真实BF16 payload边界）→ `ffnds_run_registered_video.py`（LightX2V Dense/UB1及原生噪声）→ `ffnds_stage_registered_i2v.py`（视频与噪声双重验证、原版VBench）→ `ffnds_cluster_metric_report.py`（seed/sample→source/prompt→image cluster分层统计）。
其中计划、校准数据捕获、拟合、原生生成与正式VBench分别形成可独立核验的状态。旧固定24×256拟合属于历史算法，变更cohort后重新进行算子/校准系数及独立确认版本注册。

**校准行随机性与身份**：`row_draw_id` 使用冻结的 `native_source_ordinal`、step、block和抽样seed，与实际行抽样随机数来源保持一致。当前cohort排序只用于迭代；复用前核对原生 `selected_row_ids`、`sampling_seed` 和操作数索引，以采样计划与真实捕获分别报告状态。

## 6. 已核对的一手成熟实现

- VBench++ I2V 官方数据/指标及5次采样：https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v
- VBench 官方 metric 分类与 custom-input 模式：https://github.com/Vchitect/VBench
- VBench Dynamic Degree 原函数：https://github.com/Vchitect/VBench/blob/master/vbench/dynamic_degree.py
- vLLM/LLM Compressor 校准数据选择与领域匹配：https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/
- Hugging Face Diffusers 随机生成器与随机状态复现：https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
- Wan2.2 作者代码与原生模型身份：https://github.com/Wan-Video/Wan2.2

参考实现的任务、模型、采样次数与硬件环境各自保留。成熟实现用于修正**抽样/定义/统计契约**，不将其它任务结果搬作 Wan D-Q 的已测质量。
