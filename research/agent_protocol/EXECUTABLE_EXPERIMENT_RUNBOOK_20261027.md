# FFNDS 数据/校准/真实视频/官方评价：可执行操作手册

> 本手册对应 `tools/ffnds_experiment_plans.py`、`ffnds_native_geometry.py`、`ffnds_run_registered_video.py`、`ffnds_stage_registered_i2v.py`、`ffnds_cluster_metric_report.py`。所有输出使用新目录；旧 CAL12/CAL24/UB1/R17/E0 原始RUN和视频维持原样。新脚本已接入原生 LightX2V 的验证资产、HiF4 D-Q 与 UB1 运行时；**GPU结果须按真实执行收据确认**。 GPU并行资源按[2026-10-28动态调度规则](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)采用当前真实可调用最大量，不设置预先2/4张的全局上限；单个worker的GPU需求与历史冻结scope仍分别记录。

## 2026-10-29 · S2 时间统计结构与新数据采集的可执行接口（非第五研究路线）

已经完成的真实资产：[十新图全21τ](../20261029/RAW_BF16_FULL21TAU_NEW10_OBSERVATIONS_20261029.md)、[相同Dense h的W2因子重放](../20261029/RAW_BF16_FULL21TAU_NEW10_FACTOR_REPLAY_20261029.md)，均有真实RUN/EXIT；下一步不是重复它们，而是补齐[六族时序统计与新增真实空间邻域](../20261029/S2_TEMPORAL_STATISTICAL_STRUCTURE_AND_DATA_EXPANSION_20261029.md)。

新代码：`tools/raw_bf16_temporal_statistics.py` 是不可访问GPU/模型的纯数学模块，`tests/test_raw_bf16_temporal_statistics.py` 是合成对照；`tools/e0_s2_temporal_statistics_cpu.py` 在运行前核验原父protocol、源码hash、1600原生包的已完成两part和全部源身份，并**要求ACTIVE中冻结的单独CPU运行scope**；目前只是准备的代码，未运行新T1–T4统计。建议协调者在实际worker环境按顺序执行：

```bash
# 1. 静态语法与合成测试（不会读取既有大张量）
python -m py_compile tools/raw_bf16_temporal_statistics.py tools/e0_s2_temporal_statistics_cpu.py
python -m unittest discover -s tests -p test_raw_bf16_temporal_statistics.py

# 2. 实际保存来源/数据身份的meta-only draft（需在有原始HF assets的环境）
python tools/e0_s2_temporal_statistics_cpu.py draft \
  --hf-root /home/xiang/.cache/huggingface \
  --out /tmp/e0_s2_temporal_stats_draft.json

# 3. 独立审查代码/source/预算后，按ACTIVE登记新的有限CPU scope；
#    把协议status定为FROZEN_REGISTERED并计算完整SHA，绝不改旧scope。
# 4. 只有注册且哈希一致后，受控worker执行，真实产生RUN/EXIT（不得本地假造）
CUDA_VISIBLE_DEVICES='' python tools/e0_s2_temporal_statistics_cpu.py run \
  --hf-root /home/xiang/.cache/huggingface \
  --protocol config/<REGISTERED_S2_TEMPORAL_STATS_PROTOCOL>.json
```

读数：`POSITIONS.jsonl` 保存十来源每个step×block七类tensor在全21及去τ0后的lag、频谱、时间Gram和事件；`SUMMARY.json` 按image-cluster等权再按step/position汇总。必须在实际EXIT和独立核查后才写实验结论。单次输入按真实原始包18.49GB计量，CPU-only；建议worker和外部留足预算、内存和输出空间，优先检查实际包版本、raw SHA、原BF16写回逐字节中性。运行脚本中的T1–T4**仅覆盖固定8xy轨迹**，不能凭它验证空间运动对应。T5/T6完整实现还需新空间邻域、学生前缀与分层source原生producer，须另冻scope。

**GPU**：当前实际BF16 VIDEO20等scope规定 `session_max_GPUs=2`，任何同会话GPU相关后续执行严格≤2并发；此T1–T4 CPU统计不申请GPU。通用动态调度不得覆盖真实记录。

## 2026-10-29 · 开放式新采集的可执行入口（当前优先于旧固定来源例子）

本手册以前的 CAL24、旧source-bank和UB1 runner命令是**历史兼容示例**。现行新采集按[科学问题与producer协议](EVIDENCE_DRIVEN_ACQUISITION_20261029.md)、[新增需求队列](ACQUISITION_QUEUE_20261029.json)执行，不限制在旧 CAL24 / consumer20 或五个Block。特别是S2的真实视频latent时间—空间配对、S3原生动态Q1/Q2、S4学生prefix新增tensor必须真正扩展observer，不能把旧随机行/已输出的fake-QDQ重命名为新观测。

**A. 新来源库可用性**。用实际可读、带曝光记录的原始输入清单构造/扩展新source-bank。历史基础脚本只接受符合所约定结构的 legacy canonical 来源；新数据集需先实现输入适配器，记录真实 dataset/revision、image簇、prompt/裁剪、各类别及input locator。以下只建计划，不运行模型：

~~~bash
python tools/ffnds_experiment_plans.py source-bank \
  --legacy CALIBRATION=/path/to/actual/new_calibration_canonical.json \
  --legacy DEVELOPMENT=/path/to/actual/new_development_canonical.json \
  --out /path/to/experiment/new_source_bank.json
python tools/ffnds_calibration_design.py coverage \
  --design config/calibration_design_v2_20261027.json \
  --bank /path/to/experiment/new_source_bank.json \
  --out /path/to/experiment/coverage.json
~~~

如果数据适配器或某类source不足，首先采集或接入新的真实图像来源并验证独立image簇，再重建source-bank，不使用重复裁剪或seed补数。校准拟合源和未触碰确认源需要隔离；不同来源规模12/24/48是可调整的研究对照。

**B. 新机制与原生观测**。明确必须在FFN哪个真实生产时点采数据：pre-W1输入、GELU h、Q1动态A元信息、post-W1 Q2元信息、gate、student自己输入、video latent τ与空间(y,x)配对、native K/M tile等。对已经存在的native geometry使用以下计划，**只有真正运行原生observer并通过payload验证才获得新数据**：

~~~bash
python tools/ffnds_experiment_plans.py calibration-plan \
  --bank /path/to/experiment/new_source_bank.json \
  --geometry /path/to/experiment/native_geometry.json \
  --strata i2v_subject:12,i2v_background:12 \
  --rows 256 --source-state teacher_input --seed 42 \
  --out /path/to/experiment/new_calibration_plan.json
python tools/ffnds_validate_calibration_operands.py \
  --calibration-plan /path/to/experiment/new_calibration_plan.json \
  --operands-index /path/to/experiment/actual_native_operands_index.json \
  --verify-payload --out /path/to/experiment/operands_verified.json
~~~

其中 native_geometry 和 actual_native_operands_index 必须由**这些新来源的真实采集**产生，不能用旧已暴露来源的几何或tensor模拟；如果科学目标需要21时间槽的tubelet而不是均匀随机row，先新增专门的捕获计划/observer，再单独验证视频坐标一致性；现有 calibration-plan 不是tubelet采集器。上述12/12类配额仅为示例，真实配额由本次科学分层决定。

**C. 新完整视频**。当新BF16 shared80/step160、原生NVFP4或其他方法进入视频级检验时，先确认该backend的实际candidate runner已实现并具有中性路径/160 FFN执行覆盖。老 UB1 runner**不会自动支持**新variant。可用通用现存工具冻结视频来源与种子：

~~~bash
python tools/ffnds_experiment_plans.py video-plan \
  --bank /path/to/experiment/new_source_bank.json \
  --role DEVELOPMENT --strata i2v_subject:10,i2v_background:7,camera_motion:3 \
  --samples-per-source 1 --variants dense shared80_r960 step160_r960 \
  --experiment RAWBF16_FFND_EXPAND \
  --out /path/to/experiment/new_video_plan.json
python tools/ffnds_cohort_preflight.py \
  --manifest /path/to/experiment/new_video_plan.json
~~~

新数据集能否满足以上实际strata，以及video-plan接受的具体stratum名称，须先由source-bank和脚本检验。计划生成的variant**不表示**run_registered_video.py已支持该真实算子；完成相应本地runner/核对、冻结ACTIVE中的checkpoint与source/预算，再提交实际完整视频。用同checkpoint、同seed/noise和数值recipe做Dense配对，并保存全部motion/semantic/temporal/visual指标、latency、真实RUN/EXIT。可根据新出现的source尾部、相机缺失、动作事件和核数量不确定性继续注册下一批，而不是以第一次20来源为科研上限。

**D. 实际采集收据**：每个采集需求从 NEEDS_PRODUCER → REAL_COHORT_READY → SCOPE_REGISTERED → NATIVE_CAPTURED → VALIDATED_PAYLOAD → SOURCE_HELD_ANALYZED → NEXT_DECISION 更新。任务书不越过中间状态宣称完成，失败保留原始receipt并根据竞争机制修改下一版本。每批有限预算的注册和真实GPU选择继续遵循ACTIVE与动态资源策略。


## 0. 一次性环境检查

```bash
python tests/test_ffnds_cohort_preflight.py
python tests/test_ffnds_experiment_plans.py
python tests/test_ffnds_calibration_design.py
python tests/test_ffnds_native_geometry.py
python tests/test_ffnds_stage_registered_i2v.py
python tests/test_ffnds_cluster_metric_report.py
python -m py_compile tools/ffnds_calibration_design.py tools/ffnds_experiment_plans.py tools/ffnds_native_geometry.py tools/ffnds_run_registered_video.py tools/ffnds_stage_registered_i2v.py tools/ffnds_cluster_metric_report.py
```

上述静态与CPU测试包含图像簇优先抽样、多prompt配额、重复来源诊断及类型错误的回归控制，仅覆盖协议数学/记录解析；其中聚合测试涵盖同图像多prompt不等重复次数，采样测试涵盖实际抽样与注册身份一致。模型加载、原生HiF4及官方评测须在实际集群环境另作fresh run。

## 校准方法V2的直接命令

方法：[METHOD_SNAPSHOT](../20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md)，设计：[calibration_design_v2_20261027.json](../../config/calibration_design_v2_20261027.json)。

```bash
python tools/ffnds_calibration_design.py coverage --design config/calibration_design_v2_20261027.json --bank /path/to/new_experiment/source_bank.json --out /path/to/new_experiment/CALIBRATION_COVERAGE.json
python tools/ffnds_experiment_plans.py calibration-plan --bank /path/to/new_experiment/source_bank.json --geometry /path/to/new_experiment/native_geometry.json --strata i2v_subject:12,i2v_background:12 --rows 256 --source-state teacher_input --out /path/to/new_experiment/C24_teacher.json
python tools/ffnds_experiment_plans.py calibration-plan --bank /path/to/new_experiment/source_bank.json --geometry /path/to/new_experiment/native_geometry.json --strata i2v_subject:12,i2v_background:12 --rows 256 --source-state student_input --out /path/to/new_experiment/C24_student.json
# actual native per-source group statistics exist only after native operand capture/derivation:
python tools/ffnds_calibration_design.py diagnose --statistics /path/to/actual_source_group_statistics.json --keep-groups-by-step 54,108,108,162 --bootstrap 500 --out /path/to/new_experiment/SUPPORT_DIAGNOSTIC.json
```

`coverage` 按独立image cluster判断每个stratum的候选配额；`calibration-plan` 首先抽取image cluster，再为每簇登记一条明确prompt/source，保持同一冻结bank和seed内12/24/48来源嵌套；新增图像或重分类后建立新cohort，不以旧清单的前缀性质推断跨版本嵌套。正式Image Suite的image-prompt条件选取独立于校准簇配额。

`coverage` 只证明 source bank 是否具备预设分层配额；实际native操作数和 Student 状态必须由真正捕获并通过校准索引验证。诊断 `ffnds.calibration.source_statistics.v2` 的 `group_energy[216]` 是从原生 gated group contributions 导出、以source/image聚合的非负量；测到的统计仅构成SOURCE_LEVEL_DIAGNOSTICS，不会写出拟合/官方视频已成功的结论。

## 1. 真实来源库与历史暴露

```bash
python tools/ffnds_experiment_plans.py source-bank \
  --legacy CALIBRATION=.runtime/inputs/calibration24/canonical.json \
  --legacy DEVELOPMENT=.runtime/inputs/formal20/canonical.json \
  --legacy DEVELOPMENT=.runtime/inputs/disjoint20/canonical.json \
  --out /path/to/new_experiment/source_bank.json
```

先补齐已分析的E0和R17来源曝光。如果部分来源重叠，脚本据原始source/图片文件与历史冻结图片指纹检出跨role重叠并要求在source bank中整合归属。已有评价结果用于方法选择的source保持DEVELOPMENT；真正新来源另以FROZEN_CONFIRMATION加入并保留清晰数据权属。人审相似/裁剪来源的image_id后再用于确认。

## 2. 校准来源数 × token行密度的正交设计

```bash
python tools/ffnds_native_geometry.py \
  --bank /path/to/new_experiment/source_bank.json \
  --collections runs/worker_restart_20261020/calibration24_ffn_operands/shard0/RUN.json runs/worker_restart_20261020/calibration24_ffn_operands/shard1/RUN.json \
  --out /path/to/new_experiment/native_geometry.json

python tools/ffnds_experiment_plans.py calibration-plan \
  --bank /path/to/new_experiment/source_bank.json \
  --geometry /path/to/new_experiment/native_geometry.json \
  --strata i2v_subject:6,i2v_background:6 --rows 512 --seed 42 \
  --out /path/to/new_experiment/cal12_rows512.json
python tools/ffnds_experiment_plans.py calibration-plan \
  --bank /path/to/new_experiment/source_bank.json \
  --geometry /path/to/new_experiment/native_geometry.json \
  --strata i2v_subject:12,i2v_background:12 --rows 256 --seed 42 \
  --out /path/to/new_experiment/cal24_rows256.json
```

48×128只在已采集**48个独立来源及全部160个真实native grid**后运行。相同来源的行坐标由冻结的 `native_source_ordinal`、step/block和固定seed确定，128/256/512设计具有前缀嵌套关系。每个source使用冻结metadata中的原始 `native_source_ordinal`，即使增删其它source或改动校准cohort也保持相同抽样随机过程。输入 `calibration.json` 为校准**采样计划**，新来源的 V1/V2/Y 操作数需经实际native hook按 `selected_row_ids` 采集，并用原Q1/GELU/HiF4 Q2/W2/gate数值契约验证。历史 `fit_step_budget.py` 固定24×256，继续承担旧UB1复现；新设计拟合时必须提供显式source与行索引接口，不能直接把该计划冒充已完成的模型重拟合。

## 2.1 原生校准操作数核对（新拟合的实际入口）

从对应的原生observer生成 `operands_index.json`，每条绑定 `source_id,step,block,selected_row_ids,n_rows,population_rows,source_state,expert,tensor_boundary,operand_path`。对应的 `tensor_boundary` 为 `original_native_Q1_V1_Q2_V2_W2_Y_b2_gate`。

```bash
python tools/ffnds_validate_calibration_operands.py \
  --calibration-plan /path/to/new_experiment/cal24_rows256.json \
  --operands-index /path/to/new_experiment/operands_index.json \
  --verify-payload --out /path/to/new_experiment/operands_checked.json
```

**拟合准入**：完整160位置、每来源真实选择行坐标、BF16 `V1/V2/Y/gate/b2` header与模型QDQ边界均通过后，才将该cohort作为新校准拟合输入。对没有新采集的12×512/48×128明确标 `AWAITING_NATIVE_CAPTURE`，不能从旧256抽样更名冒充。新的cohort-aware support/gain/R256拟合与部署仍需实现、数值验证并注册独立版本，旧UB1不随数据计划自动变更。

## 3. 冻结真正独立的 Dense/UB1 视频任务

```bash
python tools/ffnds_experiment_plans.py video-plan \
  --bank /path/to/new_experiment/source_bank.json \
  --role DEVELOPMENT --strata i2v_subject:10,i2v_background:10 \
  --samples-per-source 1 --variants dense UB1 --experiment ED1 \
  --out /path/to/new_experiment/video_plan.json
python tools/ffnds_cohort_preflight.py --manifest /path/to/new_experiment/video_plan.json
```

正式VBench-I2V五采样时使用 `--samples-per-source 5 --role OFFICIAL_BENCHMARK`，并采用官方完整Image Suite/原始metadata/对应指标分组。**五条生成任务每条使用不同seed，Dense/UB1对应sample index保持相同的seed和初始噪声**。仅生成plan不会出现真实RUN、MP4或质量成绩。

在具备已验证LightX2V/HiF4/模型资产、单worker GPU及UB1冻结policy的环境：

```bash
python tools/ffnds_run_registered_video.py --plan /path/to/new_experiment/video_plan.json \
  --job-index 0 --out-root /path/to/new_experiment/native_runs \
  --ub1-policy runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json --dry-run
# 在冻结资源预算与ACTIVE范围后，逐项运行实际任务；去掉--dry-run才启动真实推理。
```

真实 GPU 运行另传 `--registration /path/to/frozen_gpu_scope.json`。登记为 `ffnds.native.scope.v1`，含 `status=REGISTERED_BOUNDED_GPU_EXECUTION`、`experiment=ED1`、从冻结计划提取的 `allowed_run_ids`、`max_visible_GPUs=1`、`user_authorization=USER_RESUME_OFFLINE_IFC_20261027`。入口与 ACTIVE 现行科研权限核对。注册文件由实际预算/任务/资源的事前冻结生成。

每条job保存 `STARTED.json`、`RUN.json`、实际`video.mp4`、`initial_noise.safetensors`；同时记录4步trace和原生160个FFN位置覆盖。执行失败保留 `FAILED.json`，重试需新attempt目录/版本。

## 4. 严格来源+原生噪声配对，再启动官方评价

```bash
python tools/ffnds_stage_registered_i2v.py stage \
  --plan /path/to/new_experiment/video_plan.json \
  --run-root /path/to/new_experiment/native_runs \
  --official-full-info /path/to/pinned/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --out /path/to/new_experiment/vbench_stage
python tools/ffnds_stage_registered_i2v.py evaluate \
  --stage-root /path/to/new_experiment/vbench_stage \
  --vbench-checkout /path/to/pinned/VBench \
  --dimensions imaging_quality aesthetic_quality motion_smoothness dynamic_degree
# 检查打印的固定命令，再带 --execute 在有权重和依赖的GPU环境运行
```

`stage` 逐job读取实际RUN及视频与初始噪声文件，使用**全文件字节比较**验证Dense/UB1真实初始噪声相同，并检查同方法五个sample互不复制；逐dimension准备原版VBench的custom-input视频与图像。完整官方数据+五独立视频才进入 `vbench_standard`，局部subset使用 `custom_input`。官方 `evaluate_i2v.py` 的 `--output_path` 独立指定，单维输出不同目录。

报告须另检查真实VBench结果与镜像/权重版本：`EVALUATION.json`只表示官方调用成功，不冒称9维聚合通过；终端视频质量仍结合 `ffnds_video_metrics.py` 全帧、连续光流、动态程度、主体/背景及实际系统成本。

## 5. 来源级统计（现在可直接用于历史20）

```bash
python tools/ffnds_cluster_metric_report.py \
  --pairs reports/worker_20261020/disjoint20_UB1_paired_sources.csv \
  --bootstrap 2000 --seed 42 --out /path/to/new_experiment/ub1_disjoint_cluster.json
```

支持源级 `source_group,dimension,D_Q,candidate` 与新评价的 `source_id,image_id,seed,metric,dense,candidate`；以 `seed × sample_index` 联合识别source/prompt的真实重复观测；先按source/prompt均值，再对同一 `image_id` 内多个source/prompt等权，最后对image cluster等权并按cluster bootstrap。输出相对变化百分比、逐来源尾部、95%区间和真实独立样本数量。对于未运行official维度，维持 `NOT_RUN`。

## 6. 代理人执行规范

执行顺序为**还原真实原生source和数据结构 → 明确发现/校准/选择/独立确认的用途 → 从现有真实OPERANDS建立试验几何 → 生成并检查冻结计划 → 逐项真实推理与落盘 → 实际独立质量评价 → 来源层级推断 → 最有判别力的下一科学问题**。先把有实际证据的发现说清，再给同预算竞争解释与所需最小新增真实数据。新运行前读取 `research/ACTIVE_RESEARCH.json` 并注册可用资源、计划、资产，保持旧结果与失败的历史身份。

成熟实现来源：
- https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md
- https://raw.githubusercontent.com/Vchitect/VBench/master/evaluate_i2v.py
- https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
