# FFNDS 数据/校准/真实视频/官方评价：可执行操作手册

> 本手册对应 `tools/ffnds_experiment_plans.py`、`ffnds_native_geometry.py`、`ffnds_run_registered_video.py`、`ffnds_stage_registered_i2v.py`、`ffnds_cluster_metric_report.py`。所有输出使用新目录；旧 CAL12/CAL24/UB1/R17/E0 原始RUN和视频维持原样。新脚本已接入原生 LightX2V 的验证资产、HiF4 D-Q 与 UB1 运行时；**GPU结果须按真实执行收据确认**。 GPU并行资源按[2026-10-28动态调度规则](DYNAMIC_GPU_RESOURCE_POLICY_20261028.md)采用当前真实可调用最大量，不设置预先2/4张的全局上限；单个worker的GPU需求与历史冻结scope仍分别记录。

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
