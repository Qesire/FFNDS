# 用完整视频指标更新折叠 FFN 系数

本包将已有 L64 线性修正的少量系数作为待优化参数，按固定 CAL 视频的分项指标提出并验收更新。包含矩阵导出、真实 native binding 适配、官方 VBench worker、指标导入、约束局部搜索和有限预算外部评估协议。

**当前交付是实现及非模型单元测试；真实 Wan 视频质量、CUDA 兼容性和系统收益仍待模型环境验证。** 数学动机、收益条件、反例和确认方案见[研究说明](../../research/20261117/G20_VIDEO_METRIC_PARAMETER_FEEDBACK_20261117.md)。本包不改变执行范围或既有实验状态。

## 1. 参数实际改变什么

对一个站点，采用行向量约定 `delta_x @ A`：

\[
\Delta A_L=(P_{64}/\gamma_L)C_LQ_{32}^{\mathsf T},\qquad
A(\theta_g)=Q_{\mathrm{BF16}}(A_{\mathrm{q1}}+\theta_g\Delta A_L).
\]

`P64` 是 `[5120,64]`，`gamma_L` 是 `[64]`，`C_L` 是 `[64,32]`，`Q32` 是 `[5120,32]`；除法沿 64 维广播。L64 表示输入特征宽度，修正矩阵秩至多为 32。导出不转置矩阵。

示例用两个参数：`high` 控制 high/4、high/19、high/34；`low` 控制 low/4、low/19、low/34。其余 74 个站点沿用传入的完整 base manifest。可配置六个站点独立系数，但首轮最坏预算会由 6 组变成 14 组视频。

- `theta=0` 精确使用参数 kit 中保存的 `A_Q1_BF16`。
- `theta=1` 精确使用 kit 中保存的 `A_L_BF16`。
- 其他值在 CPU FP64 重建后，按 FP32 → BF16 舍入；端点重建与保存位模式的差异另行记录，不宣称与旧 GPU 中间计算逐位相同。
- kit 的 CURRENT_Q1 不是历史 2026-11-07 部署 Q1r2。因此这个六站点混合模型的 `theta=0` 不能标成历史 Q1r2；`theta=1` 也不是全 80 站点 L64。

矩阵输出仍是原消费者的 dense raw-BF16 `.npy`，manifest 顶层只有完整的 `high/0…39`、`low/0…39` 80 个站点。导出元数据另存。部署时不增加一条修正支路，主 GEMM 形状与缓存容量保持原有拓扑；实际 latency/VRAM 需要原生测量。

## 2. 文件与入口

| 文件 | 作用 |
| --- | --- |
| `matrices.py` | 读取 `.pt` kit，折叠系数，校验完整矩阵清单、payload SHA、端点位模式；NumPy `.npz` 仅供小型测试 |
| `native.py` | 对已构建并验证的 LightX2V pipeline 执行完整固定 cohort；调用原 `NativeBinding`、`ResidentMatrices` 和视频摘要函数 |
| `vbench_worker.py` | 在评分环境调用固定 checkout 的真实 `compute_*`；不近似重写官方模型 |
| `metrics.py` | 导入分项分数和本地诊断，统一 MUSIQ 尺度，按 sample → source → image-cluster 聚合 |
| `optimizer.py` | 坐标有限差分、对角正曲率模型、约束信赖域投影和实际候选验收 |
| `workflow.py` | 有限预算 argv oracle、固定初始参照、内容寻址缓存、初始噪声配对和失败次数记账 |
| `__main__.py` | `check`、`export`、`run`、`import-vbench`、`assemble`、`score` |

仅配置检查和单元测试需要 Python + NumPy。读取生产 `.pt` kit 另需 PyTorch；生成需完整 FFNDS checkout、原 CUDA/LightX2V 环境和模型资产；官方评分需 VBench 依赖及权重。生成与评分可以放在不同 Python 环境，外部 oracle 用显式 argv 调度。

```bash
python -m unittest discover -s tools/video_metric_feedback/tests -v
python -m tools.video_metric_feedback check --config tools/video_metric_feedback/examples/config.example.json
```

第二条只验证配置结构并显示预算，不打开模型或 payload。[示例配置](examples/config.example.json)与[cohort 模板](examples/cohort.template.json)含明确占位路径和假设阈值；不能作为已绑定的真实实验范围。

## 3. 优化和约束

每个目标指定 `name`、`direction`、`weight`、`scale`。`max` 表示越大越好，`min` 表示越小越好。`scale` 是事先确定的有意义变化尺度，不从本轮候选结果估计。优化器最小化相对初始视频的加权、定向、尺度化损失。

每个 guard 可指定 `max_drop`、`min_value`、`max_value`，以及 `level: cluster` 或 `aggregate`。`max_drop` 使用该指标的原始归一化单位：对 `max` 指标允许相对初始分数下降这么多；对 `min` 指标允许上升这么多。**参照始终是初始参数生成的固定视频，不随接受的候选移动。** `cluster` 对每个独立 image-cluster 分别检查。

默认一轮、两个参数，最多 `1 + 2×2 + 1 = 6` 次完整 cohort 评估：初始中心、四个正负探针、一个候选。每组 N 段视频，则最多 6N 段候选生成；已有 Dense 参照视频和官方评测计算另计。后续完整轮最坏再需 `2p+1` 组。边界使用实际步长，单侧可行时用单侧差分。

一次更新需要：实际目标改善超过 `min_improvement`，真实 guard 全部通过，包含先验项的实际下降为正，与局部模型预测的下降比达到 `acceptance_ratio`，并且实际 BF16 矩阵内容发生变化。线性化约束只用于提出候选，不替代真实候选检查。

`optimization_trace.jsonl` 保留梯度、曲率、预测/实际下降、逐 cluster 组合损失改善、量化平台和约束情况。`optimization_result.json` 保存选中的系数、候选目录和状态。`no_proposal` 或 `rejected` 都是有效结果，不能解释为已得到更好的视频。

## 4. 指标选择

建议先用 I2V 主体/背景与成像质量形成目标，把 Dense 配对的连续光流误差和相邻帧变化误差作为约束。示例权重、尺度和容忍量只展示字段含义，必须在确认视频结果前固定。

| 指标 | 本包含义及用法 |
| --- | --- |
| `i2v_subject`、`i2v_background` | 官方分数；包含视频帧间项，仍需配合运动检查 |
| `imaging_quality` | 官方逐视频 MUSIQ 原始值除以 100；`raw_metrics` 保留原值 |
| `subject_consistency`、`motion_smoothness` | 可报告；仅提高这些分数不能证明运动质量改善 |
| `dynamic_degree` | 官方逐视频结果为 bool；保留 0/1 语义，不把它当连续光流量 |
| `camera_motion` | 官方离散行为结果；cohort 必须显式给出官方 `camera_instruction`，不从 prompt 猜标签 |
| `flow_l2` | 既有本地诊断的 Dense 配对 Farneback 流场差异，单位为其降采样像素；越小越好 |
| `temporal_delta_rmse` | 候选与 Dense 的相邻 RGB 帧增量差异，RGB 已归一化；越小越好 |
| `flow_magnitude_ratio` | 候选/Dense 连续运动幅值比；Dense 近静态时分母不稳定，不能对所有来源统一解释 |

本地诊断来源是 [`tools/ffnds_video_metrics.py`](../ffnds_video_metrics.py)，与官方 VBench 维度分别标记。`eligible_metrics` 在 cohort 中冻结；一个来源不适用的维度不能临时补分，也不能在候选失败后删除该来源。多个 seed 或 prompt 不增加独立图像数。

## 5. 接入已注册的生成 producer

`workflow.ExternalOracle` 导出候选后，将 `request.json` 路径和预期 `result.json` 路径替换进 `oracle.command`。命令是 argv 数组，`shell=False`，要求进程成功退出并生成完整结果；不会自动重试或追加候选。

request 包含参数、矩阵清单、矩阵内容指纹、评估键、CAL cohort、指标定义和 `oracle_identity`。调用方在已有 producer 中构建并验证 pipeline 后调用：

```python
from pathlib import Path
from tools.video_metric_feedback.native import generate_batch
from tools.video_metric_feedback.workflow import read_json

request = read_json(request_path)
videos_manifest = generate_batch(pipe, request, Path(output_dir) / "native")
```

`pipe` 由调用方现有的注册 producer 提供；其 checkpoint、LoRA、分配、完整 recipe 和 scope 校验不由本包重建。历史 `runner_panel.py` 固定了旧 14 source-seed 范围，不能给它追加本来不存在的 `--matrix-manifest` 参数来运行新候选。

native 适配限定于当前四步 HHLL、两 expert×40 层、宽 5120、默认 FFN 输入 `[31668,5120]`、832×464、81 帧、16 fps、CFG 关闭的路径。每候选一组 ResidentMatrices，每 source/seed/sample 一组全新的 NativeBinding/anchor slab；检查 160 个站点、80 次 anchor、80 次 hit、80 次 native FFN、160 次 prediction。每个样本捕获实际初始 latent SHA；仅相同 seed 不算已证明噪声配对。失败时恢复三个 patched class 方法并关闭矩阵。

生成 manifest 标记 `VIDEOS_GENERATED_METRICS_PENDING`，包含视频文件、初始噪声、计数和运行源码身份。`generation_wall_seconds` 是带仪器的生成墙钟，不是 FFN 边界延迟或公平速度结果。

## 6. 评分与组装 result

在固定的评分环境运行：

```bash
python -m tools.video_metric_feedback.vbench_worker \
  --request /path/to/request.json \
  --videos-manifest /path/to/native/videos_manifest.json \
  --result /path/to/official_scores.json \
  --vbench-repo /path/to/VBench \
  --revision fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490 \
  --device cuda:0 --resolution 7-4 --local
```

worker 验证完整生成 manifest、矩阵清单 SHA、VBench HEAD 和 tracked 文件未改；每个被请求的官方维度只评分适用样本。I2V 输入图走官方 `custom_image_path`；camera 通过携带显式标签的 symlink 文件名评分并恢复原视频路径。它保存官方原始结果、源码 SHA 和调用清单。

固定 VBench SHA 不会自动固定其 `torch.hub` 外部仓库、权重或整个数值环境。调用方必须在 `oracle.identity` 中绑定环境及缓存资产，且不能在一次搜索中更换。`--local` 是官方本地缓存选项，缓存缺失时官方函数仍可能下载。

若需要本地运动约束，使用既有诊断工具逐视频对照固定 Dense：

```bash
python tools/ffnds_video_metrics.py evaluate \
  --candidate /path/to/candidate.mp4 --reference /path/to/paired_dense.mp4 \
  --identity-json /path/to/actual_runtime_identity.json \
  --out /path/to/local_report.json
```

这里的 identity 与 cost 必须来自真实运行收据，包含该工具所需的 prompt/model/protocol/run 身份、FFN 边界时间、setup 与峰值显存等字段。native helper 的生成墙钟不能替代缺失的这些成本，不应填 0 伪造完整收据。

最后写入 ExternalOracle 等待的 result 路径：

```bash
python -m tools.video_metric_feedback assemble \
  --request /path/to/request.json \
  --videos-manifest /path/to/native/videos_manifest.json \
  --official-bundle /path/to/official_scores.json \
  --local-report /path/to/source1_report.json /path/to/source2_report.json \
  --output /path/to/result.json
```

也支持仅本地诊断的组装，省略 `--official-bundle`；若未使用本地指标则省略 `--local-report`。缺失任一所需分数会失败。组装和缓存均保留 synthetic 标记，测试注入结果不能进入 video 模式。

## 7. 启动搜索和复用

将已验证的 cohort、80 站点 base manifest、六个参数 kit、输入图/Dense 视频路径及 producer argv 填入实际配置后：

```bash
python -m tools.video_metric_feedback check --config /path/to/verified_config.json
python -m tools.video_metric_feedback export --config /path/to/verified_config.json \
  --theta '{"high":1.0,"low":1.0}' --output /path/to/export_preview
python -m tools.video_metric_feedback run --config /path/to/verified_config.json \
  --output /path/to/fixed_cal_feedback
```

`export` 是 CPU 参数操作，不生成视频；完整 `run` 通过配置的 producer 进行模型调用，只在对应已注册的有限范围内使用。现有 G20 cohort 元数据中的未绑定 payload/image-cluster 字段需要依据真实资产填齐，本包不猜测它们。

缓存键包含完整配置、输入 payload SHA 和实际矩阵内容指纹。不同系数若舍入为相同 BF16 矩阵，可复用同一评分；provenance 指纹另记系数与参数 kit。改 cohort/recipe/identity 不能复用原输出目录。每个外部启动（包括失败）写入 `oracle_attempts.jsonl`，顺序重启仍受原预算限制。一次搜索目录只允许一个写入进程，不支持多个并发搜索共享目录。

选中系数只表示 CAL 上通过该次验收。冻结它们以后，再在未用于选择的 image-cluster 上比较初始参数、更新参数与 Dense，报告全部指标、逐 cluster 差异、连续运动、人审和同路径 SYS 成本；不能在看到确认结果后继续更新并仍把它称作独立确认。
