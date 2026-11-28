# 下一步实施交接

## 当前完成度

学校首轮已完成 12 条原生注意力背景轨迹、192 个配对面板、七臂拟合和全键见证。结果、实际执行调整和完整收据见[机制实验结果](results/README.md)，机器状态见[运行状态](RUN_STATUS.json)。

原预案的 FIN-C1 在 BF16 wide8192 背景胜过 OUT-ABS，在 Dense 背景退化，双背景成片门槛未通过。高噪声 OUT-ABS、低噪声 FIN-C0 是另行登记的探索性回放。以下分段保留冻结方案的实施顺序，在线学生、成片、计时和独立确认仍待新证据。

## 0. wide8192 运行前检查（本次修订）

以 `wide8192_binding.json` 和仓库 `research/CURRENT_HANDOFF.json` 为本轮执行身份。学校作业目录要包含 `generate_reference.py`、`exact_port.py`、`fourstep_hit.py`、`fourstep_anchor.py`、`fourstep_backend.py`、`generate.py` 等，且 `sha256sum -c MANIFEST.sha256` 通过。

- BF16 主臂：`python generate.py vbi_0392 --variant wide8192`，收据核对 `variant=wide8192`、`tile_rows=8192`、80 个锚点和 80 个命中。
- 显存对照：`python generate.py vbi_0392 --variant wide8192_all_int8_joined`，收据核对 joined 80、fallback 0，u0/y0/z0k INT8、128 通道组 BF16 scale、P64 原值 BF16。
- Dense 主臂：用 `render(..., method=None)` 明确走原生 FFN。不要把 `wide8192` 误解为保留 8192 FFN 通道。

正式注意力 hook 应使用每臂自身输入，在固定 BF16 cache 臂内完成 OUT/FIN 比较；INT8 为独立条件性内存与画质交互实验。

## 1. 在目标环境解析一次原生绑定

从现有学校工作区的实际 `base.json` 读取 `v2_reference_dir`、当前 CAL manifest、有效配置、模型/数据根和原生 runner。按 `repository_binding_audit.json` 中现有 SHA 校验身份。

旧 camera 原始捕获根从旧 `camera_cal_trial/capture_submission.json["stage"]` 解析；其预期后缀是 `camera_cal_capture_d4184a8a43e06873`。这个旧面板用于取坐标和 FFN 参照；新注意力 producer 使用本包已冻结的 panel index 即可继续，不依赖下载旧 FFN tensor 才能开始采集。

将实际 attention callable、backend/version、Q/K/V 边界、mask、context 分支、softmax_scale 和 LSE 约定写入 `runtime_binding.frozen.json`。LSE 从内核得到的 base/scale 必须和数学自然对数对象对应。解析 image 路径，验证六源图像和原始 prompt SHA，保存实际初始噪声文件与 tensor hash。

## 2. 实现一个共享采集 producer

沿当前原生 transformer hook，在指定四层和两个专家记录两个端点。Dense/wide8192 作为两个 FFN 背景，attention 维持 Dense 输出。代码将新增：

- 在有效 Q/K/V 边界绑定字段；读取逐头 O 与 LSE，或用保持全 keys 的独立 query-panel 精确计算获得标签。
- anchor pooled top-k 支持与 pair 内缓存；selected 输出与 FIN raw64 流式归约。
- P64/H448 分角色记录；八个 self raw witness 的四头完整 K/V；cross 条件去重。
- 当前 self gate、实际 W_O 张量绑定和 source/latent/context fingerprints。
- 有界 writer、每资产 SHA 和 capture index；模型返回输出维持本轨迹的定义。

以 CAL0032、layer13 high/low 作为第一条正式采集的接口检查。先观察实际 O/LSE、支持、轴和数值约定，再完成同一轨迹其余层以及其余五源。当前计划不先额外生成一组丢弃的校验视频。

## 3. 从固定资产完成机制分析

CAL3 生成四套共享变换、target 尺度与 λ；随后冻结。七个拟合臂和两个零拟合参考在完全相同的 capture index 上运行。输出 P64→H448 主结果、逐头纯输出 floor、all-H 估计器参考、观测谱，以及 C1 同 Hessian 的 zero-mass-RHS 诊断。

完整 K/V witness 计算实际 finite secant、一阶作用与余项。把结果按 source 聚合，与 W_O/gate 后的点/空间/时间误差共同解释。校准、参数选择、模型结果和人工视频观察各自保留来源身份。

## 4. 集成一个 NEW 和一个 REF

以固定宽度 endpoint feature 和 32×32 Schur 求解接入 self-attention 全 40 层。anchor 0/2 返回原生输出；hit 1/3 仅以当前 P64 拟合并精确覆盖其输出，保留自身 anchor 和输入。current projected feature 与 anchor 使用相同 BF16 量化。

先做两个 DEV 哨兵，再补第三个；复用身份完全相同的 Dense/FFNDS 视频。诊断路径可另采 H448，时间路径关闭 capture/shadow。完整周期的时间与峰值、视频人工判断完成后，冻结候选配方并调用 fresh-confirm 选择脚本。

## 5. 本包可直接运行的静态命令

```bash
python3 prepare_plan_assets.py
python3 compute_asset_budget.py
python3 validate_plan.py
python3 freeze_confirmation.py --help
```

这些命令只准备和验证静态资产，不执行模型。已交付的投影矩阵字节是当前版本的权威身份；若在不同 BLAS/NumPy 环境重新生成出现字节变化，将其登记为新资产版本。

确认集有真实输入后，使用：

```bash
python3 freeze_confirmation.py \
  --historical-bound VB_BOUND_INPUTS.json \
  --cohort24 COHORT.json \
  --candidates canonical_sources.json \
  --observed-clusters observed_clusters.json \
  --recipe confirmation_selection_recipe.json \
  --method-recipe candidate_recipe.frozen.json \
  --output confirmation_sources.frozen.json
```

canonical candidate 字段和分层规则见脚本帮助及配套 recipe。先锁定方法配方，再在候选运行结果产生前冻结六个新 source clusters。执行 GPU 任务时沿用学校当前真实队列与已分配资源绑定，不把历史任务的队列约束复制成新计划的事实。
