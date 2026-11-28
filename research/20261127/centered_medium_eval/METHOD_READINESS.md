# 完整方法的冻结与入口核查

核查对象为研究分支 `4ca6b1b14b90965269a7b480f4fd5bfc75f64f37`。本轮只完成源码、既有结果和入口的静态核查，未提交学校 GPU 作业。

## 决策

**现有机制已经形成覆盖四步、40 层、注意力与 FFN 联合路径的成型候选，适合冻结并进入中等规模的质量和效率评测。** 当前合理目标是确认其总体收益、质量边界和来源之间的波动。已有对照足以支持停止继续增加自由度；这里的“框架内足够优化”指经过当前有辨识力的候选比较后，保留了信息来源清楚、运行路径完整、成本可核实的最简有效实现。

剩余工程工作是将已验证路径装入统一评测入口并核对绑定。这项工作不要求再设计参数搜索或机制实验。本文附带的薄封装已完成 Python 语法和配置一致性检查，学校运行状态尚未核验。

## 要冻结的完整对象

| 位置 | 注意力 | FFN | 作用 |
|---|---|---|---|
| high step 0 | 原生完整注意力，建立路由 | 原生计算并写 BF16 锚点槽 | 保留完整现场信息 |
| high step 1 | 固定路由 selected attention + OUT-ABS + 原 P64 | wide8192 命中 | 当前观测恢复 |
| low step 2 | 原生完整注意力，建立路由、质量和 centered 历史槽 | 原生计算并写 BF16 锚点槽 | 为 low 周期重新锚定 |
| low step 3 | centered 对角运输 + OUT-D + 原 P64 | wide8192 命中 | 运输历史并由当前观测纠偏 |

固定项：Wan2.2-I2V-A14B 原四步蒸馏调度；BF16 `wide8192`；40 层；输入/输出几何 832×464、81 帧、16 fps；seed 42；attention token 数 31,668，40 heads，每 head 128 维；query block 128、selected top 31 blocks；原 64 个 observer 位置；原 CAL、基及 λ=0.1。原生锚点为 step 0/2，命中为 step 1/3。cross attention 保持原生。

low 的底座来自已经运行过的 `MassCenteredLow`：在锚点，将遗漏贡献去除 value 均值并按通道尺度标准化，存为 BF16；在命中时使用当前 value 的均值和尺度解码，再由同一 P64 读出修正。继承原 `rho` 的数值处理与必要回退，不增加 rho 截断；尺度阈值为 `1e-12`。最终 P64 行由当前完整键注意力结果覆盖。

每个完整候选片应记录 attention 160 sites、80 anchors、80 hits、5,120 native probe queries；FFN 80 anchors、80 hits、tile_rows=8192。完整执行仍有全部四步，两个原生锚点是方法的一部分。

## 三个容易混用的入口

| 源码/资产 | 实际对象 | 本轮处理 |
|---|---|---|
| `C_CONFIRM_METHOD.json`、`attention_c_dense.py` | 旧 C：high OUT-ABS + low **raw** OUT-D | 复用配对运行结构；替换候选绑定及方法身份 |
| `tools/ffnds_fourstep/generate.py --variant wide8192` | 原生 attention + wide8192 FFN | 可作为 W 臂；完整候选需外层 attention 绑定 |
| job 201500 的 `MassCenteredLow` | 可选择 raw 或 centered；三个统计/编解码函数使用 `torch.compile` | 冻结其 centered 路径 |
| job 204740 的 `EndToEnd` | diagonal/paired 机制诊断，两臂都直接使用 FP32 点运算并保存轨迹 | 用于机制和自由轨迹证据；保留为诊断实现 |

**关键细节：`MassCenteredLow.__init__` 的默认 `mode` 是 `"raw"`。** 因而 `MassCenteredLow(...).bind("OUT-D")` 本身仍会运行旧 C；必须显式设置 `online.mode = "centered"`。

201500 的 `get_moments`、`encode_cache`、`decode_base` 分别包装 `moments`、`encode`、`decode`，均为 `torch.compile(fullgraph=True, dynamic=False)`。正式路径令 `online.layer = None`，不使用 `LowTail`，不传 audit/profile/route archive，不调用 `save_trajectory`。204740 的 diagonal 与此路径的数学目标相同，计算图、舍入和测量负担有差别，不能把 204740 的耗时当成 201500 的正式效率记录，也不能要求两套实现全程逐位一致。

## 当前完整运行证据能支持什么

201500 在两个 DEV、每源两次颠倒顺序的同卡配对中，验证了 high 和前三级状态与旧 C 逐位一致，两个候选重复也逐位一致。固定轨迹八个站点的拟合后注意力误差能量均改善，同块 Dense 出口七处改善；一处 Dense 块误差增加约 2.00%，说明应在最终指标上评估净效应。

| 同卡配对区间 | 旧 C | centered | centered / C |
|---|---:|---:|---:|
| low 两步周期 | 13.3410 s | 13.1474 s | 0.98549 |
| 四步 GPU | 25.4588 s | 25.2731 s | 0.99271 |
| 完整请求，含保存 | 33.2576 s | 33.0572 s | 0.99397 |
| 同进程峰值分配 | 65.41 GiB | 64.20 GiB | 约 0.9815 |

该表是两来源开发读数，表明 centered 可以作为当前可部署路径继续评测；正式质量分布、总体加速与失败率由冻结后的独立来源评测给出。旧 C 的六源记录可以提供规模和对照设计依据，其方法身份应继续标为 raw C。

本轮对更复杂完整 value map 的自由轨迹检验没有建立稳定增益，因此主评测只保留 D、W、M 三臂：同四步 Dense、仅 wide8192 FFN、完整 centered。raw C 可作为已有消融资产保留，无需加入默认全量三臂之外的新分支。

## 附带薄封装

`centered_runtime.py` 与 `FROZEN_METHOD_BINDING.json` 应放在同一目录。该封装复用学校**已经跑通 job 201500 的运行目录**；它不是独立的模型部署包。运行目录需要原生成器及检查点、LightX2V/权重、FFN CAL、`attention_P_k.npy`，以及原 `attention_joint_action` 导入依赖链。只复制两个 attention 文件不能构成完整环境。

封装完成以下绑定：

1. 在导入前校验 `attention_online_structured.py`、`attention_mass_centered_low.py`、`attention_panel_fit.py` 以及 `FIT.sha256`，核对载入 CAL 与 P64。
2. 创建一次 `Experiment`、`MassCenteredLow` 和 `wide8192` 端口，供同进程配对复用。
3. `render(..., arm="centered")` 显式选择 centered；D/W 臂不绑定 attention，raw 仅为可选诊断臂。
4. 默认禁用本方法新增的张量捕获和审计路径，防止按旧 FFN-only 缓存复用；保留原生成收据，并写带完整方法指纹和来源 manifest SHA256 的并列收据。
5. 检查实际四步 attention/FFN 计数、几何、seed 42、四次专家选择 `[0, 0, 1, 1]`，以及四组 flow/latent/step_ms 记录。
6. 新 manifest 可以包含原运行器已登记来源；仅当图像 SHA、prompt SHA 和 seed 相同时复用其来源身份，冲突立即停止。

调用接口如下，路径参数均须指向学校现有资产；此处没有启动它：

```python
from centered_runtime import FrozenCenteredRuntime

runtime = FrozenCenteredRuntime(runtime_dir, fit_dir)
runtime.register_manifest(private_source_manifest)  # 新评测来源；原 DEV 已在运行器中
row = runtime.render(source_id, output_group, arm="centered",
                     capture_final_latent=True)  # 质量生成
```

单臂命令行也已实现，`--runtime-dir`、`--fit-dir`、`--source`、`--group` 必填；`--arm` 默认 `centered`。质量生成可传 `--capture-final-latent`，只在 step 3 更新结束时复制最终 latent 到 CPU，并在 `Experiment.render` 完成后保存 `<source>.final_latent.safetensors`。侧车 `final_latent` 字段记录路径、SHA256、字节数、shape、dtype 和 tensor_name=`latent`。hook 在 `finally` 恢复。此项不保存初始张量、flow 或中间步张量。

正式计时必须使用默认 `capture_final_latent=False`，此时不安装该 hook；原 `Experiment` 用于身份核对的对称 GPU trace witnesses 保留原样。带质量捕获的生成收据标记 `quality_capture_in_timing_window=true`，不能混入正式速度表。正式评测驱动在同一进程中对 `dense`、`wide8192`、`centered` 调用上述接口，依据冻结来源顺序和轮次轮换运行次序，每轮使用独立 `group`。

由于该结构与既有 `c_dense` 一样在同一进程常驻 wide 端口，其峰值内存应注明相同常驻上下文；独立部署峰值需另行以清洁进程测量。这不影响同卡顺序平衡的时间对照。

## 进入正式批次前的唯一集成检查

对 DEV0392、DEV1013，使用新封装各生成一次 centered，依据 201500 原始收据中的 centered `initial`、`flows`、`latents` 与 `equal_trace` 核对同一路径身份，并核对上述全程计数。该检查验证新入口确实接上已有方法，不重新调参数。采用同一 GPU 软件环境；如果身份不同，先定位源码、生成器、检查点、CAL 或环境差异。

随后冻结方法及来源 manifest，直接运行中等规模三臂方案。原 `C_CONFIRM_METHOD.json` 的精确旧 C trace 断言不能原样施加给 centered；应使用 201500 的 centered 收据作为集成参照。

学校私有 `generate_reference.py`、权重和运行目录未出现在本工作区，故本轮检查范围是代码结构、接口使用、方法参数与配置指纹。学校端集成核查与正式批次均未启动。

## 源码身份

共同提交：`4ca6b1b14b90965269a7b480f4fd5bfc75f64f37`。

| 文件 | Git blob |
|---|---|
| `research/20261124/attention_wide8192/results/joint39/analysis_20261126/joint_action_review/mass_centered_runtime/attention_mass_centered_low.py` | `72e486c1d6b33e6ae943fdd5f64a5a5ba87a81dd` |
| `research/20261124/attention_wide8192/results/joint39/attention_online_structured.py` | `7e7c611ec2aa7e8c8754bd19963cbce6ed49ded4` |
| `research/20261124/attention_wide8192/results/attention_panel_fit.py` | `f1ba6bd34237a2a58767022dbd2e0f99530cf08d` |
| `research/20261124/attention_wide8192/results/joint39/analysis_20261126/joint_action_review/mass_centered_runtime/attention_joint_action.py` | `e3dca8967bbfce79e68bc401842ac1dfed1e955a` |
| `tools/ffnds_fourstep/generate.py` | `d0e830011b08c1d1ebbbe1e22b5a9cefdcdfa39d` |

实际源文件 SHA256、CAL 指纹及参数见相邻 `FROZEN_METHOD_BINDING.json`。201500 计时和固定轨迹数据来自同目录的 `MASS_CENTERED_REPORT.md`、`MASS_CENTERED_SUMMARY.json`；204740 的实现差异由 `analysis_20261127/paired_value_transport/experiments/attention_paired_value_end_to_end.py` 直接核查。
