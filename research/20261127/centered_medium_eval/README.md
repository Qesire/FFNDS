# Centered + wide8192：方法冻结与中等规模评测

**50 源自动评测已完成。** 150 条原片、七项 VBench 和运动诊断通过校验。完整 centered 在完整请求上达到 1.167× 加速。七项质量指标的总体非劣判定为 INCONCLUSIVE，仍待 24 源人工盲审。[整体结果与逐源指标](QUALITY_REPORT_206224.md)。

**正式计时已完成。** 学校 A100 的 108 次配对请求中，完整 centered 相对 Dense 的完整请求加速为 1.167×，四步 GPU 加速为 1.229×。[正式计时结果与逐次记录](TIMING_206224.md)。

2026-11-27。依据研究分支提交 `4ca6b1b14b90965269a7b480f4fd5bfc75f64f37` 的已执行结果。

**当前方法已经形成贯通四步、40 层、注意力和后续 FFN 的完整候选。** 学校端已冻结 50 个独立图像簇并启动三臂评测。[学校作业与核验记录](SCHOOL_EXECUTION_50.json) · [公开来源清单](COHORT_50_PUBLIC.json)。

## 1. 冻结什么，为什么现在冻结

方法核心是：利用结构与锚点预测当前作用，在少量位置读取当前精确响应，再恢复尚未解释的变化。对一个模块可写为

\[
\widehat Y_t=B_t(X_t;C_a)
+\mathcal R_t\!\left[P^\top\bigl(Y_t-B_t(X_t;C_a)\bigr)\right],
\qquad \widehat Y_{t,P}\leftarrow Y_{t,P}.
\]

这里 $Y_t\in\mathbb R^{N\times d_{out}}$ 是模块输出，$B_t$ 是可在线计算的底座，$C_a$ 是锚点状态，$P\in\mathbb R^{N\times64}$ 选择原 64 个位置，$\mathcal R_t$ 是依据当前可见特征构造的残差恢复映射。公式只概括信息流；FFN 的点边联合读出与 attention 的按头恢复各自保留已经验证的具体实现。

| 四步位置 | 自注意力 | FFN |
|---|---|---|
| high step 0 | 原生完整计算，建立路由 | 原生计算、写逐位置 BF16 锚点 |
| high step 1 | selected attention + 当前 OUT-ABS + P64 | wide8192 命中与 P64 联合读出 |
| low step 2 | 原生完整计算、建立 centered 历史状态 | 原生计算、重新写锚点 |
| low step 3 | centered 对角运输 + OUT-D + P64 | wide8192 命中与 P64 联合读出 |

low 历史状态保存扣除遗漏质量对应均值、并按 value 通道尺度标准化后的贡献；命中时用当前均值和尺度还原，再用真实 P64 修正剩余变化。cross attention、原四步 HHLL、40 层、固定 CAL、固定 P64 与 BF16 语义一并冻结。`wide8192` 是执行 tile 行数，FFN 选中通道数仍是 2560。

判断“已足够优化”的依据是当前预算内的机制比较：

- job 201500 已跑通实际 centered 完整生成。八个受测站点的 attention 误差均改善，七个同输入 Dense 块出口改善；已有两来源用户成片反馈。
- job 204487/204740 将更高自由度的完整 value 矩阵运输推进到恢复、后继块和自由轨迹。该版本相对对角版本的十二站 Dense 块误差几何能量比为 1.0841；四源终点 latent 距离改善 1/4、RGB 距离改善 0/4。当前完整矩阵分支未提供足以替换对角版本的稳定增益。
- 当前 P64 读出、锚点信息与非线性后继已经得到真实算子和成片层面的辨析。继续扩大矩阵、rank、λ 或局部读出搜索的预期价值，低于测定现有完整版本的跨来源质量与完整请求成本。

因此，本轮把“框架内已经完成主要取舍、达到可评测稳定版本”作为冻结标准。理论算量上限与工程性能仍有空间；版本冻结的目的正是用统一方法身份测清楚剩余空间，而不是声称全局最优。

## 2. 当前性能证据的准确归属

| 方法比较 | 实际来源 | 四步 GPU | 预热完整请求 |
|---|---|---:|---:|
| 旧完整 raw C / 同四步 Dense | 6 源 × 2 轮，job 199260 | 1.227× 加速 | 1.156× 加速 |
| 当前 centered / 旧 raw C | 2 DEV × 2 轮，job 201500 | 时间比 0.99271 | 时间比 0.99397 |

旧 raw C 的七项六源平均分差绝对值均小于 0.31 个百分点。centered 的固定轨迹改善及两条开发视频反馈支持其成为当前候选。50 源实验将首次在同一批输入上直接测量完整 centered / Dense / FFN-only，避免把不同批次的比率相乘后称为新实测。

固定两锚点和现有模块边界下，已有主要矩阵算量估算约节省 28%，对应的乐观算量参照约节省 37.02%；这些数值界定结构空间，不替代实际墙钟结果。蒸馏与本方法的合并系统加速应在匹配的原教师分母上另测，本轮主分母明确为同一个四步蒸馏模型。

## 3. 一次完成的中等规模评测

| 内容 | 固定配置 | 请求数 |
|---|---|---:|
| 质量主集 | 50 独立图像簇 × seed42 × D/W/M | 150 |
| 正式计时 | 预选 12 源 × 3 轮 × D/W/M | 108 |
| 集成身份与预热 | 两个 DEV 的 centered 身份核对；D/W 各一次预热 | 4 |
| 本次驱动合计 | 包含准备请求；评分另行执行 | 262 |

D 为同四步 Dense；W 为原生 attention + wide8192 FFN；M 为完整 centered + wide8192。主比较 M/D 回答整体收益，W/D 与 M/W 回答组件贡献。

50 源按场景、自然、主体三组 17/16/17 预选，候选类别范围包括官方 11 类；报告实际各类别数量。按图像簇抽样，合并重复图像别名，硬排除 CAL 与当前 DEV，登记历史曝光并优先选择登记中未曝光的来源。所有分支、prompt、种子与方法参数在看新结果前冻结。主运行仅保留最终 latent 和原片；完整内部轨迹不默认落盘。

七项既有官方函数在全部 50 个 custom image/video 配对上评分：subject consistency、background consistency、aesthetic quality、motion smoothness、imaging quality、I2V subject、I2V background。主表报告各臂每维 50 个真实样本，原 benchmark 标签子集作附表。研究协议是 **CUSTOM_PAIRED**，每源每臂一个真实视频；官方完整五视频协议与榜单总分保留其原定义。

质量统计以独立图像簇为单位，在三个固定分层内做 10,000 次配对 bootstrap。三臂和七维共享抽样索引。研究预选非劣容差为 −0.005（−0.5 个百分点）：全部七维 95% CI 下界超过该界支持指标非劣；具体维度 CI 上界低于该界表明该维存在超过容差的下降；其余按 INCONCLUSIVE 报告。此容差是本研究选择，尚无本批结果。

完整请求时间以每源每臂三轮秒数的算术均值形成比率，再跨来源取几何均值。预登记工程推进目标为点估计至少 1.10×，95% CI 下界高于 1.00×。同时报告四步 GPU、原始秒数、来源分布、冷启动和峰值；同进程常驻 wide 端口的峰值单独注明，不能据此推断 Dense 的独立部署内存。

既有 Farnebäck 诊断复用全部 81 帧及时间窗口；预选 24 源盲审动作完成、运动幅度和连续性、细节及遮挡。所有 50 源原片和逐源分数保留。运动量与 Dense 轨迹距离用于解释偏差，画质判断结合官方函数和用户原片判读。

核心 258 请求按既有约 35–45 秒/片的量级规划约 2.5–3.23 GPU 小时；模型加载、首次编译、准备请求及评分单独核算。这是预算估算。完成固定批次后报告结果，不将 CI 跨界自动变成反复扩样或调参。

## 4. 统一入口已整理，学校端复用现有资产

`centered_runtime.py` 显式绑定 `MassCenteredLow.mode="centered"`，这是完整候选的关键身份。原生成 CLI 的 `--variant wide8192` 对应 W；旧 `attention_c_dense.py` 对应 raw C。封装复用 job 201500 已跑通的完整运行目录、原权重和 CAL，不重写 kernel。

`run_medium_eval.py` 将两个 DEV 的 centered 收据与 job 201500 做一次身份复核，然后执行固定三臂批次。身份核对并预热后先执行正式计时。正式计时禁用诊断和潜变量捕获，保留原生成器各臂对称的 GPU trace witnesses。完成 36 个计时配对后写出 `TIMING_SUMMARY.json`，随后生成质量视频并保存 step 3 的最终 latent。

| 文件 | 用途 |
|---|---|
| [FROZEN_METHOD_BINDING.json](FROZEN_METHOD_BINDING.json) | 唯一候选、四步调度、核心源码与 CAL 指纹 |
| [METHOD_READINESS.md](METHOD_READINESS.md) | 完整方法、运行入口和证据范围 |
| [MEDIUM_EVALUATION_SPEC.json](MEDIUM_EVALUATION_SPEC.json) / [设计](MEDIUM_EVALUATION_DESIGN.md) | 样本、统计单位、预算、判据与停止条件 |
| [EXPOSURE_REGISTRY.json](EXPOSURE_REGISTRY.json) | 当前可核验的 CAL/DEV/历史曝光登记及来源 |
| [prepare_cohort.py](prepare_cohort.py) | 学校候选表 → 50 源私有/公开清单、12 源计时和 24 源盲审子集 |
| [materialize_school_catalog.py](materialize_school_catalog.py) | 将固定 VBench 元数据与学校实际 16:9 图像资产绑定为候选表 |
| [centered_runtime.py](centered_runtime.py) / [run_medium_eval.py](run_medium_eval.py) | 既有在线方法的绑定与三臂平衡运行 |
| [build_metric_binding.py](build_metric_binding.py) | 原片、收据、最终 latent 与七维官方评分的绑定 |
| [run_motion_diagnostics.py](run_motion_diagnostics.py) | 复用已有连续运动计算，生成逐源/逐帧诊断 |
| [summarize_medium_eval.py](summarize_medium_eval.py) | 配对分数、分层 CI、真实速度和最终 latent 距离 |
| [METRIC_AND_DATA_BINDING.md](METRIC_AND_DATA_BINDING.md) | 已有离线评分资产、七维调用和视频索引 |
| [LOCAL_VALIDATION.json](LOCAL_VALIDATION.json) | 原 100 源版本的历史 CPU 验证 |

学校候选表来自封存的 VBench 元数据和现有 16:9 图像。`prepare_cohort.py` 已实读入选图像并输出 50 源清单，评分器对应的 50 张图像也已通过相同 SHA-256 核验。入选来源均未命中已有曝光登记。

主要执行顺序：

```sh
python prepare_cohort.py --candidates "$FFNDS_CANDIDATES" \
  --exclusions EXPOSURE_REGISTRY.json --method-binding FROZEN_METHOD_BINDING.json \
  --output-dir "$FFNDS_EVAL_ROOT/cohort"

python run_medium_eval.py --runtime-dir "$FFNDS_RUNTIME_201500" \
  --fit-dir "$FFNDS_FROZEN_FIT" --cohort "$FFNDS_EVAL_ROOT/cohort/cohort.json" \
  --identity-receipt "$FFNDS_MASS_201500_RECEIPT" \
  --out "$FFNDS_EVAL_ROOT/runs.json"
```

其后按 [评分绑定说明](METRIC_AND_DATA_BINDING.md) 产生 `metric_binding.json` 并运行已有 `eval_anchor10.py vbench` 七个维度。调用新汇总器时加入 `--with-latents`，同时汇总最终 latent 相对 Dense 的 L2。`build_metric_binding.py` 输出原片索引；学校发布实际原片后补入可访问链接，和逐来源结果共同交付。

公开仓库保存本目录和去除 locator/prompt 的清单；学校运行目录保留真实输入清单、三臂原片、最终 latent、评分缓存和收据。正式提交论文前，可在此固定测试集上增加完整 SparsePR/Chipmunk 对照，并另测教师 + 蒸馏 + 本方法的匹配系统倍率。本次三臂评测独立提供本方法的完整质量与成本事实。

## 5. 固定证据入口

- [centered 完整运行、联合块出口和配对时间](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/analysis_20261126/joint_action_review/mass_centered_runtime/MASS_CENTERED_REPORT.md)。
- [最新完整 value 运输与 SparsePR 恢复对照复核](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/analysis_20261127/paired_value_transport/result_review/NEW_RESULTS_REVIEW.md)。
- [已测速度及各自分母](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/analysis_20261126/speed_and_core/MEASURED_SPEED_AND_DENOMINATORS.md)。
- [真实算量与理论边界](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/analysis_20261126/speed_and_core/SPEED_LIMITS_AND_CORE.md)。
- [VBench I2V 官方协议与 custom_input](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/README.md)。
