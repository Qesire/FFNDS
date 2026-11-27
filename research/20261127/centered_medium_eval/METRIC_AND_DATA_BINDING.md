# 中等规模评测：指标与数据的实际绑定

核查对象：`Qesire/FFNDS`，研究 head `4ca6b1b14b90965269a7b480f4fd5bfc75f64f37`。本轮仅做源码、存档和 CPU 接口核查，新增 GPU 作业与新增真实指标均为 0。

## 结论与最短执行路线

现有资产足以支撑中等规模 `CUSTOM_PAIRED` 评测。复用已经执行过的七个 VBench 原始评分函数、离线评分环境、图像与 prompt 索引；增加一个薄的绑定适配器，把 `FrozenCenteredRuntime` 的完整方法回执交给原评分器。主要新增工作是给冻结 centered 版本生成成对视频、执行评分和交付可播放原片。

七项指标可以对选定的全部 50 个图像—视频对评分，并在主表中给出相同的 50 源配对均值。保留原始官方维度标签，附表再报告原标签子集与各维度样本数。这是“官方评分函数在固定自定义样本上的评测”，与官方完整五样本套件有明确区别。

建议主表保留全部七项，重点解释输入条件保持、画面质量、运动连续性四项：`i2v_subject`、`i2v_background`、`imaging_quality`、`motion_smoothness`；其余三项提供时间一致性和美学侧面的补充。另用现成全帧运动诊断与盲审检查“该动的内容是否在动、动作是否完成”。同种子 RGB 与时序差分只负责解释相对 Dense 的轨迹偏离。

## 1. 已有七项原始指标与实际入口

实际评分器是 `research/20261124/attention_wide8192/results/joint39/c_dense_baseline/eval_anchor10.py` 的 `vbench` 子命令。它对非 `i2v_` 名称导入 `vbench.<dim>.compute_<dim>`，对 I2V 名称导入 `vbench2_beta_i2v.<dim>.compute_<dim>`；逐视频保留原始结果，并核对本地聚合与官方返回值一致。

| 现成维度 | 资产清单中的模型 | 科学用途 | 与目标之间的距离 |
|---|---|---|---|
| `i2v_subject` | DINO ViT-B/16 | 输入图像与视频特征保持、帧间特征连续性 | 全图特征；不是分割后的主体成功率 |
| `i2v_background` | DreamSim ensemble | 输入图像与视频的外观保持、帧间连续性 | 全图特征；不隔离背景区域 |
| `imaging_quality` | MUSIQ SPAQ | 无参考画质代理 | 不直接判断 prompt 动作或运动幅度 |
| `motion_smoothness` | AMT-S | 时间连续性代理 | 静止视频也可能平滑 |
| `subject_consistency` | DINO ViT-B/16 | 视频内相邻帧与首帧特征一致性 | 特征稳定不等于任务正确 |
| `background_consistency` | CLIP ViT-B/32 | 视频内背景/全图外观一致性代理 | 同样受镜头变化与真实运动影响 |
| `aesthetic_quality` | CLIP ViT-L/14 + LAION 线性读出 | 美学代理 | 不直接说明恢复机制或语义动作成功 |

表中的模型映射来自仓库 `tools/offline_i2v_metric_loaders.py` 与历史资产清单；本轮拟用的具体权重仍由实际 `metric_scope` 哈希与逐项权重哈希最终绑定。

`eval_anchor10.py` 的评分子命令读取 `binding.arms` 和 `binding.records`，没有固定十源限制；固定十源/七臂的是该文件原有 `prepare` 子命令。`eval_attention_c_dense.py.prepare` 又固定了六源、12 个计时 pair、seed42。新增的 `build_metric_binding.py` 只替换这层准备逻辑。

### 为什么全部 50 源都能做七维 custom 评分

核对官方固定版本 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`：`vbench.VBench.build_full_info_json(mode='custom_input')` 会把所请求的维度赋给每条自定义视频。I2V 的 source tags 是标准套件的样本选择元数据，不是以下评分公式的必要输入。

对 DINO 或 DreamSim 的归一化全图特征，官方两个 I2V 维度都采用

\[
s=0.4\max_t c(I_0,I_t)
 +0.3\operatorname{mean}_{t\geq1}c(I_{t-1},I_t)
 +0.3\min_{t\geq1}c(I_{t-1},I_t),
\]

其中余弦相似度截断在非负值。它们读取输入图像和视频，不读取 prompt 动作类别，也不使用主体/背景分割掩码。因此主表采用预先声明的 `custom_all_bound_sources` 有公式依据；附表保留 `original_dimension_tags` 可以解释与官方原子集的关系。此前“每维都必须只用原官方标签子集”的建议过于严格，本报告以这里的源代码核查为准。

这也解释了补充运动与人审的必要性：若第一帧忠实、后续变化偏小，上式仍可能很高；七维不能单独证明语义动作完成或恢复了全部有用动态信息。

## 2. 历史评分版本与当前绑定必须分别记录

旧 C/Dense 评分作业 `199277` 绑定的 metric scope SHA256 是：

`bf008598e5183cf6b0c88a08f8b525b6341acc65e890b4bb620562883654de1d`。

评分脚本会核验 `metric_package_versions`、`metric_code_sha256`、`metric_asset_sha256`、`quality_reference_sha256`，包括 `metric_loaders.py` 与 `metric_contract.py`。它固定单 A100、TF32 关闭、评分随机种子及实际图片字节。repo 当前归档只给出该 scope 哈希；本轮未取得学校实际 scope 内容，不能把历史资产清单自动宣称为这次已验证的运行环境。

仓库已有可重新核验的离线资产清单 `research/20261023/R17_metric_asset_preparation.json`，及观察到的环境版本 `environment/vbench_i2v_official.pip-versions.txt`。历史清单版本是 VBench0.1.5、DreamSim0.2.1、PyTorch2.7.1+cu118、torchvision0.22.1+cu118、pyiqa0.1.10、timm1.0.12；该版本文件明确说明它不是干净机器安装锁。完整资产路径、字节数和 SHA 已提取到 `METRIC_DATA_AUDIT.json`，供运行前逐项比较；它是本轮审计记录，与实际运行生成的 `METRIC_BINDING.json` 分开。

另有 `tools/video_metric_feedback/vbench_worker.py` 能对固定官方 checkout 做九维及 temporal_flickering 评分，支持 `custom_image_path` 和显式 `camera_instruction`。它的 `--local` 仍允许官方初始化器下载缺失资产，不能替代已验证的离线 scope。当前七维已能回答中等规模主问题，dynamic_degree/camera_motion 可在有实际需求时补充；无需因此重开整套指标工程。

## 3. 有实际依据的数据池

`C_CONFIRM_SOURCES.archive.json` 保存的是一次真实来源选择快照：

| 快照字段 | 数量 | 含义 |
|---|---:|---|
| `candidate_source_count` | 1,118 | 图像、prompt 等来源记录 |
| `candidate_cluster_count` | 355 | 当时有可用裁剪图像的独立图像簇 |
| `excluded_cluster_count` | 56 | 历史、校准/开发、已观察来源的排除并集 |
| `fresh_candidate_cluster_count` | 299 | 当时未被排除的图像簇 |
| 三个选样层的可用簇 | 52 / 53 / 98 | structured / natural / subject |
| 旧确认阶段已选 | 6 | 每层两个；如今均已观察 |

因此同一快照在减去这六个已观察簇后，剩余未观察簇的**上界**是 293；三层中的相应上界是 50、51、96，共 197。后续实验可能进一步减少它们，所以新清单必须合并最新暴露记录并重新核验实际图片路径。这个数据基础支持准备 50 簇的中规模评测，不受更早局部协议中“仅六个空闲簇”的数字限制。

旧六源为 `vbi_0128/C00322`、`vbi_0512/C00160`、`vbi_0816/C00079`、`vbi_0592/C00290`、`vbi_0877/C00073`、`vbi_0987/C00283`。它们可以作为开发回归/流程对照，不再标成全新来源。

原快照提供完整 56 簇排除列表与 digest、来源候选文件 SHA `45badf27bbad0a79dd9aaff9062967433dd55b7db47069a87e9724ff08034256`；归档刻意省略 `image_path`、完整 prompt 和输入清单路径。学校未删字段的原清单才是可执行输入；repo 哈希和计数足以证明资产来源与选择过程，不能凭这些哈希虚构可执行路径或新选中名单。

新 `sources` manifest 至少保留：source_id、source_cluster_id、image_path、image_name、image_sha256、prompt、prompt_sha256、category、original_dimension_tags、exposure_history。原图像簇是独立统计单位，同图不同 prompt、seed、crop 属于嵌套重复。

## 4. 现有分数与资产分别可以怎样复用

| 可复用对象 | 条件与范围 |
|---|---|
| 图像/prompt 索引、离线指标权重、原评分函数 | 核对实际字节、版本和哈希后继续使用 |
| 旧六源视频、逐源七维分数 | 对应同一旧 C / Dense 输出；用于历史描述、开发回归、评分器对照 |
| 相同 Dense 原片的分数 | 必须同图/prompt/seed/权重/采样/尺寸、同视频字节与同指标环境；重新绑定来源后可复用为该输出的质量观察 |
| 旧 C 的分数当作 centered 结果 | 方法输出已经变化，应对实际 centered 原片重评分 |
| 历史计时 | 提供预算依据；新接口正式速度要按同卡、同边界配对重新计时 |
| timing repeat1 的同字节视频 | 若与 repeat0 的 MP4/RGB 完全相同可复用数值计算，不增加独立样本数 |

推荐 metric cache key 包含视频 SHA、输入图像 SHA、dimension、scorer/source SHA、离线权重 SHA、评分 seed、预处理及归一化版本。原评分器的部分结果恢复使用整个 binding SHA，改变 cohort/arm 后会拒绝直接恢复；如果将来需跨 cohort 复用，导出逐视频缓存并明确重建绑定，不能仅凭 source_id 拼接旧表。

旧七维六源 C−Dense 差值在 −0.302 到 +0.127 个百分点，全部区间都跨越 0。这为扩大确认样本提供依据；这些观察不能替代新 centered 方法的指标，也不构成已证明视觉等价。

## 5. 三类观察各回答什么

**成对近似误差。** 继续使用已有 decoded RGB MAE、PSNR、temporal_delta_MAE。后者精确定义为候选与 Dense 的相邻帧变化之差的绝对均值。小值支持复现同噪声参考轨迹；大值也可能来自内容运动相位差。它不是无参考生成质量或“闪烁程度”的别名。

**任务与视觉质量。** 七维全部列出原始值、成对差值及来源簇 bootstrap 区间；主表不拼成自创加权总分。`tools/ffnds_video_metrics.py` 已提供全帧连续运动：max-side320 Farneback 位移均值、成对光流 L2、候选/Dense 运动幅度比，逐帧及16帧窗。这个结果是估计光流的诊断，需同时保留原始分子分母；低运动参考的比值不稳定，应单列。运动幅度并非越大越好，目标是保留该输入的正确动作。

**人审。** 对完整视频做方法名隐藏、左右顺序平衡的成对评审。最小记录是 motion_amplitude、motion_continuity、layout_detail、occlusion、prompt_action_success，以及 prefer_left/tie/prefer_right 和失败原因。相同画面质量但动作丢失是独立失败类型；主表高分不能覆盖它。报告来源簇数量、真实视频数量和审阅覆盖率。

本轮计划可冻结数值容忍范围；若它们只是研究者选取的工程容忍值，就按此名称报告。旧 cache 研究的 1 个百分点阈值不是已经验证的感知等价界限。

## 6. 薄绑定适配器与执行接口

新增 `build_metric_binding.py` 读取实际 runtime 的 `source.full_method.json`，并核对同目录 `source.json` 和 `source.mp4`。它检查 source manifest、method binding、runner、runtime generate.py、来源图像、prompt、四步轨迹、各臂共同初始噪声、视频与解码指纹；不重写任何七维指标。

输入 manifest 使用 `sources:[{source_id,source_cluster_id,image_path,image_name,image_sha256,prompt,prompt_sha256,category,original_dimension_tags,...}]`。`--run` 直接读取 `run_medium_eval.py` 的完整 run JSON，只取 `quality` 下的三个臂，按 `output_root/record.group/source.mp4` 还原实际文件；不从计时重复中选择视频。质量侧车的 `final_latent` 也被核验并带入 binding。适配器当前对应 runtime 的 seed42、单真实视频/来源/臂；扩展多个 seed 时需另行绑定，不能复制现有视频。

```bash
python build_metric_binding.py \
  --run /ABS/MEDIUM_RUN.json \
  --cohort /ABS/SOURCES.json \
  --method-binding /ABS/FROZEN_METHOD_BINDING.json \
  --runner /ABS/centered_runtime.py \
  --metric-scope /ABS/CURRENT_METRIC_SCOPE.json \
  --metric-scope-sha256 ACTUAL_CURRENT_SCOPE_SHA256 \
  --evaluator /ABS/eval_anchor10.py \
  --evaluator-sha256 70de41c15205f3fd58ed150396252b081b556915fc6ec163e0b0335a9b80e4db \
  --verify-decoded-rgb \
  --binding /ABS/METRIC_BINDING.json

python /ABS/eval_anchor10.py vbench \
  --binding /ABS/METRIC_BINDING.json \
  --out /ABS/quality \
  --dim i2v_subject
```

最后一条按七维逐项运行；作业调度与 GPU 提交另由既有学校入口执行。命令中的 `/ABS` 与 `ACTUAL_*` 是明确的现场参数，不是已存在路径或已执行结果。

输出另带 `.video_index.json`，将 source、cluster、arm、实际原片路径、MP4/RGB 哈希及 `review_url` 连在一起。路径和哈希供校验，`review_url` 才供用户直接审片；缺链接时状态保持 `LINKS_PENDING`。评测交付时应补齐每臂原片可访问链接，附对照版可以便利比较，但原片也应可打开。

### 直接复用已有运动函数

`run_motion_diagnostics.py` 调用原 `tools/ffnds_video_metrics.py.evaluate`，导出全部帧、80 个相邻变化和原运动/配对指标，不修改光流或聚合公式。原工具完整 CLI 同时强制要求 FFN 分项耗时；中规模质量生成阶段不测这项，适配器因此采用单独的运动诊断 schema，省去 runtime 家族，并明确 `cost_reported=false`。未测成本没有用零值替代。

```bash
python run_motion_diagnostics.py \
  --binding /ABS/METRIC_BINDING.json \
  --implementation /ABS/FFNDS/tools/ffnds_video_metrics.py \
  --implementation-sha256 0c6ab066674adb2a04097978ced52abcd7cbf1125918b74d59cf976a56ffec66 \
  --out /ABS/motion
```

它为 Dense 记录固有运动量，为 wide8192 和 centered 记录固有运动及相对 Dense 的差异，生成逐源逐臂报告与 `MOTION_INDEX.json`；指标输入身份来自同一个绑定中的 prompt、seed、方法 release manifest 和逐片完整回执。`generation_run_sha256` 贯穿视频、七维和运动数据，便于汇总时拒绝混入旧运行。

## 证据入口

- [现有七维评分器](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/c_dense_baseline/eval_anchor10.py)
- [旧 C/Dense 协议](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/c_dense_baseline/C_DENSE_PROTOCOL.archive.json)
- [旧 C/Dense 逐源分数](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/c_dense_baseline/C_DENSE_SUMMARY.json)
- [355/299 图像簇的选择快照](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261124/attention_wide8192/results/joint39/C_CONFIRM_SOURCES.archive.json)
- [离线权重清单](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/research/20261023/R17_metric_asset_preparation.json)
- [现成全帧运动诊断](https://github.com/Qesire/FFNDS/blob/4ca6b1b14b90965269a7b480f4fd5bfc75f64f37/tools/ffnds_video_metrics.py)
- [官方 custom_input 入口](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/__init__.py)
- [官方 I2V subject 公式](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/i2v_subject.py) / [background 公式](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v/i2v_background.py)
