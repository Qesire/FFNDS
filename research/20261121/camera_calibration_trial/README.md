# 镜头运动轨迹加入校准的等预算对照

捕获前在本地冻结的登记使用 3 张校准图和 2 张同一 VBench 归档、按来源簇留出的新图，共 13 条 Dense 教师轨迹。抽取两张验证图前排除了历史 G20 和既有 24 图面板覆盖的 54 个不同来源簇。已提交作业 182597 做原生采集，作业 182601 依赖前者成功完成，在相同的验证输入上拟合并比较两套折叠权重。本提交归档实验开始时的登记和代码。此前 0951 的诊断、人审和性能数据单独保存在[已完成实验快照](https://github.com/Qesire/FFNDS/tree/b8839ed819c2a1fc0396001d0aadb622a55f2989/reports/20261121/anchored_video_experiments)。

| 用途 | 原图及来源 | 种子与条件 |
|---|---|---|
| 校准 | 0032 抽象色彩、0304 雪中车辆、0456 厨房 | A 各取原提示 42 和 43。B 共享原提示 42，种子 43 依次追加拉近、左摇、右摇。每组共 6 条。 |
| 保留验证 | 0905 海边行走的企鹅、1014 马球骑手 | 每图种子 44，分别用原提示和左摇。每个来源的所有条件均留出。 |

预设读出只考察高噪声专家第 0、13、26、39 层，去噪步 0 提供锚点、步 1 提供命中端点。每条轨迹在每层读取 64 个固定探针和 448 个其他位置，在同一教师状态上独立报告门后点值、空间差、时间差、沿真实方向的投影和正交误差。四层在每张图上属于重复测量，独立验证来源数为 2。[完整实验登记](outputs/FFNDS_camera_cal_trial/实验登记.md) · [逐条英文提示、种子、图像及来源簇 SHA](work/camera_cal_trial/protocol.json) · [提交回执与依赖](execution.json)

## 复算

1. 安装本项目对应的 Wan2.2 / LightX2V 原生四步环境和匹配的 VBench 16∶9 首帧。用 [base 身份](runtime/base_identity.json) 核对独立归档的模型实现、调度、配置、离线参数和 64 行位置。A、B 共用这些输入。完整模型、离线权重和原生教师面板保存在执行环境。
2. [公开协议](work/camera_cal_trial/protocol.json)使用相对于 VBench 数据目录的图像路径。将每个来源和轨迹的 `image_path` 加上实际 VBench 根目录后，以 UTF-8、`ensure_ascii=False, indent=2` 和末尾一个 LF 写入捕获目录的 `protocol.json`。原运行的协议 SHA-256 为 `d1582cc65e7f6481b7c4024731e02f7074bd593ae0f9e74c9ef08baf2c280569`。路径重定位会产生新的协议指纹，保留新捕获回执配套核验。用原根目录还原的文件已复验得到原指纹。
3. 在匹配原资产的捕获目录中放置 `base.json`、[捕获代码](work/camera_cal_trial/capture.py)和 [runtime 原生模块](runtime)。在带匹配环境和 Slurm 变量的 A100 节点上，核对[捕获包文件指纹](outputs/FFNDS_camera_cal_trial/capture_MANIFEST.sha256)，对重定位文件重新记录本地指纹，执行 `python -B capture.py`。读取 `output/CAPTURE_INDEX.json`，完整要求为 13 条轨迹、52 次去噪调用、52 个逐层端点面板。每条轨迹必须使用自己的锚点、命中 gate 和 64 行观测。
4. 将[拟合程序](work/camera_cal_trial/fit_camera.py)与[冻结折叠实现](work/camera_cal_trial/folded_runtime.py)置于同一目录。执行 `python -B fit_camera.py --capture-stage CAPTURE_DIR --output RESULT_DIR`。每次留出一整个校准来源估计收缩系数。冻结特征和点值＋同潜帧空间边的拟合目标，独立评分时才引入时间差。[拟合包指纹](outputs/FFNDS_camera_cal_trial/fit_MANIFEST.sha256)
5. 以 `RESULT_DIR/RESULT.json` 和 `high00/13/26/39` 对应的四份 JSON 核查逐层、逐源、逐条件的 A、B 读数。将 `RESULT.json` 放在 `outputs/FFNDS_camera_cal_trial/fit/` 后运行[配对汇总](work/camera_cal_trial/summarize_trial.py)。[教师成片背景运动测量](work/camera_cal_trial/measure_camera.py)可传入 `--directory CAPTURE_DIR/output/dense --output dense_motion.json`。该测量检查提示产生的视野运动。拟合回放未生成闭环 A/B 候选视频，动作、纹理与速度需用后续配对成片实测。

原执行使用的脚本列于[作业回执](execution.json)及两份冻结清单。仓库内的相对路径版协议和公开材料各有自己的 [SHA-256](SHA256SUMS)。学校提交脚本依赖执行端的资源配置，仓库提供其已提交版本作为记录。
