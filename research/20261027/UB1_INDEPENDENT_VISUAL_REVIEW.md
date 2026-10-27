# UB1 独立逐帧视觉复核 · 2026-10-27

**结果（VIDEO，两个既有 formal20 来源）**：主体与背景整体都可辨认，未在缩小的逐帧接触表中看到明显场景崩坏。犀牛例有显著运动差：dense 在后半段移动头部和后腿，UB1 姿态更接近静止；背景住宅例两者构图与外观相近。此观察支持“可保持大体画面，但运动不一定跟随 dense”，不等于全人群或完整质量验收。

**方法**：OpenCV 4.13 CPU 逐帧解码四个原 MP4，各 81 帧、832×464、16 fps。两侧相同缩放至 416×232，不裁剪，左 dense、右 UB1；按 9 连续帧生成 18 页并逐页实际查看。帧编号 0–80。文件 SHA 与现存 comparison manifest 一致。接触表有 JPEG 编码和缩放，不是原分辨率验收，也不提供原速播放体验；细微闪烁、运动顺滑度和微小纹理质量须原速播放/进一步评价。

| formal20 来源 | seed | dense 视频 SHA256 | UB1 视频 SHA256 |
|---|---:|---|---|
| input00，subject，source_index 882，犀牛 | 42 | ae644663e7a14e79e11f1f2013fa6e3b2645f9befeddd66ede17ff72d47df296 | 72d63b8852ac8c4dc894c2d2d599556c1da9ee826df685950fc0833b71440111 |
| input10，background，source_index 80，山坡住宅 | 42 | 61d5cce2aa9eb477a7f6a6e879f108a0093e22c17a20f080105176ee2f4ccbe6 | 5754b71cb67d1fd5e0a9cab0f808e43a9f69dde2ea8d66bf3538d79296d8bdb9 |

配对身份、prompt、路径及原始收据见 `reports/worker_20261020/LR2_formal20_input00_comparison.json` 和 `LR2_formal20_input10_comparison.json`；记录指出历史原输入图像 SHA 缺失，因此原图历史字节身份未证实。两臂同 source/seed 由当时配对记录提供。

**逐帧所见**：
- 犀牛：0–20 帧两臂均可识别为低头吃草的犀牛，背景草丛和整体光照相近；20–53 帧 dense 头部和嘴角姿态逐渐变化，UB1 头部较少变化；54–80 帧 dense 后腿位置与身体轮廓逐渐改变，UB1 后腿和躯干轮廓更稳定。两臂都保留犀牛主体；UB1 有小幅变化，不能称完全冻结。两臂视角总体稳定；在此显示分辨率未见严重主体断裂或明显大范围亮度跳变。
- 住宅：0–80 帧两臂都保留山体、雪、房屋与屋顶布局；背景全局构图变化较小，画面含细小局部纹理变化。该低运动样本不能评价大运动；未见肉眼明显的整体闪白或大规模结构崩坏。细节及实时闪烁未判定。

**既有指标（非本次运行）**：原 formal20 聚合质量在当时界内；另一 supplementary20 的 imaging 相对 dense 下降 3.166%，见 `research/20261022/POST_TRAIN_DYNAMIC_FFN_PAPER_PLAN.md`。两例来自前者，不能外推后者或整个来源分布。该原视觉复核任务是模型逐帧图像审阅，不是人工观看、官方评分或正式质量通过；该任务未新增生成、训练、官方打分或 GPU 计算。随后独立的运动补评见下节。

**材料**：`.runtime/ub1_visual_review_20261027/00_0.jpg` 至 `00_72.jpg` 与 `10_0.jpg` 至 `10_72.jpg`；每页 9 帧。犀牛运动差最直观在 `00_54.jpg` 和 `00_72.jpg`；住宅尾段在 `10_72.jpg`。下一项最直接判别是人以原 16 fps 完整播放同源配对，尤其比较犀牛动作幅度与住宅细闪；另选 supplementary20 且保留原 cohort 身份可检验成像不足是否有可见对应。

## 提示词、逐视频分数与成本核对

两条 prompt 分别为 **“a large rhino grazing in the grass near a bush”** 和 **“a bunch of houses that are on a hillside”**。第一条有吃草动作语义；UB1 中犀牛呈低头吃草姿态，接触表可见小幅变化，但仅凭逐帧静图不能确定真实咀嚼/持续进食是否清楚，因此不判定其违背 prompt。dense 比 UB1 动作幅度更大是配对观察，不自动等于 UB1 质量较差。第二条描述静态住宅布局，不要求镜头运动；低运动本身不构成违背。

原 formal20 此次归档的是 aesthetic、imaging、motion smoothness **三项**，原始结果含逐视频条目，配对表按 source group 汇总；不是只有 cohort 均值。五个文件名副本来自同一 source，不是五个独立生成。逐视频结果指向 `formal20_*_vbench3/videos/`，已核验两臂两例的 `-0.mp4` SHA 与本次审阅原视频分别完全一致。以下值为原表 D-Q → UB1（imaging 是 MUSIQ 原分除 100）：

| source | aesthetic | imaging | motion smoothness |
|---|---|---|---|
| 犀牛 | 0.61718 → 0.60682 | 0.75357 → 0.74579 | 0.98099 → 0.98269 |
| 住宅 | 0.68497 → 0.70899 | 0.78191 → 0.76493 | 0.98093 → 0.97421 |

来源 `reports/worker_20261020/formal20_UB1_paired_sources.csv`、`UB1_quality_evidence_manifest.json` 及其中逐视频 raw results。上述高 motion smoothness 不能测出动作幅度或证明 prompt 里的吃草动作；该 formal20 三项报告未提供 dynamic degree、subject consistency、background consistency 的逐视频结果，虽然 protocol 中列出相关软件文件，不能视作已执行。原三项 formal20 汇总曾满足其当时 2% 工程阈值，不扩大为九维视频质量或人工验收。

**方法与成本身份**：`UB1_policy.json` 定义静态按 step 分配 54/108/108/162 个 64-channel groups（共 216 组），并使用 BF16 rank-256 双线性残差；平均主 GEMM MAC 为 dense 的 50%，残差另为 1.85185% 的 dense 主 FFN MAC。其固定支持和低秩残差与后来的 E0 affine54（全输入 affine 加 54 组 nonlinear residual，预算为 94/216）是不同 operator，不得互相继承质量/速度。历史 `UB1_joint_engineering_acceptance_RUN.json` 的完整 160 位置、每臂三次测试，dense 平均 10.1776 s、UB1 6.5827 s，1.5461×；其中在线辅助与保守共享准备计入。这是特定 full-FFN benchmark，不是 20 条视频逐条速度。

完整视频耗时的可核验例是另一个 **旧 calibration24 source0 seed42**，非本次 formal20 两例：`research/20261023/R17E0_system_timing_prefix.json` 在 A100 80GB PCIe 热态、每臂五次技术重复，dense 完整生成中位 50.8582 s，UB1 47.2799 s，1.07568×；管线含编码保存及最终同步，已驻留模型并先热身，不包含模型加载。此单来源技术重复不能代表 formal20 的视频速度分布。


## 本轮原版运动补评及独立核对

本轮对上述四个已有视频使用一张 GPU 实际运行原版 VBench dynamic degree；这与上面的 LLM 静态逐帧复核是不同任务。原始 RUN SHA256 为 `47b4b5389f5c75586851177f9193c3c19fb2fba0d1f7b1d259cab7e56f539b18`，冻结协议及全部执行源码快照的 SHA 均一致。账本为1次 RAFT 初始化、1次 checkpoint 加载、160次 forward、3200次迭代及4个视频；GPU监测19次采样无冲突，物理使用峰值1484 MiB。外部 launcher 报告退出0。

| 来源 | dense 的40对均值 | UB1 的40对均值 | UB1 / dense | 原版二值结果（dense / UB1） |
|---|---:|---:|---:|---|
| 犀牛 | 3.3932 | 1.2937 | 38.1% | False / False |
| 住宅 | 1.9950 | 0.1833 | 9.19% | False / False |

这里均值是原函数逐对取图像光流幅值最高5%像素的均值，再对40个采样帧对取平均；它是对原始逐对结果的补充描述。160个逐对值均低于原版10.875像素阈值，四个原版 binary 因而都为 False；同一二值输出掩盖两臂幅值差。此图像光流不单独测量主体动作质量，也不单独判断 prompt 遵循：住宅 prompt 是静态描述，犀牛的 grazing 姿态本身不能判为失败。两来源此前已暴露，完整 formal20 的 dynamic degree 和 cohort 质量仍未运行、未获验收。

本轮原始来源、逐对值与运行身份见 [运动补评结果](UB1_DYNAMIC_DEGREE_RESULTS.md) 和 [机器记录](UB1_DYNAMIC_DEGREE_RESULTS.json)。
