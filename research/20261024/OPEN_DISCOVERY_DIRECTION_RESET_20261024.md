# FFNDS 研究阶段复位：先发现机制，再选择算法

日期：2026-10-24。依据仓库固定的 AutoResearcher `5bd7ec537ae63f2b6a6dd579f0e886aff58a177c`，尤其是 `research-expansion` 的 OPEN_DISCOVERY、`problem-reformulation` 的表征发现，以及 `mechanism-research` 的源码重建。

## 总问题（保持不变）

研究 Wan2.2-I2V-A14B 四步蒸馏模型的真实 FFN 中，**为何某些计算可能对最终视频并非必要**，这种结构能否被低成本地识别、重构或利用，从而降低原生 FFN / 整体推理成本而不明显损害视频质量。

这里的对象不是“找到一个 mask”；不是“恢复被砍通道的 dense 输出”；不是“必须利用 FP4”。**量化格式是潜在的信息来源和数值机制，而非预定算法。**

研究必须把以下三个问题耦合、而非排成三道审批门槛：
- 信息：模型已计算或已存储了什么、它暗示了哪一种结构？哪些现象不必先设计动作就能辨别？
- 利用：某种结构最终若支持计算改变，能否以足够低的观测、选择、变换与物理成本利用？
- 计算：真实的冗余、非必要性、近似等价或传输不敏感究竟发生在哪里；它对视频质量意味着什么？

**当前阶段：DISCOVERY / MECHANISM EXPLANATION，明确早于 operator design。**

## 之前研究顺序为何偏离

| 过早收敛的选择 | 科学问题被替换成什么 | 正确放置 |
| --- | --- | --- |
| G6D scale-Gram proxy | 量化尺度能否预测隐藏组 W1 能量 / 排名 | 候选信息假说；在发现真实量化—FFN联系后再选用 |
| G6E Q2 W2 panel | Q2 尺度能否使某个 W2 K panel 被跳过 | 候选计算动作；不能代表 Q2 的全部科学价值 |
| G6F Q1 W1 K panel / Q2 transitions | 能否预测某组 K-panel 省算引起的 Q2 编码跃迁 | 候选因果解释与干预；不能预设 K 面板是正确省算单元 |
| FP4 code orbit / cell certificate / group sharing | 码字能否完全相等或精确被证书化 | 数学上的特殊现象；完整相等并非探索必要条件 |
| 历史 UB1 / dense restoration | 哪些密集行应当补算 | 既有实验/旧动作族的证据，不是科研任务的定义 |

这些提案和历史材料**保留原版本和学术价值**，但没有哪个已被证明是新阶段的默认策略。本次不删除、改写旧版本、不制造 agent 支持票，也不改 `ACTIVE_RESEARCH.json` / `QUEUE.md` 的并行共享状态。

## 开放探索的四个方向

### A · 量化产生了什么信息结构？（第一个入口）

关注量化**作为表示与变换**：真实源路径中的 scale / exponent / groupwise max、码字占据、符号/饱和、舍入状态及共享尺度耦合。假设可以来自三种对照：
- 原始数值幅度与 RMS / amax 的等价重新编码（侧信息不提供新的廉价决策差异）；
- 量化块内部的异质性、指数迁移、码字分布或空间/时间变化，包含常规 RMS 未捕获的有用结构；
- 量化过程造成的离散边界、饱和和跨通道状态耦合，使下游计算响应不同于连续 BF16。

**当前只问“它究竟是什么 / 在哪里出现 / 如何区别”，不规定要拿它挑哪个通道或 K 面板。**

### B · FFN 自身存在什么计算与表征结构？

以 `W1→GELU→Q2→W2` 的真实数据流为对象：W1/W2 权重几何、GELU 负端与符号、通道相依性/抵消、有效子空间、尺度变化与归一化/残差。问题是**功能上的相依与冗余**，不是先选稀疏行。

### C · 四步蒸馏和视频时空结构改变了什么？

去噪步、高/低噪声专家、block、latent 时间与空间区域的动态变化；蒸馏是否集中、消除或重新组织了 FFN 特征与时间信息；历史运动/背景/光照的失真可能揭示什么性质。旧结果不能证明蒸馏必然“损失稀疏空间”，也不能套用普通 LLM 激活分布。

### D · 哪些数值变化会被视频系统放大，哪些近似不可见？

当前 FFN 输出与 gate/residual、后续 norm/attention、四步轨迹和视觉稳定性之间的因果关系。探查方向与耦合，不止幅度；探索残差商空间和传播，而不先要求近零局部 RMSE。该方向为 A/B/C 的意义提供解释，但不因早期只剩标量标签而强制设定唯一指标。

**工作次序：** 先从 A 进入，沿真实源码和资产发现现象；B/D 为不可缺的解释视角，C 作为蒸馏视频独有的交叉比较。保留它们作为独立竞争性假说，不须马上合成一个方法。

## 何时才进入算子设计？

当至少一个候选机制能陈述：
1. 真实的源/数学观察，不只是算法名称或 FP4“看起来有元数据”；
2. 不同于最近已有方法的**竞争性机制解释**与可区分预测；
3. 该现象可能提供何种计算自由度（可先是开放集合，不必选 K/N、mask、补算）；
4. 可验证它是格式伪影、普通 RMS 再表示、模型权重性质、蒸馏时空效应，还是下游影响；
5. 独立 agent 的论证可以挑战现象或前提；不必为提出探索种子先跑 GPU 或取得视频接受。

**不要把以上变成另一套繁琐准入审批**：有价值的数学构想可随时作为 seed 保存，只是不应凭它直接改全项目默认算法路线。

## 现在唯一明确的第一任务

**E0：量化—FFN 结构现象与现有证据的源审查。**
见 [FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md](FIRST_TASK_E0_QUANTIZATION_FFN_DISCOVERY_20261024.md)。

完成 E0 不等于“选出算法”，而是产生少数彼此不同、在已保存数据与源码上有区别性预测的科学解释，并指出下一份最值得观察的信息。缺乏原始 tensor 时要明确证据缺口，绝不拿 QDQ BF16 scalar summaries 冒充原生 MXFP4/NVFP4 的 scale/code 测量。

## 约束与事实来源

- 原生 FFN 参考：`tools/packed_ffn_reference.py`；HiF4 runtime `upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/wan_i2v.py`；HiF4 scale source `upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu`。
- 当前历史实证：`research/20261022/R12_HiF4_group_separability.json`；`research/20261022/R13_spatial_sharing_limits.json`；`reports/worker_20261020/HP1_proxy_RUN.json`、`HP2_proxy_RUN.json`、`NP1_proxy_RUN.json`；`research/20261023/C0_coupling_offline_results.json`；`research/20261024/C1_boundary_target_audit.json`；`research/20261024/OFFLINE_SAVED_ASSET_QUANT_STRUCTURE_R1.json`。
- 原生格式对照：OCP MXFP4 E2M1/E8M0 per32 与 NVIDIA NVFP4 E2M1/E4M3 per16 + global factor；其实际推理 kernel/packing 与当前 A100 HiF4 QDQ→BF16 不同。
- 实验权限：`research/20261024/EXPERIMENT_PAUSE.json` 为全面暂停的原记录；用户后续仅明确恢复保存资产的 CPU 算术复核（`OFFLINE_ONLY_RESUME_SCOPE_R1.json`），没有恢复新的模型/GPU 运行。**E0 只读、源码/已有记录/数学分析，不安排新实验或任何队列。**
