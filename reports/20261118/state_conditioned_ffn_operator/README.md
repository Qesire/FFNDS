# 六站点状态条件 FFN 算子实验结果

**2026-11-18 · `CLOSED_AT_P2` · `NO_TRANSFER_GAIN`**。N32 在冻结开发集的局部回放中降低了相对误差。四个候选在固定三源的自由生成中均未建立可用的视频收益。P3、P4 在 P2 停止规则下关闭，SYS 未触发。[冻结实验方案](../../../research/20261118/STATE_CONDITIONED_FFN_OPERATOR_EXPERIMENT_PLAN.md) · [结案记录](output/P2_DECISION.json)

## 局部回放

P0 验证了严格 Q1 的 80 份产物和 132 对原生 P512 端点。六个候选位点的旧 A 与严格 Q1 的 BF16 矩阵逐位一致，本轮沿用旧 L64。P1 在同一 CAL11 上拟合 L0、L+、Q32 和 N32，在冻结的 DEV11 上测量局部风险。下表为各来源等权的相对风险。

| 算子 | FP64 拟合诊断 | BF16 原生回放 |
|---|---:|---:|
| L0 | 0.33845 | 0.33854 |
| L+ | 0.33665 | 0.33675 |
| Q32 | 0.33412 | 0.33420 |
| N32 | 0.32430 | 0.32440 |

N32 的 BF16 局部相对风险较 L0 低 4.18%。FP64 下，Q32 和 N32 均在 11 个开发来源上优于各自从 CAL 拟合的 D-only 对照。[P0 校验](output/P0_BINDINGS_SUMMARY.json) · [逐来源读出](output/READOUT.csv) · [BF16 验收](output/BF16_ACCEPTANCE.json) · [P1 汇总](output/P1_LOCAL_SUMMARY.json)

## 自由生成和完整视频人审

P2 在学校 Slurm 执行四臂各三条四步视频，共 12 条新视频、48 次前向和 1,920 次 FFN 调用。固定来源为 `vbi_0392`、`vbi_1013` 和 `vbi_0800`，每条视频使用 seed42、相配初噪和严格复现的 Wan2.2 A14B HHLL BF16 配方。候选作用于六个位点，其余 74 个 expert/层位点使用严格 Q1。全部 144 次候选位置的当前 gate 写回通过原生逐位核对。九个评分作业均正常结束。315 个配对数值覆盖 21 条新旧视频。

| 算子 | `vbi_0392` 主体时序一致性 |
|---|---:|
| Dense | 0.9604 |
| Global-P64 | 0.9588 |
| Q1r2 | 0.8713 |
| L0 | 0.8806 |
| L+ | 0.8759 |
| Q32 | 0.8687 |
| N32 | 0.8843 |

用户在 `vbi_0392` 观察到 Dense 特有的晕眩重影。Global-P64 的画面破碎较轻，整体最接近 Dense。Q1r2 与四个新候选出现严重画面破碎。`vbi_0800` 和 `vbi_1013` 的肉眼差异不明显。Q32 和 N32 在 `vbi_0392` 的局部 BF16 风险分别为 0.33807 和 0.33010，均等于各自的 DEV11 中位数。这个个例表明，局部回放风险遗漏了自由生成中的具体视觉故障。结果范围为六位点、固定 DEV3 和本轮预算。

[人审原文](output/HUMAN_REVIEW_DEV3.json) · [逐视频配对评分](output/QUALITY_PAIRED.csv) · [DEV3 汇总](output/DEV3_SUMMARY.json) · [原生执行验收](output/P2_NATIVE_ACCEPTANCE.json) · [评分协议溯源](output/QUALITY_LINEAGE.json)

## 执行证据

P0 作业 `173009` 和 P1 作业 `173055` 正常退出。早期 P0 作业 `172989` 的清单序列类型比较失败，P1 作业 `173031` 的 CUDA 分配器缺少显式设备索引。修正后分别在同一实验范围完成核验和回放。P2 的原生作业、评分作业、各自的范围与复审哈希、原始 RUN 和 EXIT 哈希均见执行清单。

仓库收录实现、评分或回放表、媒体与参数哈希。完整 MP4、latent、张量和原始回执保存在学校实验归档。公开版评分入口通过配置定位 `ffprobe` 和 Python，沿用冻结的工具字节校验。执行清单记录评分入口的原始与发布哈希。

[执行清单](output/EXECUTION_MANIFEST.json) · [算子实现](../../../tools/ffnds_state_operator) · [P2 生产范围准备](../../../tools/ffnds_state_operator/P2_STAGE_NOTES.md) · [评分入口](../../../tools/ffnds_state_operator/QUALITY_RUN_NOTES.md)


## 后续算子设计

当前结果保持上述结案状态。根据 Global-P64 的响应机制，下一项研究已定义为当前真实观测驱动的割线算子：[完整设计](../../../research/20261118/OBSERVED_SECANT_OPERATOR_DESIGN.md) · [有限实验方案](../../../research/20261118/OBSERVED_SECANT_EXPERIMENT_PLAN.md) · [FP64 参考核心](../../../tools/ffnds_observed_operator/README.md)。新范围为既有资产局部判别、9 条主开发视频及最多6条条件视频；本次只完成设计与代数参考实现，尚未运行新的模型实验。

