# R15 handoff — paired videos completed

The actual method now runs through the full native four-step video chain. Read [docs/77](../../docs/77_R15完整四位置响应与配对视频跑通记录.md), the [native result](R15V0_native_rollout_results.json), the [complete decoded-frame review](R15V0_full_duration_review.json), and the candidate registry. Use these actual outputs to guide implementation progress.

R15V0 completed exactly one dense D-Q video and one candidate video on old calibration24 source0, seed42. Both are 832×464,81 frames,16fps. The candidate ran at block20 in all four steps:8 actual boundaries,20 physical candidate linears. Each arm has1600 logical native linear/QDQ boundaries and zero cache misses, weight transfers or placeholder fallbacks. Initial noise SHA matches; each student's states evolve freely without dense resets or output splicing. Two fixed weight-only rank256 factorizations completed, with zero training. Peak allocation was63.026GiB; HF generation/factor/review artifacts total30.11MB.

All81 paired frames, spanning0.000–5.000s, were visually inspected through seven timecoded contact pages, plus full-resolution face/fish crops every8 frames. No obvious additional catastrophic subject, motion, scene or stable-treeline lighting failure was visible. Fish detail drifts in both clips. The review explicitly identifies agent inspection of the complete decoded frame sequence; it does not claim human real-time playback. This is one old-source, four-of160-position generation and agent frame review. The metric catalog requires human full-video review before formal smoke PASS; that review remains pending, so the registry has not promoted the candidate. It does not establish whole-model dynamic routing, population official quality, subtle flicker safety or native speed/Pareto.

The complete playable left-dense/right-candidate video, originals, contact pages and detail page remain under the HF cache. Public receipts contain neutral relative locators and SHA values. Decoder success did not automatically grant visual PASS. The75.333s/55.368s generation wall observations include fixed cold/warm order and observation work: they are not a speed benchmark, and their ratio must not be called method acceleration.

The [direct authority](R15V0_authorization.json) preserves the human reply “跑啊等啥快点啊” to the fixed-pair question. The [registered protocol](R15V0_registered_protocol.json) and approved draft are immutable. The first occupied-card refusal occurred before model Python, with zero generations or factorizations; its [receipt](R15V0_dispatch_preflight_failure.json) is preserved. The idle-card coordinator then completed one model process without retry. Slow startup was installed Transformers/shared-filesystem traversal. A read-only metadata warmup and short read-only syscall trace changed no code, weights or dependencies. Future separately authorized runs should explicitly bind/select the previously verified local source view before imports; this run did not change paths midway.

The fixed budget is spent and model eligibility is empty. There is no automatic repeat, source/rank/position sweep, training, whole-model rollout, official quality model or benchmark. Future empirical work needs its own concrete scope and user authority. Continue source and implementation work from the actual paired outputs; this four-site smoke is not full sparse-method validation.

Offline verification: `python3 tools/check_research_routes.py`, `python3 tools/check_r15_artifacts.py`, `python3 tools/check_r15_rollout.py`, `python3 tools/check_candidate_rollout_gate.py`, and the R15/contract unit tests. Entry points are `tools/run_paired_mean_contrast_rollout.py`, `tools/rollout_mean_contrast.py`, `tools/inspect_paired_rollout_videos.py` and `tools/export_paired_rollout.py`.

Earlier CPU/backend evidence remains separate:256 saved groups, original HiF4 counterexamples,6 fusion controls,12 QDQ controls and42 synthetic timings. The fused four-row chain's34.4184ms versus dense53.1074ms is random-tensor evidence. R14N0, D0 and other original finite closures, protocols and results remain unchanged.

The latest metric runbook and two concurrent upstream commits were reconciled. Its usage verifier passes16 historical scalar checks and the synthetic mathematical fixtures. A subsequent CPU-only analysis of the actual81-frame pair reports pooled RGB MSE280.645667, MAE8.047402, pooled-MSE PSNR23.649220dB and mean per-frame PSNR23.852673dB; worst MSE isframe72 at4.5s. All81 decoded frames and all four free latent states differ. See [actual RGB fidelity](R15V0_decoded_pair_difference.json). This post-hoc F1 observation includes texture and motion divergence; it is not official/perceptual quality, an independent-source result or a success threshold. No additional model/evaluator call was made.

## Preserved metric-library handoff from12015d8

The following is the separately completed metric-review snapshot. Its zero-new-model-run statements describe that review, before the independent native R15V0 result above. Its sources, methods and pending official evaluations remain applicable.

# R15：学术指标族、已存证据复算和跨层证据门禁（2026-10-23）

**研究结论：**指标定义和实测证据必须分开。已建立 10 个指标族、93 条定义/统计方法；这不是 93 次测量，也没有任何新候选视频或原生模型速度。原来 34 条 L0–L5 claim/观测目录仍管理已经观测的数据，新的学术族目录负责指导论文指标选择。

## 执行事实

- 指标定义：`research/metric_system/academic_metric_families.json`、同源 CSV、`docs/77_FFNDS学术指标族与评测协议_20261023.md`；凡指标选择包含学术套件，必须锁定具体实现、版本、单位、输入与统计单位。
- 实际复算：`tools/audit_r15_metric_evidence.py` 从 R14N0 原始每条 stripe 记录和 R14 合成时间戳的**已存标量**计算来源内 MSE、p50/p95/max、动作族差距和选择器 regret；详见 `docs/78_指标审计与已有实验映射_R15_20261023.md`、`research/20261023/metric_evidence_recalculation_R15.json`。
- 证据检查：`tools/check_metric_contract.py` 和 `tools/metric_evidence_validation.py`，新增防止 NaN、跨阶段代称、false 完整生成、错用独立样本、未绑定真实时延及不完整 Pareto 声明等用例。
- 已观测 R14N0 是 L1 局部写回损失，R14 1.913× 是随机张量合成算子链测速。当前无 R15 新原生模型运行/视频或独立来源 VBench 值。
- 本轮研究讨论按 AutoResearcher v2.3 的 CR（真实计算对象）、DF（严格数学与单位）、CRL（最小反例检查）、GAPI（真实质量与同前提速度目标）、MEAR（不能从负局部误差推出普适方法失败）审阅。

## 下一步实验优先级

1. 对被选中的**新**动态 FFN 备选，先落实真实四步自由生成配对视频并审查主体、运动、静态背景/光照明显崩溃；需另行取得 GPU/模型作业授权。
2. 对视频无明显崩溃的候选，按官方 VBench-I2V 九维在各自官方来源子集采样，并增加静止 ROI 闪烁诊断和来源分层统计。
3. 结合完整 FFN 和端到端时间、同硬件同条件的最佳静态/已发表可复现方法作质量—速度 Pareto。不可用 synthetic backend speed 或 L1 SSE 取代。

## 非目标与保留事项

不修改 R14N0、R12D0、R11 等历史冻结协议和结果；不重新打开旧候选、不自行注册 GPU、训练或新视频；也不把局部 SSE 结果的一次失败扩大成一切空间响应方法的反证。原始证据与新指标族目录均可使用离线代码追溯。

## R16 指标实际使用核验

[执行指引](../../docs/80_R15指标实际使用验收与Agent执行指引_20261023.md) 和 [复算记录](R16_metric_usage_acceptance.json)新增 16 条原生保存标量映射核对、R14 synthetic 速度比值核对，以及 PSNR、静态 ROI、空间误差分解、来源混合加权的纯合成反例。它们并非任何新自由生成或官方 VBench 实测。另修订四个纯像素 F1 指标不需要网络 backbone 的协议字段。Agent 下一阶段应在选中不同新候选并获独立授权后做完整视频 smoke，之后实施固定版本官方九维和真正原生加速评价。
