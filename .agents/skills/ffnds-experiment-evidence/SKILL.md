---
name: ffnds-experiment-evidence
description: Construct and audit FFNDS source sampling, calibration, held-out data, paired video generation and metric evidence using real implementations and constructive scientific prompts.
---

# FFNDS · Sampling, calibration, evaluation and data evidence

Activate for any FFNDS question involving calibration, sample rows, source cohorts, model states, video generation/evaluation, generalization, official scores, or the reliability of an observed effect. Start with [data protocol](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md) and the applicable original source/config.

## Positive execution sequence

1. **State the research estimand and its observational unit**. E.g., UB1 motion preserved per image-prompt source across independent seeds, or state-conditioned local FFN error per source×step×block. Report what existing data already establishes.
2. **Recover the actual computation and data path**: file+entrypoint; upstream native Q1/GELU/Q2/W2/gate; source→crop→prompt→noise→state→row draw→fit→rollout→decoded frames→metric. Identify real counts, not path count or filenames.
3. **Choose the appropriate scope**: discovery, calibration, development, frozen confirmation, official benchmark. Preserve historical exposure; condition the experiment on real availability.
4. **Build discriminating matched contrasts** with independent sources and clustered uncertainty. Decouple source count, sampled rows and student-state correction; compare paired Dense/candidate at the same noise, model/scheduler and video parameters.
5. **Use mature implementation semantics**: VBench-I2V official 5 actual samples/condition versus custom one-video-per-source worker; explicit missing dimensions; continuous motion alongside official dynamic degree; domain-aligned calibration (LLM Compressor) and controlled generator reset (Diffusers).
6. **Preflight cheaply**: `python tools/ffnds_cohort_preflight.py --manifest path.json`. Register reused and genuinely independent outputs; inspect file bytes for suspected aliases with `--verify-file-content`. Pass confirms manifest consistency only.
7. **Interpret first, then guide**: observed effect → strongest competing mechanism(s) → opposing predictions → smallest independent discriminating evidence → next action. Tag HISTORICAL/STATIC_ONLY/NATIVE/VIDEO/OPEN at claim level, not as a block of caveats.

## Prompt operator for the next agent

> 从真实源码、已有原始数据及成熟官方实现重建本次研究的抽样单位、校准来源、Student/Dense 状态、视频生成种子与质量指标；先陈述有意义的现象和新的结构解释。为每一个候选机制提出能区分其与最强竞争解释的来源级对照，固定真正独立的source、训练/开发/确认用途、必要的随机数与指标集合；优先用现有可重用资产作低成本判别。明确20源×一个独立视频与100个文件名、6,144嵌套行与独立来源数、单维官方函数与正式九维的区别。以来源级统计和完整视频时序/成本陈述证据，输出可执行的最小下一步与资产locator；未知项给出能够获得证据的实际动作。

Historical results are protected as findings of their original protocols. Update forward-facing guidance and separately version any re-evaluation; new checks do not relabel old generated videos.


## Concrete experiment implementation (2026-10-27)

Use the executable [FFNDS implementation runbook](../../../research/agent_protocol/EXECUTABLE_EXPERIMENT_RUNBOOK_20261027.md). Source identity: `ffnds_experiment_plans.py source-bank`; select independent image clusters before choosing their image-prompt conditioning for new calibration/development cohorts, then preserve the selected `native_source_ordinal` as the stable row-draw identity. For official VBench Image Suite preserve required multiple prompt conditions per image and report both counts. Calibration real geometry and nested positions: `ffnds_native_geometry.py` + `calibration-plan`. Frozen real generation jobs: `video-plan` + `ffnds_run_registered_video.py` (actual LightX2V D-Q/UB1). Exact native noise and distinct-clip verification: `ffnds_stage_registered_i2v.py stage`. Actual official metric calls: `ffnds_stage_registered_i2v.py evaluate --execute`. Source-cluster inference: `ffnds_cluster_metric_report.py` (nested seed × sample, then equal-weight source/prompt within image cluster, then equal-weight image clusters). Report each tool's real status and produced evidence. When creating a new calibrated model, use a new cohort-aware fit implementation; the historical step-budget fit remains frozen at 24×256.

Native operand ingestion now uses `ffnds_validate_calibration_operands.py` to bind real Q1/V1, Q2/V2, Y, gate and b2 BF16 tensors to the **registered sampled row coordinates** before any new fit. Legacy `fit_step_budget.py` remains the frozen 24×256 comparator.

## Calibration v2 default research behavior

Load [the calibration METHOD_SNAPSHOT](../../../research/20261027/CALIBRATION_V2_METHOD_SNAPSHOT.md) and [registered factorial design](../../../config/calibration_design_v2_20261027.json) when discussing how much calibration data to use. Each component has a separate target: support signed group cross-term and source stability, gain/bias heldout signed error, rank256 effective dimension and heldout risk, teacher/student same-coordinate native risk, and full-video motion/deployment. Start with existing real evidence and compute feasible diagnostics using `python tools/ffnds_calibration_design.py coverage --design ... --bank ... --out ...`; diagnose actual source-level group energy only with `diagnose --statistics ... --keep-groups-by-step 54,108,108,162 --out ...`. Decide expansion through matched source-vs-row studies (12×512/24×256/48×128, 24×128/24×512) and a separately labeled motion-domain arm. Keep all unfitted and ungathered arms PLANNED/NOT_MEASURED. State what the new experiment can distinguish and its first real producer.
