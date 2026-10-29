---
name: ffnds-experiment-evidence
description: Construct and audit FFNDS source sampling, calibration, held-out data, paired video generation and metric evidence using real implementations and constructive scientific prompts.
---

# FFNDS · Sampling, calibration, evaluation and data evidence


**最新用户优先级（2026-10-29）：只测FFN自身稀疏，不做原作者LLM/FLUX或非FFN系统加速作为前置。** [FFN-only基线方案](../../../research/20261029/WAN_NATIVE_TRAINING_FREE_BASELINE_REPRIORITIZATION_20261029.md)：P0 SharQ L74的Wan native sparse FP4+Dense residual总成本[#59](https://github.com/Qesire/FFNDS/issues/59)，P0 TEAL后GELU置零全160视频质量与真正大M sparse W2[#58](https://github.com/Qesire/FFNDS/issues/58)，P1 Chipmunk paid proxy的Wan FFN跨来源验证[#57](https://github.com/Qesire/FFNDS/issues/57)。已有TEAL四来源视频评价EXIT0，但masked Dense W2不是物理加速；SharQ论文Wan2.2 T2V+SageAttention总加速不能直接报本仓I2V FFN-only成绩。候选必须在原Wan四步双expert全160处实际作用，覆盖动作/语义与FLOPs/内核/端到端真实开销；每个新CPU/GPU实验另行独立反例审查、recheck、ACTIVE冻结、真实RUN/EXIT。本文旧Chipmunk+TEAL共同P0及LLM原版复现已被本段替代。



**最新 P0 免大训练基线（2026-10-29）**：Chipmunk 与 TEAL 同为最高优先级，见 [#57](https://github.com/Qesire/FFNDS/issues/57)、[#58](https://github.com/Qesire/FFNDS/issues/58) 及 [ACTIVE](../../../research/ACTIVE_RESEARCH.json)。作者原条件与Wan2.2四步分scope；Chipmunk作者Wan2.1实例默认MLP关闭，必须独立打开MLP-only并重新验算；TEAL pre-W1 与 post-GELU 分开核实真正W1/W2算量、FP16/BF16与完整视频。新CPU/GPU研究实验每个scope在执行前重新独立评议，reviewer recheck后才能注册ACTIVE并运行；S2尚未执行的新因果GPU实验降P1，旧结果保留。[TEAL作者FP16单token有限kernel](../../../research/20261029/TEAL_AUTHOR_FP16_KERNEL_PROBE_RESULTS_20261029.md)已复审后实际EXIT0并闭环；完整LLM、Chipmunk原系统及Wan迁移仍未复现，scope状态以ACTIVE和实际报告为准。



**2026-10-29 每个真实实验执行前重审本次实验的强制约束**：先加载 [EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW](../../../research/agent_protocol/EXPERIMENT_PREEXECUTION_INDEPENDENT_REVIEW_20261029.md)。主Agent提交完整实验自审与两个相反预测；不同真实上下文的独立 reviewer 检查原论文/作者源码一致性、真实 Wan 计算图/两个 expert、source-cluster、Dense/neutral/强baseline、真实W1/W2物理省算、读入/路由/缓存/原生系统成本、输入SHA与GPU范围；给出反例和处理意见；主Agent修订，reviewer recheck，随后才能向 ACTIVE 登记新有限scope，launch前重查版本指纹。既有论文方法通过评议不等于当前实验通过；未审结保持 REVIEW_REQUESTED，阅读/代数/静态单测继续，无新 CPU/GPU 研究实验。尤其按[Chipmunk/TEAL训练自由FFN新基线](../../../research/20261029/INDEPENDENT_METHOD_REVIEW_AND_FFN_BASELINE_20261029.md)分离论文作者条件与Wan迁移、FFN-only与联合attention/step-cache、preW1与postGELU；未审结scope保持 REVIEW_REQUESTED；已闭合scope按实际报告记录，不外推到其他scope或作者benchmark。


Activate for any FFNDS question involving calibration, sample rows, source cohorts, model states, video generation/evaluation, generalization, official scores, or the reliability of an observed effect. The live [I/F/C route portfolio](../../../research/20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md) distinguishes local restoration, motion, operator speed and deployment claims; choose estimands separately for each. Start with [data protocol](../../../research/agent_protocol/DATA_SAMPLING_CALIBRATION_EVALUATION.md) and the applicable original source/config.

**开放采集的默认动作（2026-10-29）**：[问题驱动采集协议](../../../research/agent_protocol/EVIDENCE_DRIVEN_ACQUISITION_20261029.md)与[采集机器队列](../../../research/agent_protocol/ACQUISITION_QUEUE_20261029.json)。先问科学问题需要哪些真正独立source、视频时间/空间配对、teacher/student状态和native前后数值边界。历史source不够或producer没有该变量时，主动建立新cohort并实现native capture，依独立来源分类、质量/条件分层和source-held反例逐批扩张。6/12/24/48/20等是历史或可比较的实验设计点，而非研究采集上限；新增来源与RUN必须区分新发现、已暴露开发、冻结确认和官方基准。

**新数据的优先证据**：每批记录机制冲突的哪两种预测、所需张量在什么时刻真正生产、采集坐标、来源/seed、模型+backend、学生状态、成本、原生RUN/EXIT及变更后的科学判断。已有数据无法回答时提出可执行producer与数据请求，而非只把它写成OPEN。

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
